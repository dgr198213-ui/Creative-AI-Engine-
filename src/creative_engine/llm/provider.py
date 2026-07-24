"""Abstracción multi-proveedor LLM con retry, rate limiting y router.

Soporta cualquier API OpenAI-compatible (OpenAI, DeepSeek, Qwen, Ollama...).
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..core.config import LLMProviderConfig
from ..core.exceptions import (
    LLMAuthError,
    LLMEmptyResponseError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
)

logger = structlog.get_logger(__name__)

# Parámetros con manejo especial (sustitución, no eliminación genérica):
# max_tokens/max_completion_tokens tienen su propio swap en _call_api.
_SPECIAL_PARAMS = frozenset({"max_tokens", "max_completion_tokens"})

# Mensajes reales de OpenAI para un parámetro/valor no soportado por el
# modelo, p.ej. "Unsupported parameter: 'max_tokens' is not supported..."
# o "Unsupported value: 'temperature' does not support 0.9...". Captura el
# nombre del parámetro entre comillas simples.
_UNSUPPORTED_PARAM_RE = re.compile(r"Unsupported (?:parameter|value): '([^']+)'")


def _is_retryable_same_provider(exc: BaseException) -> bool:
    """Predicado de reintento de `_call_api` (mismo proveedor, una vez).

    Casi todo lo indisponible-transitorio vale la pena reintentarlo
    contra el MISMO proveedor (rate limit, red). La excepción: contenido
    vacío con `finish_reason="length"` — el modelo gastó TODO el
    presupuesto de tokens en razonamiento interno invisible; reintentar
    con el mismo prompt y el mismo `max_tokens` repetiría el mismo
    resultado vacío. En ese caso concreto no vale la pena gastar una
    llamada más contra este proveedor: mejor rotar directamente
    (`LLMModelRouter.run` ya hace failover ante LLMEmptyResponseError).
    """
    if isinstance(exc, LLMEmptyResponseError):
        return exc.details.get("finish_reason") != "length"
    return isinstance(exc, (LLMRateLimitError, httpx.ConnectError))


# Proveedores "razonadores" (temperature en _unsupported_params, la única
# señal disponible sin una lista de modelos hardcodeada) gastan parte del
# presupuesto de max_tokens/max_completion_tokens en razonamiento interno
# invisible antes de producir contenido visible. Multiplicador aplicado al
# `max_tokens` pedido, con un suelo generoso — así un writer con
# max_tokens=2000 no se queda sin margen para contenido real.
_REASONING_TOKEN_MULTIPLIER = 3
_REASONING_TOKEN_FLOOR = 4096


def _temperature_style_hint(temperature: float) -> str:
    """Traduce la temperatura a instrucción de prompt.

    Para modelos "razonadores" que ignoran/rechazan `temperature` (p.ej. la
    familia gpt-5.6, que solo acepta 1), la palanca de diversidad que usan
    los operadores de mutación del QDEngine (temperaturas altas = más
    riesgo, bajas = más conservador) se pierde. Esto la recupera como
    instrucción explícita en el prompt.
    """
    if temperature < 0.4:
        return "Sé riguroso y conservador; prioriza solidez sobre originalidad."
    if temperature > 0.8:
        return (
            "Arriesga: prioriza enfoques inusuales y exploratorios "
            "aunque sean menos seguros."
        )
    return ""


@dataclass
class LLMResponse:
    """Respuesta estandarizada de cualquier proveedor."""

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class LLMProvider:
    """Proveedor LLM con retry automático y rate limiting."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._last_request_time: float = 0.0
        self._min_interval = config.min_interval_seconds
        # Qué parámetro de límite de tokens acepta este proveedor. La API real
        # de OpenAI exige `max_completion_tokens`; los compatibles (Gemini,
        # Z.ai...) usan `max_tokens`. `type=openai` en la config es solo la
        # pista inicial: si el proveedor rechaza el parámetro con un 400, nos
        # autoadaptamos y recordamos la elección (ver _call_api). Así un
        # proveedor OpenAI sin TYPE configurado se corrige solo en la primera
        # llamada en vez de quedar deshabilitado todo el run.
        self._use_max_completion_tokens: bool = config.type == "openai"

        # Parámetros que este proveedor ha rechazado alguna vez (400
        # "Unsupported parameter/value") y que ya no se envían más: p.ej.
        # `temperature` en la familia gpt-5.6, que solo acepta el valor 1.
        # Persiste para el resto de la vida del provider, igual que el
        # swap de max_tokens. Ver `_call_api` para la autoadaptación.
        self._unsupported_params: set[str] = set()

        # Contadores de coste (llamadas lógicas y tokens): usados por el
        # arnés de benchmark (bench/harness.py) para medir el presupuesto
        # gastado por brazo. Cuentan la llamada final que tuvo éxito, no
        # los reintentos internos por autoadaptación de parámetros.
        self.total_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

        # httpx descarta el path de base_url si la petición empieza por "/".
        # Normalizamos: base con barra final + ruta relativa sin barra inicial,
        # para que rutas como .../v1beta/openai/ + chat/completions se preserven
        # (necesario p.ej. con la capa compatible-OpenAI de Gemini).
        base = (config.base_url or "https://api.openai.com/v1").rstrip("/") + "/"

        self._client = httpx.AsyncClient(
            base_url=base,
            timeout=httpx.Timeout(config.timeout_seconds),
            headers={
                "Authorization": f"Bearer {config.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )

        self._log = logger.bind(provider=config.name, model=config.model)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_providers: int | None = None,  # solo relevante para RoledLLM/router
    ) -> str:
        """Genera texto a partir de un prompt.

        `max_providers` no aplica aquí (un LLMProvider suelto no enruta):
        se acepta e ignora para que un llamador funcione igual con
        `LLMProvider` o `RoledLLM` sin distinguir cuál recibió — ver
        `LLMModelRouter.run`.
        """
        async with self._semaphore:
            await self._rate_limit()
            response = await self._call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature if temperature is not None else self._config.temperature,
                max_tokens=max_tokens or self._config.max_tokens,
            )
            self._record_usage(response)
            return response.content

    async def generate_structured(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        max_providers: int | None = None,  # ver generate()
    ) -> dict[str, Any]:
        """Genera una respuesta estructurada (JSON).

        Usa un parser tolerante que extrae el JSON aunque el modelo lo
        envuelva en ```json ... ``` o añada texto alrededor (habitual en
        Gemini y otros modelos que ignoran response_format).
        """
        from ..evolution.mutation import parse_llm_json

        async with self._semaphore:
            await self._rate_limit()
            response = await self._call_api(
                prompt=prompt,
                system_prompt=system_prompt or "Responde únicamente en JSON válido.",
                temperature=0.3,
                max_tokens=self._config.max_tokens,
                response_format=response_format,
            )
            self._record_usage(response)
            try:
                return parse_llm_json(response.content)
            except Exception as e:
                self._log.error("structured_parse_failed", raw=response.content[:300])
                raise LLMError(f"Respuesta no es JSON válido: {e}") from e

    def _record_usage(self, response: LLMResponse) -> None:
        self.total_calls += 1
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens

    @retry(
        retry=retry_if_exception(_is_retryable_same_provider),
        # Pocos reintentos internos: el disyuntor del router gestiona la
        # indisponibilidad sostenida. Así el failover tarda segundos, no minutos.
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        _token_param_retry: bool = False,
        _dropped_params_this_call: frozenset[str] = frozenset(),
    ) -> LLMResponse:
        """Llamada real a la API con retry."""
        start = time.perf_counter()

        # Si `temperature` quedó marcada como no soportada (modelos
        # razonadores como gpt-5.6, que solo aceptan el valor 1), la
        # palanca de diversidad se traduce a una instrucción de prompt en
        # vez de perderse: mutación/cruce siguen pudiendo pedir más o
        # menos riesgo aunque el proveedor ignore el parámetro numérico.
        effective_system_prompt = system_prompt
        if "temperature" in self._unsupported_params:
            hint = _temperature_style_hint(temperature)
            if hint:
                effective_system_prompt = (
                    f"{effective_system_prompt}\n\n{hint}"
                    if effective_system_prompt
                    else hint
                )

        messages: list[dict[str, str]] = []
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Causa raíz probable del contenido vacío: en modelos "razonadores"
        # (temperature en _unsupported_params — la única señal disponible
        # sin hardcodear una lista de modelos), max_tokens/
        # max_completion_tokens incluye el razonamiento interno invisible.
        # Un tope ajustado (p.ej. el del writer, 2000) se agota pensando y
        # no deja nada para el contenido visible. Se amplía el presupuesto
        # enviado a la API — el `max_tokens` que ve el resto del motor
        # (contadores de coste, logs) sigue siendo el pedido originalmente.
        effective_max_tokens = max_tokens
        if "temperature" in self._unsupported_params:
            effective_max_tokens = max(
                max_tokens * _REASONING_TOKEN_MULTIPLIER, _REASONING_TOKEN_FLOOR
            )

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
        }
        # La API real de OpenAI rechaza `max_tokens` en modelos recientes
        # (400 invalid_request_error) y exige `max_completion_tokens`. El
        # resto de proveedores compatibles siguen usando `max_tokens`.
        # El flag se autoadapta si el proveedor rechaza el parámetro.
        if self._use_max_completion_tokens:
            payload["max_completion_tokens"] = effective_max_tokens
        else:
            payload["max_tokens"] = effective_max_tokens
        if response_format:
            payload["response_format"] = response_format
        if self._config.extra_body:
            payload.update(self._config.extra_body)

        # Los GLM de Z.ai activan razonamiento por defecto: >120s por
        # respuesta en free tier → timeouts sistemáticos en evaluación.
        # Para nuestras tareas (JSON corto) el thinking no aporta nada,
        # así que lo desactivamos salvo que el usuario lo pida explícito.
        if (
            self._config.model.lower().startswith("glm")
            and "thinking" not in payload
        ):
            payload["thinking"] = {"type": "disabled"}

        # Parámetros ya rechazados por este proveedor en llamadas previas
        # (o antes en esta misma cadena de reintentos): fuera del payload,
        # ni siquiera se intentan de nuevo.
        for dropped in self._unsupported_params:
            payload.pop(dropped, None)

        try:
            resp = await self._client.post("chat/completions", json=payload)
        except httpx.ConnectError:
            raise
        except (httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError) as e:
            # Errores de red transitorios: reintentables como una indisponibilidad.
            raise LLMRateLimitError(
                f"Fallo de red con {self._config.name}: "
                f"{type(e).__name__}: {e or 'sin detalle'}",
                details={"provider": self._config.name, "error_type": type(e).__name__},
            ) from e
        except httpx.HTTPError as e:
            # str(e) puede venir vacío en algunos errores → incluir siempre el tipo.
            detail = str(e) or repr(e) or type(e).__name__
            raise LLMError(
                f"Error HTTP con {self._config.name}: {type(e).__name__}: {detail}",
                details={"provider": self._config.name, "error_type": type(e).__name__},
            ) from e

        if resp.status_code == 429:
            raise LLMRateLimitError(
                f"Rate limit excedido en {self._config.name}",
                details={"status": 429, "provider": self._config.name},
            )

        if resp.status_code in (401, 403):
            raise LLMAuthError(
                f"API key inválida o sin permisos en {self._config.name} "
                f"(HTTP {resp.status_code}). Revisa la variable de la clave.",
                details={"status": resp.status_code, "provider": self._config.name},
            )

        if resp.status_code == 400:
            try:
                err_body = resp.json()
            except Exception:
                err_body = {}
            err_type = (
                (err_body.get("error") or {}).get("type")
                if isinstance(err_body, dict)
                else None
            )
            if err_type == "invalid_request_error":
                # Autoadaptación del parámetro de límite de tokens: si el
                # proveedor rechaza `max_tokens` (OpenAI real) o
                # `max_completion_tokens` (algunos compatibles), cambiamos el
                # flag, lo recordamos para el resto de la vida del provider y
                # reintentamos la MISMA petición una única vez. Sin esto, un
                # proveedor OpenAI sin `type=openai` en la config quedaría
                # deshabilitado para todo el run por un error de forma.
                err_msg = (err_body.get("error") or {}).get("message", "") or ""
                rejected_max_tokens = (
                    not _token_param_retry
                    and "'max_tokens'" in err_msg
                    and "max_completion_tokens" in err_msg
                    and not self._use_max_completion_tokens
                )
                rejected_max_completion = (
                    not _token_param_retry
                    and "'max_completion_tokens'" in err_msg
                    and self._use_max_completion_tokens
                )
                if rejected_max_tokens or rejected_max_completion:
                    self._use_max_completion_tokens = rejected_max_tokens
                    self._log.warning(
                        "token_param_auto_adapted",
                        provider=self._config.name,
                        now_using=(
                            "max_completion_tokens"
                            if self._use_max_completion_tokens
                            else "max_tokens"
                        ),
                        hint="añade CREATIVE_LLM__<NOMBRE>__TYPE=openai para "
                        "evitar esta llamada extra en cada arranque",
                    )
                    return await self._call_api(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        _token_param_retry=True,
                        _dropped_params_this_call=_dropped_params_this_call,
                    )

                # Autoadaptación genérica: cualquier otro parámetro que el
                # modelo rechace (p.ej. `temperature` en gpt-5.6, que solo
                # acepta 1) se elimina del payload en vez de tumbar el
                # proveedor. max_tokens/max_completion_tokens quedan fuera:
                # esos se sustituyen (arriba), nunca se eliminan. Tope de 3
                # parámetros distintos por llamada — un proveedor patológico
                # que rechace todo no reintenta indefinidamente.
                match = _UNSUPPORTED_PARAM_RE.search(err_msg)
                param = match.group(1) if match else None
                if (
                    param
                    and param in payload
                    and param not in _SPECIAL_PARAMS
                    and param not in _dropped_params_this_call
                    and len(_dropped_params_this_call) < 3
                ):
                    self._unsupported_params.add(param)
                    self._log.warning(
                        "param_auto_dropped",
                        provider=self._config.name,
                        param=param,
                    )
                    return await self._call_api(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        _token_param_retry=_token_param_retry,
                        _dropped_params_this_call=_dropped_params_this_call | {param},
                    )

                raise LLMInvalidRequestError(
                    f"Petición inválida en {self._config.name} "
                    f"(400 invalid_request_error): "
                    f"{(err_body.get('error') or {}).get('message', '')}",
                    details={
                        "status": 400,
                        "provider": self._config.name,
                        "error_type": err_type,
                    },
                )

        # 503/500: sobrecarga o error temporal del proveedor (p.ej. Gemini
        # 'high demand'). Reintentable con backoff igual que el rate limit.
        if resp.status_code in (500, 502, 503, 504):
            raise LLMRateLimitError(
                f"Proveedor {self._config.name} no disponible temporalmente "
                f"(HTTP {resp.status_code})",
                details={"status": resp.status_code, "provider": self._config.name},
            )

        if resp.status_code != 200:
            raise LLMError(
                f"Error API {self._config.name}: {resp.status_code} - {resp.text[:200]}",
                details={"status": resp.status_code},
            )

        data = resp.json()
        latency = (time.perf_counter() - start) * 1000

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Respuesta con formato inesperado: {e}") from e

        usage = data.get("usage", {})

        # 200 OK con contenido vacío: visto en producción con modelos
        # "razonadores" (p.ej. terra/gpt-5.6 tras perder `temperature` por
        # autoadaptación) que consumen el presupuesto de tokens en
        # razonamiento interno invisible y no dejan nada en `content`. No
        # es un error de forma ni una indisponibilidad declarada, pero
        # tampoco es usable — se trata como LLMRateLimitError para
        # reutilizar el mismo reintento (@retry de este método) y la
        # misma rotación de proveedor (LLMModelRouter.run) que ya existen,
        # en vez de que el llamador (p.ej. WriterAgent) acepte un
        # resultado vacío como si fuera válido.
        if not content or not content.strip():
            finish_reason = (
                (data.get("choices") or [{}])[0].get("finish_reason")
                if isinstance(data.get("choices"), list)
                else None
            )
            # El modelo consumió tokens reales (razonamiento invisible)
            # aunque no haya contenido visible — sin contabilizarlo aquí,
            # el guard de presupuesto (llm/budget.py) subestima justo las
            # llamadas más caras (las que se van en pensar y no producen
            # nada). Se registra ANTES de lanzar, con el mismo mecanismo
            # que una respuesta normal (_record_usage).
            self._record_usage(
                LLMResponse(
                    content="",
                    model=data.get("model", self._config.model),
                    provider=self._config.name,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            )
            self._log.warning(
                "llm_empty_content",
                finish_reason=finish_reason,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
            raise LLMEmptyResponseError(
                f"{self._config.name} devolvió contenido vacío "
                f"(finish_reason={finish_reason})",
                details={"provider": self._config.name, "finish_reason": finish_reason},
            )

        self._log.debug(
            "llm_call_completed",
            latency_ms=round(latency, 1),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

        return LLMResponse(
            content=content,
            model=data.get("model", self._config.model),
            provider=self._config.name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
        )

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> LLMProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


class LLMRouter:
    """Enruta requests a múltiples proveedores LLM."""

    def __init__(self, providers: dict[str, LLMProvider]) -> None:
        if not providers:
            raise LLMError("LLMRouter requiere al menos un proveedor")
        self._providers = providers
        self._default_name: str = next(iter(providers))

    def get(self, name: str | None = None) -> LLMProvider:
        if name and name in self._providers:
            return self._providers[name]
        return self._providers[self._default_name]

    async def generate(
        self,
        prompt: str,
        provider_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        return await self.get(provider_name).generate(prompt, **kwargs)

    async def close_all(self) -> None:
        for p in self._providers.values():
            await p.close()
