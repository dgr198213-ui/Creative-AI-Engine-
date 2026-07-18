"""Abstracción multi-proveedor LLM con retry, rate limiting y router.

Soporta cualquier API OpenAI-compatible (OpenAI, DeepSeek, Qwen, Ollama...).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..core.config import LLMProviderConfig
from ..core.exceptions import LLMAuthError, LLMError, LLMRateLimitError

logger = structlog.get_logger(__name__)


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
    ) -> str:
        """Genera texto a partir de un prompt."""
        async with self._semaphore:
            await self._rate_limit()
            response = await self._call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature if temperature is not None else self._config.temperature,
                max_tokens=max_tokens or self._config.max_tokens,
            )
            return response.content

    async def generate_structured(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
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
            try:
                return parse_llm_json(response.content)
            except Exception as e:
                self._log.error("structured_parse_failed", raw=response.content[:300])
                raise LLMError(f"Respuesta no es JSON válido: {e}") from e

    @retry(
        retry=retry_if_exception_type((LLMRateLimitError, httpx.ConnectError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    async def _call_api(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Llamada real a la API con retry."""
        start = time.perf_counter()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

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
