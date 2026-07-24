"""Contenido vacío en una respuesta 200 OK (incidente run_431a9c5d,
24-jul-2026, y su seguimiento en la revisión de Qodo del PR #9).

Visto en producción con modelos "razonadores" que gastan el presupuesto
de tokens en razonamiento interno invisible. El proveedor lo trata como
LLMEmptyResponseError (subclase de LLMRateLimitError) para reutilizar
el reintento existente (LLMProvider._call_api) y la rotación de
proveedor (LLMModelRouter.run) — con matices añadidos tras la revisión:

1. El usage de los intentos vacíos se contabiliza ANTES de lanzar (el
   guard de presupuesto no debe subestimar las llamadas que más cuestan).
2. finish_reason="length" en una respuesta vacía salta el reintento
   contra el mismo proveedor: rota directamente (ese reintento repetiría
   el mismo resultado, es una llamada cara desperdiciada).
3. El tope de tokens se amplía para proveedores marcados como
   "razonadores" (temperature en _unsupported_params).
4. Una respuesta vacía NO activa el disyuntor de cooldown creciente: un
   proveedor sano no debe quedar en cuarentena por esto.
"""

from __future__ import annotations

import time as _time

import httpx
import pytest
import structlog.testing

from creative_engine.core.config import LLMProviderConfig
from creative_engine.core.exceptions import LLMEmptyResponseError, LLMError
from creative_engine.llm.provider import (
    _REASONING_TOKEN_FLOOR,
    _REASONING_TOKEN_MULTIPLIER,
    LLMProvider,
)
from creative_engine.llm.router import LLMModelRouter

# finish_reason="length": el caso real del incidente — el modelo agotó
# max_tokens/max_completion_tokens razonando y no dejó nada visible.
# Reintentar con el mismo prompt/tope repetiría el mismo resultado.
EMPTY_RESPONSE_LENGTH = {
    "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 300, "total_tokens": 305},
    "model": "gpt-5.6-sol",
}

# Vacío por otra causa (p.ej. un filtro de contenido puntual): SÍ vale la
# pena reintentar una vez contra el mismo proveedor.
EMPTY_RESPONSE_OTHER = {
    "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    "model": "gpt-5.6-sol",
}

OK_RESPONSE = {
    "choices": [{"message": {"content": "Informe real con contenido."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    "model": "gpt-5.6-sol",
}


def _provider(name: str, handler) -> LLMProvider:
    config = LLMProviderConfig(name=name, model="gpt-test", min_interval_seconds=0.0)
    provider = LLMProvider(config)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1/"
    )
    return provider


class TestEmptyContentDetection:
    async def test_length_finish_reason_skips_same_provider_retry(self) -> None:
        """El caso real del incidente: no vale la pena reintentar contra
        el mismo proveedor si se quedó sin presupuesto razonando."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=EMPTY_RESPONSE_LENGTH)

        provider = _provider("terra", handler)

        with pytest.raises(LLMEmptyResponseError):
            await provider.generate("prompt de prueba")

        assert calls["n"] == 1  # sin reintento: se rinde de inmediato

    async def test_non_length_empty_content_retries_same_provider(self) -> None:
        """Vacío por otra causa (finish_reason distinto de "length"): el
        reintento simple contra el mismo proveedor SÍ sigue teniendo sentido."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=EMPTY_RESPONSE_OTHER)

        provider = _provider("terra", handler)

        with pytest.raises(LLMEmptyResponseError):
            await provider.generate("prompt de prueba")

        assert calls["n"] == 2  # @retry sí reintenta una vez

    async def test_recovers_on_retry_with_same_provider(self) -> None:
        """A veces el segundo intento SÍ trae contenido: no hace falta
        rotar de proveedor si el reintento simple ya resuelve."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=EMPTY_RESPONSE_OTHER)
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider("terra", handler)
        result = await provider.generate("prompt de prueba")

        assert result == "Informe real con contenido."
        assert calls["n"] == 2

    async def test_whitespace_only_content_also_treated_as_empty(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "   \n  "}, "finish_reason": "stop"}],
                    "usage": {},
                    "model": "gpt-test",
                },
            )

        provider = _provider("terra", handler)

        with pytest.raises(LLMEmptyResponseError):
            await provider.generate("prompt de prueba")

    async def test_non_empty_content_is_unaffected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider("terra", handler)
        result = await provider.generate("prompt de prueba")

        assert result == "Informe real con contenido."


class TestEmptyContentUsageAccounting:
    """Hallazgo de la revisión: el usage de los intentos vacíos debe
    contabilizarse ANTES de lanzar, o el guard de presupuesto subestima
    justo las llamadas más caras (tokens de razonamiento sin contenido)."""

    async def test_usage_recorded_before_raising(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=EMPTY_RESPONSE_LENGTH)

        provider = _provider("terra", handler)

        with pytest.raises(LLMEmptyResponseError):
            await provider.generate("prompt de prueba")

        assert provider.total_calls == 1
        assert provider.total_prompt_tokens == 5
        assert provider.total_completion_tokens == 300

    async def test_usage_accumulates_across_retries(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=EMPTY_RESPONSE_OTHER)

        provider = _provider("terra", handler)

        with pytest.raises(LLMEmptyResponseError):
            await provider.generate("prompt de prueba")

        # 2 intentos (finish_reason distinto de "length" sí reintenta),
        # cada uno consumió tokens reales — ambos deben contar.
        assert provider.total_calls == 2
        assert provider.total_prompt_tokens == 10
        assert provider.total_completion_tokens == 4

    async def test_llm_empty_content_event_logged(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=EMPTY_RESPONSE_LENGTH)

        provider = _provider("terra", handler)

        with structlog.testing.capture_logs() as logs, pytest.raises(LLMEmptyResponseError):
            await provider.generate("prompt de prueba")

        events = [log for log in logs if log.get("event") == "llm_empty_content"]
        assert len(events) == 1
        assert events[0]["finish_reason"] == "length"
        assert events[0]["prompt_tokens"] == 5
        assert events[0]["completion_tokens"] == 300


class TestReasoningTokenBoost:
    """Causa raíz probable: en razonadores, max_tokens/max_completion_tokens
    incluye el razonamiento invisible — un tope ajustado (p.ej. el del
    writer, 2000) se agota pensando. Se amplía para proveedores marcados
    como razonadores (temperature en _unsupported_params)."""

    async def test_boosts_max_tokens_for_reasoning_flagged_provider(self) -> None:
        sent_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            sent_payloads.append(json.loads(request.content))
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider("terra", handler)
        provider._unsupported_params.add("temperature")  # ya se detectó antes

        await provider.generate("prompt de prueba", max_tokens=2000)

        sent = sent_payloads[0].get("max_tokens") or sent_payloads[0].get(
            "max_completion_tokens"
        )
        assert sent == max(2000 * _REASONING_TOKEN_MULTIPLIER, _REASONING_TOKEN_FLOOR)

    async def test_does_not_boost_for_normal_provider(self) -> None:
        sent_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            sent_payloads.append(json.loads(request.content))
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider("terra", handler)

        await provider.generate("prompt de prueba", max_tokens=2000)

        sent = sent_payloads[0].get("max_tokens") or sent_payloads[0].get(
            "max_completion_tokens"
        )
        assert sent == 2000

    async def test_boost_floor_applies_to_small_requests(self) -> None:
        sent_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            sent_payloads.append(json.loads(request.content))
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider("terra", handler)
        provider._unsupported_params.add("temperature")

        await provider.generate("prompt de prueba", max_tokens=500)  # 500*3=1500 < suelo

        sent = sent_payloads[0].get("max_tokens") or sent_payloads[0].get(
            "max_completion_tokens"
        )
        assert sent == _REASONING_TOKEN_FLOOR


class TestRouterRotatesOnEmptyContent:
    async def test_router_fails_over_to_next_provider(self) -> None:
        """Fase del incidente run_431a9c5d: terra devuelve vacío
        persistentemente; el router debe rotar a luna y traer contenido real."""

        def always_empty(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=EMPTY_RESPONSE_LENGTH)

        def always_ok(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=OK_RESPONSE)

        terra = _provider("terra", always_empty)
        luna = _provider("luna", always_ok)

        router = LLMModelRouter(providers={"terra": terra, "luna": luna})
        result = await router.run("writer", "generate", "prompt de prueba")

        assert result == "Informe real con contenido."

    async def test_all_providers_empty_raises(self) -> None:
        def always_empty(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=EMPTY_RESPONSE_LENGTH)

        terra = _provider("terra", always_empty)
        luna = _provider("luna", always_empty)

        router = LLMModelRouter(providers={"terra": terra, "luna": luna})

        with pytest.raises(LLMError):
            await router.run("writer", "generate", "prompt de prueba")

    async def test_max_providers_caps_attempts(self) -> None:
        """Límite explícito de intentos por llamada (review Qodo): con
        max_providers=2, un tercer proveedor de la cadena ni se intenta."""
        attempts = {"terra": 0, "luna": 0, "zai": 0}

        def make_handler(name: str):
            def handler(request: httpx.Request) -> httpx.Response:
                attempts[name] += 1
                return httpx.Response(200, json=EMPTY_RESPONSE_LENGTH)

            return handler

        terra = _provider("terra", make_handler("terra"))
        luna = _provider("luna", make_handler("luna"))
        zai = _provider("zai", make_handler("zai"))

        router = LLMModelRouter(providers={"terra": terra, "luna": luna, "zai": zai})

        with pytest.raises(LLMError):
            await router.run("writer", "generate", "prompt de prueba", max_providers=2)

        assert attempts["terra"] == 1
        assert attempts["luna"] == 1
        assert attempts["zai"] == 0  # nunca se llegó a intentar

    async def test_empty_response_does_not_trigger_growing_cooldown(self) -> None:
        """Confirma que una respuesta vacía no pone al proveedor en
        cuarentena creciente como un rate limit sostenido — un proveedor
        sano no debe perder minutos por un informe vacío puntual."""

        def always_empty(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=EMPTY_RESPONSE_LENGTH)

        def always_ok(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=OK_RESPONSE)

        terra = _provider("terra", always_empty)
        luna = _provider("luna", always_ok)

        router = LLMModelRouter(providers={"terra": terra, "luna": luna})
        before = _time.monotonic()
        result = await router.run("writer", "generate", "prompt de prueba")
        assert result == "Informe real con contenido."

        breaker = router._breakers["terra"]
        assert breaker.failures == 0
        assert breaker.open_until <= before  # sin cooldown activo

    async def test_true_rate_limit_still_triggers_cooldown(self) -> None:
        """Control: un 429 de verdad SÍ debe seguir activando el
        disyuntor — solo el contenido vacío queda exento."""

        def always_429(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        def always_ok(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=OK_RESPONSE)

        terra = _provider("terra", always_429)
        luna = _provider("luna", always_ok)

        router = LLMModelRouter(providers={"terra": terra, "luna": luna})
        await router.run("writer", "generate", "prompt de prueba")

        breaker = router._breakers["terra"]
        assert breaker.failures == 1
        assert breaker.open_until > _time.monotonic()
