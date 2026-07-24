"""Contenido vacío en una respuesta 200 OK (incidente run_431a9c5d,
24-jul-2026): visto en producción con modelos "razonadores" que gastan
el presupuesto de tokens en razonamiento interno invisible. El proveedor
debe tratarlo como una LLMEmptyResponseError (subclase de
LLMRateLimitError) para reutilizar el reintento existente
(LLMProvider._call_api) y la rotación de proveedor (LLMModelRouter.run).
"""

from __future__ import annotations

import httpx
import pytest

from creative_engine.core.config import LLMProviderConfig
from creative_engine.core.exceptions import LLMEmptyResponseError, LLMError
from creative_engine.llm.provider import LLMProvider
from creative_engine.llm.router import LLMModelRouter

EMPTY_RESPONSE = {
    "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 300, "total_tokens": 305},
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
    async def test_persistent_empty_content_raises_after_retry(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=EMPTY_RESPONSE)

        provider = _provider("terra", handler)

        with pytest.raises(LLMEmptyResponseError):
            await provider.generate("prompt de prueba")

        # El decorador @retry de _call_api reintenta una vez contra el
        # MISMO proveedor antes de rendirse (stop_after_attempt(2)).
        assert calls["n"] == 2

    async def test_recovers_on_retry_with_same_provider(self) -> None:
        """A veces el segundo intento SÍ trae contenido: no hace falta
        rotar de proveedor si el reintento simple ya resuelve."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=EMPTY_RESPONSE)
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


class TestRouterRotatesOnEmptyContent:
    async def test_router_fails_over_to_next_provider(self) -> None:
        """Fase del incidente run_431a9c5d: terra devuelve vacío
        persistentemente; el router debe rotar a luna y traer contenido real."""

        def always_empty(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=EMPTY_RESPONSE)

        def always_ok(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=OK_RESPONSE)

        terra = _provider("terra", always_empty)
        luna = _provider("luna", always_ok)

        router = LLMModelRouter(providers={"terra": terra, "luna": luna})
        result = await router.run("writer", "generate", "prompt de prueba")

        assert result == "Informe real con contenido."

    async def test_all_providers_empty_raises(self) -> None:
        def always_empty(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=EMPTY_RESPONSE)

        terra = _provider("terra", always_empty)
        luna = _provider("luna", always_empty)

        router = LLMModelRouter(providers={"terra": terra, "luna": luna})

        with pytest.raises(LLMError):
            await router.run("writer", "generate", "prompt de prueba")
