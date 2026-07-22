"""Autoadaptación del parámetro de límite de tokens (max_tokens vs
max_completion_tokens) en LLMProvider.

Incidente 21/22-jul-2026: terra y luna (OpenAI real) sin `type=openai`
enviaban `max_tokens`, recibían 400 invalid_request_error y quedaban
deshabilitados todo el run. El provider ahora detecta ese 400 concreto,
cambia el parámetro, reintenta una vez y recuerda la elección.
"""

import httpx
import pytest

from creative_engine.core.config import LLMProviderConfig
from creative_engine.core.exceptions import LLMInvalidRequestError
from creative_engine.llm.provider import LLMProvider

ERR_MAX_TOKENS = {
    "error": {
        "type": "invalid_request_error",
        "message": (
            "Unsupported parameter: 'max_tokens' is not supported with this "
            "model. Use 'max_completion_tokens' instead."
        ),
        "param": "max_tokens",
    }
}

ERR_OTHER_400 = {
    "error": {"type": "invalid_request_error", "message": "Invalid 'messages'."}
}

OK_RESPONSE = {
    "choices": [{"message": {"content": "hola"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    "model": "gpt-test",
}


def _provider_with_transport(handler, provider_type: str = "generic") -> LLMProvider:
    config = LLMProviderConfig(
        name="terra", model="gpt-test", type=provider_type, min_interval_seconds=0.0
    )
    provider = LLMProvider(config)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com/v1/",
    )
    return provider


class TestTokenParamAutoAdapt:
    @pytest.mark.asyncio
    async def test_adapts_on_max_tokens_rejection_and_remembers(self) -> None:
        """Sin type=openai: primer 400 → reintento con max_completion_tokens
        → éxito; las llamadas siguientes ya no envían max_tokens."""
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            body = json.loads(request.content)
            payloads.append(body)
            if "max_tokens" in body:
                return httpx.Response(400, json=ERR_MAX_TOKENS)
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider_with_transport(handler, provider_type="generic")
        result = await provider.generate("hola")
        assert result == "hola"
        # 1ª llamada con max_tokens (rechazada), 2ª adaptada
        assert "max_tokens" in payloads[0]
        assert "max_completion_tokens" in payloads[1]
        assert "max_tokens" not in payloads[1]

        # La elección se recuerda: la siguiente llamada va directa
        await provider.generate("otra")
        assert "max_completion_tokens" in payloads[2]
        assert len(payloads) == 3

    @pytest.mark.asyncio
    async def test_type_openai_needs_no_adaptation(self) -> None:
        """Con type=openai la primera llamada ya usa max_completion_tokens."""
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            payloads.append(json.loads(request.content))
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider_with_transport(handler, provider_type="openai")
        await provider.generate("hola")
        assert "max_completion_tokens" in payloads[0]
        assert len(payloads) == 1

    @pytest.mark.asyncio
    async def test_unrelated_400_still_raises(self) -> None:
        """Un 400 que no es del parámetro de tokens sigue levantando
        LLMInvalidRequestError (y por tanto deshabilita el proveedor)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json=ERR_OTHER_400)

        provider = _provider_with_transport(handler)
        with pytest.raises(LLMInvalidRequestError):
            await provider.generate("hola")

    @pytest.mark.asyncio
    async def test_no_infinite_loop_if_both_params_rejected(self) -> None:
        """Proveedor patológico que rechaza ambos parámetros: un solo
        reintento y error, nunca bucle."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            calls["n"] += 1
            body = json.loads(request.content)
            if "max_tokens" in body:
                return httpx.Response(400, json=ERR_MAX_TOKENS)
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Unsupported parameter: 'max_completion_tokens'.",
                    }
                },
            )

        provider = _provider_with_transport(handler)
        with pytest.raises(LLMInvalidRequestError):
            await provider.generate("hola")
        assert calls["n"] == 2
