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

ERR_TEMPERATURE = {
    "error": {
        "type": "invalid_request_error",
        "message": (
            "Unsupported value: 'temperature' does not support 0.9 with this "
            "model. Only the default (1) value is supported."
        ),
        "param": "temperature",
    }
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


class TestGenericParamAutoAdapt:
    """Autoadaptación generalizada: cualquier 400 'Unsupported parameter/value'
    elimina ese parámetro del payload, no solo max_tokens.

    Incidente 22-jul-2026 (continuación): la familia gpt-5.6 solo acepta
    temperature=1 y terra fue deshabilitado con "400 Unsupported value:
    'temperature' does not support 0.9". Mañana puede ser otro parámetro.
    """

    @pytest.mark.asyncio
    async def test_temperature_400_drops_param_and_succeeds(self) -> None:
        """400 de temperature -> reintento sin temperature -> éxito."""
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            body = json.loads(request.content)
            payloads.append(body)
            if "temperature" in body:
                return httpx.Response(400, json=ERR_TEMPERATURE)
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider_with_transport(handler)
        result = await provider.generate("hola", temperature=0.9)

        assert result == "hola"
        assert "temperature" in payloads[0]
        assert "temperature" not in payloads[1]
        assert "temperature" in provider._unsupported_params

    @pytest.mark.asyncio
    async def test_temperature_remembered_no_second_400(self) -> None:
        """Tras el drop, la siguiente llamada ya no incluye temperature ni
        vuelve a disparar un 400."""
        payloads: list[dict] = []
        rejections = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            body = json.loads(request.content)
            payloads.append(body)
            if "temperature" in body:
                rejections["n"] += 1
                return httpx.Response(400, json=ERR_TEMPERATURE)
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider_with_transport(handler)
        await provider.generate("hola", temperature=0.9)
        await provider.generate("otra", temperature=0.9)

        assert rejections["n"] == 1  # solo la primera llamada disparó el 400
        assert len(payloads) == 3  # rechazada + reintento + segunda directa
        assert "temperature" not in payloads[-1]

    @pytest.mark.asyncio
    async def test_high_temperature_dropped_adds_risk_hint(self) -> None:
        """Con temperature dropeada y t alta, el system prompt pide arriesgar."""
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            body = json.loads(request.content)
            payloads.append(body)
            if "temperature" in body:
                return httpx.Response(400, json=ERR_TEMPERATURE)
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider_with_transport(handler)
        await provider.generate(
            "hola", system_prompt="Eres un generador.", temperature=0.95
        )

        system_msg = next(
            m["content"] for m in payloads[-1]["messages"] if m["role"] == "system"
        )
        assert "Eres un generador." in system_msg
        assert "Arriesga" in system_msg

    @pytest.mark.asyncio
    async def test_low_temperature_dropped_adds_conservative_hint(self) -> None:
        """Con temperature dropeada y t baja, el system prompt pide rigor."""
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            body = json.loads(request.content)
            payloads.append(body)
            if "temperature" in body:
                return httpx.Response(400, json=ERR_TEMPERATURE)
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider_with_transport(handler)
        await provider.generate(
            "hola", system_prompt="Eres un generador.", temperature=0.2
        )

        system_msg = next(
            m["content"] for m in payloads[-1]["messages"] if m["role"] == "system"
        )
        assert "Eres un generador." in system_msg
        assert "riguroso y conservador" in system_msg

    @pytest.mark.asyncio
    async def test_mid_temperature_dropped_adds_no_hint(self) -> None:
        """Temperatura media (0.4-0.8): sin instrucción de estilo añadida."""
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            body = json.loads(request.content)
            payloads.append(body)
            if "temperature" in body:
                return httpx.Response(400, json=ERR_TEMPERATURE)
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider_with_transport(handler)
        await provider.generate(
            "hola", system_prompt="Eres un generador.", temperature=0.6
        )

        system_msg = next(
            m["content"] for m in payloads[-1]["messages"] if m["role"] == "system"
        )
        assert system_msg == "Eres un generador."

    @pytest.mark.asyncio
    async def test_max_tokens_swap_and_temperature_drop_coexist(self) -> None:
        """El swap especial de max_tokens sigue funcionando aunque el mismo
        provider además haya dropeado temperature genéricamente."""
        payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            body = json.loads(request.content)
            payloads.append(body)
            if "max_tokens" in body:
                return httpx.Response(400, json=ERR_MAX_TOKENS)
            if "temperature" in body:
                return httpx.Response(400, json=ERR_TEMPERATURE)
            return httpx.Response(200, json=OK_RESPONSE)

        provider = _provider_with_transport(handler, provider_type="generic")
        result = await provider.generate("hola", temperature=0.9)

        assert result == "hola"
        assert len(payloads) == 3
        assert "max_tokens" in payloads[0]
        assert "max_completion_tokens" in payloads[1] and "temperature" in payloads[1]
        assert "max_completion_tokens" in payloads[2]
        assert "temperature" not in payloads[2]

    @pytest.mark.asyncio
    async def test_pathological_provider_stops_after_3_generic_drops(self) -> None:
        """Un proveedor que rechaza parámetro tras parámetro se detiene a
        los 3 intentos y propaga el error, nunca en bucle."""
        calls = {"n": 0}
        reject_order = ["foo", "bar", "baz", "qux"]

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            idx = calls["n"] - 1
            if idx < len(reject_order):
                param = reject_order[idx]
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"Unsupported parameter: '{param}' is not supported.",
                        }
                    },
                )
            return httpx.Response(200, json=OK_RESPONSE)

        config = LLMProviderConfig(
            name="terra",
            model="gpt-test",
            min_interval_seconds=0.0,
            extra_body={"foo": 1, "bar": 2, "baz": 3, "qux": 4},
        )
        provider = LLMProvider(config)
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.openai.com/v1/",
        )

        with pytest.raises(LLMInvalidRequestError):
            await provider.generate("hola")

        assert calls["n"] == 4  # 3 adaptaciones + el intento final que falla
        assert provider._unsupported_params == {"foo", "bar", "baz"}
