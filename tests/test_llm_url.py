"""Tests de construcción de URL del proveedor LLM (multi-backend)."""

import pytest

from creative_engine.core.config import LLMProviderConfig, SecretStr
from creative_engine.llm.provider import LLMProvider


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai",  # sin barra final
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
        (None, "https://api.openai.com/v1/chat/completions"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
    ],
)
async def test_chat_completions_url(base_url: str | None, expected: str) -> None:
    """La ruta base (p.ej. /v1beta/openai/) debe preservarse en el POST."""
    config = LLMProviderConfig(
        name="test",
        api_key=SecretStr("k"),
        base_url=base_url,
    )
    provider = LLMProvider(config)
    try:
        req = provider._client.build_request("POST", "chat/completions")
        assert str(req.url) == expected
    finally:
        await provider.close()


async def test_structured_parses_markdown_wrapped_json() -> None:
    """Gemini envuelve el JSON en ```json ... ```; debe parsearse igual."""
    from unittest.mock import AsyncMock, patch

    from creative_engine.llm.provider import LLMProvider, LLMResponse

    provider = LLMProvider(LLMProviderConfig(name="gemini", api_key=SecretStr("k")))
    fake = LLMResponse(
        content='```json\n{"score": 0.82, "feedback": "buena idea"}\n```',
        model="g",
        provider="gemini",
    )
    try:
        with patch.object(provider, "_call_api", AsyncMock(return_value=fake)):
            data = await provider.generate_structured("evalúa esto")
        assert data["score"] == 0.82
        assert data["feedback"] == "buena idea"
    finally:
        await provider.close()


async def test_structured_parses_json_with_preamble() -> None:
    """Texto antes del JSON tampoco debe romper el parseo."""
    from unittest.mock import AsyncMock, patch

    from creative_engine.llm.provider import LLMProvider, LLMResponse

    provider = LLMProvider(LLMProviderConfig(name="test", api_key=SecretStr("k")))
    fake = LLMResponse(
        content='Claro, aquí está:\n{"score": 0.5, "feedback": "ok"}',
        model="g",
        provider="test",
    )
    try:
        with patch.object(provider, "_call_api", AsyncMock(return_value=fake)):
            data = await provider.generate_structured("evalúa")
        assert data["score"] == 0.5
    finally:
        await provider.close()


async def test_network_error_becomes_retryable(monkeypatch) -> None:
    """Timeouts/errores de red se convierten en LLMRateLimitError (reintentable)."""
    from unittest.mock import AsyncMock, patch

    import httpx

    from creative_engine.core.exceptions import LLMRateLimitError
    from creative_engine.llm.provider import LLMProvider

    async def _no_sleep(*a, **k):
        return None

    monkeypatch.setattr("tenacity.asyncio.sleep", _no_sleep, raising=False)
    provider = LLMProvider(LLMProviderConfig(name="zai", api_key=SecretStr("k")))
    # una sola llamada interna sin reintentos: probamos la conversión directa
    try:
        with patch.object(
            provider._client, "post", AsyncMock(side_effect=httpx.ReadError(""))
        ), pytest.raises(LLMRateLimitError):
            await provider._call_api.__wrapped__(provider, "hola")
    finally:
        await provider.close()


async def test_http_error_message_never_empty() -> None:
    """Un HTTPError sin mensaje debe producir un error con el tipo, no vacío."""
    from unittest.mock import AsyncMock, patch

    import httpx

    from creative_engine.core.exceptions import LLMError

    provider = LLMProvider(LLMProviderConfig(name="zai", api_key=SecretStr("k")))
    try:
        with patch.object(
            provider._client, "post", AsyncMock(side_effect=httpx.HTTPError(""))
        ), pytest.raises(LLMError) as exc:
            await provider._call_api.__wrapped__(provider, "hola")
        assert "HTTPError" in str(exc.value)  # el tipo aparece aunque el msg sea vacío
    finally:
        await provider.close()


async def test_extra_body_merged_into_payload() -> None:
    """extra_body (p.ej. desactivar thinking de GLM) debe llegar al payload."""
    from unittest.mock import patch

    from creative_engine.llm.provider import LLMProvider

    provider = LLMProvider(
        LLMProviderConfig(
            name="zai",
            api_key=SecretStr("k"),
            extra_body={"thinking": {"type": "disabled"}},
        )
    )
    captured: dict = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
                "model": "glm-4.5-flash",
            }

    async def fake_post(url, json=None):
        captured.update(json or {})
        return FakeResp()

    try:
        with patch.object(provider._client, "post", side_effect=fake_post):
            await provider._call_api.__wrapped__(provider, "hola")
        assert captured.get("thinking") == {"type": "disabled"}
    finally:
        await provider.close()


async def test_glm_thinking_disabled_by_default() -> None:
    """Modelos GLM: el modo razonador se desactiva solo (timeouts en free tier)."""
    from unittest.mock import patch

    from creative_engine.llm.provider import LLMProvider

    provider = LLMProvider(
        LLMProviderConfig(name="zai", api_key=SecretStr("k"), model="glm-4.5-flash")
    )
    captured: dict = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    async def fake_post(url, json=None):
        captured.update(json or {})
        return FakeResp()

    try:
        with patch.object(provider._client, "post", side_effect=fake_post):
            await provider._call_api.__wrapped__(provider, "hola")
        assert captured.get("thinking") == {"type": "disabled"}
    finally:
        await provider.close()


async def test_openai_type_uses_max_completion_tokens() -> None:
    """Proveedores tipo "openai" envían max_completion_tokens, no max_tokens.

    Reproduce el fallo de producción: terra (OpenAI gpt-5.6-sol) rechazaba
    max_tokens con 400 invalid_request_error.
    """
    from unittest.mock import patch

    from creative_engine.llm.provider import LLMProvider

    provider = LLMProvider(
        LLMProviderConfig(
            name="terra", api_key=SecretStr("k"), type="openai", model="gpt-5.6-sol"
        )
    )
    captured: dict = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    async def fake_post(url, json=None):
        captured.update(json or {})
        return FakeResp()

    try:
        with patch.object(provider._client, "post", side_effect=fake_post):
            await provider._call_api.__wrapped__(provider, "hola", max_tokens=123)
        assert captured.get("max_completion_tokens") == 123
        assert "max_tokens" not in captured
    finally:
        await provider.close()


async def test_generic_type_keeps_max_tokens() -> None:
    """El resto de proveedores (por defecto "generic") siguen usando max_tokens."""
    from unittest.mock import patch

    from creative_engine.llm.provider import LLMProvider

    provider = LLMProvider(LLMProviderConfig(name="zai", api_key=SecretStr("k")))
    captured: dict = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    async def fake_post(url, json=None):
        captured.update(json or {})
        return FakeResp()

    try:
        with patch.object(provider._client, "post", side_effect=fake_post):
            await provider._call_api.__wrapped__(provider, "hola", max_tokens=456)
        assert captured.get("max_tokens") == 456
        assert "max_completion_tokens" not in captured
    finally:
        await provider.close()


async def test_400_invalid_request_error_raises_specific_exception() -> None:
    """Un 400 con body invalid_request_error debe distinguirse de otros 400."""
    from unittest.mock import AsyncMock, patch

    from creative_engine.core.exceptions import LLMInvalidRequestError
    from creative_engine.llm.provider import LLMProvider

    provider = LLMProvider(
        LLMProviderConfig(name="terra", api_key=SecretStr("k"), model="gpt-5.6-sol")
    )

    class FakeResp:
        status_code = 400

        @staticmethod
        def json():
            return {
                "error": {
                    "message": "Unsupported parameter: 'max_tokens'.",
                    "type": "invalid_request_error",
                }
            }

        text = "invalid_request_error"

    try:
        with (
            patch.object(provider._client, "post", AsyncMock(return_value=FakeResp())),
            pytest.raises(LLMInvalidRequestError),
        ):
            await provider._call_api.__wrapped__(provider, "hola")
    finally:
        await provider.close()


async def test_400_without_invalid_request_type_stays_generic_error() -> None:
    """Un 400 sin el tipo invalid_request_error no debe activar la rotación especial."""
    from unittest.mock import AsyncMock, patch

    from creative_engine.core.exceptions import LLMError, LLMInvalidRequestError
    from creative_engine.llm.provider import LLMProvider

    provider = LLMProvider(LLMProviderConfig(name="zai", api_key=SecretStr("k")))

    class FakeResp:
        status_code = 400

        @staticmethod
        def json():
            return {"error": {"message": "algo distinto", "type": "other_error"}}

        text = "other_error"

    try:
        with patch.object(provider._client, "post", AsyncMock(return_value=FakeResp())):
            with pytest.raises(LLMError) as exc:
                await provider._call_api.__wrapped__(provider, "hola")
            assert not isinstance(exc.value, LLMInvalidRequestError)
    finally:
        await provider.close()


async def test_glm_thinking_respects_explicit_config() -> None:
    """Si el usuario configura thinking explícitamente, se respeta."""
    from unittest.mock import patch

    from creative_engine.llm.provider import LLMProvider

    provider = LLMProvider(
        LLMProviderConfig(
            name="zai",
            api_key=SecretStr("k"),
            model="glm-4.7-flash",
            extra_body={"thinking": {"type": "enabled"}},
        )
    )
    captured: dict = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    async def fake_post(url, json=None):
        captured.update(json or {})
        return FakeResp()

    try:
        with patch.object(provider._client, "post", side_effect=fake_post):
            await provider._call_api.__wrapped__(provider, "hola")
        assert captured.get("thinking") == {"type": "enabled"}
    finally:
        await provider.close()
