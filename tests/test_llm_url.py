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
