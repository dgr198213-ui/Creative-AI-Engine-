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
