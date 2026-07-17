"""Tests del diagnóstico y de la resiliencia ante claves inválidas."""

from unittest.mock import AsyncMock

from creative_engine.core.config import Settings
from creative_engine.core.exceptions import LLMAuthError, LLMRateLimitError
from creative_engine.diagnostics import check_provider, routing_report
from creative_engine.llm.router import LLMModelRouter


def _provider(name: str) -> AsyncMock:
    p = AsyncMock()
    p.generate.return_value = "ok"
    p.close = AsyncMock()
    return p


class TestAuthFailover:
    async def test_invalid_key_fails_over_to_next_provider(self) -> None:
        """Reproduce el fallo de producción: Groq con clave inválida (401).

        Una clave rota en un proveedor no debe matar el run si hay otro.
        """
        broken = _provider("groq")
        broken.generate.side_effect = LLMAuthError("Invalid API Key")
        working = _provider("gemini")
        working.generate.return_value = "respuesta válida"

        router = LLMModelRouter({"groq": broken, "gemini": working})
        result = await router.for_role("generator").generate("x")

        assert result == "respuesta válida"
        working.generate.assert_awaited_once()


class TestCheckProvider:
    async def test_ok(self) -> None:
        p = _provider("gemini")
        result = await check_provider("gemini", p)
        assert result["status"] == "ok"
        assert "latency_ms" in result

    async def test_auth_error_classified(self) -> None:
        p = _provider("groq")
        p.generate.side_effect = LLMAuthError("Invalid API Key")
        result = await check_provider("groq", p)
        assert result["status"] == "auth_error"
        assert "API_KEY" in result["detail"]

    async def test_rate_limited_classified(self) -> None:
        p = _provider("gemini")
        p.generate.side_effect = LLMRateLimitError("saturado")
        result = await check_provider("gemini", p)
        assert result["status"] == "rate_limited"


class TestRoutingReport:
    def _settings(self, providers: list[str], spec: str = "") -> Settings:
        from creative_engine.core.config import LLMProviderConfig, SecretStr

        s = Settings()
        s.llm = {n: LLMProviderConfig(name=n, api_key=SecretStr("k")) for n in providers}
        s.routing_spec = spec
        return s

    def test_detects_unknown_provider_in_spec(self) -> None:
        s = self._settings(["default", "groq"], "evaluator=qroq,default")  # errata
        report = routing_report(s)
        assert any("inexistentes" in i for i in report["issues"])

    def test_warns_multi_provider_without_spec(self) -> None:
        s = self._settings(["default", "groq"], "")
        report = routing_report(s)
        assert any("sin CREATIVE_ROUTING_SPEC" in i for i in report["issues"])
        assert report["default_chain"] == ["default", "groq"]

    def test_clean_config_no_issues(self) -> None:
        s = self._settings(["gemini", "groq"], "evaluator=groq,gemini;generator=gemini,groq")
        report = routing_report(s)
        assert report["issues"] == []
        assert report["parsed"]["evaluator"] == ["groq", "gemini"]
