"""Guardrails de seguridad de la API pública (Fase 5, bloque 2).

C1 (API key), C2 (cap de presupuesto por run) y A2 (docs off en
producción) ya estaban implementados de una auditoría previa, pero sin
tests que los protejan de una regresión — este archivo los cubre. Sin
red ni BD: middleware y guardrails se prueban con FastAPI/Starlette en
memoria (ASGITransport), y el cap de presupuesto es una función pura.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import structlog.testing
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from httpx import ASGITransport, AsyncClient

from creative_engine.api.auth import ApiKeyMiddleware
from creative_engine.api.guardrails import enforce_request_budget
from creative_engine.core import config
from creative_engine.core.config import Settings
from creative_engine.core.models import DomainName, EvolutionRequest


def _make_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.get("/")
    async def root() -> PlainTextResponse:
        return PlainTextResponse("panel")

    @app.get("/static/app.js")
    async def static_js() -> PlainTextResponse:
        return PlainTextResponse("// js")

    @app.get("/api/v1/ideas")
    async def protected() -> PlainTextResponse:
        return PlainTextResponse("secreto")

    return app


@pytest.fixture(autouse=True)
def _reset_settings():
    """Aísla el singleton de settings entre tests (auth lee get_settings())."""
    yield
    config.reset_settings()


class TestApiKeyMiddleware:
    async def test_open_without_api_key_configured(self) -> None:
        s = Settings.load()
        s.api_key = ""
        config._settings = s

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/ideas")
            assert r.status_code == 200

    async def test_rejects_missing_key_when_configured(self) -> None:
        s = Settings.load()
        s.api_key = "secreta123"
        config._settings = s

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/ideas")
            assert r.status_code == 401

    async def test_rejects_wrong_key(self) -> None:
        s = Settings.load()
        s.api_key = "secreta123"
        config._settings = s

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/ideas", headers={"X-API-Key": "incorrecta"})
            assert r.status_code == 401

    async def test_accepts_correct_key_via_header(self) -> None:
        s = Settings.load()
        s.api_key = "secreta123"
        config._settings = s

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/ideas", headers={"X-API-Key": "secreta123"})
            assert r.status_code == 200

    async def test_accepts_correct_key_via_query_param(self) -> None:
        """Necesario para enlaces de descarga (GET simple sin cabeceras)."""
        s = Settings.load()
        s.api_key = "secreta123"
        config._settings = s

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/ideas?api_key=secreta123")
            assert r.status_code == 200

    @pytest.mark.parametrize("path", ["/health", "/", "/static/app.js"])
    async def test_public_paths_bypass_even_with_key_configured(self, path: str) -> None:
        s = Settings.load()
        s.api_key = "secreta123"
        config._settings = s

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get(path)
            assert r.status_code == 200


class TestApiKeyMissingWarning:
    async def test_warns_when_api_key_not_configured(self) -> None:
        from creative_engine.api.app import _warn_if_api_key_missing

        s = Settings.load()
        s.api_key = ""

        with structlog.testing.capture_logs() as logs:
            _warn_if_api_key_missing(s)

        events = [log["event"] for log in logs]
        assert "api_key_not_configured_endpoints_open" in events

    async def test_silent_when_api_key_configured(self) -> None:
        from creative_engine.api.app import _warn_if_api_key_missing

        s = Settings.load()
        s.api_key = "secreta123"

        with structlog.testing.capture_logs() as logs:
            _warn_if_api_key_missing(s)

        events = [log["event"] for log in logs]
        assert "api_key_not_configured_endpoints_open" not in events


class TestEvolutionBudgetCap:
    def test_rejects_request_over_cap(self) -> None:
        config._settings = None  # usa el cap por defecto (2000)
        request = EvolutionRequest(
            challenge="Un reto de prueba con longitud suficiente para validar",
            domain=DomainName.GENERIC,
            population_size=500,
            generations=200,  # 500 x 200 = 100.000 evaluaciones, muy por encima del cap
        )
        with pytest.raises(HTTPException) as exc_info:
            enforce_request_budget(request)
        assert exc_info.value.status_code == 422

    def test_accepts_request_within_cap(self) -> None:
        config._settings = None
        request = EvolutionRequest(
            challenge="Un reto de prueba con longitud suficiente para validar",
            domain=DomainName.GENERIC,
            population_size=10,
            generations=5,  # 50 evaluaciones, muy por debajo del cap
        )
        enforce_request_budget(request)  # no debe lanzar


class TestDocsDisabledInProduction:
    def test_docs_disabled_when_debug_false(self) -> None:
        s = Settings.load()
        s.debug = False
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()
        app.state.repository = AsyncMock()
        assert app.docs_url is None
        assert app.redoc_url is None
        assert app.openapi_url is None

    def test_docs_enabled_when_debug_true(self) -> None:
        s = Settings.load()
        s.debug = True
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()
        app.state.repository = AsyncMock()
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"
