"""Tests del panel web y endpoints HTTP con app FastAPI y LLM simulado."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from creative_engine.core import config
from creative_engine.core.config import LLMProviderConfig, Settings
from creative_engine.core.exceptions import IdeaNotFoundError
from creative_engine.core.models import EvaluationScores, Idea


class _NullRepo:
    def __init__(self) -> None:
        self._store: dict[str, Idea] = {}

    async def store_idea(self, idea: Idea) -> Idea:
        self._store[idea.id] = idea
        return idea

    async def get_idea(self, idea_id: str) -> Idea:
        if idea_id not in self._store:
            raise IdeaNotFoundError(idea_id)
        return self._store[idea_id]

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...


@pytest.fixture
def app_client():
    s = Settings.load()
    s.llm = {"default": LLMProviderConfig(name="sim", api_key=config.SecretStr("x"))}
    config._settings = s

    from creative_engine.api.app import create_app

    app = create_app()
    app.state.repository = _NullRepo()
    transport = ASGITransport(app=app)
    return app, AsyncClient(transport=transport, base_url="http://test")


async def test_panel_served(app_client) -> None:
    _, client = app_client
    async with client:
        r = await client.get("/")
        assert r.status_code == 200
        assert "Creative AI Engine" in r.text
        assert "text/html" in r.headers["content-type"]


async def test_static_js_served(app_client) -> None:
    _, client = app_client
    async with client:
        r = await client.get("/static/app.js")
        assert r.status_code == 200
        assert "handleEvent" in r.text


async def test_health(app_client) -> None:
    _, client = app_client
    async with client:
        r = await client.get("/health")
        assert r.json()["status"] == "ok"


async def test_report_404_for_missing_idea(app_client) -> None:
    _, client = app_client
    async with client:
        r = await client.post("/api/v1/ideas/idea_missing/report")
        assert r.status_code == 404


async def test_report_on_demand(app_client) -> None:
    app, client = app_client
    idea = Idea(
        title="Bici solar plegable",
        description="Una bicicleta con paneles solares integrados y plegable.",
    )
    idea.evaluation = EvaluationScores(utility=0.8, feasibility=0.6, market_fit=0.5)
    await app.state.repository.store_idea(idea)

    sim = AsyncMock()
    sim.generate.return_value = "INFORME EJECUTIVO simulado."
    sim.close = AsyncMock()

    async with client:
        with patch("creative_engine.llm.factory.LLMProvider", return_value=sim):
            r = await client.post(f"/api/v1/ideas/{idea.id}/report")
    assert r.status_code == 200
    assert "simulado" in r.json()["report"]


async def test_history_endpoint_503_without_repo() -> None:
    """Sin persistencia, los endpoints de histórico responden 503 claro."""
    from httpx import ASGITransport, AsyncClient

    s = Settings.load()
    s.llm = {"default": LLMProviderConfig(name="sim", api_key=config.SecretStr("x"))}
    config._settings = s

    from creative_engine.api.app import create_app

    app = create_app()
    app.state.repository = None  # simula BD no disponible

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/ideas/idea_x")
        assert r.status_code == 503
        assert "persistencia" in r.json()["detail"].lower()

        # el panel y health siguen funcionando sin BD
        assert (await client.get("/")).status_code == 200
        assert (await client.get("/health")).json()["status"] == "ok"
