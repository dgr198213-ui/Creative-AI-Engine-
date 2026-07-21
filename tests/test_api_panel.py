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

    async def get_run_status(self, run_id: str) -> dict | None:
        return None

    async def get_stats(self, run_id: str | None = None) -> dict:
        return {"total": len(self._store), "elites": 0, "discarded": 0}

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


async def test_budget_cap_rejects_oversized_request() -> None:
    """Auditoría C2: population_size x generations no puede desbordar el
    tope configurado, ni siquiera con un EvolutionRequest "legítimo"."""
    from httpx import ASGITransport, AsyncClient

    s = Settings.load()
    s.llm = {"default": LLMProviderConfig(name="sim", api_key=config.SecretStr("x"))}
    config._settings = s

    from creative_engine.api.app import create_app

    app = create_app()

    payload = {
        "challenge": "Reto de prueba con longitud suficiente para pasar la validación",
        "population_size": 500,
        "generations": 200,
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/evolution/start", json=payload)

    assert r.status_code == 422
    assert "presupuesto" in r.json()["detail"].lower()


async def test_budget_cap_allows_request_within_limit() -> None:
    """Un request dentro del tope no debe chocar con el guardrail: debe
    llegar hasta el intento de construir el motor (503 sin LLM real
    configurado), no quedarse en el 422 del presupuesto."""
    from httpx import ASGITransport, AsyncClient

    s = Settings.load()
    s.llm = {}  # sin proveedores: si pasa el guardrail, falla con 503, no red
    config._settings = s

    from creative_engine.api.app import create_app

    app = create_app()

    payload = {
        "challenge": "Reto de prueba con longitud suficiente para pasar la validación",
        "population_size": 8,
        "generations": 3,
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/evolution/start", json=payload)

    assert r.status_code == 503


async def test_rate_limit_blocks_after_threshold() -> None:
    """Auditoría C2: /evolution/* debe limitar peticiones por IP.

    Sin LLM real configurado los requests que pasen el rate limit fallan
    rápido con 503 (motor no construible); lo que importa aquí es que,
    tras agotar el límite, la petición ni siquiera llega a intentarlo.
    """
    from httpx import ASGITransport, AsyncClient

    s = Settings.load()
    s.llm = {}  # sin proveedores: los que pasen el rate limit devuelven 503
    s.evolution.rate_limit_per_minute = 2
    config._settings = s

    from creative_engine.api.app import create_app

    app = create_app()

    payload = {"challenge": "Reto de prueba con longitud suficiente para validar"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.post("/api/v1/evolution/start", json=payload)
        r2 = await client.post("/api/v1/evolution/start", json=payload)
        r3 = await client.post("/api/v1/evolution/start", json=payload)

    assert r1.status_code == 503
    assert r2.status_code == 503
    assert r3.status_code == 429
    assert "límite" in r3.json()["detail"].lower()


async def test_rate_limit_is_per_app_not_global() -> None:
    """Cada app (cada test) tiene su propio limiter: no hay estado global
    compartido que contamine tests entre sí."""
    from httpx import ASGITransport, AsyncClient

    s = Settings.load()
    s.llm = {}
    s.evolution.rate_limit_per_minute = 1
    config._settings = s

    from creative_engine.api.app import create_app

    app_a = create_app()
    app_b = create_app()
    payload = {"challenge": "Reto de prueba con longitud suficiente para validar"}

    async with AsyncClient(transport=ASGITransport(app=app_a), base_url="http://test") as client:
        ra = await client.post("/api/v1/evolution/start", json=payload)
    async with AsyncClient(transport=ASGITransport(app=app_b), base_url="http://test") as client:
        rb = await client.post("/api/v1/evolution/start", json=payload)

    assert ra.status_code == 503  # consumió su único hueco, sin 429
    assert rb.status_code == 503  # app distinta: limiter fresco, tampoco 429


class TestApiKeyAuth:
    """Auditoría C1: sin CREATIVE_API_KEY la API queda abierta (como hoy);
    al configurarla, /api/v1/* exige X-API-Key salvo rutas públicas."""

    async def test_open_by_default_without_api_key(self) -> None:
        s = Settings.load()
        s.llm = {}
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/stats")
            assert r.status_code != 401

    async def test_blocks_without_key_when_configured(self) -> None:
        s = Settings.load()
        s.llm = {}
        s.api_key = "secreta-123"
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()
        app.state.repository = _NullRepo()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/stats")
            assert r.status_code == 401

    async def test_allows_with_correct_header(self) -> None:
        s = Settings.load()
        s.llm = {}
        s.api_key = "secreta-123"
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()
        app.state.repository = _NullRepo()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get(
                "/api/v1/stats", headers={"X-API-Key": "secreta-123"}
            )
            assert r.status_code != 401

    async def test_wrong_key_rejected(self) -> None:
        s = Settings.load()
        s.llm = {}
        s.api_key = "secreta-123"
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()
        app.state.repository = _NullRepo()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/stats", headers={"X-API-Key": "otra-cosa"})
            assert r.status_code == 401

    async def test_allows_with_query_param_for_download_links(self) -> None:
        """El enlace <a href> de descarga del export no puede llevar cabeceras;
        debe poder autenticarse con ?api_key= en su lugar."""
        s = Settings.load()
        s.llm = {}
        s.api_key = "secreta-123"
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()
        app.state.repository = _NullRepo()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/api/v1/stats?api_key=secreta-123")
            assert r.status_code != 401

    async def test_health_and_panel_stay_public(self) -> None:
        s = Settings.load()
        s.llm = {}
        s.api_key = "secreta-123"
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()
        app.state.repository = _NullRepo()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/")).status_code == 200
            assert (await client.get("/static/app.js")).status_code == 200


class TestDocsHiddenInProduction:
    """Auditoría A2: /docs, /redoc y /openapi.json solo con debug=True."""

    async def test_docs_hidden_when_not_debug(self) -> None:
        s = Settings.load()
        s.debug = False
        s.llm = {}
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/docs")).status_code == 404
            assert (await client.get("/redoc")).status_code == 404
            assert (await client.get("/openapi.json")).status_code == 404

    async def test_docs_available_when_debug(self) -> None:
        s = Settings.load()
        s.debug = True
        s.llm = {}
        config._settings = s

        from creative_engine.api.app import create_app

        app = create_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/docs")).status_code == 200
            assert (await client.get("/openapi.json")).status_code == 200


async def test_export_run_markdown() -> None:
    """El export del run devuelve un informe Markdown con las familias."""
    from httpx import ASGITransport, AsyncClient

    from creative_engine.core.models import EvaluationScores, Idea, IdeaStatus

    s = Settings.load()
    s.llm = {"default": LLMProviderConfig(name="sim", api_key=config.SecretStr("x"))}
    config._settings = s

    from creative_engine.api.app import create_app

    app = create_app()
    repo = _NullRepo()
    app.state.repository = repo

    # simular élites de un run
    elites = []
    for i, desc in enumerate(
        [[0.1, 0.1, 0.1], [0.12, 0.1, 0.1], [0.9, 0.9, 0.9]]
    ):
        idea = Idea(
            title=f"Idea élite {i}",
            description=f"Descripción de la idea élite número {i} del run.",
            advantages=["Ventaja"],
            status=IdeaStatus.ELITE,
            run_id="run_test",
        )
        idea.evaluation = EvaluationScores(utility=0.7, feasibility=0.6, market_fit=0.5)
        idea.behavior_descriptor = desc
        elites.append(idea)

    async def fake_get_elites(run_id, limit=50):
        return elites if run_id == "run_test" else []

    repo.get_elites_by_run = fake_get_elites

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/runs/run_test/export")
        assert r.status_code == 200
        assert "text/markdown" in r.headers["content-type"]
        assert "attachment" in r.headers["content-disposition"]
        body = r.text
        assert "# Exploración creativa" in body
        assert "Idea élite" in body
        assert "Solidez" in body

        # run inexistente → 404
        r = await client.get("/api/v1/runs/run_nope/export")
        assert r.status_code == 404


async def test_export_run_failed_returns_status_not_404() -> None:
    """M4 de la auditoría: un run que falló (0 élites, cadena de proveedores
    agotada) debe exportar el estado failed, no un 404 indistinguible de
    'este run no existe'."""
    from httpx import ASGITransport, AsyncClient

    s = Settings.load()
    s.llm = {"default": LLMProviderConfig(name="sim", api_key=config.SecretStr("x"))}
    config._settings = s

    from creative_engine.api.app import create_app

    app = create_app()
    repo = _NullRepo()
    app.state.repository = repo

    async def fake_get_elites(run_id, limit=50):
        return []  # el run falló: nunca se generó ni una idea

    async def fake_get_run_status(run_id):
        if run_id == "run_failed":
            return {
                "run_id": "run_failed",
                "status": "failed",
                "error": "La cadena de proveedores LLM se agotó por completo.",
            }
        return None

    repo.get_elites_by_run = fake_get_elites
    repo.get_run_status = fake_get_run_status

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/runs/run_failed/export")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "failed"
        assert "agotó" in data["error"]

        # sin estado persistido y sin élites → sigue siendo 404 (run inexistente)
        r_missing = await client.get("/api/v1/runs/run_missing/export")
        assert r_missing.status_code == 404


async def test_recovery_families_include_id_and_advantages() -> None:
    """El endpoint de recuperación debe traer id y advantages del representante
    (si faltan, el panel manda 'undefined' al generar informe → 404)."""
    from httpx import ASGITransport, AsyncClient

    from creative_engine.core.models import EvaluationScores, Idea, IdeaStatus

    s = Settings.load()
    s.llm = {"default": LLMProviderConfig(name="sim", api_key=config.SecretStr("x"))}
    config._settings = s

    from creative_engine.api.app import create_app

    app = create_app()
    repo = _NullRepo()
    app.state.repository = app.state.repository = repo

    elites = []
    for i, desc in enumerate([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]):
        idea = Idea(
            title=f"Idea {i}",
            description=f"Descripción larga de la idea élite {i} del run.",
            advantages=["Ventaja clara", "Otra ventaja"],
            status=IdeaStatus.ELITE,
            run_id="run_rec",
        )
        idea.evaluation = EvaluationScores(utility=0.7, feasibility=0.6, market_fit=0.5)
        idea.behavior_descriptor = desc
        elites.append(idea)

    async def fake_get_elites(run_id, limit=50):
        return elites if run_id == "run_rec" else []

    repo.get_elites_by_run = fake_get_elites

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/runs/run_rec/families")
        assert r.status_code == 200
        data = r.json()
        assert data["family_count"] >= 1
        for fam in data["families"]:
            rep = fam["representative"]
            assert rep["id"], "el representante debe tener id (si no, informe → undefined)"
            assert "advantages" in rep, "el representante debe incluir advantages"
