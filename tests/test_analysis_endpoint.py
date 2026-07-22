"""Tests del endpoint POST /api/v1/analyze (Analista Funcional).

Con CREATIVE_ANALYST_ENABLED=false (default) responde 404: el flujo
actual de evolución no debe verse afectado por esta feature apagada.
"""

from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient

from creative_engine.core import config
from creative_engine.core.config import LLMProviderConfig, Settings


def _settings(analyst_enabled: bool, with_llm: bool = True) -> Settings:
    s = Settings.load()
    s.llm = (
        {"default": LLMProviderConfig(name="sim", api_key=config.SecretStr("x"))}
        if with_llm
        else {}
    )
    s.analyst_enabled = analyst_enabled
    return s


async def test_analyze_returns_404_when_flag_disabled() -> None:
    config._settings = _settings(analyst_enabled=False)

    from creative_engine.api.app import create_app

    app = create_app()
    payload = {"challenge": "mi tienda online no vende nada desde hace semanas"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/analyze", json=payload)

    assert r.status_code == 404


async def test_analyze_returns_503_without_llm_when_flag_enabled() -> None:
    config._settings = _settings(analyst_enabled=True, with_llm=False)

    from creative_engine.api.app import create_app

    app = create_app()
    payload = {"challenge": "mi tienda online no vende nada desde hace semanas"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/analyze", json=payload)

    assert r.status_code == 503


async def test_analyze_returns_profile_and_mirror_when_enabled() -> None:
    config._settings = _settings(analyst_enabled=True)

    from creative_engine.api.app import create_app

    app = create_app()

    sim = AsyncMock()
    sim.generate_structured.return_value = {
        "topografia": {"que_ocurre": "las ventas cayeron", "frecuencia": "recurrente"},
        "hipotesis_funcional": {"mecanismo": "el checkout falla", "confianza": 0.8},
        "friccion": {"impacto_principal": "dinero", "descripcion_impacto": "ingresos"},
        "reto_reformulado": "Reducir la fricción del checkout",
        "preguntas_pendientes": [],
    }
    sim.close = AsyncMock()

    payload = {"challenge": "mi tienda online no vende nada desde hace semanas"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("creative_engine.llm.factory.LLMProvider", return_value=sim):
            r = await client.post("/api/v1/analyze", json=payload)

    assert r.status_code == 200
    data = r.json()
    assert data["profile"]["reto_reformulado"] == "Reducir la fricción del checkout"
    assert data["profile"]["reto_original"] == payload["challenge"]
    assert "Esto es lo que he entendido" in data["espejo_render"]


async def test_analyze_correction_cycle_via_http() -> None:
    """Un segundo POST con correction + previous_profile produce v2."""
    config._settings = _settings(analyst_enabled=True)

    from creative_engine.api.app import create_app

    app = create_app()

    sim = AsyncMock()
    sim.generate_structured.return_value = {
        "hipotesis_funcional": {"mecanismo": "confirmado", "confianza": 0.9},
        "reto_reformulado": "Reformulación corregida",
    }
    sim.close = AsyncMock()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("creative_engine.llm.factory.LLMProvider", return_value=sim):
            first = await client.post(
                "/api/v1/analyze",
                json={"challenge": "mis empleados no rinden y no sé por qué"},
            )
            assert first.status_code == 200
            profile_v1 = first.json()["profile"]

            second = await client.post(
                "/api/v1/analyze",
                json={
                    "challenge": "mis empleados no rinden y no sé por qué",
                    "correction": "el problema es la falta de formación, no la motivación",
                    "previous_profile": profile_v1,
                },
            )

    assert second.status_code == 200
    profile_v2 = second.json()["profile"]
    assert profile_v2["version"] == profile_v1["version"] + 1
    assert profile_v2["reto_reformulado"] == "Reformulación corregida"
