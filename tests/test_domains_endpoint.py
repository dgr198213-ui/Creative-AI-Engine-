"""GET /api/v1/domains (Fase 6, bloque 4): el panel construye sus
botones de dominio desde aquí en vez de tenerlos fijos en el HTML."""

from __future__ import annotations

from pathlib import Path

from httpx import ASGITransport, AsyncClient

from creative_engine.core import config
from creative_engine.core.config import LLMProviderConfig, Settings


def _write_pack(domains_dir: Path, name: str, display_name: str, examples: list[str]) -> None:
    pack_dir = domains_dir / name
    pack_dir.mkdir(parents=True)
    (pack_dir / "domain.yaml").write_text(
        f"name: {name}\n"
        f'display_name: "{display_name}"\n'
        f'description: "Descripción de {display_name}"\n'
        "behavior_dimensions:\n"
        "  - name: a\n    bins: 5\n"
        "  - name: b\n    bins: 5\n",
        encoding="utf-8",
    )
    if examples:
        items = "\n".join(f"- {e}" for e in examples)
        (pack_dir / "examples.yaml").write_text(items + "\n", encoding="utf-8")


def _configure_settings(configs_dir: Path) -> None:
    s = Settings.load()
    s.llm = {"default": LLMProviderConfig(name="sim", api_key=config.SecretStr("x"))}
    config._settings = s


async def test_lists_loaded_packs_with_examples(tmp_path, monkeypatch) -> None:
    domains_dir = tmp_path / "domains"
    _write_pack(domains_dir, "generic", "General", ["Reto genérico de ejemplo"])
    _write_pack(domains_dir, "marketing", "Marketing", ["Campaña de ejemplo"])

    import creative_engine.core.config as config_module

    monkeypatch.setattr(config_module, "_CONFIGS_DIR", tmp_path)
    _configure_settings(tmp_path)

    from creative_engine.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v1/domains")

    assert r.status_code == 200
    body = r.json()
    names = {d["name"] for d in body}
    assert names == {"generic", "marketing"}
    generic = next(d for d in body if d["name"] == "generic")
    assert generic["display_name"] == "General"
    assert generic["examples"] == ["Reto genérico de ejemplo"]


async def test_falls_back_to_embedded_generic_without_packs(tmp_path, monkeypatch) -> None:
    """Sin configs/domains/ en absoluto: el endpoint no debe romperse,
    debe reportar el genérico embebido."""
    import creative_engine.core.config as config_module

    monkeypatch.setattr(config_module, "_CONFIGS_DIR", tmp_path)  # sin domains/ dentro
    _configure_settings(tmp_path)

    from creative_engine.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v1/domains")

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["name"] == "generic"
