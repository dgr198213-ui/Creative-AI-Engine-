"""Endpoint de domain packs cargados (Fase 6, bloque 4).

El panel construye sus botones de dominio y sus retos de ejemplo desde
aquí, en vez de tenerlos fijos en el HTML — así un domain pack nuevo
(`configs/domains/<nombre>/`) aparece en el panel sin tocar `src/`.
"""

from __future__ import annotations

from fastapi import APIRouter

from ...core.config import get_settings

router = APIRouter()


@router.get("/domains")
async def list_domains() -> list[dict]:
    """Domain packs cargados: nombre, título, descripción y ejemplos.

    `Settings.load()` garantiza al menos un pack (arranca con un
    RuntimeError si el registro no encuentra ninguno — ver
    `core/config.py`), así que no hace falta un fallback aquí.
    """
    settings = get_settings()
    packs = settings.list_packs()
    return [pack.to_summary_dict() for _, pack in sorted(packs.items())]
