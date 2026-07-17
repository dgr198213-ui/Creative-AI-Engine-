"""Endpoint de diagnóstico de configuración."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ...core.config import get_settings
from ...diagnostics import run_doctor

router = APIRouter()


@router.get("/diagnostics")
async def diagnostics(check_llm: bool = Query(default=False)) -> dict:
    """Estado de proveedores, enrutado y base de datos.

    `check_llm=true` hace una llamada mínima a cada proveedor para verificar
    sus claves (consume una petición diminuta por proveedor).
    """
    return await run_doctor(get_settings(), check_llm=check_llm)
