"""Endpoint de estado del guard de presupuesto (Fase 5, bloque 3)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...core.config import get_settings
from ...llm.budget import get_budget_status

router = APIRouter()


@router.get("/budget")
async def get_budget(request: Request) -> dict:
    """Gasto estimado acumulado del periodo actual, límite y estado.

    `status`: "ok" | "warning" (≥80% del límite) | "downgraded" (límite
    superado — con `CREATIVE_BUDGET_ENFORCE=true`, los proveedores de
    pago quedan excluidos de todas las cadenas hasta el siguiente periodo).
    """
    settings = get_settings()
    status = await get_budget_status(settings, request.app.state.repository)
    return status.to_dict()
