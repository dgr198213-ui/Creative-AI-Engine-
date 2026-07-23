"""Endpoint del Analista Funcional (diseño 22-jul-2026, flag CREATIVE_ANALYST_ENABLED).

Perfila un reto antes de generar ideas: recibe el texto tal cual del
usuario (y, opcionalmente, una corrección sobre un análisis previo) y
devuelve el perfil estructurado + el texto del espejo de confirmación.

Con el flag apagado (default) responde 404: el flujo actual no se toca.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...core.config import get_settings
from ...core.models import ChallengeProfile
from ..guardrails import enforce_evolution_rate_limit

router = APIRouter()


class AnalyzeRequest(BaseModel):
    challenge: str = Field(..., min_length=10, max_length=5000)
    # Domain pack (Fase 6): si declara profile_fields, el Analista
    # extiende el esquema JSON con un bloque `dominio` — ver D4.
    domain: str = "generic"
    # Ciclo único de corrección (§2 del diseño): si vienen ambos, el
    # Analista produce un perfil v2 a partir del v1 + la corrección.
    correction: str | None = Field(default=None, max_length=2000)
    previous_profile: ChallengeProfile | None = None


class AnalyzeResponse(BaseModel):
    profile: ChallengeProfile
    espejo_render: str


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(enforce_evolution_rate_limit)],
)
async def analyze_challenge(request_body: AnalyzeRequest) -> AnalyzeResponse:
    """Perfila un reto (o lo re-perfila con una corrección) en una llamada LLM."""
    settings = get_settings()

    if not settings.analyst_enabled:
        raise HTTPException(
            status_code=404,
            detail="Analista Funcional no habilitado (CREATIVE_ANALYST_ENABLED=false)",
        )
    if not settings.llm:
        raise HTTPException(
            status_code=503,
            detail="No hay proveedores LLM configurados (CREATIVE_LLM__*)",
        )

    from ...analysis.analyst import FunctionalAnalystAgent
    from ...analysis.mirror import render_mirror
    from ...llm.factory import build_router, role_llms

    domain_cfg = settings.get_domain(request_body.domain)

    llm_router = build_router(settings)
    try:
        analyst_llm = role_llms(llm_router)["analyst"]
        agent = FunctionalAnalystAgent(analyst_llm)
        profile = await agent.analyze(
            challenge=request_body.challenge,
            domain=domain_cfg,
            correction=request_body.correction,
            previous_profile=request_body.previous_profile,
        )
    finally:
        await llm_router.close_all()

    return AnalyzeResponse(profile=profile, espejo_render=render_mirror(profile))
