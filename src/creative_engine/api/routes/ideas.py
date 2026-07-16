"""Endpoints de consulta de ideas."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from ...core.exceptions import IdeaNotFoundError
from ...core.models import IdeaDetailResponse
from ...memory.repository import IdeaRepository

router = APIRouter()


@router.get("/ideas/{idea_id}", response_model=IdeaDetailResponse)
async def get_idea(idea_id: str, request: Request) -> IdeaDetailResponse:
    """Detalle completo de una idea con relacionadas y linaje."""
    repo: IdeaRepository = request.app.state.repository

    try:
        idea = await repo.get_idea(idea_id)
    except IdeaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    related = await repo.search_similar(idea, limit=5)
    lineage = await repo.get_lineage(idea_id)

    return IdeaDetailResponse(idea=idea, related_ideas=related, evolution_lineage=lineage)


@router.get("/runs/{run_id}/elites")
async def get_run_elites(
    run_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """Ideas élite de una ejecución: el abanico de resultados diversos."""
    repo: IdeaRepository = request.app.state.repository
    elites = await repo.get_elites_by_run(run_id, limit=limit)

    return {
        "run_id": run_id,
        "count": len(elites),
        "ideas": [
            {
                "id": e.id,
                "title": e.title,
                "fitness": e.fitness,
                "novelty": e.evaluation.novelty if e.evaluation else None,
                "generation": e.generation,
                "description": e.description[:200],
                "evaluation": e.evaluation.model_dump() if e.evaluation else None,
            }
            for e in elites
        ],
    }


@router.get("/runs/{run_id}/families")
async def get_run_families(
    run_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=300),
    distance_threshold: float = Query(default=0.25, ge=0.05, le=1.0),
    with_reports: bool = Query(default=False),
) -> dict:
    """Agrupa las élites de un run en familias automáticas de enfoques.

    Convierte la lista plana de ideas en grupos por proximidad semántica:
    "N enfoques distintos, la mejor idea de cada uno". El número de
    familias es automático (emerge del umbral de distancia).

    `with_reports=true` genera un informe ejecutivo por familia con el
    WriterAgent — cuesta una llamada LLM por familia, por eso es opcional.
    """
    from ...evolution.clustering import group_into_families

    repo: IdeaRepository = request.app.state.repository
    elites = await repo.get_elites_by_run(run_id, limit=limit)

    if not elites:
        raise HTTPException(status_code=404, detail=f"Sin élites para el run {run_id}")

    families = group_into_families(elites, distance_threshold=distance_threshold)

    if with_reports:
        await _attach_reports(families)

    return {
        "run_id": run_id,
        "family_count": len(families),
        "total_elites": len(elites),
        "families": [
            {
                "family_id": fam.family_id,
                "size": fam.size,
                "avg_fitness": round(fam.avg_fitness, 4),
                "representative": {
                    "id": fam.representative.id,
                    "title": fam.representative.title,
                    "fitness": fam.representative.fitness,
                    "novelty": (
                        fam.representative.evaluation.novelty
                        if fam.representative.evaluation
                        else None
                    ),
                    "description": fam.representative.description[:300],
                },
                "report": fam.report,
                "members": [
                    {
                        "id": m.id,
                        "title": m.title,
                        "fitness": m.fitness,
                        "generation": m.generation,
                    }
                    for m in fam.members
                ],
            }
            for fam in families
        ],
    }


async def _attach_reports(families: list) -> None:
    """Genera un informe por familia con el WriterAgent (una llamada LLM cada uno)."""
    from ...agents.writer import WriterAgent
    from ...core.config import get_settings
    from ...llm.provider import LLMProvider

    settings = get_settings()
    if not settings.llm:
        return  # sin LLM configurado: se devuelven las familias sin informe

    config = next(iter(settings.llm.values()))
    llm = LLMProvider(config)
    try:
        writer = WriterAgent(llm)
        for fam in families:
            fam.report = await writer.write_report(fam.representative)
    finally:
        await llm.close()


@router.post("/ideas/{idea_id}/report")
async def generate_report(idea_id: str, request: Request) -> dict:
    """Genera un informe ejecutivo para una idea (una llamada LLM, bajo demanda)."""
    from ...agents.writer import WriterAgent
    from ...core.config import get_settings
    from ...llm.provider import LLMProvider

    repo: IdeaRepository = request.app.state.repository
    try:
        idea = await repo.get_idea(idea_id)
    except IdeaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    settings = get_settings()
    if not settings.llm:
        raise HTTPException(status_code=503, detail="No hay proveedor LLM configurado")

    llm = LLMProvider(next(iter(settings.llm.values())))
    try:
        report = await WriterAgent(llm).write_report(idea)
    finally:
        await llm.close()

    return {"idea_id": idea_id, "title": idea.title, "report": report}


@router.get("/stats")
async def get_stats(request: Request, run_id: str | None = None) -> dict:
    """Estadísticas globales o por ejecución."""
    repo: IdeaRepository = request.app.state.repository
    return await repo.get_stats(run_id=run_id)
