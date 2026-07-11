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


@router.get("/stats")
async def get_stats(request: Request, run_id: str | None = None) -> dict:
    """Estadísticas globales o por ejecución."""
    repo: IdeaRepository = request.app.state.repository
    return await repo.get_stats(run_id=run_id)
