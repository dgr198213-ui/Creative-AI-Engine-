"""Endpoints de memoria y recomendación."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from ...core.exceptions import IdeaNotFoundError
from ...memory.recommendation import RecommendationEngine
from ...memory.repository import IdeaRepository

router = APIRouter()


@router.get("/memory/recommendations/{idea_id}")
async def recommend(
    idea_id: str,
    request: Request,
    limit: int = Query(default=10, ge=1, le=50),
    diversity_weight: float = Query(default=0.3, ge=0.0, le=1.0),
) -> dict:
    """Recomienda ideas relacionadas (similitud semántica + diversidad)."""
    repo: IdeaRepository = request.app.state.repository

    try:
        idea = await repo.get_idea(idea_id)
    except IdeaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    engine = RecommendationEngine(repository=repo)
    recommendations = await engine.recommend_for_idea(
        idea, limit=limit, diversity_weight=diversity_weight
    )

    return {
        "idea_id": idea_id,
        "count": len(recommendations),
        "recommendations": [
            {
                "id": rec.id,
                "title": rec.title,
                "fitness": rec.fitness,
                "score": round(score, 4),
                "description": rec.description[:200],
            }
            for rec, score in recommendations
        ],
    }
