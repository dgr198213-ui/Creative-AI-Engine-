"""Endpoints de evolución."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, HTTPException, Request

from ...core.models import DomainName, EvolutionRequest, EvolutionResponse
from ...memory.repository import IdeaRepository

if TYPE_CHECKING:
    from ...evolution.qd_engine import QDEngine

logger = structlog.get_logger(__name__)
router = APIRouter()


async def _build_qd_engine(request: Request) -> QDEngine:
    """Construye el motor QD con todas sus dependencias."""
    from ...agents.evaluator_orchestrator import EvaluatorOrchestrator
    from ...agents.feasibility import FeasibilityAgent
    from ...agents.generator import IdeaGeneratorAgent
    from ...agents.innovation import InnovationAgent
    from ...agents.market import MarketAgent
    from ...core.config import get_settings
    from ...evolution.crossover import CrossoverEngine
    from ...evolution.encoders import IdeaEncoder
    from ...evolution.mutation import MutationEngine
    from ...evolution.qd_engine import QDEngine
    from ...llm.provider import LLMProvider

    settings = get_settings()

    if not settings.llm:
        raise HTTPException(
            status_code=503,
            detail="No hay proveedores LLM configurados (CREATIVE_LLM__*)",
        )

    first_config = next(iter(settings.llm.values()))
    llm = LLMProvider(first_config)

    evaluator = EvaluatorOrchestrator(
        agents={
            "innovation": InnovationAgent(llm),
            "feasibility": FeasibilityAgent(llm),
            "market": MarketAgent(llm),
        }
    )

    return QDEngine(
        generator=IdeaGeneratorAgent(llm),
        evaluator=evaluator,
        mutation=MutationEngine(llm, max_concurrent=first_config.max_concurrent),
        crossover=CrossoverEngine(llm, max_concurrent=first_config.max_concurrent),
        encoder=IdeaEncoder(),
        repository=request.app.state.repository,
    )


@router.post("/evolution/start")
async def start_evolution(request_body: EvolutionRequest, request: Request) -> dict:
    """Ejecuta una evolución completa y devuelve el resumen.

    Nota MVP: la ejecución es síncrona dentro de la petición. Para retos
    grandes usar el CLI o reducir población/generaciones.
    """
    engine = await _build_qd_engine(request)
    state = await engine.run_evolution(request_body)

    top_ideas = sorted(state.archive, key=lambda c: c.fitness, reverse=True)[:10]

    return {
        "run_id": state.run_id,
        "status": "completed",
        "generations": state.generation,
        "total_ideas_generated": len(state.all_ideas),
        "coverage": state.coverage,
        "qd_score": state.qd_score,
        "best_fitness": state.best_fitness,
        "elite_count": len(state.archive),
        "top_ideas": [
            {
                "id": c.elite.id,
                "title": c.elite.title,
                "fitness": c.fitness,
                "novelty": c.elite.evaluation.novelty if c.elite.evaluation else None,
                "description": c.elite.description[:200],
            }
            for c in top_ideas
        ],
    }


@router.get("/evolution/{run_id}", response_model=EvolutionResponse)
async def get_evolution_status(run_id: str, request: Request) -> EvolutionResponse:
    """Resumen de una evolución persistida."""
    repo: IdeaRepository = request.app.state.repository
    elites = await repo.get_elites_by_run(run_id, limit=50)
    stats = await repo.get_stats(run_id=run_id)

    if not elites and not stats.get("total"):
        raise HTTPException(status_code=404, detail=f"Evolución {run_id} no encontrada")

    max_generation = max((e.generation for e in elites), default=0)

    return EvolutionResponse(
        run_id=run_id,
        challenge="",
        domain=elites[0].domain if elites else DomainName.GENERIC,
        generations_completed=max_generation,
        total_ideas_generated=int(stats.get("total") or 0),
        elite_count=int(stats.get("elites") or 0),
        coverage=0.0,
        qd_score=0.0,
        best_fitness=float(stats.get("max_fitness") or 0.0),
        top_ideas=elites[:10],
        completed_at=datetime.now(UTC),
    )
