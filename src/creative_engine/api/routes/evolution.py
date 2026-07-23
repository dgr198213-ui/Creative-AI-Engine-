"""Endpoints de evolución."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from ...core.models import DomainName, EvolutionRequest, EvolutionResponse
from ..deps import require_repo
from ..guardrails import enforce_evolution_rate_limit, enforce_request_budget

if TYPE_CHECKING:
    from ...evolution.qd_engine import QDEngine

logger = structlog.get_logger(__name__)
router = APIRouter()


async def _build_qd_engine(request: Request, on_generation=None) -> QDEngine:
    """Construye el motor QD con enrutamiento de LLM por rol y failover."""
    from ...agents.combined_evaluator import CombinedEvaluatorAgent
    from ...agents.evaluator_orchestrator import EvaluatorOrchestrator
    from ...agents.generator import IdeaGeneratorAgent
    from ...core.config import get_settings
    from ...evolution.crossover import CrossoverEngine
    from ...evolution.encoders import get_shared_encoder
    from ...evolution.mutation import MutationEngine
    from ...evolution.qd_engine import QDEngine
    from ...llm.budget import get_budget_status
    from ...llm.factory import build_router, role_llms

    settings = get_settings()

    if not settings.llm:
        raise HTTPException(
            status_code=503,
            detail="No hay proveedores LLM configurados (CREATIVE_LLM__*)",
        )

    # Guard de presupuesto (Fase 5, bloque 3): si el gasto estimado del
    # periodo ya superó CREATIVE_BUDGET_LIMIT, construir el router SIN los
    # proveedores de pago — el run se completa igual con los gratuitos.
    budget_status = await get_budget_status(settings, request.app.state.repository)
    router = build_router(settings, budget_excluded=set(budget_status.excluded_providers))
    roles = role_llms(router)
    gen_llm = roles["generator"]
    eval_llm = roles["evaluator"]

    # Concurrencia base tomada del primer proveedor (los frenos reales
    # viven en cada LLMProvider vía semáforo y min_interval).
    max_concurrent = next(iter(settings.llm.values())).max_concurrent

    # Evaluador combinado: 3 dimensiones en 1 llamada (ahorro en free tier).
    evaluator = EvaluatorOrchestrator(
        agents={"combined": CombinedEvaluatorAgent(eval_llm)}
    )

    engine = QDEngine(
        generator=IdeaGeneratorAgent(gen_llm),
        evaluator=evaluator,
        mutation=MutationEngine(gen_llm, max_concurrent=max_concurrent),
        crossover=CrossoverEngine(gen_llm, max_concurrent=max_concurrent),
        encoder=get_shared_encoder(),
        repository=request.app.state.repository,
        on_generation=on_generation,
    )
    # Guardamos el router para cerrarlo (y contabilizar su gasto) tras el run.
    engine._llm_router = router  # type: ignore[attr-defined]
    return engine


async def _close_and_record_spend(engine: QDEngine, request: Request) -> None:
    """Cierra el router del run y persiste el gasto estimado (bloque 3).

    Compartido entre /evolution/start y /evolution/stream: ambos
    construyen el motor con `_build_qd_engine` y deben registrar el
    consumo real de proveedores de pago antes de cerrarlos.
    """
    from ...core.config import get_settings
    from ...llm.budget import record_run_spend

    llm_router = getattr(engine, "_llm_router", None)
    if llm_router is None:
        return
    try:
        await record_run_spend(llm_router, get_settings(), request.app.state.repository)
    finally:
        await llm_router.close_all()


@router.post("/evolution/start", dependencies=[Depends(enforce_evolution_rate_limit)])
async def start_evolution(request_body: EvolutionRequest, request: Request) -> dict:
    """Ejecuta una evolución completa y devuelve el resumen.

    Nota MVP: la ejecución es síncrona dentro de la petición. Para retos
    grandes usar el CLI o reducir población/generaciones.
    """
    enforce_request_budget(request_body)
    engine = await _build_qd_engine(request)
    try:
        state = await engine.run_evolution(request_body)
    finally:
        await _close_and_record_spend(engine, request)

    top_ideas = sorted(state.archive, key=lambda c: c.fitness, reverse=True)[:10]

    return {
        "run_id": state.run_id,
        "status": state.status,
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
    repo = require_repo(request)
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
