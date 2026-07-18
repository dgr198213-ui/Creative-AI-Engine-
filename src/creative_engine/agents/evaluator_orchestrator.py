"""Orquestador de evaluación multiagente.

Coordina agentes especializados para producir una evaluación de CALIDAD.
La NOVEDAD no se decide aquí: la calcula el motor QD de forma objetiva
(distancia de embedding al archivo) tras la codificación.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import structlog

from ..core.config import get_settings
from ..core.events import Event, EventType, get_event_bus
from ..core.models import EvaluationScores, Idea, IdeaStatus
from .base import AgentResult, BaseAgent

logger = structlog.get_logger(__name__)


class EvaluatorOrchestrator:
    """Coordina la evaluación de una idea mediante múltiples agentes."""

    # Mapeo agente → dimensión de evaluación (solo dimensiones de calidad)
    AGENT_DIMENSION_MAP: ClassVar[dict[str, str]] = {
        "innovation": "utility",
        "feasibility": "feasibility",
        "market": "market_fit",
    }

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self._agents = agents
        self._bus = get_event_bus()
        self._log = logger.bind(agents=list(agents.keys()))

    async def evaluate_idea(
        self,
        idea: Idea,
        context: dict[str, Any] | None = None,
        parallel: bool = True,
    ) -> EvaluationScores:
        """Evalúa una idea con todos los agentes registrados."""
        idea.status = IdeaStatus.EVALUATING
        context = context or {}

        if parallel:
            results = await self._evaluate_parallel(idea, context)
        else:
            results = await self._evaluate_sequential(idea, context)

        scores = self._aggregate_results(idea, results, context)

        idea.evaluation = scores
        idea.status = IdeaStatus.EVALUATED

        await self._bus.publish(
            Event(
                type=EventType.IDEA_EVALUATED,
                data={
                    "idea_id": idea.id,
                    "fitness": scores.weighted_score,
                    "agents_completed": len([r for r in results if r.success]),
                },
                source="EvaluatorOrchestrator",
            )
        )

        return scores

    async def _evaluate_parallel(
        self, idea: Idea, context: dict[str, Any]
    ) -> list[AgentResult]:
        tasks = [agent.safe_evaluate(idea, context) for agent in self._agents.values()]
        return list(await asyncio.gather(*tasks))

    async def _evaluate_sequential(
        self, idea: Idea, context: dict[str, Any]
    ) -> list[AgentResult]:
        results = []
        for agent in self._agents.values():
            results.append(await agent.safe_evaluate(idea, context))
        return results

    def _aggregate_results(
        self,
        idea: Idea,
        results: list[AgentResult],
        context: dict[str, Any],
    ) -> EvaluationScores:
        """Agrega los resultados de los agentes en un EvaluationScores."""
        domain_name = context.get("domain", idea.domain)
        settings = get_settings()
        domain_cfg = settings.get_domain(domain_name)
        weights = context.get("custom_weights") or domain_cfg.evaluation_weights

        scores_data: dict[str, float] = {
            "novelty": 0.0,  # la asigna el motor QD (objetiva)
            "utility": 0.0,
            "feasibility": 0.0,
            "complexity": 0.5,
            "impact": 0.0,
            "market_fit": 0.0,
            "sustainability": 0.5,
            "scalability": 0.5,
        }

        agent_feedback: dict[str, str] = {}

        for result in results:
            if not result.success:
                agent_feedback[result.agent_name] = f"Error: {result.error}"
                continue

            # Agente combinado: trae las 3 dimensiones en metadata de una vez.
            if result.agent_name == "combined":
                for dim in ("utility", "feasibility", "market_fit"):
                    if dim in result.metadata:
                        scores_data[dim] = max(0.0, min(1.0, float(result.metadata[dim])))
                agent_feedback["utility"] = result.feedback
                if result.metadata.get("feasibility_feedback"):
                    agent_feedback["feasibility"] = result.metadata["feasibility_feedback"]
                if result.metadata.get("market_feedback"):
                    agent_feedback["market"] = result.metadata["market_feedback"]
                continue

            dimension = self.AGENT_DIMENSION_MAP.get(result.agent_name)
            if dimension and result.score is not None:
                scores_data[dimension] = max(0.0, min(1.0, result.score))

            if result.feedback:
                agent_feedback[result.agent_name] = result.feedback

        # Impacto derivado de calidad (sin agente dedicado en el MVP)
        scores_data["impact"] = (
            scores_data["utility"] * 0.5
            + scores_data["market_fit"] * 0.3
            + scores_data["feasibility"] * 0.2
        )
        scores_data["complexity"] = idea.features.complexity_level

        return EvaluationScores(
            **scores_data,
            weights=weights,
            agent_feedback=agent_feedback,
        )
