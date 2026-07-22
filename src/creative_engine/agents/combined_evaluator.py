"""Evaluador combinado: utilidad + viabilidad + mercado en UNA llamada.

En free tiers, cada llamada LLM cuenta. Los tres agentes evaluadores
separados (innovation/feasibility/market) hacían 3 llamadas por idea.
Este agente las funde en una sola petición que devuelve las tres
puntuaciones a la vez, reduciendo el coste de evaluación a un tercio.

Se puede seguir usando el orquestador con los 3 agentes separados si se
prefiere granularidad; este es el modo económico por defecto.
"""

from __future__ import annotations

import contextlib
from typing import Any

import structlog

from ..core.models import Idea
from ..llm.provider import LLMProvider
from .base import AgentResult, BaseAgent

logger = structlog.get_logger(__name__)

COMBINED_SYSTEM = """Eres un comité de tres expertos que evalúa ideas creativas:
- Un estratega de innovación (juzga la UTILIDAD: ¿resuelve un dolor real?)
- Un ingeniero senior (juzga la VIABILIDAD técnica con tecnología actual)
- Un analista de mercado (juzga el ENCAJE con el mercado objetivo)

Puntúa con honestidad y criterio. No infles las notas."""

COMBINED_PROMPT = """Evalúa esta idea en tres dimensiones (cada una de 0 a 1):

Título: {title}
Descripción: {description}
Ventajas: {advantages}
Tecnologías: {technologies}
Mercados objetivo: {markets}

Contexto del reto: {challenge}
{profile_block}
Responde SOLO en JSON:
{{
    "utility": 0.0,
    "utility_feedback": "por qué es o no útil",
    "feasibility": 0.0,
    "feasibility_feedback": "análisis de viabilidad técnica",
    "market_fit": 0.0,
    "market_feedback": "encaje con el mercado",
    "estimated_complexity": 0.5
}}"""


class CombinedEvaluatorAgent(BaseAgent):
    """Evalúa utilidad, viabilidad y mercado en una sola llamada LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(name="combined", llm=llm, timeout_seconds=120.0)

    async def evaluate(
        self, idea: Idea, context: dict[str, Any] | None = None
    ) -> AgentResult:
        context = context or {}
        profile_hint = context.get("profile_hint") or ""
        prompt = COMBINED_PROMPT.format(
            title=idea.title,
            description=idea.description,
            advantages="\n".join(f"- {a}" for a in idea.advantages) or "—",
            technologies=", ".join(idea.features.technologies) or "No especificadas",
            markets=", ".join(idea.features.target_markets) or "No especificados",
            challenge=context.get("challenge", ""),
            profile_block=f"Perfil del reto (Analista Funcional): {profile_hint}\n" if profile_hint else "",
        )

        try:
            data = await self._llm.generate_structured(
                prompt=prompt, system_prompt=COMBINED_SYSTEM
            )

            def _score(key: str) -> float:
                return max(0.0, min(1.0, float(data.get(key, 0.5))))

            est = data.get("estimated_complexity")
            if est is not None:
                with contextlib.suppress(TypeError, ValueError):
                    idea.features.complexity_level = max(0.0, min(1.0, float(est)))

            return AgentResult(
                agent_name=self._name,
                idea_id=idea.id,
                success=True,
                score=_score("utility"),  # score principal; el resto en metadata
                feedback=str(data.get("utility_feedback", "")),
                metadata={
                    "utility": _score("utility"),
                    "feasibility": _score("feasibility"),
                    "market_fit": _score("market_fit"),
                    "feasibility_feedback": data.get("feasibility_feedback", ""),
                    "market_feedback": data.get("market_feedback", ""),
                },
            )
        except Exception as e:
            return AgentResult(
                agent_name=self._name, idea_id=idea.id, success=False, error=str(e)
            )
