"""Agente de Viabilidad: analiza la factibilidad técnica."""

from __future__ import annotations

import contextlib
from typing import Any

import structlog

from ..core.models import Idea
from ..llm.provider import LLMProvider
from .base import AgentResult, BaseAgent

logger = structlog.get_logger(__name__)

FEASIBILITY_SYSTEM = """Eres un ingeniero senior experto en análisis de viabilidad técnica.
Evalúa si una idea puede construirse con la tecnología ACTUAL o en los próximos 2-3 años.
Sé realista pero no excesivamente conservador.

Criterios:
- ¿Existen los materiales/componentes necesarios?
- ¿El proceso de fabricación es estándar o requiere I+D?
- ¿Hay barreras regulatorias importantes?
- ¿La complejidad es manejable?"""

FEASIBILITY_PROMPT = """Evalúa la VIABILIDAD TÉCNICA de esta idea (0 a 1):

Título: {title}
Descripción: {description}
Tecnologías: {technologies}
Materiales: {materials}

Contexto del reto: {challenge}

Responde SOLO en JSON:
{{
    "score": 0.0,
    "feedback": "Análisis de viabilidad técnica.",
    "technical_risks": ["riesgo 1", "riesgo 2"],
    "required_advances": ["qué avances tecnológicos necesita, si los hay"],
    "estimated_complexity": 0.5
}}"""


class FeasibilityAgent(BaseAgent):
    """Agente que evalúa la viabilidad técnica."""

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(name="feasibility", llm=llm, timeout_seconds=20.0)

    async def evaluate(
        self, idea: Idea, context: dict[str, Any] | None = None
    ) -> AgentResult:
        context = context or {}
        prompt = FEASIBILITY_PROMPT.format(
            title=idea.title,
            description=idea.description,
            technologies=", ".join(idea.features.technologies) or "No especificadas",
            materials=", ".join(idea.features.materials) or "No especificados",
            challenge=context.get("challenge", ""),
        )

        try:
            data = await self._llm.generate_structured(
                prompt=prompt, system_prompt=FEASIBILITY_SYSTEM
            )
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))

            est_complexity = data.get("estimated_complexity")
            if est_complexity is not None:
                with contextlib.suppress(TypeError, ValueError):
                    idea.features.complexity_level = max(0.0, min(1.0, float(est_complexity)))

            return AgentResult(
                agent_name=self._name,
                idea_id=idea.id,
                success=True,
                score=score,
                feedback=str(data.get("feedback", "")),
                metadata=data,
            )
        except Exception as e:
            return AgentResult(
                agent_name=self._name, idea_id=idea.id, success=False, error=str(e)
            )
