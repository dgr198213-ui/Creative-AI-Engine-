"""Agente de Innovación: evalúa la UTILIDAD real de la idea."""

from __future__ import annotations

from typing import Any

import structlog

from ..core.models import Idea
from ..llm.provider import LLMProvider
from .base import AgentResult, BaseAgent

logger = structlog.get_logger(__name__)

INNOVATION_SYSTEM = """Eres un experto en estrategia de innovación. Tu única misión es evaluar
la UTILIDAD de una idea. No te importa si es novedosa o viable,
solo si es verdaderamente útil para el usuario objetivo.

Una idea útil:
- Resuelve un dolor real o significativo
- El usuario estaría dispuesto a pagar por ella
- Mejora sustancialmente el status quo
- No es una solución buscando problema"""

INNOVATION_PROMPT = """Evalúa la UTILIDAD de esta idea (0 a 1):

Título: {title}
Descripción: {description}
Ventajas: {advantages}
Hipótesis de valor: {value_hypothesis}

Contexto del reto: {challenge}

Responde SOLO en JSON:
{{
    "score": 0.0,
    "feedback": "Por qué es o no útil. ¿Resuelve un dolor real?",
    "missing_utilities": ["qué utilidad podría añadirse"],
    "target_user_validation": "si el usuario objetivo realmente lo necesitaría"
}}"""


class InnovationAgent(BaseAgent):
    """Agente que evalúa la utilidad y el potencial de mejora."""

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(name="innovation", llm=llm, timeout_seconds=60.0)

    async def evaluate(
        self, idea: Idea, context: dict[str, Any] | None = None
    ) -> AgentResult:
        context = context or {}
        vh_text = ""
        if idea.value_hypothesis:
            vh_text = (
                f"Usuario: {idea.value_hypothesis.target_user}\n"
                f"Problema: {idea.value_hypothesis.problem_solved}\n"
                f"Valor: {idea.value_hypothesis.value_proposition}"
            )

        prompt = INNOVATION_PROMPT.format(
            title=idea.title,
            description=idea.description,
            advantages="\n".join(f"- {a}" for a in idea.advantages) or "—",
            value_hypothesis=vh_text or "No definida",
            challenge=context.get("challenge", ""),
        )

        try:
            data = await self._llm.generate_structured(
                prompt=prompt, system_prompt=INNOVATION_SYSTEM
            )
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
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
