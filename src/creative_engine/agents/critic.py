"""Agente Crítico (opcional): feedback cualitativo sobre debilidades.

Nota de diseño: la puntuación de NOVEDAD ya NO la decide este agente.
La novedad se calcula de forma objetiva en el motor (distancia de
embedding al archivo de élites). Este agente aporta crítica cualitativa
y conceptos similares conocidos, útil en el informe final.
"""

from __future__ import annotations

from typing import Any

import structlog

from ..core.models import Idea
from ..llm.provider import LLMProvider
from .base import AgentResult, BaseAgent

logger = structlog.get_logger(__name__)

CRITIC_SYSTEM_PROMPT = """Eres un crítico creativo experto. Tu trabajo es detectar debilidades,
supuestos frágiles y conceptos ya existentes similares a la idea.
Sé severo pero constructivo."""

CRITIC_EVAL_PROMPT = """Critica esta idea:

Título: {title}
Descripción: {description}
Ventajas: {advantages}
Tecnologías: {technologies}
Mercados objetivo: {markets}

Contexto del reto: {challenge}

Responde SOLO en JSON:
{{
    "score": 0.5,
    "feedback": "Análisis de 2-3 frases sobre las debilidades principales.",
    "similar_concepts": ["concepto similar 1", "concepto similar 2"],
    "weak_assumptions": ["supuesto frágil 1"]
}}"""


class CriticAgent(BaseAgent):
    """Agente de crítica cualitativa (no participa en el fitness por defecto)."""

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(name="critic", llm=llm, timeout_seconds=60.0)

    async def evaluate(
        self, idea: Idea, context: dict[str, Any] | None = None
    ) -> AgentResult:
        context = context or {}
        prompt = CRITIC_EVAL_PROMPT.format(
            title=idea.title,
            description=idea.description,
            advantages="\n".join(f"- {a}" for a in idea.advantages) or "—",
            technologies=", ".join(idea.features.technologies) or "No especificadas",
            markets=", ".join(idea.features.target_markets) or "No especificados",
            challenge=context.get("challenge", ""),
        )

        try:
            data = await self._llm.generate_structured(
                prompt=prompt, system_prompt=CRITIC_SYSTEM_PROMPT
            )
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            return AgentResult(
                agent_name=self._name,
                idea_id=idea.id,
                success=True,
                score=score,
                feedback=str(data.get("feedback", "")),
                metadata={
                    "similar_concepts": data.get("similar_concepts", []),
                    "weak_assumptions": data.get("weak_assumptions", []),
                },
            )
        except Exception as e:
            return AgentResult(
                agent_name=self._name, idea_id=idea.id, success=False, error=str(e)
            )
