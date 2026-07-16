"""Agente de Mercado: evalúa el encaje mercado-idea."""

from __future__ import annotations

from typing import Any

import structlog

from ..core.models import Idea
from ..llm.provider import LLMProvider
from .base import AgentResult, BaseAgent

logger = structlog.get_logger(__name__)

MARKET_SYSTEM = """Eres un analista de mercado experto. Evalúa el encaje de una idea
con su mercado objetivo. Considera tamaño de mercado, competencia,
voluntad de pago y barreras de entrada.

Sé realista sobre la adopción. La mayoría de las innovaciones
tardan años en ser adoptadas."""

MARKET_PROMPT = """Evalúa el MARKET FIT de esta idea (0 a 1):

Título: {title}
Descripción: {description}
Mercados objetivo: {markets}
Ventajas clave: {advantages}

Contexto del reto: {challenge}

Responde SOLO en JSON:
{{
    "score": 0.0,
    "feedback": "Análisis de encaje con el mercado.",
    "target_audience_size": "grande/mediano/pequeño/nicho",
    "competitive_landscape": "descripción breve de la competencia",
    "adoption_barriers": ["barrera 1", "barrera 2"]
}}"""


class MarketAgent(BaseAgent):
    """Agente que evalúa el encaje con el mercado."""

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__(name="market", llm=llm, timeout_seconds=90.0)

    async def evaluate(
        self, idea: Idea, context: dict[str, Any] | None = None
    ) -> AgentResult:
        context = context or {}
        prompt = MARKET_PROMPT.format(
            title=idea.title,
            description=idea.description,
            markets=", ".join(idea.features.target_markets) or "No especificados",
            advantages="\n".join(f"- {a}" for a in idea.advantages[:3]) or "—",
            challenge=context.get("challenge", ""),
        )

        try:
            data = await self._llm.generate_structured(
                prompt=prompt, system_prompt=MARKET_SYSTEM
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
