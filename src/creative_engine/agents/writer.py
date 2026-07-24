"""Agente Escritor: produce informes ejecutivos de ideas evaluadas."""

from __future__ import annotations

import structlog

from ..core.models import Idea
from ..llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

WRITER_PROMPT = """Eres un consultor de innovación senior. Escribe un informe ejecutivo
para la siguiente idea creativa que ha sido evaluada por un comité de expertos.

== IDEA ==
Título: {title}
Descripción: {description}
Ventajas: {advantages}
Limitaciones: {limitations}
Hipótesis de Valor: {value_hypothesis}
Generación Evolutiva: {generation}

== EVALUACIONES ==
- Novedad objetiva (distancia semántica al resto de élites): {novelty}/1.0
- Utilidad {utility}/1.0: {innovation_feedback}
- Viabilidad {feasibility}/1.0: {feasibility_feedback}
- Market Fit {market_fit}/1.0: {market_feedback}

== INSTRUCCIONES ==
Escribe un informe que incluya:
1. Resumen Ejecutivo (2 frases)
2. Descripción del Concepto (para un público no técnico)
3. Análisis de Viabilidad y Riesgos
4. Estrategia de Mercado Sugerida
5. Próximos Pasos Recomendados

Tono: profesional, objetivo, orientado a la acción."""


class WriterAgent:
    """Agente que genera informes detallados de ideas."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._log = logger.bind(agent="writer")

    async def write_report(self, idea: Idea, max_tokens: int = 2000) -> str:
        """Genera un informe ejecutivo para una idea evaluada."""
        if not idea.evaluation:
            return "Error: la idea no ha sido evaluada aún."

        e = idea.evaluation
        vh_text = ""
        if idea.value_hypothesis:
            vh_text = (
                f"Usuario: {idea.value_hypothesis.target_user}. "
                f"Problema: {idea.value_hypothesis.problem_solved}. "
                f"Propuesta: {idea.value_hypothesis.value_proposition}."
            )

        prompt = WRITER_PROMPT.format(
            title=idea.title,
            description=idea.description,
            advantages="\n".join(f"- {a}" for a in idea.advantages) or "—",
            limitations="\n".join(f"- {li}" for li in idea.limitations) or "—",
            value_hypothesis=vh_text or "No definida",
            generation=idea.generation,
            novelty=round(e.novelty, 2),
            utility=round(e.utility, 2),
            feasibility=round(e.feasibility, 2),
            market_fit=round(e.market_fit, 2),
            innovation_feedback=e.agent_feedback.get("innovation", "N/A"),
            feasibility_feedback=e.agent_feedback.get("feasibility", "N/A"),
            market_feedback=e.agent_feedback.get("market", "N/A"),
        )

        try:
            report = await self._llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=max_tokens,
            )
            # El proveedor (LLMProvider._call_api) ya reintenta y rota de
            # proveedor ante contenido vacío (LLMEmptyResponseError, ver
            # core/exceptions.py) — esto es un cinturón de seguridad
            # adicional: el writer NUNCA da por bueno un informe de
            # longitud cero, ni siquiera si algún día `self._llm` es un
            # proveedor sin ese mecanismo (p.ej. un doble en tests).
            if not report or not report.strip():
                self._log.error("report_generation_empty", idea_id=idea.id)
                return "Informe no disponible: el proveedor no generó contenido."
            self._log.info("report_generated", idea_id=idea.id, length=len(report))
            return report
        except Exception as e:
            self._log.error("report_generation_failed", idea_id=idea.id, error=str(e))
            return f"Error generando el informe: {e}"
