"""WriterAgent (incidente run_431a9c5d, 24-jul-2026): el writer nunca
debe dar por bueno un informe de longitud cero. La defensa principal es
el proveedor (LLMProvider._call_api reintenta y el router rota — ver
test_empty_response.py); esto cubre el cinturón de seguridad del propio
WriterAgent, por si `self._llm` alguna vez no pasa por ese mecanismo.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from creative_engine.agents.writer import WriterAgent
from creative_engine.core.exceptions import LLMEmptyResponseError
from creative_engine.core.models import EvaluationScores, Idea


def _evaluated_idea() -> Idea:
    idea = Idea(
        title="Idea de prueba",
        description="Descripción de prueba suficientemente larga para validar el modelo.",
    )
    idea.evaluation = EvaluationScores(
        novelty=0.5, utility=0.7, feasibility=0.6, market_fit=0.5
    )
    return idea


class TestWriterNeverAcceptsEmptyReport:
    async def test_returns_report_on_success(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "Informe ejecutivo con contenido real."
        agent = WriterAgent(llm)

        report = await agent.write_report(_evaluated_idea())

        assert report == "Informe ejecutivo con contenido real."

    async def test_empty_string_never_returned_as_success(self) -> None:
        """Cinturón de seguridad: si el LLM (por lo que sea) devuelve una
        cadena vacía sin lanzar excepción, el writer no la acepta."""
        llm = AsyncMock()
        llm.generate.return_value = ""
        agent = WriterAgent(llm)

        report = await agent.write_report(_evaluated_idea())

        assert report != ""
        assert "no generó contenido" in report

    async def test_whitespace_only_never_returned_as_success(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "   \n  "
        agent = WriterAgent(llm)

        report = await agent.write_report(_evaluated_idea())

        assert report.strip() != ""
        assert "no generó contenido" in report

    async def test_persistent_empty_response_error_surfaces_as_clear_message(self) -> None:
        """Cuando el proveedor/router agotan reintento + rotación (ver
        test_empty_response.py) y siguen sin contenido, LLMEmptyResponseError
        llega hasta aquí — el writer lo convierte en un mensaje claro, nunca
        en una cadena vacía silenciosa."""
        llm = AsyncMock()
        llm.generate.side_effect = LLMEmptyResponseError("todos los proveedores vacíos")
        agent = WriterAgent(llm)

        report = await agent.write_report(_evaluated_idea())

        assert report != ""
        assert "Error generando el informe" in report

    async def test_unevaluated_idea_still_handled(self) -> None:
        llm = AsyncMock()
        agent = WriterAgent(llm)
        idea = Idea(title="Sin evaluar", description="Idea sin evaluación todavía, de prueba.")

        report = await agent.write_report(idea)

        assert report != ""
        llm.generate.assert_not_called()
