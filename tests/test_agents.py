"""Tests de los agentes evaluadores y del orquestador."""

from creative_engine.agents.base import AgentResult
from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
from creative_engine.agents.feasibility import FeasibilityAgent
from creative_engine.agents.innovation import InnovationAgent
from creative_engine.agents.market import MarketAgent
from creative_engine.core.models import Idea


async def test_agent_returns_valid_result(mock_llm_provider) -> None:
    agent = InnovationAgent(mock_llm_provider)

    idea = Idea(
        title="Bicicleta Voladora",
        description="Bicicleta que utiliza hélices para volar sobre el tráfico urbano.",
        domain="generic",
    )

    result = await agent.safe_evaluate(idea, context={"challenge": "Movilidad urbana"})

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.agent_name == "innovation"
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert result.feedback != ""
    assert result.latency_ms > 0


async def test_agent_handles_llm_error(mock_llm_provider) -> None:
    mock_llm_provider.generate_structured.side_effect = Exception("API Down")

    agent = FeasibilityAgent(mock_llm_provider)
    idea = Idea(title="Test", description="Test error handling suficientemente largo")

    result = await agent.safe_evaluate(idea)

    assert result.success is False
    assert "API Down" in (result.error or "")


async def test_orchestrator_aggregates_quality_scores(mock_llm_provider) -> None:
    orchestrator = EvaluatorOrchestrator(
        agents={
            "innovation": InnovationAgent(mock_llm_provider),
            "feasibility": FeasibilityAgent(mock_llm_provider),
            "market": MarketAgent(mock_llm_provider),
        }
    )

    idea = Idea(
        title="Bicicleta Solar",
        description="Bicicleta urbana con panel solar integrado en el cuadro delantero.",
    )

    scores = await orchestrator.evaluate_idea(idea, context={"challenge": "Movilidad"})

    # El mock devuelve 0.75 para todos los agentes
    assert scores.utility == 0.75
    assert scores.feasibility == 0.75
    assert scores.market_fit == 0.75
    # La novedad NO la asigna el orquestador (la calcula el motor QD)
    assert scores.novelty == 0.0
    # Impacto derivado: 0.75*0.5 + 0.75*0.3 + 0.75*0.2 = 0.75
    assert scores.impact == 0.75
    assert idea.evaluation is scores
    assert idea.fitness > 0.0
