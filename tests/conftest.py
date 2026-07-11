"""Fixtures compartidas para los tests."""

import hashlib
from unittest.mock import AsyncMock

import pytest

from creative_engine.core.models import (
    DomainName,
    EvaluationScores,
    Idea,
    IdeaStatus,
)


@pytest.fixture
def mock_llm_provider():
    """Mock de un proveedor LLM que retorna JSON válido."""
    provider = AsyncMock()

    provider.generate.return_value = """```json
{
    "title": "Idea Mutada de Prueba",
    "description": "Descripción de la idea generada por el mock del LLM para testing.",
    "advantages": ["Ventaja mock 1", "Ventaja mock 2"],
    "limitations": ["Limitación mock 1"],
    "mutation_description": "Mutación simulada"
}
```"""

    provider.generate_structured.return_value = {
        "score": 0.75,
        "feedback": "Feedback simulado del agente.",
        "similar_concepts": [],
        "estimated_complexity": 0.6,
    }

    return provider


def fake_embed(text: str) -> list[float]:
    """Embedding determinista de 384 dims derivado del hash del texto."""
    digest = hashlib.sha256(text.encode()).digest()
    values = []
    seed_bytes = (digest * ((384 * 2) // len(digest) + 1))[: 384 * 2]
    for i in range(384):
        pair = seed_bytes[i * 2 : i * 2 + 2]
        values.append((int.from_bytes(pair, "big") / 65535.0) * 2 - 1)
    return values


@pytest.fixture
def deterministic_embed():
    return fake_embed


@pytest.fixture
def evaluated_idea() -> Idea:
    """Idea completamente evaluada lista para MAP-Elites."""
    idea = Idea(
        title="Bicicleta Modular Urbana",
        description=(
            "Sistema de bicicleta urbana con componentes modulares "
            "intercambiables para adaptarse a diferentes necesidades "
            "de transporte diario."
        ),
        advantages=["Adaptable", "Fácil reparación", "Vida útil extendida"],
        limitations=["Peso superior", "Complejidad en uniones"],
        status=IdeaStatus.EVALUATED,
        generation=2,
        domain=DomainName.INDUSTRIAL_DESIGN,
    )
    idea.evaluation = EvaluationScores(
        novelty=0.85,
        utility=0.90,
        feasibility=0.70,
        complexity=0.65,
        impact=0.80,
        market_fit=0.75,
        sustainability=0.85,
        scalability=0.60,
    )
    idea.behavior_descriptor = [0.85, 0.70]
    idea.genome_vector = [0.1] * 384
    return idea
