"""Test end-to-end del streaming SSE con motor simulado (sin red, sin BD)."""

import json
from unittest.mock import AsyncMock

import pytest

from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
from creative_engine.agents.feasibility import FeasibilityAgent
from creative_engine.agents.generator import IdeaGeneratorAgent
from creative_engine.agents.innovation import InnovationAgent
from creative_engine.agents.market import MarketAgent
from creative_engine.core.models import DomainName, EvolutionRequest
from creative_engine.evolution.crossover import CrossoverEngine
from creative_engine.evolution.encoders import IdeaEncoder
from creative_engine.evolution.mutation import MutationEngine
from creative_engine.evolution.qd_engine import QDEngine

_TOPICS = [
    ("Bicicleta solar", "Bicicleta urbana con paneles solares integrados."),
    ("Trueque vecinal", "Plataforma para intercambiar objetos entre vecinos."),
    ("Dron sanitario", "Dron autónomo que entrega medicamentos en zonas aisladas."),
    ("Huerto vertical", "Sistema hidropónico apilable para fachadas."),
]
_c = {"n": 0}


def _gen(prompt: str, **kwargs) -> str:
    if "array" in prompt:
        items = []
        for _ in range(3):
            t = _TOPICS[_c["n"] % len(_TOPICS)]
            _c["n"] += 1
            items.append(
                {
                    "title": f"{t[0]} v{_c['n']}",
                    "description": t[1] + f" Variante {_c['n']}.",
                    "advantages": ["A", "B"],
                    "limitations": ["X"],
                    "features": {"technologies": ["tech"], "complexity_level": 0.5},
                }
            )
        return json.dumps(items)
    t = _TOPICS[_c["n"] % len(_TOPICS)]
    _c["n"] += 1
    return json.dumps(
        {
            "title": f"{t[0]} mutada v{_c['n']}",
            "description": t[1] + f" Evolución {_c['n']}.",
            "advantages": ["Aa"],
            "limitations": ["Xx"],
            "mutation_description": "cambio",
        }
    )


@pytest.fixture
def sim_llm():
    p = AsyncMock()
    p.generate.side_effect = _gen
    p.generate_structured.return_value = {"score": 0.7, "feedback": "ok", "estimated_complexity": 0.5}
    return p


async def test_callback_receives_families_per_generation(sim_llm, deterministic_embed) -> None:
    """El callback on_generation debe recibir familias en vivo cada generación."""
    from creative_engine.evolution.clustering import group_into_families

    snapshots: list[int] = []

    async def on_generation(generation: int, cells: list) -> None:
        families = group_into_families([c.elite for c in cells])
        snapshots.append(generation)
        # Cada snapshot debe tener al menos una familia con representante válido
        assert all(f.representative.title for f in families)

    evaluator = EvaluatorOrchestrator(
        agents={
            "innovation": InnovationAgent(sim_llm),
            "feasibility": FeasibilityAgent(sim_llm),
            "market": MarketAgent(sim_llm),
        }
    )
    engine = QDEngine(
        generator=IdeaGeneratorAgent(sim_llm),
        evaluator=evaluator,
        mutation=MutationEngine(sim_llm),
        crossover=CrossoverEngine(sim_llm),
        encoder=IdeaEncoder(embed_fn=deterministic_embed),
        repository=None,
        on_generation=on_generation,
    )

    request = EvolutionRequest(
        challenge="Soluciones para movilidad urbana sostenible",
        domain=DomainName.GENERIC,
        population_size=6,
        generations=3,
    )
    state = await engine.run_evolution(request)

    # Se recibió un snapshot por cada generación (1, 2, 3)
    assert snapshots == [1, 2, 3]
    assert state.generation == 3
    assert len(state.archive) >= 2
