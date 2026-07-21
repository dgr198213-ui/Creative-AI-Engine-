"""Test de integración del ciclo QD completo con LLM simulado (sin red, sin BD)."""

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
    ("Bicicleta solar plegable", "Bicicleta urbana con paneles solares flexibles integrados."),
    ("Red de trueque vecinal", "Plataforma para intercambiar objetos y servicios entre vecinos."),
    ("Dron sanitario rural", "Dron autónomo que entrega medicamentos en zonas aisladas."),
    ("Huerto vertical modular", "Sistema hidropónico apilable para fachadas y balcones."),
    ("Ropa con sensores térmicos", "Prendas que regulan la temperatura corporal activamente."),
    ("Cocina comunitaria móvil", "Food-truck cooperativo gestionado por barrios."),
]

_counter = {"n": 0}


def _fake_generate(prompt: str, **kwargs) -> str:
    """Devuelve un array de ideas para el generador y un objeto para mutación/cruce."""
    if "Genera" in prompt and "array" in prompt:
        n = 3
        items = []
        for _ in range(n):
            topic = _TOPICS[_counter["n"] % len(_TOPICS)]
            _counter["n"] += 1
            items.append(
                {
                    "title": f"{topic[0]} v{_counter['n']}",
                    "description": topic[1] + f" Variante número {_counter['n']}.",
                    "advantages": ["Ventaja A", "Ventaja B"],
                    "limitations": ["Limitación X"],
                    "value_hypothesis": {
                        "target_user": "Usuarios urbanos",
                        "problem_solved": "Un problema real",
                        "value_proposition": "Valor diferencial claro",
                    },
                    "features": {"technologies": ["tech"], "complexity_level": 0.5},
                }
            )
        return json.dumps(items)

    topic = _TOPICS[_counter["n"] % len(_TOPICS)]
    _counter["n"] += 1
    return json.dumps(
        {
            "title": f"{topic[0]} mutada v{_counter['n']}",
            "description": topic[1] + f" Evolución {_counter['n']} con giro inesperado.",
            "advantages": ["Ventaja evolucionada"],
            "limitations": ["Limitación"],
            "mutation_description": "cambio simulado",
        }
    )


@pytest.fixture
def sim_llm():
    provider = AsyncMock()
    provider.generate.side_effect = _fake_generate
    provider.generate_structured.return_value = {
        "score": 0.7,
        "feedback": "Feedback simulado.",
        "estimated_complexity": 0.5,
    }
    return provider


async def test_full_evolution_cycle(sim_llm, deterministic_embed) -> None:
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
        repository=None,  # sin BD en el test
    )

    request = EvolutionRequest(
        challenge="Diseña soluciones innovadoras para movilidad urbana sostenible",
        domain=DomainName.GENERIC,
        population_size=6,
        generations=2,
    )

    state = await engine.run_evolution(request)

    assert state.is_running is False
    assert state.completed_at is not None
    assert state.generation == 2
    assert len(state.all_ideas) > 0
    assert len(state.archive) >= 2, "Debe haber múltiples élites diversas"
    assert 0.0 < state.coverage <= 1.0
    assert state.qd_score > 0.0
    assert state.best_fitness > 0.0

    # Todas las élites tienen evaluación completa y novedad objetiva asignada
    for cell in state.archive:
        assert cell.elite.evaluation is not None
        assert 0.0 <= cell.elite.evaluation.novelty <= 1.0
        assert len(cell.elite.behavior_descriptor) == 3

    # Las élites ocupan celdas distintas → el abanico de ideas es diverso
    cells = {cell.cell_index for cell in state.archive}
    assert len(cells) == len(state.archive)


async def test_population_rebuild_after_outage(sim_llm, deterministic_embed) -> None:
    """Si el archivo queda vacío (apagón de proveedores en la población
    inicial), la siguiente generación reconstruye la población completa con
    inyección fresca en vez de gotear 1 idea por generación."""
    from unittest.mock import patch

    from creative_engine.agents.combined_evaluator import CombinedEvaluatorAgent

    evaluator = EvaluatorOrchestrator(agents={"combined": CombinedEvaluatorAgent(sim_llm)})
    generator = IdeaGeneratorAgent(sim_llm)
    engine = QDEngine(
        generator=generator,
        evaluator=evaluator,
        mutation=MutationEngine(sim_llm),
        crossover=CrossoverEngine(sim_llm),
        encoder=IdeaEncoder(embed_fn=deterministic_embed),
        repository=None,
    )

    # Población inicial fallida (apagón): devuelve 0 ideas
    original = generator.generate_population
    calls: list[int] = []

    async def flaky(challenge, domain, count, variation_hint=""):
        calls.append(count)
        if len(calls) == 1:
            return []  # apagón total en la generación 0
        return await original(
            challenge=challenge, domain=domain, count=count, variation_hint=variation_hint
        )

    with patch.object(generator, "generate_population", side_effect=flaky):
        state = await engine.run_evolution(
            EvolutionRequest(
                challenge="Movilidad urbana sostenible e innovadora",
                domain=DomainName.GENERIC,
                population_size=6,
                generations=1,
            )
        )

    # La generación 1 pidió la población COMPLETA (6), no el goteo de 1
    assert calls[0] == 6  # intento inicial
    assert 6 in calls[1:], f"esperada reconstrucción con count=6, llamadas: {calls}"
    assert len(state.archive) >= 3  # el run se recuperó de verdad
