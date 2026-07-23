"""Regresión del dominio `generic` (Fase 6, domain packs).

Requisito no negociable del diseño: tras reemplazar el enum `DomainName`
por un registro dinámico de packs (`configs/domains/`), un run en el
dominio `generic` con LLM mockeado debe producir EXACTAMENTE el mismo
resultado que antes de la refactorización — hay un benchmark en curso y
su comparabilidad depende de ello.

Semilla fija en el RNG del motor (`engine._rng`) + embeddings
deterministas (`conftest.fake_embed`) + LLM mockeado con respuestas
deterministas → el resultado completo es reproducible byte a byte. Los
valores esperados se capturaron ejecutando este mismo test contra el
código anterior a la Fase 6; si cualquiera cambia, el dominio genérico
dejó de comportarse igual.

Usa la cadena literal "generic" (no un enum) a propósito: así el test
sigue siendo válido sin modificarlo ni antes ni después de retirar
`DomainName`, lo cual es la prueba misma de que el comportamiento no cambió.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import numpy as np

from creative_engine.agents.combined_evaluator import CombinedEvaluatorAgent
from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
from creative_engine.agents.generator import IdeaGeneratorAgent
from creative_engine.core.models import EvolutionRequest
from creative_engine.evolution.crossover import CrossoverEngine
from creative_engine.evolution.encoders import IdeaEncoder
from creative_engine.evolution.mutation import MutationEngine
from creative_engine.evolution.qd_engine import QDEngine

_TOPICS = [
    ("Bicicleta solar plegable", "Bicicleta urbana con paneles solares flexibles integrados."),
    ("Red de trueque vecinal", "Plataforma para intercambiar objetos y servicios entre vecinos."),
    ("Dron sanitario rural", "Dron autónomo que entrega medicamentos en zonas aisladas."),
    ("Huerto vertical modular", "Sistema hidropónico apilable para fachadas y balcones."),
]


def _make_generate():
    counter = {"n": 0}

    def _generate(prompt: str, **kwargs) -> str:
        if "Genera" in prompt and "array" in prompt:
            items = []
            for _ in range(4):
                topic = _TOPICS[counter["n"] % len(_TOPICS)]
                counter["n"] += 1
                items.append(
                    {
                        "title": f"{topic[0]} v{counter['n']}",
                        "description": topic[1] + f" Variante {counter['n']}.",
                        "advantages": ["Ventaja A", "Ventaja B"],
                        "limitations": ["Limitación X"],
                        "features": {"technologies": ["tech"], "complexity_level": 0.5},
                    }
                )
            return json.dumps(items)
        topic = _TOPICS[counter["n"] % len(_TOPICS)]
        counter["n"] += 1
        return json.dumps(
            {
                "title": f"{topic[0]} mutada v{counter['n']}",
                "description": topic[1] + f" Evolución {counter['n']}.",
                "advantages": ["Ventaja evolucionada"],
                "limitations": ["Limitación"],
                "mutation_description": "cambio simulado",
            }
        )

    return _generate


def _make_generate_structured():
    def _generate_structured(prompt: str, **kwargs) -> dict:
        return {
            "utility": 0.7,
            "utility_feedback": "ok",
            "feasibility": 0.6,
            "feasibility_feedback": "ok",
            "market_fit": 0.55,
            "market_feedback": "ok",
            "estimated_complexity": 0.5,
        }

    return _generate_structured


def _build_engine(llm, encoder: IdeaEncoder) -> QDEngine:
    evaluator = EvaluatorOrchestrator(agents={"combined": CombinedEvaluatorAgent(llm)})
    engine = QDEngine(
        generator=IdeaGeneratorAgent(llm),
        evaluator=evaluator,
        mutation=MutationEngine(llm),
        crossover=CrossoverEngine(llm),
        encoder=encoder,
        repository=None,
    )
    engine._rng = np.random.default_rng(42)  # determinismo total en selección de padres
    return engine


async def test_generic_domain_produces_identical_result(deterministic_embed) -> None:
    llm = AsyncMock()
    llm.generate.side_effect = _make_generate()
    llm.generate_structured.side_effect = _make_generate_structured()

    engine = _build_engine(llm, IdeaEncoder(embed_fn=deterministic_embed))
    request = EvolutionRequest(
        challenge="Diseña soluciones innovadoras para movilidad urbana sostenible",
        domain="generic",
        population_size=6,
        generations=2,
    )

    state = await engine.run_evolution(request)

    # Valores capturados ejecutando este test contra el código anterior a
    # la Fase 6 (domain packs). Cualquier cambio aquí es una regresión de
    # comportamiento del dominio `generic`, no un ajuste de test.
    assert state.status == "completed"
    assert state.generation == 2
    assert state.domain == "generic"
    assert len(state.all_ideas) == 18
    assert len(state.archive) == 18
    assert round(state.coverage, 6) == round(18 / 800, 6)
    assert round(state.qd_score, 4) == 11.1510
    assert round(state.best_fitness, 4) == 0.6195

    titles_and_fitness = sorted(
        (cell.elite.title, round(cell.fitness, 4)) for cell in state.archive
    )
    assert titles_and_fitness == [
        ("Bicicleta solar plegable mutada v13", 0.6195),
        ("Bicicleta solar plegable mutada v21", 0.6195),
        ("Bicicleta solar plegable mutada v9", 0.6195),
        ("Bicicleta solar plegable v1", 0.6195),
        ("Bicicleta solar plegable v5", 0.6195),
        ("Dron sanitario rural mutada v11", 0.6195),
        ("Dron sanitario rural mutada v19", 0.6195),
        ("Dron sanitario rural v23", 0.6195),
        ("Dron sanitario rural v3", 0.6195),
        ("Huerto vertical modular mutada v12", 0.6195),
        ("Huerto vertical modular mutada v20", 0.6195),
        ("Huerto vertical modular v4", 0.6195),
        ("Red de trueque vecinal mutada v10", 0.6195),
        ("Red de trueque vecinal mutada v18", 0.6195),
        ("Red de trueque vecinal mutada v22", 0.6195),
        ("Red de trueque vecinal v14", 0.6195),
        ("Red de trueque vecinal v2", 0.6195),
        ("Red de trueque vecinal v6", 0.6195),
    ]
