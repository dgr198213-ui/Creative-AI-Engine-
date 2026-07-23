"""Test: la evaluación de un lote respeta el límite de concurrencia.

Evaluar todas las ideas a la vez saturaba los rate limits y provocaba
timeouts en cascada (visto en producción). El motor debe limitar cuántas
ideas evalúa en paralelo.
"""

import asyncio

from creative_engine.core.config import get_settings, reset_settings
from creative_engine.core.models import EvolutionRequest


async def test_eval_batch_respects_concurrency_limit(deterministic_embed, monkeypatch) -> None:
    import json
    from unittest.mock import AsyncMock

    from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
    from creative_engine.agents.feasibility import FeasibilityAgent
    from creative_engine.agents.generator import IdeaGeneratorAgent
    from creative_engine.agents.innovation import InnovationAgent
    from creative_engine.agents.market import MarketAgent
    from creative_engine.evolution import encoders as enc
    from creative_engine.evolution.crossover import CrossoverEngine
    from creative_engine.evolution.encoders import IdeaEncoder
    from creative_engine.evolution.mutation import MutationEngine
    from creative_engine.evolution.qd_engine import QDEngine

    reset_settings()
    settings = get_settings()
    settings.evolution.max_concurrent_evaluations = 2  # límite bajo para el test

    # Rastreamos cuántas evaluaciones de idea corren a la vez.
    active = {"now": 0, "peak": 0}

    def gen(prompt, **kw):
        if "array" in prompt:
            return json.dumps(
                [
                    {
                        "title": f"Idea {i} diversa",
                        "description": f"Descripción de la idea {i} suficientemente larga.",
                        "advantages": ["A"],
                        "limitations": ["X"],
                        "features": {"complexity_level": 0.5},
                    }
                    for i in range(4)
                ]
            )
        return json.dumps(
            {
                "title": "Idea mutada",
                "description": "Evolución con giro distinto y detallado.",
                "advantages": ["Aa"],
                "limitations": ["Xx"],
                "mutation_description": "c",
            }
        )

    llm = AsyncMock()
    llm.generate.side_effect = gen

    async def slow_structured(*a, **k):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return {"score": 0.7, "feedback": "ok", "estimated_complexity": 0.5}

    llm.generate_structured.side_effect = slow_structured

    monkeypatch.setattr(enc.IdeaEncoder, "_embed", lambda self, text: deterministic_embed(text))

    evaluator = EvaluatorOrchestrator(
        agents={
            "innovation": InnovationAgent(llm),
            "feasibility": FeasibilityAgent(llm),
            "market": MarketAgent(llm),
        }
    )
    engine = QDEngine(
        generator=IdeaGeneratorAgent(llm),
        evaluator=evaluator,
        mutation=MutationEngine(llm),
        crossover=CrossoverEngine(llm),
        encoder=IdeaEncoder(),
        repository=None,
    )

    await engine.run_evolution(
        EvolutionRequest(
            challenge="Movilidad urbana sostenible e innovadora",
            domain="generic",
            population_size=4,
            generations=1,
        )
    )

    reset_settings()
    # Cada idea = 3 agentes en paralelo; con límite de 2 ideas → pico ≤ 6.
    # Sin el límite, las 4+ ideas correrían a la vez (pico ≥ 12).
    assert active["peak"] <= 6, f"pico de concurrencia {active['peak']} supera el límite"
