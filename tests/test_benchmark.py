"""Tests del benchmark motor QD vs prompt único."""

import json
from unittest.mock import AsyncMock

import pytest

from creative_engine.benchmark import (
    distinct_cells,
    pairwise_diversity,
    run_benchmark,
)
from creative_engine.core.config import default_generic_domain
from creative_engine.core.models import Idea


def _idea(title: str, genome: list[float], descriptor: list[float]) -> Idea:
    idea = Idea(title=title, description=f"Descripción larga de {title} para validar.")
    idea.genome_vector = genome
    idea.behavior_descriptor = descriptor
    return idea


class TestDiversityMetrics:
    def test_identical_ideas_zero_diversity(self) -> None:
        g = [1.0, 0.0, 0.0]
        ideas = [_idea("Aaa", g, [0.5, 0.5, 0.5]), _idea("Bbb", g, [0.5, 0.5, 0.5])]
        mean_d, min_d = pairwise_diversity(ideas)
        assert mean_d == pytest.approx(0.0, abs=1e-9)
        assert min_d == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_ideas_half_diversity(self) -> None:
        ideas = [
            _idea("Aaa", [1.0, 0.0], [0.1, 0.1, 0.1]),
            _idea("Bbb", [0.0, 1.0], [0.9, 0.9, 0.9]),
        ]
        mean_d, _ = pairwise_diversity(ideas)
        assert mean_d == pytest.approx(0.5, abs=1e-6)  # (1-0)/2

    def test_single_idea_zero(self) -> None:
        assert pairwise_diversity([_idea("Aaa", [1.0], [0.5, 0.5, 0.5])]) == (0.0, 0.0)

    def test_min_detects_clones(self) -> None:
        """Tres ideas: dos clones y una lejana → media alta pero mínima ~0."""
        g = [1.0, 0.0]
        ideas = [
            _idea("Clon1", g, [0.1, 0.1, 0.1]),
            _idea("Clon2", g, [0.1, 0.1, 0.1]),
            _idea("Lejana", [0.0, 1.0], [0.9, 0.9, 0.9]),
        ]
        mean_d, min_d = pairwise_diversity(ideas)
        assert min_d == pytest.approx(0.0, abs=1e-9)
        assert mean_d > 0.3


class TestDistinctCells:
    def test_counts_unique_cells(self) -> None:
        domain = default_generic_domain()
        ideas = [
            _idea("Aaa", [1.0], [0.05, 0.05, 0.05]),
            _idea("Bbb", [1.0], [0.06, 0.05, 0.05]),  # misma celda que Aaa
            _idea("Ccc", [1.0], [0.95, 0.95, 0.95]),
        ]
        assert distinct_cells(ideas, domain) == 2

    def test_ignores_missing_descriptor(self) -> None:
        domain = default_generic_domain()
        idea = _idea("Aaa", [1.0], [])
        assert distinct_cells([idea], domain) == 0


class TestRunBenchmark:
    async def test_full_benchmark_with_mocks(self, deterministic_embed) -> None:
        from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
        from creative_engine.agents.feasibility import FeasibilityAgent
        from creative_engine.agents.generator import IdeaGeneratorAgent
        from creative_engine.agents.innovation import InnovationAgent
        from creative_engine.agents.market import MarketAgent
        from creative_engine.evolution.crossover import CrossoverEngine
        from creative_engine.evolution.encoders import IdeaEncoder
        from creative_engine.evolution.mutation import MutationEngine
        from creative_engine.evolution.qd_engine import QDEngine

        topics = [
            "bicicleta solar urbana",
            "red de trueque vecinal",
            "dron sanitario rural",
            "huerto vertical modular",
            "ropa con sensores térmicos",
            "cocina comunitaria móvil",
        ]
        counter = {"n": 0}

        def gen(prompt, **kw):
            if "array" in prompt:
                items = []
                for _ in range(3):
                    t = topics[counter["n"] % len(topics)]
                    counter["n"] += 1
                    items.append(
                        {
                            "title": f"{t} v{counter['n']}",
                            "description": f"Idea sobre {t}, variante {counter['n']} bien detallada.",
                            "advantages": ["A"],
                            "limitations": ["X"],
                            "features": {"complexity_level": 0.5},
                        }
                    )
                return json.dumps(items)
            t = topics[counter["n"] % len(topics)]
            counter["n"] += 1
            return json.dumps(
                {
                    "title": f"{t} mutada v{counter['n']}",
                    "description": f"Evolución de {t} con giro {counter['n']}.",
                    "advantages": ["Aa"],
                    "limitations": ["Xx"],
                    "mutation_description": "cambio",
                }
            )

        llm = AsyncMock()
        llm.generate.side_effect = gen
        llm.generate_structured.return_value = {
            "score": 0.7,
            "feedback": "ok",
            "estimated_complexity": 0.5,
        }

        domain = default_generic_domain()
        encoder = IdeaEncoder(embed_fn=deterministic_embed)
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
            encoder=encoder,
            repository=None,
        )

        result = await run_benchmark(
            challenge="Movilidad urbana sostenible e innovadora",
            domain=domain,
            generator=IdeaGeneratorAgent(llm),
            encoder=encoder,
            engine=engine,
            evaluator=evaluator,
            n_ideas=6,
            population=6,
            generations=1,
        )

        assert result.baseline.n_ideas > 0
        assert result.engine.n_ideas > 0
        assert 0.0 <= result.baseline.mean_pairwise_distance <= 1.0
        assert 0.0 <= result.engine.mean_pairwise_distance <= 1.0
        assert result.baseline.distinct_cells >= 1
        assert result.engine.distinct_cells >= 1
        assert result.engine.mean_fitness is not None
        assert result.verdict  # hay veredicto textual
        # serializable a JSON
        payload = json.dumps(result.to_dict())
        assert "verdict" in payload
