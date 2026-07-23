"""Tests de la puerta de sorpresa adaptativa (evaluar solo lo sorprendente)."""

import json
from unittest.mock import AsyncMock

from creative_engine.evolution.surprise import SurpriseGate, min_distance_to_elites


class TestMinDistance:
    def test_empty_archive_max_surprise(self) -> None:
        assert min_distance_to_elites([1.0, 0.0], []) == 1.0

    def test_identical_vector_zero_distance(self) -> None:
        assert min_distance_to_elites([1.0, 0.0], [[1.0, 0.0]]) < 1e-9

    def test_orthogonal_half_distance(self) -> None:
        d = min_distance_to_elites([1.0, 0.0], [[0.0, 1.0]])
        assert abs(d - 0.5) < 1e-9

    def test_min_over_multiple_elites(self) -> None:
        d = min_distance_to_elites([1.0, 0.0], [[0.0, 1.0], [1.0, 0.1]])
        assert d < 0.01  # el segundo está casi encima


class TestGate:
    def test_near_duplicate_not_surprising(self) -> None:
        gate = SurpriseGate(threshold=0.10)
        assert not gate.is_surprising([1.0, 0.0], [[1.0, 0.05]])
        assert gate.evaluations_saved == 1

    def test_far_idea_surprising(self) -> None:
        gate = SurpriseGate(threshold=0.10)
        assert gate.is_surprising([1.0, 0.0], [[0.0, 1.0]])
        assert gate.evaluations_saved == 0

    def test_empty_archive_always_surprising(self) -> None:
        gate = SurpriseGate(threshold=0.10)
        assert gate.is_surprising([1.0, 0.0], [])

    def test_adapt_stagnation_lowers_threshold(self) -> None:
        gate = SurpriseGate(threshold=0.10, min_threshold=0.02, step=0.02)
        gate.adapt(stagnated=True)
        assert abs(gate.threshold - 0.08) < 1e-9

    def test_adapt_progress_raises_threshold(self) -> None:
        gate = SurpriseGate(threshold=0.10, max_threshold=0.20, step=0.02)
        gate.adapt(stagnated=False)
        assert abs(gate.threshold - 0.12) < 1e-9

    def test_adapt_respects_bounds(self) -> None:
        gate = SurpriseGate(threshold=0.03, min_threshold=0.02, max_threshold=0.20, step=0.02)
        gate.adapt(stagnated=True)
        gate.adapt(stagnated=True)
        assert gate.threshold == 0.02  # no baja del mínimo
        for _ in range(20):
            gate.adapt(stagnated=False)
        assert gate.threshold == 0.20  # no sube del máximo


class TestGateInEngine:
    async def test_duplicates_skip_llm_evaluation(
        self, deterministic_embed, monkeypatch
    ) -> None:
        """Ideas semánticamente idénticas a élites existentes NO deben gastar
        una evaluación LLM (la puerta las descarta antes)."""
        from creative_engine.agents.combined_evaluator import CombinedEvaluatorAgent
        from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
        from creative_engine.agents.generator import IdeaGeneratorAgent
        from creative_engine.core.config import get_settings, reset_settings
        from creative_engine.core.models import EvolutionRequest
        from creative_engine.evolution import encoders as enc
        from creative_engine.evolution.crossover import CrossoverEngine
        from creative_engine.evolution.encoders import IdeaEncoder
        from creative_engine.evolution.mutation import MutationEngine
        from creative_engine.evolution.qd_engine import QDEngine

        reset_settings()
        settings = get_settings()
        settings.evolution.surprise_gate_enabled = True
        settings.evolution.surprise_threshold = 0.10

        # El generador devuelve SIEMPRE los mismos 3 textos → tras la
        # población inicial, todo lo nuevo es un duplicado exacto.
        fixed = [
            {
                "title": "Idea fija A",
                "description": "Primera idea fija con contenido detallado.",
                "advantages": ["A"],
                "limitations": ["X"],
                "features": {"complexity_level": 0.5},
            },
            {
                "title": "Idea fija B",
                "description": "Segunda idea fija completamente distinta de la primera.",
                "advantages": ["B"],
                "limitations": ["Y"],
                "features": {"complexity_level": 0.5},
            },
            {
                "title": "Idea fija C",
                "description": "Tercera idea fija sobre otro tema totalmente diferente.",
                "advantages": ["C"],
                "limitations": ["Z"],
                "features": {"complexity_level": 0.5},
            },
        ]

        llm = AsyncMock()
        llm.generate.side_effect = lambda prompt, **kw: (
            json.dumps(fixed)
            if "array" in prompt
            else json.dumps(
                {
                    "title": "Idea fija A",
                    "description": "Primera idea fija con contenido detallado.",
                    "advantages": ["A"],
                    "limitations": ["X"],
                    "mutation_description": "sin cambio real",
                }
            )
        )
        llm.generate_structured.return_value = {
            "utility": 0.7,
            "utility_feedback": "ok",
            "feasibility": 0.6,
            "feasibility_feedback": "ok",
            "market_fit": 0.5,
            "market_feedback": "ok",
            "estimated_complexity": 0.5,
        }

        monkeypatch.setattr(
            enc.IdeaEncoder, "_embed", lambda self, text: deterministic_embed(text)
        )

        evaluator = EvaluatorOrchestrator(
            agents={"combined": CombinedEvaluatorAgent(llm)}
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
                generations=2,
            )
        )
        reset_settings()

        # Población inicial: 3 evaluaciones (únicas). Las generaciones
        # posteriores producen solo duplicados exactos → la puerta los
        # descarta y NO deben sumar más de 1-2 evaluaciones extra.
        evals = llm.generate_structured.await_count
        assert evals <= 5, f"esperadas ≤5 evaluaciones (3 iniciales + margen), hubo {evals}"
        assert engine._surprise_gate is not None
        assert engine._surprise_gate.evaluations_saved >= 2, (
            f"la puerta debió ahorrar ≥2 evaluaciones, ahorró "
            f"{engine._surprise_gate.evaluations_saved}"
        )

    async def test_gate_disabled_evaluates_everything(self, monkeypatch) -> None:
        from creative_engine.core.config import get_settings, reset_settings
        from creative_engine.evolution.qd_engine import QDEngine

        reset_settings()
        settings = get_settings()
        settings.evolution.surprise_gate_enabled = False

        engine = QDEngine(
            generator=AsyncMock(),
            evaluator=AsyncMock(),
            mutation=AsyncMock(),
            crossover=AsyncMock(),
            encoder=AsyncMock(),
            repository=None,
        )
        assert engine._surprise_gate is None
        reset_settings()
