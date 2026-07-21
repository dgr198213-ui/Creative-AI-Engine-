"""Tests de la memoria entre runs (grounding del generador)."""

import json
from unittest.mock import AsyncMock

from creative_engine.core.models import Idea
from creative_engine.evolution.grounding import build_memory_hint, select_related_elites


def _elite(title: str, genome: list[float]) -> Idea:
    idea = Idea(title=title, description=f"Descripción completa de {title} para el test.")
    idea.genome_vector = genome
    return idea


class TestSelectRelated:
    def test_picks_most_similar_above_floor(self) -> None:
        challenge_vec = [1.0, 0.0]
        past = [
            _elite("Muy afín", [0.95, 0.05]),
            _elite("Algo afín", [0.7, 0.7]),
            _elite("Nada que ver", [-1.0, 0.0]),
        ]
        related = select_related_elites(challenge_vec, past, k=2, min_similarity=0.25)
        titles = [i.title for i in related]
        assert titles == ["Muy afín", "Algo afín"]

    def test_floor_filters_unrelated(self) -> None:
        """Mejor ninguna memoria que memoria irrelevante."""
        related = select_related_elites(
            [1.0, 0.0], [_elite("Opuesta", [-1.0, 0.0])], k=3, min_similarity=0.25
        )
        assert related == []

    def test_empty_inputs(self) -> None:
        assert select_related_elites([], [_elite("Idea", [1.0])], k=3) == []
        assert select_related_elites([1.0], [], k=3) == []

    def test_ignores_elites_without_genome(self) -> None:
        no_genome = Idea(title="Sin genoma", description="Élite antigua sin vector aún.")
        related = select_related_elites([1.0, 0.0], [no_genome], k=3)
        assert related == []


class TestMemoryHint:
    def test_none_when_empty(self) -> None:
        assert build_memory_hint([]) is None

    def test_contains_titles_and_repulsion(self) -> None:
        hint = build_memory_hint([_elite("Bici solar plegable", [1.0])])
        assert "Bici solar plegable" in hint
        assert "PROHIBIDO" in hint  # inspiración + repulsión, no plantilla


class TestEngineIntegration:
    async def test_memory_hint_reaches_generator_prompt(
        self, deterministic_embed, monkeypatch
    ) -> None:
        """Con élites pasadas afines, el prompt de la población inicial debe
        llevar la memoria (título de la élite + instrucción de no repetir)."""
        from creative_engine.agents.combined_evaluator import CombinedEvaluatorAgent
        from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
        from creative_engine.agents.generator import IdeaGeneratorAgent
        from creative_engine.core.config import reset_settings
        from creative_engine.core.models import DomainName, EvolutionRequest
        from creative_engine.evolution import encoders as enc
        from creative_engine.evolution.crossover import CrossoverEngine
        from creative_engine.evolution.encoders import IdeaEncoder
        from creative_engine.evolution.mutation import MutationEngine
        from creative_engine.evolution.qd_engine import QDEngine

        reset_settings()
        challenge = "Movilidad urbana sostenible e innovadora"

        # Élite pasada MÁXIMAMENTE afín: su genoma = embedding del reto
        past_elite = _elite("Tranvía solar de barrio", deterministic_embed(challenge))

        repo = AsyncMock()
        repo.get_recent_elites.return_value = [past_elite]
        repo.store_idea = AsyncMock(side_effect=lambda i: i)

        def gen(prompt, **kw):
            items = [
                {
                    "title": f"Idea {i} distinta",
                    "description": f"Descripción de la idea {i} suficientemente larga.",
                    "advantages": ["A"],
                    "limitations": ["X"],
                    "features": {"complexity_level": 0.5},
                }
                for i in range(4)
            ]
            if "array" in prompt:
                return json.dumps(items)
            return json.dumps(
                {
                    "title": "Mutada distinta",
                    "description": "Evolución con giro totalmente diferente.",
                    "advantages": ["Aa"],
                    "limitations": ["Xx"],
                    "mutation_description": "c",
                }
            )

        llm = AsyncMock()
        llm.generate.side_effect = gen
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

        engine = QDEngine(
            generator=IdeaGeneratorAgent(llm),
            evaluator=EvaluatorOrchestrator(
                agents={"combined": CombinedEvaluatorAgent(llm)}
            ),
            mutation=MutationEngine(llm),
            crossover=CrossoverEngine(llm),
            encoder=IdeaEncoder(),
            repository=repo,
        )

        await engine.run_evolution(
            EvolutionRequest(
                challenge=challenge,
                domain=DomainName.GENERIC,
                population_size=4,
                generations=1,
            )
        )
        reset_settings()

        # La primera llamada de generación debe llevar la memoria
        first_prompt = llm.generate.await_args_list[0].kwargs.get(
            "prompt"
        ) or llm.generate.await_args_list[0].args[0]
        assert "Tranvía solar de barrio" in first_prompt
        assert "PROHIBIDO" in first_prompt
        repo.get_recent_elites.assert_awaited_once()

    async def test_no_repo_no_memory_no_crash(self, deterministic_embed, monkeypatch) -> None:
        """Sin persistencia, el motor genera igual, sin hint de memoria."""
        from creative_engine.agents.combined_evaluator import CombinedEvaluatorAgent
        from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
        from creative_engine.agents.generator import IdeaGeneratorAgent
        from creative_engine.core.config import reset_settings
        from creative_engine.core.models import DomainName, EvolutionRequest
        from creative_engine.evolution import encoders as enc
        from creative_engine.evolution.crossover import CrossoverEngine
        from creative_engine.evolution.encoders import IdeaEncoder
        from creative_engine.evolution.mutation import MutationEngine
        from creative_engine.evolution.qd_engine import QDEngine

        reset_settings()

        def gen(prompt, **kw):
            if "array" in prompt:
                return json.dumps(
                    [
                        {
                            "title": f"Idea {i}",
                            "description": f"Descripción {i} suficientemente larga aquí.",
                            "advantages": ["A"],
                            "limitations": ["X"],
                            "features": {"complexity_level": 0.5},
                        }
                        for i in range(4)
                    ]
                )
            return json.dumps(
                {
                    "title": "Mutada",
                    "description": "Otra evolución bien distinta y detallada.",
                    "advantages": ["Aa"],
                    "limitations": ["Xx"],
                    "mutation_description": "c",
                }
            )

        llm = AsyncMock()
        llm.generate.side_effect = gen
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

        engine = QDEngine(
            generator=IdeaGeneratorAgent(llm),
            evaluator=EvaluatorOrchestrator(
                agents={"combined": CombinedEvaluatorAgent(llm)}
            ),
            mutation=MutationEngine(llm),
            crossover=CrossoverEngine(llm),
            encoder=IdeaEncoder(),
            repository=None,
        )
        state = await engine.run_evolution(
            EvolutionRequest(
                challenge="Movilidad urbana sostenible e innovadora",
                domain=DomainName.GENERIC,
                population_size=4,
                generations=1,
            )
        )
        reset_settings()
        assert len(state.archive) >= 2
        first_prompt = llm.generate.await_args_list[0].kwargs.get(
            "prompt"
        ) or llm.generate.await_args_list[0].args[0]
        assert "MEMORIA DE EXPLORACIONES" not in first_prompt


class TestRecentElitesQuery:
    """La query de memoria entre runs debe castear el parámetro nullable.

    asyncpg no puede inferir el tipo de un parámetro usado en 'X IS NULL'
    sin cast explícito → AmbiguousParameterError (visto en producción).
    El CAST(... AS TEXT) es obligatorio; este test evita reintroducir el bug.
    """

    def test_query_casts_nullable_param(self) -> None:
        import inspect

        from creative_engine.memory import repository

        source = inspect.getsource(repository.IdeaRepository.get_recent_elites)
        assert "CAST(:exclude_run_id AS TEXT) IS NULL" in source, (
            "el parámetro nullable debe castearse o asyncpg da "
            "AmbiguousParameterError"
        )
