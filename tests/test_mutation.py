"""Tests del motor de mutación y del parseo JSON."""

import pytest

from creative_engine.core.exceptions import LLMResponseParseError
from creative_engine.core.models import Idea, IdeaStatus, MutationType
from creative_engine.evolution.mutation import MutationEngine, parse_llm_json


class TestParseLLMJson:
    def test_parses_plain_json(self) -> None:
        data = parse_llm_json('{"title": "X", "description": "Y"}')
        assert data["title"] == "X"

    def test_parses_markdown_fenced_json(self) -> None:
        raw = '```json\n{"title": "X", "description": "Y"}\n```'
        data = parse_llm_json(raw)
        assert data["description"] == "Y"

    def test_parses_json_with_preamble(self) -> None:
        raw = 'Aquí tienes la idea:\n{"title": "X", "description": "Y"}'
        data = parse_llm_json(raw)
        assert data["title"] == "X"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(LLMResponseParseError):
            parse_llm_json("esto no es json")


class TestMutationEngine:
    @pytest.fixture
    def parent(self) -> Idea:
        return Idea(
            title="Bicicleta Modular",
            description="Sistema de bicicleta urbana con componentes intercambiables.",
            advantages=["Adaptable"],
            generation=2,
            run_id="run_test",
        )

    async def test_mutate_builds_child(self, mock_llm_provider, parent: Idea) -> None:
        engine = MutationEngine(mock_llm_provider)
        child = await engine.mutate(parent, MutationType.FUNCTIONALITY)

        assert child.id != parent.id
        assert child.parent_ids == [parent.id]
        assert child.generation == parent.generation + 1
        assert child.status == IdeaStatus.MUTATED
        assert child.mutation_type == MutationType.FUNCTIONALITY
        assert child.run_id == parent.run_id
        assert child.title == "Idea Mutada de Prueba"

    async def test_batch_mutate_skips_failures(self, mock_llm_provider, parent: Idea) -> None:
        engine = MutationEngine(mock_llm_provider)

        # Segunda llamada falla, primera y tercera funcionan
        good = mock_llm_provider.generate.return_value
        mock_llm_provider.generate.side_effect = [good, Exception("API Down"), good]

        # La excepción genérica no es LLMError → gather la propagaría; usamos
        # LLMError para simular el fallo controlado
        from creative_engine.core.exceptions import LLMError

        mock_llm_provider.generate.side_effect = [good, LLMError("API Down"), good]

        parents = [parent, parent.model_copy(), parent.model_copy()]
        children = await engine.batch_mutate(parents)
        assert len(children) == 2
