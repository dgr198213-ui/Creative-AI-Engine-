"""Tests de los modelos de datos Pydantic."""

import pytest

from creative_engine.core.models import (
    DEFAULT_WEIGHTS,
    BehaviorDimension,
    DomainConfig,
    EvaluationScores,
    Idea,
    IdeaFeatures,
    IdeaStatus,
    MutationType,
    ValueHypothesis,
)


class TestIdea:
    def test_create_minimal(self) -> None:
        idea = Idea(title="Test", description="Una idea de prueba con más de 10 caracteres.")
        assert idea.id.startswith("idea_")
        assert idea.status == IdeaStatus.DRAFT
        assert idea.generation == 0
        assert idea.fitness == 0.0  # sin evaluación

    def test_create_full(self) -> None:
        idea = Idea(
            title="Bicicleta Solar",
            description="Bicicleta urbana con panel solar integrado en el cuadro.",
            advantages=["Energía limpia", "Autonomía extendida"],
            limitations=["Coste elevado", "Peso mayor"],
            value_hypothesis=ValueHypothesis(
                target_user="Commuters urbanos",
                problem_solved="Dependencia de la red eléctrica",
                value_proposition="Movilidad autónoma y sostenible",
            ),
            features=IdeaFeatures(
                technologies=["IoT", "Solar Fotovoltaica"],
                complexity_level=0.7,
            ),
            generation=3,
            parent_ids=["idea_parent1"],
            mutation_type=MutationType.TECHNOLOGY,
        )
        assert len(idea.advantages) == 2
        assert idea.generation == 3
        assert idea.content_hash

    def test_fitness_quality_only(self) -> None:
        """El fitness ignora la novedad (peso 0 por defecto)."""
        idea = Idea(title="Test", description="Descripción suficientemente larga aquí.")
        idea.evaluation = EvaluationScores(
            novelty=0.0,
            utility=1.0,
            feasibility=1.0,
            impact=1.0,
            market_fit=1.0,
            sustainability=1.0,
            scalability=1.0,
        )
        assert idea.fitness == pytest.approx(1.0)

        # Cambiar solo la novedad NO cambia el fitness
        idea.evaluation.novelty = 1.0
        assert idea.fitness == pytest.approx(1.0)

    def test_content_hash_deterministic(self) -> None:
        idea1 = Idea(title="AAA", description="Misma descripción larga")
        idea2 = Idea(title="AAA", description="Misma descripción larga")
        assert idea1.content_hash == idea2.content_hash

    def test_content_hash_different(self) -> None:
        idea1 = Idea(title="AAA", description="Descripción uno larga")
        idea2 = Idea(title="BBB", description="Descripción dos larga")
        assert idea1.content_hash != idea2.content_hash


class TestEvaluationScores:
    def test_default_weights_sum_to_one(self) -> None:
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=0.01)

    def test_novelty_excluded_from_default_fitness(self) -> None:
        scores = EvaluationScores(novelty=1.0)
        assert scores.weighted_score == pytest.approx(0.0)

    def test_weighted_score(self) -> None:
        scores = EvaluationScores(utility=1.0)
        assert scores.weighted_score == pytest.approx(0.30)

    def test_as_vector(self) -> None:
        scores = EvaluationScores(novelty=0.5, utility=0.5)
        vec = scores.as_vector
        assert len(vec) == 8
        assert vec[0] == 0.5
        assert vec[1] == 0.5

    def test_custom_weights(self) -> None:
        scores = EvaluationScores(
            novelty=1.0,
            weights={
                "novelty": 1.0,
                "utility": 0.0,
                "feasibility": 0.0,
                "impact": 0.0,
                "market_fit": 0.0,
                "sustainability": 0.0,
                "scalability": 0.0,
            },
        )
        assert scores.weighted_score == pytest.approx(1.0)


class TestDomainConfig:
    def _dims(self, bins_a: int = 5, bins_b: int = 10) -> list[BehaviorDimension]:
        return [
            BehaviorDimension(name="a", bins=bins_a),
            BehaviorDimension(name="b", bins=bins_b),
        ]

    def test_grid_shape(self) -> None:
        config = DomainConfig(
            name="generic",
            display_name="Test",
            behavior_dimensions=self._dims(),
        )
        assert config.grid_shape == (5, 10)
        assert config.total_cells == 50

    def test_default_descriptor_mode_is_embedding(self) -> None:
        config = DomainConfig(
            name="generic",
            display_name="Test",
            behavior_dimensions=self._dims(),
        )
        assert config.descriptor_mode == "embedding"

    def test_invalid_weights_rejected(self) -> None:
        with pytest.raises(ValueError):
            DomainConfig(
                name="generic",
                display_name="Test",
                behavior_dimensions=self._dims(),
                evaluation_weights={"novelty": 0.5, "utility": 0.6},  # suma 1.1
            )
