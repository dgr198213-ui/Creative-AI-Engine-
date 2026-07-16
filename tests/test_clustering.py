"""Tests del agrupado automático de élites en familias."""

import pytest

from creative_engine.core.models import DomainName, EvaluationScores, Idea, IdeaStatus
from creative_engine.evolution.clustering import group_into_families


def make_elite(title: str, descriptor: list[float], utility: float = 0.7) -> Idea:
    idea = Idea(
        title=title,
        description=f"Descripción de {title} suficientemente larga para validar.",
        status=IdeaStatus.ELITE,
        domain=DomainName.GENERIC,
    )
    idea.evaluation = EvaluationScores(utility=utility, feasibility=0.6, market_fit=0.5)
    idea.behavior_descriptor = descriptor
    return idea


class TestGroupIntoFamilies:
    def test_empty_input(self) -> None:
        assert group_into_families([]) == []

    def test_ignores_ideas_without_descriptor(self) -> None:
        idea = make_elite("Sin descriptor", [])
        idea.behavior_descriptor = []
        assert group_into_families([idea]) == []

    def test_single_idea_one_family(self) -> None:
        families = group_into_families([make_elite("Única", [0.5, 0.5, 0.5])])
        assert len(families) == 1
        assert families[0].size == 1
        assert families[0].family_id == 0

    def test_two_close_ideas_merge(self) -> None:
        elites = [
            make_elite("Aaa", [0.10, 0.10, 0.10]),
            make_elite("Bbb", [0.12, 0.11, 0.09]),  # muy cerca de A
        ]
        families = group_into_families(elites, distance_threshold=0.25)
        assert len(families) == 1
        assert families[0].size == 2

    def test_two_far_ideas_separate(self) -> None:
        elites = [
            make_elite("Aaa", [0.05, 0.05, 0.05]),
            make_elite("Bbb", [0.95, 0.95, 0.95]),  # esquina opuesta
        ]
        families = group_into_families(elites, distance_threshold=0.25)
        assert len(families) == 2

    def test_automatic_family_count(self) -> None:
        """Tres clusters bien separados → exactamente tres familias."""
        elites = [
            make_elite("Fam-A1", [0.05, 0.05, 0.05]),
            make_elite("Fam-A2", [0.08, 0.06, 0.07]),
            make_elite("Fam-B1", [0.90, 0.10, 0.50]),
            make_elite("Fam-B2", [0.88, 0.12, 0.48]),
            make_elite("Fam-C1", [0.50, 0.90, 0.90]),
        ]
        families = group_into_families(elites, distance_threshold=0.25)
        assert len(families) == 3
        assert sum(f.size for f in families) == 5

    def test_representative_is_highest_fitness(self) -> None:
        elites = [
            make_elite("Débil", [0.10, 0.10, 0.10], utility=0.3),
            make_elite("Fuerte", [0.12, 0.11, 0.09], utility=0.95),
        ]
        families = group_into_families(elites, distance_threshold=0.25)
        assert len(families) == 1
        assert families[0].representative.title == "Fuerte"

    def test_families_sorted_by_representative_fitness(self) -> None:
        elites = [
            make_elite("Baja", [0.05, 0.05, 0.05], utility=0.2),
            make_elite("Alta", [0.95, 0.95, 0.95], utility=0.95),
        ]
        families = group_into_families(elites, distance_threshold=0.25)
        assert families[0].representative.title == "Alta"
        assert families[1].representative.title == "Baja"
        assert [f.family_id for f in families] == [0, 1]

    def test_higher_threshold_fewer_families(self) -> None:
        elites = [
            make_elite("Aaa", [0.10, 0.10, 0.10]),
            make_elite("Bbb", [0.40, 0.40, 0.40]),
            make_elite("Ccc", [0.70, 0.70, 0.70]),
        ]
        tight = group_into_families(elites, distance_threshold=0.15)
        loose = group_into_families(elites, distance_threshold=0.9)
        assert len(tight) > len(loose)
        assert len(loose) == 1

    def test_avg_fitness(self) -> None:
        elites = [
            make_elite("Aaa", [0.10, 0.10, 0.10], utility=1.0),
            make_elite("Bbb", [0.12, 0.11, 0.09], utility=0.0),
        ]
        families = group_into_families(elites, distance_threshold=0.25)
        assert families[0].size == 2
        # media de dos fitness (utility pesa 0.30 en DEFAULT_WEIGHTS)
        expected = (elites[0].fitness + elites[1].fitness) / 2
        assert families[0].avg_fitness == pytest.approx(expected)
