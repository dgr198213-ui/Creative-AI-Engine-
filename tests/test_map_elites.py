"""Tests del archivo MAP-Elites."""

import pytest

from creative_engine.core.exceptions import BehaviorDescriptorError, PopulationEmptyError
from creative_engine.core.models import DomainName, EvaluationScores, Idea, IdeaStatus
from creative_engine.evolution.map_elites import MAPElitesArchive


@pytest.fixture
def archive() -> MAPElitesArchive:
    return MAPElitesArchive(grid_shape=(10, 10), dimension_names=["d1", "d2"])


def make_idea(
    utility: float,
    feasibility: float,
    descriptor: list[float],
    title: str = "Idea de prueba",
) -> Idea:
    idea = Idea(
        title=title,
        description="Una idea de prueba para verificar el archivo MAP-Elites",
        status=IdeaStatus.EVALUATED,
        domain=DomainName.GENERIC,
    )
    idea.evaluation = EvaluationScores(utility=utility, feasibility=feasibility)
    idea.behavior_descriptor = descriptor
    return idea


class TestMAPElitesArchive:
    def test_initialization(self, archive: MAPElitesArchive) -> None:
        assert archive.total_cells == 100
        assert archive.coverage == 0.0
        assert len(archive.occupied_cells) == 0

    def test_insert_first_idea(self, archive: MAPElitesArchive) -> None:
        idea = make_idea(0.8, 0.6, [0.8, 0.6])
        assert archive.try_insert(idea) is True
        assert archive.coverage == pytest.approx(0.01)
        assert archive.best_fitness == pytest.approx(idea.fitness)

    def test_insert_better_replaces_same_cell(self, archive: MAPElitesArchive) -> None:
        # 0.80*9=7.2→bin 7 y 0.82*9=7.38→bin 7: MISMA celda
        base = make_idea(0.5, 0.5, [0.80, 0.60])
        archive.try_insert(base)

        better = make_idea(0.9, 0.9, [0.82, 0.60], title="Mejor idea")
        assert archive.try_insert(better) is True
        assert archive.coverage == pytest.approx(0.01)  # sigue siendo 1 celda
        assert archive.best_fitness == pytest.approx(better.fitness)

    def test_insert_worse_same_cell_rejected(self, archive: MAPElitesArchive) -> None:
        base = make_idea(0.9, 0.9, [0.80, 0.60])
        archive.try_insert(base)

        worse = make_idea(0.1, 0.1, [0.82, 0.60], title="Peor idea")
        assert archive.try_insert(worse) is False
        assert archive.coverage == pytest.approx(0.01)

    def test_insert_worse_different_cell_accepted(self, archive: MAPElitesArchive) -> None:
        """Peor fitness global pero celda distinta → se acepta (es la gracia de QD)."""
        base = make_idea(0.9, 0.9, [0.80, 0.60])
        archive.try_insert(base)

        different = make_idea(0.1, 0.1, [0.20, 0.10], title="Distinta región")
        assert archive.try_insert(different) is True
        assert archive.coverage == pytest.approx(0.02)

    def test_discretize_corners(self, archive: MAPElitesArchive) -> None:
        assert archive.discretize([0.0, 0.0]) == (0, 0)
        assert archive.discretize([1.0, 1.0]) == (9, 9)
        assert archive.discretize([0.5, 0.5]) == (4, 4)  # int(0.5*9)=4

    def test_discretize_invalid_dimension(self, archive: MAPElitesArchive) -> None:
        with pytest.raises(BehaviorDescriptorError):
            archive.discretize([0.5])

    def test_discretize_out_of_range(self, archive: MAPElitesArchive) -> None:
        with pytest.raises(BehaviorDescriptorError):
            archive.discretize([1.5, 0.5])
        with pytest.raises(BehaviorDescriptorError):
            archive.discretize([0.5, -0.1])

    def test_select_for_mutation(self, archive: MAPElitesArchive) -> None:
        for i in range(5):
            idea = make_idea(0.2 + i * 0.1, 0.5, [0.1 + i * 0.2, 0.5], title=f"Idea {i}")
            archive.try_insert(idea)

        selected = archive.select_for_mutation(n=3)
        assert len(selected) == 3
        assert len({i.id for i in selected}) == 3  # únicas

    def test_select_for_mutation_empty_raises(self, archive: MAPElitesArchive) -> None:
        with pytest.raises(PopulationEmptyError):
            archive.select_for_mutation(n=2)

    def test_select_pair_for_crossover(self, archive: MAPElitesArchive) -> None:
        assert archive.select_pair_for_crossover() is None

        archive.try_insert(make_idea(0.5, 0.5, [0.1, 0.1]))
        assert archive.select_pair_for_crossover() is None  # necesita 2

        archive.try_insert(make_idea(0.6, 0.6, [0.9, 0.9]))
        pair = archive.select_pair_for_crossover()
        assert pair is not None
        assert pair[0].id != pair[1].id

    def test_qd_score_accumulates(self, archive: MAPElitesArchive) -> None:
        i1 = make_idea(0.5, 0.5, [0.1, 0.1])
        i2 = make_idea(0.8, 0.8, [0.9, 0.9])

        archive.try_insert(i1)
        assert archive.qd_score == pytest.approx(i1.fitness)

        archive.try_insert(i2)
        assert archive.qd_score == pytest.approx(i1.fitness + i2.fitness)

    def test_qd_score_after_replacement(self, archive: MAPElitesArchive) -> None:
        old = make_idea(0.2, 0.2, [0.80, 0.60])
        new = make_idea(0.9, 0.9, [0.82, 0.60])
        archive.try_insert(old)
        archive.try_insert(new)
        assert archive.qd_score == pytest.approx(new.fitness)

    def test_elite_genomes(self, archive: MAPElitesArchive) -> None:
        idea = make_idea(0.5, 0.5, [0.5, 0.5])
        idea.genome_vector = [0.1, 0.2, 0.3]
        archive.try_insert(idea)
        genomes = archive.elite_genomes()
        assert genomes == [[0.1, 0.2, 0.3]]

    def test_to_numpy_arrays_empty(self, archive: MAPElitesArchive) -> None:
        g, d, f = archive.to_numpy_arrays()
        assert g.shape == (0, 1)
        assert d.shape == (0, 2)
        assert f.shape == (0,)

    def test_to_numpy_arrays_with_data(self, archive: MAPElitesArchive) -> None:
        idea = make_idea(0.8, 0.6, [0.8, 0.6])
        idea.genome_vector = [0.1, 0.2, 0.3]
        archive.try_insert(idea)

        g, d, f = archive.to_numpy_arrays()
        assert g.shape == (1, 3)
        assert d.shape == (1, 2)
        assert f[0] == pytest.approx(idea.fitness)


class TestCuriositySelection:
    """Selección por curiosidad: prioriza regiones poco exploradas."""

    def _archive_with_cluster_and_outlier(self):
        """Archivo con un grupo denso y una élite aislada."""
        from creative_engine.core.models import EvaluationScores, Idea

        def _make(title: str, descriptor: list[float]) -> Idea:
            idea = Idea(title=title, description=f"Descripción de {title} con detalle.")
            idea.behavior_descriptor = descriptor
            idea.genome_vector = [1.0, 0.0]
            idea.evaluation = EvaluationScores(
                utility=0.5, feasibility=0.5, market_fit=0.5
            )
            return idea

        archive = MAPElitesArchive(grid_shape=(10, 10, 8))
        # clúster denso: 4 élites en celdas adyacentes
        cluster = [
            [0.05, 0.05, 0.05],
            [0.15, 0.05, 0.05],
            [0.05, 0.15, 0.05],
            [0.15, 0.15, 0.05],
        ]
        for i, d in enumerate(cluster):
            archive.try_insert(_make(f"Idea clúster {i}", d))
        # élite aislada en la esquina opuesta
        archive.try_insert(_make("Idea aislada", [0.95, 0.95, 0.95]))
        return archive

    def test_curious_prefers_isolated_elites(self) -> None:
        """Con fitness igual, la élite aislada debe seleccionarse mucho más
        a menudo que su cuota uniforme (1/5)."""
        import numpy as np

        archive = self._archive_with_cluster_and_outlier()
        rng = np.random.default_rng(42)
        picks = 0
        trials = 400
        for _ in range(trials):
            selected = archive.select_curious(1, rng, fitness_weight=0.0)
            if selected[0].title == "Idea aislada":
                picks += 1
        # cuota uniforme sería ~20%; con curiosidad pura debe superar 30%
        assert picks / trials > 0.30, f"aislada elegida solo {picks}/{trials}"

    def test_curious_respects_fitness_weight(self) -> None:
        """Con fitness_weight=1.0 se comporta como selección por fitness."""
        import numpy as np

        archive = self._archive_with_cluster_and_outlier()
        rng = np.random.default_rng(7)
        selected = archive.select_curious(3, rng, fitness_weight=1.0)
        assert len(selected) == 3

    def test_curious_empty_archive_raises(self) -> None:
        import pytest

        from creative_engine.core.exceptions import PopulationEmptyError

        archive = MAPElitesArchive(grid_shape=(10, 10, 8))
        with pytest.raises(PopulationEmptyError):
            archive.select_curious(1)
