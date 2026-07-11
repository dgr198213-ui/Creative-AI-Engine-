"""Tests de codificadores: proyección determinista y novedad objetiva."""

import pytest

from creative_engine.core.config import default_generic_domain
from creative_engine.core.models import Idea
from creative_engine.evolution.encoders import (
    IdeaEncoder,
    objective_novelty,
    project_to_descriptor,
)


class TestProjection:
    def test_descriptor_in_range(self, deterministic_embed) -> None:
        genome = deterministic_embed("una idea cualquiera sobre movilidad")
        descriptor = project_to_descriptor(genome, 3)
        assert len(descriptor) == 3
        assert all(0.0 <= v <= 1.0 for v in descriptor)

    def test_projection_is_deterministic(self, deterministic_embed) -> None:
        genome = deterministic_embed("bicicleta solar plegable")
        d1 = project_to_descriptor(genome, 3)
        d2 = project_to_descriptor(genome, 3)
        assert d1 == d2

    def test_different_content_different_descriptor(self, deterministic_embed) -> None:
        d1 = project_to_descriptor(deterministic_embed("bicicleta solar urbana"), 3)
        d2 = project_to_descriptor(deterministic_embed("campaña de marketing viral"), 3)
        assert d1 != d2


class TestObjectiveNovelty:
    def test_empty_archive_is_maximally_novel(self, deterministic_embed) -> None:
        genome = deterministic_embed("idea inicial")
        assert objective_novelty(genome, []) == 1.0

    def test_identical_idea_has_zero_novelty(self, deterministic_embed) -> None:
        genome = deterministic_embed("idea repetida")
        novelty = objective_novelty(genome, [genome])
        assert novelty == pytest.approx(0.0, abs=1e-6)

    def test_different_idea_has_positive_novelty(self, deterministic_embed) -> None:
        g1 = deterministic_embed("bicicleta eléctrica modular")
        g2 = deterministic_embed("plataforma de streaming de ópera")
        novelty = objective_novelty(g1, [g2])
        assert 0.0 < novelty <= 1.0

    def test_novelty_in_valid_range(self, deterministic_embed) -> None:
        query = deterministic_embed("query")
        archive = [deterministic_embed(f"idea {i}") for i in range(10)]
        novelty = objective_novelty(query, archive, k=3)
        assert 0.0 <= novelty <= 1.0


class TestIdeaEncoder:
    def test_encode_idea_with_injected_embedder(self, deterministic_embed) -> None:
        encoder = IdeaEncoder(embed_fn=deterministic_embed)
        domain = default_generic_domain()

        idea = Idea(
            title="Bicicleta Solar",
            description="Bicicleta urbana con panel solar integrado en el cuadro.",
            advantages=["Energía limpia"],
        )
        encoded = encoder.encode_idea(idea, domain)

        assert len(encoded.genome_vector) == 384
        assert len(encoded.behavior_descriptor) == len(domain.behavior_dimensions)
        assert all(0.0 <= v <= 1.0 for v in encoded.behavior_descriptor)

    def test_semantic_diversity_maps_to_different_cells(self, deterministic_embed) -> None:
        """Ideas de contenido distinto deben tender a celdas distintas."""
        from creative_engine.evolution.map_elites import MAPElitesArchive

        encoder = IdeaEncoder(embed_fn=deterministic_embed)
        domain = default_generic_domain()
        archive = MAPElitesArchive(grid_shape=domain.grid_shape)

        contents = [
            ("Bicicleta solar", "Bicicleta urbana con paneles solares integrados en ruedas."),
            ("App de trueque", "Aplicación móvil para intercambiar objetos entre vecinos."),
            ("Dron mensajero", "Dron autónomo para entregas médicas en zonas rurales."),
            ("Huerto vertical", "Sistema hidropónico modular para fachadas de edificios."),
        ]

        cells = set()
        for title, desc in contents:
            idea = Idea(title=title, description=desc)
            encoder.encode_idea(idea, domain)
            cells.add(archive.discretize(idea.behavior_descriptor))

        # Con 4 contenidos muy distintos esperamos al menos 3 celdas diferentes
        assert len(cells) >= 3
