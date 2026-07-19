"""Implementación de MAP-Elites en Python puro.

MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) mantiene
un grid N-dimensional del espacio de comportamiento donde cada celda
conserva el individuo con mayor fitness observado en esa región.

Referencia: Mouret & Clune, "Illuminating search spaces by mapping
elites", arXiv:1504.04909.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from ..core.exceptions import BehaviorDescriptorError, PopulationEmptyError
from ..core.models import Idea, MAPElitesCell

logger = structlog.get_logger(__name__)


@dataclass
class GridCell:
    """Una celda del grid MAP-Elites."""

    index: tuple[int, ...]
    fitness: float = -np.inf
    content: Idea | None = None
    occupancy_count: int = 0


class MAPElitesArchive:
    """Archivo MAP-Elites con grid N-dimensional."""

    def __init__(
        self,
        grid_shape: tuple[int, ...],
        dimension_names: list[str] | None = None,
    ) -> None:
        if not grid_shape or len(grid_shape) < 2:
            raise ValueError("grid_shape debe tener al menos 2 dimensiones")

        self.grid_shape = grid_shape
        self.dimension_names = dimension_names or [f"dim_{i}" for i in range(len(grid_shape))]
        self.total_cells = int(np.prod(grid_shape))

        self._cells: dict[tuple[int, ...], GridCell] = {}
        for idx in np.ndindex(*self.grid_shape):
            self._cells[tuple(idx)] = GridCell(index=tuple(idx))

        self._best_fitness: float = -np.inf
        self._total_evaluations: int = 0

        self._log = logger.bind(grid_shape=grid_shape, total_cells=self.total_cells)

    # ── Métricas ────────────────────────────────────────────────────

    @property
    def coverage(self) -> float:
        """Fracción de celdas ocupadas [0, 1]."""
        occupied = sum(1 for c in self._cells.values() if c.content is not None)
        return occupied / self.total_cells

    @property
    def qd_score(self) -> float:
        """Suma de fitness de todas las celdas ocupadas."""
        return float(
            sum(c.fitness for c in self._cells.values() if c.content is not None)
        )

    @property
    def best_fitness(self) -> float:
        return self._best_fitness if self._best_fitness != -np.inf else 0.0

    @property
    def occupied_cells(self) -> list[MAPElitesCell]:
        return [
            MAPElitesCell(cell_index=cell.index, elite=cell.content, fitness=cell.fitness)
            for cell in self._cells.values()
            if cell.content is not None
        ]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_cells": self.total_cells,
            "occupied": sum(1 for c in self._cells.values() if c.content is not None),
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "best_fitness": self.best_fitness,
            "total_evaluations": self._total_evaluations,
        }

    def elite_genomes(self) -> list[list[float]]:
        """Genomas (embeddings) de todas las élites — para novedad objetiva."""
        return [
            c.content.genome_vector
            for c in self._cells.values()
            if c.content is not None and c.content.genome_vector
        ]

    # ── Operaciones ─────────────────────────────────────────────────

    def discretize(self, behavior_descriptor: list[float]) -> tuple[int, ...]:
        """Convierte un descriptor continuo [0,1]^N en índice de celda."""
        bd = np.asarray(behavior_descriptor, dtype=np.float64)

        if bd.shape[0] != len(self.grid_shape):
            raise BehaviorDescriptorError(
                f"Descriptor tiene {bd.shape[0]} dimensiones, "
                f"se esperaban {len(self.grid_shape)}"
            )

        if np.any(bd < 0.0) or np.any(bd > 1.0):
            raise BehaviorDescriptorError(
                f"Valores del descriptor fuera de [0,1]: "
                f"min={bd.min():.4f}, max={bd.max():.4f}"
            )

        indices = (bd * (np.array(self.grid_shape) - 1)).astype(int)
        indices = np.clip(indices, 0, np.array(self.grid_shape) - 1)
        return tuple(int(i) for i in indices)

    def try_insert(self, idea: Idea) -> bool:
        """Inserta la idea si su celda está vacía o mejora al elite actual."""
        self._total_evaluations += 1

        cell_idx = self.discretize(idea.behavior_descriptor)
        cell = self._cells[cell_idx]
        cell.occupancy_count += 1

        fitness = idea.fitness

        if cell.content is None or fitness > cell.fitness:
            is_new = cell.content is None
            cell.fitness = fitness
            cell.content = idea
            self._best_fitness = max(self._best_fitness, fitness)

            if is_new:
                self._log.debug(
                    "new_cell_occupied",
                    cell=cell_idx,
                    fitness=round(fitness, 4),
                    coverage=round(self.coverage, 4),
                )
            return True

        return False

    def get_elite(self, cell_index: tuple[int, ...]) -> Idea | None:
        cell = self._cells.get(cell_index)
        return cell.content if cell else None

    def get_random_occupied_cell(
        self, rng: np.random.Generator | None = None
    ) -> Idea | None:
        occupied = self.occupied_cells
        if not occupied:
            return None
        rng = rng or np.random.default_rng()
        return occupied[int(rng.integers(len(occupied)))].elite

    def select_for_mutation(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> list[Idea]:
        """Selecciona hasta n élites, con probabilidad proporcional al fitness."""
        occupied = self.occupied_cells
        if not occupied:
            raise PopulationEmptyError("No hay élites para mutar")

        rng = rng or np.random.default_rng()
        fitnesses = np.array([c.fitness for c in occupied])
        fitnesses_shifted = fitnesses - fitnesses.min() + 1e-8
        probs = fitnesses_shifted / fitnesses_shifted.sum()

        size = min(n, len(occupied))
        indices = rng.choice(len(occupied), size=size, p=probs, replace=False)
        return [occupied[int(i)].elite for i in indices]

    def _local_density(self, cell_index: tuple[int, ...]) -> int:
        """Nº de celdas OCUPADAS en el vecindario inmediato (Chebyshev r=1)."""
        density = 0
        for cell in self.occupied_cells:
            other_index = cell.cell_index
            if other_index == cell_index:
                continue
            if all(abs(a - b) <= 1 for a, b in zip(cell_index, other_index, strict=False)):
                density += 1
        return density

    def select_curious(
        self,
        n: int,
        rng: np.random.Generator | None = None,
        fitness_weight: float = 0.5,
    ) -> list[Idea]:
        """Selección por curiosidad: prioriza regiones POCO exploradas.

        Inspirado en el `sample_underexplored` de los híbridos
        TurboEvolve + MAP-Elites: los padres se eligen combinando fitness
        con la inversa de la densidad local — las élites en zonas
        despobladas del espacio de comportamiento tienen más probabilidad
        de reproducirse, empujando la cobertura hacia lo inexplorado.
        """
        occupied = self.occupied_cells
        if not occupied:
            raise PopulationEmptyError("No hay élites para mutar")

        rng = rng or np.random.default_rng()

        fitnesses = np.array([c.fitness for c in occupied])
        f_norm = (fitnesses - fitnesses.min() + 1e-8)
        f_norm = f_norm / f_norm.sum()

        densities = np.array(
            [self._local_density(c.cell_index) for c in occupied], dtype=np.float64
        )
        curiosity = 1.0 / (1.0 + densities)  # menos vecinos → más curiosidad
        c_norm = curiosity / curiosity.sum()

        probs = fitness_weight * f_norm + (1.0 - fitness_weight) * c_norm
        probs = probs / probs.sum()

        size = min(n, len(occupied))
        indices = rng.choice(len(occupied), size=size, p=probs, replace=False)
        return [occupied[int(i)].elite for i in indices]

    def select_pair_for_crossover(
        self,
        rng: np.random.Generator | None = None,
    ) -> tuple[Idea, Idea] | None:
        """Par de élites de celdas diferentes para cruce."""
        occupied = self.occupied_cells
        if len(occupied) < 2:
            return None
        rng = rng or np.random.default_rng()
        pair = rng.choice(len(occupied), size=2, replace=False)
        return occupied[int(pair[0])].elite, occupied[int(pair[1])].elite

    def to_numpy_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exporta (genotypes, descriptors, fitnesses) para análisis externo."""
        occupied = self.occupied_cells
        n_dims = len(self.grid_shape)
        if not occupied:
            return np.empty((0, 1)), np.empty((0, n_dims)), np.empty(0)

        n = len(occupied)
        genome_dim = len(occupied[0].elite.genome_vector) or 1

        genotypes = np.zeros((n, genome_dim))
        descriptors = np.zeros((n, n_dims))
        fitnesses = np.zeros(n)

        for i, cell in enumerate(occupied):
            gv = cell.elite.genome_vector
            genotypes[i, : len(gv)] = gv
            descriptors[i] = cell.elite.behavior_descriptor
            fitnesses[i] = cell.fitness

        return genotypes, descriptors, fitnesses
