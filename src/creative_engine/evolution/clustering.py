"""Agrupado automático de ideas élite en "familias" de enfoques.

Las élites de un run ocupan celdas distintas del espacio de comportamiento,
pero muchas caen cerca unas de otras: representan variaciones del mismo
enfoque. Este módulo las agrupa en familias para presentar el abanico de
resultados organizado ("5 enfoques, la mejor idea de cada uno") en lugar
de una lista plana.

Diseño:
- Clustering aglomerativo de enlace simple (single-linkage) sobre la
  distancia euclídea del `behavior_descriptor` (ya en [0,1]^N).
- El NÚMERO de familias es automático: emerge de un umbral de distancia,
  no de un `k` impuesto. Ideas más separadas → más familias.
- Sin dependencias nuevas: solo numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.models import Idea

# Umbral por defecto de distancia en el espacio de comportamiento.
# El descriptor vive en [0,1]^N; la distancia máxima posible es sqrt(N).
# 0.25 agrupa ideas razonablemente próximas sin fusionar enfoques distintos.
DEFAULT_DISTANCE_THRESHOLD = 0.25


@dataclass
class IdeaFamily:
    """Un grupo de ideas élite que comparten un enfoque similar."""

    family_id: int
    representative: Idea  # la de mayor fitness del grupo
    members: list[Idea] = field(default_factory=list)
    report: str | None = None  # informe opcional del WriterAgent

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def avg_fitness(self) -> float:
        if not self.members:
            return 0.0
        return float(np.mean([m.fitness for m in self.members]))


def _pairwise_distances(descriptors: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclídeas entre descriptores."""
    diff = descriptors[:, None, :] - descriptors[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _single_linkage_labels(distances: np.ndarray, threshold: float) -> list[int]:
    """Clustering aglomerativo de enlace simple mediante union-find.

    Une cualquier par de puntos cuya distancia sea <= threshold; los
    clusters resultantes son las componentes conexas del grafo de umbral.
    """
    n = distances.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= threshold:
                union(i, j)

    # Reetiquetar raíces a 0..k-1 de forma estable
    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])
    return labels


def group_into_families(
    elites: list[Idea],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[IdeaFamily]:
    """Agrupa ideas élite en familias automáticas por proximidad semántica.

    Args:
        elites: Ideas élite con `behavior_descriptor` no vacío.
        distance_threshold: Distancia máxima para considerar dos ideas
            del mismo enfoque. Mayor umbral → menos familias, más grandes.

    Returns:
        Familias ordenadas por fitness del representante (desc).
        Ideas sin descriptor válido se ignoran.
    """
    valid = [e for e in elites if e.behavior_descriptor]
    if not valid:
        return []

    if len(valid) == 1:
        return [IdeaFamily(family_id=0, representative=valid[0], members=[valid[0]])]

    descriptors = np.array([e.behavior_descriptor for e in valid], dtype=np.float64)
    distances = _pairwise_distances(descriptors)
    labels = _single_linkage_labels(distances, distance_threshold)

    grouped: dict[int, list[Idea]] = {}
    for idea, label in zip(valid, labels, strict=True):
        grouped.setdefault(label, []).append(idea)

    families: list[IdeaFamily] = []
    for members in grouped.values():
        members_sorted = sorted(members, key=lambda i: i.fitness, reverse=True)
        families.append(
            IdeaFamily(
                family_id=0,  # se reasigna tras ordenar
                representative=members_sorted[0],
                members=members_sorted,
            )
        )

    # Ordenar familias por fitness del representante y reasignar IDs estables
    families.sort(key=lambda f: f.representative.fitness, reverse=True)
    for idx, family in enumerate(families):
        family.family_id = idx

    return families
