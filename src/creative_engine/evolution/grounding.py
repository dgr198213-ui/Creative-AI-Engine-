"""Memoria entre runs: grounding del generador con élites pasadas.

"Un chat olvida; este motor acumula". Antes de generar la población
inicial de un reto nuevo, el motor recupera élites de ejecuciones
anteriores y selecciona —por similitud de embeddings, cálculo local y
gratis— las relacionadas con el reto actual. Se inyectan en el prompt
como inspiración + repulsión: punto de partida mental, prohibido
repetirlas.

Coste en llamadas LLM: cero. Solo enriquece prompts que ya se hacen.
"""

from __future__ import annotations

import numpy as np
import structlog

from ..core.models import Idea

logger = structlog.get_logger(__name__)


def select_related_elites(
    challenge_vector: list[float],
    past_elites: list[Idea],
    k: int = 3,
    min_similarity: float = 0.25,
) -> list[Idea]:
    """Las k élites pasadas más afines al reto (similitud coseno ≥ mínimo).

    El mínimo evita inyectar ideas de retos sin relación alguna: mejor
    ninguna memoria que memoria irrelevante contaminando el prompt.
    """
    if not challenge_vector or not past_elites:
        return []

    q = np.asarray(challenge_vector, dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    q = q / q_norm

    candidates: list[tuple[float, Idea]] = []
    for idea in past_elites:
        if not idea.genome_vector:
            continue
        g = np.asarray(idea.genome_vector, dtype=np.float64)
        g_norm = np.linalg.norm(g)
        if g_norm == 0:
            continue
        similarity = float(np.dot(q, g / g_norm))
        if similarity >= min_similarity:
            candidates.append((similarity, idea))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [idea for _, idea in candidates[:k]]


def build_memory_hint(related: list[Idea]) -> str | None:
    """Construye el hint de inspiración + repulsión para el generador."""
    if not related:
        return None

    listado = "; ".join(
        f"«{idea.title}: {idea.description[:100]}»" for idea in related
    )
    return (
        f"MEMORIA DE EXPLORACIONES ANTERIORES — en retos parecidos ya "
        f"surgieron estos enfoques: {listado}. Úsalos como punto de partida "
        f"mental de lo que ya se sabe, pero está PROHIBIDO repetirlos o "
        f"parafrasearlos: genera ideas que exploren ángulos, mecanismos o "
        f"públicos que ahí no aparezcan."
    )
