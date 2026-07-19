"""Puerta de sorpresa adaptativa: evaluar con LLM solo lo que sorprende.

Del informe de investigación (TurboEvolve, y la idea 'EcoThreshold' que el
propio motor propuso): las evaluaciones LLM son el recurso caro; los
embeddings son locales y gratis. Antes de gastar una evaluación en una
idea nueva, medimos su distancia semántica a las élites existentes:

- Si está demasiado cerca de algo ya explorado (poca sorpresa), se
  descarta SIN evaluar — no habría desplazado a nadie y ahorramos la llamada.
- Si está lejos (sorprendente), pasa a evaluación normal.

El umbral se adapta solo, estilo planificador online de TurboEvolve:

- Generación SIN celdas nuevas (estancamiento) → bajar el umbral: dejar
  pasar más candidatos, explorar más.
- Generación con progreso → subir el umbral: ahorrar más presupuesto.
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def min_distance_to_elites(
    genome: list[float], elite_genomes: list[list[float]]
) -> float:
    """Distancia coseno mínima ((1-sim)/2, rango [0,1]) a las élites."""
    if not genome or not elite_genomes:
        return 1.0  # archivo vacío o sin genoma comparable: máxima sorpresa

    g = np.asarray(genome, dtype=np.float64)
    g_norm = np.linalg.norm(g)
    if g_norm == 0:
        return 1.0
    g = g / g_norm

    elites = np.asarray(elite_genomes, dtype=np.float64)
    norms = np.linalg.norm(elites, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    elites = elites / norms

    sims = elites @ g
    dists = (1.0 - sims) / 2.0
    return float(dists.min())


class SurpriseGate:
    """Decide si una idea merece una evaluación LLM, con umbral adaptativo."""

    def __init__(
        self,
        threshold: float = 0.10,
        min_threshold: float = 0.02,
        max_threshold: float = 0.20,
        step: float = 0.02,
    ) -> None:
        self.threshold = threshold
        self._min = min_threshold
        self._max = max_threshold
        self._step = step
        self.evaluations_saved = 0
        self._log = logger.bind(component="SurpriseGate")

    def is_surprising(
        self, genome: list[float], elite_genomes: list[list[float]]
    ) -> bool:
        """True si la idea está lo bastante lejos de lo ya explorado."""
        distance = min_distance_to_elites(genome, elite_genomes)
        surprising = distance >= self.threshold
        if not surprising:
            self.evaluations_saved += 1
        return surprising

    def adapt(self, stagnated: bool) -> None:
        """Ajusta el umbral según el progreso de la última generación."""
        old = self.threshold
        if stagnated:
            # Sin celdas nuevas: abrir la puerta, explorar más.
            self.threshold = max(self._min, self.threshold - self._step)
        else:
            # Progresando: cerrar un poco, ahorrar presupuesto.
            self.threshold = min(self._max, self.threshold + self._step)
        if self.threshold != old:
            self._log.debug(
                "surprise_threshold_adapted",
                old=round(old, 3),
                new=round(self.threshold, 3),
                stagnated=stagnated,
            )
