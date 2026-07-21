"""Operadores adaptativos: el motor aprende qué operador le funciona.

La versión viable de la "meta-cognición" que el propio motor propuso
(Learned QD aterrizado a free tier, y también recomendación de la
auditoría externa): en vez de un reparto fijo entre mutación, cruce e
inyección fresca, el motor observa qué operador está produciendo élites
en ESTE reto y redistribuye el presupuesto de la siguiente generación
hacia lo que funciona.

Coste en llamadas LLM: cero — solo cambia cómo se reparten las que ya
se hacen. Con suelo de exploración: ningún operador se abandona del
todo (podría volver a ser útil más adelante en el run).
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

OPERATORS = ("mutation", "crossover", "fresh")

# Peso de la historia previa al decaer cada generación (recencia)
_DECAY = 0.5
# Suelo: proporción mínima del presupuesto que conserva cada operador
_MIN_SHARE = 0.15


class OperatorScheduler:
    """Redistribuye el presupuesto generacional según el éxito reciente."""

    def __init__(self, base_rates: dict[str, float]) -> None:
        # Tasas base del dominio (mutation_rate, crossover_rate, random_injection_rate)
        self._base = {op: max(0.0, base_rates.get(op, 0.0)) for op in OPERATORS}
        self._attempts: dict[str, float] = dict.fromkeys(OPERATORS, 0.0)
        self._inserted: dict[str, float] = dict.fromkeys(OPERATORS, 0.0)
        self._log = logger.bind(component="OperatorScheduler")

    def record(self, operator: str, attempted: int, inserted: int) -> None:
        """Registra el resultado de un operador en la generación actual."""
        if operator not in self._attempts or attempted <= 0:
            return
        self._attempts[operator] += attempted
        self._inserted[operator] += inserted

    def decay(self) -> None:
        """Desvanece la historia: lo reciente pesa más que lo antiguo."""
        for op in OPERATORS:
            self._attempts[op] *= _DECAY
            self._inserted[op] *= _DECAY

    def success_rate(self, operator: str) -> float | None:
        """Tasa de inserción observada, o None sin datos suficientes."""
        attempts = self._attempts.get(operator, 0.0)
        if attempts < 1.0:
            return None
        return self._inserted[operator] / attempts

    def allocate(self, pop_size: int) -> dict[str, int]:
        """Reparte el presupuesto de la generación entre operadores.

        Peso de cada operador = tasa base x (0.5 + tasa de éxito reciente).
        Sin datos aún, se usa la tasa base tal cual. Suelo del 15% para
        que ningún operador desaparezca (exploración garantizada).
        """
        weights: dict[str, float] = {}
        for op in OPERATORS:
            success = self.success_rate(op)
            factor = 1.0 if success is None else (0.5 + success)
            weights[op] = self._base[op] * factor

        total = sum(weights.values()) or 1.0
        shares = {op: w / total for op, w in weights.items()}

        # Suelo de exploración y renormalización
        for op in OPERATORS:
            if self._base[op] > 0:
                shares[op] = max(shares[op], _MIN_SHARE)
        total = sum(shares.values())
        shares = {op: s / total for op, s in shares.items()}

        allocation = {op: round(pop_size * shares[op]) for op in OPERATORS}
        # La inyección fresca conserva al menos 1 (diversidad garantizada)
        if self._base["fresh"] > 0:
            allocation["fresh"] = max(1, allocation["fresh"])

        self._log.debug(
            "operator_allocation",
            allocation=allocation,
            success={op: self.success_rate(op) for op in OPERATORS},
        )
        return allocation
