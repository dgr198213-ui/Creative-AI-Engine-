"""Tests de los operadores adaptativos (meta-cognición viable)."""

from creative_engine.evolution.adaptive_operators import OperatorScheduler

BASE = {"mutation": 0.4, "crossover": 0.3, "fresh": 0.3}


class TestScheduler:
    def test_without_data_uses_base_rates(self) -> None:
        s = OperatorScheduler(BASE)
        alloc = s.allocate(10)
        # Sin historia: reparto ≈ tasas base
        assert alloc["mutation"] == 4
        assert alloc["crossover"] == 3
        assert alloc["fresh"] == 3

    def test_successful_operator_gains_budget(self) -> None:
        s = OperatorScheduler(BASE)
        # La mutación produce élites; el cruce no
        s.record("mutation", attempted=4, inserted=4)   # 100% éxito
        s.record("crossover", attempted=3, inserted=0)  # 0% éxito
        s.record("fresh", attempted=3, inserted=1)
        alloc = s.allocate(10)
        assert alloc["mutation"] > alloc["crossover"], alloc
        assert alloc["mutation"] >= 4  # ganó presupuesto

    def test_floor_prevents_operator_extinction(self) -> None:
        """Ni siquiera un operador que fracasa siempre desaparece del todo."""
        s = OperatorScheduler(BASE)
        for _ in range(5):
            s.record("crossover", attempted=5, inserted=0)
            s.record("mutation", attempted=5, inserted=5)
            s.decay()
        alloc = s.allocate(10)
        assert alloc["crossover"] >= 1, f"el cruce no debe extinguirse: {alloc}"

    def test_fresh_injection_always_at_least_one(self) -> None:
        s = OperatorScheduler(BASE)
        s.record("fresh", attempted=10, inserted=0)
        s.record("mutation", attempted=5, inserted=5)
        alloc = s.allocate(6)
        assert alloc["fresh"] >= 1  # diversidad garantizada

    def test_decay_fades_old_history(self) -> None:
        s = OperatorScheduler(BASE)
        s.record("mutation", attempted=10, inserted=0)  # mal comienzo
        for _ in range(6):
            s.decay()
        # historia casi desvanecida → success_rate sin datos suficientes
        assert s.success_rate("mutation") is None

    def test_record_ignores_unknown_operator(self) -> None:
        s = OperatorScheduler(BASE)
        s.record("inexistente", attempted=3, inserted=3)  # no debe romper
        assert s.allocate(10)["mutation"] == 4
