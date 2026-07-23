"""Guard de presupuesto (Fase 5, bloque 3): degradación automática a
proveedores gratuitos al superar el límite estimado. Sin red ni BD real:
un repositorio en memoria hace de Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from creative_engine.core.config import LLMProviderConfig, Settings
from creative_engine.core.config import SecretStr as _SecretStr
from creative_engine.llm.budget import (
    estimate_cost_usd,
    get_budget_status,
    period_key,
    record_run_spend,
)
from creative_engine.llm.factory import build_router


class _FakeBudgetRepository:
    """Mismo contrato que IdeaRepository para record_provider_spend/
    get_period_spend, sin BD real."""

    def __init__(self) -> None:
        self.spend: dict[tuple[str, str], float] = {}
        self.calls: list[dict] = []

    async def record_provider_spend(
        self, provider: str, period_key: str, cost_usd: float,
        prompt_tokens: int, completion_tokens: int,
    ) -> None:
        self.calls.append(
            {
                "provider": provider, "period_key": period_key, "cost_usd": cost_usd,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            }
        )
        key = (provider, period_key)
        self.spend[key] = self.spend.get(key, 0.0) + cost_usd

    async def get_period_spend(self, period_key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for (provider, key), cost in self.spend.items():
            if key == period_key:
                out[provider] = out.get(provider, 0.0) + cost
        return out


class _FakeProvider:
    """Doble de LLMProvider con contadores reales, sin red."""

    def __init__(self) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        self.total_calls += 1
        self.total_prompt_tokens += 1000
        self.total_completion_tokens += 500
        return "ok"

    async def generate_structured(self, prompt: str, **kwargs) -> dict:
        self.total_calls += 1
        self.total_prompt_tokens += 1000
        self.total_completion_tokens += 500
        return {}

    async def close(self) -> None:
        return None


def _settings_with_providers() -> Settings:
    s = Settings.load()
    s.llm = {
        "free_provider": LLMProviderConfig(name="free", api_key=_SecretStr("x")),
        "paid_provider": LLMProviderConfig(
            name="paid", api_key=_SecretStr("x"), price_in=5.0, price_out=15.0
        ),
    }
    return s


class TestPeriodKeyAndEstimate:
    def test_monthly_period_key(self) -> None:
        now = datetime(2026, 7, 23, tzinfo=UTC)
        assert period_key("monthly", now) == "2026-07"

    def test_daily_period_key(self) -> None:
        now = datetime(2026, 7, 23, tzinfo=UTC)
        assert period_key("daily", now) == "2026-07-23"

    def test_unknown_period_defaults_to_monthly(self) -> None:
        now = datetime(2026, 7, 23, tzinfo=UTC)
        assert period_key("weekly", now) == "2026-07"

    def test_estimate_cost(self) -> None:
        # 1M tokens de prompt a $5/1M + 500k de completion a $15/1M = 5 + 7.5
        cost = estimate_cost_usd(price_in=5.0, price_out=15.0,
                                   prompt_tokens=1_000_000, completion_tokens=500_000)
        assert cost == pytest.approx(12.5)

    def test_free_provider_is_not_paid(self) -> None:
        cfg = LLMProviderConfig(name="x", api_key=_SecretStr("x"))
        assert cfg.is_paid is False

    def test_provider_with_price_is_paid(self) -> None:
        cfg = LLMProviderConfig(name="x", api_key=_SecretStr("x"), price_in=1.0)
        assert cfg.is_paid is True


class TestGetBudgetStatus:
    async def test_ok_without_limit_configured(self) -> None:
        settings = _settings_with_providers()
        settings.budget_limit = 0.0
        repo = _FakeBudgetRepository()
        await repo.record_provider_spend("paid_provider", period_key(settings.budget_period), 999.0, 1, 1)

        status = await get_budget_status(settings, repo)
        assert status.state == "ok"
        assert status.excluded_providers == []

    async def test_ok_without_repository(self) -> None:
        settings = _settings_with_providers()
        settings.budget_limit = 10.0
        status = await get_budget_status(settings, None)
        assert status.state == "ok"
        assert status.spent_usd == 0.0

    async def test_warning_at_80_percent(self) -> None:
        settings = _settings_with_providers()
        settings.budget_limit = 10.0
        repo = _FakeBudgetRepository()
        await repo.record_provider_spend("paid_provider", period_key(settings.budget_period), 8.5, 1, 1)

        status = await get_budget_status(settings, repo)
        assert status.state == "warning"
        assert status.excluded_providers == []

    async def test_downgraded_excludes_paid_providers_when_enforced(self) -> None:
        settings = _settings_with_providers()
        settings.budget_limit = 10.0
        settings.budget_enforce = True
        repo = _FakeBudgetRepository()
        await repo.record_provider_spend("paid_provider", period_key(settings.budget_period), 11.0, 1, 1)

        status = await get_budget_status(settings, repo)
        assert status.state == "downgraded"
        assert status.excluded_providers == ["paid_provider"]

    async def test_downgraded_keeps_routing_intact_when_not_enforced(self) -> None:
        """CREATIVE_BUDGET_ENFORCE=false: sigue contabilizando y avisando,
        pero no toca el routing."""
        settings = _settings_with_providers()
        settings.budget_limit = 10.0
        settings.budget_enforce = False
        repo = _FakeBudgetRepository()
        await repo.record_provider_spend("paid_provider", period_key(settings.budget_period), 11.0, 1, 1)

        status = await get_budget_status(settings, repo)
        assert status.state == "downgraded"
        assert status.excluded_providers == []


class TestRecordRunSpend:
    async def test_records_only_paid_providers(self) -> None:
        settings = _settings_with_providers()
        repo = _FakeBudgetRepository()

        from creative_engine.llm.router import LLMModelRouter

        free = _FakeProvider()
        paid = _FakeProvider()
        await free.generate("hola")
        await paid.generate("hola")

        router = LLMModelRouter(
            providers={"free_provider": free, "paid_provider": paid}
        )

        await record_run_spend(router, settings, repo)

        assert len(repo.calls) == 1
        assert repo.calls[0]["provider"] == "paid_provider"
        assert repo.calls[0]["cost_usd"] > 0

    async def test_noop_without_repository(self) -> None:
        settings = _settings_with_providers()
        from creative_engine.llm.router import LLMModelRouter

        router = LLMModelRouter(
            providers={"free_provider": _FakeProvider(), "paid_provider": _FakeProvider()}
        )
        await record_run_spend(router, settings, None)  # no debe lanzar


class TestAcceptanceDowngradeCompletesRun:
    """Criterio de aceptación del diseño: simula gasto por encima del
    límite y verifica que la siguiente llamada no usa proveedores de pago
    y que el run se completa igualmente."""

    async def test_next_call_avoids_paid_providers_and_completes(self) -> None:
        settings = _settings_with_providers()
        settings.budget_limit = 10.0
        settings.budget_enforce = True
        repo = _FakeBudgetRepository()
        # Gasto ya por encima del límite en el periodo actual.
        await repo.record_provider_spend(
            "paid_provider", period_key(settings.budget_period), 50.0, 1, 1
        )

        status = await get_budget_status(settings, repo)
        assert status.state == "downgraded"

        from unittest.mock import patch

        free = _FakeProvider()
        paid = _FakeProvider()
        with patch(
            "creative_engine.llm.factory.LLMProvider",
            side_effect=lambda cfg: paid if cfg.name == "paid" else free,
        ):
            router = build_router(settings, budget_excluded=set(status.excluded_providers))

        result = await router.for_role("generator").generate("prueba")

        assert result == "ok"
        assert free.total_calls == 1
        assert paid.total_calls == 0  # nunca se llegó a llamar al proveedor de pago
