"""Guard de presupuesto (Fase 5, bloque 3): degradación automática a
proveedores gratuitos cuando el gasto estimado supera un límite.

Opción B del diseño (23-jul-2026): precio por millón de tokens
configurable por proveedor (`CREATIVE_LLM__<N>__PRICE_IN` /
`PRICE_OUT`). Un proveedor sin precio declarado se trata como gratuito y
nunca cuenta para el guard. Esto renuncia a la exactitud contable de
consultar la API de uso de cada proveedor (opción C del diseño) — la
cifra es una ESTIMACIÓN (tokens x precio configurado), no la factura
real — a cambio de que el mecanismo funcione igual con Gemini, Z.ai,
OpenAI o cualquier proveedor futuro sin integraciones específicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import structlog

from ..core.config import Settings

if TYPE_CHECKING:
    from ..memory.repository import IdeaRepository
    from .router import LLMModelRouter

logger = structlog.get_logger(__name__)

BudgetState = Literal["ok", "warning", "downgraded"]


def period_key(period: str, now: datetime | None = None) -> str:
    """Clave de la ventana temporal para acumular gasto.

    "daily" → "2026-07-23"; cualquier otro valor (incluido "monthly" o
    desconocido) → "2026-07", mensual por defecto.
    """
    ts = now or datetime.now(UTC)
    if period == "daily":
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m")


def estimate_cost_usd(
    price_in: float, price_out: float, prompt_tokens: int, completion_tokens: int
) -> float:
    """Coste estimado en USD: tokens x precio declarado por millón."""
    return (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out


@dataclass
class BudgetStatus:
    spent_usd: float
    limit_usd: float
    period: str
    period_key: str
    state: BudgetState
    excluded_providers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "spent_usd": round(self.spent_usd, 4),
            "limit_usd": self.limit_usd,
            "period": self.period,
            "period_key": self.period_key,
            "status": self.state,
            "excluded_providers": self.excluded_providers,
        }


async def get_budget_status(
    settings: Settings, repository: IdeaRepository | None
) -> BudgetStatus:
    """Consulta el gasto acumulado del periodo actual y decide si degradar.

    Sin límite configurado (`budget_limit=0`) o sin BD para consultar el
    gasto acumulado: siempre "ok" y sin exclusiones — el guard solo actúa
    cuando hay ambas cosas. Con `budget_enforce=False`, sigue reportando
    "downgraded" si el gasto supera el límite (contabilidad intacta) pero
    no excluye proveedores (routing intacto).
    """
    key = period_key(settings.budget_period)
    limit = settings.budget_limit

    spent = 0.0
    if repository is not None:
        try:
            by_provider = await repository.get_period_spend(key)
            spent = sum(by_provider.values())
        except Exception as e:
            logger.warning("budget_spend_query_failed", error=str(e))

    if limit <= 0.0:
        return BudgetStatus(
            spent_usd=spent, limit_usd=limit, period=settings.budget_period,
            period_key=key, state="ok",
        )

    if spent >= limit:
        state: BudgetState = "downgraded"
    elif spent >= limit * settings.budget_warning_ratio:
        state = "warning"
    else:
        state = "ok"

    excluded: list[str] = []
    if state == "downgraded" and settings.budget_enforce:
        excluded = [name for name, cfg in settings.llm.items() if cfg.is_paid]

    if state == "downgraded":
        logger.warning(
            "budget_exhausted_downgrade",
            spent_usd=round(spent, 4),
            limit_usd=limit,
            period_key=key,
            enforced=settings.budget_enforce,
            excluded_providers=excluded,
        )
    elif state == "warning":
        logger.warning(
            "budget_warning", spent_usd=round(spent, 4), limit_usd=limit, period_key=key
        )

    return BudgetStatus(
        spent_usd=spent,
        limit_usd=limit,
        period=settings.budget_period,
        period_key=key,
        state=state,
        excluded_providers=excluded,
    )


async def record_run_spend(
    router: LLMModelRouter, settings: Settings, repository: IdeaRepository | None
) -> None:
    """Persiste el gasto estimado de este run, por proveedor de pago.

    Best-effort y opcional (igual que el resto de la persistencia del
    proyecto): sin `repository`, no hace nada — el guard sigue funcionando
    dentro de un mismo proceso vía los contadores en memoria del router,
    pero la acumulación entre runs/reinicios necesita BD.
    """
    if repository is None:
        return

    key = period_key(settings.budget_period)
    for name, provider in router.providers.items():
        cfg = settings.llm.get(name)
        if cfg is None or not cfg.is_paid:
            continue  # gratuitos no cuentan para el guard ni se persisten

        cost = estimate_cost_usd(
            cfg.price_in,
            cfg.price_out,
            provider.total_prompt_tokens,
            provider.total_completion_tokens,
        )
        if provider.total_calls == 0:
            continue

        try:
            await repository.record_provider_spend(
                provider=name,
                period_key=key,
                cost_usd=cost,
                prompt_tokens=provider.total_prompt_tokens,
                completion_tokens=provider.total_completion_tokens,
            )
        except Exception as e:
            logger.warning("budget_spend_record_failed", provider=name, error=str(e))
