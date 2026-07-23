"""Agregación, veredicto e informe Markdown del benchmark de 3 brazos.

Criterios de éxito para que el Analista se quede en el motor (§3 del
diseño 22-jul-2026):

1. En retos vagos: C > B en qd_score y utilidad ciega (≥ +15%, señal no ruido).
2. En controles bien formulados: C no empeora a B más de un 5% (el
   Analista no debe estropear un input ya bueno).
3. B > A en diversidad y utilidad en ambos tipos (valida el motor mismo).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .harness import BenchChallengeResult

ARM_LABELS = (("A", "A — Prompt único"), ("B", "B — Motor solo"), ("C", "C — Motor + Analista"))


@dataclass
class ArmAggregate:
    """Medias de un brazo sobre un subconjunto de resultados (p.ej. un tipo de reto)."""

    arm: str
    n: int
    mean_diversity: float
    mean_blind_utility: float | None
    mean_qd_score: float | None
    mean_calls: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    # Presupuesto objetivo (Fase 5, bloque 1): consumo real de B para el
    # mismo reto. None en B misma (es la referencia, no tiene objetivo).
    mean_budget_calls: float | None


@dataclass
class BenchVerdict:
    vagos_c_beats_b: bool | None
    controles_c_no_peor: bool | None
    b_beats_a_vagos: bool | None
    b_beats_a_controles: bool | None
    summary: str


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def aggregate(
    results: list[BenchChallengeResult], reto_tipo: str | None = None
) -> dict[str, ArmAggregate]:
    """Agrega por brazo (A/B/C), opcionalmente filtrando por tipo de reto."""
    filtered = [r for r in results if reto_tipo is None or r.reto_tipo == reto_tipo]

    out: dict[str, ArmAggregate] = {}
    for arm_key, _ in ARM_LABELS:
        arms = [r.arms[arm_key] for r in filtered]
        out[arm_key] = ArmAggregate(
            arm=arm_key,
            n=len(arms),
            mean_diversity=_mean([a.mean_pairwise_distance for a in arms]) or 0.0,
            mean_blind_utility=_mean([a.blind_utility for a in arms]),
            mean_qd_score=_mean([a.qd_score for a in arms]),
            mean_calls=_mean([float(a.cost.calls) for a in arms]) or 0.0,
            mean_prompt_tokens=_mean([float(a.cost.prompt_tokens) for a in arms]) or 0.0,
            mean_completion_tokens=_mean([float(a.cost.completion_tokens) for a in arms]) or 0.0,
            mean_budget_calls=_mean(
                [float(a.budget_calls) if a.budget_calls is not None else None for a in arms]
            ),
        )
    return out


def _pct_gain(new: float | None, base: float | None) -> float | None:
    if new is None or base is None or base == 0:
        return None
    return (new - base) / abs(base)


def compute_verdict(results: list[BenchChallengeResult]) -> BenchVerdict:
    """Evalúa los 3 criterios de éxito del §3 sobre el set completo."""
    vagos = aggregate(results, "vago")
    controles = aggregate(results, "control")

    qd_gain = _pct_gain(vagos["C"].mean_qd_score, vagos["B"].mean_qd_score)
    utility_gain = _pct_gain(vagos["C"].mean_blind_utility, vagos["B"].mean_blind_utility)
    vagos_c_beats_b = (
        qd_gain is not None
        and utility_gain is not None
        and qd_gain >= 0.15
        and utility_gain >= 0.15
    )

    qd_drop = _pct_gain(controles["C"].mean_qd_score, controles["B"].mean_qd_score)
    utility_drop = _pct_gain(controles["C"].mean_blind_utility, controles["B"].mean_blind_utility)
    controles_c_no_peor = (
        qd_drop is not None
        and utility_drop is not None
        and qd_drop >= -0.05
        and utility_drop >= -0.05
    )

    def _b_beats_a(agg: dict[str, ArmAggregate]) -> bool | None:
        util_gain = _pct_gain(agg["B"].mean_blind_utility, agg["A"].mean_blind_utility)
        if util_gain is None:
            return None
        return (agg["B"].mean_diversity - agg["A"].mean_diversity) > 0 and util_gain > 0

    b_beats_a_vagos = _b_beats_a(vagos)
    b_beats_a_controles = _b_beats_a(controles)

    lines = []
    if qd_gain is not None and utility_gain is not None:
        lines.append(
            f"- Criterio 1 (vagos, C > B): "
            f"{'cumplido' if vagos_c_beats_b else 'no cumplido'} "
            f"(qd_score {qd_gain:+.1%}, utilidad ciega {utility_gain:+.1%})"
        )
    else:
        lines.append("- Criterio 1 (vagos, C > B): sin datos suficientes")

    if qd_drop is not None and utility_drop is not None:
        lines.append(
            f"- Criterio 2 (controles, C no empeora >5%): "
            f"{'cumplido' if controles_c_no_peor else 'no cumplido'} "
            f"(qd_score {qd_drop:+.1%}, utilidad ciega {utility_drop:+.1%})"
        )
    else:
        lines.append("- Criterio 2 (controles, C no empeora >5%): sin datos suficientes")

    lines.append(
        f"- Criterio 3 (B > A en diversidad y utilidad, ambos tipos): "
        f"vagos {'cumplido' if b_beats_a_vagos else 'no cumplido'}, "
        f"controles {'cumplido' if b_beats_a_controles else 'no cumplido'}"
    )

    return BenchVerdict(
        vagos_c_beats_b=vagos_c_beats_b,
        controles_c_no_peor=controles_c_no_peor,
        b_beats_a_vagos=b_beats_a_vagos,
        b_beats_a_controles=b_beats_a_controles,
        summary="\n".join(lines),
    )


def _fmt(value: float | None, suffix: str = "") -> str:
    return f"{value:.3f}{suffix}" if value is not None else "—"


def _aggregate_table(agg: dict[str, ArmAggregate]) -> list[str]:
    lines = [
        "| Brazo | Diversidad media | Utilidad ciega (0-10) | QD-Score | "
        "Presupuesto objetivo | Llamadas reales | Tokens medios (prompt+completion) |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm_key, label in ARM_LABELS:
        a = agg[arm_key]
        tokens = a.mean_prompt_tokens + a.mean_completion_tokens
        budget = f"{a.mean_budget_calls:.1f}" if a.mean_budget_calls is not None else "—"
        lines.append(
            f"| {label} | {_fmt(a.mean_diversity)} | {_fmt(a.mean_blind_utility)} | "
            f"{_fmt(a.mean_qd_score)} | {budget} | {a.mean_calls:.1f} | {tokens:.0f} |"
        )
    return lines


def render_markdown(results: list[BenchChallengeResult], set_name: str) -> str:
    """Informe Markdown exportable con las tablas agregadas y el veredicto."""
    date = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_challenges = len({r.challenge for r in results})
    verdict = compute_verdict(results)

    lines = [
        f"# Benchmark — {set_name}",
        "",
        f"- **Fecha:** {date}",
        f"- **Retos:** {n_challenges} · **Ejecuciones totales:** {len(results)}",
        "",
        "Tres brazos con el mismo presupuesto aproximado: **A** prompt único, "
        "**B** motor QD solo, **C** motor QD + Analista Funcional. Coste real "
        "(llamadas, tokens) medido por brazo — ver tablas.",
        "",
        "## Retos vagos",
        "",
        *_aggregate_table(aggregate(results, "vago")),
        "",
        "## Retos control (bien formulados)",
        "",
        *_aggregate_table(aggregate(results, "control")),
        "",
        "## Veredicto",
        "",
        verdict.summary,
        "",
        "---",
        "",
        "*Generado por Creative AI Engine — benchmark de 3 brazos.*",
        "",
    ]
    return "\n".join(lines)
