"""Tests del arnés de benchmark de 3 brazos (diseño 22-jul-2026 §3, §6).

Extremo a extremo con LLM simulado (sin red, sin BD) + tests puros de
agregación/veredicto/informe sobre resultados sintéticos.
"""

import json
from unittest.mock import patch

import pytest

from creative_engine.bench.config import BenchChallenge, BenchSetConfig
from creative_engine.bench.harness import ArmCost, BenchArmResult, BenchChallengeResult

_TOPICS = [
    ("Checkout simplificado", "Rediseño del proceso de pago en dos pasos."),
    ("Programa de referidos", "Incentivo para que clientes traigan a otros clientes."),
    ("Bundle de suscripción", "Paquete mensual con descuento por volumen."),
    ("Chatbot de soporte", "Asistente que resuelve dudas frecuentes al instante."),
]
_counter = {"n": 0}


def _fake_generate(prompt: str, **kwargs) -> str:
    """Array de ideas para generación/auto-mejora, objeto para mutación/cruce."""
    if "Genera" in prompt and "array" in prompt:
        items = []
        for _ in range(3):
            topic = _TOPICS[_counter["n"] % len(_TOPICS)]
            _counter["n"] += 1
            items.append(
                {
                    "title": f"{topic[0]} v{_counter['n']}",
                    "description": topic[1] + f" Variante {_counter['n']}.",
                    "advantages": ["Ventaja A"],
                    "limitations": ["Limitación X"],
                    "features": {"technologies": ["tech"], "complexity_level": 0.5},
                }
            )
        return json.dumps(items)

    topic = _TOPICS[_counter["n"] % len(_TOPICS)]
    _counter["n"] += 1
    return json.dumps(
        {
            "title": f"{topic[0]} mutada v{_counter['n']}",
            "description": topic[1] + f" Evolución {_counter['n']}.",
            "advantages": ["Ventaja evolucionada"],
            "limitations": ["Limitación"],
            "mutation_description": "cambio simulado",
        }
    )


def _fake_structured(prompt: str, **kwargs) -> dict:
    lowered = prompt.lower()
    if "perfil funcional" in lowered or "topografia" in lowered:
        return {
            "topografia": {"que_ocurre": "las ventas bajaron", "frecuencia": "recurrente"},
            "hipotesis_funcional": {"mecanismo": "el checkout es confuso", "confianza": 0.7},
            "friccion": {"impacto_principal": "dinero", "descripcion_impacto": "ingresos"},
            "reto_reformulado": "Reducir la fricción del proceso de pago",
            "preguntas_pendientes": [],
        }
    if "accionabilidad" in lowered:
        return {"puntuaciones": [{"accionabilidad": 7, "pertinencia": 8} for _ in range(3)]}
    return {
        "utility": 0.7,
        "utility_feedback": "ok",
        "feasibility": 0.6,
        "feasibility_feedback": "ok",
        "market_fit": 0.5,
        "market_feedback": "ok",
        "estimated_complexity": 0.5,
    }


class _CountingSimProvider:
    """Doble de LLMProvider con contadores de coste reales (no un Mock):
    router.total_calls/total_tokens leen estos atributos de verdad."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        self.total_calls += 1
        self.total_prompt_tokens += 10
        self.total_completion_tokens += 20
        return _fake_generate(prompt, **kwargs)

    async def generate_structured(self, prompt: str, **kwargs) -> dict:
        self.total_calls += 1
        self.total_prompt_tokens += 10
        self.total_completion_tokens += 20
        return _fake_structured(prompt, **kwargs)

    async def close(self) -> None:
        return None


class TestRunBenchSetEndToEnd:
    async def test_three_arms_produce_results(self, deterministic_embed) -> None:
        from creative_engine.bench.harness import run_bench_set
        from creative_engine.core.config import LLMProviderConfig, Settings
        from creative_engine.core.config import SecretStr as _SecretStr
        from creative_engine.evolution import encoders as enc

        settings = Settings.load()
        settings.llm = {"default": LLMProviderConfig(name="sim", api_key=_SecretStr("x"))}

        set_config = BenchSetConfig(
            name="test_set",
            domain="generic",
            retos=[
                BenchChallenge(texto="Mi tienda online no vende nada", tipo="vago"),
                BenchChallenge(
                    texto="Reto de control bien formulado y específico para probar",
                    tipo="control",
                ),
            ],
            repeticiones=1,
            poblacion_motor=4,
            generaciones_motor=1,
            ideas_por_brazo=3,
        )

        orig_embed = enc.IdeaEncoder._embed
        enc.IdeaEncoder._embed = lambda self, text: deterministic_embed(text)
        try:
            with patch(
                "creative_engine.llm.factory.LLMProvider",
                side_effect=lambda cfg: _CountingSimProvider(),
            ):
                results = await run_bench_set(set_config, settings)
        finally:
            enc.IdeaEncoder._embed = orig_embed

        assert len(results) == 2
        tipos = {r.reto_tipo for r in results}
        assert tipos == {"vago", "control"}

        for r in results:
            assert set(r.arms) == {"A", "B", "C"}
            for arm in r.arms.values():
                assert arm.cost.calls > 0
                assert arm.n_ideas >= 0

        # El brazo C debe haber gastado al menos 1 llamada más que B (el
        # análisis cuenta dentro del presupuesto, tal como pide el diseño).
        vago = next(r for r in results if r.reto_tipo == "vago")
        assert vago.arms["C"].cost.calls >= vago.arms["B"].cost.calls


class _FakeBenchRepository:
    """Repositorio en memoria: mismo contrato que IdeaRepository para
    save_bench_result/get_bench_results, sin BD real — permite probar
    persistencia incremental y reanudación sin red ni Postgres."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def save_bench_result(
        self, set_name: str, challenge: str, reto_tipo: str, repetition: int, arms: dict
    ) -> None:
        self.rows.append(
            {
                "set_name": set_name,
                "challenge": challenge,
                "reto_tipo": reto_tipo,
                "repetition": repetition,
                "arms": arms,
            }
        )

    async def get_bench_results(self, set_name: str) -> list[dict]:
        return [dict(r) for r in self.rows if r["set_name"] == set_name]


class TestRunBenchSetPersistenceAndResume:
    async def test_persists_each_challenge_as_it_completes(self, deterministic_embed) -> None:
        """Con repository: cada reto/repetición queda en BD antes de que
        termine el set completo (no solo al final)."""
        from creative_engine.bench.harness import run_bench_set
        from creative_engine.core.config import LLMProviderConfig, Settings
        from creative_engine.core.config import SecretStr as _SecretStr
        from creative_engine.evolution import encoders as enc

        settings = Settings.load()
        settings.llm = {"default": LLMProviderConfig(name="sim", api_key=_SecretStr("x"))}

        set_config = BenchSetConfig(
            name="test_set_persist",
            domain="generic",
            retos=[BenchChallenge(texto="Mi tienda online no vende nada", tipo="vago")],
            repeticiones=2,
            poblacion_motor=4,
            generaciones_motor=1,
            ideas_por_brazo=3,
        )

        repo = _FakeBenchRepository()
        orig_embed = enc.IdeaEncoder._embed
        enc.IdeaEncoder._embed = lambda self, text: deterministic_embed(text)
        try:
            with patch(
                "creative_engine.llm.factory.LLMProvider",
                side_effect=lambda cfg: _CountingSimProvider(),
            ):
                results = await run_bench_set(set_config, settings, repository=repo)
        finally:
            enc.IdeaEncoder._embed = orig_embed

        # Las 2 repeticiones quedaron persistidas, no solo devueltas en RAM.
        assert len(repo.rows) == 2
        assert {r["repetition"] for r in repo.rows} == {0, 1}
        assert len(results) == 2

    async def test_resumes_skipping_already_persisted(self, deterministic_embed) -> None:
        """Si una repetición ya está en BD, run_bench_set no la vuelve a
        ejecutar — el set es reanudable tras una caída a mitad."""
        from creative_engine.bench.harness import run_bench_set
        from creative_engine.core.config import LLMProviderConfig, Settings
        from creative_engine.core.config import SecretStr as _SecretStr
        from creative_engine.evolution import encoders as enc

        settings = Settings.load()
        settings.llm = {"default": LLMProviderConfig(name="sim", api_key=_SecretStr("x"))}

        set_config = BenchSetConfig(
            name="test_set_resume",
            domain="generic",
            retos=[BenchChallenge(texto="Mi tienda online no vende nada", tipo="vago")],
            repeticiones=2,
            poblacion_motor=4,
            generaciones_motor=1,
            ideas_por_brazo=3,
        )

        repo = _FakeBenchRepository()
        # Simula que la repetición 0 ya se completó en una corrida anterior
        # (p.ej. el proceso murió por OOM justo después de persistirla).
        await repo.save_bench_result(
            set_name="test_set_resume",
            challenge="Mi tienda online no vende nada",
            reto_tipo="vago",
            repetition=0,
            arms={
                "A": {
                    "arm": "A_prompt_unico", "n_ideas": 3,
                    "mean_pairwise_distance": 0.1, "min_pairwise_distance": 0.05,
                    "blind_utility": 5.0, "cost": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 20},
                    "elapsed_s": 1.0, "qd_score": None, "coverage": None, "titles": ["x"],
                },
                "B": {
                    "arm": "B_motor_solo", "n_ideas": 3,
                    "mean_pairwise_distance": 0.1, "min_pairwise_distance": 0.05,
                    "blind_utility": 5.0, "cost": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 20},
                    "elapsed_s": 1.0, "qd_score": 0.3, "coverage": 0.1, "titles": ["x"],
                },
                "C": {
                    "arm": "C_motor_analista", "n_ideas": 3,
                    "mean_pairwise_distance": 0.1, "min_pairwise_distance": 0.05,
                    "blind_utility": 5.0, "cost": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 20},
                    "elapsed_s": 1.0, "qd_score": 0.3, "coverage": 0.1, "titles": ["x"],
                },
            },
        )

        orig_embed = enc.IdeaEncoder._embed
        enc.IdeaEncoder._embed = lambda self, text: deterministic_embed(text)
        try:
            with patch(
                "creative_engine.llm.factory.LLMProvider",
                side_effect=lambda cfg: _CountingSimProvider(),
            ):
                results = await run_bench_set(set_config, settings, repository=repo)
        finally:
            enc.IdeaEncoder._embed = orig_embed

        # Solo se ejecutó (y persistió) la repetición 1: la 0 se saltó.
        assert len(repo.rows) == 2
        assert {r["repetition"] for r in repo.rows} == {0, 1}
        assert len(results) == 2


def _arm(calls: int, diversity: float, blind: float | None, qd: float | None) -> BenchArmResult:
    return BenchArmResult(
        arm="x",
        n_ideas=3,
        mean_pairwise_distance=diversity,
        min_pairwise_distance=diversity * 0.5,
        blind_utility=blind,
        cost=ArmCost(calls=calls, prompt_tokens=calls * 100, completion_tokens=calls * 50),
        elapsed_s=1.0,
        qd_score=qd,
        titles=["a", "b", "c"],
    )


def _challenge_result(reto_tipo: str, a: BenchArmResult, b: BenchArmResult, c: BenchArmResult) -> BenchChallengeResult:
    return BenchChallengeResult(
        challenge="reto de prueba",
        reto_tipo=reto_tipo,
        repetition=0,
        arms={"A": a, "B": b, "C": c},
    )


class TestAggregateAndVerdict:
    def test_aggregate_computes_means(self) -> None:
        from creative_engine.bench.report import aggregate

        results = [
            _challenge_result(
                "vago",
                _arm(5, 0.3, 6.0, 0.5),
                _arm(8, 0.5, 7.0, 0.6),
                _arm(9, 0.5, 8.0, 0.7),
            ),
            _challenge_result(
                "vago",
                _arm(5, 0.3, 6.0, 0.5),
                _arm(8, 0.5, 7.0, 0.6),
                _arm(9, 0.5, 8.0, 0.7),
            ),
        ]
        agg = aggregate(results, "vago")
        assert agg["B"].mean_diversity == pytest.approx(0.5)
        assert agg["C"].mean_blind_utility == pytest.approx(8.0)
        assert agg["B"].mean_calls == pytest.approx(8.0)

    def test_verdict_criterion1_met_when_c_beats_b_by_15_percent(self) -> None:
        from creative_engine.bench.report import compute_verdict

        # C supera a B en qd_score y utilidad ciega en >=15% (vagos);
        # controles: C no empeora; B > A en ambos tipos.
        results = [
            _challenge_result(
                "vago",
                _arm(5, 0.3, 5.0, 0.4),   # A
                _arm(8, 0.5, 6.0, 0.5),   # B
                _arm(9, 0.5, 7.5, 0.65),  # C: +25% util, +30% qd sobre B
            ),
            _challenge_result(
                "control",
                _arm(5, 0.3, 5.0, 0.4),
                _arm(8, 0.5, 7.0, 0.6),
                _arm(9, 0.5, 6.9, 0.59),  # C casi igual a B (dentro del 5%)
            ),
        ]
        verdict = compute_verdict(results)
        assert verdict.vagos_c_beats_b is True
        assert verdict.controles_c_no_peor is True
        assert verdict.b_beats_a_vagos is True
        assert verdict.b_beats_a_controles is True

    def test_verdict_criterion1_not_met_when_gain_too_small(self) -> None:
        from creative_engine.bench.report import compute_verdict

        results = [
            _challenge_result(
                "vago",
                _arm(5, 0.3, 5.0, 0.4),
                _arm(8, 0.5, 6.0, 0.5),
                _arm(9, 0.5, 6.1, 0.51),  # C solo +1.7% sobre B: ruido, no señal
            ),
        ]
        verdict = compute_verdict(results)
        assert verdict.vagos_c_beats_b is False

    def test_verdict_criterion2_fails_when_c_hurts_controls(self) -> None:
        from creative_engine.bench.report import compute_verdict

        results = [
            _challenge_result(
                "control",
                _arm(5, 0.3, 5.0, 0.4),
                _arm(8, 0.5, 8.0, 0.7),
                _arm(9, 0.5, 6.0, 0.5),  # C empeora a B bastante más de un 5%
            ),
        ]
        verdict = compute_verdict(results)
        assert verdict.controles_c_no_peor is False

    def test_verdict_handles_missing_blind_utility(self) -> None:
        from creative_engine.bench.report import compute_verdict

        results = [
            _challenge_result(
                "vago",
                _arm(5, 0.3, None, None),
                _arm(8, 0.5, None, None),
                _arm(9, 0.5, None, None),
            ),
        ]
        verdict = compute_verdict(results)
        assert verdict.vagos_c_beats_b is False
        assert "sin datos suficientes" not in verdict.summary or verdict.summary


class TestRenderMarkdown:
    def test_report_contains_sections_and_verdict(self) -> None:
        from creative_engine.bench.report import render_markdown

        results = [
            _challenge_result(
                "vago",
                _arm(5, 0.3, 5.0, 0.4),
                _arm(8, 0.5, 6.0, 0.5),
                _arm(9, 0.5, 7.5, 0.65),
            ),
            _challenge_result(
                "control",
                _arm(5, 0.3, 5.0, 0.4),
                _arm(8, 0.5, 7.0, 0.6),
                _arm(9, 0.5, 6.9, 0.59),
            ),
        ]
        report = render_markdown(results, "vagos")

        assert "# Benchmark — vagos" in report
        assert "## Retos vagos" in report
        assert "## Retos control" in report
        assert "## Veredicto" in report
        assert "A — Prompt único" in report
        assert "B — Motor solo" in report
        assert "C — Motor + Analista" in report
