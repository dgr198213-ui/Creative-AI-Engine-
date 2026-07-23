"""Arnés de benchmark de 3 brazos (diseño 22-jul-2026 §3):

- A (prompt único): el reto tal cual a un LLM; N ideas + 1 pasada de
  auto-mejora (equivale a un usuario hábil de ChatGPT con buen prompt).
- B (motor solo): pipeline QD actual, reto sin procesar.
- C (motor + Analista): perfil del Analista Funcional inyectado; la
  llamada del Analista cuenta dentro del presupuesto del brazo.

Presupuesto: aproximadamente igualado por diseño (misma escala de
trabajo — N ideas ~ población del motor —, mismos proveedores y routing
en los tres brazos), no por un conteo exacto de llamadas HTTP: el motor
evolutivo y un prompt único tienen dinámicas de llamada estructuralmente
distintas y forzar una igualdad exacta sería ilusorio. El coste real
(llamadas y tokens) de cada brazo se mide con los contadores del router
y se reporta explícitamente, para que cualquier desigualdad quede
visible y auditable en vez de asumida.

El juez ciego (bench/judge.py) usa el rol "writer": no participa en la
generación/evaluación de ningún brazo, así que puntúa sin haber influido
en lo que juzga.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from ..agents.combined_evaluator import CombinedEvaluatorAgent
from ..agents.evaluator_orchestrator import EvaluatorOrchestrator
from ..agents.generator import IdeaGeneratorAgent
from ..analysis.analyst import FunctionalAnalystAgent
from ..benchmark import pairwise_diversity
from ..core.config import Settings
from ..core.models import DomainConfig, DomainName, EvolutionRequest, Idea
from ..evolution.crossover import CrossoverEngine
from ..evolution.encoders import IdeaEncoder
from ..evolution.mutation import MutationEngine
from ..evolution.qd_engine import QDEngine
from ..llm.factory import build_router, role_llms
from ..llm.router import LLMModelRouter
from ..memory.repository import IdeaRepository
from .config import BenchSetConfig
from .judge import judge_blind

logger = structlog.get_logger(__name__)


@dataclass
class ArmCost:
    """Coste real medido de un brazo: llamadas lógicas y tokens."""

    calls: int
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class BenchArmResult:
    """Métricas de un brazo para un reto y una repetición."""

    arm: str
    n_ideas: int
    mean_pairwise_distance: float
    min_pairwise_distance: float
    blind_utility: float | None
    cost: ArmCost
    elapsed_s: float
    qd_score: float | None = None
    coverage: float | None = None
    titles: list[str] = field(default_factory=list)
    # Presupuesto objetivo en llamadas LLM (Fase 5, bloque 1): el consumo
    # real del brazo B para este mismo reto. None en B (es la referencia).
    budget_calls: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost"] = self.cost.to_dict()
        return d


@dataclass
class BenchChallengeResult:
    """Resultado de los 3 brazos para un reto y una repetición."""

    challenge: str
    reto_tipo: str
    repetition: int
    arms: dict[str, BenchArmResult]  # claves "A", "B", "C"

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenge": self.challenge,
            "reto_tipo": self.reto_tipo,
            "repetition": self.repetition,
            "arms": {k: v.to_dict() for k, v in self.arms.items()},
        }


def _row_to_challenge_result(row: dict[str, Any]) -> BenchChallengeResult:
    """Reconstruye un BenchChallengeResult desde una fila persistida en BD.

    Lo que se guarda (`BenchArmResult.to_dict()`) ya es ligero — títulos y
    métricas, sin genome_vector ni descripciones completas — así que el
    informe final se puede armar leyendo de BD sin mantener en memoria
    los resultados originales (con sus objetos Idea) durante todo el set.
    """
    arms = {
        key: BenchArmResult(
            arm=data["arm"],
            n_ideas=data["n_ideas"],
            mean_pairwise_distance=data["mean_pairwise_distance"],
            min_pairwise_distance=data["min_pairwise_distance"],
            blind_utility=data["blind_utility"],
            cost=ArmCost(**data["cost"]),
            elapsed_s=data["elapsed_s"],
            qd_score=data.get("qd_score"),
            coverage=data.get("coverage"),
            titles=data.get("titles", []),
            budget_calls=data.get("budget_calls"),
        )
        for key, data in row["arms"].items()
    }
    return BenchChallengeResult(
        challenge=row["challenge"],
        reto_tipo=row["reto_tipo"],
        repetition=row["repetition"],
        arms=arms,
    )


def _cost_delta(
    router: LLMModelRouter, calls_before: int, tokens_before: dict[str, int]
) -> ArmCost:
    tokens_after = router.total_tokens
    return ArmCost(
        calls=router.total_calls - calls_before,
        prompt_tokens=tokens_after["prompt_tokens"] - tokens_before["prompt_tokens"],
        completion_tokens=(
            tokens_after["completion_tokens"] - tokens_before["completion_tokens"]
        ),
    )


def _arm_metrics(
    arm: str,
    ideas: list[Idea],
    cost: ArmCost,
    elapsed_s: float,
    blind_utility: float | None,
    qd_score: float | None = None,
    coverage: float | None = None,
    budget_calls: int | None = None,
) -> BenchArmResult:
    mean_d, min_d = pairwise_diversity(ideas)
    return BenchArmResult(
        arm=arm,
        n_ideas=len(ideas),
        mean_pairwise_distance=round(mean_d, 4),
        min_pairwise_distance=round(min_d, 4),
        blind_utility=round(blind_utility, 2) if blind_utility is not None else None,
        cost=cost,
        elapsed_s=round(elapsed_s, 1),
        qd_score=round(qd_score, 3) if qd_score is not None else None,
        coverage=round(coverage, 4) if coverage is not None else None,
        titles=[i.title for i in ideas],
        budget_calls=budget_calls,
    )


def _fresh_qd_engine(
    roles: dict[str, Any], encoder: IdeaEncoder, max_concurrent: int
) -> QDEngine:
    """Motor QD con estado limpio: archivo MAP-Elites vacío para esta ejecución."""
    evaluator = EvaluatorOrchestrator(
        agents={"combined": CombinedEvaluatorAgent(roles["evaluator"])}
    )
    return QDEngine(
        generator=IdeaGeneratorAgent(roles["generator"]),
        evaluator=evaluator,
        mutation=MutationEngine(roles["generator"], max_concurrent=max_concurrent),
        crossover=CrossoverEngine(roles["generator"], max_concurrent=max_concurrent),
        encoder=encoder,
        repository=None,
    )


async def _run_arm_a(
    challenge: str,
    domain: DomainConfig,
    roles: dict[str, Any],
    encoder: IdeaEncoder,
    router: LLMModelRouter,
    n_ideas: int,
    budget_calls: int,
) -> BenchArmResult:
    """Brazo A: prompt único, repartido en rondas de respuestas
    independientes + auto-mejora hasta agotar `budget_calls` (Fase 5,
    bloque 1 — el consumo real del brazo B para este mismo reto).

    Antes A gastaba un puñado fijo de llamadas (generar + 1 auto-mejora)
    frente a las decenas que gasta el motor evolutivo: cualquier
    comparación de diversidad/utilidad medía un brazo hambriento contra
    uno bien alimentado. Repetir rondas hasta el mismo presupuesto real
    hace la comparación válida.
    """
    calls_before = router.total_calls
    tokens_before = router.total_tokens
    t0 = time.perf_counter()

    generator = IdeaGeneratorAgent(roles["generator"])
    all_ideas: list[Idea] = []
    # Tope de seguridad: como máximo una ronda por llamada de presupuesto,
    # para no bucear indefinidamente si algún proveedor falla sin llegar
    # a gastar la llamada que se le atribuye (no debería, es solo cinturón).
    max_rounds = max(1, budget_calls)
    rounds = 0
    while router.total_calls - calls_before < budget_calls and rounds < max_rounds:
        batch = await generator.generate_population(
            challenge=challenge, domain=domain, count=n_ideas
        )
        batch = await generator.refine_population(
            challenge=challenge, domain=domain, ideas=batch
        )
        all_ideas.extend(batch)
        rounds += 1

    for idea in all_ideas:
        encoder.encode_idea(idea, domain)

    elapsed = time.perf_counter() - t0
    cost = _cost_delta(router, calls_before, tokens_before)
    blind = await judge_blind(roles["writer"], challenge, all_ideas)
    return _arm_metrics(
        "A_prompt_unico", all_ideas, cost, elapsed, blind, budget_calls=budget_calls
    )


async def _run_arm_b(
    challenge: str,
    domain: DomainConfig,
    roles: dict[str, Any],
    encoder: IdeaEncoder,
    router: LLMModelRouter,
    n_ideas: int,
    population: int,
    generations: int,
) -> BenchArmResult:
    """Brazo B: motor solo, reto sin procesar."""
    calls_before = router.total_calls
    tokens_before = router.total_tokens
    t0 = time.perf_counter()

    engine = _fresh_qd_engine(roles, encoder, max_concurrent=5)
    state = await engine.run_evolution(
        EvolutionRequest(
            challenge=challenge,
            domain=domain.name,
            population_size=population,
            generations=generations,
        )
    )
    top = sorted(state.archive, key=lambda c: c.fitness, reverse=True)[:n_ideas]
    ideas = [c.elite for c in top]

    elapsed = time.perf_counter() - t0
    cost = _cost_delta(router, calls_before, tokens_before)
    blind = await judge_blind(roles["writer"], challenge, ideas)
    return _arm_metrics(
        "B_motor_solo", ideas, cost, elapsed, blind,
        qd_score=state.qd_score, coverage=state.coverage,
    )


async def _run_arm_c(
    challenge: str,
    domain: DomainConfig,
    roles: dict[str, Any],
    encoder: IdeaEncoder,
    router: LLMModelRouter,
    n_ideas: int,
    population: int,
    generations: int,
    budget_calls: int | None = None,
) -> BenchArmResult:
    """Brazo C: motor + Analista (su llamada cuenta en el presupuesto).

    `budget_calls` (consumo real de B) se adjunta solo para el informe —
    a diferencia de A, no se fuerza: C usa la misma población/generaciones
    que B, así que su coste ya es estructuralmente comparable sin
    necesidad de repetir rondas.
    """
    calls_before = router.total_calls
    tokens_before = router.total_tokens
    t0 = time.perf_counter()

    analyst = FunctionalAnalystAgent(roles["analyst"])
    profile = await analyst.analyze(challenge)

    engine = _fresh_qd_engine(roles, encoder, max_concurrent=5)
    state = await engine.run_evolution(
        EvolutionRequest(
            challenge=challenge,
            domain=domain.name,
            population_size=population,
            generations=generations,
            profile=profile,
        )
    )
    top = sorted(state.archive, key=lambda c: c.fitness, reverse=True)[:n_ideas]
    ideas = [c.elite for c in top]

    elapsed = time.perf_counter() - t0
    cost = _cost_delta(router, calls_before, tokens_before)
    blind = await judge_blind(roles["writer"], challenge, ideas)
    return _arm_metrics(
        "C_motor_analista", ideas, cost, elapsed, blind,
        qd_score=state.qd_score, coverage=state.coverage, budget_calls=budget_calls,
    )


async def run_single_challenge(
    challenge: str,
    reto_tipo: str,
    repetition: int,
    domain: DomainConfig,
    roles: dict[str, Any],
    router: LLMModelRouter,
    set_config: BenchSetConfig,
    encoder: IdeaEncoder,
) -> BenchChallengeResult:
    """Ejecuta los 3 brazos, mismo reto y misma repetición.

    B corre PRIMERO: su consumo real de llamadas LLM es la referencia de
    presupuesto para A (Fase 5, bloque 1) — sin esto A gastaba una
    fracción mínima de lo que gasta el motor y la comparación era inválida.
    """
    arm_b = await _run_arm_b(
        challenge, domain, roles, encoder, router,
        set_config.ideas_por_brazo, set_config.poblacion_motor, set_config.generaciones_motor,
    )
    arm_a = await _run_arm_a(
        challenge, domain, roles, encoder, router,
        set_config.ideas_por_brazo, budget_calls=arm_b.cost.calls,
    )
    arm_c = await _run_arm_c(
        challenge, domain, roles, encoder, router,
        set_config.ideas_por_brazo, set_config.poblacion_motor, set_config.generaciones_motor,
        budget_calls=arm_b.cost.calls,
    )

    return BenchChallengeResult(
        challenge=challenge,
        reto_tipo=reto_tipo,
        repetition=repetition,
        arms={"A": arm_a, "B": arm_b, "C": arm_c},
    )


async def run_bench_set(
    set_config: BenchSetConfig,
    settings: Settings,
    repository: IdeaRepository | None = None,
) -> list[BenchChallengeResult]:
    """Ejecuta el set completo: cada reto x N repeticiones, 3 brazos cada vez.

    Con `repository`: cada reto/repetición se persiste en BD según se
    completa (no se acumula en memoria hasta el informe final) y el set
    es reanudable — si el proceso muere a mitad (p.ej. OOM), volver a
    llamar con el mismo `set_config.name` salta lo ya persistido en vez
    de repetirlo. El informe final se reconstruye leyendo de BD.

    Sin `repository`: comportamiento legado, todo en memoria y sin
    reanudación posible (no hay dónde reanudar desde sin persistencia).
    """
    router = build_router(settings)
    roles = role_llms(router)
    domain = settings.get_domain(DomainName(set_config.domain))
    # Instancia única: el modelo de embeddings se carga una sola vez para
    # todo el set, no por brazo/repetición.
    encoder = IdeaEncoder()

    done: set[tuple[str, int]] = set()
    if repository is not None:
        existing = await repository.get_bench_results(set_config.name)
        done = {(r["challenge"], r["repetition"]) for r in existing}
        if done:
            logger.info(
                "bench_set_resuming", set_name=set_config.name, already_done=len(done)
            )

    results: list[BenchChallengeResult] = []
    try:
        for reto in set_config.retos:
            for rep in range(set_config.repeticiones):
                if (reto.texto, rep) in done:
                    logger.info(
                        "bench_challenge_skipped_resumed",
                        challenge=reto.texto[:60],
                        repetition=rep,
                    )
                    continue

                result = await run_single_challenge(
                    challenge=reto.texto,
                    reto_tipo=reto.tipo,
                    repetition=rep,
                    domain=domain,
                    roles=roles,
                    router=router,
                    set_config=set_config,
                    encoder=encoder,
                )
                logger.info(
                    "bench_challenge_completed",
                    challenge=reto.texto[:60],
                    tipo=reto.tipo,
                    repetition=rep,
                )

                if repository is not None:
                    await repository.save_bench_result(
                        set_name=set_config.name,
                        challenge=result.challenge,
                        reto_tipo=result.reto_tipo,
                        repetition=result.repetition,
                        arms={k: v.to_dict() for k, v in result.arms.items()},
                    )
                    # Ya está en BD: no hace falta conservar las ideas
                    # completas de este reto en memoria hasta el informe.
                    del result
                else:
                    results.append(result)
    finally:
        await router.close_all()

    if repository is not None:
        rows = await repository.get_bench_results(set_config.name)
        return [_row_to_challenge_result(r) for r in rows]

    return results
