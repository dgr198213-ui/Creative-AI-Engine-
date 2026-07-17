"""Benchmark: motor QD vs. prompt único ("modo ChatGPT").

Responde la pregunta existencial del proyecto de forma medible:
¿el abanico de ideas del motor evolutivo es mejor que el de un buen
prompt único pidiendo N ideas diversas?

Dos brazos con el MISMO reto, mismo codificador y mismos evaluadores:

- baseline : una pasada del generador pidiendo N ideas diversas
             (equivale a un usuario hábil de ChatGPT con un buen prompt)
- engine   : evolución QD completa; se comparan sus N mejores élites

Métricas:
- Diversidad semántica: distancia coseno media y mínima entre pares
  (embeddings). La mínima detecta "clones": dos ideas casi iguales.
- Cobertura: celdas distintas ocupadas en el grid de comportamiento.
- Calidad (opcional, cuesta llamadas LLM): fitness medio y máximo con
  los mismos agentes evaluadores para ambos brazos.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import structlog

from .agents.evaluator_orchestrator import EvaluatorOrchestrator
from .agents.generator import IdeaGeneratorAgent
from .core.models import DomainConfig, EvolutionRequest, Idea
from .evolution.encoders import IdeaEncoder
from .evolution.map_elites import MAPElitesArchive
from .evolution.qd_engine import QDEngine

logger = structlog.get_logger(__name__)


@dataclass
class ArmMetrics:
    """Métricas de un brazo del benchmark."""

    name: str
    n_ideas: int
    mean_pairwise_distance: float
    min_pairwise_distance: float
    distinct_cells: int
    mean_fitness: float | None
    best_fitness: float | None
    elapsed_s: float
    titles: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    challenge: str
    baseline: ArmMetrics
    engine: ArmMetrics
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenge": self.challenge,
            "baseline": self.baseline.to_dict(),
            "engine": self.engine.to_dict(),
            "verdict": self.verdict,
        }


def pairwise_diversity(ideas: list[Idea]) -> tuple[float, float]:
    """Distancia coseno (media, mínima) entre todos los pares de ideas.

    Rango [0,1]. Media alta = abanico amplio; mínima baja = hay clones.
    """
    genomes = [i.genome_vector for i in ideas if i.genome_vector]
    if len(genomes) < 2:
        return 0.0, 0.0

    g = np.asarray(genomes, dtype=np.float64)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    g = g / norms

    sims = g @ g.T
    dists = (1.0 - sims) / 2.0
    iu = np.triu_indices(len(genomes), k=1)
    values = dists[iu]
    return float(values.mean()), float(values.min())


def distinct_cells(ideas: list[Idea], domain: DomainConfig) -> int:
    """Celdas distintas del grid de comportamiento que ocupan las ideas."""
    archive = MAPElitesArchive(grid_shape=domain.grid_shape)
    cells = set()
    for idea in ideas:
        if idea.behavior_descriptor:
            try:
                cells.add(archive.discretize(idea.behavior_descriptor))
            except Exception:
                continue
    return len(cells)


def _arm_metrics(
    name: str,
    ideas: list[Idea],
    domain: DomainConfig,
    elapsed_s: float,
) -> ArmMetrics:
    mean_d, min_d = pairwise_diversity(ideas)
    fitnesses = [i.fitness for i in ideas if i.evaluation is not None]
    return ArmMetrics(
        name=name,
        n_ideas=len(ideas),
        mean_pairwise_distance=round(mean_d, 4),
        min_pairwise_distance=round(min_d, 4),
        distinct_cells=distinct_cells(ideas, domain),
        mean_fitness=round(float(np.mean(fitnesses)), 4) if fitnesses else None,
        best_fitness=round(max(fitnesses), 4) if fitnesses else None,
        elapsed_s=round(elapsed_s, 1),
        titles=[i.title for i in ideas],
    )


def _verdict(baseline: ArmMetrics, engine: ArmMetrics) -> str:
    """Veredicto simple y honesto a partir de las métricas."""
    parts: list[str] = []

    d_gain = engine.mean_pairwise_distance - baseline.mean_pairwise_distance
    if d_gain > 0.01:
        parts.append(f"el motor amplía la diversidad media (+{d_gain:.3f})")
    elif d_gain < -0.01:
        parts.append(f"el prompt único fue más diverso ({d_gain:.3f})")
    else:
        parts.append("diversidad media pareja")

    if engine.distinct_cells > baseline.distinct_cells:
        parts.append(
            f"cubre más regiones del espacio ({engine.distinct_cells} vs "
            f"{baseline.distinct_cells} celdas)"
        )
    elif engine.distinct_cells < baseline.distinct_cells:
        parts.append(
            f"cubre menos regiones ({engine.distinct_cells} vs "
            f"{baseline.distinct_cells})"
        )

    if engine.mean_fitness is not None and baseline.mean_fitness is not None:
        q_gain = engine.mean_fitness - baseline.mean_fitness
        if q_gain > 0.02:
            parts.append(f"y mejora la calidad media (+{q_gain:.3f})")
        elif q_gain < -0.02:
            parts.append(f"pero pierde calidad media ({q_gain:.3f})")
        else:
            parts.append("con calidad media pareja")

    return "; ".join(parts).capitalize() + "."


async def run_benchmark(
    challenge: str,
    domain: DomainConfig,
    generator: IdeaGeneratorAgent,
    encoder: IdeaEncoder,
    engine: QDEngine,
    evaluator: EvaluatorOrchestrator | None = None,
    n_ideas: int = 12,
    population: int = 12,
    generations: int = 3,
) -> BenchmarkResult:
    """Ejecuta ambos brazos y devuelve las métricas comparadas.

    Si `evaluator` es None se omite la medición de calidad (solo diversidad:
    barato, sin coste extra de evaluación para el baseline).
    """
    context = {"challenge": challenge, "domain": domain.name}

    # ── Brazo 1: prompt único (baseline) ──
    t0 = time.perf_counter()
    baseline_ideas = await generator.generate_population(
        challenge=challenge, domain=domain, count=n_ideas
    )
    for idea in baseline_ideas:
        encoder.encode_idea(idea, domain)
    if evaluator is not None:
        for idea in baseline_ideas:
            try:
                await evaluator.evaluate_idea(idea, context)
            except Exception as e:
                logger.warning("baseline_eval_failed", idea_id=idea.id, error=str(e))
    baseline_elapsed = time.perf_counter() - t0

    # ── Brazo 2: motor QD ──
    t0 = time.perf_counter()
    state = await engine.run_evolution(
        EvolutionRequest(
            challenge=challenge,
            domain=domain.name,
            population_size=population,
            generations=generations,
        )
    )
    engine_elapsed = time.perf_counter() - t0

    top_elites = [
        c.elite
        for c in sorted(state.archive, key=lambda c: c.fitness, reverse=True)[:n_ideas]
    ]

    baseline_m = _arm_metrics("prompt_unico", baseline_ideas, domain, baseline_elapsed)
    engine_m = _arm_metrics("motor_qd", top_elites, domain, engine_elapsed)

    result = BenchmarkResult(
        challenge=challenge,
        baseline=baseline_m,
        engine=engine_m,
        verdict=_verdict(baseline_m, engine_m),
    )

    logger.info(
        "benchmark_completed",
        baseline_diversity=baseline_m.mean_pairwise_distance,
        engine_diversity=engine_m.mean_pairwise_distance,
        baseline_cells=baseline_m.distinct_cells,
        engine_cells=engine_m.distinct_cells,
    )

    return result
