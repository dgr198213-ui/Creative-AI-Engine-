"""Motor QD (Quality-Diversity) principal.

Orquesta el ciclo completo de evolución. Flujo por generación:

1. Seleccionar élites del archivo → mutar + cruzar
2. Inyectar ideas nuevas del generador (diversidad)
3. Evaluar CALIDAD con agentes LLM (utilidad, viabilidad, mercado)
4. Codificar: embedding (genoma) + descriptor semántico proyectado
5. Calcular NOVEDAD objetiva contra el archivo (distancia de embedding)
6. Intentar insertar en MAP-Elites
7. Persistir en el repositorio y publicar métricas
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

from ..agents.evaluator_orchestrator import EvaluatorOrchestrator
from ..agents.generator import IdeaGeneratorAgent
from ..core.config import get_settings
from ..core.events import Event, EventType, get_event_bus
from ..core.exceptions import EvolutionError
from ..core.models import (
    DomainConfig,
    EvolutionRequest,
    EvolutionState,
    Idea,
    IdeaStatus,
)
from ..memory.repository import IdeaRepository
from .crossover import CrossoverEngine
from .encoders import IdeaEncoder, objective_novelty
from .map_elites import MAPElitesArchive
from .mutation import MutationEngine

logger = structlog.get_logger(__name__)


class QDEngine:
    """Motor Quality-Diversity que orquesta el ciclo evolutivo completo."""

    def __init__(
        self,
        generator: IdeaGeneratorAgent,
        evaluator: EvaluatorOrchestrator,
        mutation: MutationEngine,
        crossover: CrossoverEngine,
        encoder: IdeaEncoder,
        repository: IdeaRepository | None = None,
    ) -> None:
        self._generator = generator
        self._evaluator = evaluator
        self._mutation = mutation
        self._crossover = crossover
        self._encoder = encoder
        self._repository = repository
        self._bus = get_event_bus()
        self._settings = get_settings()
        self._rng = np.random.default_rng()
        self._log = logger.bind(component="QDEngine")

    async def run_evolution(self, request: EvolutionRequest) -> EvolutionState:
        """Ejecuta el ciclo evolutivo completo y devuelve el estado final."""
        domain = self._settings.get_domain(request.domain)
        if request.custom_weights:
            domain = domain.model_copy(update={"evaluation_weights": request.custom_weights})

        state = EvolutionState(
            challenge=request.challenge,
            domain=request.domain,
            total_generations=request.generations or domain.default_generations,
            population_size=request.population_size or domain.default_population_size,
            is_running=True,
        )

        archive = MAPElitesArchive(
            grid_shape=domain.grid_shape,
            dimension_names=[d.name for d in domain.behavior_dimensions],
        )

        self._log.info(
            "evolution_started",
            run_id=state.run_id,
            challenge=request.challenge[:80],
            domain=request.domain.value,
            grid_shape=domain.grid_shape,
            generations=state.total_generations,
            pop_size=state.population_size,
        )

        await self._bus.publish(
            Event(
                type=EventType.EVOLUTION_STARTED,
                data={
                    "run_id": state.run_id,
                    "challenge": request.challenge[:100],
                    "domain": request.domain.value,
                },
                source="QDEngine",
            )
        )

        context: dict[str, Any] = {
            "challenge": request.challenge,
            "domain": request.domain,
            "custom_weights": request.custom_weights,
            "run_id": state.run_id,
        }

        try:
            # ── Generación 0: población inicial ──
            initial_ideas = await self._generator.generate_population(
                challenge=request.challenge,
                domain=domain,
                count=state.population_size,
            )
            for idea in initial_ideas:
                idea.run_id = state.run_id

            await self._process_batch(initial_ideas, archive, domain, context, state)

            # ── Generaciones 1..N ──
            for gen in range(1, state.total_generations + 1):
                gen_start = time.perf_counter()
                state.generation = gen

                new_ideas = await self._create_generation(
                    archive=archive,
                    domain=domain,
                    context=context,
                    pop_size=state.population_size,
                )
                await self._process_batch(new_ideas, archive, domain, context, state)

                gen_time = time.perf_counter() - gen_start

                self._log.info(
                    "generation_completed",
                    run_id=state.run_id,
                    generation=gen,
                    new_ideas=len(new_ideas),
                    coverage=round(archive.coverage, 4),
                    qd_score=round(archive.qd_score, 3),
                    best_fitness=round(archive.best_fitness, 4),
                    elapsed_s=round(gen_time, 2),
                )

                await self._bus.publish(
                    Event(
                        type=EventType.EVOLUTION_GENERATION_COMPLETED,
                        data={
                            "run_id": state.run_id,
                            "generation": gen,
                            "coverage": archive.coverage,
                            "qd_score": archive.qd_score,
                            "best_fitness": archive.best_fitness,
                        },
                        source="QDEngine",
                    )
                )

                if gen_time > self._settings.evolution.max_generation_time_seconds:
                    self._log.warning("generation_timeout", gen=gen, elapsed=gen_time)
                    break

            # ── Finalización ──
            state.archive = archive.occupied_cells
            state.coverage = archive.coverage
            state.qd_score = archive.qd_score
            state.best_fitness = archive.best_fitness
            state.is_running = False
            state.completed_at = datetime.now(UTC)

            await self._bus.publish(
                Event(
                    type=EventType.EVOLUTION_COMPLETED,
                    data={
                        "run_id": state.run_id,
                        "coverage": state.coverage,
                        "qd_score": state.qd_score,
                        "elite_count": len(state.archive),
                    },
                    source="QDEngine",
                )
            )

            self._log.info(
                "evolution_completed",
                run_id=state.run_id,
                generations=state.generation,
                elites=len(state.archive),
                coverage=round(state.coverage, 4),
            )

            return state

        except Exception as e:
            state.is_running = False
            await self._bus.publish(
                Event(
                    type=EventType.EVOLUTION_FAILED,
                    data={"run_id": state.run_id, "error": str(e)},
                    source="QDEngine",
                )
            )
            self._log.error("evolution_failed", run_id=state.run_id, error=str(e))
            raise EvolutionError(f"Evolución fallida: {e}") from e

    async def _create_generation(
        self,
        archive: MAPElitesArchive,
        domain: DomainConfig,
        context: dict[str, Any],
        pop_size: int,
    ) -> list[Idea]:
        """Crea una nueva generación de ideas a partir del archivo."""
        cfg = self._settings.evolution
        new_ideas: list[Idea] = []
        occupied = len(archive.occupied_cells)

        n_mutations = int(pop_size * cfg.mutation_rate)
        n_crossovers = int(pop_size * cfg.crossover_rate)
        n_random = max(1, int(pop_size * cfg.random_injection_rate))

        # ── Mutaciones ──
        if n_mutations > 0 and occupied > 0:
            parents = archive.select_for_mutation(n_mutations, self._rng)
            new_ideas.extend(await self._mutation.batch_mutate(parents))

        # ── Cruces ──
        if n_crossovers > 0 and occupied >= 2:
            pairs: list[tuple[Idea, Idea]] = []
            for _ in range(n_crossovers):
                pair = archive.select_pair_for_crossover(self._rng)
                if pair:
                    pairs.append(pair)
            if pairs:
                new_ideas.extend(await self._crossover.batch_crossover(pairs))

        # ── Inyección de ideas frescas (diversidad) ──
        if n_random > 0:
            random_ideas = await self._generator.generate_population(
                challenge=context["challenge"],
                domain=domain,
                count=n_random,
                variation_hint="Explora un ángulo completamente diferente e inesperado.",
            )
            new_ideas.extend(random_ideas)

        for idea in new_ideas:
            idea.run_id = context.get("run_id", "")

        return new_ideas

    async def _process_batch(
        self,
        ideas: list[Idea],
        archive: MAPElitesArchive,
        domain: DomainConfig,
        context: dict[str, Any],
        state: EvolutionState,
    ) -> None:
        """Evalúa, codifica, calcula novedad objetiva e inserta un lote."""
        if not ideas:
            return

        # 1. Evaluación de calidad en paralelo (agentes LLM)
        await asyncio.gather(
            *(self._evaluator.evaluate_idea(idea, context) for idea in ideas),
            return_exceptions=True,
        )

        # 2. Codificar + novedad objetiva + insertar
        inserted_count = 0
        k = self._settings.evolution.novelty_k_nearest

        for idea in ideas:
            if idea.evaluation is None:
                idea.status = IdeaStatus.DISCARDED
                continue

            state.all_ideas.append(idea.id)

            try:
                self._encoder.encode_idea(idea, domain)

                # Novedad objetiva: distancia al archivo ANTES de insertarla
                idea.evaluation.novelty = objective_novelty(
                    idea.genome_vector, archive.elite_genomes(), k=k
                )

                if archive.try_insert(idea):
                    idea.status = IdeaStatus.ELITE
                    inserted_count += 1
                else:
                    idea.status = IdeaStatus.DISCARDED

                if self._repository is not None:
                    await self._repository.store_idea(idea)

            except Exception as e:
                self._log.warning("idea_processing_failed", idea_id=idea.id, error=str(e))
                idea.status = IdeaStatus.DISCARDED

        self._log.debug(
            "batch_processed",
            total=len(ideas),
            inserted=inserted_count,
            discarded=len(ideas) - inserted_count,
        )
