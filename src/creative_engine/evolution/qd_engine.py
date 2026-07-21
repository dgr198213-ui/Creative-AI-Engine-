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

# Callback opcional invocado tras cada generación con (generation, cells).
# Permite a consumidores externos (p.ej. el stream SSE) obtener el abanico
# de familias en vivo sin acoplar el motor a la capa de transporte.
from collections.abc import Awaitable, Callable  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from ..core.models import MAPElitesCell

GenerationCallback = Callable[[int, "list[MAPElitesCell]"], Awaitable[None]]


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
        on_generation: GenerationCallback | None = None,
    ) -> None:
        self._generator = generator
        self._evaluator = evaluator
        self._mutation = mutation
        self._crossover = crossover
        self._encoder = encoder
        self._repository = repository
        self._on_generation = on_generation
        self._bus = get_event_bus()
        self._settings = get_settings()
        self._rng = np.random.default_rng()
        self._log = logger.bind(component="QDEngine")

        # Puerta de sorpresa (TurboEvolve-style): evita gastar evaluaciones
        # LLM en ideas semánticamente redundantes. Configurable por env.
        evo = self._settings.evolution
        if evo.surprise_gate_enabled:
            from .surprise import SurpriseGate

            self._surprise_gate: SurpriseGate | None = SurpriseGate(
                threshold=evo.surprise_threshold,
                min_threshold=evo.surprise_threshold_min,
                max_threshold=evo.surprise_threshold_max,
                step=evo.surprise_threshold_step,
            )
        else:
            self._surprise_gate = None

    async def run_evolution(self, request: EvolutionRequest) -> EvolutionState:
        """Ejecuta el ciclo evolutivo completo y devuelve el estado final."""
        domain = self._settings.get_domain(request.domain)
        if request.custom_weights:
            domain = domain.model_copy(update={"evaluation_weights": request.custom_weights})

        import structlog.contextvars as _ctx

        state_kwargs: dict[str, Any] = {}
        if request.run_id:
            state_kwargs["run_id"] = request.run_id
        state = EvolutionState(
            **state_kwargs,
            challenge=request.challenge,
            domain=request.domain,
            total_generations=request.generations or domain.default_generations,
            population_size=request.population_size or domain.default_population_size,
            is_running=True,
        )

        # run_id en TODOS los logs del run (mutaciones, cruces, proveedor,
        # puerta de sorpresa...) vía contextvars — correlacionable aunque
        # haya varias ejecuciones simultáneas.
        _ctx.bind_contextvars(run_id=state.run_id)

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
            # ── Memoria entre runs: grounding con élites de retos pasados ──
            memory_hint = await self._cross_run_memory_hint(request.challenge, state.run_id)

            # ── Generación 0: población inicial ──
            initial_ideas = await self._generator.generate_population(
                challenge=request.challenge,
                domain=domain,
                count=state.population_size,
                variation_hint=memory_hint or "",
            )
            for idea in initial_ideas:
                idea.run_id = state.run_id

            await self._process_batch(initial_ideas, archive, domain, context, state)

            # ── Generaciones 1..N ──
            for gen in range(1, state.total_generations + 1):
                gen_start = time.perf_counter()
                state.generation = gen

                cells_before = len(archive.occupied_cells)

                new_ideas = await self._create_generation(
                    archive=archive,
                    domain=domain,
                    context=context,
                    pop_size=state.population_size,
                )
                await self._process_batch(new_ideas, archive, domain, context, state)

                # Adaptación del umbral de sorpresa según el progreso:
                # estancamiento (sin celdas nuevas) → abrir la puerta;
                # progreso → cerrarla un poco y ahorrar presupuesto.
                if self._surprise_gate is not None:
                    stagnated = len(archive.occupied_cells) <= cells_before
                    self._surprise_gate.adapt(stagnated=stagnated)

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
                            "elite_count": len(archive.occupied_cells),
                        },
                        source="QDEngine",
                    )
                )

                if self._on_generation is not None:
                    try:
                        await self._on_generation(gen, archive.occupied_cells)
                    except Exception as cb_err:
                        self._log.warning("on_generation_callback_failed", error=str(cb_err))

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
        finally:
            _ctx.unbind_contextvars("run_id")

    async def _cross_run_memory_hint(self, challenge: str, run_id: str) -> str | None:
        """Hint de inspiración+repulsión con élites afines de runs pasados.

        Cálculo local (embeddings) sobre lo ya persistido: cero llamadas LLM.
        Degrada en silencio si no hay persistencia o la consulta falla.
        """
        cfg = self._settings.evolution
        if not cfg.cross_run_memory_enabled or self._repository is None:
            return None

        try:
            from .grounding import build_memory_hint, select_related_elites

            past = await self._repository.get_recent_elites(
                limit=200, exclude_run_id=run_id
            )
            if not past:
                return None

            challenge_vector = self._encoder._embed(challenge)
            related = select_related_elites(
                challenge_vector,
                past,
                k=cfg.cross_run_memory_k,
                min_similarity=cfg.cross_run_memory_min_similarity,
            )
            if related:
                self._log.info(
                    "cross_run_memory_applied",
                    related=[idea.title[:40] for idea in related],
                    past_pool=len(past),
                )
            return build_memory_hint(related)
        except Exception as e:
            self._log.warning("cross_run_memory_failed", error=str(e))
            return None

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

        # ── Mutaciones (selección por curiosidad: regiones poco exploradas) ──
        if n_mutations > 0 and occupied > 0:
            parents = archive.select_curious(n_mutations, self._rng)
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

        # ── Inyección de ideas frescas con repulsión (best-shot invertido) ──
        # Estilo FunSearch: mostrar al generador lo mejor que ya existe,
        # pero pidiendo explícitamente alejarse de ello → diversidad real
        # sin ninguna llamada LLM adicional.
        if n_random > 0:
            top_cells = sorted(
                archive.occupied_cells, key=lambda c: c.fitness, reverse=True
            )[:3]
            if top_cells:
                existing = "; ".join(f"«{c.elite.title}»" for c in top_cells)
                hint = (
                    f"Ya existen estos enfoques: {existing}. "
                    "Genera ideas RADICALMENTE distintas a esas: otro ángulo, "
                    "otro mecanismo, otro público."
                )
            else:
                hint = "Explora un ángulo completamente diferente e inesperado."
            random_ideas = await self._generator.generate_population(
                challenge=context["challenge"],
                domain=domain,
                count=n_random,
                variation_hint=hint,
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
        """Codifica, filtra por sorpresa, evalúa e inserta un lote.

        Orden clave: la codificación (embeddings) es local y gratis; la
        evaluación LLM es el recurso caro. Codificamos primero y solo
        evaluamos las ideas que aportan sorpresa semántica frente a las
        élites ya existentes (puerta adaptativa, estilo TurboEvolve).
        """
        if not ideas:
            return

        # 0. Codificar primero (local, sin coste LLM)
        encoded: list[Idea] = []
        for idea in ideas:
            try:
                self._encoder.encode_idea(idea, domain)
                encoded.append(idea)
            except Exception as e:
                self._log.warning("idea_encoding_failed", idea_id=idea.id, error=str(e))
                idea.status = IdeaStatus.DISCARDED

        # 1. Puerta de sorpresa: descartar sin evaluar lo ya explorado
        to_evaluate: list[Idea] = []
        if self._surprise_gate is not None:
            elite_genomes = archive.elite_genomes()
            skipped = 0
            for idea in encoded:
                if self._surprise_gate.is_surprising(idea.genome_vector, elite_genomes):
                    to_evaluate.append(idea)
                else:
                    idea.status = IdeaStatus.DISCARDED
                    skipped += 1
            if skipped:
                self._log.info(
                    "surprise_gate_skipped",
                    skipped=skipped,
                    threshold=round(self._surprise_gate.threshold, 3),
                    total_saved=self._surprise_gate.evaluations_saved,
                )
        else:
            to_evaluate = encoded

        if not to_evaluate:
            return

        # 2. Evaluación de calidad con concurrencia LIMITADA.
        # Evaluar todas las ideas a la vez satura los rate limits de los
        # proveedores (cada idea = 3 llamadas de agente), provocando timeouts
        # en cascada. Limitamos cuántas ideas se evalúan simultáneamente.
        max_parallel = self._settings.evolution.max_concurrent_evaluations
        semaphore = asyncio.Semaphore(max_parallel)

        async def _eval_one(idea: Idea) -> None:
            async with semaphore:
                await self._evaluator.evaluate_idea(idea, context)

        await asyncio.gather(
            *(_eval_one(idea) for idea in to_evaluate),
            return_exceptions=True,
        )

        # 3. Novedad objetiva + insertar
        inserted_count = 0
        k = self._settings.evolution.novelty_k_nearest

        for idea in to_evaluate:
            if idea.evaluation is None:
                idea.status = IdeaStatus.DISCARDED
                continue

            state.all_ideas.append(idea.id)

            try:
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
            evaluated=len(to_evaluate),
            inserted=inserted_count,
        )
