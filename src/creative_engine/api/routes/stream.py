"""Streaming SSE del progreso de una evolución en vivo.

El motor QD acepta un callback por generación. Aquí lo usamos para
empujar a una cola async, tras cada generación, el abanico de ideas
agrupado en familias, y lo servimos como Server-Sent Events. El panel
muestra así progreso real e ideas apareciendo, sin esperas muertas.

Eventos SSE emitidos:
  - open     : conexión establecida
  - start    : {total_generations}
  - progress : {generation, coverage, best_fitness, families[...]}
  - done     : resumen final {run_id, families[...], ...}
  - error    : {message}
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...core.models import EvolutionRequest
from ...evolution.clustering import group_into_families
from ..guardrails import enforce_evolution_rate_limit, enforce_request_budget

logger = structlog.get_logger(__name__)
router = APIRouter()


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _families_payload(cells: list) -> list[dict[str, Any]]:
    """Agrupa élites en familias y las serializa para el panel (sin jerga)."""
    elites = [cell.elite for cell in cells]
    families = group_into_families(elites)
    return [
        {
            "family_id": fam.family_id,
            "size": fam.size,
            "representative": {
                "id": fam.representative.id,
                "title": fam.representative.title,
                "description": fam.representative.description[:280],
                "fitness": round(fam.representative.fitness, 3),
                "novelty": (
                    round(fam.representative.evaluation.novelty, 3)
                    if fam.representative.evaluation
                    else None
                ),
                "advantages": fam.representative.advantages[:3],
            },
            "members": [{"id": m.id, "title": m.title} for m in fam.members],
        }
        for fam in families
    ]


@router.post(
    "/evolution/stream", dependencies=[Depends(enforce_evolution_rate_limit)]
)
async def stream_evolution(request_body: EvolutionRequest, request: Request) -> StreamingResponse:
    """Lanza una evolución y transmite su progreso en vivo por SSE.

    El run_id se pre-asigna y se envía en el evento `start`: si el cliente
    pierde la conexión (móvil, red inestable), el run CONTINÚA en el
    servidor y el cliente puede recuperar los resultados guardados vía
    GET /runs/{run_id}/families.
    """
    import uuid

    enforce_request_budget(request_body)

    from .evolution import _build_qd_engine

    run_id = f"run_{uuid.uuid4().hex}"
    request_body.run_id = run_id

    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    async def on_generation(generation: int, cells: list) -> None:
        await queue.put(
            (
                "progress",
                {"generation": generation, "families": _families_payload(cells)},
            )
        )

    engine = await _build_qd_engine(request, on_generation=on_generation)

    async def run_and_signal() -> None:
        try:
            from ...core.config import get_settings

            domain_cfg = get_settings().get_domain(request_body.domain)
            total = request_body.generations or domain_cfg.default_generations
            await queue.put(("start", {"total_generations": total, "run_id": run_id}))
            state = await engine.run_evolution(request_body)
            await queue.put(
                (
                    "done",
                    {
                        "run_id": state.run_id,
                        "status": state.status,
                        "generations": state.generation,
                        "total_ideas": len(state.all_ideas),
                        "families": _families_payload(state.archive),
                    },
                )
            )
        except Exception as e:
            logger.error("stream_evolution_failed", error=str(e))
            await queue.put(("error", {"message": str(e)}))
        finally:
            router = getattr(engine, "_llm_router", None)
            if router is not None:
                await router.close_all()
            await queue.put(("__end__", {}))

    async def event_generator() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(run_and_signal())
        disconnected = False
        try:
            yield _sse("open", {"status": "connected"})
            while True:
                if await request.is_disconnected():
                    disconnected = True
                    logger.info(
                        "stream_client_disconnected_run_continues", run_id=run_id
                    )
                    break
                try:
                    kind, data = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if kind == "__end__":
                    break
                yield _sse(kind, data)
        finally:
            # Si el cliente se desconectó, el run sigue en segundo plano:
            # run_and_signal termina solo, persiste en BD y cierra el router.
            if not disconnected and not task.done():
                task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
