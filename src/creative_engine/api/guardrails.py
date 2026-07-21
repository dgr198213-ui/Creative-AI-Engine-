"""Guardrails de la API pública (auditoría C2): tope de presupuesto por
request y rate limiting básico en los endpoints de evolución.

Ambos protegen el mismo recurso escaso: las llamadas LLM. Sin ellos, un
solo EvolutionRequest legítimo (population_size≤500, generations≤200)
puede pedir hasta 100.000 evaluaciones, y nada impide repetirlo sin
límite. Rate limiting en memoria por proceso: suficiente para el
despliegue actual (un solo dyno de Railway), sin depender de Redis.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from ..core.config import get_settings
from ..core.models import EvolutionRequest


class InMemoryRateLimiter:
    """Ventana deslizante por clave (IP). Una instancia por app."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, max_requests: int, window_seconds: float) -> bool:
        """True si la petición cabe en el límite; registra el intento."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > window_seconds:
            hits.popleft()
        if len(hits) >= max_requests:
            return False
        hits.append(now)
        return True


def enforce_evolution_rate_limit(request: Request) -> None:
    """Dependencia FastAPI: limita /evolution/* a N peticiones/ventana por IP."""
    cfg = get_settings().evolution
    limiter: InMemoryRateLimiter = request.app.state.rate_limiter
    key = request.client.host if request.client else "unknown"

    if not limiter.check(key, cfg.rate_limit_per_minute, cfg.rate_limit_window_seconds):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Límite de {cfg.rate_limit_per_minute} evoluciones por "
                f"{int(cfg.rate_limit_window_seconds)}s por IP excedido. "
                "Reintenta más tarde."
            ),
        )


def enforce_request_budget(request_body: EvolutionRequest) -> None:
    """422 si population_size x generations supera el máximo permitido.

    Usa los valores efectivos (los del request, o los del dominio si no
    se especifican) para no dejar pasar un ataque que omita un campo y
    confíe en el default del dominio para inflar el otro.
    """
    settings = get_settings()
    domain = settings.get_domain(request_body.domain)
    population = request_body.population_size or domain.default_population_size
    generations = request_body.generations or domain.default_generations
    requested = population * generations
    cap = settings.evolution.max_requested_evaluations

    if requested > cap:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Presupuesto solicitado excesivo: {population} ideas x "
                f"{generations} generaciones = {requested} evaluaciones "
                f"(máximo permitido: {cap}). Reduce population_size o generations."
            ),
        )
