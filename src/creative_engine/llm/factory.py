"""Construcción del router de modelos y los LLM por rol desde la config."""

from __future__ import annotations

import structlog

from ..core.config import Settings
from .provider import LLMProvider
from .router import LLMModelRouter, RoledLLM

logger = structlog.get_logger(__name__)


def build_router(settings: Settings) -> LLMModelRouter:
    """Instancia todos los proveedores definidos y el router con su enrutado."""
    if not settings.llm:
        raise ValueError("No hay proveedores LLM configurados (CREATIVE_LLM__*)")

    providers = {name: LLMProvider(cfg) for name, cfg in settings.llm.items()}
    return LLMModelRouter(providers=providers, routing=settings.routing())


def role_llms(router: LLMModelRouter) -> dict[str, RoledLLM]:
    """Devuelve las vistas por rol que consumen los agentes.

    Cada una se comporta como un LLMProvider pero enruta + failover según
    su rol. Si un rol no tiene cadena configurada, usa el proveedor default.
    """
    return {
        "generator": router.for_role("generator"),
        "evaluator": router.for_role("evaluator"),
        "writer": router.for_role("writer"),
    }
