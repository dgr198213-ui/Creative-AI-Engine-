"""Jerarquía de excepciones del Creative AI Engine."""

from __future__ import annotations

from typing import Any


class CreativeEngineError(Exception):
    """Base para todas las excepciones del motor."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


# ── LLM ──────────────────────────────────────────────────────────────


class LLMError(CreativeEngineError):
    """Error en la comunicación con un proveedor LLM."""


class LLMRateLimitError(LLMError):
    """Se excedió el rate limit del proveedor."""


class LLMAuthError(LLMError):
    """API key inválida o sin permisos en el proveedor."""


class LLMInvalidRequestError(LLMError):
    """El proveedor rechazó la petición (400 invalid_request_error).

    Típicamente un parámetro no soportado por ese modelo/API concreto
    (p.ej. `max_tokens` en vez de `max_completion_tokens`). No es
    reintentable contra el mismo proveedor: señala una incompatibilidad,
    no una indisponibilidad transitoria.
    """


class LLMResponseParseError(LLMError):
    """La respuesta del LLM no pudo parsearse al formato esperado."""


# ── Evolución ────────────────────────────────────────────────────────


class EvolutionError(CreativeEngineError):
    """Error en el motor evolutivo."""


class PopulationEmptyError(EvolutionError):
    """La población no contiene individuos válidos."""


class EncodingError(EvolutionError):
    """Error al codificar/decodificar una idea a formato numérico."""


class BehaviorDescriptorError(EvolutionError):
    """Error al calcular los descriptores de comportamiento."""


# ── Agentes ──────────────────────────────────────────────────────────


class AgentError(CreativeEngineError):
    """Error en la ejecución de un agente."""


class AgentTimeoutError(AgentError):
    """Un agente excedió su tiempo límite de ejecución."""


# ── Memoria ──────────────────────────────────────────────────────────


class EngineMemoryError(CreativeEngineError):
    """Error en el subsistema de memoria.

    No se llama `MemoryError` para no sombrear el builtin de Python.
    """


class IdeaNotFoundError(EngineMemoryError):
    """No se encontró una idea con el ID proporcionado."""


class GraphQueryError(EngineMemoryError):
    """Error en una consulta al Knowledge Graph."""


# ── Dominio ──────────────────────────────────────────────────────────


class DomainError(CreativeEngineError):
    """Error relacionado con la configuración de dominio."""
