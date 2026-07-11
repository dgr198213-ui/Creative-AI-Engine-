"""Clase base para todos los agentes del sistema."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from ..core.events import Event, EventType, get_event_bus
from ..core.models import Idea
from ..llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


@dataclass
class AgentResult:
    """Resultado estandarizado de un agente."""

    agent_name: str
    idea_id: str
    success: bool
    score: float | None = None
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0


class BaseAgent(ABC):
    """Agente abstracto del Creative AI Engine."""

    def __init__(
        self,
        name: str,
        llm: LLMProvider,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._name = name
        self._llm = llm
        self._timeout = timeout_seconds
        self._bus = get_event_bus()
        self._log = logger.bind(agent=name)

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def evaluate(self, idea: Idea, context: dict[str, Any] | None = None) -> AgentResult:
        """Evalúa una idea y retorna resultado con puntuación y feedback."""
        ...

    async def safe_evaluate(
        self, idea: Idea, context: dict[str, Any] | None = None
    ) -> AgentResult:
        """Evaluación con manejo de errores y timeout."""
        start = time.perf_counter()

        await self._bus.publish(
            Event(
                type=EventType.AGENT_INVOKED,
                data={"agent": self._name, "idea_id": idea.id},
                source=self._name,
            )
        )

        try:
            result = await asyncio.wait_for(self.evaluate(idea, context), timeout=self._timeout)
            result.latency_ms = (time.perf_counter() - start) * 1000
        except TimeoutError:
            result = AgentResult(
                agent_name=self._name,
                idea_id=idea.id,
                success=False,
                error=f"Timeout después de {self._timeout}s",
                latency_ms=self._timeout * 1000,
            )
            self._log.warning("agent_timeout", idea_id=idea.id)
        except Exception as e:
            result = AgentResult(
                agent_name=self._name,
                idea_id=idea.id,
                success=False,
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
            self._log.error("agent_error", idea_id=idea.id, error=str(e))

        await self._bus.publish(
            Event(
                type=EventType.AGENT_COMPLETED if result.success else EventType.AGENT_FAILED,
                data={
                    "agent": self._name,
                    "idea_id": idea.id,
                    "score": result.score,
                    "error": result.error,
                },
                source=self._name,
            )
        )

        return result
