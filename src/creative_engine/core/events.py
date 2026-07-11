"""Bus de eventos internos para desacoplar subsistemas."""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class EventType(StrEnum):
    # Ciclo evolutivo
    EVOLUTION_STARTED = "evolution.started"
    EVOLUTION_GENERATION_COMPLETED = "evolution.generation_completed"
    EVOLUTION_COMPLETED = "evolution.completed"
    EVOLUTION_FAILED = "evolution.failed"

    # Ideas
    IDEA_GENERATED = "idea.generated"
    IDEA_MUTATED = "idea.mutated"
    IDEA_CROSSED = "idea.crossed"
    IDEA_EVALUATED = "idea.evaluated"
    IDEA_ELITE_SELECTED = "idea.elite_selected"
    IDEA_DISCARDED = "idea.discarded"

    # Agentes
    AGENT_INVOKED = "agent.invoked"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    # Memoria
    IDEA_STORED = "memory.idea_stored"
    GRAPH_UPDATED = "memory.graph_updated"


@dataclass(frozen=True)
class Event:
    """Evento inmutable que circula por el bus."""

    type: EventType
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "event_type": self.type.value,
            "source": self.source,
            "ts": self.timestamp.isoformat(),
            **self.data,
        }


EventHandler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Bus de eventos simple con soporte sync/async."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        logger.debug("event_published", **event.to_log())
        handlers = list(self._handlers.get(event.type, [])) + list(self._global_handlers)
        for handler in handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "event_handler_failed",
                    event=event.type.value,
                    handler=getattr(handler, "__name__", repr(handler)),
                )

    async def publish_many(self, events: list[Event]) -> None:
        for event in events:
            await self.publish(event)

    def clear(self) -> None:
        self._handlers.clear()
        self._global_handlers.clear()


_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
