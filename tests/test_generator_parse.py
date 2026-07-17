"""Tests del parser de lotes del generador (rescate de JSON truncado)."""

import json
from unittest.mock import AsyncMock

from creative_engine.agents.generator import IdeaGeneratorAgent
from creative_engine.core.config import default_generic_domain


def _agent() -> IdeaGeneratorAgent:
    return IdeaGeneratorAgent(AsyncMock())


def _item(i: int) -> dict:
    return {
        "title": f"Idea número {i}",
        "description": f"Descripción suficientemente larga de la idea {i}.",
        "advantages": ["A"],
        "limitations": ["X"],
        "features": {"complexity_level": 0.5},
    }


class TestParseBatch:
    def test_valid_array(self) -> None:
        raw = json.dumps([_item(1), _item(2)])
        ideas = _agent()._parse_batch(raw, default_generic_domain())
        assert len(ideas) == 2

    def test_truncated_array_salvages_complete_items(self) -> None:
        """Simula el fallo de producción: respuesta cortada a mitad de un objeto."""
        full = json.dumps([_item(1), _item(2), _item(3)])
        truncated = full[: int(len(full) * 0.75)]  # corta el tercer objeto
        ideas = _agent()._parse_batch(truncated, default_generic_domain())
        assert len(ideas) == 2  # rescata las dos completas
        assert ideas[0].title == "Idea número 1"

    def test_truncated_mid_first_item_returns_empty(self) -> None:
        full = json.dumps([_item(1)])
        truncated = full[:20]  # ni un objeto completo
        ideas = _agent()._parse_batch(truncated, default_generic_domain())
        assert ideas == []

    def test_markdown_wrapped_array(self) -> None:
        raw = "```json\n" + json.dumps([_item(1)]) + "\n```"
        ideas = _agent()._parse_batch(raw, default_generic_domain())
        assert len(ideas) == 1

    def test_no_array_at_all(self) -> None:
        assert _agent()._parse_batch("no hay json aquí", default_generic_domain()) == []
