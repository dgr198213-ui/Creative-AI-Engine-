"""Motor de cruce conceptual guiado por LLM.

Fusiona dos ideas padre en una descendiente que combine
los mejores elementos de ambas de forma coherente.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from ..core.events import Event, EventType, get_event_bus
from ..core.models import Idea, IdeaFeatures, IdeaStatus
from ..llm.provider import LLMProvider
from .mutation import parse_llm_json

logger = structlog.get_logger(__name__)

CROSSOVER_PROMPT = """Eres un experto en innovación que fusiona conceptos creativos.
Combina las siguientes dos ideas en UNA NUEVA idea coherente que
tome lo mejor de cada una. No es una mezcla simple: debe ser una
síntesis que genere algo mayor que la suma de sus partes.

=== IDEA A ===
Título: {title_a}
Descripción: {description_a}
Ventajas: {advantages_a}
Tecnologías: {technologies_a}
Mercados: {markets_a}

=== IDEA B ===
Título: {title_b}
Descripción: {description_b}
Ventajas: {advantages_b}
Tecnologías: {technologies_b}
Mercados: {markets_b}

Genera UNA nueva idea que combine elementos de A y B de forma innovadora.
La idea resultante debe ser viable y coherente por sí misma.

Responde SOLO en formato JSON válido:
{{
    "title": "título de la idea fusionada",
    "description": "descripción detallada de la idea fusionada",
    "advantages": ["ventaja1", "ventaja2", "ventaja3"],
    "limitations": ["limitación1"],
    "features": {{
        "technologies": ["tech_combinada1", "tech_combinada2"],
        "target_markets": ["mercado_combinado1"],
        "materials": ["material1"]
    }},
    "fusion_logic": "explicación breve de cómo se combinaron los conceptos"
}}"""


class CrossoverEngine:
    """Motor de cruce conceptual entre dos ideas usando LLM."""

    def __init__(self, llm: LLMProvider, max_concurrent: int = 5) -> None:
        self._llm = llm
        self._bus = get_event_bus()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._log = logger.bind(component="CrossoverEngine")

    async def crossover(self, parent_a: Idea, parent_b: Idea) -> Idea:
        """Fusiona dos ideas para generar una nueva."""
        prompt = CROSSOVER_PROMPT.format(
            title_a=parent_a.title,
            description_a=parent_a.description,
            advantages_a="\n".join(f"- {a}" for a in parent_a.advantages) or "—",
            technologies_a=", ".join(parent_a.features.technologies) or "No especificadas",
            markets_a=", ".join(parent_a.features.target_markets) or "No especificados",
            title_b=parent_b.title,
            description_b=parent_b.description,
            advantages_b="\n".join(f"- {a}" for a in parent_b.advantages) or "—",
            technologies_b=", ".join(parent_b.features.technologies) or "No especificadas",
            markets_b=", ".join(parent_b.features.target_markets) or "No especificados",
        )

        raw_response = await self._llm.generate(prompt)
        data = parse_llm_json(raw_response)

        child = self._build_child(parent_a, parent_b, data)

        await self._bus.publish(
            Event(
                type=EventType.IDEA_CROSSED,
                data={
                    "parent_a": parent_a.id,
                    "parent_b": parent_b.id,
                    "child": child.id,
                    "logic": str(data.get("fusion_logic", ""))[:200],
                },
                source="CrossoverEngine",
            )
        )

        self._log.info(
            "crossover_completed",
            parent_a=parent_a.id,
            parent_b=parent_b.id,
            child=child.id,
        )

        return child

    async def batch_crossover(self, pairs: list[tuple[Idea, Idea]]) -> list[Idea]:
        """Cruce concurrente en lote."""

        async def _one(a: Idea, b: Idea) -> Idea | None:
            async with self._semaphore:
                try:
                    return await self.crossover(a, b)
                except Exception as e:
                    self._log.warning(
                        "batch_crossover_skipped",
                        parent_a=a.id,
                        parent_b=b.id,
                        error=str(e),
                    )
                    return None

        results = await asyncio.gather(*(_one(a, b) for a, b in pairs))
        return [r for r in results if r is not None]

    def _build_child(self, parent_a: Idea, parent_b: Idea, data: dict[str, Any]) -> Idea:
        features = IdeaFeatures()
        feat_data = data.get("features")
        if isinstance(feat_data, dict):
            features.technologies = list(feat_data.get("technologies", []))
            features.materials = list(feat_data.get("materials", []))
            features.target_markets = list(feat_data.get("target_markets", []))
            features.complexity_level = float(feat_data.get("complexity_level", 0.5))

        max_gen = max(parent_a.generation, parent_b.generation)

        return Idea(
            title=data["title"],
            description=data["description"],
            advantages=data.get("advantages", []),
            limitations=data.get("limitations", []),
            features=features,
            status=IdeaStatus.CROSSED,
            generation=max_gen + 1,
            run_id=parent_a.run_id or parent_b.run_id,
            parent_ids=[parent_a.id, parent_b.id],
            domain=parent_a.domain,
        )
