"""Motor de mutación guiada por LLM.

A diferencia de la mutación aleatoria clásica, usa un LLM para
modificar aspectos específicos de una idea de forma coherente,
siguiendo el enfoque de "Evolutionary Thoughts" (arXiv:2505.05756).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

import structlog

from ..core.events import Event, EventType, get_event_bus
from ..core.exceptions import LLMError, LLMResponseParseError
from ..core.models import Idea, IdeaStatus, MutationType
from ..llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

MUTATION_PROMPTS: dict[MutationType, str] = {
    MutationType.FUNCTIONALITY: """Modifica la funcionalidad principal de la siguiente idea creativa.
Cambia o añade una funcionalidad que la haga más útil o interesante.
Mantén la esencia pero transforma su uso.

Idea original:
Título: {title}
Descripción: {description}
Ventajas: {advantages}
Limitaciones: {limitations}

Responde SOLO en formato JSON válido:
{{
    "title": "nuevo título",
    "description": "nueva descripción detallada",
    "advantages": ["ventaja1", "ventaja2"],
    "limitations": ["limitación1"],
    "mutation_description": "qué cambió y por qué"
}}""",
    MutationType.TECHNOLOGY: """Sustituye la tecnología principal de esta idea por una alternativa
emergente o diferente. Explica cómo la nueva tecnología cambia
las posibilidades del concepto.

Idea original:
Título: {title}
Descripción: {description}
Tecnologías actuales: {technologies}

Responde SOLO en formato JSON válido:
{{
    "title": "nuevo título",
    "description": "nueva descripción detallada",
    "advantages": ["ventaja1", "ventaja2"],
    "limitations": ["limitación1"],
    "features": {{"technologies": ["nueva_tech1", "nueva_tech2"]}},
    "mutation_description": "qué tecnología se aplicó y qué habilita"
}}""",
    MutationType.MATERIAL: """Cambia los materiales de esta idea por alternativas más sostenibles,
baratas, avanzadas o inusuales. Explica el impacto del cambio.

Idea original:
Título: {title}
Descripción: {description}
Materiales actuales: {materials}

Responde SOLO en formato JSON válido:
{{
    "title": "nuevo título",
    "description": "nueva descripción detallada",
    "advantages": ["ventaja1", "ventaja2"],
    "limitations": ["limitación1"],
    "features": {{"materials": ["nuevo_material1", "nuevo_material2"]}},
    "mutation_description": "qué materiales se cambiaron y su efecto"
}}""",
    MutationType.PROCESS: """Rediseña el proceso de fabricación, entrega o uso de esta idea.
Simplifícalo, automatízalo o inviértelo por completo.

Idea original:
Título: {title}
Descripción: {description}

Responde SOLO en formato JSON válido:
{{
    "title": "nuevo título",
    "description": "nueva descripción detallada",
    "advantages": ["ventaja1", "ventaja2"],
    "limitations": ["limitación1"],
    "mutation_description": "qué proceso se cambió y cómo"
}}""",
    MutationType.TARGET_MARKET: """Reorienta esta idea hacia un mercado objetivo completamente diferente.
Piensa en usuarios inesperados que podrían beneficiarse.

Idea original:
Título: {title}
Descripción: {description}
Mercados actuales: {markets}

Responde SOLO en formato JSON válido:
{{
    "title": "nuevo título",
    "description": "nueva descripción detallada",
    "advantages": ["ventaja1", "ventaja2"],
    "limitations": ["limitación1"],
    "features": {{"target_markets": ["nuevo_mercado1", "nuevo_mercado2"]}},
    "mutation_description": "a qué mercado se reorientó y por qué tiene sentido"
}}""",
    MutationType.BUSINESS_MODEL: """Transforma el modelo de negocio de esta idea.
Si era venta directa, hazlo suscripción. Si era premium, hazlo freemium.
Invierte la lógica económica.

Idea original:
Título: {title}
Descripción: {description}

Responde SOLO en formato JSON válido:
{{
    "title": "nuevo título",
    "description": "nueva descripción detallada con el nuevo modelo de negocio",
    "advantages": ["ventaja1", "ventaja2"],
    "limitations": ["limitación1"],
    "mutation_description": "qué modelo de negocio se aplicó"
}}""",
    MutationType.HYBRID: """Realiza una mutación combinada de esta idea. Cambia simultáneamente
dos o más aspectos (funcionalidad + tecnología, o mercado + proceso).
Sé audaz pero coherente.

Idea original:
Título: {title}
Descripción: {description}
Ventajas: {advantages}
Limitaciones: {limitations}
Tecnologías: {technologies}
Mercados: {markets}

Responde SOLO en formato JSON válido:
{{
    "title": "nuevo título",
    "description": "nueva descripción detallada",
    "advantages": ["ventaja1", "ventaja2", "ventaja3"],
    "limitations": ["limitación1"],
    "features": {{
        "technologies": ["tech1"],
        "target_markets": ["mercado1"],
        "materials": ["material1"]
    }},
    "mutation_description": "qué aspectos combinados se cambiaron"
}}""",
}


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Extrae y parsea un objeto JSON de una respuesta LLM (robusto a markdown)."""
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if json_match:
        raw = json_match.group(1).strip()

    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        raw = brace_match.group(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMResponseParseError(
            f"No se pudo parsear JSON del LLM: {e}",
            details={"raw_preview": raw[:300]},
        ) from e

    if not isinstance(data, dict):
        raise LLMResponseParseError(
            "La respuesta JSON no es un objeto",
            details={"type": type(data).__name__},
        )
    return data


class MutationEngine:
    """Motor de mutación guiada por LLM."""

    def __init__(
        self,
        llm: LLMProvider,
        allowed_mutations: list[MutationType] | None = None,
        max_concurrent: int = 5,
    ) -> None:
        self._llm = llm
        self._allowed = allowed_mutations or list(MutationType)
        self._bus = get_event_bus()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._log = logger.bind(component="MutationEngine")

    async def mutate(
        self,
        idea: Idea,
        mutation_type: MutationType | None = None,
    ) -> Idea:
        """Transforma una idea usando el LLM. Devuelve la hija (sin evaluar)."""
        if mutation_type is None or mutation_type not in self._allowed:
            mutation_type = random.choice(self._allowed)

        prompt = self._build_prompt(idea, mutation_type)

        try:
            raw_response = await self._llm.generate(prompt)
        except LLMError as e:
            self._log.error("mutation_llm_failed", idea_id=idea.id, error=str(e))
            raise

        data = parse_llm_json(raw_response)

        missing = {"title", "description"} - set(data.keys())
        if missing:
            raise LLMResponseParseError(
                f"Campos requeridos faltantes en mutación: {missing}",
                details={"received_keys": list(data.keys())},
            )

        mutated = self._build_mutated_idea(idea, data, mutation_type)

        await self._bus.publish(
            Event(
                type=EventType.IDEA_MUTATED,
                data={
                    "parent_id": idea.id,
                    "child_id": mutated.id,
                    "mutation_type": mutation_type.value,
                },
                source="MutationEngine",
            )
        )

        self._log.info(
            "mutation_completed",
            parent_id=idea.id,
            child_id=mutated.id,
            mutation_type=mutation_type.value,
        )

        return mutated

    async def batch_mutate(
        self,
        ideas: list[Idea],
        mutation_types: list[MutationType | None] | None = None,
    ) -> list[Idea]:
        """Mutación concurrente en lote (limitada por semáforo)."""
        if mutation_types is None:
            mutation_types = [None] * len(ideas)

        async def _one(idea: Idea, mtype: MutationType | None) -> Idea | None:
            async with self._semaphore:
                try:
                    return await self.mutate(idea, mtype)
                except (LLMError, LLMResponseParseError) as e:
                    self._log.warning("batch_mutate_skipped", idea_id=idea.id, error=str(e))
                    return None

        results = await asyncio.gather(
            *(_one(i, m) for i, m in zip(ideas, mutation_types, strict=False))
        )
        return [r for r in results if r is not None]

    def _build_prompt(self, idea: Idea, mutation_type: MutationType) -> str:
        template = MUTATION_PROMPTS.get(mutation_type, MUTATION_PROMPTS[MutationType.HYBRID])
        return template.format(
            title=idea.title,
            description=idea.description,
            advantages="\n".join(f"- {a}" for a in idea.advantages) or "Ninguna especificada",
            limitations="\n".join(f"- {li}" for li in idea.limitations) or "Ninguna especificada",
            technologies=", ".join(idea.features.technologies) or "No especificadas",
            materials=", ".join(idea.features.materials) or "No especificados",
            markets=", ".join(idea.features.target_markets) or "No especificados",
        )

    def _build_mutated_idea(
        self,
        parent: Idea,
        data: dict[str, Any],
        mutation_type: MutationType,
    ) -> Idea:
        features = parent.features.model_copy()
        feat_data = data.get("features")
        if isinstance(feat_data, dict):
            if "technologies" in feat_data:
                features.technologies = list(feat_data["technologies"])
            if "materials" in feat_data:
                features.materials = list(feat_data["materials"])
            if "target_markets" in feat_data:
                features.target_markets = list(feat_data["target_markets"])
            if "complexity_level" in feat_data:
                features.complexity_level = float(feat_data["complexity_level"])

        return Idea(
            title=data["title"],
            description=data["description"],
            advantages=data.get("advantages", parent.advantages),
            limitations=data.get("limitations", parent.limitations),
            features=features,
            value_hypothesis=parent.value_hypothesis,
            status=IdeaStatus.MUTATED,
            generation=parent.generation + 1,
            run_id=parent.run_id,
            parent_ids=[parent.id],
            mutation_type=mutation_type,
            domain=parent.domain,
        )
