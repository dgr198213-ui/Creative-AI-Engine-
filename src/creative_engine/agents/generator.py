"""Agente Generador: crea poblaciones de ideas diversas."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from ..core.models import DomainConfig, Idea, IdeaFeatures, IdeaStatus, ValueHypothesis
from ..llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

GENERATION_PROMPT = """Genera {count} ideas creativas y diferentes para el siguiente reto:

RETO: {challenge}
DOMINIO: {domain_name}
{variation_hint}

Cada idea debe ser GENUINAMENTE DIFERENTE de las demás.
Explora ángulos opuestos, tecnologías dispares, mercados inesperados.

Responde SOLO en JSON (un array de objetos):
[
  {{
    "title": "título conciso",
    "description": "descripción detallada de 3-5 frases",
    "advantages": ["ventaja1", "ventaja2", "ventaja3"],
    "limitations": ["limitación1", "limitación2"],
    "value_hypothesis": {{
      "target_user": "quién se beneficiaría",
      "problem_solved": "qué problema resuelve",
      "value_proposition": "propuesta de valor única"
    }},
    "features": {{
      "technologies": ["tech1", "tech2"],
      "materials": ["mat1"],
      "target_markets": ["mercado1"],
      "complexity_level": 0.5
    }}
  }}
]"""

_VARIATION_ANGLES = [
    "Enfoque en minimalismo y simplicidad extrema",
    "Enfoque en máxima funcionalidad y capacidades",
    "Enfoque en sostenibilidad y materiales ecológicos",
    "Enfoque en bajo coste y accesibilidad masiva",
    "Enfoque en premium y lujo",
    "Enfoque en tecnología emergente (IA, IoT, biotech)",
    "Enfoque en modularidad y personalización",
    "Enfoque en compartir y economía colaborativa",
    "Enfoque en mercados emergentes y países en desarrollo",
    "Enfoque en seguridad y robustez extrema",
]


class IdeaGeneratorAgent:
    """Genera poblaciones iniciales de ideas creativas usando LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._log = logger.bind(agent="generator")

    async def generate_population(
        self,
        challenge: str,
        domain: DomainConfig,
        count: int = 20,
        variation_hint: str = "",
    ) -> list[Idea]:
        """Genera una población de ideas diversas, en lotes de máximo 10."""
        batch_size = min(count, 10)
        all_ideas: list[Idea] = []

        batches_needed = (count + batch_size - 1) // batch_size

        for i in range(batches_needed):
            remaining = count - len(all_ideas)
            if remaining <= 0:
                break
            batch_count = min(batch_size, remaining)
            batch_hint = self._get_variation_hint(variation_hint, i, batches_needed)

            try:
                batch = await self._generate_batch(
                    challenge=challenge,
                    domain=domain,
                    count=batch_count,
                    variation_hint=batch_hint,
                )
                all_ideas.extend(batch)
            except Exception as e:
                self._log.warning("generation_batch_failed", batch=i, error=str(e))

        self._log.info("population_generated", requested=count, generated=len(all_ideas))
        return all_ideas

    async def _generate_batch(
        self,
        challenge: str,
        domain: DomainConfig,
        count: int,
        variation_hint: str,
    ) -> list[Idea]:
        prompt = GENERATION_PROMPT.format(
            count=count,
            challenge=challenge,
            domain_name=domain.display_name,
            variation_hint=f"ENFOQUE ADICIONAL: {variation_hint}" if variation_hint else "",
        )

        raw = await self._llm.generate(
            prompt=prompt,
            system_prompt=domain.system_prompt,
            temperature=0.9,  # alta temperatura → máxima diversidad inicial
            max_tokens=4096,
        )

        return self._parse_batch(raw, domain)

    def _parse_batch(self, raw: str, domain: DomainConfig) -> list[Idea]:
        json_match = re.search(r"\[[\s\S]*\]", raw)
        if not json_match:
            self._log.warning("no_json_array_found", raw_preview=raw[:200])
            return []

        try:
            items = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            self._log.warning("batch_parse_failed", error=str(e))
            return []

        ideas = []
        for item in items:
            try:
                ideas.append(self._item_to_idea(item, domain))
            except Exception as e:
                self._log.debug("item_parse_skipped", error=str(e))
        return ideas

    def _item_to_idea(self, item: dict[str, Any], domain: DomainConfig) -> Idea:
        vh_data = item.get("value_hypothesis") or {}
        feat_data = item.get("features") or {}

        value_hypothesis = None
        if vh_data.get("target_user") and vh_data.get("problem_solved"):
            value_hypothesis = ValueHypothesis(
                target_user=vh_data.get("target_user", ""),
                problem_solved=vh_data.get("problem_solved", ""),
                value_proposition=vh_data.get("value_proposition", "—"),
                differentiation=vh_data.get("differentiation", ""),
            )

        return Idea(
            title=item["title"],
            description=item["description"],
            advantages=item.get("advantages", []),
            limitations=item.get("limitations", []),
            value_hypothesis=value_hypothesis,
            features=IdeaFeatures(
                technologies=feat_data.get("technologies", []),
                materials=feat_data.get("materials", []),
                target_markets=feat_data.get("target_markets", []),
                complexity_level=float(feat_data.get("complexity_level", 0.5)),
            ),
            status=IdeaStatus.DRAFT,
            domain=domain.name,
        )

    @staticmethod
    def _get_variation_hint(base_hint: str, batch_index: int, total_batches: int) -> str:
        if base_hint:
            return base_hint
        return _VARIATION_ANGLES[batch_index % len(_VARIATION_ANGLES)]
