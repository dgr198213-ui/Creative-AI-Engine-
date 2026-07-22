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
        batch_size = min(count, 5)  # lotes cortos: menos truncamiento de JSON en la respuesta
        all_ideas: list[Idea] = []

        batches_needed = (count + batch_size - 1) // batch_size

        # Contrato de cardinalidad: intentos extra para rellenar lotes
        # fallidos/cortos, y nunca devolver más de lo pedido.
        max_attempts = batches_needed + 2
        attempt = 0
        while len(all_ideas) < count and attempt < max_attempts:
            remaining = count - len(all_ideas)
            batch_count = min(batch_size, remaining)
            batch_hint = self._get_variation_hint(variation_hint, attempt, batches_needed)

            try:
                batch = await self._generate_batch(
                    challenge=challenge,
                    domain=domain,
                    count=batch_count,
                    variation_hint=batch_hint,
                )
                all_ideas.extend(batch)
            except Exception as e:
                self._log.warning("generation_batch_failed", batch=attempt, error=str(e))
            attempt += 1

        all_ideas = all_ideas[:count]
        self._log.info("population_generated", requested=count, generated=len(all_ideas))
        return all_ideas

    async def refine_population(
        self, challenge: str, domain: DomainConfig, ideas: list[Idea]
    ) -> list[Idea]:
        """Una pasada de auto-mejora sobre ideas ya generadas.

        Usada por el arnés de benchmark (bench/harness.py) para el brazo
        "prompt único": N ideas + 1 pasada de auto-mejora, equivalente a
        pedirle a un LLM que revise su propia respuesta una vez. Si el
        LLM no devuelve un array parseable, se conservan las originales.
        """
        if not ideas:
            return ideas

        summary = "\n".join(f"- {i.title}: {i.description}" for i in ideas)
        prompt = (
            f"Generaste estas {len(ideas)} ideas para el reto: {challenge}\n\n"
            f"{summary}\n\n"
            "Revísalas: elimina redundancia entre ellas, mejora la "
            "accionabilidad y la claridad. Responde con el mismo formato "
            f"de array JSON de {len(ideas)} ideas (título, descripción, "
            "ventajas, limitaciones, value_hypothesis, features), mejoradas."
        )

        try:
            raw = await self._llm.generate(
                prompt=prompt,
                system_prompt=domain.system_prompt,
                temperature=0.5,
                max_tokens=4096,
            )
        except Exception as e:
            self._log.warning("refine_population_failed", error=str(e))
            return ideas

        refined = self._parse_batch(raw, domain)
        return refined if refined else ideas

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
        json_match = re.search(r"\[[\s\S]*\]?", raw)
        if not json_match:
            self._log.warning("no_json_array_found", raw_preview=raw[:200])
            return []

        payload = json_match.group(0)
        try:
            items = json.loads(payload)
        except json.JSONDecodeError as e:
            # El modelo trunca o malforma arrays largos con frecuencia.
            # Rescatamos las ideas completas que sí llegaron en vez de
            # descartar el lote entero.
            items = self._salvage_array(payload)
            if items:
                self._log.info(
                    "batch_salvaged", recovered=len(items), original_error=str(e)
                )
            else:
                self._log.warning("batch_parse_failed", error=str(e))
                return []

        ideas = []
        for item in items:
            try:
                ideas.append(self._item_to_idea(item, domain))
            except Exception as e:
                self._log.debug("item_parse_skipped", error=str(e))
        return ideas

    @staticmethod
    def _salvage_array(payload: str) -> list[dict[str, Any]]:
        """Recorta un array JSON truncado/malformado al último objeto completo.

        Prueba a cerrar el array en cada '}' desde el final hacia atrás;
        el primer recorte que parsea devuelve las ideas completas recibidas.
        """
        cut = payload.rfind("}")
        attempts = 0
        while cut != -1 and attempts < 40:
            candidate = payload[: cut + 1] + "]"
            try:
                items = json.loads(candidate)
                if isinstance(items, list):
                    return [i for i in items if isinstance(i, dict)]
                return []
            except json.JSONDecodeError:
                cut = payload.rfind("}", 0, cut)
                attempts += 1
        return []

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
