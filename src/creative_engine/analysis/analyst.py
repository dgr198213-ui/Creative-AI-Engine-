"""Agente Analista Funcional: perfila un reto vago antes de generar ideas.

Convierte "mi tienda no vende" en un perfil estructurado (topografía,
hipótesis funcional, fricción, restricciones) que generator/evaluator
usan como contexto. Una sola llamada LLM estructurada; nunca inventa
datos — lo desconocido queda en blanco/"desconocida", no se rellena con
suposiciones presentadas como hechos.
"""

from __future__ import annotations

from typing import Any

import structlog

from ..core.models import (
    ChallengeFriction,
    ChallengeProfile,
    ChallengeTopography,
    FunctionalHypothesis,
)
from ..llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

_FRECUENCIA_VALUES = {"puntual", "recurrente", "constante", "desconocida"}
_IMPACTO_VALUES = {"dinero", "tiempo", "clientes", "equipo", "reputación"}
_URGENCIA_VALUES = {"baja", "media", "alta"}

# Umbral bajo el cual se muestran preguntas_pendientes en el espejo (§1/§2
# del diseño). Por debajo, el Analista no está seguro de su hipótesis.
CONFIDENCE_QUESTIONS_THRESHOLD = 0.6

ANALYST_SYSTEM = """Eres un analista funcional senior especializado en diagnosticar
problemas de negocio a partir de descripciones vagas de personas no técnicas.

Reglas estrictas:
- NUNCA inventes datos que el usuario no ha dado. Si algo no se puede saber
  con lo que hay, dilo explícitamente (null, o "desconocida" en frecuencia)
  — no rellenes con suposiciones presentadas como hechos.
- La hipótesis de la causa de fondo es una HIPÓTESIS, no un diagnóstico
  certero: exprésala con la incertidumbre real que tiene (campo `confianza`).
- `reto_reformulado` debe conservar el vocabulario y el dominio del usuario
  donde sea posible — no lo traduzcas a jerga técnica innecesaria.
- Si tu confianza en la hipótesis es menor a 0.6, incluye hasta 2
  `preguntas_pendientes` que ayudarían a confirmarla; si es 0.6 o más, esa
  lista debe quedar vacía."""

ANALYST_PROMPT = """Analiza este reto y devuelve su perfil funcional en JSON.

RETO DEL USUARIO: {challenge}
{correction_block}
Responde SOLO con este JSON (mismas claves, mismos tipos):
{{
  "topografia": {{
    "que_ocurre": "descripción neutra y observable del problema",
    "frecuencia": "puntual | recurrente | constante | desconocida",
    "desde_cuando": "string o null",
    "donde_ocurre": "área del negocio/sistema afectada",
    "intentos_previos": ["qué ha probado ya el usuario, si lo menciona"]
  }},
  "hipotesis_funcional": {{
    "antecedente": "qué dispara o precede al problema",
    "mecanismo": "por qué se produce (hipótesis, no certeza)",
    "refuerzo": "qué beneficio oculto mantiene el problema vivo, si lo hay",
    "confianza": 0.7
  }},
  "friccion": {{
    "impacto_principal": "dinero | tiempo | clientes | equipo | reputación",
    "descripcion_impacto": "en palabras del dominio del usuario",
    "urgencia": "baja | media | alta"
  }},
  "restricciones_duras": ["restricciones explícitas o muy probables, si las hay"],
  "reto_reformulado": "el reto técnico que recibirá el motor de generación de ideas",
  "preguntas_pendientes": ["máx 2, solo si tu confianza es < 0.6"]
}}"""

_CORRECTION_BLOCK = """
TU ANÁLISIS ANTERIOR (perfil v{prev_version}): {previous_json}
CORRECCIÓN DEL USUARIO: {correction}
Genera un perfil actualizado que incorpore la corrección; no repitas
preguntas ya respondidas por ella.
"""


def _safe_literal(value: Any, allowed: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _safe_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def _safe_str_list(value: Any, max_items: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [str(v) for v in value if v is not None]
    return items[:max_items] if max_items else items


class FunctionalAnalystAgent:
    """Perfila un reto en una sola llamada LLM estructurada."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._log = logger.bind(agent="analyst")

    async def analyze(
        self,
        challenge: str,
        correction: str | None = None,
        previous_profile: ChallengeProfile | None = None,
    ) -> ChallengeProfile:
        """Produce el perfil funcional de un reto (o su v2 tras una corrección).

        Degrada con elegancia si la llamada LLM falla: devuelve un perfil
        mínimo (reto_reformulado = reto_original) en vez de propagar la
        excepción, para que el endpoint /analyze nunca caiga por esto.
        """
        correction_block = ""
        if correction and previous_profile is not None:
            correction_block = _CORRECTION_BLOCK.format(
                prev_version=previous_profile.version,
                previous_json=previous_profile.model_dump_json(
                    exclude={"reto_original"}
                ),
                correction=correction,
            )

        prompt = ANALYST_PROMPT.format(
            challenge=challenge, correction_block=correction_block
        )

        try:
            data = await self._llm.generate_structured(
                prompt=prompt, system_prompt=ANALYST_SYSTEM
            )
        except Exception as e:
            self._log.error("analysis_failed", error=str(e))
            data = {}

        reto_original = (
            previous_profile.reto_original if previous_profile is not None else challenge
        )
        version = (previous_profile.version + 1) if previous_profile is not None else 1

        profile = self._parse(data, reto_original=reto_original, version=version)
        self._log.info(
            "profile_generated",
            version=profile.version,
            confianza=profile.hipotesis_funcional.confianza,
            preguntas=len(profile.preguntas_pendientes),
        )
        return profile

    def _parse(self, data: dict[str, Any], reto_original: str, version: int) -> ChallengeProfile:
        topografia_data = data.get("topografia") or {}
        hipotesis_data = data.get("hipotesis_funcional") or {}
        friccion_data = data.get("friccion") or {}

        topografia = ChallengeTopography(
            que_ocurre=str(topografia_data.get("que_ocurre", "")),
            frecuencia=_safe_literal(
                topografia_data.get("frecuencia"), _FRECUENCIA_VALUES, "desconocida"
            ),
            desde_cuando=topografia_data.get("desde_cuando"),
            donde_ocurre=str(topografia_data.get("donde_ocurre", "")),
            intentos_previos=_safe_str_list(topografia_data.get("intentos_previos")),
        )

        confianza = _safe_confidence(hipotesis_data.get("confianza"))
        hipotesis = FunctionalHypothesis(
            antecedente=str(hipotesis_data.get("antecedente", "")),
            mecanismo=str(hipotesis_data.get("mecanismo", "")),
            refuerzo=str(hipotesis_data.get("refuerzo", "")),
            confianza=confianza,
        )

        friccion = ChallengeFriction(
            impacto_principal=_safe_literal(
                friccion_data.get("impacto_principal"), _IMPACTO_VALUES, "dinero"
            ),
            descripcion_impacto=str(friccion_data.get("descripcion_impacto", "")),
            urgencia=_safe_literal(
                friccion_data.get("urgencia"), _URGENCIA_VALUES, "media"
            ),
        )

        # Regla del agente: preguntas_pendientes solo si la confianza es baja,
        # y como mucho 2 — se hace cumplir aquí aunque el LLM se equivoque.
        preguntas = (
            _safe_str_list(data.get("preguntas_pendientes"), max_items=2)
            if confianza < CONFIDENCE_QUESTIONS_THRESHOLD
            else []
        )

        return ChallengeProfile(
            version=version,
            reto_original=reto_original,
            topografia=topografia,
            hipotesis_funcional=hipotesis,
            friccion=friccion,
            restricciones_duras=_safe_str_list(data.get("restricciones_duras")),
            reto_reformulado=str(data.get("reto_reformulado") or reto_original),
            preguntas_pendientes=preguntas,
        )
