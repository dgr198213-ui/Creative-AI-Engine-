"""Juez LLM ciego para el benchmark de 3 brazos (diseño 22-jul-2026 §3).

Puntúa accionabilidad y pertinencia del top-3 de cada brazo sin saber de
qué brazo viene ni qué método lo produjo: el prompt no menciona "motor
QD" ni "prompt único", solo título+descripción de las propuestas.
"""

from __future__ import annotations

import structlog

from ..core.models import Idea
from ..llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

JUDGE_SYSTEM = """Eres un consultor de negocio experimentado. Evalúas propuestas
de solución a un reto de negocio con honestidad y sin sesgo hacia ningún
estilo de redacción. No sabes cómo se generó cada propuesta ni debes
intentar adivinarlo — puntúa solo por su contenido."""

JUDGE_PROMPT = """RETO: {challenge}

Evalúa estas propuestas, cada una del 1 al 10 en dos dimensiones:
- accionabilidad: ¿qué tan lista está para actuar mañana mismo?
- pertinencia: ¿qué tan bien resuelve el reto concreto?

{propuestas}

Responde SOLO en JSON, un objeto por propuesta en el mismo orden:
{{"puntuaciones": [{{"accionabilidad": 0, "pertinencia": 0}}]}}"""


async def judge_blind(llm: LLMProvider, challenge: str, ideas: list[Idea]) -> float | None:
    """Media (0-10) de accionabilidad+pertinencia del top-3, o None si falla/vacío."""
    top = ideas[:3]
    if not top:
        return None

    propuestas = "\n\n".join(
        f"PROPUESTA {i}: {idea.title}\n{idea.description}"
        for i, idea in enumerate(top, 1)
    )
    prompt = JUDGE_PROMPT.format(challenge=challenge, propuestas=propuestas)

    try:
        data = await llm.generate_structured(prompt=prompt, system_prompt=JUDGE_SYSTEM)
    except Exception as e:
        logger.warning("blind_judge_failed", error=str(e))
        return None

    scores = data.get("puntuaciones") or []
    values: list[float] = []
    for s in scores:
        if not isinstance(s, dict):
            continue
        try:
            accionabilidad = max(0.0, min(10.0, float(s.get("accionabilidad", 5))))
            pertinencia = max(0.0, min(10.0, float(s.get("pertinencia", 5))))
        except (TypeError, ValueError):
            continue
        values.append((accionabilidad + pertinencia) / 2)

    return sum(values) / len(values) if values else None
