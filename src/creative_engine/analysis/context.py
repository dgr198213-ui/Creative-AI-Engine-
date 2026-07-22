"""Resume el ChallengeProfile como contexto para generator/evaluator.

El motor no conoce el Analista Funcional como concepto: solo recibe un
texto (hint) igual que cualquier otro variation_hint o contexto de
evaluación — invariante 5 del CLAUDE.md (el motor no conoce la capa que
lo alimenta).
"""

from __future__ import annotations

from ..core.models import ChallengeProfile


def profile_context_hint(profile: ChallengeProfile) -> str:
    """Resumen breve del perfil para inyectar en prompts de generación/evaluación."""
    parts: list[str] = []

    if profile.topografia.que_ocurre:
        parts.append(f"Contexto observado: {profile.topografia.que_ocurre}.")
    if profile.hipotesis_funcional.mecanismo:
        parts.append(f"Hipótesis de causa de fondo: {profile.hipotesis_funcional.mecanismo}.")
    if profile.friccion.descripcion_impacto:
        parts.append(
            f"Impacto principal ({profile.friccion.impacto_principal}): "
            f"{profile.friccion.descripcion_impacto}."
        )
    if profile.restricciones_duras:
        parts.append("Restricciones a respetar: " + "; ".join(profile.restricciones_duras) + ".")

    return " ".join(parts)
