"""Espejo de confirmación (diseño 22-jul-2026 §2): texto plano que el panel
muestra tras la llamada al Analista y antes de lanzar el run, para que el
usuario confirme o corrija en un único ciclo — la puerta no es un chat.
"""

from __future__ import annotations

from ..core.models import ChallengeProfile


def render_mirror(profile: ChallengeProfile) -> str:
    """Renderiza el espejo en texto plano con negrita ligera (**...**)."""
    lines = ["**Esto es lo que he entendido:**"]

    impacto = profile.friccion.descripcion_impacto or "el negocio en general"
    que_ocurre = profile.topografia.que_ocurre or "el problema descrito"
    lines.append(f"{que_ocurre}, que afecta sobre todo a {impacto}.")

    lines.append("")
    lines.append("**Mi hipótesis de la causa de fondo:**")
    hipotesis_txt = profile.hipotesis_funcional.mecanismo or "no tengo suficiente información aún"
    if profile.hipotesis_funcional.refuerzo:
        hipotesis_txt = (
            f"{hipotesis_txt} Y sospecho que se mantiene porque "
            f"{profile.hipotesis_funcional.refuerzo}."
        )
    lines.append(hipotesis_txt)

    if profile.preguntas_pendientes:
        lines.append("")
        lines.append("**Antes de seguir, me ayudaría saber:**")
        lines.extend(f"- {q}" for q in profile.preguntas_pendientes)

    return "\n".join(lines)
