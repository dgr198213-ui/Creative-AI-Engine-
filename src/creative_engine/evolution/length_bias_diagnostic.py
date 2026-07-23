"""Diagnóstico del sesgo de longitud en el descriptor de originalidad.

Sospecha detectada en el run `c66010b7` (22-jul-2026): dos ideas
conceptualmente primas (grafo causal de linajes, descritas corta y larga)
puntuaron 100% y 16% de originalidad. Si la longitud del texto domina
sobre el contenido al codificar, dos textos MUY parecidos en contenido
pero de longitud muy distinta saldrán lejos en el espacio de embeddings,
y dos textos de contenido DISTINTO pero longitud parecida saldrán cerca
— justo lo contrario de lo deseable para medir novedad real.

Requiere el modelo real de sentence-transformers (no el `embed_fn` falso
de los tests): ejecútalo con `creative-engine diagnose-length-bias` donde
haya red la primera vez. No participa en la suite `pytest` normal (que
corre sin red ni BD, ver CLAUDE.md) más que como guardia opcional que se
salta sola sin conexión — ver `tests/test_length_bias_diagnostic.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoders import IdeaEncoder

# Mismo concepto de fondo, longitud MUY distinta (blurb vs elaboración).
_SAME_CONCEPT_PAIRS: list[tuple[str, str]] = [
    (
        "Grafo causal de linajes: mapa visual que conecta eventos históricos "
        "por causa y efecto.",
        "Un grafo causal de linajes es una representación visual e "
        "interactiva que conecta eventos históricos entre sí mediante "
        "relaciones de causa y efecto, permitiendo al usuario explorar cómo "
        "una decisión temprana genera una cadena de consecuencias a lo largo "
        "de generaciones, identificar puntos de inflexión clave, comparar "
        "líneas temporales alternativas y entender el impacto acumulado de "
        "decisiones pasadas sobre el presente, con una interfaz pensada para "
        "historiadores, educadores y estudiantes que buscan estudiar la "
        "causalidad histórica en profundidad.",
    ),
    (
        "Dron sanitario rural: entrega medicamentos de forma autónoma en "
        "zonas aisladas.",
        "Un dron sanitario rural es un vehículo aéreo no tripulado que "
        "planifica rutas óptimas para entregar medicamentos, vacunas y "
        "muestras de laboratorio en comunidades rurales aisladas donde el "
        "transporte terrestre es lento o inseguro, coordinándose con centros "
        "de salud locales, priorizando envíos urgentes según criticidad "
        "médica y registrando cada entrega para trazabilidad sanitaria.",
    ),
]

# Conceptos DISTINTOS entre sí, longitud parecida entre pares.
_DIFFERENT_CONCEPT_PAIRS: list[tuple[str, str]] = [
    (
        "Bicicleta solar plegable: cuadro con paneles integrados que cargan "
        "la batería mientras se pedalea por la ciudad.",
        "Red de trueque vecinal: aplicación para intercambiar objetos y "
        "servicios entre vecinos sin usar dinero en efectivo.",
    ),
    (
        "Huerto vertical modular: sistema hidropónico apilable para fachadas "
        "y balcones urbanos con poco espacio disponible.",
        "Ropa con sensores térmicos: prendas que regulan la temperatura "
        "corporal activamente según la actividad física del usuario.",
    ),
]


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class LengthBiasDiagnosis:
    same_concept_similarities: list[float]
    different_concept_similarities: list[float]
    mean_same_concept: float
    mean_different_concept: float
    bias_confirmed: bool
    verdict: str


def diagnose_length_bias(encoder: IdeaEncoder | None = None) -> LengthBiasDiagnosis:
    """Codifica los pares y compara similitudes.

    Sesgo confirmado si los pares de MISMO concepto y longitud MUY
    distinta salen MENOS similares (en promedio) que los pares de
    CONCEPTO distinto y longitud parecida — señal de que la longitud pesa
    más que el contenido en el espacio de embeddings.
    """
    enc = encoder or IdeaEncoder()

    same = [_cosine(enc._embed(a), enc._embed(b)) for a, b in _SAME_CONCEPT_PAIRS]
    diff = [_cosine(enc._embed(a), enc._embed(b)) for a, b in _DIFFERENT_CONCEPT_PAIRS]

    mean_same = sum(same) / len(same)
    mean_diff = sum(diff) / len(diff)
    confirmed = mean_diff > mean_same

    if confirmed:
        verdict = (
            "SESGO DE LONGITUD CONFIRMADO: conceptos distintos de longitud "
            f"parecida salen más similares (media {mean_diff:.3f}) que el "
            f"mismo concepto en longitudes muy distintas (media "
            f"{mean_same:.3f}). La longitud domina sobre el contenido."
        )
    else:
        verdict = (
            "Sin sesgo de longitud: el mismo concepto en longitudes muy "
            f"distintas sale más similar (media {mean_same:.3f}) que "
            f"conceptos distintos de longitud parecida (media "
            f"{mean_diff:.3f}). El contenido domina, como se espera."
        )

    return LengthBiasDiagnosis(
        same_concept_similarities=same,
        different_concept_similarities=diff,
        mean_same_concept=mean_same,
        mean_different_concept=mean_diff,
        bias_confirmed=confirmed,
        verdict=verdict,
    )
