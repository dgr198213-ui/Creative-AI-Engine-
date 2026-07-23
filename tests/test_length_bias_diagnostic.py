"""Diagnóstico determinista del sesgo de longitud (Fase 5, bloque 4).

Requiere el modelo REAL de sentence-transformers (no el `embed_fn` falso
de los demás tests): sin red, este test se salta solo — no rompe la
regla del proyecto de que la suite corre sin red ni BD (ver CLAUDE.md).
Sirve de guardia de regresión permanente para cuando SÍ hay red
disponible (entorno local del autor, o donde el modelo ya esté cacheado).
"""

from __future__ import annotations

import pytest

from creative_engine.evolution.encoders import IdeaEncoder
from creative_engine.evolution.length_bias_diagnostic import diagnose_length_bias


def test_length_bias_diagnostic() -> None:
    try:
        encoder = IdeaEncoder()
        encoder._embed("prueba de disponibilidad del modelo de embeddings")
    except Exception as e:
        pytest.skip(f"modelo de embeddings no disponible (sin red en este entorno): {e}")

    result = diagnose_length_bias(encoder)
    assert not result.bias_confirmed, result.verdict
