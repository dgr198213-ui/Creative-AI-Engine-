"""Tests del Analista Funcional: modelo de perfil, agente y espejo.

Diseño 22-jul-2026 §1-2: convierte un reto vago en un perfil estructurado
(topografía, hipótesis funcional, fricción) sin inventar datos, y renderiza
un espejo de confirmación de un único ciclo de corrección.
"""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from creative_engine.analysis.analyst import FunctionalAnalystAgent
from creative_engine.analysis.mirror import render_mirror
from creative_engine.core.models import (
    ChallengeFriction,
    ChallengeProfile,
    ChallengeTopography,
    FunctionalHypothesis,
)


class TestChallengeProfileModel:
    def test_valid_full_profile(self) -> None:
        profile = ChallengeProfile(
            reto_original="mi tienda online no vende",
            topografia=ChallengeTopography(
                que_ocurre="las visitas no convierten en compras",
                frecuencia="constante",
                donde_ocurre="checkout",
                intentos_previos=["bajar precios"],
            ),
            hipotesis_funcional=FunctionalHypothesis(
                mecanismo="el proceso de pago es confuso",
                confianza=0.8,
            ),
            friccion=ChallengeFriction(
                impacto_principal="dinero",
                descripcion_impacto="ventas mensuales",
                urgencia="alta",
            ),
            reto_reformulado="Rediseñar el flujo de checkout para reducir abandono",
        )
        assert profile.version == 1
        assert profile.topografia.frecuencia == "constante"
        assert profile.preguntas_pendientes == []

    def test_defaults_are_valid(self) -> None:
        """Un perfil vacío (todo default) debe ser válido: nunca inventa."""
        profile = ChallengeProfile()
        assert profile.topografia.frecuencia == "desconocida"
        assert profile.friccion.urgencia == "media"
        assert profile.hipotesis_funcional.confianza == 0.5

    def test_invalid_frecuencia_literal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChallengeTopography(frecuencia="mensual")  # no es un valor permitido

    def test_invalid_impacto_principal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChallengeFriction(impacto_principal="salud")

    def test_confianza_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FunctionalHypothesis(confianza=1.5)

    def test_preguntas_pendientes_max_two(self) -> None:
        with pytest.raises(ValidationError):
            ChallengeProfile(preguntas_pendientes=["a", "b", "c"])


def _mock_llm(response: dict) -> AsyncMock:
    llm = AsyncMock()
    llm.generate_structured.return_value = response
    return llm


class TestFunctionalAnalystAgent:
    async def test_analyze_produces_valid_profile(self) -> None:
        llm = _mock_llm(
            {
                "topografia": {
                    "que_ocurre": "las ventas cayeron el último trimestre",
                    "frecuencia": "recurrente",
                    "desde_cuando": "hace 3 meses",
                    "donde_ocurre": "tienda online",
                    "intentos_previos": ["publicidad en redes"],
                },
                "hipotesis_funcional": {
                    "antecedente": "cambio de proveedor de pagos",
                    "mecanismo": "el nuevo checkout tiene más pasos",
                    "refuerzo": "",
                    "confianza": 0.75,
                },
                "friccion": {
                    "impacto_principal": "dinero",
                    "descripcion_impacto": "ingresos mensuales",
                    "urgencia": "alta",
                },
                "restricciones_duras": ["presupuesto limitado"],
                "reto_reformulado": "Reducir la fricción del checkout online",
                "preguntas_pendientes": [],
            }
        )
        agent = FunctionalAnalystAgent(llm)
        profile = await agent.analyze("mi tienda online no vende como antes")

        assert profile.version == 1
        assert profile.reto_original == "mi tienda online no vende como antes"
        assert profile.topografia.frecuencia == "recurrente"
        assert profile.hipotesis_funcional.confianza == 0.75
        assert profile.reto_reformulado == "Reducir la fricción del checkout online"
        assert profile.preguntas_pendientes == []

    async def test_low_confidence_keeps_pending_questions(self) -> None:
        llm = _mock_llm(
            {
                "hipotesis_funcional": {"mecanismo": "no está claro", "confianza": 0.3},
                "preguntas_pendientes": ["¿Desde cuándo notas la caída?", "¿Cambiaste algo?"],
            }
        )
        agent = FunctionalAnalystAgent(llm)
        profile = await agent.analyze("mis clientes se van y no sé por qué")

        assert profile.hipotesis_funcional.confianza == 0.3
        assert len(profile.preguntas_pendientes) == 2

    async def test_high_confidence_drops_pending_questions(self) -> None:
        """Aunque el LLM devuelva preguntas, con confianza alta se descartan."""
        llm = _mock_llm(
            {
                "hipotesis_funcional": {"mecanismo": "claro como el agua", "confianza": 0.9},
                "preguntas_pendientes": ["¿Seguro?"],
            }
        )
        agent = FunctionalAnalystAgent(llm)
        profile = await agent.analyze("mi restaurante se llena pero no gano dinero")

        assert profile.preguntas_pendientes == []

    async def test_more_than_two_questions_truncated(self) -> None:
        llm = _mock_llm(
            {
                "hipotesis_funcional": {"confianza": 0.2},
                "preguntas_pendientes": ["a", "b", "c", "d"],
            }
        )
        agent = FunctionalAnalystAgent(llm)
        profile = await agent.analyze("algo vago")
        assert len(profile.preguntas_pendientes) == 2

    async def test_invalid_literal_from_llm_falls_back_to_default(self) -> None:
        """Un LLM que no respeta el enum no debe tumbar el análisis."""
        llm = _mock_llm(
            {
                "topografia": {"frecuencia": "todos los días, creo"},
                "friccion": {"impacto_principal": "salud mental", "urgencia": "muchísima"},
            }
        )
        agent = FunctionalAnalystAgent(llm)
        profile = await agent.analyze("mis empleados no rinden")

        assert profile.topografia.frecuencia == "desconocida"
        assert profile.friccion.impacto_principal == "dinero"
        assert profile.friccion.urgencia == "media"

    async def test_llm_failure_degrades_to_minimal_profile(self) -> None:
        """Si la llamada LLM falla, se devuelve un perfil mínimo, no una excepción."""
        llm = AsyncMock()
        llm.generate_structured.side_effect = Exception("proveedor caído")

        agent = FunctionalAnalystAgent(llm)
        profile = await agent.analyze("la competencia me copia todo")

        assert profile.reto_original == "la competencia me copia todo"
        assert profile.reto_reformulado == "la competencia me copia todo"
        assert profile.preguntas_pendientes == []

    async def test_correction_cycle_produces_v2_keeping_original(self) -> None:
        """Un ciclo de corrección produce v2, preservando el reto original."""
        llm = _mock_llm(
            {
                "hipotesis_funcional": {"mecanismo": "confirmado por el usuario", "confianza": 0.9},
                "reto_reformulado": "Reto reformulado corregido",
            }
        )
        agent = FunctionalAnalystAgent(llm)

        previous = ChallengeProfile(
            version=1,
            reto_original="mi restaurante se llena pero no gano dinero",
            reto_reformulado="primer intento de reformulación",
        )

        profile_v2 = await agent.analyze(
            challenge="mi restaurante se llena pero no gano dinero",
            correction="el problema es el coste de los ingredientes, no el personal",
            previous_profile=previous,
        )

        assert profile_v2.version == 2
        assert profile_v2.reto_original == previous.reto_original
        assert profile_v2.reto_reformulado == "Reto reformulado corregido"
        # el prompt debe haber incluido el perfil previo y la corrección
        sent_prompt = llm.generate_structured.call_args.kwargs["prompt"]
        assert "primer intento de reformulación" in sent_prompt
        assert "coste de los ingredientes" in sent_prompt


class TestRenderMirror:
    def test_includes_que_ocurre_and_impacto(self) -> None:
        profile = ChallengeProfile(
            topografia=ChallengeTopography(que_ocurre="las ventas bajan"),
            friccion=ChallengeFriction(descripcion_impacto="la facturación mensual"),
            hipotesis_funcional=FunctionalHypothesis(mecanismo="hay un cuello de botella"),
        )
        text = render_mirror(profile)
        assert "las ventas bajan" in text
        assert "la facturación mensual" in text
        assert "hay un cuello de botella" in text

    def test_includes_refuerzo_when_present(self) -> None:
        profile = ChallengeProfile(
            hipotesis_funcional=FunctionalHypothesis(
                mecanismo="el proceso es lento", refuerzo="evita conflictos difíciles"
            )
        )
        text = render_mirror(profile)
        assert "evita conflictos difíciles" in text

    def test_no_pending_questions_section_when_empty(self) -> None:
        profile = ChallengeProfile()
        text = render_mirror(profile)
        assert "Antes de seguir" not in text

    def test_pending_questions_rendered_as_list(self) -> None:
        profile = ChallengeProfile(preguntas_pendientes=["¿Pregunta uno?", "¿Pregunta dos?"])
        text = render_mirror(profile)
        assert "Antes de seguir" in text
        assert "¿Pregunta uno?" in text
        assert "¿Pregunta dos?" in text
