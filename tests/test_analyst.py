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
    BehaviorDimension,
    ChallengeFriction,
    ChallengeProfile,
    ChallengeTopography,
    DomainConfig,
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


class TestDomainProfileFields:
    """D4, Fase 6: campos extra de ChallengeProfile.dominio declarados por
    el domain pack (`DomainConfig.profile_fields`), sin tocar el motor."""

    def _domain_with_fields(self, profile_fields: list[dict[str, str]]) -> DomainConfig:
        return DomainConfig(
            name="tuesdi",
            display_name="TUESDI",
            behavior_dimensions=[
                BehaviorDimension(name="a", bins=5),
                BehaviorDimension(name="b", bins=5),
            ],
            profile_fields=profile_fields,
        )

    async def test_without_domain_dominio_stays_empty(self) -> None:
        """Sin domain (comportamiento de siempre): dominio queda vacío,
        aunque el LLM devuelva un bloque 'dominio' por su cuenta."""
        llm = _mock_llm(
            {
                "reto_reformulado": "algo",
                "dominio": {"tipo_artista": "pintor"},
            }
        )
        agent = FunctionalAnalystAgent(llm)
        profile = await agent.analyze("un reto cualquiera de prueba suficientemente largo")
        assert profile.dominio == {}

    async def test_domain_without_profile_fields_keeps_dominio_empty(self) -> None:
        llm = _mock_llm({"reto_reformulado": "algo", "dominio": {"x": "y"}})
        agent = FunctionalAnalystAgent(llm)
        domain = self._domain_with_fields([])
        profile = await agent.analyze(
            "un reto cualquiera de prueba suficientemente largo", domain=domain
        )
        assert profile.dominio == {}

    async def test_domain_with_profile_fields_populates_dominio(self) -> None:
        llm = _mock_llm(
            {
                "reto_reformulado": "algo",
                "dominio": {"tipo_artista": "pintor", "aforo_tipico": "50 personas"},
            }
        )
        agent = FunctionalAnalystAgent(llm)
        domain = self._domain_with_fields(
            [
                {"nombre": "tipo_artista", "descripcion": "Género o disciplina"},
                {"nombre": "aforo_tipico", "descripcion": "Capacidad típica"},
            ]
        )
        profile = await agent.analyze(
            "un reto cualquiera de prueba suficientemente largo", domain=domain
        )
        assert profile.dominio == {"tipo_artista": "pintor", "aforo_tipico": "50 personas"}

    async def test_dominio_ignores_undeclared_fields(self) -> None:
        """El Analista solo recoge los campos que el pack declaró — un
        campo extra que el LLM invente no se cuela en el perfil."""
        llm = _mock_llm(
            {
                "reto_reformulado": "algo",
                "dominio": {"tipo_artista": "pintor", "campo_no_declarado": "x"},
            }
        )
        agent = FunctionalAnalystAgent(llm)
        domain = self._domain_with_fields(
            [{"nombre": "tipo_artista", "descripcion": "Género o disciplina"}]
        )
        profile = await agent.analyze(
            "un reto cualquiera de prueba suficientemente largo", domain=domain
        )
        assert profile.dominio == {"tipo_artista": "pintor"}

    async def test_prompt_includes_dominio_schema_block_when_declared(self) -> None:
        llm = _mock_llm({"reto_reformulado": "algo"})
        agent = FunctionalAnalystAgent(llm)
        domain = self._domain_with_fields(
            [{"nombre": "tipo_artista", "descripcion": "Género o disciplina"}]
        )
        await agent.analyze(
            "un reto cualquiera de prueba suficientemente largo", domain=domain
        )
        prompt = llm.generate_structured.call_args.kwargs["prompt"]
        assert '"dominio"' in prompt
        assert "tipo_artista" in prompt

    async def test_domain_analyst_prompt_used_as_system_prompt(self) -> None:
        llm = _mock_llm({"reto_reformulado": "algo"})
        agent = FunctionalAnalystAgent(llm)
        domain = self._domain_with_fields([])
        domain = domain.model_copy(update={"analyst_prompt": "Persona TUESDI a medida"})

        await agent.analyze(
            "un reto cualquiera de prueba suficientemente largo", domain=domain
        )

        system_prompt = llm.generate_structured.call_args.kwargs["system_prompt"]
        assert system_prompt == "Persona TUESDI a medida"


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
