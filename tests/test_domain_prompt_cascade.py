"""Cascada de prompts por dominio en los agentes (D3, Fase 6).

El motor nunca contiene texto específico de un dominio: generator y
evaluator resuelven el prompt de sistema desde el pack (con fallback a
un default embebido si el pack no declara nada), aplicando los
placeholders {reto}/{perfil}/{inspiraciones} cuando el pack los usa.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from creative_engine.agents.combined_evaluator import COMBINED_SYSTEM, CombinedEvaluatorAgent
from creative_engine.agents.generator import IdeaGeneratorAgent
from creative_engine.core.models import BehaviorDimension, DomainConfig, Idea, IdeaFeatures


def _domain(**overrides: str) -> DomainConfig:
    return DomainConfig(
        name="test",
        display_name="Test",
        behavior_dimensions=[
            BehaviorDimension(name="a", bins=5),
            BehaviorDimension(name="b", bins=5),
        ],
        **overrides,
    )


class TestGeneratorPromptCascade:
    async def test_uses_domain_generator_prompt(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = '{"title":"x","description":"y "*3,"advantages":[],"limitations":[]}'
        agent = IdeaGeneratorAgent(llm)
        domain = _domain(generator_prompt="Persona a medida del pack")

        await agent.refine_population(
            challenge="reto de prueba", domain=domain, ideas=[]
        )
        # refine_population con ideas=[] no llama al LLM; forzamos con 1 idea.
        idea = Idea(title="Idea", description="Descripción de prueba suficientemente larga.")
        await agent.refine_population(challenge="reto de prueba", domain=domain, ideas=[idea])

        system_prompt = llm.generate.call_args.kwargs["system_prompt"]
        assert system_prompt == "Persona a medida del pack"

    async def test_falls_back_to_default_when_pack_has_no_prompt(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "{}"
        agent = IdeaGeneratorAgent(llm)
        domain = _domain()  # generator_prompt="" por defecto
        idea = Idea(title="Idea", description="Descripción de prueba suficientemente larga.")

        await agent.refine_population(challenge="reto", domain=domain, ideas=[idea])

        system_prompt = llm.generate.call_args.kwargs["system_prompt"]
        assert "innovación" in system_prompt.lower()

    async def test_resolves_reto_placeholder_in_generator_prompt(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "{}"
        agent = IdeaGeneratorAgent(llm)
        domain = _domain(generator_prompt="Ideas para el reto: {reto}")
        idea = Idea(title="Idea", description="Descripción de prueba suficientemente larga.")

        await agent.refine_population(
            challenge="vender más bicicletas", domain=domain, ideas=[idea]
        )

        system_prompt = llm.generate.call_args.kwargs["system_prompt"]
        assert system_prompt == "Ideas para el reto: vender más bicicletas"


class TestEvaluatorPromptCascade:
    def _idea(self) -> Idea:
        return Idea(
            title="Idea de prueba",
            description="Descripción de prueba suficientemente larga para validar.",
            features=IdeaFeatures(),
        )

    async def test_uses_context_evaluator_prompt(self) -> None:
        llm = AsyncMock()
        llm.generate_structured.return_value = {"utility": 0.5, "feasibility": 0.5, "market_fit": 0.5}
        agent = CombinedEvaluatorAgent(llm)

        await agent.evaluate(
            self._idea(),
            context={"challenge": "reto", "evaluator_prompt": "Rúbrica a medida del pack"},
        )

        system_prompt = llm.generate_structured.call_args.kwargs["system_prompt"]
        assert system_prompt == "Rúbrica a medida del pack"

    async def test_falls_back_to_combined_system_without_context_prompt(self) -> None:
        llm = AsyncMock()
        llm.generate_structured.return_value = {"utility": 0.5, "feasibility": 0.5, "market_fit": 0.5}
        agent = CombinedEvaluatorAgent(llm)

        await agent.evaluate(self._idea(), context={"challenge": "reto"})

        system_prompt = llm.generate_structured.call_args.kwargs["system_prompt"]
        assert system_prompt == COMBINED_SYSTEM

    async def test_resolves_reto_and_perfil_placeholders(self) -> None:
        llm = AsyncMock()
        llm.generate_structured.return_value = {"utility": 0.5, "feasibility": 0.5, "market_fit": 0.5}
        agent = CombinedEvaluatorAgent(llm)

        await agent.evaluate(
            self._idea(),
            context={
                "challenge": "vender más",
                "profile_hint": "tienda online",
                "evaluator_prompt": "Reto: {reto}. Perfil: {perfil}.",
            },
        )

        system_prompt = llm.generate_structured.call_args.kwargs["system_prompt"]
        assert system_prompt == "Reto: vender más. Perfil: tienda online."
