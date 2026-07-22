"""Integración del perfil del Analista Funcional con QDEngine (diseño §4-5).

Sin `profile` en el EvolutionRequest, el motor debe comportarse EXACTAMENTE
como hoy (flag CREATIVE_ANALYST_ENABLED apagado = flujo intacto). Con
`profile`, el motor genera sobre `reto_reformulado` en vez de `challenge`,
y el perfil se inyecta como contexto adicional en generator/evaluator.
"""

import json
from unittest.mock import AsyncMock

from creative_engine.agents.combined_evaluator import CombinedEvaluatorAgent
from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
from creative_engine.agents.generator import IdeaGeneratorAgent
from creative_engine.core.models import (
    ChallengeFriction,
    ChallengeProfile,
    ChallengeTopography,
    DomainName,
    EvolutionRequest,
    FunctionalHypothesis,
)
from creative_engine.evolution.crossover import CrossoverEngine
from creative_engine.evolution.encoders import IdeaEncoder
from creative_engine.evolution.mutation import MutationEngine
from creative_engine.evolution.qd_engine import QDEngine

_TOPICS = [
    ("Checkout en dos pasos", "Simplifica el proceso de pago para reducir abandono."),
    ("Recordatorio de carrito", "Notifica al usuario que dejó productos sin comprar."),
    ("Garantía extendida", "Ofrece devolución gratuita durante 60 días."),
]
_counter = {"n": 0}


def _make_generate(captured_prompts: list[str]):
    def _generate(prompt: str, **kwargs) -> str:
        captured_prompts.append(prompt)
        if "Genera" in prompt and "array" in prompt:
            items = []
            for _ in range(3):
                topic = _TOPICS[_counter["n"] % len(_TOPICS)]
                _counter["n"] += 1
                items.append(
                    {
                        "title": f"{topic[0]} v{_counter['n']}",
                        "description": topic[1] + f" Variante {_counter['n']}.",
                        "advantages": ["Ventaja"],
                        "limitations": ["Límite"],
                        "features": {"technologies": ["tech"], "complexity_level": 0.5},
                    }
                )
            return json.dumps(items)
        topic = _TOPICS[_counter["n"] % len(_TOPICS)]
        _counter["n"] += 1
        return json.dumps(
            {
                "title": f"{topic[0]} mutada",
                "description": topic[1] + " Evolución.",
                "advantages": ["Ventaja"],
                "limitations": ["Límite"],
                "mutation_description": "cambio",
            }
        )

    return _generate


def _make_generate_structured(captured_prompts: list[str]):
    def _generate_structured(prompt: str, **kwargs) -> dict:
        captured_prompts.append(prompt)
        return {
            "utility": 0.7,
            "utility_feedback": "ok",
            "feasibility": 0.6,
            "feasibility_feedback": "ok",
            "market_fit": 0.5,
            "market_feedback": "ok",
            "estimated_complexity": 0.5,
        }

    return _generate_structured


def _build_engine(llm, encoder: IdeaEncoder) -> QDEngine:
    evaluator = EvaluatorOrchestrator(agents={"combined": CombinedEvaluatorAgent(llm)})
    return QDEngine(
        generator=IdeaGeneratorAgent(llm),
        evaluator=evaluator,
        mutation=MutationEngine(llm),
        crossover=CrossoverEngine(llm),
        encoder=encoder,
        repository=None,
    )


async def test_without_profile_uses_challenge_as_is(deterministic_embed) -> None:
    """Sin profile (flag off): comportamiento idéntico al de siempre."""
    gen_prompts: list[str] = []
    eval_prompts: list[str] = []
    llm = AsyncMock()
    llm.generate.side_effect = _make_generate(gen_prompts)
    llm.generate_structured.side_effect = _make_generate_structured(eval_prompts)

    engine = _build_engine(llm, IdeaEncoder(embed_fn=deterministic_embed))
    request = EvolutionRequest(
        challenge="Mi tienda online no vende nada desde hace semanas",
        domain=DomainName.GENERIC,
        population_size=4,
        generations=1,
    )
    state = await engine.run_evolution(request)

    assert state.status == "completed"
    assert state.challenge == request.challenge
    assert any(request.challenge in p for p in gen_prompts)
    # Sin perfil, ningún prompt debe mencionar el bloque del Analista.
    assert all("Analista Funcional" not in p for p in eval_prompts)


async def test_with_profile_generates_on_reformulated_challenge(deterministic_embed) -> None:
    """Con profile: el motor genera sobre reto_reformulado, no sobre challenge."""
    gen_prompts: list[str] = []
    eval_prompts: list[str] = []
    llm = AsyncMock()
    llm.generate.side_effect = _make_generate(gen_prompts)
    llm.generate_structured.side_effect = _make_generate_structured(eval_prompts)

    profile = ChallengeProfile(
        reto_original="mi tienda online no vende nada desde hace semanas",
        topografia=ChallengeTopography(que_ocurre="las visitas no convierten"),
        hipotesis_funcional=FunctionalHypothesis(
            mecanismo="el checkout tiene demasiados pasos", confianza=0.8
        ),
        friccion=ChallengeFriction(descripcion_impacto="ingresos mensuales en caída"),
        restricciones_duras=["presupuesto menor a 500 euros"],
        reto_reformulado="Rediseñar el flujo de checkout para reducir el abandono",
    )

    engine = _build_engine(llm, IdeaEncoder(embed_fn=deterministic_embed))
    request = EvolutionRequest(
        challenge="mi tienda online no vende nada desde hace semanas",
        domain=DomainName.GENERIC,
        population_size=4,
        generations=1,
        profile=profile,
    )
    state = await engine.run_evolution(request)

    assert state.status == "completed"
    # state.challenge conserva el texto original (trazabilidad de lo que
    # escribió el usuario), aunque el motor haya generado sobre el reformulado.
    assert state.challenge == request.challenge

    # El generador debe haber recibido el reto REFORMULADO, no el original.
    assert any(profile.reto_reformulado in p for p in gen_prompts)

    # El perfil debe llegar al evaluador como contexto adicional.
    assert any("Analista Funcional" in p for p in eval_prompts)
    assert any("checkout tiene demasiados pasos" in p for p in eval_prompts)


def test_evolution_request_deserializes_profile_from_json() -> None:
    """El panel envía el perfil como JSON plano (fetch/body); confirma que
    FastAPI/Pydantic lo deserializa igual que si se construyera en Python."""
    payload = {
        "challenge": "Mi tienda online no vende nada desde hace semanas",
        "domain": "generic",
        "profile": {
            "reto_original": "mi tienda online no vende nada desde hace semanas",
            "topografia": {"que_ocurre": "las visitas no convierten", "frecuencia": "constante"},
            "hipotesis_funcional": {"mecanismo": "checkout confuso", "confianza": 0.75},
            "friccion": {"impacto_principal": "dinero", "descripcion_impacto": "ingresos"},
            "reto_reformulado": "Reducir la fricción del checkout",
        },
    }
    request = EvolutionRequest.model_validate(payload)

    assert request.profile is not None
    assert request.profile.reto_reformulado == "Reducir la fricción del checkout"
    assert request.profile.topografia.frecuencia == "constante"
