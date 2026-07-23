"""Retención de memoria entre runs (incidente 22-jul-2026, ver CLAUDE.md).

Un run no debe dejar más memoria retenida de la que ya tenía el proceso
tras el run anterior. Con embeddings deterministas (sin torch real) la
magnitud absoluta no reproduce el incidente de producción (esa parte es
el encoder compartido, ver `encoders.get_shared_encoder`), pero el
PATRÓN de crecimiento sí: si algo empieza a acumular referencias entre
runs — una lista a nivel de módulo, un handler de eventos sin
desuscribir, un archivo MAP-Elites que no se libera — este test lo detecta.
"""

from __future__ import annotations

import gc
import json
from unittest.mock import AsyncMock

import structlog.testing

from creative_engine.agents.combined_evaluator import CombinedEvaluatorAgent
from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
from creative_engine.agents.generator import IdeaGeneratorAgent
from creative_engine.core.memory_utils import current_rss_mb
from creative_engine.core.models import DomainName, EvolutionRequest
from creative_engine.evolution.crossover import CrossoverEngine
from creative_engine.evolution.encoders import IdeaEncoder
from creative_engine.evolution.mutation import MutationEngine
from creative_engine.evolution.qd_engine import QDEngine

_TOPICS = [
    ("Bicicleta solar plegable", "Bicicleta urbana con paneles solares flexibles integrados."),
    ("Red de trueque vecinal", "Plataforma para intercambiar objetos y servicios entre vecinos."),
    ("Dron sanitario rural", "Dron autónomo que entrega medicamentos en zonas aisladas."),
]
_counter = {"n": 0}


def _fake_generate(prompt: str, **kwargs) -> str:
    if "Genera" in prompt and "array" in prompt:
        items = []
        for _ in range(4):
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


def _sim_llm() -> AsyncMock:
    provider = AsyncMock()
    provider.generate.side_effect = _fake_generate
    provider.generate_structured.return_value = {
        "utility": 0.7,
        "utility_feedback": "ok",
        "feasibility": 0.6,
        "feasibility_feedback": "ok",
        "market_fit": 0.5,
        "market_feedback": "ok",
        "estimated_complexity": 0.5,
    }
    return provider


def _build_engine(encoder: IdeaEncoder) -> QDEngine:
    """Misma forma que `_build_qd_engine` de la API: agentes nuevos por
    run, encoder COMPARTIDO — mirror del fix de producción
    (`encoders.get_shared_encoder`)."""
    llm = _sim_llm()
    evaluator = EvaluatorOrchestrator(agents={"combined": CombinedEvaluatorAgent(llm)})
    return QDEngine(
        generator=IdeaGeneratorAgent(llm),
        evaluator=evaluator,
        mutation=MutationEngine(llm),
        crossover=CrossoverEngine(llm),
        encoder=encoder,
        repository=None,
    )


async def test_second_run_does_not_grow_memory_vs_first(deterministic_embed) -> None:
    """Dos runs seguidos con el mismo encoder: el segundo no debe pesar
    más que el primero más allá de un margen razonable de ruido."""
    encoder = IdeaEncoder(embed_fn=deterministic_embed)
    request_kwargs: dict = {
        "challenge": "Movilidad urbana sostenible e innovadora",
        "domain": DomainName.GENERIC,
        "population_size": 6,
        "generations": 2,
    }

    gc.collect()
    engine_1 = _build_engine(encoder)
    state_1 = await engine_1.run_evolution(EvolutionRequest(**request_kwargs))
    assert state_1.status == "completed"
    del engine_1, state_1
    gc.collect()
    rss_after_1 = current_rss_mb()

    engine_2 = _build_engine(encoder)
    state_2 = await engine_2.run_evolution(EvolutionRequest(**request_kwargs))
    assert state_2.status == "completed"
    del engine_2, state_2
    gc.collect()
    rss_after_2 = current_rss_mb()

    if rss_after_1 is None or rss_after_2 is None:
        return  # plataforma sin /proc/self/status (no Linux): nada que validar

    growth_mb = rss_after_2 - rss_after_1
    assert growth_mb < 30.0, (
        f"la RAM creció {growth_mb:.1f} MB entre el run 1 y el run 2: "
        "algo está reteniendo referencias entre runs"
    )


async def test_run_evolution_logs_memory_footprint(deterministic_embed) -> None:
    """`run_memory_footprint` debe quedar en el log con ambas medidas, para
    poder verificar la retención de memoria desde los logs sin depender
    del panel de Railway."""
    encoder = IdeaEncoder(embed_fn=deterministic_embed)
    engine = _build_engine(encoder)

    with structlog.testing.capture_logs() as logs:
        state = await engine.run_evolution(
            EvolutionRequest(
                challenge="Movilidad urbana sostenible e innovadora",
                domain=DomainName.GENERIC,
                population_size=4,
                generations=1,
            )
        )

    assert state.status == "completed"
    footprint = [log for log in logs if log.get("event") == "run_memory_footprint"]
    assert len(footprint) == 1
    entry = footprint[0]
    assert "rss_start_mb" in entry
    assert "rss_end_mb" in entry
    assert "rss_delta_mb" in entry
