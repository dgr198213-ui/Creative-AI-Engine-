"""Validación del pack `tuesdi` (Fase 6, bloque 5).

Prueba de aceptación de la fase: una abstracción con una sola
implementación (generic) no está validada. Este archivo prueba que
`configs/domains/tuesdi/` — visibilidad de artistas independientes — se
carga y funciona de punta a punta usando SOLO el registro dinámico de
domain packs, sin una sola línea de `src/` escrita para este dominio en
particular (el pack se creó copiando el directorio, tal como exige el
diseño).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from creative_engine.agents.combined_evaluator import CombinedEvaluatorAgent
from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
from creative_engine.agents.generator import IdeaGeneratorAgent
from creative_engine.analysis.analyst import FunctionalAnalystAgent
from creative_engine.core.config import Settings, reset_settings
from creative_engine.core.models import EvolutionRequest
from creative_engine.evolution.crossover import CrossoverEngine
from creative_engine.evolution.encoders import IdeaEncoder
from creative_engine.evolution.mutation import MutationEngine
from creative_engine.evolution.qd_engine import QDEngine

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TUESDI_DIR = _REPO_ROOT / "configs" / "domains" / "tuesdi"


def _fresh_settings() -> Settings:
    reset_settings()
    return Settings.load()


def test_tuesdi_pack_directory_exists_and_is_self_contained() -> None:
    """El pack existe y trae sus propios prompts/profile/examples/bench
    — no depende de que un pack 'base' exista para funcionar."""
    assert (_TUESDI_DIR / "domain.yaml").exists()
    assert (_TUESDI_DIR / "prompts" / "generator.md").exists()
    assert (_TUESDI_DIR / "prompts" / "evaluator.md").exists()
    assert (_TUESDI_DIR / "prompts" / "analyst.md").exists()
    assert (_TUESDI_DIR / "profile.yaml").exists()
    assert (_TUESDI_DIR / "examples.yaml").exists()
    assert (_TUESDI_DIR / "bench.yaml").exists()


def test_tuesdi_loads_via_real_settings() -> None:
    settings = _fresh_settings()
    domain = settings.get_domain("tuesdi")

    assert domain.name == "tuesdi"
    assert "TUESDI" in domain.display_name
    assert domain.grid_shape == (10, 10, 8)
    assert "artista independiente" in domain.generator_prompt.lower()
    assert "mánager independiente" in domain.evaluator_prompt.lower()
    assert "discográfica" not in domain.evaluator_prompt.lower()  # regla propia, no genérica
    assert {f["nombre"] for f in domain.profile_fields} == {"tipo_artista", "aforo_tipico"}


def test_tuesdi_pack_appears_in_registry_summary() -> None:
    settings = _fresh_settings()
    pack = settings.get_pack("tuesdi")
    assert pack is not None
    summary = pack.to_summary_dict()
    assert summary["name"] == "tuesdi"
    assert len(summary["examples"]) == 3


_TOPICS = [
    ("Serie de conciertos íntimos", "Micro-conciertos en casas para construir público leal."),
    ("Club de mecenas mensual", "Suscripción para fans que financian el próximo lanzamiento."),
    ("Colaboración cruzada", "Featuring con otro artista independiente del mismo circuito."),
]


def _fake_generate(prompt: str, **kwargs) -> str:
    counter = _fake_generate.counter
    if "Genera" in prompt and "array" in prompt:
        items = []
        for _ in range(3):
            topic = _TOPICS[counter["n"] % len(_TOPICS)]
            counter["n"] += 1
            items.append(
                {
                    "title": f"{topic[0]} v{counter['n']}",
                    "description": topic[1] + f" Variante {counter['n']}.",
                    "advantages": ["Ventaja"],
                    "limitations": ["Límite"],
                    "features": {"technologies": [], "complexity_level": 0.4},
                }
            )
        return json.dumps(items)
    topic = _TOPICS[counter["n"] % len(_TOPICS)]
    counter["n"] += 1
    return json.dumps(
        {
            "title": f"{topic[0]} mutada",
            "description": topic[1] + " Evolución.",
            "advantages": ["Ventaja"],
            "limitations": ["Límite"],
            "mutation_description": "cambio",
        }
    )


_fake_generate.counter = {"n": 0}


async def test_tuesdi_full_run_completes_with_mocked_llm(deterministic_embed) -> None:
    """Motor QD completo sobre el dominio tuesdi, LLM simulado: prueba
    dura de la fase — funciona sin ninguna línea de src/ específica de
    este dominio."""
    _fresh_settings()

    llm = AsyncMock()
    llm.generate.side_effect = _fake_generate
    llm.generate_structured.return_value = {
        "utility": 0.65,
        "utility_feedback": "ok",
        "feasibility": 0.6,
        "feasibility_feedback": "ok",
        "market_fit": 0.5,
        "market_feedback": "ok",
        "estimated_complexity": 0.4,
    }

    evaluator = EvaluatorOrchestrator(agents={"combined": CombinedEvaluatorAgent(llm)})
    engine = QDEngine(
        generator=IdeaGeneratorAgent(llm),
        evaluator=evaluator,
        mutation=MutationEngine(llm),
        crossover=CrossoverEngine(llm),
        encoder=IdeaEncoder(embed_fn=deterministic_embed),
        repository=None,
    )

    request = EvolutionRequest(
        challenge="Toco en bares pequeños pero nadie fuera de mis amigos viene a verme",
        domain="tuesdi",
        population_size=6,
        generations=1,
    )
    state = await engine.run_evolution(request)

    assert state.status == "completed"
    assert state.domain == "tuesdi"
    assert len(state.archive) > 0

    # El evaluador recibió la rúbrica del pack (no la del comité genérico).
    eval_prompt = llm.generate_structured.call_args.kwargs["system_prompt"]
    assert "mánager independiente" in eval_prompt.lower()


async def test_tuesdi_analyst_extends_profile_with_domain_fields() -> None:
    settings = _fresh_settings()
    domain = settings.get_domain("tuesdi")

    llm = AsyncMock()
    llm.generate_structured.return_value = {
        "reto_reformulado": "Conseguir público nuevo más allá del círculo cercano",
        "dominio": {"tipo_artista": "música, indie folk", "aforo_tipico": "30-40 personas"},
    }
    agent = FunctionalAnalystAgent(llm)

    profile = await agent.analyze(
        "Toco en bares pequeños pero nadie fuera de mis amigos viene a verme",
        domain=domain,
    )

    assert profile.dominio == {
        "tipo_artista": "música, indie folk",
        "aforo_tipico": "30-40 personas",
    }
    system_prompt = llm.generate_structured.call_args.kwargs["system_prompt"]
    assert "descubrimiento" in system_prompt.lower()
