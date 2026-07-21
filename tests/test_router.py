"""Tests del enrutamiento por rol y failover entre proveedores."""

from unittest.mock import AsyncMock

import pytest

from creative_engine.core.config import Settings
from creative_engine.core.exceptions import LLMError, LLMRateLimitError
from creative_engine.llm.router import LLMModelRouter


def _provider(name: str, generate_return="ok") -> AsyncMock:
    p = AsyncMock()
    p.generate.return_value = generate_return
    p.generate_structured.return_value = {"provider": name}
    p.close = AsyncMock()
    return p


class TestRoutingSpec:
    def test_parse_multiple_roles(self) -> None:
        s = Settings()
        s.routing_spec = "evaluator=groq,gemini;generator=gemini,groq;writer=gemini"
        assert s.routing() == {
            "evaluator": ["groq", "gemini"],
            "generator": ["gemini", "groq"],
            "writer": ["gemini"],
        }

    def test_parse_empty(self) -> None:
        assert Settings().routing() == {}

    def test_parse_ignores_malformed(self) -> None:
        s = Settings()
        s.routing_spec = "evaluator=groq; ; garbage ;writer="
        assert s.routing() == {"evaluator": ["groq"]}


class TestModelRouter:
    def test_default_chain_when_no_routing(self) -> None:
        providers = {"gemini": _provider("gemini")}
        router = LLMModelRouter(providers)
        assert router._chain_for("evaluator") == ["gemini"]

    def test_default_chain_includes_all_providers(self) -> None:
        """Sin routing configurado, la cadena por defecto incluye todos los
        proveedores → failover automático sin configuración."""
        providers = {"gemini": _provider("gemini"), "groq": _provider("groq")}
        router = LLMModelRouter(providers)
        assert router._chain_for("generator") == ["gemini", "groq"]

    async def test_failover_works_without_routing_spec(self) -> None:
        """Reproduce el fallo de producción: routing={} pero dos proveedores.
        Antes el failover no saltaba; ahora debe saltar al segundo."""
        p1 = _provider("gemini")
        p1.generate.side_effect = LLMRateLimitError("saturado")
        p2 = _provider("groq", generate_return="respuesta de groq")
        router = LLMModelRouter({"gemini": p1, "groq": p2})  # sin routing
        result = await router.for_role("generator").generate("x")
        assert result == "respuesta de groq"

    def test_role_uses_its_chain(self) -> None:
        providers = {"gemini": _provider("gemini"), "groq": _provider("groq")}
        router = LLMModelRouter(providers, routing={"evaluator": ["groq", "gemini"]})
        assert router._chain_for("evaluator") == ["groq", "gemini"]
        # rol sin cadena → default = todos los proveedores (failover automático)
        assert router._chain_for("writer") == ["gemini", "groq"]

    def test_invalid_provider_in_chain_dropped(self) -> None:
        providers = {"gemini": _provider("gemini")}
        router = LLMModelRouter(providers, routing={"evaluator": ["nonexistent", "gemini"]})
        assert router._chain_for("evaluator") == ["gemini"]

    async def test_for_role_returns_working_view(self) -> None:
        providers = {"groq": _provider("groq", generate_return="hola")}
        router = LLMModelRouter(providers, routing={"generator": ["groq"]})
        llm = router.for_role("generator")
        result = await llm.generate("prompt")
        assert result == "hola"
        providers["groq"].generate.assert_awaited_once()

    async def test_failover_on_rate_limit(self) -> None:
        """Si el primer proveedor agota rate limit, salta al siguiente."""
        p1 = _provider("gemini")
        p1.generate.side_effect = LLMRateLimitError("saturado")
        p2 = _provider("groq", generate_return="respuesta de groq")

        router = LLMModelRouter(
            {"gemini": p1, "groq": p2},
            routing={"evaluator": ["gemini", "groq"]},
        )
        llm = router.for_role("evaluator")
        result = await llm.generate("evalúa")

        assert result == "respuesta de groq"
        p1.generate.assert_awaited_once()
        p2.generate.assert_awaited_once()

    async def test_no_failover_on_non_availability_error(self) -> None:
        """Errores que no son de disponibilidad NO hacen failover."""
        p1 = _provider("gemini")
        p1.generate.side_effect = LLMError("prompt inválido")
        p2 = _provider("groq")

        router = LLMModelRouter(
            {"gemini": p1, "groq": p2},
            routing={"generator": ["gemini", "groq"]},
        )
        llm = router.for_role("generator")

        with pytest.raises(LLMError):
            await llm.generate("x")
        p2.generate.assert_not_awaited()  # no se intentó el segundo

    async def test_all_providers_exhausted_raises(self) -> None:
        p1 = _provider("gemini")
        p1.generate.side_effect = LLMRateLimitError("saturado")
        p2 = _provider("groq")
        p2.generate.side_effect = LLMRateLimitError("saturado")

        router = LLMModelRouter(
            {"gemini": p1, "groq": p2},
            routing={"evaluator": ["gemini", "groq"]},
        )
        llm = router.for_role("evaluator")

        with pytest.raises(LLMError, match="no están disponibles"):
            await llm.generate("x")

    async def test_close_all(self) -> None:
        providers = {"a": _provider("a"), "b": _provider("b")}
        router = LLMModelRouter(providers)
        await router.close_all()
        providers["a"].close.assert_awaited_once()
        providers["b"].close.assert_awaited_once()

    def test_empty_providers_raises(self) -> None:
        with pytest.raises(LLMError):
            LLMModelRouter({})


class TestRouterInQDCycle:
    """El router debe funcionar como LLM en el ciclo QD completo."""

    async def test_full_cycle_routes_by_role(self, deterministic_embed) -> None:
        import json

        from creative_engine.agents.evaluator_orchestrator import EvaluatorOrchestrator
        from creative_engine.agents.feasibility import FeasibilityAgent
        from creative_engine.agents.generator import IdeaGeneratorAgent
        from creative_engine.agents.innovation import InnovationAgent
        from creative_engine.agents.market import MarketAgent
        from creative_engine.core.models import DomainName, EvolutionRequest
        from creative_engine.evolution import encoders as enc
        from creative_engine.evolution.crossover import CrossoverEngine
        from creative_engine.evolution.encoders import IdeaEncoder
        from creative_engine.evolution.mutation import MutationEngine
        from creative_engine.evolution.qd_engine import QDEngine

        def gen(prompt, **kw):
            if "array" in prompt:
                return json.dumps(
                    [
                        {
                            "title": f"Idea {i} sobre movilidad",
                            "description": f"Descripción diversa número {i} para el reto.",
                            "advantages": ["A"],
                            "limitations": ["X"],
                            "features": {"complexity_level": 0.5},
                        }
                        for i in range(3)
                    ]
                )
            return json.dumps(
                {
                    "title": "Idea mutada distinta",
                    "description": "Una evolución con un giro inesperado y diferente.",
                    "advantages": ["Aa"],
                    "limitations": ["Xx"],
                    "mutation_description": "cambio",
                }
            )

        # generator: solo gemini responde generate; evaluator: solo groq responde structured
        gemini = _provider("gemini")
        gemini.generate.side_effect = gen
        gemini.generate_structured.side_effect = LLMRateLimitError("saturado")
        groq = _provider("groq")
        groq.generate.side_effect = gen
        groq.generate_structured.return_value = {
            "score": 0.7,
            "feedback": "ok",
            "estimated_complexity": 0.5,
        }

        router = LLMModelRouter(
            {"gemini": gemini, "groq": groq},
            routing={"generator": ["gemini", "groq"], "evaluator": ["groq", "gemini"]},
        )

        orig = enc.IdeaEncoder._embed
        enc.IdeaEncoder._embed = lambda self, text: deterministic_embed(text)
        try:
            evaluator = EvaluatorOrchestrator(
                agents={
                    "innovation": InnovationAgent(router.for_role("evaluator")),
                    "feasibility": FeasibilityAgent(router.for_role("evaluator")),
                    "market": MarketAgent(router.for_role("evaluator")),
                }
            )
            engine = QDEngine(
                generator=IdeaGeneratorAgent(router.for_role("generator")),
                evaluator=evaluator,
                mutation=MutationEngine(router.for_role("generator")),
                crossover=CrossoverEngine(router.for_role("generator")),
                encoder=IdeaEncoder(),
                repository=None,
            )
            state = await engine.run_evolution(
                EvolutionRequest(
                    challenge="Movilidad urbana sostenible e innovadora",
                    domain=DomainName.GENERIC,
                    population_size=6,
                    generations=2,
                )
            )
        finally:
            enc.IdeaEncoder._embed = orig
            await router.close_all()

        assert len(state.archive) >= 2
        # Todas las élites evaluadas con calidad real (vino de groq)
        evaluated = [c for c in state.archive if c.elite.evaluation and c.elite.evaluation.utility > 0]
        assert len(evaluated) == len(state.archive)
        assert groq.generate_structured.await_count > 0


class TestInvalidRequestDisablesProvider:
    """400 invalid_request_error: rotar y deshabilitar para el resto del run."""

    async def test_400_invalid_request_rotates_to_next_provider(self) -> None:
        from creative_engine.core.exceptions import LLMInvalidRequestError

        p1 = _provider("terra")
        p1.generate.side_effect = LLMInvalidRequestError("max_tokens no soportado")
        p2 = _provider("zai", generate_return="respuesta de zai")

        router = LLMModelRouter(
            {"terra": p1, "zai": p2},
            routing={"generator": ["terra", "zai"]},
        )
        result = await router.for_role("generator").generate("x")

        assert result == "respuesta de zai"
        p1.generate.assert_awaited_once()
        p2.generate.assert_awaited_once()

    async def test_disabled_provider_stays_disabled_for_rest_of_run(self) -> None:
        """A diferencia del disyuntor por rate limit, no hay enfriamiento:
        el proveedor no se vuelve a intentar en todo el run."""
        from creative_engine.core.exceptions import LLMInvalidRequestError

        p1 = _provider("terra")
        p1.generate.side_effect = LLMInvalidRequestError("max_tokens no soportado")
        p2 = _provider("zai", generate_return="respuesta de zai")

        router = LLMModelRouter(
            {"terra": p1, "zai": p2},
            routing={"generator": ["terra", "zai"]},
        )
        llm = router.for_role("generator")

        await llm.generate("primera")
        await llm.generate("segunda")

        # terra solo se intentó una vez: quedó deshabilitado para el run.
        assert p1.generate.await_count == 1
        assert p2.generate.await_count == 2

    async def test_all_providers_invalid_request_raises_clean_error(self) -> None:
        from creative_engine.core.exceptions import LLMInvalidRequestError

        p1 = _provider("terra")
        p1.generate.side_effect = LLMInvalidRequestError("max_tokens no soportado")

        router = LLMModelRouter({"terra": p1})
        with pytest.raises(LLMError):
            await router.for_role("generator").generate("x")

    async def test_disabled_for_run_tracked_internally(self) -> None:
        """El proveedor que devuelve 400 invalid_request_error se marca
        internamente como deshabilitado para el run (provider_disabled_for_run)."""
        from creative_engine.core.exceptions import LLMInvalidRequestError

        p1 = _provider("terra")
        p1.generate.side_effect = LLMInvalidRequestError("max_tokens no soportado")
        p2 = _provider("zai", generate_return="ok")

        router = LLMModelRouter(
            {"terra": p1, "zai": p2}, routing={"generator": ["terra", "zai"]}
        )
        assert "terra" not in router._disabled_for_run
        await router.for_role("generator").generate("x")
        assert "terra" in router._disabled_for_run


class TestCircuitBreaker:
    """Disyuntor por proveedor: no reintentar contra proveedores caídos."""

    async def test_open_breaker_skips_provider(self) -> None:
        """Tras fallar, el proveedor entra en enfriamiento y la siguiente
        operación va DIRECTA al alternativo sin intentar el caído."""
        p1 = _provider("gemini")
        p1.generate.side_effect = LLMRateLimitError("saturado")
        p2 = _provider("groq", generate_return="ok groq")

        router = LLMModelRouter(
            {"gemini": p1, "groq": p2},
            routing={"generator": ["gemini", "groq"]},
        )
        llm = router.for_role("generator")

        # 1ª llamada: gemini falla (abre disyuntor), failover a groq
        assert await llm.generate("a") == "ok groq"
        assert p1.generate.await_count == 1

        # 2ª llamada: gemini está en enfriamiento → NO se intenta
        assert await llm.generate("b") == "ok groq"
        assert p1.generate.await_count == 1  # sigue en 1: no se reintentó
        assert p2.generate.await_count == 2

    async def test_breaker_closes_after_cooldown_success(self) -> None:
        """Pasado el enfriamiento, la sonda (half-open) puede cerrar el circuito."""
        p1 = _provider("gemini")
        p1.generate.side_effect = [LLMRateLimitError("saturado"), "gemini ok"]
        p2 = _provider("groq", generate_return="ok groq")

        router = LLMModelRouter(
            {"gemini": p1, "groq": p2},
            routing={"generator": ["gemini", "groq"]},
        )
        llm = router.for_role("generator")

        await llm.generate("a")  # abre el disyuntor de gemini
        # Simular que el enfriamiento expiró
        router._breakers["gemini"].open_until = 0.0

        result = await llm.generate("b")  # sonda: gemini responde
        assert result == "gemini ok"
        assert router._breakers["gemini"].failures == 0  # circuito cerrado

    async def test_all_open_forces_probe(self) -> None:
        """Con todos los proveedores en enfriamiento, se fuerza una sonda
        en vez de fallar la operación sin intentar nada."""
        p1 = _provider("gemini", generate_return="volvió gemini")
        router = LLMModelRouter({"gemini": p1})
        # Forzar disyuntor abierto lejos en el futuro
        import time as _time

        router._breakers["gemini"].open_until = _time.monotonic() + 999
        router._breakers["gemini"].failures = 1

        result = await router.for_role("generator").generate("x")
        assert result == "volvió gemini"

    async def test_repeated_failures_double_cooldown(self) -> None:
        p1 = _provider("gemini")
        p1.generate.side_effect = LLMRateLimitError("saturado")
        router = LLMModelRouter({"gemini": p1})
        llm = router.for_role("generator")

        import contextlib
        import time as _time

        for expected_min in (60, 120):
            router._breakers["gemini"].open_until = 0.0  # permitir sonda
            with contextlib.suppress(Exception):
                await llm.generate("x")
            remaining = router._breakers["gemini"].open_until - _time.monotonic()
            assert remaining >= expected_min - 2, (
                f"enfriamiento esperado ≥{expected_min}s, quedó {remaining:.0f}s"
            )
