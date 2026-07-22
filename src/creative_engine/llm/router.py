"""Enrutamiento de LLM por rol con failover entre proveedores.

Problema que resuelve: en free tiers estrictos (Gemini) un run se rompe
por rate limits y sobrecargas. Además, distintos roles del motor tienen
necesidades distintas:

- generator : creatividad → un modelo "listo" (p.ej. Gemini)
- evaluator : volumen alto (≈75% de las llamadas), respuestas cortas →
              un modelo rápido y con límites generosos (p.ej. Groq)
- writer    : calidad de redacción → el mejor disponible

`RoledLLM` expone exactamente la misma interfaz que `LLMProvider`
(`generate`, `generate_structured`, `close`), así que los agentes lo
reciben sin cambiar nada. Por dentro:

1. Elige la cadena de proveedores del rol (o cae a la cadena por defecto).
2. Intenta cada proveedor en orden; si uno agota sus reintentos por rate
   limit / indisponibilidad, salta al siguiente (failover).

Todo se configura por variables de entorno, sin tocar código:
    CREATIVE_LLM__GEMINI__...      (define un proveedor llamado "gemini")
    CREATIVE_LLM__GROQ__...        (define uno llamado "groq")
    CREATIVE_ROUTING__EVALUATOR=groq,gemini
    CREATIVE_ROUTING__GENERATOR=gemini,groq
    CREATIVE_ROUTING__WRITER=gemini
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from ..core.exceptions import (
    LLMAuthError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
)
from .provider import LLMProvider

logger = structlog.get_logger(__name__)

# Roles que el motor puede solicitar. Cualquier otro cae a la cadena default.
# "analyst": Analista Funcional (perfila el reto antes de generar ideas) —
# recomendado un razonador primero (p.ej. CREATIVE_ROUTING_SPEC=analyst=luna,default,zai).
KNOWN_ROLES = ("generator", "evaluator", "writer", "analyst")


class RoledLLM:
    """Vista de un rol sobre el router: se comporta como un LLMProvider.

    Los agentes reciben una de estas instancias y llaman a `generate` /
    `generate_structured` con normalidad. El failover es transparente.
    """

    def __init__(self, router: LLMModelRouter, role: str) -> None:
        self._router = router
        self._role = role

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self._router.run(self._role, "generate", prompt, **kwargs)

    async def generate_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return await self._router.run(self._role, "generate_structured", prompt, **kwargs)

    async def close(self) -> None:
        # El cierre real de proveedores lo gestiona el router (compartidos).
        return None


class _BreakerState:
    """Estado del disyuntor de un proveedor (CLOSED → OPEN → half-open)."""

    __slots__ = ("failures", "open_until")

    def __init__(self) -> None:
        self.open_until: float = 0.0
        self.failures: int = 0


class LLMModelRouter:
    """Orquesta varios proveedores LLM con enrutamiento por rol y failover.

    Incluye un disyuntor por proveedor: tras agotar reintentos por rate
    limit / indisponibilidad / clave inválida, el proveedor entra en
    enfriamiento (60s, duplicándose hasta 300s) y el router lo salta sin
    intentarlo — evita el patrón "reintenta contra el proveedor limitado
    y vuelve a fallar" detectado en producción. Pasado el enfriamiento,
    la siguiente petición actúa de sonda (half-open): si triunfa, el
    disyuntor se cierra; si falla, el enfriamiento se duplica.
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        routing: dict[str, list[str]] | None = None,
    ) -> None:
        if not providers:
            raise LLMError("LLMModelRouter requiere al menos un proveedor")

        self._providers = providers
        # Cadena por defecto: TODOS los proveedores en orden de definición.
        # Así, aunque no haya routing configurado, tener un segundo proveedor
        # ya da failover automático ante saturación del primero.
        self._default_chain: list[str] = list(providers)

        # Cadena por rol; se valida contra proveedores existentes.
        self._routing: dict[str, list[str]] = {}
        for role, chain in (routing or {}).items():
            valid = [name for name in chain if name in providers]
            if valid:
                self._routing[role] = valid
            else:
                logger.warning(
                    "routing_role_no_valid_providers",
                    role=role,
                    requested=chain,
                    available=list(providers),
                )

        self._breakers: dict[str, _BreakerState] = {
            name: _BreakerState() for name in providers
        }
        # Proveedores que rechazan la petición con 400 invalid_request_error
        # no se recuperan con el tiempo (a diferencia de un rate limit): es
        # una incompatibilidad de parámetros con esa API concreta. Se
        # deshabilitan para el resto del run, no solo con enfriamiento.
        self._disabled_for_run: set[str] = set()

        self._log = logger.bind(
            providers=list(providers),
            routing={r: c for r, c in self._routing.items()},
        )
        self._log.info("model_router_ready")

    def for_role(self, role: str) -> RoledLLM:
        """Devuelve una vista tipo LLMProvider para un rol."""
        return RoledLLM(self, role)

    @property
    def total_calls(self) -> int:
        """Llamadas lógicas acumuladas en todos los proveedores del router.

        Usado por el arnés de benchmark (bench/harness.py) para medir el
        coste de cada brazo por diferencia (antes/después de ejecutarlo).
        """
        return sum(p.total_calls for p in self._providers.values())

    @property
    def total_tokens(self) -> dict[str, int]:
        """Tokens acumulados (prompt/completion) en todos los proveedores."""
        return {
            "prompt_tokens": sum(p.total_prompt_tokens for p in self._providers.values()),
            "completion_tokens": sum(
                p.total_completion_tokens for p in self._providers.values()
            ),
        }

    def _chain_for(self, role: str) -> list[str]:
        return self._routing.get(role, self._default_chain)

    async def run(self, role: str, method: str, prompt: str, **kwargs: Any) -> Any:
        """Ejecuta `method` sobre la cadena del rol, con failover.

        Salta al siguiente proveedor ante fallos de disponibilidad
        (rate limit / sobrecarga) y de autenticación (clave inválida):
        una clave rota en un proveedor no debe matar el run si hay otro.
        Otros errores (p.ej. parseo) se propagan sin failover.
        """
        chain = self._chain_for(role)
        last_error: Exception | None = None
        attempted_any = False

        candidates = [name for name in chain if name not in self._disabled_for_run]
        if not candidates:
            raise LLMError(
                f"Todos los proveedores del rol '{role}' están deshabilitados "
                f"para este run (400 invalid_request_error): {chain}",
                details={"role": role, "chain": chain},
            )
        now = time.monotonic()

        # Si TODOS los candidatos están en enfriamiento, forzamos una sonda
        # (half-open) contra el que antes salga del enfriamiento: mejor un
        # intento que fallar la operación sin probar nada.
        if all(now < self._breakers[n].open_until for n in candidates):
            candidates = [min(candidates, key=lambda n: self._breakers[n].open_until)]
            self._log.info(
                "circuit_forced_probe", role=role, provider=candidates[0]
            )

        for idx, name in enumerate(candidates):
            breaker = self._breakers[name]
            now = time.monotonic()

            if now < breaker.open_until and len(candidates) > 1:
                # Disyuntor abierto: saltar sin gastar reintentos.
                self._log.debug(
                    "circuit_open_skipping",
                    role=role,
                    provider=name,
                    remaining_s=round(breaker.open_until - now, 1),
                )
                continue

            provider = self._providers[name]
            fn = getattr(provider, method)
            attempted_any = True
            try:
                result = await fn(prompt, **kwargs)
                if breaker.failures:
                    self._log.info("circuit_closed", provider=name)
                breaker.failures = 0
                breaker.open_until = 0.0
                if idx > 0:
                    self._log.info("failover_succeeded", role=role, provider=name)
                return result
            except (LLMRateLimitError, LLMAuthError) as e:
                last_error = e
                breaker.failures += 1
                cooldown = min(60.0 * (2 ** (breaker.failures - 1)), 300.0)
                breaker.open_until = time.monotonic() + cooldown
                self._log.warning(
                    "provider_unavailable_failover",
                    role=role,
                    provider=name,
                    cooldown_s=round(cooldown),
                    next_provider=(
                        candidates[idx + 1] if idx < len(candidates) - 1 else None
                    ),
                    error=str(e),
                )
                continue
            except LLMInvalidRequestError as e:
                # 400 invalid_request_error no se arregla esperando (a
                # diferencia de un rate limit): deshabilitar el proveedor
                # para el resto del run y rotar al siguiente de la cadena.
                last_error = e
                self._disabled_for_run.add(name)
                self._log.warning(
                    "provider_disabled_for_run",
                    role=role,
                    provider=name,
                    next_provider=(
                        candidates[idx + 1] if idx < len(candidates) - 1 else None
                    ),
                    error=str(e),
                )
                continue
            except LLMError:
                # Error no relacionado con disponibilidad → no hacer failover.
                raise

        # Se agotó toda la cadena por indisponibilidad (o enfriamiento).
        raise LLMError(
            f"Todos los proveedores del rol '{role}' no están disponibles: {chain}"
            + ("" if attempted_any else " (todos en enfriamiento)"),
            details={"role": role, "chain": chain, "last_error": str(last_error)},
        )

    async def close_all(self) -> None:
        for provider in self._providers.values():
            await provider.close()
