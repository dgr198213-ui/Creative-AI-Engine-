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

from typing import Any

import structlog

from ..core.exceptions import LLMError, LLMRateLimitError
from .provider import LLMProvider

logger = structlog.get_logger(__name__)

# Roles que el motor puede solicitar. Cualquier otro cae a la cadena default.
KNOWN_ROLES = ("generator", "evaluator", "writer")


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


class LLMModelRouter:
    """Orquesta varios proveedores LLM con enrutamiento por rol y failover."""

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

        self._log = logger.bind(
            providers=list(providers),
            routing={r: c for r, c in self._routing.items()},
        )
        self._log.info("model_router_ready")

    def for_role(self, role: str) -> RoledLLM:
        """Devuelve una vista tipo LLMProvider para un rol."""
        return RoledLLM(self, role)

    def _chain_for(self, role: str) -> list[str]:
        return self._routing.get(role, self._default_chain)

    async def run(self, role: str, method: str, prompt: str, **kwargs: Any) -> Any:
        """Ejecuta `method` sobre la cadena del rol, con failover.

        Salta al siguiente proveedor solo ante fallos de disponibilidad
        (rate limit / sobrecarga agotados tras reintentos). Otros errores
        (p.ej. parseo) se propagan: no tiene sentido reintentar en otro
        proveedor una respuesta mal formada del prompt.
        """
        chain = self._chain_for(role)
        last_error: Exception | None = None

        for idx, name in enumerate(chain):
            provider = self._providers[name]
            fn = getattr(provider, method)
            try:
                result = await fn(prompt, **kwargs)
                if idx > 0:
                    self._log.info("failover_succeeded", role=role, provider=name)
                return result
            except LLMRateLimitError as e:
                last_error = e
                is_last = idx == len(chain) - 1
                self._log.warning(
                    "provider_unavailable_failover",
                    role=role,
                    provider=name,
                    next_provider=None if is_last else chain[idx + 1],
                    error=str(e),
                )
                continue
            except LLMError:
                # Error no relacionado con disponibilidad → no hacer failover.
                raise

        # Se agotó toda la cadena por indisponibilidad.
        raise LLMError(
            f"Todos los proveedores del rol '{role}' no están disponibles: {chain}",
            details={"role": role, "chain": chain, "last_error": str(last_error)},
        )

    async def close_all(self) -> None:
        for provider in self._providers.values():
            await provider.close()
