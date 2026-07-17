"""Diagnóstico de configuración: proveedores, enrutado y base de datos.

Responde en segundos a "¿qué está mal configurado?" en vez de deducirlo
de logs de runs fallidos. Disponible como comando CLI (`doctor`) y como
endpoint (`GET /api/v1/diagnostics`).
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from .core.config import Settings
from .llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


async def check_provider(name: str, provider: LLMProvider) -> dict[str, Any]:
    """Prueba un proveedor con una llamada mínima y clasifica el resultado."""
    from .core.exceptions import LLMAuthError, LLMRateLimitError

    start = time.perf_counter()
    try:
        await provider.generate("Responde solo: ok", max_tokens=5, temperature=0.0)
        return {
            "provider": name,
            "status": "ok",
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "detail": "responde correctamente",
        }
    except LLMAuthError as e:
        return {
            "provider": name,
            "status": "auth_error",
            "detail": f"clave inválida o sin permisos — revisa la variable API_KEY ({e})",
        }
    except LLMRateLimitError as e:
        return {
            "provider": name,
            "status": "rate_limited",
            "detail": f"la clave funciona pero el proveedor está saturado ahora ({e})",
        }
    except Exception as e:
        return {
            "provider": name,
            "status": "error",
            "detail": f"{type(e).__name__}: {e}",
        }


def routing_report(settings: Settings) -> dict[str, Any]:
    """Estado del enrutado por rol tal como lo verá el motor."""
    providers = list(settings.llm)
    spec = settings.routing_spec
    parsed = settings.routing()

    issues: list[str] = []
    if spec and not parsed:
        issues.append(
            "CREATIVE_ROUTING_SPEC tiene contenido pero no se pudo parsear "
            "(formato esperado: 'rol=prov1,prov2;rol2=prov3')"
        )
    for role, chain in parsed.items():
        unknown = [p for p in chain if p not in providers]
        if unknown:
            issues.append(
                f"el rol '{role}' referencia proveedores inexistentes: {unknown} "
                f"(definidos: {providers})"
            )
    if len(providers) > 1 and not parsed:
        issues.append(
            "hay varios proveedores pero sin CREATIVE_ROUTING_SPEC: todos los "
            "roles usarán la cadena por defecto (failover automático en orden "
            f"de definición: {providers})"
        )

    return {
        "spec_raw": spec or "(no definida)",
        "parsed": parsed,
        "default_chain": providers,
        "issues": issues,
    }


async def check_database(settings: Settings) -> dict[str, Any]:
    """Comprueba la conexión a PostgreSQL sin efectos secundarios graves."""
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(settings.database.postgres_url, pool_size=1)
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"status": "ok", "detail": "PostgreSQL responde"}
        finally:
            await engine.dispose()
    except Exception as e:
        return {
            "status": "error",
            "detail": f"{type(e).__name__}: {e}",
        }


async def run_doctor(settings: Settings, check_llm: bool = True) -> dict[str, Any]:
    """Diagnóstico completo. `check_llm=False` evita llamadas a los proveedores."""
    report: dict[str, Any] = {
        "providers_defined": list(settings.llm),
        "routing": routing_report(settings),
        "database": await check_database(settings),
        "provider_checks": [],
    }

    if check_llm and settings.llm:
        providers = {name: LLMProvider(cfg) for name, cfg in settings.llm.items()}
        try:
            for name, provider in providers.items():
                report["provider_checks"].append(await check_provider(name, provider))
        finally:
            for provider in providers.values():
                await provider.close()

    return report
