"""Aplicación FastAPI del Creative AI Engine."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


def _warn_if_api_key_missing(settings: Settings) -> None:
    """Aviso explícito en el log si /api/v1/* queda sin autenticar.

    Auditoría C1 / Fase 5 bloque 2: sin CREATIVE_API_KEY, la API arranca
    igual (no rompe uso local/tests) pero deja constancia en el log en
    vez de un silencio que solo se nota al auditar — importante ahora
    que terra y luna facturan dinero real.
    """
    if not settings.api_key:
        logger.warning("api_key_not_configured_endpoints_open")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifecycle: inicializar y limpiar recursos.

    El arranque tolera que la base de datos no esté lista todavía
    (Railway/PaaS no garantizan orden de arranque): reintenta la
    inicialización con backoff y, si aun así falla, arranca sin
    persistencia en vez de caerse — el motor sigue devolviendo ideas.
    """
    import asyncio

    settings = get_settings()
    logger.info("starting_creative_engine", debug=settings.debug)
    _warn_if_api_key_missing(settings)

    from ..memory.repository import IdeaRepository

    repo: IdeaRepository | None = None
    for attempt in range(1, 6):
        try:
            candidate = IdeaRepository()
            await candidate.initialize()
            repo = candidate
            break
        except Exception as e:
            wait = min(2**attempt, 20)
            logger.warning(
                "repository_init_retry",
                attempt=attempt,
                wait_s=wait,
                error=str(e),
            )
            await asyncio.sleep(wait)

    if repo is None:
        logger.warning("repository_unavailable_starting_without_persistence")

    app.state.repository = repo

    logger.info("creative_engine_ready", persistence=repo is not None)
    yield

    if repo is not None:
        await repo.close()
    logger.info("creative_engine_shutdown")


def create_app() -> FastAPI:
    """Factory de la aplicación FastAPI."""
    settings = get_settings()

    app = FastAPI(
        title="Creative AI Engine",
        description=(
            "Motor de generación creativa: múltiples ideas élite y diversas "
            "para un reto, mediante Quality-Diversity y agentes LLM"
        ),
        version=__version__,
        lifespan=lifespan,
        # Auditoría A2: el esquema OpenAPI interactivo no debe quedar expuesto
        # en producción (combinado con una API sin auth, es un mapa completo
        # de la superficie de ataque). Solo se sirve con debug=True.
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
    )

    from .guardrails import InMemoryRateLimiter

    app.state.rate_limiter = InMemoryRateLimiter()

    # Auditoría C1: API key obligatoria en /api/v1/* si CREATIVE_API_KEY está
    # configurada (añadido antes que CORS para no autenticar preflight OPTIONS,
    # que el middleware ya deja pasar explícitamente).
    from .auth import ApiKeyMiddleware

    app.add_middleware(ApiKeyMiddleware)

    # CORS: abierto en desarrollo; restringir orígenes antes de exponer públicamente.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .routes.analysis import router as analysis_router
    from .routes.budget import router as budget_router
    from .routes.diagnostics import router as diagnostics_router
    from .routes.domains import router as domains_router
    from .routes.evolution import router as evolution_router
    from .routes.ideas import router as ideas_router
    from .routes.memory import router as memory_router
    from .routes.stream import router as stream_router

    app.include_router(evolution_router, prefix="/api/v1", tags=["Evolution"])
    app.include_router(stream_router, prefix="/api/v1", tags=["Streaming"])
    app.include_router(ideas_router, prefix="/api/v1", tags=["Ideas"])
    app.include_router(memory_router, prefix="/api/v1", tags=["Memory"])
    app.include_router(diagnostics_router, prefix="/api/v1", tags=["Diagnostics"])
    app.include_router(analysis_router, prefix="/api/v1", tags=["Analysis"])
    app.include_router(budget_router, prefix="/api/v1", tags=["Budget"])
    app.include_router(domains_router, prefix="/api/v1", tags=["Domains"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "engine": f"Creative AI Engine v{__version__}"}

    # Panel web (una sola pantalla) servido como estáticos
    from pathlib import Path

    from fastapi import Response
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> Response:
            # 204 en vez de 404: sin favicon dedicado, sin ruido en los logs.
            return Response(status_code=204)

    return app


app = create_app()
