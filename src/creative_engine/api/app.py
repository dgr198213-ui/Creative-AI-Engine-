"""Aplicación FastAPI del Creative AI Engine."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..core.config import get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifecycle: inicializar y limpiar recursos."""
    settings = get_settings()
    logger.info("starting_creative_engine", debug=settings.debug)

    from ..memory.repository import IdeaRepository

    repo = IdeaRepository()
    await repo.initialize()
    app.state.repository = repo

    logger.info("creative_engine_ready")
    yield

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
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS: abierto en desarrollo; restringir orígenes antes de exponer públicamente.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .routes.evolution import router as evolution_router
    from .routes.ideas import router as ideas_router
    from .routes.memory import router as memory_router

    app.include_router(evolution_router, prefix="/api/v1", tags=["Evolution"])
    app.include_router(ideas_router, prefix="/api/v1", tags=["Ideas"])
    app.include_router(memory_router, prefix="/api/v1", tags=["Memory"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "engine": f"Creative AI Engine v{__version__}"}

    return app


app = create_app()
