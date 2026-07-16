"""Dependencias compartidas de la API."""

from __future__ import annotations

from fastapi import HTTPException, Request

from ..memory.repository import IdeaRepository


def require_repo(request: Request) -> IdeaRepository:
    """Devuelve el repositorio o 503 si no hay persistencia disponible.

    En despliegues sin base de datos lista, el motor sigue generando
    ideas en vivo (streaming), pero los endpoints que consultan histórico
    necesitan persistencia y responden 503 de forma explícita.
    """
    repo = getattr(request.app.state, "repository", None)
    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="La base de datos no está disponible; el histórico y los informes "
            "requieren persistencia. La generación en vivo sí funciona.",
        )
    return repo
