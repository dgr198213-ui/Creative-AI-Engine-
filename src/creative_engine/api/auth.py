"""Autenticación por API key para la API pública (auditoría C1).

Sin `CREATIVE_API_KEY` configurada, la API queda abierta (uso local,
tests, CI): el fix solo entra en juego cuando el operador define la
clave, que es justamente cuando la API queda expuesta a internet
(Railway) sin nada más delante.

Acepta la clave por cabecera `X-API-Key` (uso normal desde JS/curl) o por
query param `api_key` (necesario para el enlace de descarga del export,
que el navegador sigue como GET simple sin poder añadir cabeceras).
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..core.config import get_settings

# Rutas públicas incluso con API key activa: healthcheck (lo consulta
# Railway sin credenciales) y el panel estático (HTML/JS, no consume LLM).
_PUBLIC_PATHS = {"/health", "/", "/favicon.ico"}
_PUBLIC_PREFIXES = ("/static/",)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Exige X-API-Key en /api/v1/* cuando CREATIVE_API_KEY está configurada."""

    async def dispatch(self, request: Request, call_next) -> Response:
        api_key = get_settings().api_key
        if not api_key:
            return await call_next(request)

        path = request.url.path
        if (
            request.method == "OPTIONS"  # preflight CORS: no lleva credenciales
            or path in _PUBLIC_PATHS
            or path.startswith(_PUBLIC_PREFIXES)
        ):
            return await call_next(request)

        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key") or ""
        if not hmac.compare_digest(provided, api_key):
            return JSONResponse(
                {
                    "detail": "API key inválida o ausente. Añade la cabecera "
                    "X-API-Key (o ?api_key= en enlaces de descarga)."
                },
                status_code=401,
            )

        return await call_next(request)
