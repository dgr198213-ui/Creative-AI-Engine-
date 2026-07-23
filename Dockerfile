# ── Build stage: aquí vive build-essential, no en la imagen final ──
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# hatchling necesita src/ presente en el build
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# ── Runtime stage: sin build-essential (auditoría M1: ~250 MB menos,
# menor superficie de ataque en la imagen que corre en producción) ──
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY configs/ configs/

# Precargar el modelo de embeddings en la imagen: elimina la descarga de
# HuggingFace en el primer run (latencia + dependencia de red en runtime).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Logs sin buffering: los timestamps llegan al colector sin retraso.
ENV PYTHONUNBUFFERED=1

# Domain packs (Fase 6): el paquete se instala vía pip en /usr/local
# (site-packages), separado de configs/, que vive en /app. La heurística
# de core/config.py basada en __file__ asume un checkout del repo y no
# llega hasta aquí — sin esta variable, el registro no encontraba ningún
# pack y arrancaba en silencio con un solo dominio embebido (incidente
# de producción 23-jul-2026). Explícito > adivinado.
ENV CREATIVE_CONFIGS_DIR=/app/configs

# Railway/PaaS asignan el puerto por la variable PORT.
ENV PORT=8000
EXPOSE 8000

# Shell form para que $PORT se expanda en tiempo de ejecución.
CMD creative-engine serve --host 0.0.0.0 --port ${PORT:-8000}
