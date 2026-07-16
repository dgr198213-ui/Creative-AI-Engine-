FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# hatchling necesita src/ presente en el build
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Railway/PaaS asignan el puerto por la variable PORT.
ENV PORT=8000
EXPOSE 8000

# Shell form para que $PORT se expanda en tiempo de ejecución.
CMD creative-engine serve --host 0.0.0.0 --port ${PORT:-8000}
