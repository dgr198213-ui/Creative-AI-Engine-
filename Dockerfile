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

EXPOSE 8000

CMD ["creative-engine", "serve", "--host", "0.0.0.0", "--port", "8000"]
