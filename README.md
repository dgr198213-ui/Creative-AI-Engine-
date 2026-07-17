# Creative AI Engine

Motor de generación creativa que, ante un reto, necesidad o problema, devuelve **un abanico de ideas élite genuinamente distintas entre sí** — no una única respuesta.

En lugar de pedir "la mejor idea" a un LLM (que converge hacia lo típico), el motor mantiene un ecosistema evolutivo donde las ideas compiten, se combinan y evolucionan, conservando la mejor propuesta de cada región del espacio creativo (Quality-Diversity / MAP-Elites).

## Cómo funciona

```
Reto del usuario
      │
      ▼
Generador (LLM) ──── población inicial diversa
      │
      ▼
Agentes de calidad ── utilidad · viabilidad · mercado  (LLM)
      │
      ▼
Codificador ───────── embedding (genoma) + descriptor semántico
      │
      ▼
Novedad objetiva ──── distancia de embedding al archivo (sin LLM)
      │
      ▼
MAP-Elites ────────── cada celda conserva su mejor idea
      │
      ▼
Mutación + Cruce ──── operadores guiados por LLM
      │                (nueva generación → repite el ciclo)
      ▼
Abanico final de ideas élite diversas
```

## Decisiones de diseño clave

| Decisión | Motivo |
|---|---|
| **Novedad objetiva** (distancia coseno al archivo de élites) | Un LLM puntuando "novedad" sin ver la población da ruido no calibrado. La distancia de embedding es medible y reproducible. |
| **Descriptor de comportamiento desde el embedding** (proyección determinista 384→3 dims) | Si el grid se construye con puntuaciones, dos ideas semánticamente opuestas compiten por la misma celda. Con embeddings, la diversidad del archivo es diversidad de contenido real. |
| **Fitness = calidad pura** (utilidad, viabilidad, mercado, impacto) | MAP-Elites ya garantiza la diversidad; mezclar novedad en el fitness la contaría dos veces. |
| **Mutación/cruce guiados por LLM** | Operadores semánticos coherentes en lugar de aleatorios (enfoque "Evolutionary Thoughts", arXiv:2505.05756). |
| **Defaults económicos** (población 20 × 10 generaciones) | Cada idea ≈ 4 llamadas LLM. Los defaults mantienen un run en cientos de llamadas, no decenas de miles. |

## Enrutamiento multi-proveedor (opcional)

El motor puede usar **varios proveedores LLM a la vez**, asignando cada rol
al modelo que más le conviene, con **failover automático**: si un proveedor
se satura (rate limit / 503), la petición salta al siguiente de su lista.

Roles: `generator` (creatividad), `evaluator` (volumen, ≈75% de las
llamadas → conviene un modelo rápido) y `writer` (redacción).

Ejemplo: Gemini para generar, Groq (rápido, free tier generoso) para
evaluar, con failover cruzado. Solo variables de entorno, sin tocar código:

```bash
# Proveedor Gemini
CREATIVE_LLM__GEMINI__API_KEY=...
CREATIVE_LLM__GEMINI__BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
CREATIVE_LLM__GEMINI__MODEL=gemini-flash-latest
# Proveedor Groq
CREATIVE_LLM__GROQ__API_KEY=...
CREATIVE_LLM__GROQ__BASE_URL=https://api.groq.com/openai/v1
CREATIVE_LLM__GROQ__MODEL=llama-3.3-70b-versatile
# Enrutado por rol con failover
CREATIVE_ROUTING_SPEC=generator=gemini,groq;evaluator=groq,gemini;writer=gemini
```

Con un solo proveedor no hace falta configurar nada: todos los roles lo usan.

## Requisitos

- Python ≥ 3.12
- PostgreSQL 16 (persistencia — opcional con `--no-db`)
- Neo4j 5 (grafo de relaciones — opcional)
- Una API key de cualquier proveedor OpenAI-compatible

## Inicio rápido

```bash
# 1. Configuración
cp .env.example .env        # edita tu API key

# 2. Infraestructura (opcional para probar: existe --no-db)
docker compose up -d postgres neo4j redis

# 3. Instalación
pip install -e ".[dev]"

# 4. Evolución desde CLI (sin base de datos)
creative-engine evolve \
  --challenge "Diseña una bicicleta eléctrica urbana innovadora" \
  --domain industrial_design \
  --population 12 --generations 3 \
  --no-db

# 5. O servidor API + panel web
creative-engine serve       # → http://localhost:8000  (panel)
                            # → http://localhost:8000/docs  (API)
```

## Panel web

Servido en `http://localhost:8000` cuando el motor está en marcha. Una sola pantalla, sin jerga técnica, que fluye entre tres momentos:

1. **Pregunta** — un campo grande ("¿Qué necesitas resolver?"), selector de ámbito (General / Producto / Marketing) y el botón *Generar ideas*.
2. **En vivo** — barra de progreso real y las ideas apareciendo agrupadas por enfoque conforme el motor las descubre (streaming SSE). La espera deja de ser muerta.
3. **Resultado** — el abanico organizado: "N enfoques distintos, el mejor de cada uno". Cada tarjeta expande sus variantes y ofrece *Generar informe* bajo demanda.

## API

| Endpoint | Descripción |
|---|---|
| `POST /api/v1/evolution/stream` | **Streaming SSE**: lanza la evolución y transmite el progreso e ideas en vivo |
| `POST /api/v1/evolution/start` | Ejecuta una evolución (síncrono) y devuelve las top ideas |
| `GET /api/v1/evolution/{run_id}` | Resumen de una ejecución persistida |
| `GET /api/v1/runs/{run_id}/elites` | El abanico completo de ideas élite |
| `GET /api/v1/runs/{run_id}/families` | Élites agrupadas en familias automáticas de enfoques |
| `POST /api/v1/ideas/{idea_id}/report` | Informe ejecutivo de una idea (bajo demanda) |
| `GET /api/v1/ideas/{idea_id}` | Detalle, relacionadas y linaje evolutivo |
| `GET /api/v1/memory/recommendations/{idea_id}` | Ideas relacionadas (similitud + diversidad) |
| `GET /api/v1/stats` | Estadísticas globales o por run |

## Dominios

Cada dominio se define en un YAML de `configs/` (prompt del sistema, pesos de evaluación, mutaciones permitidas, tamaño del grid). Incluidos: `generic`, `industrial_design`, `marketing`. Añadir un dominio nuevo = añadir un YAML.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v   # 48 tests, sin red ni BD
```

La suite incluye un test de integración del ciclo evolutivo completo con LLM simulado y embeddings deterministas.

## Estructura

```
src/creative_engine/
├── core/          # modelos Pydantic, config, eventos, excepciones
├── evolution/     # MAP-Elites, encoders (novedad objetiva), mutación, cruce, motor QD
├── agents/        # generador, evaluadores (utilidad/viabilidad/mercado), crítico, escritor
├── llm/           # abstracción de proveedores OpenAI-compatibles
├── memory/        # repositorio PostgreSQL, grafo Neo4j, recomendación
└── api/           # FastAPI: evolución, streaming SSE, ideas, memoria
    └── static/    # panel web (index.html + app.js)
```

## Roadmap post-MVP

1. Búsqueda vectorial real con pgvector (el schema ya lo contempla)
2. Persistencia del progreso del stream para reconexión (reload sin perder el run)
3. Coevolución adversaria (Generational Adversarial MAP-Elites, arXiv:2505.06617)
4. Learned QD para meta-aprendizaje de reglas de exploración (arXiv:2502.02190)

## Licencia

Propietaria.
