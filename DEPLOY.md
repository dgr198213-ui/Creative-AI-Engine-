# Despliegue en Railway

> **Por qué no Vercel:** este motor mantiene conexiones SSE abiertas durante
> minutos (el streaming en vivo) y necesita PostgreSQL persistente. Las
> funciones serverless de Vercel tienen límite de duración y hacen buffering,
> así que el streaming no funciona ahí. Railway ejecuta el contenedor de
> forma persistente y aloja la base de datos al lado.

## Pasos

### 1. Servicio de la app (desde GitHub)

1. En Railway: **New Project → Deploy from GitHub repo** → elige `Creative-AI-Engine-`.
2. Railway detecta el `Dockerfile` y `railway.json` automáticamente y construye la imagen.
3. En **Settings → Networking → Generate Domain** para obtener la URL pública.

### 2. Base de datos PostgreSQL (gestionada)

1. En el proyecto: **+ New → Database → Add PostgreSQL**.
2. Railway crea la BD e inyecta la variable `DATABASE_URL` automáticamente.
3. El motor la lee y le añade el driver `asyncpg` solo (ver `core/config.py`).

> Neo4j y Redis son **opcionales**. El motor arranca y genera ideas sin ellos;
> solo el grafo de relaciones (Neo4j) y algún cache (Redis) quedan inactivos.
> Si los quieres: **+ New → Database → Redis**, y para Neo4j
> **+ New → Docker Image → `neo4j:5.27-community`** con sus variables.

### 3. Variables de entorno (Settings → Variables)

Obligatoria — tu proveedor LLM (cualquier API OpenAI-compatible):

```
CREATIVE_LLM__DEFAULT__NAME=openai
CREATIVE_LLM__DEFAULT__API_KEY=sk-...
CREATIVE_LLM__DEFAULT__MODEL=gpt-4o-mini
CREATIVE_LLM__DEFAULT__MAX_CONCURRENT=5
```

`DATABASE_URL` la pone Railway sola si vinculas la BD (usa una *reference
variable* al servicio Postgres). No hace falta definir `PORT`: Railway lo
inyecta y el contenedor lo respeta.

### 4. Comprobar

- `https://<tu-dominio>.railway.app/` → el panel.
- `https://<tu-dominio>.railway.app/health` → `{"status":"ok"}`.
- `https://<tu-dominio>.railway.app/docs` → la API.

## Notas de robustez ya incluidas

- **Arranque tolerante:** si la BD no está lista, la app reintenta con backoff
  y, si aun así falla, arranca sin persistencia en vez de caerse. Los
  endpoints de histórico responden 503 claro; la generación en vivo funciona.
- **`$PORT`:** el contenedor escucha en el puerto que asigna la plataforma.
- **Healthcheck:** `railway.json` apunta a `/health` con 120 s de margen
  (la primera carga descarga el modelo de embeddings).

## Coste orientativo

El streaming mantiene un proceso vivo durante cada run. En Railway el coste
es por uso (vCPU/RAM por segundo). Un uso moderado suele quedar en el entorno
de 5–20 USD/mes; vigila los runs muy grandes (población × generaciones altas),
que alargan el proceso y multiplican las llamadas LLM.

## Troubleshooting rápido

Ante cualquier fallo, primero el diagnóstico (Console de Railway):

```bash
creative-engine doctor
```

Te dirá directamente: qué proveedor tiene la clave mal (401), cuál está
saturado, si `CREATIVE_ROUTING_SPEC` tiene erratas y si la BD responde.
O desde el navegador: `https://<tu-dominio>/api/v1/diagnostics?check_llm=true`.

Errores típicos ya vistos:
- `401 Invalid API Key` → la clave de ese proveedor está mal copiada
  (espacios, comillas) o revocada. Regenera y pega de nuevo.
- `routing={}` en los logs → falta `CREATIVE_ROUTING_SPEC` o tiene errata;
  con 2+ proveedores el failover funciona igualmente en orden de definición.
- `Rate limit excedido` → free tier saturado; el failover salta al otro
  proveedor si existe.
