# CLAUDE.md

Guía para trabajar en este repositorio con un asistente de IA. Léela antes
de proponer cambios: recoge la intención del proyecto, las invariantes que
NO deben romperse y las lecciones aprendidas desplegando en producción.

## Qué es esto

**Creative AI Engine**: dado un reto, devuelve un abanico de ideas élite
genuinamente diversas usando Quality-Diversity (MAP-Elites) + agentes LLM.
La tesis del proyecto, en palabras del autor: *"ideas más vestidas y con
evolución, no las típicas respuestas de un chat"*.

No es un chatbot ni una plataforma. Es una herramienta enfocada de
exploración de ideas. Resistir la tentación de convertirlo en algo más grande.

## Arranque rápido para trabajar

```bash
cd creative-ai-engine
PYTHONPATH=src python -m pytest tests/ -q     # 173 tests, sin red ni BD
ruff check src/ tests/                         # lint
```

Todo el desarrollo se valida sin red ni base de datos: los tests usan LLMs
simulados (`AsyncMock`) y embeddings deterministas (`conftest.fake_embed`).
Si un cambio necesita red o BD para probarse, casi siempre hay una forma
mejor de estructurarlo.

## Arquitectura en una pantalla

```
src/creative_engine/
├── core/        modelos Pydantic (models.py), config (config.py),
│                eventos (EventBus), excepciones
├── evolution/   map_elites.py (archivo + selección: fitness / curiosidad),
│                encoders.py (embeddings + novedad objetiva),
│                mutation.py, crossover.py, surprise.py (puerta adaptativa),
│                clustering.py (familias), qd_engine.py (orquesta el ciclo)
├── agents/      generator, combined_evaluator (3 dims en 1 llamada),
│                evaluator_orchestrator, writer, critic, base
├── llm/         provider.py (cliente OpenAI-compatible), router.py
│                (enrutado por rol + failover), factory.py
├── memory/      repository.py (PostgreSQL), graph.py (Neo4j, opcional),
│                recommendation
├── api/         app.py (FastAPI), auth.py (API key), guardrails.py (rate
│                limit + tope de presupuesto), routes/ (evolution, stream
│                SSE, ideas, memory, diagnostics), static/ (panel: index.html
│                + app.js)
├── benchmark.py motor QD vs prompt único (la validación de la tesis)
├── diagnostics.py  doctor: verifica claves/enrutado/BD
└── main.py      CLI: serve, evolve, benchmark, doctor
```

El flujo del motor (`qd_engine._process_batch`): **codificar (local, gratis)
→ puerta de sorpresa → evaluar solo lo sorprendente (LLM, caro) → novedad
objetiva → insertar en el archivo**. El orden importa: la evaluación LLM es
el recurso escaso; todo lo que se pueda decidir con embeddings va antes.

Memoria entre runs (`grounding.py`): antes de la población inicial, el motor
recupera élites de runs pasados y las afines al reto (similitud de
embeddings, local) van al prompt como inspiración+repulsión. Cero llamadas
LLM extra; degrada sola sin persistencia. Es el foso del producto: cada run
abona el siguiente.

## Invariantes que NO se rompen

Estas decisiones son deliberadas. Cambiarlas exige una razón muy buena:

1. **Novedad objetiva por embeddings, no por juicio del LLM.** Un LLM
   puntuando "novedad" sin ver la población da ruido. La distancia coseno al
   archivo es medible y reproducible. La novedad tiene peso 0 en el fitness
   (MAP-Elites ya aporta la diversidad; contarla en fitness sería doble).

2. **Descriptor de comportamiento desde el embedding** (proyección
   determinista 384→3 dims, seed fija). Así la diversidad del archivo es
   diversidad de contenido real, no de puntuaciones ruidosas.

3. **Fitness = calidad pura** (utilidad, viabilidad, mercado, impacto).

4. **La evaluación LLM es el cuello de botella.** Cualquier cambio que
   multiplique las llamadas de evaluación (p.ej. Evo-MCTS) está casi siempre
   descartado. Ya no operamos solo en free tiers — **terra factura dinero
   real** (ver inventario de proveedores) — así que esta regla ahora protege
   tanto cuota gratuita como coste directo. Preferir señales locales
   (embeddings).

5. **El motor no conoce la capa de transporte.** El SSE se implementa con un
   callback `on_generation` opcional; los agentes reciben un objeto con
   interfaz `generate`/`generate_structured` (sea `LLMProvider` o `RoledLLM`)
   y no saben nada de routing ni failover.

6. **Un run sobrevive a la desconexión del cliente.** No cancelar la task de
   evolución cuando el navegador se desconecta; persiste en BD y el panel
   recupera vía `GET /runs/{id}/families`.

## Configuración LLM (lo que más ha costado en producción)

Todo por variables de entorno, prefijo `CREATIVE_`, delimitador `__`:

- Un proveedor se define con `CREATIVE_LLM__<NOMBRE>__*` (NAME, API_KEY,
  BASE_URL, MODEL, TYPE, MAX_CONCURRENT, MIN_INTERVAL_SECONDS, EXTRA_BODY).
- `TYPE=openai` es OBLIGATORIO para la API real de OpenAI: activa
  `max_completion_tokens` en el payload. Sin él, el proveedor manda
  `max_tokens` y los modelos recientes de OpenAI devuelven 400.
- El enrutado por rol: `CREATIVE_ROUTING_SPEC=generator=a,b;evaluator=b,a;writer=a`.
  Con varios proveedores y sin spec, la cadena por defecto los usa todos en
  orden (failover automático).
- `EXTRA_BODY` es JSON: p.ej. `{"thinking":{"type":"disabled"}}`.

**Lecciones de producción (no repetir estos errores):**

- **httpx descarta el path de `base_url` si la petición empieza por `/`.**
  Por eso las URLs con ruta (Gemini `/v1beta/openai/`) deben acabar en barra
  y el POST usa ruta relativa `chat/completions`. Ya está resuelto en
  `provider.py`; no reintroducir el `/` inicial.
- **`asyncpg` no acepta múltiples comandos en un execute.** El schema se
  ejecuta sentencia a sentencia. No volver a mandar el bloque entero.
- **Los GLM de Z.ai "piensan" por defecto** (>120s en free tier). El provider
  desactiva `thinking` automáticamente para modelos que empiezan por `glm`.
- **Nombres de modelos de Gemini cambian rápido.** Usar el alias
  `gemini-flash-latest`, no versiones fijas que Google retira.
- **401 = clave mal copiada**, casi siempre (espacios, comillas). El failover
  ante 401 evita que tumbe el run, pero conviene arreglar la clave.
- **OpenAI real rechaza `max_tokens` en modelos recientes** (400
  invalid_request_error): exige `max_completion_tokens`. El provider lo
  gestiona SOLO si el proveedor tiene `TYPE=openai` configurado (incidente
  21-jul-2026: terra sin TYPE → 12 batches fallidos, run con 0 élites).
- **Un 400 invalid_request_error deshabilita el proveedor para el resto del
  run** (`provider_disabled_for_run`) y rota al siguiente: no se arregla
  reintentando. Un run que agota la cadena con población vacía aborta con
  estado `failed` (`evolution_aborted_empty_population`), nunca completa vacío.
- Ante cualquier duda de config: `creative-engine doctor` lo diagnostica en
  segundos. No deducir de logs de runs fallidos.

## Convenciones

- Python ≥ 3.12, Pydantic v2, tipos everywhere, ruff limpio.
- Comentarios y textos de usuario en español (el autor trabaja en español).
- Cada cambio de comportamiento va con su test, sin red ni BD.
- Formato de commit: `tipo: resumen` + cuerpo explicando el *porqué*, no solo
  el qué. Referenciar el síntoma de producción cuando aplique.
- No añadir dependencias sin necesidad clara.

## Cómo validar antes de entregar

```bash
PYTHONPATH=src python -m pytest tests/ -q     # todo verde
ruff check src/ tests/                         # sin errores
node --check src/creative_engine/api/static/app.js   # si tocaste el panel
```

Para cambios en el ciclo evolutivo, además un mini-run con LLM simulado que
confirme que produce élites y no rompe el primer run (archivo vacío).

## Inventario de proveedores en producción (Railway)

MANTENER AL DÍA: toda sesión que añada, retire o modifique un proveedor o
el routing DEBE actualizar esta tabla y las lecciones de arriba en el mismo
commit. Las sesiones no comparten contexto entre sí; este archivo es el
punto de sincronización.

| Nombre    | Servicio        | Modelo             | TYPE   | Coste     |
|-----------|-----------------|--------------------|--------|-----------|
| `default` | Gemini (OpenAI-compat) | gemini-flash-latest | —  | Free tier |
| `zai`     | Z.ai            | glm (flash)        | —      | Free tier |
| `terra`   | OpenAI (real)   | gpt-5.6-sol        | openai | DE PAGO   |
| `luna`    | **sin confirmar** — ver nota | **sin confirmar** | probablemente `openai` (pendiente verificar) | **sin confirmar** — verificar si es de pago |

**Nota sobre `luna` (21-jul-2026):** nadie en el historial de sesiones sabe
qué servicio/modelo es. Evidencia de los logs de producción de ese día:
`luna` falló con el mismo 400 `invalid_request_error` ("Unsupported
parameter: 'max_tokens' ... Use 'max_completion_tokens' instead") que
`terra`, y el mensaje de la excepción imprime "en openai" para ambos —
señal de que ninguno tiene `CREATIVE_LLM__<NOMBRE>__NAME` configurado
explícitamente (el default de `LLMProviderConfig.name` es `"openai"`).
Esto sugiere que `luna` también corre sobre un backend OpenAI real y
probablemente necesite `TYPE=openai`, pero es una inferencia a partir de
los logs, no un dato confirmado. Verificar con `railway variables | grep
LUNA` (o el panel de Railway) y rellenar esta fila antes de tocar de nuevo
el routing.

Routing actual: `generator=terra,luna,default,zai`;
`evaluator=luna,terra,default,zai`; `writer=terra,default`.
OJO: `terra` factura dinero real (y posiblemente `luna` también, a
confirmar). Cualquier cambio que multiplique llamadas del generator/writer
tiene coste directo, no solo cuota.

## Estado y siguiente paso

Desplegado y funcionando (4 proveedores con failover por rol, circuito con
cooldown, puerta de sorpresa, salvage de JSON malformado). El
ciclo base está validado en producción con runs completos e informes reales.
Lo siguiente NO es arreglar, es **usar** (retos reales) y **medir** (el modo
`benchmark` contra prompt único). Solo después, integrar más técnicas del
roadmap — y siempre con la regla 4 (no multiplicar evaluaciones) en mente.
