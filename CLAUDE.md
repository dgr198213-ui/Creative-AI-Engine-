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
PYTHONPATH=src python -m pytest tests/ -q     # 215 tests, sin red ni BD
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
├── analysis/    analyst.py (Analista Funcional), mirror.py (espejo de
│                confirmación), context.py (perfil → hint para el motor)
├── api/         app.py (FastAPI), auth.py (API key), guardrails.py (rate
│                limit + tope de presupuesto), routes/ (evolution, stream
│                SSE, ideas, memory, diagnostics, analysis), static/
│                (panel: index.html + app.js)
├── bench/       arnés de benchmark de 3 brazos (config.py, harness.py,
│                judge.py, report.py) — ver sección "Analista Funcional"
├── benchmark.py motor QD vs prompt único, 2 brazos (la validación de la tesis)
├── diagnostics.py  doctor: verifica claves/enrutado/BD
└── main.py      CLI: serve, evolve, benchmark (2 brazos), bench (3 brazos), doctor
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
- `TYPE=openai` para la API real de OpenAI activa `max_completion_tokens`
  en el payload directamente, sin coste de una llamada extra. **Ya no es
  un requisito, es una optimización**: el provider autoadapta el
  parámetro solo si `TYPE` falta o llega mal (ver lección de abajo),
  pero configurarlo bien evita ese primer 400 en cada arranque.
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
  invalid_request_error): exige `max_completion_tokens`. Incidente
  21-jul-2026: terra sin `TYPE` → 12 batches fallidos, run con 0 élites.
- **La config `TYPE` es una pista, no un requisito (incidente 22-jul-2026,
  segunda noche).** Terra siguió en 400 pese a tener `TYPE=openai` puesto
  (la variable no llegaba al contenedor — comillas/espacios/environment
  equivocado) y **luna tenía exactamente el mismo problema sin que nadie
  lo detectara**: todo el diagnóstico del primer incidente se centró en
  terra. Causa raíz única: la elección de parámetro dependía enteramente
  de que un humano configurase bien una env var por proveedor — frágil
  por construcción. Fix: `LLMProvider` ahora **autoadapta** el parámetro
  de tokens: si recibe el 400 de "usa max_completion_tokens" (o al revés),
  cambia el flag, reintenta la misma petición una vez y recuerda la
  elección de por vida del provider (log `token_param_auto_adapted`). Un
  proveedor OpenAI sin `TYPE` ya no se deshabilita todo el run — paga un
  400+reintento (~1s, gratis) solo en su primera llamada. Configurar
  `TYPE=openai` sigue siendo mejor (evita ese primer 400), pero ya no es
  obligatorio para que el proveedor funcione.
- **La autoadaptación se generalizó a cualquier parámetro, no solo tokens
  (mismo incidente, tercera vuelta).** Terra volvió a caer con "400
  Unsupported value: 'temperature' does not support 0.9": la familia
  gpt-5.6 solo acepta `temperature=1`. El patrón (un parámetro que un
  modelo concreto no soporta, detectado por texto del error, corregible
  quitándolo del payload) es el mismo que el de `max_tokens` — así que en
  vez de parchear parámetro a parámetro, `LLMProvider` ahora reconoce
  cualquier 400 "Unsupported parameter/value: 'X'", quita `X` del payload,
  lo recuerda (`self._unsupported_params`) y reintenta (log
  `param_auto_dropped`). Tope de 3 parámetros distintos por llamada, nunca
  el mismo dos veces, para no reintentar sin fin ante un proveedor
  patológico. `max_tokens`/`max_completion_tokens` quedan fuera de este
  mecanismo genérico: su fix es sustitución (uno por otro), no eliminación.
  **Caso especial de `temperature`:** eliminarla del payload sin más
  perdería la palanca de diversidad que usan mutación/cruce (temperaturas
  altas = más riesgo). Se traduce a una instrucción en el system prompt
  (`t<0.4` → conservador, `t>0.8` → arriesgar, rango medio → nada) para que
  el efecto sobrevida aunque el modelo ignore el parámetro numérico.
- **Un 400 invalid_request_error deshabilita el proveedor para el resto del
  run** (`provider_disabled_for_run`) y rota al siguiente: no se arregla
  reintentando. Esto sigue aplicando a cualquier 400 que la autoadaptación
  no pueda resolver (parámetro fuera del payload, ya intentado, o tope de
  3 agotado). Un run que agota la cadena con población vacía aborta con
  estado `failed` (`evolution_aborted_empty_population`), nunca completa
  vacío.
- Ante cualquier duda de config: `creative-engine doctor` lo diagnostica en
  segundos. No deducir de logs de runs fallidos.

## Retención de memoria entre runs (incidente 22-jul-2026)

Un run desde el panel (población 8, 3 generaciones) hacía subir la RAM de
~94 MB a ~870 MB y se quedaba ahí 90 minutos después de terminar, sin
bajar. 90 minutos más tarde, un segundo proceso (bench) provocó OOM y
Railway mató el contenedor.

**Causa raíz: `IdeaEncoder()` se reconstruía en cada llamada a
`/evolution/start` o `/evolution/stream`** (`api/routes/evolution.py`,
`_build_qd_engine`). En su primer uso carga `sentence-transformers`
(torch) — cientos de MB de RSS entre el modelo y los buffers de sus hilos
BLAS. No es una fuga de objetos Python (esos se recolectan solos por
conteo de referencias): son los asignadores nativos de torch/MKL, que NO
devuelven memoria liberada al sistema operativo — la retienen en arenas
por si se vuelve a pedir. Reconstruir el encoder run tras run repetía ese
coste sin ninguna ganancia, y la memoria nunca volvía a bajar por sí sola.

**Fix (`evolution/encoders.py::get_shared_encoder`):** el servidor mantiene
UNA instancia de `IdeaEncoder` para todo el proceso, cargada perezosamente
en el primer run y reutilizada en todos los siguientes —
`_build_qd_engine` ya no llama a `IdeaEncoder()` directamente. **Decisión
consciente:** el encoder se mantiene cargado a propósito porque recargarlo
es caro; esa memoria queda descontada del objetivo de "la RAM vuelve al
baseline". El nuevo baseline en reposo incluye el encoder cargado — el
objetivo real es que la RAM **tras un run no crezca respecto al baseline
con el encoder ya cargado** (validado por `test_memory_retention.py`: dos
runs seguidos, el segundo no debe pesar más que el primero). El CLI
(`main.py`, `bench/harness.py`) ya cargaba el encoder una sola vez por
proceso — ese patrón no cambia, solo se replica en el servidor de la API.

**Limpieza explícita al terminar cada run** (`qd_engine.py::run_evolution`,
bloque `finally`, cubre éxito/fallo/abort): se borra la referencia al
archivo MAP-Elites completo (`archive`, todas las celdas del grid, no solo
las ocupadas — ya copiadas a `state.archive`) y al contexto de generación
ANTES de llamar a `gc.collect()` + `malloc_trim(0)` (`core/memory_utils.py`)
— si no se borran antes, el frame de la función los sigue referenciando y
la recolección es un no-op. `malloc_trim` es lo que de verdad le pide a
glibc que devuelva páginas libres al SO; `gc.collect()` solo no lo
consigue con la basura fina del run (miles de objetos `Idea`, prompts,
respuestas descartadas de generaciones intermedias).

**Logging para verificar sin depender del panel:** cada run emite
`run_memory_footprint` (`rss_start_mb`, `rss_end_mb`, `rss_delta_mb`) al
terminar, vía `core/memory_utils.current_rss_mb()` (lee `/proc/self/status`;
devuelve `None` fuera de Linux, no rompe nada).

**Aparte, en el arnés de bench** (`bench/harness.py`, `main.py::bench`):
cada brazo/repetición se persiste en BD según se completa (en vez de
acumularse en una lista hasta el informe final) — el set es reanudable si
el proceso muere a mitad, y el informe se reconstruye leyendo de BD en vez
de depender de tener todo en RAM al final.

## Analista Funcional (diseño 22-jul-2026, apagado por defecto)

Convierte un reto vago de un empresario no técnico ("mi tienda no vende")
en un perfil funcional estructurado ANTES de generar ideas, sin inventar
datos. Feature flag `CREATIVE_ANALYST_ENABLED` (default `false`): apagado,
`POST /api/v1/analyze` responde 404 y el motor se comporta exactamente
igual que sin esta feature — `EvolutionRequest.profile` es `None` siempre
que el panel no lo active.

**Contrato del perfil** (`ChallengeProfile` en `core/models.py`, es
también el embrión del futuro "domain pack" — otro dominio, otro
esquema, mismo motor):

```
version, reto_original (nunca lo escribe el LLM, lo asigna el agente)
topografia:          que_ocurre, frecuencia, desde_cuando, donde_ocurre,
                      intentos_previos
hipotesis_funcional:  antecedente, mecanismo, refuerzo, confianza (0-1)
friccion:             impacto_principal, descripcion_impacto, urgencia
restricciones_duras:  lista libre
reto_reformulado:     el reto que de verdad recibe el motor
preguntas_pendientes: máx. 2, solo si confianza < 0.6 (forzado en el
                      parseo aunque el LLM se equivoque)
```

**Flujo:** `POST /analyze` (perfil + espejo de confirmación en texto) →
el panel muestra el espejo con "✅ Es esto" / "✏️ Corregir algo" (máximo
UN ciclo de corrección, la puerta no es un chat) → `POST /evolution/stream`
con el `profile` en el body. `QDEngine.run_evolution` genera sobre
`profile.reto_reformulado` (no `challenge`) e inyecta un resumen del
perfil (`analysis/context.py`) en el `variation_hint` del generador y en
el prompt del evaluador combinado. `state.challenge` conserva el texto
original tal cual lo escribió el usuario (trazabilidad).

**Benchmark de 3 brazos** (`creative-engine bench --set configs/bench/vagos.yaml`):
A (prompt único + auto-mejora) / B (motor solo) / C (motor + Analista),
mismo presupuesto aproximado, mismos proveedores. Coste real (llamadas,
tokens) medido por diferencia de contadores del router
(`LLMProvider.total_calls/total_prompt_tokens/total_completion_tokens`,
agregados en `LLMModelRouter.total_calls/total_tokens`) — no se fuerza
una igualdad exacta, que sería ilusoria entre un prompt único y un motor
evolutivo; se reporta para que cualquier desigualdad sea auditable.

**Criterio de éxito para que el Analista se quede** (`bench/report.py`):
1. En retos vagos: C > B en qd_score y utilidad ciega (≥ +15%, señal no ruido).
2. En retos control (ya bien formulados): C no empeora a B más de un 5%.
3. B > A en diversidad y utilidad en ambos tipos (valida el motor mismo).

El juez de "utilidad ciega" usa el rol `writer` (no participa en la
generación/evaluación de ningún brazo) y no sabe de qué brazo viene cada
propuesta ni cómo se generó.

## Diagnóstico de sesgo de longitud en originalidad (Fase 5, bloque 4)

Sospecha detectada en el run `c66010b7` (22-jul-2026): dos ideas
conceptualmente primas (grafo causal de linajes, corta vs larga)
puntuaron 100% y 16% de originalidad — posible señal de que el
descriptor pesa más la longitud del texto que su contenido real.

**Herramienta:** `creative-engine diagnose-length-bias`
(`evolution/length_bias_diagnostic.py`) codifica pares de mismo-concepto/
longitud-distinta contra pares de concepto-distinto/longitud-parecida con
el modelo REAL de embeddings y compara similitudes. Sesgo confirmado si
los conceptos distintos salen más similares que el mismo concepto en
longitudes distintas. Hay un test guardia (`test_length_bias_diagnostic.py`)
que corre lo mismo pero se salta solo sin red (no rompe la regla de suite
sin red — ver más abajo).

**Estado: SIN CONFIRMAR.** La sesión que implementó el diagnóstico
(23-jul-2026) no pudo ejecutarlo: este entorno de desarrollo bloquea por
política de red la descarga de `all-MiniLM-L6-v2` desde Hugging Face
(`403 Forbidden` en `huggingface.co`, ver salida de
`creative-engine diagnose-length-bias`). **Pendiente:** ejecutar el
comando donde SÍ haya red (máquina local del autor, o el contenedor de
Railway, que ya tiene el modelo cargado) y actualizar este estado con el
veredicto real. El bloque 4-corrección (normalizar longitud antes de
codificar) de la Fase 5 sigue sin implementar a propósito hasta tener
ese veredicto — corregir una métrica sin confirmar el diagnóstico
invalidaría cualquier benchmark posterior.

## Presupuesto igualado entre brazos del bench (Fase 5, bloque 1)

La prueba de humo del bench (22-jul) no era concluyente: el brazo A
gastó 3 llamadas frente a 24 de B y 30 de C. El diseño exigía presupuesto
igualado; sin él, el criterio 3 (motor > prompt único) medía el motor
contra un brazo hambriento y cualquier conclusión era inválida.

**Fix (`bench/harness.py`):** B corre PRIMERO en `run_single_challenge`;
su consumo real de llamadas (`arm_b.cost.calls`) es la referencia de
presupuesto para A. `_run_arm_a` ya no genera una sola tanda + 1
auto-mejora: repite rondas de "generar N ideas + auto-mejora" hasta
agotar ese presupuesto (con un tope de seguridad de rondas para no
bucear indefinidamente si un proveedor falla sin gastar la llamada
esperada). C no se fuerza — usa la misma población/generaciones que B,
así que su coste ya es estructuralmente comparable; solo se le adjunta
el mismo `budget_calls` para el informe.

`BenchArmResult.budget_calls` (None en B, el consumo real de B en A y C)
queda persistido y se muestra en el informe Markdown como columna
"Presupuesto objetivo" junto a "Llamadas reales". Criterio de aceptación
validado en `tests/test_bench.py::TestEqualizedBudget`: A y B no
difieren más de un 10% en un bench de 1 reto.

## Seguridad de la API pública (Fase 5, bloque 2)

Ya implementado de una auditoría previa, ahora con tests que lo protegen
de regresión (`tests/test_security.py` — no existían antes de esta fase):

- **`CREATIVE_API_KEY`** (`api/auth.py::ApiKeyMiddleware`): sin ella, la
  API arranca abierta (uso local/tests); con ella, todo `/api/v1/*`
  exige `X-API-Key` (o `?api_key=` para enlaces de descarga). Rutas
  públicas incluso con clave activa: `/health`, `/`, `/static/*`.
  Añadido en esta fase: log `api_key_not_configured_endpoints_open` al
  arrancar si la clave no está puesta — antes el estado quedaba mudo
  hasta auditar a mano.
- **Cap por run** (`api/guardrails.py::enforce_request_budget`): 422 si
  `population_size x generations` supera `CREATIVE_EVOLUTION__MAX_REQUESTED_EVALUATIONS`
  (default 2000, campo `EvolutionConfig.max_requested_evaluations`).
- **`docs_url`/`redoc_url`/`openapi_url`** en `None` cuando `debug=False`
  (`api/app.py::create_app`).

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
| `luna`    | OpenAI (real) — confirmado 22-jul | **sin confirmar** modelo exacto | openai | **sin confirmar** — verificar si es de pago |

**Nota sobre `luna` (actualizado 22-jul-2026):** la auditoría del incidente
de la segunda noche (P1) confirma que `luna` es OpenAI real: falla con el
mismo 400 `invalid_request_error` exacto que `terra` ("Unsupported
parameter: 'max_tokens' ... Use 'max_completion_tokens' instead"), y con
la autoadaptación del provider ya funciona sin depender de `TYPE`. Queda
pendiente confirmar el **modelo exacto** y si **factura dinero real** como
terra — verificar con `railway variables | grep LUNA` y rellenar antes de
tocar de nuevo el routing. Tampoco está de más poner `CREATIVE_LLM__LUNA__TYPE=openai`
explícitamente: con la autoadaptación ya no es obligatorio, pero evita el
400 inicial en cada arranque.

Routing actual: `generator=terra,luna,default,zai`;
`evaluator=luna,terra,default,zai`; `writer=terra,default`.
**Pendiente operativo:** el writer no tiene fallback gratuito
(`terra,default` — si terra cae y Gemini está saturado, el writer se
queda sin proveedor). Cambiar a `writer=terra,default,zai` en Railway.
OJO: `terra` factura dinero real (y probablemente `luna` también, a
confirmar). Cualquier cambio que multiplique llamadas del generator/writer
tiene coste directo, no solo cuota.

## Estado y siguiente paso

Desplegado y funcionando (4 proveedores con failover por rol, circuito con
cooldown, puerta de sorpresa, salvage de JSON malformado). El
ciclo base está validado en producción con runs completos e informes reales.
El Analista Funcional y el benchmark de 3 brazos están implementados y en
verde en tests, pero **apagados en producción** (`CREATIVE_ANALYST_ENABLED=false`).
Siguiente paso real: correr `creative-engine bench --set configs/bench/vagos.yaml`
contra proveedores reales y decidir con los 3 criterios de éxito si el
Analista se activa por defecto o se descarta — no activarlo "porque sí"
sin ese dato. En paralelo, seguir con lo de siempre: **usar** (retos
reales) y **medir** (el modo `benchmark` de 2 brazos). Solo después,
integrar más técnicas del roadmap — y siempre con la regla 4 (no
multiplicar evaluaciones) en mente.
