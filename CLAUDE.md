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
PYTHONPATH=src python -m pytest tests/ -q     # 296 tests, sin red ni BD
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
│                domain_registry.py (domain packs, Fase 6), eventos
│                (EventBus), excepciones
├── evolution/   map_elites.py (archivo + selección: fitness / curiosidad),
│                encoders.py (embeddings + novedad objetiva),
│                mutation.py, crossover.py, surprise.py (puerta adaptativa),
│                clustering.py (familias), qd_engine.py (orquesta el ciclo)
├── agents/      generator, combined_evaluator (3 dims en 1 llamada),
│                evaluator_orchestrator, writer, critic, base
├── llm/         provider.py (cliente OpenAI-compatible), router.py
│                (enrutado por rol + failover), factory.py, budget.py
│                (guard de presupuesto)
├── memory/      repository.py (PostgreSQL), graph.py (Neo4j, opcional),
│                recommendation
├── analysis/    analyst.py (Analista Funcional), mirror.py (espejo de
│                confirmación), context.py (perfil → hint para el motor)
├── api/         app.py (FastAPI), auth.py (API key), guardrails.py (rate
│                limit + tope de presupuesto), routes/ (evolution, stream
│                SSE, ideas, memory, diagnostics, analysis, budget,
│                domains), static/ (panel: index.html + app.js)
├── bench/       arnés de benchmark de 3 brazos (config.py, harness.py,
│                judge.py, report.py) — ver sección "Analista Funcional"
├── benchmark.py motor QD vs prompt único, 2 brazos (la validación de la tesis)
├── diagnostics.py  doctor: verifica claves/enrutado/BD
└── main.py      CLI: serve, evolve, benchmark (2 brazos), bench (3 brazos),
                  doctor, domain (list/validate), diagnose-length-bias
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

## Guard de presupuesto (Fase 5, bloque 3)

Antes, la estrategia "usar proveedores de pago hasta agotar el
presupuesto y luego pasar a gratis" era manual: vigilar el usage y
editar `CREATIVE_ROUTING_SPEC` a mano en el momento justo. Ahora se
automatiza con la opción B del diseño: **precio por millón de tokens
configurable por proveedor**, no contador de tokens crudos (mezclaría
proveedores caros y gratis) ni consulta a la API de usage de cada
proveedor (acopla a uno concreto). El coste es una ESTIMACIÓN (tokens x
precio declarado), no la factura real.

**Variables nuevas:**
- `CREATIVE_LLM__<NOMBRE>__PRICE_IN` / `PRICE_OUT`: USD por millón de
  tokens de prompt/completion. Sin declarar (0.0 por defecto), el
  proveedor se trata como gratuito y nunca cuenta para el guard.
- `CREATIVE_BUDGET_LIMIT`: límite estimado en USD por periodo. `0`
  (por defecto) = sin límite, el guard nunca degrada (solo contabiliza
  si hay BD).
- `CREATIVE_BUDGET_PERIOD`: `monthly` (por defecto) o `daily` — ventana
  de acumulación del gasto.
- `CREATIVE_BUDGET_WARNING_RATIO`: umbral de aviso, 0.8 por defecto.
- `CREATIVE_BUDGET_ENFORCE`: `true` por defecto. En `false`, desactiva
  la degradación (el routing no cambia) pero la contabilidad y los
  avisos siguen intactos — escape para casos donde se quiere solo
  observar el gasto antes de dejar que el guard actúe.

**Comportamiento (`llm/budget.py`):** en cada `/evolution/start` o
`/evolution/stream`, antes de construir el router
(`api/routes/evolution.py::_build_qd_engine`), se consulta el gasto
acumulado del periodo en BD (`get_budget_status`). Si supera
`CREATIVE_BUDGET_LIMIT` y `CREATIVE_BUDGET_ENFORCE=true`, el router se
construye SIN los proveedores de pago (`LLMModelRouter(budget_excluded=...)`)
— el motor sigue funcionando con los gratuitos, nunca falla el run por
esto salvo que TODOS los proveedores de un rol sean de pago. Al 80% del
límite, log `budget_warning` sin tocar el routing. Al terminar cada run
(`_close_and_record_spend`, compartido entre `/start` y `/stream`), se
persiste el gasto real de los proveedores de pago con
`record_run_spend` — sin BD, el guard sigue activo dentro del proceso
por los contadores en memoria del router, pero la acumulación entre
runs/reinicios necesita persistencia.

**Endpoint:** `GET /api/v1/budget` → `spent_usd`, `limit_usd`, `period`,
`period_key`, `status` (`ok` | `warning` | `downgraded`),
`excluded_providers`.

**Pendiente operativo:** rellenar `PRICE_IN`/`PRICE_OUT` para `terra` y
`luna` en Railway con el precio real de gpt-5.6-sol (y confirmar el
modelo de `luna`, ver inventario de proveedores) — sin esto, el guard
los sigue tratando como gratuitos y nunca los excluye.

## Domain Packs (Fase 6, 23-jul-2026)

Un dominio ya no es un enum en código (`DomainName`, retirado): es un
directorio autocontenido bajo `configs/domains/<nombre>/`, cargado por
`core/domain_registry.py` al arrancar. Requisito no negociable de esta
fase, verificado por `tests/test_generic_domain_regression.py` (semilla
fija + LLM mockeado, valores exactos): el dominio `generic` se comporta
EXACTAMENTE igual que antes de la refactorización.

### Por qué (las tres limitaciones que resolvía)

1. `DomainName` obligaba a tocar `src/` para añadir un dominio.
2. Un dominio solo variaba pesos/mutaciones/dimensiones — generator,
   evaluator y analyst usaban el MISMO texto para todos los dominios
   (solo el generador tenía `system_prompt`; evaluator y analyst ni eso).
3. El perfil del Analista Funcional era fijo — el mismo esquema para una
   campaña de marketing que para un artista independiente.

### Cómo crear un domain pack

```
configs/domains/<nombre>/
├── domain.yaml          # obligatorio: name, display_name, description,
│                          descriptor_mode, behavior_dimensions,
│                          evaluation_weights, allowed_mutations,
│                          default_population_size, default_generations
├── prompts/
│   ├── generator.md     # opcional; persona del generador. Si falta,
│   │                      hereda de configs/domains/base/
│   ├── evaluator.md     # opcional; rúbrica del evaluador en lenguaje
│   │                      del dominio. Ídem, hereda de "base"
│   └── analyst.md       # opcional; cómo interrogar a un usuario de
│                          este dominio. Ídem
├── profile.yaml         # opcional: campos extra de ChallengeProfile.dominio
│                          (formato: campos: [{nombre, descripcion}, ...])
├── examples.yaml         # opcional: lista de retos de ejemplo (panel)
└── bench.yaml            # opcional: set de retos (creative-engine bench)
```

Todo salvo `domain.yaml` es opcional y hereda de `configs/domains/base/`
(el pack fundacional — su `domain.yaml` declara `name: generic`). **Añadir
un dominio no toca ni una línea de `src/`** — es exactamente lo que probó
el pack `tuesdi` (visibilidad de artistas independientes), creado ANTES
de migrar `base`/`marketing`/`industrial_design` al nuevo formato, a
propósito: si la abstracción de cascada estuviera mal diseñada, un pack
que no puede apoyarse en `base` lo habría revelado.

Los prompts admiten los placeholders `{reto}`, `{perfil}`,
`{inspiraciones}` (`domain_registry.format_domain_prompt`), resueltos
con el reto/perfil/pista de variación de cada llamada. Un placeholder
desconocido (typo) no rompe el run: el prompt se usa sin resolver
(degradación con elegancia) — pero `creative-engine domain validate`
lo detecta antes de llegar a producción.

**Validar antes de desplegar:**
```bash
creative-engine domain validate configs/domains/<nombre>
creative-engine domain list       # todos los packs cargados
```
Detecta esquema inválido (`domain.yaml`), placeholders desconocidos en
los prompts, y prompts que no se pueden formatear.

**Arranque ruidoso:** un pack con `domain.yaml`/`profile.yaml`/
`examples.yaml` inválido hace FALLAR `Settings.load()` (nunca se salta
en silencio — un dominio mal configurado que "funciona" con los
defaults de otro es peor que no arrancar). **Cero packs encontrados
también hace fallar el arranque** (`RuntimeError`, desde el incidente de
producción del 23-jul-2026 — ver abajo): ya no existe el fallback
silencioso al "generic" embebido que había al cerrar la Fase 6.

**Incidente de producción (23-jul-2026): el registro no cargaba ningún
pack en Railway.** `GET /api/v1/domains` devolvía solo el "generic"
embebido pese a que `configs/domains/{base,marketing,industrial_design,
tuesdi}` existían en el contenedor (`ls` lo confirmaba) — sin ningún
warning ni error en los logs. Causa raíz: `_CONFIGS_DIR` en
`core/config.py` se calculaba con `Path(__file__).parent x4` asumiendo
un checkout del repo (`src/creative_engine/core/config.py` → subir 4 →
raíz del repo). El `Dockerfile` instala el paquete vía `pip install
--prefix=/install .` (queda en `site-packages` dentro de la imagen
runtime) y copia `configs/` **aparte** a `/app/configs` — los cuatro
`.parent` desde `site-packages/creative_engine/core/config.py` no
llegan ni de lejos a `/app/configs`. El registro interpretó "ruta
equivocada" como "sin `configs/domains/`", que antes era un caso
VÁLIDO (fallback a generic) — la ambigüedad entre "no existe" y "ruta
mal calculada" ocultó el bug durante todo un despliegue.

**Fix (`core/config.py::Settings.load`):**
1. `CREATIVE_CONFIGS_DIR` (variable de entorno nueva) tiene prioridad
   sobre la heurística de `__file__` — control explícito en vez de
   adivinar rutas. Fijada en el `Dockerfile` a `/app/configs`, que es
   donde el propio Dockerfile ya copia `configs/`.
2. Cero packs encontrados ahora es SIEMPRE un `RuntimeError` al
   arrancar (ver arriba) — así una ruta mal resuelta revienta el deploy
   de inmediato en vez de degradar en silencio a un solo dominio.
3. Log `domains_loaded` (con la lista de packs) en cada arranque
   exitoso, para verificar desde logs sin depender de `/api/v1/domains`.

Lección: un fallback "silencioso pero válido" (configs/domains/
ausente → generic embebido) es exactamente lo que necesita un bug de
ruta para pasar desapercibido — dos causas indistinguibles (ausencia
real vs. ruta equivocada) no deben compartir el mismo camino silencioso.

### Perfil extensible del Analista (D4)

`ChallengeProfile.dominio: dict` recoge los campos que el pack declare
en `profile.yaml`. `FunctionalAnalystAgent.analyze()` acepta un `domain`
opcional: si trae `profile_fields`, extiende el esquema JSON que le pide
al LLM con un bloque `dominio` y solo recoge los campos DECLARADOS (lo
que el LLM invente de más no se cuela en el perfil). Sin `domain` (o sin
campos declarados), comportamiento idéntico al de siempre.

### Endpoint y panel

`GET /api/v1/domains` devuelve los packs cargados (nombre, título,
descripción, ejemplos). El panel (`app.js::loadDomains`) construye sus
chips de dominio y los retos de ejemplo desde ahí al cargar — ya no hay
dominios fijos en `index.html`. El servidor garantiza al menos un pack
al arrancar (ver "Arranque ruidoso" arriba), así que el endpoint nunca
necesita un fallback propio.

### Packs disponibles hoy

`base` (fundacional, `name: generic`), `marketing`, `industrial_design`
(ambos heredan evaluator/analyst de `base`, solo traen su propio
`generator.md`), y `tuesdi` (visibilidad de artistas independientes —
trae los tres prompts propios + `profile.yaml` + `bench.yaml`, prueba de
aceptación de la fase).

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

| Nombre    | Servicio        | Modelo             | TYPE   | Coste     | PRICE_IN/OUT (guard presupuesto) |
|-----------|-----------------|--------------------|--------|-----------|----------------------------------|
| `default` | Gemini (OpenAI-compat) | gemini-flash-latest | —  | Free tier | sin declarar (gratuito, correcto) |
| `zai`     | Z.ai            | glm (flash)        | —      | Free tier | sin declarar (gratuito, correcto) |
| `terra`   | OpenAI (real)   | gpt-5.6-sol        | openai | DE PAGO   | **sin declarar — el guard lo trata como gratuito y nunca lo excluye** |
| `luna`    | OpenAI (real) — confirmado 22-jul | **sin confirmar** modelo exacto | openai | **sin confirmar** — verificar si es de pago | sin declarar |

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

Fase 6 (Domain Packs) completa: `DomainName` retirado, registro dinámico
desde `configs/domains/`, prompts en cascada, perfil extensible (D4),
`GET /api/v1/domains` + panel dinámico, pack `tuesdi` como validación de
la abstracción, `base`/`marketing`/`industrial_design` migrados al
nuevo formato. El benchmark de 2 y 3 brazos sigue siendo válido: el
dominio `generic` se comporta exactamente igual (regresión verificada).

**Pendiente antes de activar el Analista:** correr
`creative-engine bench --set configs/bench/vagos.yaml` contra
proveedores reales y decidir con los 3 criterios de éxito si se activa
por defecto o se descarta — no activarlo "porque sí" sin ese dato.
**Pendiente del diagnóstico de sesgo de longitud** (Fase 5, bloque 4):
sigue sin confirmar, ejecutar `creative-engine diagnose-length-bias`
donde haya red. En paralelo, seguir con lo de siempre: **usar** (retos
reales, ahora también con el pack `tuesdi`) y **medir** (el modo
`benchmark` de 2 brazos, y `bench --set configs/domains/tuesdi/bench.yaml`
para ese dominio). Solo después, integrar más técnicas del roadmap — y
siempre con la regla 4 (no multiplicar evaluaciones) en mente.
