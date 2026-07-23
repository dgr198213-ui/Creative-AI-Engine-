"""Configuración centralizada con soporte para dominios y YAML."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .domain_registry import DomainPack, DomainPackError, load_domain_packs
from .models import BehaviorDimension, DomainConfig

# raíz del repo: src/creative_engine/core/config.py → ../../../configs
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"


class LLMProviderConfig(BaseModel):
    """Configuración de un proveedor LLM (API OpenAI-compatible)."""

    name: str = "openai"
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    # "openai" para la API real de OpenAI (modelos recientes rechazan
    # `max_tokens` con 400 invalid_request_error; exigen `max_completion_tokens`).
    # El resto de proveedores compatibles (Gemini, Groq, Z.ai...) usan "generic".
    type: str = "generic"
    max_tokens: int = 4096
    temperature: float = 0.8
    max_concurrent: int = 5
    timeout_seconds: float = 60.0
    # Intervalo mínimo entre peticiones (segundos). Súbelo para APIs con
    # rate limits estrictos como el free tier de Gemini (p.ej. 4.0 = ~15/min).
    min_interval_seconds: float = 0.1
    # Parámetros extra a incluir en cada petición (JSON). Ej. para desactivar
    # el modo razonador de los GLM de Z.ai (lentísimo en free tier):
    # CREATIVE_LLM__ZAI__EXTRA_BODY={"thinking":{"type":"disabled"}}
    extra_body: dict = Field(default_factory=dict)
    # Precio por millón de tokens (Fase 5, bloque 3 — guard de presupuesto
    # opción B): distingue lo que cuesta dinero de lo que no, que es la
    # decisión que el guard tiene que tomar. Sin precio declarado (0.0), el
    # proveedor se trata como gratuito y nunca cuenta para el guard.
    price_in: float = Field(default=0.0, ge=0.0)  # USD / 1M tokens de prompt
    price_out: float = Field(default=0.0, ge=0.0)  # USD / 1M tokens de completion

    @property
    def is_paid(self) -> bool:
        return self.price_in > 0.0 or self.price_out > 0.0


class DatabaseConfig(BaseModel):
    """Configuración de bases de datos."""

    postgres_url: str = "postgresql+asyncpg://engine:engine_dev@localhost:5432/creative_engine"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_dev"
    redis_url: str = "redis://localhost:6379/0"


class EvolutionConfig(BaseModel):
    """Configuración global del motor evolutivo."""

    max_concurrent_evaluations: int = Field(default=3, ge=1, le=100)
    # Puerta de sorpresa: solo se evalúan con LLM las ideas suficientemente
    # lejanas (distancia coseno) a las élites existentes. Umbral adaptativo.
    surprise_gate_enabled: bool = True
    surprise_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    surprise_threshold_min: float = Field(default=0.02, ge=0.0, le=1.0)
    surprise_threshold_max: float = Field(default=0.20, ge=0.0, le=1.0)
    surprise_threshold_step: float = Field(default=0.02, ge=0.0, le=0.2)
    # Memoria entre runs: inyectar élites afines de ejecuciones pasadas en
    # el prompt de la población inicial (inspiración + repulsión). Cero
    # llamadas LLM extra; requiere persistencia (sin BD se omite sola).
    cross_run_memory_enabled: bool = True
    cross_run_memory_k: int = Field(default=3, ge=1, le=10)
    cross_run_memory_min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)
    # Operadores adaptativos: redistribuir el presupuesto generacional hacia
    # el operador (mutación/cruce/fresco) que está produciendo élites.
    adaptive_operators_enabled: bool = True
    mutation_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    random_injection_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    max_generation_time_seconds: float = 300.0
    # k vecinos para el cálculo objetivo de novedad
    novelty_k_nearest: int = Field(default=5, ge=1, le=50)

    # ── Guardrails de la API pública (auditoría C2) ──────────────────
    # Sin este tope, un EvolutionRequest legítimo (population_size≤500,
    # generations≤200) puede pedir hasta 100.000 evaluaciones LLM de golpe.
    max_requested_evaluations: int = Field(default=2000, ge=1)
    # Rate limit por IP en /evolution/* (en memoria, un solo proceso):
    # cada run consume decenas/cientos de llamadas LLM, así que sin esto
    # cualquiera con la URL puede agotar la cuota gratuita de los proveedores.
    rate_limit_per_minute: int = Field(default=10, ge=1)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0.0)


class Settings(BaseSettings):
    """Configuración global cargada desde env vars y archivos YAML."""

    model_config = SettingsConfigDict(
        env_prefix="CREATIVE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Creative AI Engine"
    debug: bool = False
    log_level: str = "INFO"
    # Auditoría C1: sin esta clave configurada, la API queda abierta (uso
    # local, tests, CI). Al definirla (CREATIVE_API_KEY), todo /api/v1/*
    # exige la cabecera X-API-Key — pensado para el momento en que la API
    # queda expuesta a internet (Railway) sin nada más delante.
    api_key: str = ""
    # Feature flag del Analista Funcional (diseño 22-jul-2026): por defecto
    # apagado — el flujo actual (reto → motor directo) no se toca. Con esto
    # en false, POST /api/v1/analyze responde 404.
    analyst_enabled: bool = False

    # ── Guard de presupuesto (Fase 5, bloque 3, opción B) ────────────
    # Límite estimado en USD por periodo. 0 = sin límite (el guard no
    # degrada nunca; el gasto se sigue contabilizando si hay BD).
    budget_limit: float = Field(default=0.0, ge=0.0)
    # "monthly" (por defecto) o "daily" — ventana de acumulación del gasto.
    budget_period: str = "monthly"
    # Umbral de aviso (log budget_warning, sin tocar el routing).
    budget_warning_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    # Escape: desactiva la DEGRADACIÓN automática a proveedores gratuitos
    # al superar el límite, sin perder la contabilidad (se sigue sumando
    # gasto y avisando). CREATIVE_BUDGET_ENFORCE=false para desactivarla.
    budget_enforce: bool = True

    llm: dict[str, LLMProviderConfig] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)

    # Nombre de dominio (string libre, Fase 6) → config resuelta. Nunca se
    # asigna a mano fuera de `load()`/tests: la fuente de verdad son los
    # domain packs en `configs/domains/` (`domain_registry.load_domain_packs`).
    domains: dict[str, DomainConfig] = Field(default_factory=dict)
    # Packs completos (con ejemplos, para el panel/GET /api/v1/domains) —
    # PrivateAttr porque DomainPack no es un modelo Pydantic serializable.
    _packs: dict[str, DomainPack] = PrivateAttr(default_factory=dict)

    # Enrutamiento por rol. Formato: "rol=prov1,prov2;rol2=prov3".
    # Ej: "evaluator=groq,gemini;generator=gemini,groq;writer=gemini".
    # Vacío = todos los roles usan el proveedor por defecto.
    routing_spec: str = ""

    def routing(self) -> dict[str, list[str]]:
        """Parsea `routing_spec` a {rol: [proveedores en orden de failover]}."""
        result: dict[str, list[str]] = {}
        for part in self.routing_spec.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            role, chain = part.split("=", 1)
            role = role.strip().lower()
            providers = [p.strip() for p in chain.split(",") if p.strip()]
            if role and providers:
                result[role] = providers
        return result

    @classmethod
    def load(cls) -> Settings:
        """Carga configuración desde env + archivos YAML de dominios."""
        settings = cls()

        # Compatibilidad con PaaS (Railway/Render/Heroku): estos inyectan
        # DATABASE_URL y REDIS_URL. Los mapeamos a la config del proyecto y
        # normalizamos el driver de Postgres a asyncpg.
        import os

        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            settings.database.postgres_url = db_url

        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            settings.database.redis_url = redis_url

        # Domain packs (Fase 6). Incidente de producción (23-jul-2026):
        # `_CONFIGS_DIR` se calculaba relativo a `__file__` asumiendo un
        # checkout del repo; el Dockerfile instala el paquete vía `pip
        # install` (queda en site-packages) y copia `configs/` aparte a
        # `/app/configs` — los cuatro `.parent` ya no llegaban a
        # `/app/configs`, así que el escaneo encontraba `configs/domains/`
        # como si no existiera (ruta equivocada, no ausencia real) y
        # arrancaba en silencio con un único dominio embebido. Sin log,
        # sin warning: parecía un arranque normal.
        # `CREATIVE_CONFIGS_DIR` da control explícito (fijado en el
        # Dockerfile a `/app/configs`) sin adivinar rutas por heurística.
        configs_dir_override = os.environ.get("CREATIVE_CONFIGS_DIR")
        configs_dir = Path(configs_dir_override) if configs_dir_override else _CONFIGS_DIR

        try:
            packs = load_domain_packs(configs_dir)
        except DomainPackError as e:
            raise RuntimeError(f"Domain pack mal formado: {e}") from e

        # Arranque RUIDOSO si el escaneo no encuentra NINGÚN pack — ya sea
        # porque configs/domains/ no existe, está vacío, o (el bug real de
        # este incidente) la ruta resuelta es la equivocada. Antes esto
        # caía en silencio al "generic" embebido (default_generic_domain);
        # ese fallback enmascaró el bug de ruta en producción durante todo
        # un despliegue sin que ningún log lo delatara. Nunca más: cero
        # packs es un error de configuración, no un estado válido.
        if not packs:
            raise RuntimeError(
                f"No se encontró ningún domain pack válido en "
                f"{configs_dir / 'domains'}. El registro de dominios "
                "necesita al menos uno para arrancar. Si el paquete se "
                "instaló separado de configs/ (p.ej. vía pip — ver "
                "Dockerfile), define CREATIVE_CONFIGS_DIR apuntando al "
                "directorio configs/ real."
            )

        settings._packs = packs
        settings.domains = {name: pack.config for name, pack in packs.items()}

        import structlog

        structlog.get_logger(__name__).info(
            "domains_loaded",
            packs=sorted(packs),
            configs_dir=str(configs_dir),
        )

        return settings

    def get_domain(self, name: str) -> DomainConfig:
        return self.domains.get(name, self.domains["generic"])

    def get_pack(self, name: str) -> DomainPack | None:
        return self._packs.get(name)

    def list_packs(self) -> dict[str, DomainPack]:
        return dict(self._packs)


def default_generic_domain() -> DomainConfig:
    """Dominio genérico embebido: garantiza arranque sin configs/domains/.

    Texto idéntico al del pack `configs/domains/base/` — este fallback
    solo entra en juego si el repo se ejecuta sin ningún directorio de
    domain packs (p.ej. instalado como paquete suelto).
    """
    return DomainConfig(
        name="generic",
        display_name="Creatividad General",
        description="Configuración genérica para cualquier dominio creativo",
        descriptor_mode="embedding",
        behavior_dimensions=[
            BehaviorDimension(name="semantica_1", bins=10),
            BehaviorDimension(name="semantica_2", bins=10),
            BehaviorDimension(name="semantica_3", bins=8),
        ],
        generator_prompt=(
            "Eres un experto en innovación y creatividad computacional. "
            "Generas ideas que son simultáneamente novedosas, útiles y viables. "
            "Cada idea debe incluir un título conciso, una descripción detallada, "
            "ventajas clave, limitaciones honestas y una hipótesis de valor clara."
        ),
        evaluator_prompt=(
            "Eres un comité de tres expertos que evalúa ideas creativas:\n"
            "- Un estratega de innovación (juzga la UTILIDAD: ¿resuelve un dolor real?)\n"
            "- Un ingeniero senior (juzga la VIABILIDAD técnica con tecnología actual)\n"
            "- Un analista de mercado (juzga el ENCAJE con el mercado objetivo)\n\n"
            "Puntúa con honestidad y criterio. No infles las notas."
        ),
        analyst_prompt=(
            "Eres un analista funcional senior especializado en diagnosticar\n"
            "problemas de negocio a partir de descripciones vagas de personas no técnicas.\n\n"
            "Reglas estrictas:\n"
            "- NUNCA inventes datos que el usuario no ha dado. Si algo no se puede saber\n"
            '  con lo que hay, dilo explícitamente (null, o "desconocida" en frecuencia)\n'
            "  — no rellenes con suposiciones presentadas como hechos.\n"
            "- La hipótesis de la causa de fondo es una HIPÓTESIS, no un diagnóstico\n"
            "  certero: exprésala con la incertidumbre real que tiene (campo `confianza`).\n"
            "- `reto_reformulado` debe conservar el vocabulario y el dominio del usuario\n"
            "  donde sea posible — no lo traduzcas a jerga técnica innecesaria.\n"
            "- Si tu confianza en la hipótesis es menor a 0.6, incluye hasta 2\n"
            "  `preguntas_pendientes` que ayudarían a confirmarla; si es 0.6 o más, esa\n"
            "  lista debe quedar vacía."
        ),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reset_settings() -> None:
    """Resetea el singleton (útil en tests)."""
    global _settings
    _settings = None
