"""Configuración centralizada con soporte para dominios y YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import BehaviorDimension, DomainConfig, DomainName

# raíz del repo: src/creative_engine/core/config.py → ../../../configs
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"


class LLMProviderConfig(BaseModel):
    """Configuración de un proveedor LLM (API OpenAI-compatible)."""

    name: str = "openai"
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None
    model: str = "gpt-4o-mini"
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
    mutation_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    random_injection_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    max_generation_time_seconds: float = 300.0
    # k vecinos para el cálculo objetivo de novedad
    novelty_k_nearest: int = Field(default=5, ge=1, le=50)


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

    llm: dict[str, LLMProviderConfig] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)

    domains: dict[DomainName, DomainConfig] = Field(default_factory=dict)

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

        if _CONFIGS_DIR.exists():
            for yaml_file in sorted(_CONFIGS_DIR.glob("*.yaml")):
                try:
                    domain_cfg = DomainConfig.model_validate(
                        yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                    )
                    settings.domains[domain_cfg.name] = domain_cfg
                except Exception as e:
                    import structlog

                    structlog.get_logger(__name__).warning(
                        "domain_config_invalid", file=str(yaml_file), error=str(e)
                    )

        if DomainName.GENERIC not in settings.domains:
            settings.domains[DomainName.GENERIC] = default_generic_domain()

        return settings

    def get_domain(self, name: DomainName) -> DomainConfig:
        return self.domains.get(name, self.domains[DomainName.GENERIC])


def default_generic_domain() -> DomainConfig:
    """Dominio genérico embebido: garantiza arranque sin configs/."""
    return DomainConfig(
        name=DomainName.GENERIC,
        display_name="Creatividad General",
        description="Configuración genérica para cualquier dominio creativo",
        descriptor_mode="embedding",
        behavior_dimensions=[
            BehaviorDimension(name="semantica_1", bins=10),
            BehaviorDimension(name="semantica_2", bins=10),
            BehaviorDimension(name="semantica_3", bins=8),
        ],
        system_prompt=(
            "Eres un experto en innovación y creatividad computacional. "
            "Generas ideas que son simultáneamente novedosas, útiles y viables. "
            "Cada idea debe incluir un título conciso, una descripción detallada, "
            "ventajas clave, limitaciones honestas y una hipótesis de valor clara."
        ),
        evaluation_criteria=[
            "La idea resuelve un problema real o crea valor significativo",
            "La idea es técnicamente viable con la tecnología actual o próxima",
            "La idea tiene potencial de impacto medible",
        ],
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
