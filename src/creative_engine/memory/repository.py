"""Repositorio de ideas: almacenamiento persistente en PostgreSQL."""

from __future__ import annotations

import json
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..core.config import get_settings
from ..core.exceptions import IdeaNotFoundError
from ..core.models import (
    EvaluationScores,
    Idea,
    IdeaFeatures,
    IdeaStatus,
    MutationType,
    ValueHypothesis,
)

logger = structlog.get_logger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ideas (
    id                  VARCHAR(64) PRIMARY KEY,
    title               VARCHAR(200) NOT NULL,
    description         TEXT NOT NULL,
    advantages          JSONB DEFAULT '[]',
    limitations         JSONB DEFAULT '[]',
    value_hypothesis    JSONB,
    features            JSONB DEFAULT '{}',
    status              VARCHAR(20) NOT NULL DEFAULT 'draft',
    generation          INTEGER NOT NULL DEFAULT 0,
    run_id              VARCHAR(64) NOT NULL DEFAULT '',
    parent_ids          JSONB DEFAULT '[]',
    mutation_type       VARCHAR(30),
    domain              VARCHAR(30) NOT NULL DEFAULT 'generic',
    evaluation          JSONB,
    genome_vector       JSONB DEFAULT '[]',
    behavior_descriptor JSONB DEFAULT '[]',
    content_hash        VARCHAR(16) NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ideas_run_id ON ideas(run_id);
CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);
CREATE INDEX IF NOT EXISTS idx_ideas_domain ON ideas(domain);
CREATE INDEX IF NOT EXISTS idx_ideas_generation ON ideas(generation);
CREATE INDEX IF NOT EXISTS idx_ideas_fitness ON ideas(
    ((evaluation->>'weighted_score')::FLOAT)
);

CREATE TABLE IF NOT EXISTS run_status (
    run_id      VARCHAR(64) PRIMARY KEY,
    status      VARCHAR(20) NOT NULL DEFAULT 'running',
    error       TEXT,
    challenge   TEXT NOT NULL DEFAULT '',
    domain      VARCHAR(30) NOT NULL DEFAULT 'generic',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bench_results (
    id           BIGSERIAL PRIMARY KEY,
    set_name     VARCHAR(100) NOT NULL,
    challenge    TEXT NOT NULL,
    reto_tipo    VARCHAR(20) NOT NULL,
    repetition   INTEGER NOT NULL DEFAULT 0,
    arms         JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bench_results_set_name ON bench_results(set_name);

CREATE TABLE IF NOT EXISTS provider_spend (
    provider           VARCHAR(64) NOT NULL,
    period_key         VARCHAR(20) NOT NULL,
    cost_usd           DOUBLE PRECISION NOT NULL DEFAULT 0,
    prompt_tokens      BIGINT NOT NULL DEFAULT 0,
    completion_tokens  BIGINT NOT NULL DEFAULT 0,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (provider, period_key)
);
"""


def _as_json(value: Any) -> Any:
    """Los drivers pueden devolver JSONB como str o como objeto ya parseado."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


class IdeaRepository:
    """Repositorio async de ideas con PostgreSQL."""

    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        url = database_url or settings.database.postgres_url

        self._engine = create_async_engine(url, echo=settings.debug, pool_size=10)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._log = logger.bind(component="IdeaRepository")

    async def initialize(self) -> None:
        """Crea las tablas e índices si no existen.

        asyncpg no permite múltiples comandos en un solo prepared statement,
        así que ejecutamos cada sentencia (CREATE TABLE, cada CREATE INDEX)
        de forma individual en lugar de enviar el bloque entero de golpe.
        """
        statements = [s.strip() for s in _SCHEMA_SQL.split(";") if s.strip()]
        async with self._engine.begin() as conn:
            for statement in statements:
                await conn.execute(text(statement))
        self._log.info("repository_initialized", statements=len(statements))

    async def store_idea(self, idea: Idea) -> Idea:
        """Almacena o actualiza una idea (upsert)."""
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO ideas (
                        id, title, description, advantages, limitations,
                        value_hypothesis, features, status, generation,
                        run_id, parent_ids, mutation_type, domain,
                        evaluation, genome_vector, behavior_descriptor,
                        content_hash, created_at, updated_at
                    ) VALUES (
                        :id, :title, :description, CAST(:advantages AS jsonb),
                        CAST(:limitations AS jsonb), CAST(:value_hypothesis AS jsonb),
                        CAST(:features AS jsonb), :status, :generation,
                        :run_id, CAST(:parent_ids AS jsonb), :mutation_type,
                        :domain, CAST(:evaluation AS jsonb), CAST(:genome_vector AS jsonb),
                        CAST(:behavior_descriptor AS jsonb), :content_hash,
                        :created_at, :updated_at
                    ) ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        advantages = EXCLUDED.advantages,
                        limitations = EXCLUDED.limitations,
                        features = EXCLUDED.features,
                        status = EXCLUDED.status,
                        evaluation = EXCLUDED.evaluation,
                        genome_vector = EXCLUDED.genome_vector,
                        behavior_descriptor = EXCLUDED.behavior_descriptor,
                        updated_at = EXCLUDED.updated_at
                """),
                {
                    "id": idea.id,
                    "title": idea.title,
                    "description": idea.description,
                    "advantages": json.dumps(idea.advantages),
                    "limitations": json.dumps(idea.limitations),
                    "value_hypothesis": json.dumps(
                        idea.value_hypothesis.model_dump() if idea.value_hypothesis else None
                    ),
                    "features": json.dumps(idea.features.model_dump()),
                    "status": idea.status.value,
                    "generation": idea.generation,
                    "run_id": idea.run_id,
                    "parent_ids": json.dumps(idea.parent_ids),
                    "mutation_type": idea.mutation_type.value if idea.mutation_type else None,
                    "domain": idea.domain,
                    "evaluation": json.dumps(
                        idea.evaluation.model_dump() if idea.evaluation else None
                    ),
                    "genome_vector": json.dumps(idea.genome_vector),
                    "behavior_descriptor": json.dumps(idea.behavior_descriptor),
                    "content_hash": idea.content_hash,
                    "created_at": idea.created_at,
                    "updated_at": idea.updated_at,
                },
            )
            await session.commit()

        return idea

    async def get_idea(self, idea_id: str) -> Idea:
        """Recupera una idea por ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM ideas WHERE id = :id"), {"id": idea_id}
            )
            row = result.fetchone()
            if not row:
                raise IdeaNotFoundError(f"Idea {idea_id} no encontrada")
            return self.row_to_idea(row)

    async def get_elites_by_run(self, run_id: str, limit: int = 50) -> list[Idea]:
        """Ideas élite de una ejecución, ordenadas por fitness."""
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT * FROM ideas
                    WHERE run_id = :run_id AND status = 'elite'
                    ORDER BY (evaluation->>'weighted_score')::FLOAT DESC NULLS LAST
                    LIMIT :limit
                """),
                {"run_id": run_id, "limit": limit},
            )
            return [self.row_to_idea(row) for row in result.fetchall()]

    async def get_recent_elites(
        self, limit: int = 200, exclude_run_id: str | None = None
    ) -> list[Idea]:
        """Élites recientes de runs anteriores (memoria entre ejecuciones).

        Alimenta el grounding del generador: ideas ya descubiertas en retos
        pasados que pueden servir de inspiración/repulsión en retos nuevos.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT * FROM ideas
                    WHERE status = 'elite'
                      AND (CAST(:exclude_run_id AS TEXT) IS NULL
                           OR run_id != CAST(:exclude_run_id AS TEXT))
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"exclude_run_id": exclude_run_id, "limit": limit},
            )
            return [self.row_to_idea(row) for row in result.fetchall()]

    async def get_lineage(self, idea_id: str, max_depth: int = 25) -> list[Idea]:
        """Linaje de una idea siguiendo el primer padre (con tope de profundidad)."""
        lineage: list[Idea] = []
        visited = {idea_id}
        current_id: str | None = idea_id

        while current_id and len(lineage) < max_depth:
            try:
                idea = await self.get_idea(current_id)
            except IdeaNotFoundError:
                break
            lineage.append(idea)

            if idea.parent_ids:
                next_id = idea.parent_ids[0]
                if next_id in visited:
                    break
                visited.add(next_id)
                current_id = next_id
            else:
                break

        return lineage

    async def search_similar(self, idea: Idea, limit: int = 10) -> list[Idea]:
        """Búsqueda básica por dominio y estado (pgvector: roadmap)."""
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT * FROM ideas
                    WHERE domain = :domain
                      AND status = 'elite'
                      AND id != :exclude_id
                    ORDER BY (evaluation->>'weighted_score')::FLOAT DESC NULLS LAST
                    LIMIT :limit
                """),
                {"domain": idea.domain, "exclude_id": idea.id, "limit": limit},
            )
            return [self.row_to_idea(row) for row in result.fetchall()]

    async def get_stats(self, run_id: str | None = None) -> dict[str, Any]:
        """Estadísticas del repositorio (globales, o de un run si se indica)."""
        async with self._session_factory() as session:
            # Sin f-string en el WHERE (auditoría M2): un único texto SQL fijo,
            # el filtro opcional se resuelve por bind param, igual que en
            # get_recent_elites.
            result = await session.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'elite') as elites,
                        COUNT(*) FILTER (WHERE status = 'discarded') as discarded,
                        AVG((evaluation->>'weighted_score')::FLOAT) as avg_fitness,
                        MAX((evaluation->>'weighted_score')::FLOAT) as max_fitness
                    FROM ideas
                    WHERE CAST(:run_id AS TEXT) IS NULL OR run_id = CAST(:run_id AS TEXT)
                """),
                {"run_id": run_id},
            )
            row = result.fetchone()
            return dict(row._mapping) if row else {}

    async def save_run_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
        challenge: str = "",
        domain: str = "generic",
    ) -> None:
        """Guarda el estado final de un run (running/completed/failed).

        Permite a `/runs/{run_id}/export` distinguir un run que falló sin
        generar ideas (devolver el estado failed) de un run inexistente
        (404), sin depender solo de si hay élites en `ideas`.
        """
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO run_status (run_id, status, error, challenge, domain, updated_at)
                    VALUES (:run_id, :status, :error, :challenge, :domain, NOW())
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        error = EXCLUDED.error,
                        updated_at = EXCLUDED.updated_at
                """),
                {
                    "run_id": run_id,
                    "status": status,
                    "error": error,
                    "challenge": challenge,
                    "domain": domain,
                },
            )
            await session.commit()

    async def get_run_status(self, run_id: str) -> dict[str, Any] | None:
        """Estado final persistido de un run, si existe."""
        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM run_status WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            row = result.fetchone()
            return dict(row._mapping) if row else None

    async def save_bench_result(
        self,
        set_name: str,
        challenge: str,
        reto_tipo: str,
        repetition: int,
        arms: dict[str, Any],
    ) -> None:
        """Persiste el resultado de los 3 brazos para un reto/repetición."""
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO bench_results
                        (set_name, challenge, reto_tipo, repetition, arms)
                    VALUES
                        (:set_name, :challenge, :reto_tipo, :repetition, CAST(:arms AS jsonb))
                """),
                {
                    "set_name": set_name,
                    "challenge": challenge,
                    "reto_tipo": reto_tipo,
                    "repetition": repetition,
                    "arms": json.dumps(arms),
                },
            )
            await session.commit()

    async def get_bench_results(self, set_name: str) -> list[dict[str, Any]]:
        """Todos los resultados persistidos de un set de benchmark."""
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT challenge, reto_tipo, repetition, arms, created_at
                    FROM bench_results
                    WHERE set_name = :set_name
                    ORDER BY created_at ASC
                """),
                {"set_name": set_name},
            )
            rows = []
            for row in result.fetchall():
                d = dict(row._mapping)
                d["arms"] = _as_json(d["arms"]) or {}
                rows.append(d)
            return rows

    async def record_provider_spend(
        self,
        provider: str,
        period_key: str,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Acumula el gasto estimado de un proveedor en un periodo (upsert).

        Fase 5, bloque 3 (guard de presupuesto): el coste es una
        ESTIMACIÓN (tokens x precio configurado), no la factura real —
        ver CLAUDE.md. Suma sobre lo ya acumulado en vez de sobrescribir,
        para que llamadas de runs distintos en el mismo periodo se agreguen.
        """
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO provider_spend
                        (provider, period_key, cost_usd, prompt_tokens, completion_tokens, updated_at)
                    VALUES
                        (:provider, :period_key, :cost_usd, :prompt_tokens, :completion_tokens, NOW())
                    ON CONFLICT (provider, period_key) DO UPDATE SET
                        cost_usd = provider_spend.cost_usd + EXCLUDED.cost_usd,
                        prompt_tokens = provider_spend.prompt_tokens + EXCLUDED.prompt_tokens,
                        completion_tokens = provider_spend.completion_tokens + EXCLUDED.completion_tokens,
                        updated_at = NOW()
                """),
                {
                    "provider": provider,
                    "period_key": period_key,
                    "cost_usd": cost_usd,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )
            await session.commit()

    async def get_period_spend(self, period_key: str) -> dict[str, float]:
        """Gasto estimado (USD) por proveedor acumulado en el periodo dado."""
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT provider, cost_usd FROM provider_spend WHERE period_key = :period_key"
                ),
                {"period_key": period_key},
            )
            return {row.provider: float(row.cost_usd) for row in result.fetchall()}

    @staticmethod
    def row_to_idea(row: Any) -> Idea:
        """Convierte una fila SQL en un objeto Idea."""
        eval_data = _as_json(row.evaluation)
        evaluation = None
        if eval_data:
            eval_data.pop("weighted_score", None)  # computed field
            eval_data.pop("as_vector", None)
            evaluation = EvaluationScores.model_validate(eval_data)

        vh_data = _as_json(row.value_hypothesis)
        value_hypothesis = ValueHypothesis.model_validate(vh_data) if vh_data else None

        features = IdeaFeatures.model_validate(_as_json(row.features) or {})

        return Idea(
            id=row.id,
            title=row.title,
            description=row.description,
            advantages=_as_json(row.advantages) or [],
            limitations=_as_json(row.limitations) or [],
            value_hypothesis=value_hypothesis,
            features=features,
            status=IdeaStatus(row.status),
            generation=row.generation,
            run_id=row.run_id,
            parent_ids=_as_json(row.parent_ids) or [],
            mutation_type=MutationType(row.mutation_type) if row.mutation_type else None,
            domain=row.domain,
            evaluation=evaluation,
            genome_vector=_as_json(row.genome_vector) or [],
            behavior_descriptor=_as_json(row.behavior_descriptor) or [],
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def close(self) -> None:
        await self._engine.dispose()
