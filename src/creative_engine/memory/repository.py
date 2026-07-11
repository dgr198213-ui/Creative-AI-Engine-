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
    DomainName,
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
        """Crea las tablas si no existen."""
        async with self._engine.begin() as conn:
            await conn.execute(text(_SCHEMA_SQL))
        self._log.info("repository_initialized")

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
                    "domain": idea.domain.value,
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
                {"domain": idea.domain.value, "exclude_id": idea.id, "limit": limit},
            )
            return [self.row_to_idea(row) for row in result.fetchall()]

    async def get_stats(self, run_id: str | None = None) -> dict[str, Any]:
        """Estadísticas del repositorio."""
        async with self._session_factory() as session:
            where = "WHERE run_id = :run_id" if run_id else ""
            params: dict[str, Any] = {"run_id": run_id} if run_id else {}

            result = await session.execute(
                text(f"""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'elite') as elites,
                        COUNT(*) FILTER (WHERE status = 'discarded') as discarded,
                        AVG((evaluation->>'weighted_score')::FLOAT) as avg_fitness,
                        MAX((evaluation->>'weighted_score')::FLOAT) as max_fitness
                    FROM ideas {where}
                """),
                params,
            )
            row = result.fetchone()
            return dict(row._mapping) if row else {}

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
            domain=DomainName(row.domain),
            evaluation=evaluation,
            genome_vector=_as_json(row.genome_vector) or [],
            behavior_descriptor=_as_json(row.behavior_descriptor) or [],
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def close(self) -> None:
        await self._engine.dispose()
