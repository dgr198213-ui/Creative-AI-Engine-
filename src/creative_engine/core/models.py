"""Modelos de datos centrales del Creative AI Engine.

Decisiones de diseño (v0.1, "opción B"):

1. La NOVEDAD no la puntúa un LLM: se calcula de forma objetiva como
   distancia de embedding al archivo de élites. Es informativa y NO
   forma parte del fitness (peso 0 por defecto).
2. El fitness es CALIDAD PURA: utilidad, viabilidad, mercado, impacto.
3. El descriptor de comportamiento se deriva del CONTENIDO (embedding
   proyectado), no de las puntuaciones → la diversidad del archivo
   MAP-Elites es diversidad semántica real.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

# ── Tipos básicos ───────────────────────────────────────────────────

type IdeaId = str
type RunId = str
type AgentName = str


def _new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex
    return f"{prefix}_{raw}" if prefix else raw


# ── Enums ───────────────────────────────────────────────────────────


class IdeaStatus(StrEnum):
    DRAFT = "draft"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    ELITE = "elite"
    MUTATED = "mutated"
    CROSSED = "crossed"
    DISCARDED = "discarded"
    ARCHIVED = "archived"


class MutationType(StrEnum):
    FUNCTIONALITY = "functionality"
    TECHNOLOGY = "technology"
    MATERIAL = "material"
    PROCESS = "process"
    TARGET_MARKET = "target_market"
    BUSINESS_MODEL = "business_model"
    HYBRID = "hybrid"


class DomainName(StrEnum):
    INDUSTRIAL_DESIGN = "industrial_design"
    MARKETING = "marketing"
    ARCHITECTURE = "architecture"
    VIDEOGAMES = "videogames"
    RESEARCH = "research"
    STARTUPS = "startups"
    GENERIC = "generic"


# ── Evaluación ──────────────────────────────────────────────────────

# Pesos por defecto: SOLO calidad. `novelty` queda fuera del fitness
# a propósito: la diversidad ya la garantiza MAP-Elites y la novedad
# se mide de forma objetiva (distancia de embedding al archivo).
DEFAULT_WEIGHTS: dict[str, float] = {
    "novelty": 0.0,
    "utility": 0.30,
    "feasibility": 0.25,
    "impact": 0.20,
    "market_fit": 0.15,
    "sustainability": 0.05,
    "scalability": 0.05,
}


class EvaluationScores(BaseModel):
    """Puntuaciones multidimensionales de una idea, normalizadas en [0, 1].

    `novelty` es calculada por el motor (objetiva, basada en embeddings),
    no por un agente LLM. El resto proviene de los agentes evaluadores.
    """

    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    utility: float = Field(default=0.0, ge=0.0, le=1.0)
    feasibility: float = Field(default=0.0, ge=0.0, le=1.0)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    impact: float = Field(default=0.0, ge=0.0, le=1.0)
    market_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    sustainability: float = Field(default=0.0, ge=0.0, le=1.0)
    scalability: float = Field(default=0.0, ge=0.0, le=1.0)

    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # Feedback textual por agente
    agent_feedback: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def weighted_score(self) -> float:
        """Puntuación global ponderada (fitness de calidad)."""
        scores = {
            "novelty": self.novelty,
            "utility": self.utility,
            "feasibility": self.feasibility,
            "impact": self.impact,
            "market_fit": self.market_fit,
            "sustainability": self.sustainability,
            "scalability": self.scalability,
        }
        total_weight = sum(self.weights.get(k, 0.0) for k in scores)
        if total_weight == 0:
            return 0.0
        return sum(scores[k] * self.weights.get(k, 0.0) for k in scores) / total_weight

    @computed_field  # type: ignore[prop-decorator]
    @property
    def as_vector(self) -> list[float]:
        """Vector de puntuaciones para cálculos numéricos."""
        return [
            self.novelty,
            self.utility,
            self.feasibility,
            self.complexity,
            self.impact,
            self.market_fit,
            self.sustainability,
            self.scalability,
        ]


# ── Idea ────────────────────────────────────────────────────────────


class ValueHypothesis(BaseModel):
    """Hipótesis de valor asociada a una idea."""

    target_user: str = Field(..., min_length=1)
    problem_solved: str = Field(..., min_length=1)
    value_proposition: str = Field(..., min_length=1)
    differentiation: str = Field(default="")


class IdeaFeatures(BaseModel):
    """Características estructuradas extraídas de la idea."""

    primary_function: str = ""
    technologies: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    target_markets: list[str] = Field(default_factory=list)
    complexity_level: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty_indicators: list[str] = Field(default_factory=list)


class Idea(BaseModel):
    """Modelo central: una idea creativa dentro del ecosistema evolutivo."""

    id: IdeaId = Field(default_factory=lambda: _new_id("idea"))
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=5000)

    advantages: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    value_hypothesis: ValueHypothesis | None = None
    features: IdeaFeatures = Field(default_factory=IdeaFeatures)

    # Metadatos evolutivos
    status: IdeaStatus = IdeaStatus.DRAFT
    generation: int = Field(default=0, ge=0)
    run_id: RunId = ""
    parent_ids: list[IdeaId] = Field(default_factory=list)
    mutation_type: MutationType | None = None
    domain: DomainName = DomainName.GENERIC

    # Evaluación (agentes LLM + novelty objetiva del motor)
    evaluation: EvaluationScores | None = None

    # Representación numérica (para QD)
    genome_vector: list[float] = Field(default_factory=list)
    behavior_descriptor: list[float] = Field(default_factory=list)

    # Metadatos temporales
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fitness(self) -> float:
        """Aptitud global: media ponderada de la evaluación (calidad)."""
        if self.evaluation is None:
            return 0.0
        return self.evaluation.weighted_score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """Hash del contenido para detección de duplicados."""
        canonical = f"{self.title}|{self.description}|{sorted(self.advantages)}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @model_validator(mode="after")
    def sync_timestamps(self) -> Idea:
        if self.updated_at < self.created_at:
            self.updated_at = self.created_at
        return self

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()


# ── Población y Archivo ─────────────────────────────────────────────


class MAPElitesCell(BaseModel):
    """Una celda ocupada del archivo MAP-Elites."""

    cell_index: tuple[int, ...]
    elite: Idea
    fitness: float


class EvolutionState(BaseModel):
    """Estado completo de una ejecución evolutiva."""

    run_id: RunId = Field(default_factory=lambda: _new_id("run"))
    challenge: str = ""
    domain: DomainName = DomainName.GENERIC
    generation: int = 0
    total_generations: int = 10
    population_size: int = 20
    archive: list[MAPElitesCell] = Field(default_factory=list)
    all_ideas: list[IdeaId] = Field(default_factory=list)
    coverage: float = 0.0
    qd_score: float = 0.0
    best_fitness: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    is_running: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_complete(self) -> bool:
        return self.generation >= self.total_generations


# ── Configuración de Dominio ────────────────────────────────────────


class BehaviorDimension(BaseModel):
    """Una dimensión del espacio de comportamiento (MAP-Elites).

    En modo `embedding` (por defecto) el valor proviene de una proyección
    determinista del embedding de la idea; `source_metric` se ignora.
    En modo `metrics` se usa la métrica indicada de EvaluationScores.
    """

    name: str
    description: str = ""
    source_metric: str = ""
    bins: int = Field(default=10, ge=2, le=100)
    min_value: float = 0.0
    max_value: float = 1.0


class DomainConfig(BaseModel):
    """Configuración completa para un dominio creativo."""

    name: DomainName
    display_name: str
    description: str = ""

    # Cómo se calcula el descriptor de comportamiento:
    #   "embedding": proyección determinista del embedding (diversidad semántica real)
    #   "metrics":   métricas de evaluación (modo legado / interpretable)
    descriptor_mode: Literal["embedding", "metrics"] = "embedding"

    behavior_dimensions: list[BehaviorDimension] = Field(min_length=2, max_length=5)

    evaluation_weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    allowed_mutations: list[MutationType] = Field(default_factory=lambda: list(MutationType))

    # Defaults económicos: ~20 x 10 ≈ cientos de llamadas LLM, no decenas de miles.
    default_population_size: int = Field(default=20, ge=4, le=10000)
    default_generations: int = Field(default=4, ge=1, le=1000)

    system_prompt: str = ""
    evaluation_criteria: list[str] = Field(default_factory=list)

    @field_validator("evaluation_weights")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Los pesos deben sumar ~1.0, obtuve {total:.4f}")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(dim.bins for dim in self.behavior_dimensions)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cells(self) -> int:
        result = 1
        for dim in self.behavior_dimensions:
            result *= dim.bins
        return result


# ── Solicitudes y Respuestas API ────────────────────────────────────


class EvolutionRequest(BaseModel):
    """Solicitud para iniciar una evolución."""

    challenge: str = Field(..., min_length=10, max_length=5000)
    domain: DomainName = DomainName.GENERIC
    population_size: int | None = Field(default=None, ge=4, le=500)
    generations: int | None = Field(default=None, ge=1, le=200)
    custom_weights: dict[str, float] | None = None


class EvolutionResponse(BaseModel):
    """Respuesta resumen de una evolución completada."""

    run_id: RunId
    challenge: str
    domain: DomainName
    generations_completed: int
    total_ideas_generated: int
    elite_count: int
    coverage: float
    qd_score: float
    best_fitness: float
    top_ideas: list[Idea]
    completed_at: datetime | None = None


class IdeaDetailResponse(BaseModel):
    """Detalle completo de una idea."""

    idea: Idea
    related_ideas: list[Idea] = Field(default_factory=list)
    evolution_lineage: list[Idea] = Field(default_factory=list)


Idea.model_rebuild()
