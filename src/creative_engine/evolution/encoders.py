"""Codificadores: puente entre ideas textuales y representaciones numéricas.

Opción B del diseño:

- `genome_vector`: embedding semántico normalizado de la idea.
- `behavior_descriptor` (modo "embedding"): proyección aleatoria
  DETERMINISTA del genoma a N dimensiones, aplastada a [0,1] con una
  sigmoide. Dos ideas semánticamente distintas caen en celdas distintas
  del grid → la diversidad de MAP-Elites es diversidad de contenido real.
- `objective_novelty`: distancia coseno media a los k élites más
  cercanos del archivo. Sustituye el juicio subjetivo de un LLM.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import structlog

from ..core.exceptions import EncodingError
from ..core.models import DomainConfig, Idea

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Semilla fija: la proyección debe ser idéntica entre procesos y sesiones,
# de lo contrario los descriptores almacenados dejarían de ser comparables.
_PROJECTION_SEED = 1234
_SIGMOID_GAIN = 1.7  # con embeddings normalizados, z~N(0,1) → buena dispersión en (0,1)

_projection_cache: dict[tuple[int, int], np.ndarray] = {}


def _projection_matrix(in_dim: int, out_dim: int) -> np.ndarray:
    key = (in_dim, out_dim)
    if key not in _projection_cache:
        rng = np.random.default_rng(_PROJECTION_SEED)
        _projection_cache[key] = rng.normal(size=(in_dim, out_dim))
    return _projection_cache[key]


def project_to_descriptor(genome: list[float], n_dims: int) -> list[float]:
    """Proyecta un embedding normalizado a un descriptor en [0,1]^n_dims."""
    g = np.asarray(genome, dtype=np.float64)
    if g.size == 0:
        raise EncodingError("Genoma vacío: no se puede proyectar a descriptor")

    norm = np.linalg.norm(g)
    if norm > 0:
        g = g / norm

    z = g @ _projection_matrix(g.shape[0], n_dims)
    descriptor = 1.0 / (1.0 + np.exp(-_SIGMOID_GAIN * z))
    return [float(min(max(v, 0.0), 1.0)) for v in descriptor]


def objective_novelty(
    genome: list[float],
    archive_genomes: list[list[float]],
    k: int = 5,
) -> float:
    """Novedad objetiva: distancia coseno media a los k élites más cercanos.

    Rango [0,1]: 0 = idéntica a lo ya archivado, 1 = ortogonal/opuesta.
    Con archivo vacío devuelve 1.0 (todo es novedoso al principio).
    """
    if not archive_genomes:
        return 1.0

    g = np.asarray(genome, dtype=np.float64)
    g_norm = np.linalg.norm(g)
    if g_norm == 0:
        return 0.0
    g = g / g_norm

    dists: list[float] = []
    for other in archive_genomes:
        o = np.asarray(other, dtype=np.float64)
        o_norm = np.linalg.norm(o)
        if o_norm == 0 or o.shape != g.shape:
            continue
        sim = float(np.dot(g, o / o_norm))  # [-1, 1]
        dists.append((1.0 - sim) / 2.0)  # → [0, 1]

    if not dists:
        return 1.0

    dists.sort()
    nearest = dists[: max(1, min(k, len(dists)))]
    return float(np.clip(np.mean(nearest), 0.0, 1.0))


class IdeaEncoder:
    """Codifica ideas a representaciones numéricas para el motor QD.

    `embed_fn` permite inyectar una función de embedding (tests, otros
    backends). Si no se proporciona, se carga sentence-transformers de
    forma perezosa — así el resto del sistema no depende de torch.
    """

    def __init__(
        self,
        embedding_model: str = _DEFAULT_MODEL,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._log = logger.bind(model=embedding_model)
        self._embed_fn = embed_fn
        self._model_name = embedding_model
        self._model = None  # carga perezosa

    def _embed(self, text: str) -> list[float]:
        if self._embed_fn is not None:
            return list(self._embed_fn(text))

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                self._log.info("encoder_loaded")
            except Exception as e:
                raise EncodingError(
                    f"No se pudo cargar el modelo de embeddings '{self._model_name}': {e}",
                    details={"model": self._model_name},
                ) from e

        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_genome(self, idea: Idea) -> list[float]:
        """Embedding del contenido completo de la idea."""
        try:
            combined = f"TÍTULO: {idea.title}\nDESCRIPCIÓN: {idea.description}"

            if idea.advantages:
                combined += "\nVENTAJAS: " + "; ".join(idea.advantages)

            if idea.features and idea.features.technologies:
                combined += "\nTECNOLOGÍAS: " + ", ".join(idea.features.technologies)

            return self._embed(combined)
        except EncodingError:
            raise
        except Exception as e:
            raise EncodingError(
                f"Error codificando genoma de idea {idea.id}: {e}",
                details={"idea_id": idea.id},
            ) from e

    def compute_behavior_descriptor(self, idea: Idea, domain: DomainConfig) -> list[float]:
        """Descriptor de comportamiento según el modo del dominio."""
        n_dims = len(domain.behavior_dimensions)

        if domain.descriptor_mode == "embedding":
            if not idea.genome_vector:
                raise EncodingError(
                    f"Idea {idea.id} sin genoma: codifica el genoma antes del descriptor"
                )
            return project_to_descriptor(idea.genome_vector, n_dims)

        # Modo legado "metrics": derivado de las puntuaciones de evaluación
        if idea.evaluation is None:
            raise EncodingError(
                f"No se puede calcular descriptor: idea {idea.id} no está evaluada"
            )

        descriptor = []
        for dim in domain.behavior_dimensions:
            value = getattr(idea.evaluation, dim.source_metric, 0.5)
            normalized = (value - dim.min_value) / (dim.max_value - dim.min_value + 1e-8)
            descriptor.append(float(min(max(normalized, 0.0), 1.0)))
        return descriptor

    def encode_idea(self, idea: Idea, domain: DomainConfig) -> Idea:
        """Codifica completamente una idea (genoma + descriptor) in-place."""
        idea.genome_vector = self.encode_genome(idea)
        idea.behavior_descriptor = self.compute_behavior_descriptor(idea, domain)
        return idea


class SearchResultScorer:
    """Puntúa similitud entre ideas (para el motor de recomendación)."""

    def score_similarity(self, query_idea: Idea, candidate: Idea) -> float:
        q = np.array(query_idea.genome_vector)
        c = np.array(candidate.genome_vector)

        if q.shape != c.shape or q.size == 0:
            return 0.0

        norm_q = np.linalg.norm(q)
        norm_c = np.linalg.norm(c)
        if norm_q == 0 or norm_c == 0:
            return 0.0

        return float(np.dot(q, c) / (norm_q * norm_c))

    def score_behavior_distance(self, idea_a: Idea, idea_b: Idea) -> float:
        a = np.array(idea_a.behavior_descriptor)
        b = np.array(idea_b.behavior_descriptor)

        if a.shape != b.shape or a.size == 0:
            return 1.0

        max_dist = np.sqrt(len(a))
        return float(np.linalg.norm(a - b) / max_dist)
