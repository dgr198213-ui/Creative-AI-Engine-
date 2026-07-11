"""Motor de recomendación: sugiere ideas relacionadas para reutilización.

Estrategia híbrida:
1. Búsqueda semántica (similitud de embeddings) sobre el repositorio.
2. Bonus de diversidad: se favorecen candidatas alejadas en el espacio
   de comportamiento para no recomendar clones de la idea consultada.
"""

from __future__ import annotations

import structlog

from ..core.models import Idea
from ..evolution.encoders import SearchResultScorer
from .graph import IdeaKnowledgeGraph
from .repository import IdeaRepository

logger = structlog.get_logger(__name__)


class RecommendationEngine:
    """Combina similitud semántica y diversidad para recomendar ideas."""

    def __init__(
        self,
        repository: IdeaRepository,
        graph: IdeaKnowledgeGraph | None = None,
    ) -> None:
        self._repository = repository
        self._graph = graph
        self._scorer = SearchResultScorer()
        self._log = logger.bind(component="RecommendationEngine")

    async def recommend_for_idea(
        self,
        query_idea: Idea,
        limit: int = 10,
        diversity_weight: float = 0.3,
    ) -> list[tuple[Idea, float]]:
        """Recomienda ideas relacionadas con una idea dada.

        Args:
            query_idea: Idea de referencia.
            limit: Máximo de recomendaciones.
            diversity_weight: Peso [0,1] del bonus por lejanía en el
                espacio de comportamiento (0 = solo similitud).

        Returns:
            Lista de (idea, score) ordenada por relevancia.
        """
        diversity_weight = max(0.0, min(1.0, diversity_weight))

        candidates = await self._repository.search_similar(query_idea, limit=limit * 3)

        scored: list[tuple[Idea, float]] = []
        for idea in candidates:
            similarity = self._scorer.score_similarity(query_idea, idea)
            behavior_dist = self._scorer.score_behavior_distance(query_idea, idea)
            final_score = similarity * (1 - diversity_weight) + behavior_dist * diversity_weight
            scored.append((idea, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def related_from_graph(self, idea_id: str, limit: int = 10) -> list[dict]:
        """Ideas relacionadas estructuralmente (linaje/cruces) vía Neo4j."""
        if self._graph is None:
            return []
        return await self._graph.find_related_ideas(idea_id, limit=limit)
