"""Knowledge Graph: relaciones entre ideas usando Neo4j.

Relaciones semánticas soportadas:
- EVOLUCIONA_DE (mutación)
- COMBINA_ELEMENTOS_DE (cruce)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ..core.config import get_settings
from ..core.exceptions import GraphQueryError
from ..core.models import Idea

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)


class IdeaKnowledgeGraph:
    """Interfaz con Neo4j para el grafo de conocimiento de ideas."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        settings = get_settings()
        self._uri = uri or settings.database.neo4j_uri
        self._user = user or settings.database.neo4j_user
        self._password = password or settings.database.neo4j_password
        self._driver: AsyncDriver | None = None
        self._log = logger.bind(component="KnowledgeGraph")

    async def connect(self) -> None:
        # Import perezoso: `neo4j` es una dependencia opcional (extra
        # "graph"). Nada en el flujo activo del motor la necesita: solo
        # se paga el import si alguien conecta este grafo de verdad.
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
        await self._driver.verify_connectivity()
        self._log.info("graph_connected", uri=self._uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def store_idea_node(self, idea: Idea) -> None:
        """Crea o actualiza un nodo de idea en el grafo."""
        if not self._driver:
            return

        query = """
        MERGE (i:Idea {id: $id})
        SET i.title = $title,
            i.domain = $domain,
            i.fitness = $fitness,
            i.generation = $generation,
            i.novelty = $novelty,
            i.created_at = datetime($created_at)
        """
        params = {
            "id": idea.id,
            "title": idea.title,
            "domain": idea.domain.value,
            "fitness": idea.fitness,
            "generation": idea.generation,
            "novelty": idea.evaluation.novelty if idea.evaluation else 0.0,
            "created_at": idea.created_at.isoformat(),
        }

        try:
            async with self._driver.session() as session:
                await session.run(query, params)
        except Exception as e:
            self._log.error("graph_store_failed", idea_id=idea.id, error=str(e))
            raise GraphQueryError(f"Error almacenando idea {idea.id}: {e}") from e

    async def store_evolution_relationship(
        self,
        child_id: str,
        parent_ids: list[str],
        rel_type: str = "EVOLUCIONA_DE",
    ) -> None:
        """Crea relaciones de evolución entre ideas."""
        if not self._driver or not parent_ids:
            return

        if rel_type not in {"EVOLUCIONA_DE", "COMBINA_ELEMENTOS_DE"}:
            raise GraphQueryError(f"Tipo de relación no permitido: {rel_type}")

        query = f"""
        MATCH (child:Idea {{id: $child_id}})
        MATCH (parent:Idea {{id: $parent_id}})
        MERGE (child)-[r:{rel_type}]->(parent)
        """
        for pid in parent_ids:
            try:
                async with self._driver.session() as session:
                    await session.run(query, {"child_id": child_id, "parent_id": pid})
            except Exception as e:
                self._log.warning("graph_rel_failed", child=child_id, parent=pid, error=str(e))

    async def find_related_ideas(
        self, idea_id: str, max_depth: int = 2, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Encuentra ideas relacionadas navegando el grafo."""
        if not self._driver:
            return []

        max_depth = max(1, min(int(max_depth), 4))
        limit = max(1, min(int(limit), 100))

        query = f"""
        MATCH (i:Idea {{id: $idea_id}})-[r*1..{max_depth}]-(related:Idea)
        WHERE related.id <> $idea_id
        RETURN DISTINCT related.id AS id, related.title AS title,
               related.fitness AS fitness, related.domain AS domain
        LIMIT {limit}
        """

        try:
            async with self._driver.session() as session:
                result = await session.run(query, {"idea_id": idea_id})
                return await result.data()
        except Exception as e:
            self._log.error("graph_query_failed", error=str(e))
            raise GraphQueryError(f"Error consultando grafo: {e}") from e

    async def get_evolution_lineage(self, idea_id: str) -> list[dict[str, Any]]:
        """Árbol genealógico completo de una idea."""
        if not self._driver:
            return []

        query = """
        MATCH path = (ancestor:Idea)<-[:EVOLUCIONA_DE*]-(descendant:Idea {id: $idea_id})
        UNWIND nodes(path) AS node
        RETURN DISTINCT node.id AS id, node.title AS title,
               node.generation AS generation, node.fitness AS fitness
        ORDER BY node.generation ASC
        """

        try:
            async with self._driver.session() as session:
                result = await session.run(query, {"idea_id": idea_id})
                return await result.data()
        except Exception as e:
            self._log.error("lineage_query_failed", error=str(e))
            return []
