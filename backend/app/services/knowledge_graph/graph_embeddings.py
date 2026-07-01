import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GraphEmbeddingService:
    """Generates and manages embeddings for graph entities."""

    def __init__(self, embedding_service=None, graph_service=None):
        self.embedding_service = embedding_service
        self.graph_service = graph_service

    async def embed_entity(self, entity: dict) -> list[float] | None:
        """Generate embedding for a single entity based on its name and description."""
        if not self.embedding_service:
            return None
        text = f"{entity.get('name', '')} {entity.get('description', '')}"
        if not text.strip():
            return None
        try:
            return self.embedding_service.encode_query(text)
        except Exception as e:
            logger.warning(f"Entity embedding failed: {e}")
            return None

    async def embed_all_entities(self) -> int:
        """Generate embeddings for all entities in the graph."""
        if not self.embedding_service or not self.graph_service:
            return 0

        embedded_count = 0
        offset = 0
        batch_size = 50

        while True:
            entities, total = await self.graph_service.get_all_entities(
                limit=batch_size, offset=offset
            )
            if not entities:
                break

            for entity in entities:
                embedding = await self.embed_entity(entity)
                if embedding:
                    embedded_count += 1

            offset += batch_size
            if offset >= total:
                break

        logger.info(f"Embedded {embedded_count} graph entities")
        return embedded_count

    def build_entity_text(self, entity: dict) -> str:
        """Build a text representation of an entity for embedding."""
        parts = [entity.get("name", "")]
        if entity.get("description"):
            parts.append(entity["description"])
        if entity.get("entity_type"):
            parts.append(f"Type: {entity['entity_type']}")
        if entity.get("aliases"):
            parts.append(f"Aliases: {', '.join(entity['aliases'])}")
        return " | ".join(parts)
