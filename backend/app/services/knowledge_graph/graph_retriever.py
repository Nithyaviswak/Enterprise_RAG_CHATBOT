import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Hybrid graph + vector retrieval with automatic strategy selection."""

    def __init__(self, graph_service=None, embedding_service=None, vector_store=None):
        self.graph_service = graph_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def classify_query(self, query: str) -> str:
        """Classify query to determine retrieval strategy."""
        query_lower = query.lower()

        entity_indicators = [
            "who is", "what is", "tell me about", "describe",
            "organization", "company", "person", "location",
            "product", "technology", "tool", "framework",
        ]

        relationship_indicators = [
            "relationship between", "connection", "how does",
            "work for", "works at", "acquired", "owned by",
            "partner", "collaborate", "invest",
        ]

        comparison_indicators = [
            "compare", "difference between", "vs", "versus",
            "similarities", "different",
        ]

        for indicator in relationship_indicators:
            if indicator in query_lower:
                return "graph_traversal"

        for indicator in entity_indicators:
            if indicator in query_lower:
                return "graph_entity"

        for indicator in comparison_indicators:
            if indicator in query_lower:
                return "graph_comparison"

        if len(query.split()) <= 5:
            return "vector"

        return "hybrid"

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        entity_types: list[str] | None = None,
        use_hybrid_search: bool = True,
    ) -> dict:
        """Retrieve using automatic strategy selection.

        Returns dict with entities, relationships, paths, vector_results, query_type.
        """
        query_type = self.classify_query(query) if use_hybrid_search else "vector"
        result = {
            "entities": [],
            "relationships": [],
            "paths": [],
            "vector_results": [],
            "query_type": query_type,
        }

        if query_type in ("graph_entity", "graph_traversal", "graph_comparison", "hybrid"):
            if self.graph_service:
                try:
                    entities = await self.graph_service.search_entities(
                        query=query,
                        entity_types=entity_types,
                        limit=top_k,
                    )
                    result["entities"] = entities

                    if entities and query_type in ("graph_traversal", "graph_comparison", "hybrid"):
                        entity_ids = [e["id"] for e in entities[:3]]
                        all_entities = []
                        all_relationships = []
                        for eid in entity_ids:
                            ents, rels = await self.graph_service.get_entity_neighborhood(
                                entity_id=eid,
                                max_hops=2 if query_type == "graph_traversal" else 1,
                                max_nodes=30,
                            )
                            all_entities.extend(ents)
                            all_relationships.extend(rels)

                        seen_ids = set()
                        result["entities"] = [
                            e for e in all_entities
                            if e["id"] not in seen_ids and not seen_ids.add(e["id"])
                        ]

                        seen_rel_keys = set()
                        result["relationships"] = [
                            r for r in all_relationships
                            if r.get("id") not in seen_rel_keys
                            and not seen_rel_keys.add(r.get("id"))
                        ]

                        if query_type == "graph_traversal" and self.graph_service:
                            paths = await self.graph_service.multi_hop_query(
                                query=query, max_hops=3, top_k=5
                            )
                            result["paths"] = paths

                except Exception as e:
                    logger.warning(f"Graph retrieval failed, falling back to vector: {e}")
                    query_type = "vector"

        if query_type in ("vector", "hybrid") or not result["entities"]:
            if self.embedding_service and self.vector_store:
                try:
                    query_embedding = self.embedding_service.encode_query(query)
                    vector_results = self.vector_store.query(
                        query_embedding=query_embedding,
                        n_results=top_k,
                    )
                    result["vector_results"] = vector_results
                except Exception as e:
                    logger.warning(f"Vector retrieval failed: {e}")

        return result

    async def graph_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        top_k: int = 10,
    ) -> dict:
        """Pure graph search with multi-hop traversal."""
        result = {
            "entities": [],
            "relationships": [],
            "paths": [],
            "vector_results": [],
            "query_type": "graph",
        }

        if not self.graph_service:
            return result

        try:
            entities = await self.graph_service.search_entities(
                query=query,
                entity_types=entity_types,
                limit=top_k,
            )
            result["entities"] = entities

            if entities:
                entity_ids = [e["id"] for e in entities[:5]]
                all_entities = []
                all_relationships = []
                for eid in entity_ids:
                    ents, rels = await self.graph_service.get_entity_neighborhood(
                        entity_id=eid, max_hops=2, max_nodes=30
                    )
                    all_entities.extend(ents)
                    all_relationships.extend(rels)

                seen_ids = set()
                result["entities"] = [
                    e for e in all_entities
                    if e["id"] not in seen_ids and not seen_ids.add(e["id"])
                ]

                seen_rel_keys = set()
                result["relationships"] = [
                    r for r in all_relationships
                    if r.get("id") not in seen_rel_keys
                    and not seen_rel_keys.add(r.get("id"))
                ]

                paths = await self.graph_service.multi_hop_query(
                    query=query, max_hops=3, top_k=5
                )
                result["paths"] = paths

        except Exception as e:
            logger.error(f"Graph search failed: {e}")

        return result

    async def vector_fallback(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Pure vector search fallback."""
        if not self.embedding_service or not self.vector_store:
            return []
        try:
            query_embedding = self.embedding_service.encode_query(query)
            return self.vector_store.query(
                query_embedding=query_embedding,
                n_results=top_k,
            )
        except Exception as e:
            logger.warning(f"Vector fallback failed: {e}")
            return []
