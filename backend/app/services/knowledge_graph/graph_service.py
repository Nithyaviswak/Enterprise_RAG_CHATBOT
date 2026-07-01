import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from app.config import get_settings

logger = logging.getLogger(__name__)


class GraphService:
    """Neo4j Knowledge Graph service for entity and relationship storage."""

    def __init__(self):
        settings = get_settings()
        self._driver: Optional[AsyncDriver] = None
        self._uri = settings.neo4j_uri
        self._user = settings.neo4j_user
        self._password = settings.neo4j_password
        self._database = settings.neo4j_database

    async def initialize(self):
        """Create the Neo4j driver and ensure indexes exist."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
            max_connection_lifetime=3600,
            connection_acquisition_timeout=60,
        )
        await self._verify_connectivity()
        await self._create_indexes()
        logger.info("Neo4j graph database initialized")

    async def close(self):
        if self._driver:
            await self._driver.close()

    async def _verify_connectivity(self):
        async with self._driver.session(database=self._database) as session:
            result = await session.run("RETURN 1 AS connected")
            record = await result.single()
            if record and record["connected"] == 1:
                logger.info("Neo4j connection verified")
            else:
                raise ConnectionError("Neo4j connectivity check failed")

    async def _create_indexes(self):
        async with self._driver.session(database=self._database) as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relation_type)",
            ]
            for cypher in constraints:
                try:
                    await session.run(cypher)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        aliases: list[str] | None = None,
        chunk_ids: list[str] | None = None,
        source_document: str = "",
        metadata: dict | None = None,
    ) -> dict:
        entity_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET
                    e.id = $entity_id,
                    e.entity_type = $entity_type,
                    e.description = $description,
                    e.aliases = $aliases,
                    e.chunk_ids = $chunk_ids,
                    e.source_document = $source_document,
                    e.metadata = $metadata,
                    e.created_at = $now,
                    e.updated_at = $now
                ON MATCH SET
                    e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
                    e.aliases = apoc.coll.union(e.aliases, $aliases),
                    e.chunk_ids = apoc.coll.union(e.chunk_ids, $chunk_ids),
                    e.updated_at = $now
                RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                       e.description AS description, e.aliases AS aliases,
                       e.chunk_ids AS chunk_ids, e.source_document AS source_document,
                       e.created_at AS created_at
                """,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                description=description,
                aliases=aliases or [],
                chunk_ids=chunk_ids or [],
                source_document=source_document,
                metadata=metadata or {},
                now=now,
            )
            record = await result.single()
            return dict(record) if record else {}

    async def create_relationship(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        description: str = "",
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> dict:
        rel_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (s:Entity {name: $source_name})
                MATCH (t:Entity {name: $target_name})
                MERGE (s)-[r:RELATES_TO {relation_type: $relation_type}]->(t)
                ON CREATE SET
                    r.id = $rel_id,
                    r.description = $description,
                    r.weight = $weight,
                    r.metadata = $metadata,
                    r.created_at = $now
                ON MATCH SET
                    r.weight = r.weight + $weight,
                    r.description = CASE WHEN $description <> '' THEN $description ELSE r.description END,
                    r.metadata = $metadata
                RETURN r.id AS id, r.relation_type AS relation_type,
                       r.description AS description, r.weight AS weight,
                       r.created_at AS created_at,
                       s.name AS source_name, t.name AS target_name
                """,
                source_name=source_name,
                target_name=target_name,
                relation_type=relation_type,
                rel_id=rel_id,
                description=description,
                weight=weight,
                metadata=metadata or {},
                now=now,
            )
            record = await result.single()
            return dict(record) if record else {}

    async def find_entity(self, name: str) -> dict | None:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (e:Entity {name: $name}) RETURN e.id AS id, e.name AS name, "
                "e.entity_type AS entity_type, e.description AS description, "
                "e.aliases AS aliases, e.chunk_ids AS chunk_ids, "
                "e.source_document AS source_document, e.created_at AS created_at",
                name=name,
            )
            record = await result.single()
            return dict(record) if record else None

    async def search_entities(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        async with self._driver.session(database=self._database) as session:
            if entity_types:
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE (e.name CONTAINS $query OR any(alias IN e.aliases WHERE alias CONTAINS $query))
                    AND e.entity_type IN $entity_types
                    RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                           e.description AS description, e.aliases AS aliases,
                           e.source_document AS source_document
                    ORDER BY size(e.name) ASC
                    LIMIT $limit
                    """,
                    query=query,
                    entity_types=entity_types,
                    limit=limit,
                )
            else:
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $query OR any(alias IN e.aliases WHERE alias CONTAINS $query)
                    RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                           e.description AS description, e.aliases AS aliases,
                           e.source_document AS source_document
                    ORDER BY size(e.name) ASC
                    LIMIT $limit
                    """,
                    query=query,
                    limit=limit,
                )
            return [dict(record) for record in await result.fetch(limit)]

    async def get_entity_neighborhood(
        self,
        entity_id: str,
        max_hops: int = 2,
        max_nodes: int = 50,
    ) -> tuple[list[dict], list[dict]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (center:Entity {id: $entity_id})
                CALL apoc.path.subgraph(center, {
                    maxLevel: $max_hops,
                    whitelistNodes: ['Entity'],
                    limit: $max_nodes
                }) YIELD node, relationship
                WITH COLLECT(DISTINCT node) AS nodes, COLLECT(DISTINCT relationship) AS rels
                RETURN
                    [n IN nodes | {
                        id: n.id, name: n.name, entity_type: n.entity_type,
                        description: n.description, aliases: n.aliases,
                        source_document: n.source_document
                    }] AS entities,
                    [r IN rels WHERE r IS NOT NULL | {
                        id: r.id, source_name: startNode(r).name, target_name: endNode(r).name,
                        relation_type: r.relation_type, description: r.description,
                        weight: r.weight
                    }] AS relationships
                """,
                entity_id=entity_id,
                max_hops=max_hops,
                max_nodes=max_nodes,
            )
            record = await result.single()
            if record:
                return record["entities"], record["relationships"]
            return [], []

    async def multi_hop_query(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 10,
    ) -> list[list[dict]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH path = (start:Entity)-[*1..$max_hops]-(end:Entity)
                WHERE start.name CONTAINS $query OR end.name CONTAINS $query
                WITH path, length(path) AS path_len
                ORDER BY path_len ASC
                LIMIT $top_k
                RETURN [node IN nodes(path) | {
                    id: node.id, name: node.name, entity_type: node.entity_type
                }] AS path_nodes,
                [rel IN relationships(path) | {
                    source_name: startNode(rel).name,
                    target_name: endNode(rel).name,
                    relation_type: rel.relation_type
                }] AS path_rels
                """,
                query=query,
                max_hops=max_hops,
                top_k=top_k,
            )
            paths = []
            async for record in result:
                path = []
                for i, node in enumerate(record["path_nodes"]):
                    path.append({**node, "type": "entity"})
                    if i < len(record["path_rels"]):
                        path.append({**record["path_rels"][i], "type": "relationship"})
                if path:
                    paths.append(path)
            return paths

    async def get_all_entities(
        self,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        async with self._driver.session(database=self._database) as session:
            count_result = await session.run(
                """
                MATCH (e:Entity)
                WHERE $entity_type IS NULL OR e.entity_type = $entity_type
                RETURN count(e) AS total
                """,
                entity_type=entity_type,
            )
            count_record = await count_result.single()
            total = count_record["total"] if count_record else 0

            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE $entity_type IS NULL OR e.entity_type = $entity_type
                RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                       e.description AS description, e.aliases AS aliases,
                       e.source_document AS source_document, e.created_at AS created_at
                ORDER BY e.created_at DESC
                SKIP $offset
                LIMIT $limit
                """,
                entity_type=entity_type,
                limit=limit,
                offset=offset,
            )
            entities = [dict(record) for record in await result.fetch(limit)]
            return entities, total

    async def get_stats(self) -> dict:
        async with self._driver.session(database=self._database) as session:
            node_count = await session.run("MATCH (e:Entity) RETURN count(e) AS count")
            node_record = await node_count.single()
            total_nodes = node_record["count"] if node_record else 0

            edge_count = await session.run(
                "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS count"
            )
            edge_record = await edge_count.single()
            total_edges = edge_record["count"] if edge_record else 0

            type_counts_result = await session.run(
                """
                MATCH (e:Entity)
                RETURN e.entity_type AS entity_type, count(e) AS count
                ORDER BY count DESC
                """
            )
            entity_type_counts = {
                record["entity_type"]: record["count"]
                async for record in type_counts_result
            }

            rel_counts_result = await session.run(
                """
                MATCH ()-[r:RELATES_TO]->()
                RETURN r.relation_type AS relation_type, count(r) AS count
                ORDER BY count DESC
                """
            )
            relation_type_counts = {
                record["relation_type"]: record["count"]
                async for record in rel_counts_result
            }

            doc_count_result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.source_document <> ''
                RETURN count(DISTINCT e.source_document) AS count
                """
            )
            doc_record = await doc_count_result.single()

            return {
                "node_count": total_nodes,
                "edge_count": total_edges,
                "entity_type_counts": entity_type_counts,
                "relation_type_counts": relation_type_counts,
                "documents_processed": doc_record["count"] if doc_record else 0,
            }

    async def delete_entity(self, entity_id: str) -> bool:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $entity_id})
                DETACH DELETE e
                RETURN count(e) AS deleted
                """,
                entity_id=entity_id,
            )
            record = await result.single()
            return record["deleted"] > 0 if record else False

    async def clear_graph(self):
        async with self._driver.session(database=self._database) as session:
            await session.run("MATCH (n) DETACH DELETE n")
            logger.info("Knowledge graph cleared")
