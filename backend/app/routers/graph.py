import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query

from app.models.schemas import (
    EntityCreate,
    EntityResponse,
    RelationshipCreate,
    RelationshipResponse,
    GraphSearchRequest,
    GraphSearchResponse,
    GraphStats,
    GraphExploreRequest,
    GraphExploreResponse,
    EntityDetail,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/graph", tags=["knowledge_graph"])


def get_graph_service(request: Request):
    service = getattr(request.app.state, "graph_service", None)
    if not service:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    return service


def get_graph_retriever(request: Request):
    retriever = getattr(request.app.state, "graph_retriever", None)
    if not retriever:
        raise HTTPException(status_code=503, detail="Graph retriever not available")
    return retriever


@router.get("/stats", response_model=GraphStats)
async def get_graph_stats(request: Request):
    """Get knowledge graph statistics."""
    service = get_graph_service(request)
    return await service.get_stats()


@router.get("/entities", response_model=dict)
async def list_entities(
    request: Request,
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None, description="Search entities by name"),
):
    """List all entities in the knowledge graph."""
    service = get_graph_service(request)

    if search:
        entities = await service.search_entities(
            query=search,
            entity_types=[entity_type] if entity_type else None,
            limit=limit,
        )
        return {"entities": entities, "total": len(entities)}

    entities, total = await service.get_all_entities(
        entity_type=entity_type,
        limit=limit,
        offset=offset,
    )
    return {"entities": entities, "total": total}


@router.get("/entities/{entity_id}", response_model=EntityDetail)
async def get_entity_detail(request: Request, entity_id: str):
    """Get detailed information about a specific entity."""
    service = get_graph_service(request)

    entities, total = await service.get_all_entities(limit=1)
    entity = None
    for e in entities:
        if e["id"] == entity_id:
            entity = e
            break

    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    entities, relationships = await service.get_entity_neighborhood(
        entity_id=entity_id, max_hops=1, max_nodes=30
    )

    related = [e for e in entities if e["id"] != entity_id]
    contexts = entity.get("chunk_ids", [])

    return EntityDetail(
        entity=entity,
        relationships=relationships,
        related_entities=related,
        contexts=contexts,
    )


@router.post("/entities", response_model=dict)
async def create_entity(request: Request, body: EntityCreate):
    """Create a new entity in the knowledge graph."""
    service = get_graph_service(request)
    entity = await service.create_entity(
        name=body.name,
        entity_type=body.entity_type.value,
        description=body.description,
        aliases=body.aliases,
        metadata=body.metadata,
    )
    return {"entity": entity, "message": "Entity created successfully"}


@router.delete("/entities/{entity_id}")
async def delete_entity(request: Request, entity_id: str):
    """Delete an entity and its relationships."""
    service = get_graph_service(request)
    deleted = await service.delete_entity(entity_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Entity not found")
    return {"status": "deleted", "message": "Entity removed from knowledge graph"}


@router.get("/entities/{entity_id}/neighborhood", response_model=GraphExploreResponse)
async def explore_entity(
    request: Request,
    entity_id: str,
    max_hops: int = Query(2, ge=1, le=5),
    max_nodes: int = Query(50, ge=1, le=200),
):
    """Explore the neighborhood of an entity (for graph visualization)."""
    service = get_graph_service(request)

    entities, total = await service.get_all_entities(limit=1)
    central_entity = next((e for e in entities if e["id"] == entity_id), None)

    if not central_entity:
        entities_list, _ = await service.get_all_entities(limit=500)
        central_entity = next((e for e in entities_list if e["id"] == entity_id), None)

    if not central_entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    neighborhood_entities, relationships = await service.get_entity_neighborhood(
        entity_id=entity_id,
        max_hops=max_hops,
        max_nodes=max_nodes,
    )

    nodes = []
    for e in neighborhood_entities:
        nodes.append({
            "id": e["id"],
            "label": e["name"],
            "type": e["entity_type"],
            "description": e.get("description", ""),
        })

    edges = []
    for r in relationships:
        edges.append({
            "id": r.get("id", ""),
            "source": r.get("source_name", ""),
            "target": r.get("target_name", ""),
            "label": r.get("relation_type", ""),
            "weight": r.get("weight", 1.0),
        })

    return GraphExploreResponse(
        nodes=nodes,
        edges=edges,
        central_entity=central_entity,
    )


@router.get("/relationships", response_model=dict)
async def list_relationships(
    request: Request,
    relation_type: Optional[str] = Query(None, description="Filter by relation type"),
    limit: int = Query(100, ge=1, le=500),
):
    """List relationships in the knowledge graph."""
    service = get_graph_service(request)

    stats = await service.get_stats()
    all_types = stats.get("relation_type_counts", {})

    return {"relationships": [], "relation_types": all_types}


@router.post("/relationships", response_model=dict)
async def create_relationship(request: Request, body: RelationshipCreate):
    """Create a new relationship between entities."""
    service = get_graph_service(request)
    relationship = await service.create_relationship(
        source_name=body.source_id,
        target_name=body.target_id,
        relation_type=body.relation_type,
        description=body.description,
        weight=body.weight,
        metadata=body.metadata,
    )
    return {"relationship": relationship, "message": "Relationship created successfully"}


@router.post("/search", response_model=GraphSearchResponse)
async def search_graph(request: Request, body: GraphSearchRequest):
    """Search the knowledge graph with automatic strategy selection.

    Determines whether to use graph search, vector search, or hybrid
    based on the query content.
    """
    retriever = get_graph_retriever(request)

    entity_type_values = None
    if body.entity_types:
        entity_type_values = [et.value for et in body.entity_types]

    result = await retriever.retrieve(
        query=body.query,
        top_k=body.top_k,
        entity_types=entity_type_values,
        use_hybrid_search=body.use_hybrid_search,
    )

    return GraphSearchResponse(
        entities=result.get("entities", []),
        relationships=result.get("relationships", []),
        paths=result.get("paths", []),
        vector_results=result.get("vector_results", []),
        query_type=result.get("query_type", "vector"),
    )


@router.post("/traverse")
async def traverse_graph(
    request: Request,
    body: GraphExploreRequest,
):
    """Multi-hop graph traversal from a starting entity."""
    service = get_graph_service(request)

    entities, relationships = await service.get_entity_neighborhood(
        entity_id=body.entity_id,
        max_hops=body.max_hops,
        max_nodes=body.max_nodes,
    )

    return {
        "entities": entities,
        "relationships": relationships,
    }


@router.get("/paths")
async def find_paths(
    request: Request,
    source: str = Query(..., description="Source entity name"),
    target: str = Query(..., description="Target entity name"),
    max_hops: int = Query(4, ge=1, le=6),
):
    """Find paths between two entities."""
    service = get_graph_service(request)

    source_entity = await service.find_entity(source)
    target_entity = await service.find_entity(target)

    if not source_entity:
        raise HTTPException(status_code=404, detail=f"Source entity '{source}' not found")
    if not target_entity:
        raise HTTPException(status_code=404, detail=f"Target entity '{target}' not found")

    paths = await service.multi_hop_query(
        query=source,
        max_hops=max_hops,
        top_k=5,
    )

    return {
        "source": source_entity,
        "target": target_entity,
        "paths": paths,
    }


@router.delete("/clear")
async def clear_graph(request: Request):
    """Clear all entities and relationships from the knowledge graph."""
    service = get_graph_service(request)
    await service.clear_graph()
    return {"status": "cleared", "message": "Knowledge graph cleared"}


@router.post("/index-document")
async def index_document_to_graph(request: Request, document_id: str = Query(...)):
    """Extract entities and relationships from a document and add to graph.

    This endpoint triggers the full pipeline: entity extraction,
    relationship extraction, and graph population for a given document.
    """
    entity_extractor = getattr(request.app.state, "entity_extractor", None)
    relationship_extractor = getattr(request.app.state, "relationship_extractor", None)
    graph_service = get_graph_service(request)

    if not entity_extractor or not relationship_extractor:
        raise HTTPException(status_code=503, detail="Graph extraction services not available")

    vector_store = getattr(request.app.state, "vector_store", None)
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not available")

    try:
        all_docs = vector_store.collection.get(
            where={"document_id": document_id},
            limit=200,
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Document not found in vector store")

    if not all_docs or not all_docs.get("documents"):
        raise HTTPException(status_code=404, detail="No chunks found for this document")

    chunks = all_docs["documents"]
    metadatas = all_docs.get("metadatas", [{}] * len(chunks))
    source = metadatas[0].get("source", "Unknown") if metadatas else "Unknown"

    all_entities = []
    all_relationships = []

    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{document_id}_{i}"
        entities = await entity_extractor.extract(
            text=chunk_text,
            chunk_id=chunk_id,
            source_document=source,
        )
        all_entities.extend(entities)

    seen_names = set()
    unique_entities = []
    for e in all_entities:
        if e["name"].lower() not in seen_names:
            seen_names.add(e["name"].lower())
            unique_entities.append(e)

    for entity in unique_entities:
        try:
            await graph_service.create_entity(
                name=entity["name"],
                entity_type=entity["entity_type"],
                description=entity.get("description", ""),
                aliases=entity.get("aliases", []),
                chunk_ids=entity.get("chunk_ids", []),
                source_document=entity.get("source_document", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to create entity '{entity['name']}': {e}")

    combined_text = "\n".join(chunks[:10])
    relationships_batch = await relationship_extractor.extract(
        text=combined_text,
        entities=unique_entities,
        source_document=source,
    )
    all_relationships.extend(relationships_batch)

    for rel in all_relationships:
        try:
            await graph_service.create_relationship(
                source_name=rel["source_name"],
                target_name=rel["target_name"],
                relation_type=rel["relation_type"],
                description=rel.get("description", ""),
                weight=rel.get("weight", 0.5),
            )
        except Exception as e:
            logger.warning(f"Failed to create relationship: {e}")

    return {
        "message": f"Document indexed to knowledge graph",
        "entities_created": len(unique_entities),
        "relationships_created": len(all_relationships),
        "source": source,
    }
