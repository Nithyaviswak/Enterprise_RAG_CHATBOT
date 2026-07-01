"""Pydantic models for request/response schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Chat Schemas ──────────────────────────────────────────────

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    use_ragflow: bool = True


class ChatMessage(BaseModel):
    id: str
    role: MessageRole
    content: str
    sources: list[dict] = []
    created_at: datetime


class Conversation(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationList(BaseModel):
    conversations: list[Conversation]


# ── Document Schemas ──────────────────────────────────────────

class DocumentStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    chunks_count: int = 0
    message: str = ""


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: DocumentStatus
    chunks_count: int
    created_at: datetime


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


# ── Fine-tuning Schemas ──────────────────────────────────────

class FinetuneRequest(BaseModel):
    base_model: str = "BAAI/bge-small-en-v1.5"
    epochs: int = Field(default=2, ge=1, le=20)
    batch_size: int = Field(default=10, ge=1, le=64)


class FinetuneStatus(BaseModel):
    status: str
    progress: float = 0.0
    message: str = ""
    model_path: Optional[str] = None


# ── Health Check ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    services: dict[str, str] = {}


# ── Retrieval Schemas ─────────────────────────────────────────

class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float
    metadata: dict = {}


# ── Knowledge Graph Schemas ───────────────────────────────────

class EntityType(str, Enum):
    ORGANIZATION = "Organization"
    PERSON = "Person"
    PRODUCT = "Product"
    LOCATION = "Location"
    EVENT = "Event"
    DOCUMENT = "Document"
    CONCEPT = "Concept"
    DATE = "Date"
    TECHNOLOGY = "Technology"


class EntityBase(BaseModel):
    name: str
    entity_type: EntityType
    description: str = ""
    aliases: list[str] = []
    metadata: dict = {}


class EntityCreate(EntityBase):
    pass


class EntityResponse(EntityBase):
    id: str
    chunk_ids: list[str] = []
    created_at: str = ""
    source_document: str = ""


class RelationshipBase(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    metadata: dict = {}


class RelationshipCreate(RelationshipBase):
    pass


class RelationshipResponse(RelationshipBase):
    id: str
    created_at: str = ""


class GraphSearchRequest(BaseModel):
    query: str
    entity_types: list[EntityType] | None = None
    max_hops: int = 2
    top_k: int = 10
    use_hybrid_search: bool = True


class GraphSearchResponse(BaseModel):
    entities: list[EntityResponse] = []
    relationships: list[RelationshipResponse] = []
    paths: list[list[dict]] = []
    vector_results: list[RetrievedChunk] = []
    query_type: str = "graph"


class GraphStats(BaseModel):
    node_count: int
    edge_count: int
    entity_type_counts: dict[str, int]
    relation_type_counts: dict[str, int]
    documents_processed: int


class EntityDetail(BaseModel):
    entity: EntityResponse
    relationships: list[RelationshipResponse]
    related_entities: list[EntityResponse]
    contexts: list[str] = []


class GraphExploreRequest(BaseModel):
    entity_id: str
    max_hops: int = 2
    max_nodes: int = 50


class GraphExploreResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    central_entity: EntityResponse
