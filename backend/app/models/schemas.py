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
