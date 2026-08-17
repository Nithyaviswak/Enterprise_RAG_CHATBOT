"""Document Ingestion Router — Upload, process, and manage documents.

Uses the page-aware, dedup-aware ingestion pipeline in ``app.rag.ingestion``.
"""

import os
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Request, UploadFile, File, HTTPException, BackgroundTasks

from app.config import get_settings
from app.models.database import add_document, update_document_status, get_documents, delete_document
from app.rag.ingestion import (
    ALLOWED_EXTENSIONS,
    build_chunks,
    content_hash,
    is_allowed,
    is_duplicate_in_store,
    parse_document,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


async def _process_document(
    doc_id: str,
    file_path: str,
    file_type: str,
    filename: str,
    request_app,
):
    """Background task to process an uploaded document."""
    try:
        # 1. Parse text with page-level metadata
        parsed = parse_document(file_path, file_type, filename=filename)
        doc_hash = content_hash(parsed.full_text)

        embedding_service = request_app.state.embedding_service
        vector_store = request_app.state.vector_store

        # 2. Duplicate detection — skip re-indexing identical content
        if is_duplicate_in_store(vector_store.collection, doc_id, doc_hash):
            await update_document_status(doc_id, "ready", 0, duplicate=True)
            logger.info(f"Duplicate document skipped: '{filename}' (already indexed)")
            return

        # 3. Chunk with provenance metadata (source, page, chunk_id, document_id)
        chunks = build_chunks(parsed, doc_id)
        if not chunks:
            await update_document_status(doc_id, "error", 0)
            return

        # 4. Embed + store in ChromaDB
        texts = [c["content"] for c in chunks]
        embeddings = embedding_service.encode_documents(texts)
        ids = [c["id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        vector_store.add_documents(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        # 5. Try uploading to RAGFlow too (if available, non-critical)
        ragflow = request_app.state.ragflow_client
        if ragflow.is_available:
            try:
                datasets = await ragflow.list_datasets()
                if datasets:
                    dataset_id = datasets[0].get("id")
                    if dataset_id:
                        await ragflow.upload_document(dataset_id, file_path, filename)
            except Exception as e:
                logger.warning(f"RAGFlow upload failed (non-critical): {e}")

        # 6. Update status
        await update_document_status(doc_id, "ready", len(chunks))
        logger.info(f"Document '{filename}' processed: {len(chunks)} chunks indexed")

    except Exception as e:
        logger.error(f"Document processing error: {e}", exc_info=True)
        await update_document_status(doc_id, "error", 0)


@router.post("/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload and process a document (PDF, DOCX, TXT, MD, CSV)."""
    filename = file.filename or "document"
    ext = os.path.splitext(filename)[1].lower()
    if not is_allowed(filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    settings = get_settings()
    os.makedirs(settings.resolve_path(settings.upload_dir), exist_ok=True)

    file_id = uuid.uuid4().hex
    file_path = os.path.join(settings.resolve_path(settings.upload_dir), f"{file_id}{ext}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    doc = await add_document(
        filename=filename,
        file_type=ext,
        size_bytes=len(content),
        file_path=file_path,
    )

    background_tasks.add_task(
        _process_document,
        doc["id"],
        file_path,
        ext,
        filename,
        request.app,
    )

    return {
        "id": doc["id"],
        "filename": filename,
        "status": "processing",
        "message": "Document uploaded and being processed",
    }


@router.get("")
async def list_documents(request: Request):
    """List all uploaded documents."""
    documents = await get_documents()
    return {"documents": documents}


@router.delete("/{document_id}")
async def remove_document(request: Request, document_id: str):
    """Delete a document and its embeddings."""
    file_path = await delete_document(document_id)

    try:
        vector_store = request.app.state.vector_store
        vector_store.delete_by_metadata({"document_id": document_id})
    except Exception as e:
        logger.warning(f"Error removing embeddings: {e}")

    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    return {"status": "deleted"}