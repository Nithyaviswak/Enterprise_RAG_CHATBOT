"""Document Ingestion Router — Upload, process, and manage documents."""

import os
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Request, UploadFile, File, HTTPException, BackgroundTasks
from pypdf import PdfReader
from docx import Document as DocxDocument

from app.config import get_settings
from app.models.database import add_document, update_document_status, get_documents, delete_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def _extract_text(file_path: str, file_type: str) -> str:
    """Extract text content from a file."""
    if file_type == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif file_type == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    elif file_type in {".txt", ".md", ".csv"}:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks


async def _process_document(
    doc_id: str,
    file_path: str,
    file_type: str,
    filename: str,
    request_app,
):
    """Background task to process an uploaded document."""
    try:
        # 1. Extract text
        text = _extract_text(file_path, file_type)
        if not text.strip():
            await update_document_status(doc_id, "error", 0)
            return

        # 2. Chunk text
        chunks = _chunk_text(text)
        if not chunks:
            await update_document_status(doc_id, "error", 0)
            return

        # 3. Generate embeddings using sentence-transformers
        embedding_service = request_app.state.embedding_service
        embeddings = embedding_service.encode_documents(chunks)

        # 4. Store in ChromaDB
        vector_store = request_app.state.vector_store
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": filename,
                "document_id": doc_id,
                "chunk_index": i,
                "file_type": file_type,
            }
            for i in range(len(chunks))
        ]

        vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids,
        )

        # 5. Try uploading to RAGFlow too (if available)
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
    """Upload and process a document (PDF, DOCX, TXT)."""
    # Validate file extension
    filename = file.filename or "document"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save file to disk
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)

    file_id = uuid.uuid4().hex
    file_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create database record
    doc = await add_document(
        filename=filename,
        file_type=ext,
        size_bytes=len(content),
        file_path=file_path,
    )

    # Process in background
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

    # Remove from vector store
    try:
        vector_store = request.app.state.vector_store
        vector_store.delete_by_metadata({"document_id": document_id})
    except Exception as e:
        logger.warning(f"Error removing embeddings: {e}")

    # Remove file from disk
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    return {"status": "deleted"}
