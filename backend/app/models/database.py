"""SQLite database for chat history using aiosqlite."""

import aiosqlite
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings

DB_PATH = get_settings().resolve_path("data/chat_history.db")


async def init_db():
    """Initialize database tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploading',
                chunks_count INTEGER DEFAULT 0,
                file_path TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await db.commit()


async def create_conversation(title: str = "New Chat") -> dict:
    """Create a new conversation."""
    conv_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        await db.commit()
    return {"id": conv_id, "title": title, "created_at": now, "updated_at": now}


async def update_conversation_title(conv_id: str, title: str):
    """Update conversation title."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conv_id),
        )
        await db.commit()


async def get_conversations() -> list[dict]:
    """Get all conversations with message counts."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_conversation(conv_id: str) -> Optional[dict]:
    """Get a single conversation."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def delete_conversation(conv_id: str):
    """Delete a conversation and its messages."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        await db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        await db.commit()


async def add_message(conversation_id: str, role: str, content: str, sources: str = "[]") -> dict:
    """Add a message to a conversation."""
    msg_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, sources, now),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await db.commit()
    return {"id": msg_id, "role": role, "content": content, "sources": sources, "created_at": now}


async def get_messages(conversation_id: str) -> list[dict]:
    """Get all messages for a conversation."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


# ── Document DB operations ────────────────────────────────────

async def add_document(filename: str, file_type: str, size_bytes: int, file_path: str) -> dict:
    """Add a document record."""
    doc_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO documents (id, filename, file_type, size_bytes, status, file_path, created_at) VALUES (?, ?, ?, ?, 'processing', ?, ?)",
            (doc_id, filename, file_type, size_bytes, file_path, now),
        )
        await db.commit()
    return {"id": doc_id, "filename": filename, "file_type": file_type, "size_bytes": size_bytes, "status": "processing", "created_at": now}


async def update_document_status(doc_id: str, status: str, chunks_count: int = 0, duplicate: bool = False):
    """Update document processing status.

    Args:
        doc_id: Document id.
        status: New status (uploading/processing/ready/error).
        chunks_count: Number of indexed chunks.
        duplicate: Mark the document as a duplicate (skipped re-indexing).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        if duplicate:
            await db.execute("UPDATE documents SET status='ready', chunks_count=? WHERE id=?", (0, doc_id))
        else:
            await db.execute(
                "UPDATE documents SET status = ?, chunks_count = ? WHERE id = ?",
                (status, chunks_count, doc_id),
            )
        await db.commit()


async def get_documents() -> list[dict]:
    """Get all documents."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def delete_document(doc_id: str) -> Optional[str]:
    """Delete a document record, returns file_path for cleanup."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT file_path FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()
        if row:
            await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await db.commit()
            return row["file_path"]
        return None
