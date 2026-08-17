"""Chat API Router — Streaming chat with RAG pipeline."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
async def chat(request: Request, body: ChatRequest):
    """Send a message and receive a streaming RAG-powered response.

    Returns Server-Sent Events (SSE) with:
    - metadata: conversation_id and sources
    - token: streamed response tokens
    - title: auto-generated conversation title
    - done: completion signal
    """
    chat_service = request.app.state.chat_service
    debug_mode = getattr(request.app.state, "rag_debug_mode", False)

    return StreamingResponse(
        chat_service.chat_stream(
            message=body.message,
            conversation_id=body.conversation_id,
            use_ragflow=body.use_ragflow,
            debug_mode=body.debug_mode if body.debug_mode is not None else debug_mode,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history")
async def get_conversations(request: Request):
    """Get all conversations."""
    chat_service = request.app.state.chat_service
    conversations = await chat_service.get_conversations()
    return {"conversations": conversations}


@router.get("/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str):
    """Get messages for a specific conversation."""
    chat_service = request.app.state.chat_service
    messages = await chat_service.get_conversation_messages(conversation_id)
    return {"messages": messages}


@router.delete("/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str):
    """Delete a conversation and its messages."""
    chat_service = request.app.state.chat_service
    await chat_service.delete_conversation(conversation_id)
    return {"status": "deleted"}
