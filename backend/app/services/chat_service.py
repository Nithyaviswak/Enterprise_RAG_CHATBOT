"""
Chat Service — RAG Pipeline Orchestration (production).

Routes every message through the production RAG pipeline and streams the
result back over SSE. Each request is traced (request_id, stage latencies,
failure categories) and recorded in the metrics store.

SSE events:
- data: {"type": "metadata", "conversation_id", "sources", "confidence",
          "request_id", "latency_ms", "failure_type", "refused"}
- data: {"type": "token", "content": "..."}
- data: {"type": "title", "title": "..."}
- data: {"type": "debug", ...}     (only when debug mode is enabled)
- data: {"type": "done"}
- data: {"type": "error", "message": "..."}
"""

import json
import logging
from typing import AsyncGenerator, Optional

from app.models.database import (
    create_conversation,
    update_conversation_title,
    get_conversations,
    get_conversation,
    delete_conversation,
    add_message,
    get_messages,
)
from app.observability.tracing import Tracer
from app.rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)

_TOKEN_CHUNK_SIZE = 80


class ChatService:
    """Orchestrates the RAG chatbot pipeline with tracing and guardrails."""

    def __init__(
        self,
        gemini_service,
        retrieval_service,
        graph_retriever=None,
        rag_pipeline: Optional[RagPipeline] = None,
        metrics_store=None,
        settings=None,
    ):
        self.gemini = gemini_service
        self.retrieval = retrieval_service
        self.graph_retriever = graph_retriever
        self.rag_pipeline = rag_pipeline
        self.metrics_store = metrics_store
        from app.config import get_settings

        self.settings = get_settings()

    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_ragflow: bool = True,
        debug_mode: Optional[bool] = None,
    ) -> AsyncGenerator[str, None]:
        """Process a chat message through the full RAG pipeline with streaming."""
        tracer = Tracer.new(message)
        request_id = tracer.request_id
        debug_mode = self.settings.rag_debug_mode if debug_mode is None else debug_mode

        try:
            # 1. Create or get conversation
            if not conversation_id:
                conv = await create_conversation()
                conversation_id = conv["id"]
                is_new = True
            else:
                conv = await get_conversation(conversation_id)
                if not conv:
                    conv = await create_conversation()
                    conversation_id = conv["id"]
                is_new = True if not conv else False

            # 2. Save user message
            await add_message(conversation_id, "user", message)

            # 3. Run the production RAG pipeline (retrieval → guardrails → generation)
            history = await get_messages(conversation_id)
            history_formatted = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history[:-1]  # exclude the message we just added
            ]

            result = await self.rag_pipeline.run(
                query=message,
                history=history_formatted,
                use_ragflow=use_ragflow,
                debug=debug_mode,
                tracer=tracer,
            )

            # 4. Record metrics for observability / dashboard
            self._record_metrics(result, request_id, tracer)

            # 5. Emit metadata with sources + confidence + latency
            sources = [
                {
                    "source": s.get("source", "Unknown"),
                    "page": s.get("page"),
                    "score": s.get("score", 0.0),
                    "id": s.get("id"),
                }
                for s in result.sources
            ]
            yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conversation_id, 'sources': sources, 'confidence': result.confidence, 'request_id': request_id, 'latency_ms': result.trace.get('total_latency_ms'), 'failure_type': result.failure_type, 'refused': result.refused, 'answered': result.answered})}\n\n"

            # 6. Debug payload for developer mode (never for normal users)
            if debug_mode and result.debug:
                yield f"data: {json.dumps({'type': 'debug', 'debug': result.debug})}\n\n"

            # 7. Stream the answer (chunked so the UI keeps its streaming feel)
            answer = result.answer
            for i in range(0, len(answer), _TOKEN_CHUNK_SIZE):
                yield f"data: {json.dumps({'type': 'token', 'content': answer[i:i + _TOKEN_CHUNK_SIZE]})}\n\n"

            # 8. Save assistant response with source attribution
            await add_message(
                conversation_id,
                "assistant",
                answer,
                json.dumps(sources),
            )

            # 9. Generate title for new conversations
            if is_new:
                try:
                    title = await self.gemini.generate_title(message)
                    await update_conversation_title(conversation_id, title)
                    yield f"data: {json.dumps({'type': 'title', 'title': title})}\n\n"
                except Exception as e:
                    logger.warning(f"Title generation failed: {e}")

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Chat pipeline error: {e}", exc_info=True)
            tracer.set_failure("RETRIEVAL_FAILURE", str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    def _record_metrics(self, result, request_id: str, tracer: Tracer):
        """Record a safe metrics entry for the dashboard."""
        if not self.metrics_store:
            return
        from datetime import datetime, timezone

        entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieval_latency_ms": result.trace.get("stages", {}).get("retrieval", {}).get("latency_ms"),
            "generation_latency_ms": result.trace.get("stages", {}).get("generation", {}).get("latency_ms"),
            "total_latency_ms": result.trace.get("total_latency_ms"),
            "chunks": len(result.sources),
            "retrieval_confidence": result.confidence.get("retrieval_confidence"),
            "model_used": self.settings.gemini_model,
            "failure_type": result.failure_type,
            "answered": result.answered,
            "grounded_ratio": result.hallucination.get("grounded_ratio"),
            "refused": result.refused,
        }
        self.metrics_store.record(entry)

    async def get_conversations(self) -> list[dict]:
        """Get all conversations."""
        return await get_conversations()

    async def get_conversation_messages(self, conversation_id: str) -> list[dict]:
        """Get all messages for a conversation."""
        return await get_messages(conversation_id)

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation."""
        await delete_conversation(conversation_id)