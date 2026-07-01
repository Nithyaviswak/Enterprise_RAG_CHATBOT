"""
Chat Service — RAG Pipeline Orchestration.

Orchestrates the complete RAG flow:
1. Receive user query
2. Retrieve relevant context
3. Inject context into Gemini prompt
4. Stream response back
5. Store conversation history
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
from app.services.gemini_service import GeminiService
from app.services.retrieval_service import RetrievalService
from app.services.knowledge_graph.graph_retriever import GraphRetriever

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates the RAG chatbot pipeline."""

    def __init__(
        self,
        gemini_service: GeminiService,
        retrieval_service: RetrievalService,
        graph_retriever: GraphRetriever | None = None,
    ):
        self.gemini = gemini_service
        self.retrieval = retrieval_service
        self.graph_retriever = graph_retriever

    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_ragflow: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Process a chat message through the full RAG pipeline with streaming.

        Yields SSE-formatted events:
        - data: {"type": "metadata", "conversation_id": "...", "sources": [...]}
        - data: {"type": "token", "content": "..."}
        - data: {"type": "done"}
        - data: {"type": "error", "message": "..."}
        """
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

            # 3. Retrieve relevant context (vector + graph in parallel)
            context = []
            graph_entities = []
            graph_relationships = []

            async def retrieve_vector():
                try:
                    return await self.retrieval.retrieve(
                        query=message,
                        top_k=5,
                        use_ragflow=use_ragflow,
                    )
                except Exception as e:
                    logger.warning(f"Vector retrieval failed: {e}")
                    return []

            async def retrieve_graph():
                if not self.graph_retriever:
                    return [], [], []
                try:
                    result = await self.graph_retriever.retrieve(
                        query=message,
                        top_k=5,
                        use_hybrid_search=True,
                    )
                    return (
                        result.get("entities", []),
                        result.get("relationships", []),
                        result.get("paths", []),
                    )
                except Exception as e:
                    logger.warning(f"Graph retrieval failed: {e}")
                    return [], [], []

            import asyncio
            vector_task = asyncio.create_task(retrieve_vector())
            graph_task = asyncio.create_task(retrieve_graph())
            context = await vector_task
            graph_entities, graph_relationships, graph_paths = await graph_task

            # 4. Build enriched context with graph data
            graph_context_str = ""
            if graph_entities:
                entity_lines = [
                    f"- {e['name']} ({e.get('entity_type', 'entity')})"
                    for e in graph_entities[:10]
                ]
                graph_context_str = "\nKnown entities in context:\n" + "\n".join(entity_lines)

            if graph_relationships:
                rel_lines = [
                    f"- {r.get('source_name', '?')} --[{r.get('relation_type', 'related_to')}]--> {r.get('target_name', '?')}"
                    for r in graph_relationships[:10]
                ]
                graph_context_str += "\nRelationships:\n" + "\n".join(rel_lines)

            if graph_context_str and context:
                context[-1]["content"] += f"\n\n{graph_context_str}"

            sources_summary = [
                {"source": c.get("source", ""), "score": round(c.get("score", 0), 3)}
                for c in context
            ]

            yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conversation_id, 'sources': sources_summary})}\n\n"

            # 5. Get conversation history for context
            history = await get_messages(conversation_id)
            history_formatted = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history[:-1]  # Exclude the message we just added
            ]

            # 6. Stream Gemini response
            full_response = ""
            async for token in self.gemini.chat_stream(
                message=message,
                history=history_formatted,
                context=context,
            ):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # 7. Save assistant response
            await add_message(
                conversation_id,
                "assistant",
                full_response,
                json.dumps(sources_summary),
            )

            # 8. Generate title for new conversations
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
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    async def get_conversations(self) -> list[dict]:
        """Get all conversations."""
        return await get_conversations()

    async def get_conversation_messages(self, conversation_id: str) -> list[dict]:
        """Get all messages for a conversation."""
        return await get_messages(conversation_id)

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation."""
        await delete_conversation(conversation_id)
