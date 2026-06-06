"""
Google Gemini API Service — LLM Integration.

Uses the google-genai SDK for streaming chat completions
with RAG context injection.
"""

import json
import logging
from typing import AsyncGenerator
from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)

# System prompt template with RAG context injection
RAG_SYSTEM_PROMPT = """You are an intelligent AI assistant powered by a Retrieval-Augmented Generation (RAG) system.

When context is provided, use it to give accurate, well-sourced answers. Always:
1. Base your responses on the provided context when available
2. Cite your sources by referencing document names or sections
3. If the context doesn't contain enough information, say so clearly
4. Be conversational, clear, and helpful
5. Format responses with markdown for readability
6. Use code blocks with language tags for code snippets

{context_section}"""

CONTEXT_TEMPLATE = """
## Retrieved Context

The following information was retrieved from the knowledge base to help answer the user's query:

{context}

---
Use the above context to inform your response. Cite sources where applicable."""


class GeminiService:
    """Service for Google Gemini API interactions."""

    def __init__(self):
        settings = get_settings()
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        logger.info(f"Gemini service initialized with model: {self.model}")

    def _build_system_prompt(self, context: list[dict] | None = None) -> str:
        """Build system prompt with optional RAG context."""
        if context and len(context) > 0:
            context_text = "\n\n".join(
                f"**[Source: {c.get('source', 'Unknown')}]** (Relevance: {c.get('score', 0):.2f})\n{c.get('content', '')}"
                for c in context
            )
            context_section = CONTEXT_TEMPLATE.format(context=context_text)
        else:
            context_section = "\nNo specific context was retrieved. Answer based on your general knowledge."

        return RAG_SYSTEM_PROMPT.format(context_section=context_section)

    async def chat_stream(
        self,
        message: str,
        history: list[dict] | None = None,
        context: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat response from Gemini with RAG context."""
        system_prompt = self._build_system_prompt(context)

        # Build conversation contents
        contents = []
        if history:
            for msg in history[-20:]:  # Keep last 20 messages for context window
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

        # Add current message
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        )

        try:
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=4096,
                ),
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            yield f"\n\n⚠️ Error communicating with Gemini API: {str(e)}"

    async def chat(
        self,
        message: str,
        history: list[dict] | None = None,
        context: list[dict] | None = None,
    ) -> str:
        """Non-streaming chat response."""
        full_response = ""
        async for chunk in self.chat_stream(message, history, context):
            full_response += chunk
        return full_response

    async def generate_title(self, message: str) -> str:
        """Generate a short title for a conversation based on the first message."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"Generate a concise title (max 6 words) for a conversation that starts with: '{message[:200]}'. Return ONLY the title, no quotes or extra formatting.",
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=30,
                ),
            )
            return response.text.strip().strip('"\'') if response.text else "New Chat"
        except Exception as e:
            logger.error(f"Title generation error: {e}")
            return message[:50] + "..." if len(message) > 50 else message
