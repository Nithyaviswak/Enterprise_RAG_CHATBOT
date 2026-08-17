"""Prompt construction and grounded generation.

Builds an explicit, grounded system prompt from the final context — with
numbered sources and page numbers — that instructs the LLM to cite evidence and
refuse when the context is insufficient. Returns the answer together with
metadata (prompt, latency, model) for observability and debugging.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

GROUNDED_INSTRUCTIONS = """You are a factual assistant. Follow these rules strictly:
1. Answer ONLY using the retrieved context below. Do not use general knowledge to add facts that are absent from the context.
2. Cite your sources inline with [n] after each claim, where n is the source number.
3. If the retrieved context does not contain enough information to answer reliably, respond with: "I don't have enough information in the provided documents to answer this reliably."
4. Do not invent citations, page numbers, or facts.
5. Be concise, precise, and helpful. Format with markdown when helpful."""


@dataclass
class GenerationResult:
    """Result of a generation call including trace metadata."""

    answer: str
    system_prompt: str = ""
    latency_ms: float = 0.0
    model: str = ""
    generation_failure: bool = False
    failure_reason: str = ""


def build_context_block(contexts: list[dict]) -> str:
    """Format final context chunks with numbered sources and page info."""
    lines = []
    for i, c in enumerate(contexts, start=1):
        source = c.get("source", "Unknown")
        page = c.get("metadata", {}).get("page")
        score = c.get("score", 0.0)
        page_txt = f", Page {page}" if page else ""
        lines.append(
            f"[{i}] Source: {source}{page_txt} (relevance {score:.2f})\n{c.get('content', '')}"
        )
    return "\n\n---\n\n".join(lines)


def build_system_prompt(contexts: list[dict]) -> str:
    """Construct the grounded system prompt for a set of context chunks."""
    if not contexts:
        return GROUNDED_INSTRUCTIONS + "\n\nNo context was retrieved."
    context_block = build_context_block(contexts)
    return (
        GROUNDED_INSTRUCTIONS
        + "\n\n## Retrieved Context\n\n"
        + context_block
    )


class PromptBuilder:
    """Encapsulates prompt construction for RAG generation (testable)."""

    def __init__(self):
        self.settings = get_settings()

    def build(self, query: str, contexts: list[dict]) -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)``."""
        return build_system_prompt(contexts), query

    def build_grounded_sources(self, contexts: list[dict]) -> list[dict]:
        """Return a source list for attribution (filename + page)."""
        return [
            {
                "id": i + 1,
                "source": c.get("source", "Unknown"),
                "page": c.get("metadata", {}).get("page"),
                "score": round(c.get("score", 0.0), 3),
                "content_excerpt": (c.get("content", "") or "")[:160],
            }
            for i, c in enumerate(contexts)
        ]


class Generator:
    """Generates grounded answers via the configured LLM service."""

    def __init__(self, llm_service, prompt_builder: Optional[PromptBuilder] = None):
        self.llm_service = llm_service
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.settings = get_settings()
        self._model = getattr(llm_service, "model", self.settings.gemini_model)

    async def generate(
        self,
        query: str,
        contexts: list[dict],
        history: Optional[list[dict]] = None,
    ) -> GenerationResult:
        """Generate an answer for ``query`` grounded in ``contexts``."""
        system_prompt, user_prompt = self.prompt_builder.build(query, contexts)

        start = time.perf_counter()
        full_response: list[str] = []
        generation_error = ""
        try:
            async for token in self.llm_service.chat_stream(
                message=user_prompt,
                history=history,
                context=contexts,
                system_prompt=system_prompt,
            ):
                # The service emits a prefixed error marker on failure.
                if token.startswith("\n\n⚠️ Error"):
                    generation_error = token
                else:
                    full_response.append(token)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Generation failed: %s", e)
            generation_error = str(e)

        latency = (time.perf_counter() - start) * 1000
        answer = "".join(full_response).strip()
        failed = bool(generation_error) or not answer
        return GenerationResult(
            answer=answer,
            system_prompt=system_prompt,
            latency_ms=round(latency, 2),
            model=self._model,
            generation_failure=failed,
            failure_reason=generation_error or ("empty response" if not answer else ""),
        )