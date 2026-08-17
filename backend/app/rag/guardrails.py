"""Guardrails applied between retrieval and generation.

Decides whether it is safe to generate an answer for the retrieved context:

- :attr:`EMPTY_CONTEXT` — nothing relevant was retrieved.
- :attr:`LOW_CONFIDENCE` — retrieval confidence is below the configured
  threshold; answering would risk hallucination.
- Excessive context is clamped (not refused) before prompt construction.

When a guardrail trips, the pipeline returns the configured refusal message
instead of fabricating an answer.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.rag.failures import FailureType

logger = logging.getLogger(__name__)


@dataclass
class GuardrailDecision:
    """Outcome of the guardrail evaluation."""

    should_refuse: bool
    failure_type: Optional[FailureType] = None
    reason: str = ""
    refusal_message: Optional[str] = None
    context_chunks: list[dict] | None = None


class Guardrails:
    """Applies retrieval guardrails to decide if generation may proceed."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    def evaluate(self, retrieval: "RetrievalResult", query: str) -> GuardrailDecision:
        """Evaluate guardrail conditions for a retrieval result."""
        chunks = retrieval.chunks
        confidence = retrieval.retrieval_confidence

        # 1. Empty context
        if not chunks:
            logger.warning("GUARDRAIL failure_type=EMPTY_CONTEXT reason=no_context")
            return GuardrailDecision(
                should_refuse=True,
                failure_type=FailureType.EMPTY_CONTEXT,
                reason="No relevant context retrieved",
                refusal_message=self.settings.refusal_message,
                context_chunks=[],
            )

        # 2. Low retrieval confidence (insufficient / weak evidence)
        if confidence < self.settings.low_confidence_threshold:
            logger.warning(
                "GUARDRAIL failure_type=LOW_CONFIDENCE confidence=%.2f reason=low_retrieval_score",
                confidence,
            )
            return GuardrailDecision(
                should_refuse=True,
                failure_type=FailureType.LOW_CONFIDENCE,
                reason=f"Retrieval confidence {confidence:.2f} below threshold "
                f"{self.settings.low_confidence_threshold:.2f}",
                refusal_message=self.settings.refusal_message,
                context_chunks=chunks,
            )

        # 3. Excessive query guardrail (defensive)
        if len(query) > 2000:
            return GuardrailDecision(
                should_refuse=True,
                failure_type=FailureType.INVALID_DOCUMENT,
                reason="Query exceeds maximum allowed length",
                refusal_message=self.settings.refusal_message,
                context_chunks=chunks,
            )

        # Everything else proceeds; context is clamped before generation.
        return GuardrailDecision(
            should_refuse=False,
            context_chunks=chunks,
        )