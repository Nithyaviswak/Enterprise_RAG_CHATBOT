"""End-to-end RAG pipeline orchestrator.

    query → hybrid retrieval → guardrails → grounded generation
          → hallucination check → confidence → source attribution

Every run produces a :class:`RagResult` carrying the answer, sources,
confidence, failure category, and a trace (plus full internals when debug mode
is enabled). Nothing sensitive is logged.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings
from app.observability.tracing import Tracer
from app.rag.failures import FailureType
from app.rag.generator import Generator, PromptBuilder
from app.rag.guardrails import Guardrails
from app.rag.hallucination import HallucinationDetector, is_refusal_answer
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class RagResult:
    """Complete outcome of one RAG pipeline run."""

    answer: str
    sources: list[dict]
    confidence: dict
    failure_type: Optional[str]
    failure_reason: str
    refused: bool
    hallucination: dict
    trace: dict
    answered: bool
    contexts: list[dict] = field(default_factory=list)
    debug: Optional[dict] = None


class RagPipeline:
    """Orchestrates retrieval → guardrails → generation → verification."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        guardrails: Optional[Guardrails] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.guardrails = guardrails or Guardrails()
        self.hallucination = hallucination_detector or HallucinationDetector()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.settings = get_settings()

    async def run(
        self,
        query: str,
        history: Optional[list[dict]] = None,
        top_k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
        use_ragflow: bool = True,
        debug: bool = False,
        tracer: Optional[Tracer] = None,
    ) -> RagResult:
        """Run the full RAG pipeline for a query."""
        tracer = tracer or Tracer.new(query)
        contexts: list[dict] = []
        failure_type: Optional[str] = None
        failure_reason = ""
        refused = False

        # ── 1. Retrieval ───────────────────────────────────────────
        stage = tracer.begin("retrieval")
        try:
            retrieval = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                where=metadata_filter,
                use_ragflow=use_ragflow,
            )
            tracer.end(
                stage,
                chunks=len(retrieval.chunks),
                scores=[c.get("score", 0.0) for c in retrieval.chunks],
            )
        except Exception as e:
            tracer.end(stage, status="error")
            tracer.set_failure(FailureType.RETRIEVAL_FAILURE.value, str(e))
            return self._refusal(
                query, failure_type=FailureType.RETRIEVAL_FAILURE.value,
                reason=str(e), tracer=tracer,
            )

        # ── 2. Guardrails ──────────────────────────────────────────
        decision = self.guardrails.evaluate(retrieval, query)
        if decision.should_refuse:
            tracer.set_failure(decision.failure_type.value, decision.reason)
            refused = True
            confidence = self.hallucination.combined_confidence(
                retrieval.retrieval_confidence,
                self.hallucination.analyze("", []),
                answered=False,
            )
            confidence["retrieval_confidence"] = retrieval.retrieval_confidence
            result = RagResult(
                answer=decision.refusal_message or self.settings.refusal_message,
                sources=[],
                confidence=confidence,
                failure_type=decision.failure_type.value,
                failure_reason=decision.reason,
                refused=True,
                hallucination={"risk_level": "high", "grounded_ratio": 0.0},
                trace=tracer.summary(),
                answered=False,
                contexts=[],
                debug=self._debug_payload(query, [], None, None, confidence, decision.failure_type.value, tracer) if debug else None,
            )
            return result

        # ── 3. Final context (clamped to budget) ───────────────────
        contexts = self.retriever.get_context(retrieval.chunks)

        # ── 4. Generation ──────────────────────────────────────────
        stage = tracer.begin("generation")
        generation = await self.generator.generate(query, contexts, history)
        tracer.end(
            stage,
            status="error" if generation.generation_failure else "ok",
            failure_type=generation.failure_reason or "",
        )

        answer = generation.answer
        answered = bool(answer) and not generation.generation_failure
        if generation.generation_failure:
            failure_type = FailureType.LLM_FAILURE.value
            failure_reason = generation.failure_reason
            tracer.set_failure(failure_type, failure_reason)
            answer = self.settings.refusal_message
            refused = True

        # ── 5. Hallucination detection ─────────────────────────────
        report = self.hallucination.analyze(answer, contexts)
        if not refused and is_refusal_answer(answer):
            refused = True

        confidence = self.hallucination.combined_confidence(
            retrieval.retrieval_confidence, report, answered=answered and not refused
        )

        # Flag high-risk answers even when we still surface them.
        if answered and not refused and report.risk_level == "high" and not is_refusal_answer(answer):
            failure_type = FailureType.HALLUCINATION_RISK.value
            failure_reason = "; ".join(report.risk_factors)
            tracer.set_failure(failure_type, failure_reason)

        # ── 6. Source attribution ──────────────────────────────────
        sources = self.prompt_builder.build_grounded_sources(contexts) if contexts else []
        if sources:
            tracer.info("sources", count=len(sources), names=[s["source"] for s in sources])

        trace = tracer.summary()
        debug_payload = self._debug_payload(
            query, contexts, retrieval, generation.system_prompt, confidence, failure_type, tracer
        ) if debug else None

        return RagResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            failure_type=failure_type,
            failure_reason=failure_reason,
            refused=refused,
            hallucination={
                "risk_level": report.risk_level,
                "grounded_ratio": report.grounded_ratio,
                "unsupported_claims": report.unsupported_claims[:5],
                "claims_checked": len(report.claims),
            },
            trace=trace,
            answered=answered and not refused,
            contexts=contexts,
            debug=debug_payload,
        )

    # ─── helpers ─────────────────────────────────────────────────────

    def _refusal(self, query: str, failure_type: str, reason: str, tracer: Tracer) -> RagResult:
        return RagResult(
            answer=self.settings.refusal_message,
            sources=[],
            confidence={"overall_confidence": 0.0, "confidence_level": "low"},
            failure_type=failure_type,
            failure_reason=reason,
            refused=True,
            hallucination={"risk_level": "high", "grounded_ratio": 0.0},
            trace=tracer.summary(),
            answered=False,
            contexts=[],
        )

    def _debug_payload(
        self,
        query: str,
        contexts: list[dict],
        retrieval,
        system_prompt: Optional[str],
        confidence: dict,
        failure_type: Optional[str],
        tracer: Tracer,
    ) -> dict:
        """Full debugging view (only returned when debug mode is enabled)."""
        return {
            "query": query,
            "request_id": tracer.request_id,
            "retrieved_documents": [
                {
                    "source": c.get("source"),
                    "page": c.get("metadata", {}).get("page"),
                    "score": c.get("score"),
                    "rerank_score": c.get("rerank_score"),
                    "original_score": c.get("original_score"),
                    "retrieval_method": c.get("retrieval_method"),
                    "excerpt": (c.get("content") or "")[:300],
                }
                for c in contexts
            ],
            "retrieval_confidence": retrieval.retrieval_confidence if retrieval else None,
            "retrieval_methods": retrieval.methods_used if retrieval else [],
            "similarity_scores": [c.get("score", 0.0) for c in contexts],
            "reranking_scores": [c.get("rerank_score") for c in contexts if c.get("rerank_score") is not None],
            "final_context": [c.get("content") for c in contexts],
            "system_prompt": system_prompt,
            "confidence": confidence,
            "failure_type": failure_type,
            "stage_times": tracer.summary().get("stages", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }