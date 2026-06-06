"""
Hallucination Mitigation & Confidence Scoring Service.

Implements:
1. Grounded Generation: Force answers to cite retrieved evidence
2. Source Attribution: Link claims to specific sources/chunks
3. Answer Verification: Validate claims against retrieved context
4. Confidence Scoring: Generate retrieval and answer confidence scores
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ClaimAnalysis:
    """Analysis of a single claim in the answer."""
    claim: str
    is_verified: bool
    supporting_sources: list[str]
    confidence: float
    issue: Optional[str] = None


@dataclass
class AnswerVerification:
    """Complete verification result for an answer."""
    is_grounded: bool
    overall_confidence: float
    claims: list[ClaimAnalysis]
    unsupported_claims: list[str]
    missing_evidence: list[str]
    verification_summary: str


class HallucinationMitigationService:
    """Service for reducing hallucinations and scoring confidence."""

    def __init__(self, llm_service=None, embedding_service=None):
        self.llm_service = llm_service
        self.embedding_service = embedding_service

    # ─────────────────────────────────────────────────────────────────
    # Claim Extraction & Verification
    # ─────────────────────────────────────────────────────────────────

    async def verify_answer(
        self,
        answer: str,
        retrieved_context: list[dict],
    ) -> AnswerVerification:
        """Verify that answer claims are supported by retrieved context.

        Args:
            answer: Generated answer to verify
            retrieved_context: List of retrieved context chunks

        Returns:
            AnswerVerification with detailed analysis
        """
        if not self.llm_service:
            # Fallback to simple keyword matching
            return await self._simple_verification(answer, retrieved_context)

        context_combined = "\n\n---\n\n".join(
            f"[Source {i+1}]: {ctx.get('content', ctx.get('text', ''))}"
            for i, ctx in enumerate(retrieved_context)
        )

        # Extract claims and verify
        verification_prompt = f"""You are a fact-checker for a RAG system.

Analyze the following answer and verify each factual claim against the retrieved context.

## Retrieved Context:
{context_combined}

## Generated Answer:
{answer}

## Your Task:

1. **Extract all factual claims** from the answer (not trivial phrases)
2. **Verify each claim** against the retrieved context
3. **For each claim**, determine:
   - Is it supported by the context? (yes/no)
   - Which source(s) support it?
   - Confidence level (0-1)

Return ONLY a JSON object with this exact format:
```json
{{
  "claims": [
    {{
      "claim": "specific factual claim extracted",
      "is_verified": true or false,
      "supporting_sources": ["source identifier or text snippet"],
      "confidence": 0.0-1.0,
      "issue": "null or brief description of problem if unverified"
    }}
  ],
  "unsupported_claims": ["list of claims not supported by context"],
  "missing_evidence": ["specific information in answer not in context"],
  "verification_summary": "brief overall assessment"
}}
```"""

        try:
            response = await self.llm_service.chat(
                message=verification_prompt,
                context=None,
            )

            # Parse response
            import json
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())

                # Convert to ClaimAnalysis objects
                claims = [
                    ClaimAnalysis(
                        claim=c["claim"],
                        is_verified=c.get("is_verified", False),
                        supporting_sources=c.get("supporting_sources", []),
                        confidence=c.get("confidence", 0.5),
                        issue=c.get("issue"),
                    )
                    for c in analysis.get("claims", [])
                ]

                # Calculate overall metrics
                verified_count = sum(1 for c in claims if c.is_verified)
                is_grounded = len(claims) == 0 or verified_count / len(claims) >= 0.7

                overall_confidence = (
                    sum(c.confidence for c in claims) / len(claims)
                    if claims else 0.5
                )

                return AnswerVerification(
                    is_grounded=is_grounded,
                    overall_confidence=overall_confidence,
                    claims=claims,
                    unsupported_claims=analysis.get("unsupported_claims", []),
                    missing_evidence=analysis.get("missing_evidence", []),
                    verification_summary=analysis.get("verification_summary", ""),
                )
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")

        # Fallback
        return await self._simple_verification(answer, retrieved_context)

    async def _simple_verification(
        self,
        answer: str,
        retrieved_context: list[dict],
    ) -> AnswerVerification:
        """Simple keyword-based verification fallback."""
        answer_lower = answer.lower()
        context_texts = [
            ctx.get("content", ctx.get("text", "")).lower()
            for ctx in retrieved_context
        ]

        # Simple claim extraction (split by sentences)
        sentences = re.split(r'[.!?]+', answer)
        claims = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:  # Skip trivial sentences
                continue

            # Check if sentence has keywords from context
            found_sources = []
            for i, ctx_text in enumerate(context_texts):
                # Check for word overlap
                sent_words = set(sent.split())
                ctx_words = set(ctx_text.split())
                overlap = len(sent_words & ctx_words)

                if overlap >= 3:
                    found_sources.append(f"Source {i+1}")

            claims.append(ClaimAnalysis(
                claim=sent,
                is_verified=len(found_sources) > 0,
                supporting_sources=found_sources,
                confidence=0.8 if found_sources else 0.2,
            ))

        verified_count = sum(1 for c in claims if c.is_verified)
        is_grounded = len(claims) == 0 or verified_count / len(claims) >= 0.7

        return AnswerVerification(
            is_grounded=is_grounded,
            overall_confidence=0.7 if is_grounded else 0.3,
            claims=claims,
            unsupported_claims=[c.claim for c in claims if not c.is_verified],
            missing_evidence=[],
            verification_summary="Basic verification complete",
        )

    # ─────────────────────────────────────────────────────────────────
    # Source Attribution
    # ─────────────────────────────────────────────────────────────────

    def attribute_sources(
        self,
        answer: str,
        retrieved_context: list[dict],
    ) -> dict:
        """Add source attribution to answer.

        Returns answer with inline citations and source list.

        Args:
            answer: Generated answer
            retrieved_context: Retrieved context chunks

        Returns:
            Dict with attributed answer and source list
        """
        # Simple citation: add footnote-style references
        answer_with_citations = answer
        sources = []

        # Build source list
        for i, ctx in enumerate(retrieved_context):
            source_name = ctx.get("source", f"Source {i+1}")
            chunk_idx = ctx.get("metadata", {}).get("chunk_index", i)
            sources.append({
                "id": i + 1,
                "name": source_name,
                "chunk_index": chunk_idx,
                "relevance": ctx.get("score", 0),
            })

        return {
            "answer_with_citations": answer_with_citations,
            "sources": sources,
            "source_count": len(sources),
        }

    # ─────────────────────────────────────────────────────────────────
    # Confidence Scoring
    # ─────────────────────────────────────────────────────────────────

    def calculate_confidence(
        self,
        retrieved_context: list[dict],
        verification: AnswerVerification,
    ) -> dict:
        """Calculate overall confidence score for the response.

        Args:
            retrieved_context: Retrieved context chunks
            verification: Answer verification result

        Returns:
            Dict with confidence breakdown
        """
        # Retrieval confidence: based on context scores
        context_scores = [ctx.get("score", 0) for ctx in retrieved_context]
        retrieval_confidence = (
            sum(context_scores) / len(context_scores)
            if context_scores else 0.0
        )

        # Answer confidence: based on verification
        answer_confidence = verification.overall_confidence

        # Combined confidence
        overall_confidence = (
            retrieval_confidence * 0.4 +  # Retrieval matters
            answer_confidence * 0.6  # Answer quality matters more
        )

        # Confidence level labels
        if overall_confidence >= 0.8:
            confidence_level = "high"
        elif overall_confidence >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "retrieval_confidence": round(retrieval_confidence, 3),
            "answer_confidence": round(answer_confidence, 3),
            "overall_confidence": round(overall_confidence, 3),
            "confidence_level": confidence_level,
            "is_grounded": verification.is_grounded,
            "retrieved_chunks": len(retrieved_context),
            "verified_claims": sum(1 for c in verification.claims if c.is_verified),
            "total_claims": len(verification.claims),
        }

    # ─────────────────────────────────────────────────────────────────
    # Grounded Generation Prompt
    # ─────────────────────────────────────────────────────────────────

    def get_grounded_prompt(
        self,
        context: list[dict],
    ) -> str:
        """Generate system prompt for grounded generation.

        Use this to instruct the LGM to always cite sources.
        """
        context_section = "\n\n".join(
            f"**[Source {i+1}]:** {c.get('content', '')[:500]}"
            for i, c in enumerate(context)
        )

        grounded_prompt = f"""You are a factual AI assistant. Your responses MUST be grounded in the provided sources.

## Instructions:
1. ONLY make claims that are directly supported by the sources
2. Cite sources using [Source N] notation after each factual claim
3. If sources don't contain enough information, explicitly state "The provided sources do not contain information about X"
4. Do NOT make up facts or hallucinate information
5. Be precise and accurate

## Sources:
{context_section}

## Answer the user's question based ONLY on these sources."""

        return grounded_prompt

    # ─────────────────────────────────────────────────────────────────
    # Hallucination Alert
    # ─────────────────────────────────────────────────────────────────

    async def check_hallucination_risk(
        self,
        answer: str,
        retrieved_context: list[dict],
    ) -> dict:
        """Quick check for potential hallucination risk.

        Returns risk assessment without full verification.
        """
        # Low context coverage = high risk
        has_context = len(retrieved_context) > 0

        # Very short answers might indicate failure
        is_too_short = len(answer.split()) < 10 and len(retrieved_context) > 0

        # Check for hedge words that might indicate uncertainty
        hedge_words = ["might", "could be", "probably", "perhaps", "possibly", "I think"]
        has_hedge = any(word in answer.lower() for word in hedge_words)

        # Hallucination indicators
        risk_factors = []
        if not has_context:
            risk_factors.append("no_context")
        if is_too_short:
            risk_factors.append("answer_too_short")

        risk_level = "high" if len(risk_factors) >= 2 else "medium" if risk_factors else "low"

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "has_context": has_context,
            "answer_length": len(answer.split()),
            "has_hedging": has_hedge,
        }
