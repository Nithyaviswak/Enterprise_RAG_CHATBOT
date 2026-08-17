"""Hallucination detection and confidence scoring.

A lightweight, deterministic factuality layer that checks whether each claim in
an answer is grounded in the retrieved context:

- extract claims (sentence-level)
- measure lexical overlap between each claim and the retrieved chunks
- produce a grounding ratio, risk level, and a combined confidence score

This runs for every response so unsupported answers can be caught and reported
(e.g. ``HALLUCINATION_RISK``). It intentionally needs no LLM, keeping it fast,
cheap, and deterministic for tests and evaluation.
"""

import logging
import re
import difflib
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "been", "being", "with", "at", "by", "from",
    "as", "that", "this", "it", "its", "they", "them", "their", "you", "your",
    "we", "our", "i", "he", "she", "his", "her", "not", "no", "do", "does",
    "did", "have", "has", "had", "about", "into", "than", "then", "which",
    "what", "when", "where", "who", "whom", "how", "will", "would", "can",
    "could", "should", "may", "might", "just", "also", "very", "more", "most",
    "some", "any", "all", "each", "other", "such", "only", "come", "per",
    "please", "according",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Acknowledgements / refusals are allowed without a claim-level verdict.
_REFUSAL_MARKERS = ("don't have enough information", "do not have enough information",
                    "not enough information", "cannot answer", "can't answer")


@dataclass
class Claim:
    """A single extracted claim and its grounding verdict."""

    text: str
    grounded: bool
    support_score: float
    supporting_sources: list[str] = field(default_factory=list)


@dataclass
class HallucinationReport:
    """Result of hallucination analysis for one answer."""

    claims: list[Claim]
    grounded_ratio: float
    is_grounded: bool
    risk_level: str
    unsupported_claims: list[str]
    risk_factors: list[str]

    @property
    def secure(self) -> bool:
        return self.is_grounded and self.risk_level in {"low", "medium"}


def _content_words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w not in _STOPWORDS and len(w) > 1}


def _normalize_similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


class HallucinationDetector:
    """Deterministic grounding verifier for generated answers."""

    def __init__(self, min_overlap: Optional[int] = None):
        settings = get_settings()
        self.min_overlap = settings.hallucination_min_overlap if min_overlap is None else min_overlap

    def analyze(self, answer: str, contexts: list[dict]) -> HallucinationReport:
        """Analyze an answer against the retrieved context."""
        if not answer:
            return HallucinationReport(
                claims=[], grounded_ratio=0.0, is_grounded=False,
                risk_level="high", unsupported_claims=[], risk_factors=["empty_answer"],
            )
        if not contexts:
            return HallucinationReport(
                claims=[], grounded_ratio=0.0, is_grounded=False,
                risk_level="high", unsupported_claims=[],
                risk_factors=["no_context"],
            )

        context_texts = [c.get("content", "") for c in contexts]
        context_sources = [c.get("source", "Unknown") for c in contexts]
        context_word_sets = [_content_words(t) for t in context_texts]

        claims: list[Claim] = []
        for sentence in self._extract_claims(answer):
            verdict, score, sources = self._verify_claim(
                sentence, context_word_sets, context_texts, context_sources
            )
            claims.append(
                Claim(text=sentence, grounded=verdict, support_score=score, supporting_sources=sources)
            )

        grounded_ratio = sum(1 for c in claims if c.grounded) / len(claims) if claims else 0.0
        unsupported = [c.text for c in claims if not c.grounded]

        risk_factors: list[str] = []
        if unsupported:
            risk_factors.append(f"{len(unsupported)}_unsupported_claims")
        if grounded_ratio < 0.6:
            risk_factors.append("low_grounding_ratio")
        if self._looks_like_answer(answer) and grounded_ratio < 0.4:
            risk_factors.append("high_hallucination_likelihood")

        if not claims:
            risk_level = "medium"
        elif grounded_ratio >= 0.7:
            risk_level = "low"
        elif grounded_ratio >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "high"

        is_grounded = grounded_ratio >= 0.7
        report = HallucinationReport(
            claims=claims,
            grounded_ratio=round(grounded_ratio, 3),
            is_grounded=is_grounded,
            risk_level=risk_level,
            unsupported_claims=unsupported,
            risk_factors=risk_factors,
        )
        logger.debug(
            "Hallucination check: grounded_ratio=%.2f risk=%s",
            report.grounded_ratio, report.risk_level,
        )
        return report

    # ─── internals ───────────────────────────────────────────────────

    def _extract_claims(self, answer: str) -> list[str]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
        return [s for s in sentences if len(s.split()) > 5]

    def _verify_claim(
        self,
        claim: str,
        context_word_sets: list[set[str]],
        context_texts: list[str],
        context_sources: list[str],
    ) -> tuple[bool, float, list[str]]:
        claim_words = _content_words(re.sub(r"\[\d+\]|\[Source[^\]]*\]", "", claim))
        claim_text = claim.lower()
        best_overlap = 0
        best_ratio = 0.0
        supporting: list[str] = []

        # Refusal / hedge sentences are considered grounded by definition.
        if any(m in claim_text for m in _REFUSAL_MARKERS):
            return True, 1.0, []

        for i, ctx_words in enumerate(context_word_sets):
            overlap = len(claim_words & ctx_words)
            ratio = overlap / max(len(claim_words), 1)
            # Also reward exact phrasing matches (quotations, key passages).
            phrase_score = _normalize_similar(claim, context_texts[i][:2000])
            combined = ratio * 0.6 + phrase_score * 0.4
            if ratio > best_overlap / max(len(claim_words), 1) or combined > best_ratio:
                best_overlap = overlap
                best_ratio = combined
                if self._is_supported(combined, ratio):
                    supporting = [context_sources[i]]
                elif combined > 0.5:
                    supporting = [context_sources[i]]

        grounded = self._is_supported(best_ratio, best_overlap / max(len(claim_words), 1))
        # Require at least ``min_overlap`` real term matches to be grounded.
        grounded = grounded and best_overlap >= self.min_overlap
        return grounded, round(best_ratio, 3), supporting

    def _is_supported(self, similarity: float, term_ratio: float) -> bool:
        return similarity >= 0.25 or term_ratio >= 0.35

    def _looks_like_answer(self, answer: str) -> bool:
        """Heuristic: a substantive answer with low grounding is high risk."""
        return len(answer.split()) >= 15

    def combined_confidence(
        self,
        retrieval_confidence: float,
        report: HallucinationReport,
        answered: bool,
    ) -> dict:
        """Combine retrieval confidence with answer grounding into one score."""
        if not answered:
            overall = round(retrieval_confidence * 0.4, 3)
            level = "low"
        else:
            overall = round(
                retrieval_confidence * 0.4 + report.grounded_ratio * 0.6, 3
            )
            if overall >= 0.75:
                level = "high"
            elif overall >= 0.5:
                level = "medium"
            else:
                level = "low"

        return {
            "overall_confidence": max(0.0, min(1.0, overall)),
            "confidence_level": level,
            "retrieval_confidence": round(retrieval_confidence, 3),
            "grounding_ratio": report.grounded_ratio,
            "risky": report.risk_level in {"high", "medium"} and answered,
        }


def is_refusal_answer(answer: str) -> bool:
    """Return True when the answer is the standard low-confidence refusal."""
    normalized = " ".join(answer.lower().split())
    return any(marker in normalized for marker in _REFUSAL_MARKERS)