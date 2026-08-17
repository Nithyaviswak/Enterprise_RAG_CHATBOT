"""Deterministic RAG metrics (faithfulness, relevancy, precision, recall).

These metrics are computed locally without any paid API, so evaluation is
reproducible and safe to run in CI. Faithfulness uses the lexical grounding
verifier from ``app.rag.hallucination``; relevancy uses local embeddings
cosine similarity (falling back to lexical overlap); precision/recall use
term coverage of the reference answer by the retrieved context.

An optional RAGAS adapter is provided for LLM-judge quality metrics, enabled
explicitly with ``--framework ragas`` when RAGAS and a compatible LLM are
configured. Both paths produce the same summary schema.
"""

import logging
from typing import Optional

from app.rag.hallucination import HallucinationDetector, is_refusal_answer

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
    "please", "according", "please",
}


def _words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w not in _STOPWORDS and len(w) > 1}


def _cosine(q: list[float], a: list[float]) -> float:
    qn = sum(x * x for x in q) ** 0.5 or 1.0
    an = sum(x * x for x in a) ** 0.5 or 1.0
    return sum(x * y for x, y in zip(q, a)) / (qn * an)


def _embed(embedding_service, text: str) -> Optional[list[float]]:
    if embedding_service is None:
        return None
    try:
        return embedding_service.encode([text])[0]
    except Exception as e:  # pragma: no cover
        logger.warning("Embedding unavailable for metric: %s", e)
        return None


def _lexical_similarity(a: str, b: str) -> float:
    a_words = _words(a)
    b_words = _words(b)
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words)


def normalize_contexts(contexts) -> list[dict]:
    """Accept chunk dicts or plain strings and normalize to dicts."""
    out = []
    for c in contexts:
        if isinstance(c, str):
            out.append({"content": c})
        else:
            out.append(c)
    return out


class Metrics:
    """Computes deterministic RAG quality metrics for evaluation samples."""

    def __init__(self, embedding_service=None, detector: Optional[HallucinationDetector] = None):
        self.embedding_service = embedding_service
        self.detector = detector or HallucinationDetector()

    def compute_sample(
        self,
        question: str,
        reference_answer: str,
        contexts,
        generated_answer: str,
        expected_in_corpus: bool = True,
    ) -> dict:
        """Compute all metrics for a single sample."""
        contexts = normalize_contexts(contexts)
        claimed_grounding = True
        if contexts:
            report = self.detector.analyze(generated_answer, contexts)
            grounding_ratio = report.grounded_ratio
            claimed_grounding = report.is_grounded
        else:
            report = None
            grounding_ratio = 1.0 if (not generated_answer or is_refusal_answer(generated_answer)) else 0.0

        # Faithfulness — a refusal or empty answer makes no claims, so it is faithful.
        if not generated_answer.strip() or is_refusal_answer(generated_answer):
            faithfulness = 1.0
        else:
            faithfulness = grounding_ratio

        # Answer relevancy — how well the answer addresses the question.
        if not generated_answer.strip() or is_refusal_answer(generated_answer):
            answer_relevancy = 0.4 if not expected_in_corpus else 0.3
        else:
            q_emb = _embed(self.embedding_service, question)
            a_emb = _embed(self.embedding_service, generated_answer)
            if q_emb and a_emb:
                answer_relevancy = max(0.0, _cosine(q_emb, a_emb))
            else:
                answer_relevancy = _lexical_similarity(question, generated_answer)

        # Context precision — are retrieved chunks relevant to the reference,
        # with higher-ranked chunks weighted more (RAGAS-style precision@k proxy)?
        ref_words = _words(reference_answer) or _words(question)
        if not contexts:
            context_precision = 0.0
        else:
            weights_sum = 0.0
            weighted_relevance = 0.0
            for i, c in enumerate(contexts, start=1):
                chunk_words = _words(c.get("content", ""))
                overlap = len(ref_words & chunk_words) if ref_words else 0
                relevance = min(1.0, overlap / max(len(ref_words), 1))
                w = 1.0 / i
                weights_sum += w
                weighted_relevance += relevance * w
            context_precision = weighted_relevance / weights_sum if weights_sum else 0.0

        # Context recall — how much of the reference is covered by ALL context?
        if not ref_words:
            context_recall = 1.0
        elif not contexts:
            context_recall = 0.0
        else:
            covered = set()
            for c in contexts:
                covered |= _words(c.get("content", ""))
            context_recall = len(ref_words & covered) / len(ref_words)

        answer = generated_answer or ""
        refused = is_refusal_answer(answer) or not expected_in_corpus and not answer.strip()
        hallucination_flag = (
            expected_in_corpus
            and bool(answer.strip())
            and not refused
            and grounding_ratio < 0.6
        )

        # Answer coverage of reference (informational).
        ref_words_full = _words(reference_answer)
        answer_words = _words(answer)
        answer_coverage = len(ref_words_full & answer_words) / len(ref_words_full) if ref_words_full else 1.0

        return {
            "id": None,
            "question": question,
            "faithfulness": round(float(faithfulness), 3),
            "answer_relevancy": round(float(answer_relevancy), 3),
            "context_precision": round(float(context_precision), 3),
            "context_recall": round(float(context_recall), 3),
            "hallucination_flag": bool(hallucination_flag),
            "refused": bool(refused),
            "grounded_ratio": round(float(grounding_ratio), 3),
            "answer_coverage": round(float(answer_coverage), 3),
            "expected_in_corpus": bool(expected_in_corpus),
            "retrieved_chunks": len(contexts),
        }


def aggregate(sample_metrics: list[dict]) -> dict:
    """Summarize per-sample metrics into an aggregate report."""
    n = len(sample_metrics)
    if n == 0:
        return {}

    def mean(key):
        return round(sum(m[key] for m in sample_metrics) / n, 3)

    hallucination_rate = round(
        sum(1 for m in sample_metrics if m["hallucination_flag"]) / n, 3
    )
    refusal_rate = round(sum(1 for m in sample_metrics if m["refused"]) / n, 3)
    retrieval_success = round(
        sum(1 for m in sample_metrics if m["retrieved_chunks"] > 0) / n, 3
    )

    return {
        "total_samples": n,
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "hallucination_rate": hallucination_rate,
        "refusal_rate": refusal_rate,
        "retrieval_success_rate": retrieval_success,
        "avg_retrieved_chunks": round(sum(m["retrieved_chunks"] for m in sample_metrics) / n, 2),
        "avg_grounding_ratio": mean("grounded_ratio"),
    }