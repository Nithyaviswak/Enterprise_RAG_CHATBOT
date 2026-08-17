"""Cross-encoder re-ranking with a deterministic embedding fallback.

The cross-encoder (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``) gives the
strongest re-ranking signal when available. When it can't be loaded (lightweight
CI / tests), a cosine-similarity re-scoring on top of the query embedding is used
so the pipeline still runs deterministically offline.
"""

import logging
from typing import Callable, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Re-ranks retrieved chunks given a query."""

    def __init__(
        self,
        embedding_service=None,
        model_name: Optional[str] = None,
        enabled: bool = True,
        score_fn: Optional[Callable[[list[tuple[str, str]]], list[float]]] = None,
    ):
        """
        Args:
            embedding_service: Used for the cosine fallback.
            model_name: Cross-encoder HF model id.
            enabled: Whether re-ranking is active.
            score_fn: Injectable scorer for deterministic tests. Takes a list of
                ``(query, doc)`` pairs and returns a list of scores.
        """
        settings = get_settings()
        self.embedding_service = embedding_service
        self.model_name = model_name or settings.reranker_model
        self.enabled = enabled
        self._score_fn = score_fn
        self._cross_encoder = None

    def _load_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded: %s", self.model_name)
        except Exception as e:
            logger.warning("Cross-encoder unavailable (%s) — using embedding fallback", e)
            self._cross_encoder = False
        return self._cross_encoder

    def _scorer(self) -> Optional[Callable[[list[tuple[str, str]]], list[float]]]:
        if self._score_fn:
            return self._score_fn
        ce = self._load_cross_encoder()
        if ce:
            return ce.predict
        if self.embedding_service is not None:
            return self._cosine_scorer
        return None

    def _cosine_scorer(self, pairs: list[tuple[str, str]]) -> list[float]:
        queries = [p[0] for p in pairs]
        docs = [p[1] for p in pairs]
        q_emb = self.embedding_service.encode(queries)
        d_emb = self.embedding_service.encode(docs)
        scores = []
        for qv, dv in zip(q_emb, d_emb):
            dot = sum(a * b for a, b in zip(qv, dv))
            qn = sum(a * a for a in qv) ** 0.5 or 1.0
            dn = sum(b * b for b in dv) ** 0.5 or 1.0
            scores.append(dot / (qn * dn))
        return scores

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: Optional[int] = None,
        original_weight: float = 0.0,
    ) -> list[dict]:
        """Re-rank results in place and return them sorted.

        Each result dict gains ``rerank_score`` and ``combined_score``.
        ``combined_score = (1 - original_weight) * rerank_score +
        original_weight * original_score``.
        """
        if not self.enabled or not results:
            return results

        scorer = self._scorer()
        if scorer is None:
            return results

        try:
            pairs = [(query, r.get("content", "")) for r in results]
            scores = scorer(pairs)
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
                r["original_score"] = r.get("score", 0.0)
                r["combined_score"] = (
                    (1.0 - original_weight) * float(s)
                    + original_weight * r.get("score", 0.0)
                )
            results.sort(key=lambda r: r.get("combined_score", 0.0), reverse=True)
        except Exception as e:
            logger.warning("Re-ranking failed (%s) — keeping original order", e)

        if top_k is not None:
            results = results[:top_k]
        return results