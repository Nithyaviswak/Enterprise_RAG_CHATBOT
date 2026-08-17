"""Hybrid retriever: semantic + BM25 with thresholding and dedup.

Pipeline: over-fetch candidates → similarity threshold → optional metadata
filter → hybrid fusion → dedup → re-rank → top-k.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.rag.reranker import Reranker
from app.rag.failures import RetrievalFailure

logger = logging.getLogger(__name__)

# ChromaDB cosine distance maps to similarity in [0, 1]: sim = 1 - distance.
_MAX_BM25_DOCS = 1000


@dataclass
class RetrievalResult:
    """Result container from the retriever."""

    chunks: list[dict] = field(default_factory=list)
    retrieval_confidence: float = 0.0
    latency_ms: float = 0.0
    methods_used: list[str] = field(default_factory=list)
    raw_semantic: list[dict] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.chunks


class Retriever:
    """Production hybrid retriever built on the existing vector store."""

    def __init__(
        self,
        vector_store,
        embedding_service,
        ragflow_client=None,
        reranker: Optional[Reranker] = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.ragflow_client = ragflow_client
        self.reranker = reranker or Reranker(embedding_service=embedding_service)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        hybrid: Optional[bool] = None,
        similarity_threshold: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        use_reranking: bool = True,
        use_ragflow: bool = True,
    ) -> RetrievalResult:
        """Retrieve and rank chunks for a query.

        Args:
            query: User query.
            top_k: Final number of chunks (default: settings.retrieval_top_k).
            where: Optional metadata filter (e.g. {"source": "file.pdf"}).
            hybrid: Combine semantic + BM25 (default: settings.retrieval_hybrid).
            similarity_threshold: Drop semantic hits below this cosine similarity
                (default: settings.similarity_threshold).
            semantic_weight: Semantic share in the hybrid fusion (0-1).
            use_reranking: Apply the cross-encoder re-ranker.
            use_ragflow: Include RAGFlow results when available.
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        hybrid = settings.retrieval_hybrid if hybrid is None else hybrid
        similarity_threshold = (
            settings.similarity_threshold if similarity_threshold is None else similarity_threshold
        )
        semantic_weight = (
            settings.retrieval_semantic_weight if semantic_weight is None else semantic_weight
        )

        start = time.perf_counter()
        methods: list[str] = []

        # ── 1. Semantic search ─────────────────────────────────────
        try:
            q_emb = self.embedding_service.encode_query(query)
        except Exception as e:
            raise RetrievalFailure(f"Query embedding failed: {e}", query=query)

        raw_semantic: list[dict] = []
        semantic_filtered: list[dict] = []
        try:
            raw_semantic = self.vector_store.query(
                query_embedding=q_emb,
                n_results=max(top_k * 4, settings.rerank_top_k * 2),
                where=where,
            )
            for r in raw_semantic:
                r.setdefault("retrieval_method", "semantic")
                if r.get("score", 0.0) >= similarity_threshold:
                    semantic_filtered.append(r)
            methods.append("semantic")
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)

        # ── 2. BM25 keyword search ────────────────────────────────
        keyword: list[dict] = []
        if hybrid:
            try:
                keyword = self._bm25_search(query, settings.rerank_top_k * 2, where=where)
                for r in keyword:
                    r["retrieval_method"] = "keyword"
                methods.append("keyword")
            except Exception as e:
                logger.warning("BM25 search failed: %s", e)

        # ── 3. RAGFlow (optional external retriever) ───────────────
        ragflow_results: list[dict] = []
        if use_ragflow and self.ragflow_client is not None and self.ragflow_client.is_available:
            try:
                ragflow_results = self._ragflow_search(query, top_k * 2)
                for r in ragflow_results:
                    r["retrieval_method"] = "ragflow"
                methods.append("ragflow")
            except Exception as e:
                logger.warning("RAGFlow retrieval failed: %s", e)

        # ── 4. Hybrid fusion + dedup ──────────────────────────────
        fused = self._fuse(semantic_filtered, keyword, ragflow_results, semantic_weight)
        deduped = self._dedupe(fused)

        # ── 5. Re-rank and trim to top-k ──────────────────────────
        if use_reranking and deduped:
            deduped = self.reranker.rerank(query, deduped, top_k=settings.rerank_top_k)
        else:
            deduped.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            deduped = deduped[:settings.rerank_top_k]

        chunks = deduped[:top_k]
        confidence = self._confidence(chunks)
        latency = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            chunks=chunks,
            retrieval_confidence=confidence,
            latency_ms=round(latency, 2),
            methods_used=methods,
            raw_semantic=raw_semantic,
        )

    # ─── internals ─────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int, where: Optional[dict] = None) -> list[dict]:
        if self.vector_store.collection.count() == 0:
            return []
        all_docs = self.vector_store.collection.get(limit=_MAX_BM25_DOCS)
        if not all_docs or not all_docs["documents"]:
            return []

        docs = all_docs["documents"]
        metas = all_docs["metadatas"] or [dict()] * len(docs)
        ids = all_docs["ids"] or []

        t_docs = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(t_docs)
        scores = bm25.get_scores(query.lower().split())
        if getattr(scores, "size", 1) > 0:
            max_score = float(scores.max())
        else:
            max_score = 0.0
        if max_score <= 0:
            max_score = 1.0

        scored: list[dict] = []
        for i, s in enumerate(scores):
            meta = metas[i] or {}
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            scored.append(
                {
                    "id": ids[i] if ids and i < len(ids) else None,
                    "content": docs[i],
                    "source": meta.get("source", "Unknown"),
                    "score": float(s) / (max_score or 1.0),
                    "metadata": meta,
                }
            )
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def _ragflow_search(self, query: str, top_k: int) -> list[dict]:
        import asyncio

        result = asyncio.run(self.ragflow_client.retrieve(query=query, top_k=top_k))
        return [dict(r) for r in result]

    def _fuse(
        self,
        semantic: list[dict],
        keyword: list[dict],
        ragflow: list[dict],
        semantic_weight: float,
    ) -> list[dict]:
        """Merge results keyed by normalized content; hybrid-fuse scores."""
        merged: dict[str, dict] = {}

        def add(entry: dict, method: str, weight: float):
            key = entry.get("content", "")[:300].lower()
            if not key:
                return
            if key not in merged:
                base = dict(entry)
                base["_scores"] = {}
                merged[key] = base
            merged[key]["_scores"][method] = max(
                merged[key]["_scores"].get(method, 0.0),
                entry.get("score", 0.0),
            )
            if "score" not in merged[key] or merged[key].get("score", 0) < entry.get("score", 0):
                merged[key]["score"] = entry.get("score", 0.0)
            if not merged[key].get("source") and entry.get("source"):
                merged[key]["source"] = entry["source"]
            if not merged[key].get("metadata") and entry.get("metadata"):
                merged[key]["metadata"] = entry["metadata"]

        for r in semantic:
            add(r, "semantic", semantic_weight)
        for r in keyword:
            add(r, "keyword", 1.0 - semantic_weight)
        for r in ragflow:
            add(r, "ragflow", 0.5)

        out: list[dict] = []
        for key, entry in merged.items():
            scores = entry.pop("_scores")
            combined = entry.get("score", 0.0)
            if "semantic" in scores and "keyword" in scores:
                combined = (
                    semantic_weight * scores["semantic"]
                    + (1.0 - semantic_weight) * scores["keyword"]
                )
                entry["combined_score"] = combined
            entry["score"] = round(float(combined), 4)
            out.append(entry)
        out.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return out

    def _dedupe(self, results: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for r in results:
            key = (r.get("content", "")[:200] + r.get("source", "")).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _confidence(self, chunks: list[dict]) -> float:
        """Retrieval confidence based on mean top-1 weighted similarity."""
        if not chunks:
            return 0.0
        scores = [c.get("score", 0.0) for c in chunks]
        top1 = max(scores)
        mean = sum(scores) / len(scores)
        hit = sum(1 for s in scores if s >= 0.5) / len(scores)
        return round(min(1.0, 0.4 * top1 + 0.3 * mean + 0.3 * hit), 3)

    def get_context(self, chunks: list[dict], max_chars: Optional[int] = None) -> list[dict]:
        """Return the final context passed to the LLM (bounds total size)."""
        settings = get_settings()
        max_chars = max_chars or settings.max_context_chars
        budget = max_chars
        out: list[dict] = []
        for c in chunks:
            content = c.get("content", "")
            if len(content) > budget and out:
                content = content[:budget]
            if len(content) <= 0:
                continue
            entry = dict(c)
            entry["content"] = content
            budget -= len(content)
            out.append(entry)
            if budget <= 0:
                break
        return out

    def apply_metadata_filter(self, chunks: list[dict], filters: dict) -> list[dict]:
        """Client-side metadata filtering for results lacking a store filter."""
        if not filters:
            return chunks
        return [c for c in chunks if all(c.get("metadata", {}).get(k) == v for k, v in filters.items())]