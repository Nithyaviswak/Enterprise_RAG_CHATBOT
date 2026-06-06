"""
Retrieval Service — Hybrid Search + Re-ranking.

Combines semantic search (ChromaDB) with keyword search (BM25)
and applies cross-encoder re-ranking for optimal retrieval quality.
"""

import logging
import math
from typing import Optional

from rank_bm25 import BM25Okapi

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.ragflow_client import RAGFlowClient

logger = logging.getLogger(__name__)


class RetrievalService:
    """Hybrid retrieval with semantic + keyword search and re-ranking."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        ragflow_client: RAGFlowClient,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.ragflow_client = ragflow_client

        # Lazy-load cross-encoder for re-ranking
        self._reranker = None

    def _get_reranker(self):
        """Lazily load the cross-encoder re-ranker."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                from app.config import get_settings

                settings = get_settings()
                self._reranker = CrossEncoder(settings.reranker_model)
                logger.info(f"Cross-encoder re-ranker loaded: {settings.reranker_model}")
            except Exception as e:
                logger.warning(f"Could not load re-ranker: {e}")
        return self._reranker

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_ragflow: bool = True,
        use_reranking: bool = True,
        semantic_weight: float = 0.7,
    ) -> list[dict]:
        """Perform hybrid retrieval combining multiple sources.

        Args:
            query: User's search query.
            top_k: Number of final results to return.
            use_ragflow: Whether to also query RAGFlow.
            use_reranking: Whether to apply cross-encoder re-ranking.
            semantic_weight: Weight for semantic vs keyword results (0-1).

        Returns:
            List of retrieved chunks with content, source, and score.
        """
        all_results = []

        # 1. Semantic search via ChromaDB
        try:
            query_embedding = self.embedding_service.encode_query(query)
            semantic_results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=top_k * 2,  # Over-fetch for fusion
            )
            for r in semantic_results:
                r["retrieval_method"] = "semantic"
            all_results.extend(semantic_results)
            logger.debug(f"Semantic search returned {len(semantic_results)} results")
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        # 2. RAGFlow retrieval (if available and enabled)
        if use_ragflow and self.ragflow_client.is_available:
            try:
                ragflow_results = await self.ragflow_client.retrieve(
                    query=query, top_k=top_k
                )
                for r in ragflow_results:
                    r["retrieval_method"] = "ragflow"
                all_results.extend(ragflow_results)
                logger.debug(f"RAGFlow returned {len(ragflow_results)} results")
            except Exception as e:
                logger.warning(f"RAGFlow retrieval failed: {e}")

        # 3. BM25 keyword search on ChromaDB documents
        try:
            bm25_results = self._bm25_search(query, top_k)
            for r in bm25_results:
                r["retrieval_method"] = "keyword"
            all_results.extend(bm25_results)
            logger.debug(f"BM25 search returned {len(bm25_results)} results")
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")

        if not all_results:
            return []

        # 4. Deduplicate by content
        seen_content = set()
        unique_results = []
        for r in all_results:
            content_hash = hash(r["content"][:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(r)

        # 5. Re-rank with cross-encoder
        if use_reranking and len(unique_results) > 1:
            unique_results = self._rerank(query, unique_results)

        # 6. Return top_k results
        return unique_results[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 keyword search over stored documents."""
        # Get all documents from ChromaDB for BM25
        if self.vector_store.collection.count() == 0:
            return []

        all_docs = self.vector_store.collection.get(limit=1000)
        if not all_docs or not all_docs["documents"]:
            return []

        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"] or [{}] * len(documents)

        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        # Score query
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Get top results
        scored_results = [
            {
                "content": documents[i],
                "source": metadatas[i].get("source", "Unknown") if metadatas[i] else "Unknown",
                "score": float(scores[i]),
                "metadata": metadatas[i] or {},
            }
            for i in range(len(documents))
        ]

        # Normalize BM25 scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1
        for r in scored_results:
            r["score"] = r["score"] / max_score

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Re-rank results using cross-encoder."""
        reranker = self._get_reranker()
        if reranker is None:
            return results

        try:
            pairs = [(query, r["content"]) for r in results]
            scores = reranker.predict(pairs)

            for i, score in enumerate(scores):
                results[i]["rerank_score"] = float(score)
                results[i]["original_score"] = results[i]["score"]
                results[i]["score"] = float(score)

            results.sort(key=lambda x: x["score"], reverse=True)
            logger.debug("Re-ranking applied successfully")
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")

        return results
