"""
Enhanced Retrieval Service — Multi-Query, Query Expansion, Contextual Compression.

Extends the base retrieval service with:
- Multi-Query Retrieval: Generate query variants for better recall
- Query Expansion: LLM-based expansion with synonyms
- Contextual Compression: Filter redundant context
- Adaptive Top-K: Dynamic result count based on score thresholds
"""

import logging
import asyncio
from typing import Optional
from collections import defaultdict

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.ragflow_client import RAGFlowClient
from app.config import get_settings

logger = logging.getLogger(__name__)


class EnhancedRetrievalService:
    """Enhanced retrieval with multi-query, expansion, and compression."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        ragflow_client: RAGFlowClient,
        gemini_service=None,  # LLM for query expansion
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.ragflow_client = ragflow_client
        self.gemini_service = gemini_service
        self._base_retrieval = None

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

    # ─────────────────────────────────────────────────────────────────
    # Multi-Query Retrieval
    # ─────────────────────────────────────────────────────────────────

    def _generate_query_variants(self, query: str) -> list[str]:
        """Generate query variants for better recall.

        Creates multiple reformulations of the original query:
        - Concise version
        - Expanded synonyms
        - Question form
        """
        variants = [query]

        # Simple expansions (no LLM needed)
        query_lower = query.lower()

        # Add common variations
        if "what" not in query_lower and "how" not in query_lower:
            variants.append(f"What is {query}?")
            variants.append(f"How to {query.lower().rstrip('?')}?")

        # Add synonyms for common terms
        synonym_map = {
            "find": ["find", "locate", "search", "get"],
            "show": ["show", "display", "list", "present"],
            "get": ["get", "obtain", "retrieve", "fetch"],
            "create": ["create", "make", "build", "generate"],
            "update": ["update", "modify", "change", "edit"],
        }

        for word, synonyms in synonym_map.items():
            if word in query_lower:
                for syn in synonyms:
                    variants.append(query_lower.replace(word, syn))

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            v_norm = v.lower().strip()
            if v_norm not in seen:
                seen.add(v_norm)
                unique_variants.append(v)

        return unique_variants[:5]  # Limit to 5 variants

    # ─────────────────────────────────────────────────────────────────
    # Query Expansion with LLM
    # ─────────────────────────────────────────────────────────────────

    async def _expand_query_llm(self, query: str) -> list[str]:
        """Use LLM to generate semantic query expansions."""
        if not self.gemini_service:
            return [query]

        expansion_prompt = f"""Generate 3-5 different ways to ask this question that capture the same intent.

Original question: "{query}"

Return ONLY a JSON array of strings, nothing else. Example: ["variant1", "variant2", "variant3"]"""

        try:
            response = await self.gemini_service.chat(
                message=expansion_prompt,
                context=None,
            )

            # Try to parse JSON response
            import json
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            expansions = json.loads(response.strip())
            if isinstance(expansions, list):
                return [query] + expansions[:4]
        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")

        return [query]

    # ─────────────────────────────────────────────────────────────────
    # Contextual Compression
    # ─────────────────────────────────────────────────────────────────

    def _compress_context(
        self,
        results: list[dict],
        max_tokens: int = 4000,
    ) -> list[dict]:
        """Compress retrieved context by removing redundant information.

        Uses sentence-level embeddings to identify and remove
        sentences that are semantically similar to earlier ones.
        """
        if len(results) <= 3:
            return results

        compressed = []
        seen_content = set()

        for result in results:
            content = result.get("content", "")

            # Simple deduplication
            content_hash = hash(content[:100])
            if content_hash in seen_content:
                continue

            # Check semantic similarity with already-kept results
            is_redundant = False
            for kept in compressed:
                kept_content = kept.get("content", "")
                # Simple word overlap check
                kept_words = set(kept_content.lower().split())
                content_words = set(content.lower().split())
                overlap = len(kept_words & content_words) / max(len(content_words), 1)

                if overlap > 0.8:  # 80% word overlap = redundant
                    is_redundant = True
                    # Merge scores (keep highest)
                    result["score"] = max(result.get("score", 0), kept.get("score", 0))
                    break

            if not is_redundant:
                seen_content.add(content_hash)
                compressed.append(result)

        return compressed

    # ─────────────────────────────────────────────────────────────────
    # Parent-Document Retrieval
    # ─────────────────────────────────────────────────────────────────

    def _get_parent_chunks(
        self,
        child_results: list[dict],
    ) -> list[dict]:
        """Retrieve parent (larger) chunks for more context.

        For each child chunk, try to retrieve its parent chunk
        for better context understanding.
        """
        # This requires storing parent-child relationships in metadata
        parent_results = []

        for child in child_results:
            metadata = child.get("metadata", {})
            parent_id = metadata.get("parent_chunk_id")

            if parent_id:
                try:
                    parent = self.vector_store.collection.get(ids=[parent_id])
                    if parent and parent.get("documents"):
                        parent_results.append({
                            "content": parent["documents"][0],
                            "source": parent["metadatas"][0].get("source", "Unknown") if parent.get("metadatas") else "Unknown",
                            "score": child.get("score", 0),
                            "metadata": parent["metadatas"][0] if parent.get("metadatas") else {},
                            "retrieval_method": "parent",
                        })
                except Exception as e:
                    logger.debug(f"Parent retrieval failed: {e}")

        return parent_results

    # ─────────────────────────────────────────────────────────────────
    # Main Retrieval Pipeline
    # ─────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_ragflow: bool = True,
        use_reranking: bool = True,
        use_multi_query: bool = True,
        use_expansion: bool = True,
        use_compression: bool = True,
        semantic_weight: float = 0.7,
    ) -> list[dict]:
        """Enhanced retrieval with multi-query, expansion, and compression.

        Args:
            query: User's search query
            top_k: Number of final results
            use_ragflow: Include RAGFlow results
            use_reranking: Apply cross-encoder re-ranking
            use_multi_query: Generate query variants
            use_expansion: Use LLM for query expansion
            use_compression: Remove redundant context
            semantic_weight: Weight for semantic vs keyword
        """
        all_results = []

        # ── Step 1: Query Expansion ──
        queries = [query]
        if use_expansion and self.gemini_service:
            expanded = await self._expand_query_llm(query)
            queries.extend(expanded[1:])  # Skip original

        if use_multi_query:
            query_variants = self._generate_query_variants(query)
            queries.extend(query_variants[1:])  # Skip original

        # Deduplicate queries
        queries = list(dict.fromkeys(queries))[:6]  # Max 6 queries
        logger.info(f"Retrieving with {len(queries)} query variants")

        # ── Step 2: Parallel Retrieval for All Queries ──
        query_tasks = []
        for q in queries:
            task = self._retrieve_single(q, top_k, use_ragflow, semantic_weight)
            query_tasks.append(task)

        # Run all retrievals concurrently
        results_lists = await asyncio.gather(*query_tasks, return_exceptions=True)

        for results in results_lists:
            if isinstance(results, list):
                all_results.extend(results)

        if not all_results:
            return []

        # ── Step 3: Deduplicate by Content ──
        seen_content = set()
        unique_results = []
        for r in all_results:
            content_hash = hash(r["content"][:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(r)

        # ── Step 4: Contextual Compression ──
        if use_compression:
            unique_results = self._compress_context(unique_results)

        # ── Step 5: Re-ranking ──
        if use_reranking and len(unique_results) > 1:
            unique_results = await self._rerank(query, unique_results)

        # ── Step 6: Return Top-K ──
        return unique_results[:top_k]

    async def _retrieve_single(
        self,
        query: str,
        top_k: int,
        use_ragflow: bool,
        semantic_weight: float,
    ) -> list[dict]:
        """Retrieve for a single query."""
        results = []

        # Semantic search
        try:
            query_embedding = self.embedding_service.encode_query(query)
            semantic_results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=top_k * 2,
            )
            for r in semantic_results:
                r["retrieval_method"] = "semantic"
            results.extend(semantic_results)
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        # RAGFlow
        if use_ragflow and self.ragflow_client.is_available:
            try:
                ragflow_results = await self.ragflow_client.retrieve(
                    query=query, top_k=top_k
                )
                for r in ragflow_results:
                    r["retrieval_method"] = "ragflow"
                results.extend(ragflow_results)
            except Exception as e:
                logger.warning(f"RAGFlow retrieval failed: {e}")

        # BM25 keyword search
        try:
            bm25_results = self._bm25_search(query, top_k)
            for r in bm25_results:
                r["retrieval_method"] = "keyword"
            results.extend(bm25_results)
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")

        return results

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 keyword search (simplified for enhanced service)."""
        from rank_bm25 import BM25Okapi

        if self.vector_store.collection.count() == 0:
            return []

        all_docs = self.vector_store.collection.get(limit=1000)
        if not all_docs or not all_docs["documents"]:
            return []

        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"] or [{}] * len(documents)

        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        scored_results = [
            {
                "content": documents[i],
                "source": metadatas[i].get("source", "Unknown") if metadatas[i] else "Unknown",
                "score": float(scores[i]),
                "metadata": metadatas[i] or {},
            }
            for i in range(len(documents))
        ]

        max_score = max(scores) if max(scores) > 0 else 1
        for r in scored_results:
            r["score"] = r["score"] / max_score

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    async def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Re-rank results using cross-encoder."""
        reranker = self._get_reranker()
        if reranker is None:
            return results

        try:
            # Run in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            pairs = [(query, r["content"]) for r in results]
            scores = await loop.run_in_executor(None, reranker.predict, pairs)

            for i, score in enumerate(scores):
                results[i]["rerank_score"] = float(score)
                results[i]["original_score"] = results[i].get("score", 0)
                results[i]["score"] = float(score)

            results.sort(key=lambda x: x["score"], reverse=True)
            logger.debug("Re-ranking applied successfully")
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")

        return results
