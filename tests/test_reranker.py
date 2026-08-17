"""Unit tests for re-ranking (deterministic score_fn path)."""

from tests.conftest import FakeEmbeddingService, make_docs, make_reranker

DOCS = [
    {"id": "a", "content": "The pipeline uses hybrid retrieval and re-ranking.", "source": "t.txt", "metadata": {}},
    {"id": "b", "content": "The agent's failover system hot-swaps backup models on 429 errors.", "source": "t.txt", "metadata": {}},
    {"id": "c", "content": "Another document about weather patterns across regions.", "source": "t.txt", "metadata": {}},
]


def test_rerank_with_injected_scorer_orders_results():
    reranker = make_reranker(
        score_fn=lambda pairs: [0.9 if "failover" in p[1] else 0.5 if "retrieval" in p[1] else 0.1 for p in pairs]
    )
    results = [dict(d) for d in DOCS]
    out = reranker.rerank("how does failover work", results, top_k=2)
    assert [r["id"] for r in out] == ["b", "a"]
    assert all("rerank_score" in r for r in out)


def test_rerank_disabled_returns_original():
    reranker = make_reranker(enabled=False)
    results = [dict(d) for d in DOCS]
    out = reranker.rerank("query", results, top_k=3)
    assert out == results


def test_rerank_top_k_trims():
    reranker = make_reranker(score_fn=lambda pairs: [0.1, 0.9, 0.5])
    out = reranker.rerank("q", [dict(d) for d in DOCS], top_k=1)
    assert len(out) == 1


def test_cosine_fallback_scores():
    reranker = make_reranker()  # enabled, no score_fn, embedding fallback
    results = [dict(d) for d in DOCS]
    out = reranker.rerank("retrieval and re-ranking", results, top_k=3)
    assert all("rerank_score" in r for r in out)
    # The chunk sharing the most full terms should rank first.
    assert out[0]["id"] == "a"


def test_embedding_service_used_for_fallback():
    class Spy(FakeEmbeddingService):
        call_count = 0

        def encode(self, texts, **kwargs):
            Spy.call_count += 1
            return super().encode(texts)

    reranker = make_reranker()
    reranker.embedding_service = Spy()
    reranker.rerank("q", [dict(d) for d in DOCS][:2], top_k=2)
    assert Spy.call_count >= 1