"""Unit tests for the hybrid retriever (hermetic, deterministic)."""

from app.rag.retriever import Retriever
from tests.conftest import FakeEmbeddingService, FakeVectorStore, make_reranker

CONTENTS = [
    "The AI Research Agent performs deep research using LangGraph and Gemini.",
    "The Voice Healthcare Agent reduced latency and improved reliability for patients.",
    "Computer vision models process medical images for diagnostics.",
]


def build_retriever(**kwargs):
    store = FakeVectorStore(
        [{"id": f"c{i}", "content": c, "metadata": {"source": "kb.txt", "page": i}} for i, c in enumerate(CONTENTS)],
        embedding_service=FakeEmbeddingService(),
    )
    return Retriever(
        vector_store=store,
        embedding_service=FakeEmbeddingService(),
        reranker=make_reranker(enabled=False),
        **kwargs,
    )


def test_retrieve_semantic_match():
    retriever = build_retriever()
    result = retriever.retrieve("how does the research agent work", top_k=2, similarity_threshold=0.0)
    assert result.chunks
    assert result.methods_used
    assert result.chunks[0]["content"] == CONTENTS[0]


def test_retrieve_with_metadata_filter():
    retriever = build_retriever()
    result = retriever.retrieve("research agent", top_k=3, where={"source": "kb.txt"})
    assert result.chunks
    assert all(c["metadata"]["source"] == "kb.txt" for c in result.chunks)


def test_retrieve_empty_store():
    store = FakeVectorStore([], embedding_service=FakeEmbeddingService())
    retriever = Retriever(
        vector_store=store, embedding_service=FakeEmbeddingService(), reranker=make_reranker(enabled=False)
    )
    result = retriever.retrieve("anything", top_k=3)
    assert result.empty
    assert result.retrieval_confidence == 0.0


def test_get_context_bounds_size():
    retriever = build_retriever()
    result = retriever.retrieve("research agent", top_k=3, similarity_threshold=0.0)
    context = retriever.get_context(result.chunks, max_chars=100)
    assert context
    assert sum(len(c["content"]) for c in context) <= 100


def test_apply_metadata_filter_client_side():
    retriever = build_retriever()
    chunks = [
        {"content": "x", "metadata": {"source": "a.pdf"}},
        {"content": "y", "metadata": {"source": "b.pdf"}},
    ]
    out = retriever.apply_metadata_filter(chunks, {"source": "a.pdf"})
    assert len(out) == 1
    assert out[0]["content"] == "x"