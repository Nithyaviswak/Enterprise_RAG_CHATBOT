"""Shared pytest fixtures: deterministic, offline fakes for the RAG stack.

No ChromaDB, no HuggingFace models, no network — every test is hermetic and
fast so the suite runs in CI without secrets or downloads.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.rag.reranker import Reranker  # noqa: E402


class FakeEmbeddingService:
    """Deterministic feature-hash embeddings; cosine ~ term overlap."""

    def __init__(self, dim: int = 12):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in text.lower().split():
            h = 0
            for ch in token:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            for d in range(self.dim):
                bit = (h >> d) & 1
                v[d] += 1.0 if bit else -1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def encode_query(self, query: str, **kwargs) -> list[float]:
        return self._vector(query)

    def encode_documents(self, documents: list[str], **kwargs) -> list[list[float]]:
        return [self._vector(d) for d in documents]


class FakeCollection:
    """Minimal ChromaDB-compatible surface used by the retriever."""

    def __init__(self, docs: list[dict]):
        self._docs = docs  # each: {"id", "content", "metadata", "embedding"}
        self._emb = FakeEmbeddingService()

    def count(self) -> int:
        return len(self._docs)

    def get(self, limit=None):
        docs = self._docs[:limit] if limit else self._docs
        return {
            "ids": [d["id"] for d in docs],
            "documents": [d["content"] for d in docs],
            "metadatas": [d["metadata"] for d in docs],
        }

    def query(self, query_embeddings, n_results, where=None):
        results = []
        for d in self._docs:
            meta = d["metadata"]
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            sim = _cosine(query_embeddings[0], d["embedding"])
            results.append((sim, d))
        results.sort(key=lambda t: t[0], reverse=True)
        return {
            "ids": [[r[1]["id"] for r in results[:n_results]]],
            "documents": [[r[1]["content"] for r in results[:n_results]]],
            "metadatas": [[r[1]["metadata"] for r in results[:n_results]]],
            "distances": [[1 - r[0] for r in results[:n_results]]],
        }


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    an = sum(x * x for x in a) ** 0.5 or 1.0
    bn = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (an * bn)


class FakeVectorStore:
    """Mimics VectorStoreService.query() + .collection for the retriever."""

    def __init__(self, docs: list[dict], embedding_service=None):
        self.embedding_service = embedding_service or FakeEmbeddingService()
        for d in docs:
            d.setdefault("embedding", self.embedding_service.encode_query(d["content"]))
        self.collection = FakeCollection(docs)

    def query(self, query_embedding, n_results: int = 5, where=None) -> list[dict]:
        raw = self.collection.query([query_embedding], n_results, where=where)
        out = []
        if raw["documents"]:
            for i, doc in enumerate(raw["documents"][0]):
                meta = raw["metadatas"][0][i] if raw["metadatas"] else {}
                out.append({
                    "content": doc,
                    "source": meta.get("source", "Unknown"),
                    "score": 1 - (raw["distances"][0][i] if raw["distances"] else 0),
                    "metadata": meta,
                })
        return out

    def add_documents(self, documents, embeddings, metadatas, ids):
        raise NotImplementedError("Fake store is read-only for tests")


def make_docs(contents: list[str], source: str = "test.txt") -> list[dict]:
    emb = FakeEmbeddingService()
    return [
        {"id": f"chunk{i}", "content": c, "metadata": {"source": source, "page": i}}
        for i, c in enumerate(contents)
    ]


def make_reranker(enabled: bool = True, score_fn=None) -> Reranker:
    return Reranker(embedding_service=FakeEmbeddingService(), enabled=enabled, score_fn=score_fn)


@pytest.fixture(autouse=True)
def _no_cross_encoder(monkeypatch):
    """Keep tests hermetic: never load the real cross-encoder model.

    The reranker falls back to deterministic cosine scoring instead.
    """
    monkeypatch.setattr(Reranker, "_load_cross_encoder", lambda self: False)


@pytest.fixture
def embedding_service():
    return FakeEmbeddingService()


@pytest.fixture
def store():
    """Vector store seeded with a small deterministic corpus."""
    contents = [
        "The Autonomous AI Research Agent uses LangGraph and Google Gemini for deep web research.",
        "It handles API failures with a self-healing failover system that detects 429 errors.",
        "Nithyananda Chari R is pursuing B.Tech in AI and Machine Learning with a CGPA of 8.44.",
    ]
    return FakeVectorStore(make_docs(contents))
