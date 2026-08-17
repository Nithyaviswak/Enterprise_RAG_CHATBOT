"""Application configuration loaded from environment variables."""

import os

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable loading."""

    # App
    app_name: str = "RAG Chatbot"
    app_version: str = "1.0.0"
    debug: bool = False
    rag_debug_mode: bool = False  # Expose retrieval internals to the API (dev only)
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Google Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 2048

    # RAGFlow (Plugin)
    ragflow_base_url: str = "http://localhost:9380"
    ragflow_api_key: str = ""

    # Embedding Model (Plugin: sentence-transformers)
    embedding_model: str = "all-MiniLM-L6-v2"
    finetuned_model_path: str = ""

    # Vector Store (ChromaDB)
    chroma_persist_dir: str = "./data/chromadb"
    chroma_collection_name: str = "documents"

    # Chat History (SQLite)
    database_url: str = "sqlite+aiosqlite:///./data/chat_history.db"

    # Rate Limiting
    rate_limit: str = "30/minute"

    # Document Storage
    upload_dir: str = "./data/uploads"

    # ── Chunking (configurable, not hard-coded) ──────────────────
    chunk_size: int = 512          # words per chunk
    chunk_overlap: int = 64        # overlapping words between chunks
    chunk_min_length: int = 20     # drop chunks below this many words

    # ── Retrieval ────────────────────────────────────────────────
    retrieval_top_k: int = 5       # final context size fed to the LLM
    retrieval_hybrid: bool = True  # combine semantic + BM25
    retrieval_semantic_weight: float = 0.7
    similarity_threshold: float = 0.30   # min cosine similarity for semantic hits
    min_context_chunks: int = 1          # chunks required to attempt an answer
    metadata_filter_enabled: bool = False

    # ── Re-ranking ───────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True
    rerank_top_k: int = 8          # candidates passed to the reranker

    # ── Guardrails / Confidence ──────────────────────────────────
    low_confidence_threshold: float = 0.40
    refusal_message: str = (
        "I don't have enough information in the provided documents to "
        "answer this reliably."
    )
    max_context_chars: int = 12000

    # ── Hallucination detection ──────────────────────────────────
    hallucination_check_enabled: bool = True
    hallucination_min_overlap: int = 2   # min keyword overlap to consider a claim grounded

    # ── Evaluation / release gating ──────────────────────────────
    evaluation_dataset: str = "evaluation/dataset.json"
    evaluation_thresholds: str = "evaluation/thresholds.yaml"
    evaluation_report_dir: str = "evaluation/reports"
    evaluation_llm_judge: bool = False  # use RAGAS/LLM judge metrics when configured

    # Firebase Auth (optional)
    firebase_credentials_path: str = ""

    # Neo4j Graph Database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme-password"
    neo4j_database: str = "neo4j"
    graph_embedding_model: str = "all-MiniLM-L6-v2"
    entity_extraction_model: str = "gemini-2.5-flash"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    def resolve_path(self, path: str) -> str:
        """Resolve a possibly-relative path against the backend directory.

        Relative paths in the shipped defaults (e.g. ``./data/chromadb``) are
        interpreted relative to the ``backend/`` folder so the app works the
        same whether launched from the repo root or from ``backend/``.
        """
        if os.path.isabs(path):
            return path
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.abspath(os.path.join(backend_dir, path))

    @property
    def repo_root(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        )

    @property
    def evaluation_dataset_path(self) -> str:
        return os.path.join(self.repo_root, self.evaluation_dataset)

    @property
    def evaluation_thresholds_path(self) -> str:
        return os.path.join(self.repo_root, self.evaluation_thresholds)

    @property
    def evaluation_report_dir_path(self) -> str:
        if os.path.isabs(self.evaluation_report_dir):
            return self.evaluation_report_dir
        return os.path.join(self.repo_root, self.evaluation_report_dir)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
