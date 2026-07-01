"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable loading."""

    # App
    app_name: str = "RAG Chatbot"
    app_version: str = "1.0.0"
    debug: bool = False
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Google Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

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

    # Re-ranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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


@lru_cache()
def get_settings() -> Settings:
    return Settings()
