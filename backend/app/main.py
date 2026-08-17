"""
RAG Chatbot — FastAPI Backend Entry Point.

Production-grade RAG chatbot powered by:
- Google Gemini API (LLM)
- RAGFlow (Document parsing & retrieval plugin)
- Sentence-Transformers (Embedding plugin)
- ChromaDB (Vector store)
- LlamaIndex Fine-tuning (Embedding optimization plugin)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.models.database import init_db
from app.observability.tracing import MetricsStore
from app.rag.generator import Generator, PromptBuilder
from app.rag.guardrails import Guardrails
from app.rag.hallucination import HallucinationDetector
from app.rag.pipeline import RagPipeline
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.services.gemini_service import GeminiService
from app.services.ragflow_client import RAGFlowClient
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService
from app.services.knowledge_graph import (
    GraphService,
    EntityExtractor,
    RelationshipExtractor,
    GraphRetriever,
    GraphEmbeddingService,
)
from app.routers import chat, documents, evaluation, finetune, health, graph
from app.middleware.rate_limit import limiter, rate_limit_handler
from app.middleware.auth import FirebaseAuthMiddleware

# ── Logging Setup ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Application Lifespan ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown application services."""
    settings = get_settings()
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")

    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs(settings.resolve_path(settings.upload_dir), exist_ok=True)
    os.makedirs(settings.resolve_path(settings.chroma_persist_dir), exist_ok=True)

    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")

    # Initialize services
    # 1. Embedding service (Plugin: sentence-transformers)
    embedding_service = EmbeddingService()
    app.state.embedding_service = embedding_service
    logger.info("✅ Embedding service ready")

    # 2. Vector store (ChromaDB)
    vector_store = VectorStoreService(embedding_service=embedding_service)
    app.state.vector_store = vector_store
    logger.info("✅ Vector store ready")

    # 3. RAGFlow client (Plugin: ragflow)
    ragflow_client = RAGFlowClient()
    app.state.ragflow_client = ragflow_client
    if ragflow_client.is_available:
        logger.info("✅ RAGFlow client ready")
    else:
        logger.warning("⚠️  RAGFlow not configured — using local retrieval only")

    # 4. Retrieval service (hybrid search + re-ranking)
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        ragflow_client=ragflow_client,
    )
    app.state.retrieval_service = retrieval_service
    logger.info("✅ Retrieval service ready")

    # 5. Gemini service (LLM)
    gemini_service = GeminiService()
    app.state.gemini_service = gemini_service
    logger.info("✅ Gemini service ready")

    # 5b. Production RAG pipeline (retrieval → guardrails → generation → verify)
    rag_retriever = Retriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        ragflow_client=ragflow_client,
        reranker=Reranker(embedding_service=embedding_service),
    )
    rag_pipeline = RagPipeline(
        retriever=rag_retriever,
        generator=Generator(gemini_service, PromptBuilder()),
        guardrails=Guardrails(),
        hallucination_detector=HallucinationDetector(),
        prompt_builder=PromptBuilder(),
    )
    app.state.rag_pipeline = rag_pipeline
    app.state.metrics_store = MetricsStore.get()
    logger.info("✅ Production RAG pipeline ready")

    # 6. Knowledge Graph services
    try:
        graph_service = GraphService()
        await graph_service.initialize()
        app.state.graph_service = graph_service
        logger.info("✅ Knowledge graph service ready")
    except Exception as e:
        logger.warning(f"⚠️  Neo4j not available — graph features disabled: {e}")
        app.state.graph_service = None

    entity_extractor = EntityExtractor(gemini_service=gemini_service)
    app.state.entity_extractor = entity_extractor
    logger.info("✅ Entity extractor ready")

    relationship_extractor = RelationshipExtractor(gemini_service=gemini_service)
    app.state.relationship_extractor = relationship_extractor
    logger.info("✅ Relationship extractor ready")

    graph_embeddings = GraphEmbeddingService(
        embedding_service=embedding_service,
        graph_service=graph_service,
    )
    app.state.graph_embeddings = graph_embeddings

    graph_retriever = GraphRetriever(
        graph_service=graph_service,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    app.state.graph_retriever = graph_retriever
    logger.info("✅ Graph retriever ready")

    # 7. Chat service (orchestration)
    chat_service = ChatService(
        gemini_service=gemini_service,
        retrieval_service=retrieval_service,
        graph_retriever=graph_retriever,
        rag_pipeline=rag_pipeline,
        metrics_store=app.state.metrics_store,
    )
    app.state.chat_service = chat_service
    logger.info("✅ Chat service ready")

    logger.info("=" * 60)
    logger.info(f"🤖 {settings.app_name} is ready at http://localhost:8000")
    logger.info(f"📚 API docs at http://localhost:8000/docs")
    logger.info("=" * 60)

    yield

    # Cleanup
    graph_service = getattr(app.state, "graph_service", None)
    if graph_service:
        await graph_service.close()
    logger.info("Shutting down services...")


# ── FastAPI Application ───────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "Production-grade RAG chatbot API powered by Google Gemini, "
        "RAGFlow, Sentence-Transformers, and ChromaDB."
    ),
    version=get_settings().app_version,
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# Firebase Auth (optional — only active if configured)
app.add_middleware(FirebaseAuthMiddleware)

# ── Routers ───────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(finetune.router)
app.include_router(graph.router)
app.include_router(evaluation.router)


# ── Root Redirect ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "RAG Chatbot API",
        "version": settings.app_version,
        "docs": "/docs",
    }
