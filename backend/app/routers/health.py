"""Health Check Router."""

from fastapi import APIRouter, Request
from app.config import get_settings

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health_check(request: Request):
    """Check service health and connected plugins."""
    settings = get_settings()
    services = {}

    # Check Gemini
    services["gemini"] = "configured" if settings.gemini_api_key else "not_configured"

    # Check RAGFlow
    ragflow = request.app.state.ragflow_client
    if ragflow.is_available:
        try:
            is_healthy = await ragflow.health_check()
            services["ragflow"] = "connected" if is_healthy else "unreachable"
        except Exception:
            services["ragflow"] = "unreachable"
    else:
        services["ragflow"] = "not_configured"

    # Check ChromaDB
    try:
        stats = request.app.state.vector_store.get_stats()
        services["chromadb"] = f"connected ({stats['document_count']} docs)"
    except Exception:
        services["chromadb"] = "error"

    # Check Embedding Model
    try:
        info = request.app.state.embedding_service.get_model_info()
        services["embeddings"] = f"loaded ({info['model_name']})"
    except Exception:
        services["embeddings"] = "error"

    return {
        "status": "healthy",
        "version": settings.app_version,
        "services": services,
    }
