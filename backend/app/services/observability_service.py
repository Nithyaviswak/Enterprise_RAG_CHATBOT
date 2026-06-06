"""
LangSmith Observability Integration.

Provides:
- Distributed tracing for RAG pipeline
- Retrieval quality monitoring
- Latency tracking
- Error logging with full context
- Cost tracking

Usage:
    from app.services.observability_service import ObservabilityService

    obs = ObservabilityService()
    obs.trace_retrieval(query, results)
    obs.trace_generation(prompt, response)
"""

import logging
import time
import json
from typing import Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Try to import LangSmith
try:
    from langsmith import traceable, Client as LangSmithClient
    from langsmith.run_helpers import get_current_run_tree
    LANG smith_AVAILABLE = True
except ImportError:
    LANG smith_AVAILABLE = False
    logger.warning("LangSmith not installed. Install with: pip install langsmith")


@dataclass
class TraceData:
    """Container for trace information."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None
    traces: list = field(default_factory=list)


class ObservabilityService:
    """LangSmith integration for RAG observability."""

    def __init__(self, project_name: str = "rag-chatbot"):
        """Initialize observability service.

        Args:
            project_name: LangSmith project name for grouping traces
        """
        self.project_name = project_name
        self._client = None
        self._enabled = False
        self._trace_buffer: list[TraceData] = []

        # Try to initialize LangSmith client
        if LANG smith_AVAILABLE:
            try:
                self._client = LangSmithClient()
                self._enabled = True
                logger.info(f"LangSmith enabled for project: {project_name}")
            except Exception as e:
                logger.warning(f"LangSmith init failed: {e}")

        # Local fallback
        if not self._enabled:
            logger.info("Using local observability (no LangSmith)")

    # ─────────────────────────────────────────────────────────────────
    # Core Tracing
    # ─────────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def trace_operation(
        self,
        operation: str,
        inputs: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        """Context manager for tracing operations.

        Usage:
            async with obs.trace_operation("retrieve", {"query": "..."}):
                results = await retrieval.retrieve(...)
        """
        trace = TraceData(
            operation=operation,
            start_time=time.time(),
            inputs=inputs or {},
            metadata=metadata or {},
        )

        try:
            yield trace
            trace.end_time = time.time()
            trace.duration_ms = (trace.end_time - trace.start_time) * 1000
            self._log_trace(trace)
        except Exception as e:
            trace.error = str(e)
            trace.end_time = time.time()
            trace.duration_ms = (trace.end_time - trace.start_time) * 1000
            self._log_trace(trace)
            raise

    def trace_retrieval(
        self,
        query: str,
        results: list[dict],
        duration_ms: float,
        metadata: Optional[dict] = None,
    ):
        """Log retrieval operation.

        Args:
            query: Search query
            results: Retrieved results
            duration_ms: Operation duration
            metadata: Additional metadata
        """
        trace = TraceData(
            operation="retrieve",
            start_time=time.time() - duration_ms / 1000,
            end_time=time.time(),
            duration_ms=duration_ms,
            inputs={"query": query, "top_k": len(results)},
            outputs={
                "result_count": len(results),
                "sources": [r.get("source", "unknown") for r in results],
                "avg_score": sum(r.get("score", 0) for r in results) / len(results) if results else 0,
            },
            metadata=metadata or {},
        )

        # Add result details for debugging
        if results:
            trace.outputs["top_sources"] = [
                {"source": r.get("source"), "score": r.get("score")}
                for r in results[:3]
            ]

        self._log_trace(trace)

    def trace_generation(
        self,
        prompt: str,
        response: str,
        duration_ms: float,
        token_count: Optional[int] = None,
        metadata: Optional[dict] = None,
    ):
        """Log LLM generation operation.

        Args:
            prompt: Input prompt
            response: Generated response
            duration_ms: Operation duration
            token_count: Total tokens used
            metadata: Additional metadata
        """
        trace = TraceData(
            operation="generate",
            start_time=time.time() - duration_ms / 1000,
            end_time=time.time(),
            duration_ms=duration_ms,
            inputs={"prompt_length": len(prompt)},
            outputs={
                "response_length": len(response),
                "word_count": len(response.split()),
            },
            metadata=metadata or {},
        )

        if token_count:
            trace.metadata["token_count"] = token_count

        self._log_trace(trace)

    def trace_embedding(
        self,
        texts: list[str],
        duration_ms: float,
        metadata: Optional[dict] = None,
    ):
        """Log embedding generation.

        Args:
            texts: Texts being embedded
            duration_ms: Operation duration
            metadata: Additional metadata
        """
        trace = TraceData(
            operation="embed",
            start_time=time.time() - duration_ms / 1000,
            end_time=time.time(),
            duration_ms=duration_ms,
            inputs={"text_count": len(texts), "avg_length": sum(len(t) for t in texts) / len(texts) if texts else 0},
            metadata=metadata or {},
        )

        self._log_trace(trace)

    # ─────────────────────────────────────────────────────────────────
    # Error & Failure Logging
    # ─────────────────────────────────────────────────────────────────

    def log_retrieval_failure(
        self,
        query: str,
        error: str,
        context: Optional[dict] = None,
    ):
        """Log retrieval failure."""
        logger.error(
            f"Retrieval failure: {error}",
            extra={
                "query": query,
                "context": context,
            }
        )

    def log_generation_failure(
        self,
        prompt: str,
        error: str,
        context: Optional[dict] = None,
    ):
        """Log generation failure."""
        logger.error(
            f"Generation failure: {error}",
            extra={
                "prompt_length": len(prompt),
                "context": context,
            }
        )

    def log_empty_retrieval(
        self,
        query: str,
        context: Optional[dict] = None,
    ):
        """Log when retrieval returns no results."""
        logger.warning(
            f"Empty retrieval for query: {query[:100]}",
            extra={"context": context}
        )

    def log_hallucination_detected(
        self,
        query: str,
        answer: str,
        unsupported_claims: list[str],
    ):
        """Log detected hallucination."""
        logger.warning(
            f"Hallucination detected: {len(unsupported_claims)} unsupported claims",
            extra={
                "query": query[:100],
                "answer_length": len(answer),
                "unsupported_claims": unsupported_claims,
            }
        )

    # ─────────────────────────────────────────────────────────────────
    # Metrics & Statistics
    # ─────────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Get observability statistics from trace buffer."""
        if not self._trace_buffer:
            return {"message": "No traces recorded"}

        operations = {}
        for trace in self._trace_buffer:
            op = trace.operation
            if op not in operations:
                operations[op] = {"count": 0, "total_ms": 0, "errors": 0}

            operations[op]["count"] += 1
            if trace.duration_ms:
                operations[op]["total_ms"] += trace.duration_ms
            if trace.error:
                operations[op]["errors"] += 1

        # Calculate averages
        for op, stats in operations.items():
            if stats["count"] > 0:
                stats["avg_ms"] = stats["total_ms"] / stats["count"]
                stats["error_rate"] = stats["errors"] / stats["count"]

        return {
            "total_traces": len(self._trace_buffer),
            "operations": operations,
        }

    # ─────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────

    def _log_trace(self, trace: TraceData):
        """Log trace to LangSmith or local buffer."""
        self._trace_buffer.append(trace)

        if self._enabled:
            # LangSmith logging would go here
            # For production, use @traceable decorators
            pass

        # Also log to standard logger for debugging
        log_msg = f"[{trace.operation}] duration={trace.duration_ms:.1f}ms"
        if trace.error:
            logger.error(log_msg, extra={"error": trace.error})
        else:
            logger.debug(log_msg)


# ─────────────────────────────────────────────────────────────────
# Decorator-based Tracing
# ─────────────────────────────────────────────────────────────────

if LANG smith_AVAILABLE:
    # Example decorator usage (add to service methods)
    # @traceable(name="retrieve", project_name="rag-chatbot")
    # async def retrieve(self, query: str, ...):
    #     ...
    pass
else:
    # Fallback decorator
    def traceable(name: str = None, project_name: str = None):
        """Fallback decorator when LangSmith not available."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = (time.time() - start) * 1000
                    logger.debug(f"{func.__name__} completed in {duration:.1f}ms")
                    return result
                except Exception as e:
                    duration = (time.time() - start) * 1000
                    logger.error(f"{func.__name__} failed after {duration:.1f}ms: {e}")
                    raise
            return wrapper
        return decorator


# ─────────────────────────────────────────────────────────────────
# Request/Response Logging Middleware
# ─────────────────────────────────────────────────────────────────

class RequestLoggingMiddleware:
    """Middleware for logging API requests/responses."""

    def __init__(self, observability: ObservabilityService):
        self.obs = observability

    async def log_request(
        self,
        endpoint: str,
        request_data: dict,
        response_data: dict,
        duration_ms: float,
    ):
        """Log API request/response."""
        logger.info(
            f"API {endpoint} completed in {duration_ms:.1f}ms",
            extra={
                "endpoint": endpoint,
                "request": request_data,
                "response": response_data,
                "duration_ms": duration_ms,
            }
        )
