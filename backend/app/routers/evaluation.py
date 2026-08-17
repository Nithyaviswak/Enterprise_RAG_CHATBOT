"""
Evaluation Router — RAGAS Evaluation Endpoints.

Provides API endpoints for:
- Running single evaluation
- Batch evaluation with synthetic data
- Evaluation statistics and reporting
- Comparison before/after improvements
"""

import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.services.evaluation_service import EvaluationService, SyntheticDatasetGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


# ─────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────

class EvaluationSample(BaseModel):
    """Single evaluation sample."""
    query: str
    context: List[str]
    answer: str
    ground_truth: Optional[str] = None


class EvaluationRequest(BaseModel):
    """Evaluation request for single sample."""
    query: str
    retrieved_context: List[dict]
    generated_answer: str
    ground_truth: Optional[str] = None


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation request."""
    samples: List[EvaluationSample]


class EvaluationResponse(BaseModel):
    """Evaluation response."""
    metrics: dict
    evaluation_time_ms: float


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

def get_evaluation_service(request: Request) -> EvaluationService:
    """Get or create evaluation service."""
    if not hasattr(request.app.state, "evaluation_service"):
        # Try to get LLM service for evaluation
        llm_service = getattr(request.app.state, "gemini_service", None)
        request.app.state.evaluation_service = EvaluationService(llm_service)
    return request.app.state.evaluation_service


@router.post("/evaluate")
async def evaluate_sample(request: Request, body: EvaluationRequest):
    """Evaluate a single RAG sample.

    Measures:
    - Context Precision, Recall, Relevancy (retrieval)
    - Faithfulness, Answer Relevancy (generation)
    - Overall quality score
    """
    eval_service = get_evaluation_service(request)

    # Convert context dicts to strings
    context_strings = [c.get("content", "") for c in body.retrieved_context]

    result = await eval_service.evaluate(
        query=body.query,
        retrieved_context=context_strings,
        generated_answer=body.generated_answer,
        ground_truth=body.ground_truth,
    )

    return result.to_dict()


@router.post("/evaluate/batch")
async def evaluate_batch(request: Request, body: BatchEvaluationRequest):
    """Evaluate multiple RAG samples in batch."""
    eval_service = get_evaluation_service(request)

    # Convert samples to evaluation format
    samples = [
        {
            "query": s.query,
            "context": s.context,
            "answer": s.answer,
            "ground_truth": s.ground_truth,
        }
        for s in body.samples
    ]

    results = await eval_service.evaluate_batch(samples)

    # Calculate aggregate statistics
    total = len(results)
    if total == 0:
        return {"message": "No valid results", "count": 0}

    avg_metrics = {
        "context_precision": sum(r.context_precision for r in results) / total,
        "context_recall": sum(r.context_recall for r in results) / total,
        "context_relevancy": sum(r.context_relevancy for r in results) / total,
        "faithfulness": sum(r.faithfulness for r in results) / total,
        "answer_relevancy": sum(r.answer_relevancy for r in results) / total,
        "overall_score": sum(r.overall_score for r in results) / total,
    }

    return {
        "count": total,
        "metrics": {k: round(v, 3) for k, v in avg_metrics.items()},
        "individual_results": [r.to_dict() for r in results],
    }


@router.get("/statistics")
async def get_statistics(request: Request):
    """Get evaluation statistics."""
    eval_service = get_evaluation_service(request)
    return eval_service.get_statistics()


@router.get("/metrics/live")
async def get_live_metrics(request: Request):
    """Get live runtime metrics from the RAG metrics store (dashboard)."""
    from app.observability.tracing import MetricsStore

    store = MetricsStore.get()
    return store.summarize()


@router.get("/results")
async def get_results(request: Request, format: str = "json"):
    """Get all evaluation results.

    Args:
        format: Output format (json or csv)
    """
    eval_service = get_evaluation_service(request)
    return eval_service.export_results(format=format)


@router.post("/results/clear")
async def clear_results(request: Request):
    """Clear evaluation history."""
    eval_service = get_evaluation_service(request)
    eval_service.clear_history()
    return {"status": "cleared"}


# ─────────────────────────────────────────────────────────────────
# Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────

@router.post("/generate-dataset")
async def generate_dataset(
    request: Request,
    num_questions: int = 10,
):
    """Generate synthetic evaluation dataset from uploaded documents.

    Creates QA pairs from document content for evaluation.
    """
    llm_service = getattr(request.app.state, "gemini_service", None)
    if not llm_service:
        raise HTTPException(status_code=500, detail="LLM service not available")

    # Get documents from vector store
    vector_store = getattr(request.app.state, "vector_store", None)
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not available")

    # Get all documents
    try:
        all_docs = vector_store.collection.get(limit=1000)
        documents = [
            {
                "content": doc,
                "metadata": meta or {},
            }
            for doc, meta in zip(
                all_docs.get("documents", []),
                all_docs.get("metadatas", [{}]),
            )
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {e}")

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")

    # Generate synthetic dataset
    generator = SyntheticDatasetGenerator(llm_service)
    samples = await generator.generate_evaluation_set(
        documents=documents,
        num_questions_per_doc=num_questions // len(documents) + 1,
    )

    return {
        "count": len(samples),
        "samples": samples[:num_questions],  # Limit to requested count
    }


@router.get("/benchmark")
async def run_benchmark(request: Request):
    """Run built-in benchmark on common RAG queries.

    Returns benchmark results comparing retrieval and generation quality.
    """
    # Predefined benchmark queries
    benchmark_queries = [
        "What is the main topic of the documents?",
        "Summarize the key points",
        "What are the main conclusions?",
    ]

    eval_service = get_evaluation_service(request)
    results = []

    for query in benchmark_queries:
        # Retrieve context
        retrieval_service = getattr(request.app.state, "retrieval_service", None)
        if not retrieval_service:
            continue

        try:
            retrieved = await retrieval_service.retrieve(query=query, top_k=3)
            context_strings = [r.get("content", "") for r in retrieved]

            # Get LLM answer
            gemini_service = getattr(request.app.state, "gemini_service", None)
            if not gemini_service:
                continue

            answer = await gemini_service.chat(message=query, context=retrieved)

            # Evaluate
            result = await eval_service.evaluate(
                query=query,
                retrieved_context=context_strings,
                generated_answer=answer,
            )

            results.append(result.to_dict())
        except Exception as e:
            logger.warning(f"Benchmark query failed: {e}")

    # Calculate averages
    if results:
        avg_score = sum(r["metrics"]["end_to_end"]["overall_score"] for r in results) / len(results)
        return {
            "query_count": len(results),
            "average_score": round(avg_score, 3),
            "results": results,
        }

    return {"message": "Benchmark failed", "results": []}
