"""
RAGAS Evaluation Pipeline.

Comprehensive evaluation framework for measuring RAG system quality:
- Retrieval Metrics: Context Precision, Context Recall, Context Relevancy
- Generation Metrics: Faithfulness, Answer Relevancy, Response Correctness
- End-to-End Quality Scoring

Usage:
    from app.services.evaluation_service import EvaluationService

    evaluator = EvaluationService()
    results = await evaluator.evaluate(query, retrieved_context, generated_answer, ground_truth)
    print(results.metrics)
"""

import logging
import json
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    query: str
    retrieved_context: list[str]
    generated_answer: str
    ground_truth: Optional[str] = None

    # Retrieval metrics
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevancy: float = 0.0

    # Generation metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0

    # End-to-end
    response_correctness: float = 0.0
    overall_score: float = 0.0

    # Metadata
    evaluation_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "retrieved_context": self.retrieved_context,
            "generated_answer": self.generated_answer,
            "ground_truth": self.ground_truth,
            "metrics": {
                "retrieval": {
                    "context_precision": round(self.context_precision, 3),
                    "context_recall": round(self.context_recall, 3),
                    "context_relevancy": round(self.context_relevancy, 3),
                },
                "generation": {
                    "faithfulness": round(self.faithfulness, 3),
                    "answer_relevancy": round(self.answer_relevancy, 3),
                },
                "end_to_end": {
                    "response_correctness": round(self.response_correctness, 3),
                    "overall_score": round(self.overall_score, 3),
                },
            },
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
            "timestamp": self.timestamp,
        }


class EvaluationService:
    """RAGAS-based evaluation service for RAG systems."""

    def __init__(self, llm_service=None):
        """Initialize evaluation service.

        Args:
            llm_service: LLM service for generating evaluation signals
        """
        self.llm_service = llm_service
        self._eval_history: list[EvaluationResult] = []

        # Try to load ragas if available
        self._ragas_available = False
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_relevancy,
            )
            self._ragas_evaluate = evaluate
            self._ragas_faithfulness = faithfulness
            self._ragas_answer_relevancy = answer_relevancy
            self._ragas_context_precision = context_precision
            self._ragas_context_recall = context_recall
            self._ragas_context_relevancy = context_relevancy
            self._ragas_available = True
            logger.info("RAGAS library loaded successfully")
        except ImportError:
            logger.warning("RAGAS not installed. Using fallback evaluation.")

    async def evaluate(
        self,
        query: str,
        retrieved_context: list[str],
        generated_answer: str,
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate RAG system on a single sample.

        Args:
            query: User query
            retrieved_context: List of retrieved context strings
            generated_answer: Generated answer from LLM
            ground_truth: Optional ground truth answer

        Returns:
            EvaluationResult with all metrics
        """
        import time
        start_time = time.time()

        result = EvaluationResult(
            query=query,
            retrieved_context=retrieved_context,
            generated_answer=generated_answer,
            ground_truth=ground_truth,
        )

        # Try RAGAS first
        if self._ragas_available and self.llm_service:
            try:
                ragas_result = await self._evaluate_with_ragas(
                    query, retrieved_context, generated_answer, ground_truth
                )
                result.context_precision = ragas_result.get("context_precision", 0.0)
                result.context_recall = ragas_result.get("context_recall", 0.0)
                result.context_relevancy = ragas_result.get("context_relevancy", 0.0)
                result.faithfulness = ragas_result.get("faithfulness", 0.0)
                result.answer_relevancy = ragas_result.get("answer_relevancy", 0.0)
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed: {e}, using fallback")
                fallback_metrics = await self._evaluate_fallback(
                    query, retrieved_context, generated_answer, ground_truth
                )
                self._apply_fallback_metrics(result, fallback_metrics)
        else:
            # Fallback to LLM-based evaluation
            fallback_metrics = await self._evaluate_fallback(
                query, retrieved_context, generated_answer, ground_truth
            )
            self._apply_fallback_metrics(result, fallback_metrics)

        # Calculate overall score
        result.overall_score = self._calculate_overall_score(result)
        result.evaluation_time_ms = (time.time() - start_time) * 1000

        # Store in history
        self._eval_history.append(result)

        return result

    async def _evaluate_with_ragas(
        self,
        query: str,
        contexts: list[str],
        answer: str,
        ground_truth: Optional[str],
    ) -> dict:
        """Evaluate using RAGAS library."""
        from datasets import Dataset

        # Prepare data for RAGAS
        data = {
            "question": [query],
            "contexts": [contexts],
            "answer": [answer],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Run evaluation
        # Note: This requires an LLM to be configured in ragas
        # For production, set OPENAI_API_KEY or use custom evaluator
        result = self._ragas_evaluate(
            dataset=dataset,
            metrics=[
                self._ragas_faithfulness,
                self._ragas_answer_relevancy,
                self._ragas_context_precision,
                self._ragas_context_recall,
            ],
        )

        return result.to_dict()

    async def _evaluate_fallback(
        self,
        query: str,
        contexts: list[str],
        answer: str,
        ground_truth: Optional[str],
    ) -> dict:
        """Fallback evaluation using LLM when RAGAS unavailable."""
        if not self.llm_service:
            return self._default_metrics()

        # Combine context for analysis
        context_combined = "\n\n---\n\n".join(
            f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
        )

        # Evaluation prompt
        eval_prompt = f"""You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

Evaluate the following RAG response on a scale of 0-1 for each metric:

## User Query:
{query}

## Retrieved Context:
{context_combined}

## Generated Answer:
{answer}

{f"## Ground Truth:\n{ground_truth}" if ground_truth else ""}

## Metrics to Evaluate:

1. **Context Precision**: Are the retrieved contexts relevant to the query? (0-1)
2. **Context Recall**: Does the context contain the information needed to answer? (0-1)
3. **Context Relevancy**: Is the retrieved information focused and non-redundant? (0-1)
4. **Faithfulness**: Does the answer only make claims supported by the context? (0-1)
5. **Answer Relevancy**: Is the answer directly relevant to the user's question? (0-1)
{f"6. **Response Correctness**: Does the answer match the ground truth? (0-1)" if ground_truth else ""}

Return ONLY a JSON object with these exact keys:
```json
{{
  "context_precision": 0.0-1.0,
  "context_recall": 0.0-1.0,
  "context_relevancy": 0.0-1.0,
  "faithfulness": 0.0-1.0,
  "answer_relevancy": 0.0-1.0
  {"\"response_correctness\": 0.0-1.0" if ground_truth else ""}
}}
```"""

        try:
            response = await self.llm_service.chat(
                message=eval_prompt,
                context=None,
            )

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                metrics = json.loads(json_match.group())
                return metrics
        except Exception as e:
            logger.warning(f"Fallback evaluation failed: {e}")

        return self._default_metrics()

    def _apply_fallback_metrics(self, result: EvaluationResult, metrics: dict):
        """Apply fallback metrics to result."""
        result.context_precision = metrics.get("context_precision", 0.0)
        result.context_recall = metrics.get("context_recall", 0.0)
        result.context_relevancy = metrics.get("context_relevancy", 0.0)
        result.faithfulness = metrics.get("faithfulness", 0.0)
        result.answer_relevancy = metrics.get("answer_relevancy", 0.0)
        result.response_correctness = metrics.get("response_correctness", 0.0)

    def _default_metrics(self) -> dict:
        """Default metrics when evaluation fails."""
        return {
            "context_precision": 0.5,
            "context_recall": 0.5,
            "context_relevancy": 0.5,
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "response_correctness": 0.5,
        }

    def _calculate_overall_score(self, result: EvaluationResult) -> float:
        """Calculate weighted overall score."""
        weights = {
            "retrieval": 0.4,  # Retrieval quality matters
            "generation": 0.4,  # Generation quality matters
            "correctness": 0.2,  # Correctness (if available)
        }

        retrieval_score = (
            result.context_precision * 0.4 +
            result.context_recall * 0.3 +
            result.context_relevancy * 0.3
        )

        generation_score = (
            result.faithfulness * 0.6 +  # Faithfulness is critical
            result.answer_relevancy * 0.4
        )

        overall = (
            retrieval_score * weights["retrieval"] +
            generation_score * weights["generation"]
        )

        if result.ground_truth:
            overall = overall * 0.8 + result.response_correctness * 0.2

        return overall

    # ─────────────────────────────────────────────────────────────────
    # Batch Evaluation
    # ─────────────────────────────────────────────────────────────────

    async def evaluate_batch(
        self,
        samples: list[dict],
    ) -> list[EvaluationResult]:
        """Evaluate multiple samples in batch.

        Args:
            samples: List of dicts with keys: query, context, answer, ground_truth

        Returns:
            List of EvaluationResult
        """
        tasks = [
            self.evaluate(
                query=sample["query"],
                retrieved_context=sample["context"],
                generated_answer=sample["answer"],
                ground_truth=sample.get("ground_truth"),
            )
            for sample in samples
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, EvaluationResult)]
        logger.info(f"Evaluated {len(valid_results)}/{len(samples)} samples")

        return valid_results

    # ─────────────────────────────────────────────────────────────────
    # Statistics & Reporting
    # ─────────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Get evaluation statistics."""
        if not self._eval_history:
            return {"message": "No evaluations run yet"}

        metrics = {
            "context_precision": [],
            "context_recall": [],
            "context_relevancy": [],
            "faithfulness": [],
            "answer_relevancy": [],
            "overall_score": [],
        }

        for result in self._eval_history:
            metrics["context_precision"].append(result.context_precision)
            metrics["context_recall"].append(result.context_recall)
            metrics["context_relevancy"].append(result.context_relevancy)
            metrics["faithfulness"].append(result.faithfulness)
            metrics["answer_relevancy"].append(result.answer_relevancy)
            metrics["overall_score"].append(result.overall_score)

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            "total_evaluations": len(self._eval_history),
            "metrics": {
                "context_precision": {"mean": round(avg(metrics["context_precision"]), 3)},
                "context_recall": {"mean": round(avg(metrics["context_recall"]), 3)},
                "context_relevancy": {"mean": round(avg(metrics["context_relevancy"]), 3)},
                "faithfulness": {"mean": round(avg(metrics["faithfulness"]), 3)},
                "answer_relevancy": {"mean": round(avg(metrics["answer_relevancy"]), 3)},
                "overall_score": {"mean": round(avg(metrics["overall_score"]), 3)},
            },
        }

    def export_results(self, format: str = "json") -> str:
        """Export evaluation results.

        Args:
            format: Output format (json, csv)

        Returns:
            Formatted results string
        """
        if format == "json":
            return json.dumps(
                [r.to_dict() for r in self._eval_history],
                indent=2,
            )
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            if self._eval_history:
                fields = ["query", "context_precision", "context_recall",
                         "faithfulness", "answer_relevancy", "overall_score"]
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()

                for r in self._eval_history:
                    writer.writerow({
                        "query": r.query[:50] + "...",
                        "context_precision": r.context_precision,
                        "context_recall": r.context_recall,
                        "faithfulness": r.faithfulness,
                        "answer_relevancy": r.answer_relevancy,
                        "overall_score": r.overall_score,
                    })

            return output.getvalue()

        return str(self._eval_history)

    def clear_history(self):
        """Clear evaluation history."""
        self._eval_history.clear()
        logger.info("Evaluation history cleared")


# ─────────────────────────────────────────────────────────────────
# Synthetic Dataset Generator
# ─────────────────────────────────────────────────────────────────

class SyntheticDatasetGenerator:
    """Generate synthetic evaluation datasets from documents."""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    async def generate_evaluation_set(
        self,
        documents: list[dict],
        num_questions_per_doc: int = 5,
    ) -> list[dict]:
        """Generate synthetic QA pairs from documents.

        Args:
            documents: List of dicts with 'content' and 'metadata'
            num_questions_per_doc: Questions to generate per document

        Returns:
            List of QA pairs with context
        """
        if not self.llm_service:
            logger.warning("LLM service required for dataset generation")
            return []

        generated_samples = []

        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "Unknown")

            prompt = f"""Generate {num_questions_per_doc} diverse questions that can be answered using the following document.

Generate questions of different types:
- Factual questions (who, what, when, where)
- How-to questions
- Explanation questions
- Comparison questions

## Document:
{content[:2000]}...  # Truncate for prompt length

Return ONLY a JSON array of objects with this exact format:
```json
[
  {{"question": "question text", "answer": "expected answer"}},
  ...
]
```"""

            try:
                response = await self.llm_service.chat(
                    message=prompt,
                    context=None,
                )

                # Parse response
                import re
                import json
                json_match = re.search(r'\[[^\]]+\]', response, re.DOTALL)
                if json_match:
                    qa_pairs = json.loads(json_match.group())

                    for qa in qa_pairs:
                        generated_samples.append({
                            "query": qa["question"],
                            "context": [content],  # The document is the context
                            "answer": qa["answer"],
                            "ground_truth": qa["answer"],
                            "source": source,
                        })
            except Exception as e:
                logger.warning(f"Failed to generate questions for {source}: {e}")

        logger.info(f"Generated {len(generated_samples)} synthetic QA pairs")
        return generated_samples
