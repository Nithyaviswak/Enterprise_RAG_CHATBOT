"""Run the RAG evaluation suite and gate on release thresholds.

Usage (from repo root, with ``backend/venv`` active):
    backend/venv/Scripts/python.exe evaluation/evaluate.py
    backend/venv/Scripts/python.exe evaluation/evaluate.py --framework ragas
    backend/venv/Scripts/python.exe evaluation/evaluate.py --min-faithfulness 0.85

The runner loads ``evaluation/dataset.json`` and the corpus from the live
ChromaDB collection, runs every question through the real ``RagPipeline``,
scores each sample with the deterministic metrics, aggregates a scoreboard,
and fails (non-zero exit) when any gate is breached. A JSON report is written
under ``evaluation/reports/`` with a timestamp.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

import yaml  # noqa: E402
import tqdm  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.failures import FailureType  # noqa: E402
from app.rag.generator import GenerationResult, Generator, PromptBuilder  # noqa: E402
from app.rag.guardrails import Guardrails  # noqa: E402
from app.rag.hallucination import HallucinationDetector  # noqa: E402
from app.rag.pipeline import RagPipeline, RagResult  # noqa: E402
from app.rag.reranker import Reranker  # noqa: E402
from app.rag.retriever import RetrievalResult, Retriever  # noqa: E402
from app.services.gemini_service import GeminiService  # noqa: E402
from app.services.vector_store import VectorStoreService  # noqa: E402
from evaluation.metrics import Metrics, aggregate  # noqa: E402


class _OfflineRetriever:
    """Deterministic retriever used by ``--offline`` (CI gate demo).

    Maps each dataset question to its reference answer as the retrieved chunk,
    so retrieval + metrics run hermetically with no network. Correctly returns
    no chunks for out-of-corpus questions so the guardrail path is exercised.
    """

    def __init__(self, samples: list[dict]):
        self._by_question = {}
        for s in samples:
            if s.get("expected_in_corpus") and s.get("reference_answer"):
                self._by_question[s["question"].lower().strip()] = s["reference_answer"]

    def retrieve(self, query, **kwargs) -> RetrievalResult:
        ref = self._by_question.get(query.lower().strip())
        if not ref:
            return RetrievalResult(chunks=[], retrieval_confidence=0.0, methods_used=["offline"])
        return RetrievalResult(
            chunks=[{
                "id": "ref-chunk-0",
                "content": ref,
                "source": "reference-answer.txt",
                "metadata": {"page": 1},
                "score": 0.9,
            }],
            retrieval_confidence=0.9,
            methods_used=["offline"],
        )

    def get_context(self, chunks, max_chars=None):
        return chunks


class _OfflineEmbedding:
    """Deterministic feature-hash embeddings for hermetic CI evaluation."""

    def __init__(self, dim: int = 12):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in text.lower().split():
            h = 0
            for ch in token:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            for d in range(self.dim):
                v[d] += 1.0 if ((h >> d) & 1) else -1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class _OfflineGenerator:
    """Deterministic generator: echoes the retrieved context (grounded by construction)."""

    async def generate(self, query, contexts, history=None) -> GenerationResult:
        if not contexts:
            return GenerationResult(
                answer="",
                system_prompt="",
                generation_failure=True,
                failure_reason="empty context",
            )
        content = contexts[0]["content"]
        first_sentence = re.split(r"(?<=[.!?])\s+", content.strip())[0][:160]
        return GenerationResult(answer=first_sentence, system_prompt="<offline>", latency_ms=1.0, model="offline")


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["samples"]


def load_thresholds(path: Path, overrides: dict) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        thresholds = yaml.safe_load(fh) or {}
    thresholds.update({k: v for k, v in overrides.items() if v is not None})
    return thresholds


def run_sample(pipeline: RagPipeline, metrics: Metrics, sample: dict) -> dict:
    question = sample["question"]
    result: RagResult = asyncio.run(pipeline.run(question, use_ragflow=False))
    contexts = [c for c in result.contexts]
    generated = result.answer if result.answered else (result.answer or "")

    computed = metrics.compute_sample(
        question=question,
        reference_answer=sample.get("reference_answer", ""),
        contexts=contexts,
        generated_answer=generated,
        expected_in_corpus=sample.get("expected_in_corpus", True),
    )
    computed.update(
        {
            "id": sample["id"],
            "category": sample.get("category", "other"),
            "answered": result.answered,
            "refused": result.refused,
            "failure_type": result.failure_type,
            "confidence": result.confidence.get("overall_confidence", 0.0)
            if isinstance(result.confidence, dict) else 0.0,
        }
    )
    return computed


def print_scoreboard(summary: dict, thresholds: dict) -> str:
    lines = [f"  faithfulness        {summary['faithfulness']:.3f}  (gate >= {thresholds['faithfulness_min']})",
             f"  answer_relevancy    {summary['answer_relevancy']:.3f}  (gate >= {thresholds['answer_relevancy_min']})",
             f"  context_precision   {summary['context_precision']:.3f}  (gate >= {thresholds['context_precision_min']})",
             f"  context_recall      {summary['context_recall']:.3f}  (gate >= {thresholds['context_recall_min']})",
             f"  hallucination_rate  {summary['hallucination_rate']:.3f}  (gate <= {thresholds['hallucination_rate_max']})",
             f"  refusal_rate        {summary['refusal_rate']:.3f}",
             f"  retrieval_success   {summary['retrieval_success_rate']:.3f}",
             f"  avg_grounding       {summary['avg_grounding_ratio']:.3f}"]
    return "\n".join(lines)


def check_gates(summary: dict, thresholds: dict) -> list[str]:
    checks = [
        ("faithfulness", summary["faithfulness"] >= thresholds["faithfulness_min"], ">=", thresholds["faithfulness_min"]),
        ("answer_relevancy", summary["answer_relevancy"] >= thresholds["answer_relevancy_min"], ">=", thresholds["answer_relevancy_min"]),
        ("context_precision", summary["context_precision"] >= thresholds["context_precision_min"], ">=", thresholds["context_precision_min"]),
        ("context_recall", summary["context_recall"] >= thresholds["context_recall_min"], ">=", thresholds["context_recall_min"]),
        ("hallucination_rate", summary["hallucination_rate"] <= thresholds["hallucination_rate_max"], "<=", thresholds["hallucination_rate_max"]),
    ]
    return [f"  FAIL {name}: {summary[name]:.3f} {op} {limit:.3f}" for name, ok, op, limit in checks if not ok]


def write_report(summary: dict, samples: list[dict], fail: list[str], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {
        "timestamp": stamp,
        "gates_failed": fail,
        "passed": not fail,
        "summary": summary,
        "samples": samples,
    }
    path = report_dir / f"report-{stamp}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation and gate on thresholds.")
    parser.add_argument("--framework", choices=["local", "ragas"], default="local",
                        help="Metric backend (ragas requires RAGAS + a configured LLM judge).")
    parser.add_argument("--offline", action="store_true",
                        help="Use deterministic stub retriever/generator (no API, no network). "
                             "Exercises the harness, metrics and gates hermetically for CI.")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N samples.")
    parser.add_argument("--min-faithfulness", type=float, default=None)
    parser.add_argument("--min-relevancy", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--max-hallucination", type=float, default=None)
    parser.add_argument("--thresholds", default=None, help="Path to thresholds yaml (overrides settings).")
    parser.add_argument("--sample-delay", type=float, default=0.0,
                        help="Seconds to sleep between samples to respect per-minute LLM quota "
                             "(e.g. free tier is 5 req/min; use ~12 for a 24-sample run).")
    args = parser.parse_args()

    settings = get_settings()

    dataset_path = Path(settings.evaluation_dataset_path)
    thresholds_path = Path(settings.evaluation_thresholds_path)
    if args.offline:
        offline_thresholds = Path(settings.evaluation_dataset_path).parent / "thresholds.offline.yaml"
        if (Path(settings.evaluation_thresholds_path).parent / "thresholds.offline.yaml").exists():
            thresholds_path = Path(settings.evaluation_thresholds_path).parent / "thresholds.offline.yaml"

    samples = load_dataset(dataset_path)
    if args.limit:
        samples = samples[: args.limit]

    thresholds = load_thresholds(
        thresholds_path,
        {
            "faithfulness_min": args.min_faithfulness,
            "answer_relevancy_min": args.min_relevancy,
            "context_precision_min": args.min_precision,
            "context_recall_min": args.min_recall,
            "hallucination_rate_max": args.max_hallucination,
        },
    )

    if args.framework == "ragas":
        raise SystemExit(
            "RAGAS framework path is not wired up yet: install ragas and a langchain-compatible "
            "LLM judge (e.g. langchain-google-genai), then point RAGAS_LLM at it. Use --framework local for now."
        )

    print(f"[evaluate] loading {len(samples)} samples from {dataset_path}")
    if args.offline:
        print("[evaluate] OFFLINE mode — deterministic stubs, no API, no network")
        pipeline = RagPipeline(
            retriever=_OfflineRetriever(samples),
            generator=_OfflineGenerator(),
            guardrails=Guardrails(),
            hallucination_detector=HallucinationDetector(),
            prompt_builder=PromptBuilder(),
        )
        embedding_service = _OfflineEmbedding()
    else:
        print("[evaluate] building RagPipeline (retriever + generator + guardrails)...")
        embedding_service = None
        try:
            from app.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            pipeline = RagPipeline(
                retriever=Retriever(
                    vector_store=VectorStoreService(embedding_service=embedding_service),
                    embedding_service=embedding_service,
                    ragflow_client=None,
                    reranker=Reranker(embedding_service=embedding_service),
                ),
                generator=Generator(GeminiService(), PromptBuilder()),
                guardrails=Guardrails(),
                hallucination_detector=HallucinationDetector(),
                prompt_builder=PromptBuilder(),
            )
        except Exception as exc:  # pragma: no cover
            print(f"[evaluate] failed to build pipeline: {exc}")
            return 2
    metrics = Metrics(embedding_service=embedding_service)
    print(f"[evaluate] running pipeline over corpus at {settings.chroma_persist_dir}\n")

    results: list[dict] = []
    for sample in tqdm.tqdm(samples, desc="evaluating", unit="q"):
        results.append(run_sample(pipeline, metrics, sample))
        if args.sample_delay > 0:
            asyncio.run(asyncio.sleep(args.sample_delay))

    summary = aggregate(results)
    print("\n===== SCOREBOARD =====")
    print(print_scoreboard(summary, thresholds))
    failures = check_gates(summary, thresholds)
    print("\n===== GATES =====")
    if failures:
        print("\n".join(failures))
        print("RESULT: FAIL")
    else:
        print("  all gates passed")
        print("RESULT: PASS")

    report_path = write_report(summary, results, failures, Path(settings.evaluation_report_dir_path))
    print(f"report: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())