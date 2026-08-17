"""Structured tracing and metrics for RAG requests.

Every RAG request gets a ``request_id``. Stages record latencies and are
emitted as structured ``key=value`` log lines (JSON-parseable) so each request
can be traced end to end:

    INFO request_id=abc123 stage=retrieval latency_ms=320 chunks=5
    WARN request_id=abc123 failure_type=LOW_CONFIDENCE confidence=0.41

Nothing sensitive (API keys, passwords, personal data) is ever logged.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = {"api_key", "password", "token", "secret", "authorization"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if str(k).lower() in _SENSITIVE_KEYS else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def log_event(
    level: int,
    event: str,
    request_id: str = "",
    **fields,
) -> None:
    """Emit a structured ``key=value`` log record."""
    parts = [f"event={event}"]
    if request_id:
        parts.append(f"request_id={request_id}")
    for k, v in _redact(fields).items():
        if isinstance(v, (list, dict)):
            try:
                v = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                v = str(v)
        parts.append(f"{k}={v}")
    logger.log(level, " ".join(parts))


@dataclass
class StageTrace:
    """Timing and outcome for a single pipeline stage."""

    stage: str
    start: float
    end: Optional[float] = None
    latency_ms: Optional[float] = None
    status: str = "ok"
    metadata: dict = field(default_factory=dict)


class Tracer:
    """Per-request trace context."""

    def __init__(self, request_id: str, question: str = ""):
        self.request_id = request_id
        self.question = question
        self.started_at = time.perf_counter()
        self.stages: list[StageTrace] = []
        self.failure_type: Optional[str] = None
        self.failure_reason: str = ""

    @classmethod
    def new(cls, question: str = "") -> "Tracer":
        return cls(uuid.uuid4().hex[:12], question)

    def begin(self, stage: str) -> StageTrace:
        trace = StageTrace(stage=stage, start=time.perf_counter())
        self.stages.append(trace)
        return trace

    def end(self, trace: StageTrace, status: str = "ok", **metadata):
        trace.end = time.perf_counter()
        trace.latency_ms = (trace.end - trace.start) * 1000
        trace.status = status
        trace.metadata.update(metadata)
        log_event(
            logging.INFO if status == "ok" else logging.WARNING,
            "stage",
            self.request_id,
            stage=trace.stage,
            latency_ms=f"{trace.latency_ms:.1f}ms",
            status=status,
            **{k: v for k, v in metadata.items() if k in {"chunks", "scores", "failure_type", "confidence"}},
        )

    def warn(self, event: str, **fields):
        log_event(logging.WARNING, event, self.request_id, **fields)

    def info(self, event: str, **fields):
        log_event(logging.INFO, event, self.request_id, **fields)

    def error(self, event: str, **fields):
        log_event(logging.ERROR, event, self.request_id, **fields)

    def set_failure(self, failure_type: str, reason: str = ""):
        self.failure_type = failure_type
        self.failure_reason = reason
        log_event(
            logging.WARNING if failure_type not in {"LLM_FAILURE", "RETRIEVAL_FAILURE"} else logging.ERROR,
            "failure",
            self.request_id,
            failure_type=failure_type,
            reason=reason,
        )

    def summary(self) -> dict:
        """Return a compact, safe tracing summary for the response/debug view."""
        return {
            "request_id": self.request_id,
            "total_latency_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
            "stages": {
                t.stage: {
                    "latency_ms": round(t.latency_ms or 0, 2),
                    "status": t.status,
                    **t.metadata,
                }
                for t in self.stages
            },
            "failure_type": self.failure_type,
        }


class MetricsStore:
    """In-memory rolling metrics used by the developer evaluation dashboard."""

    _instance: Optional["MetricsStore"] = None

    def __init__(self, max_size: int = 5000):
        self.records: list[dict] = []
        self.counters: dict[str, int] = {}
        self.max_size = max_size

    @classmethod
    def get(cls) -> "MetricsStore":
        if cls._instance is None:
            cls._instance = MetricsStore()
        return cls._instance

    def record(self, entry: dict):
        """Store a safe per-request metrics entry."""
        safe = {
            "request_id": entry.get("request_id"),
            "timestamp": entry.get("timestamp"),
            "retrieval_latency_ms": entry.get("retrieval_latency_ms"),
            "generation_latency_ms": entry.get("generation_latency_ms"),
            "total_latency_ms": entry.get("total_latency_ms"),
            "chunks": entry.get("chunks"),
            "retrieval_confidence": entry.get("retrieval_confidence"),
            "model_used": entry.get("model_used"),
            "failure_type": entry.get("failure_type"),
            "answered": entry.get("answered"),
            "grounded_ratio": entry.get("grounded_ratio"),
            "refused": entry.get("refused"),
        }
        self.records.append(safe)
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]
        if entry.get("failure_type"):
            self.counters[entry["failure_type"]] = self.counters.get(entry["failure_type"], 0) + 1
        if entry.get("refused"):
            self.counters["REFUSED"] = self.counters.get("REFUSED", 0) + 1

    def summarize(self) -> dict:
        """Aggregate metrics for the evaluation dashboard."""
        records = self.records
        n = len(records)
        if n == 0:
            return {"total_requests": 0}

        def avg(key):
            vals = [r.get(key) for r in records if r.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        grounded = [r for r in records if r.get("grounded_ratio") is not None]
        return {
            "total_requests": n,
            "avg_retrieval_latency_ms": avg("retrieval_latency_ms"),
            "avg_generation_latency_ms": avg("generation_latency_ms"),
            "avg_total_latency_ms": avg("total_latency_ms"),
            "avg_retrieval_confidence": avg("retrieval_confidence"),
            "avg_grounding_ratio": avg("grounded_ratio"),
            "refusal_rate": round(len([r for r in records if r.get("refused")]) / n, 3),
            "hallucination_risk_rate": round(
                len([r for r in grounded if r["grounded_ratio"] < 0.6]) / len(grounded), 3
            ) if grounded else 0.0,
            "failure_counts": {k: v for k, v in self.counters.items() if v},
            "recent": records[-50:],
        }