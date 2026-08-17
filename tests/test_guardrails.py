"""Unit tests for guardrails."""

from types import SimpleNamespace

from app.rag.failures import FailureType
from app.rag.guardrails import Guardrails


def settings_override(**kwargs):
    base = {
        "low_confidence_threshold": 0.40,
        "refusal_message": "I don't have enough information in the provided documents to answer this reliably.",
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def make_retrieval(chunks, confidence):
    return SimpleNamespace(chunks=chunks, retrieval_confidence=confidence)


def test_empty_context_refuses():
    g = Guardrails(settings_override())
    decision = g.evaluate(make_retrieval([], 0.0), "question")
    assert decision.should_refuse
    assert decision.failure_type == FailureType.EMPTY_CONTEXT


def test_low_confidence_refuses():
    g = Guardrails(settings_override())
    chunk = {"content": "some retrieved text about a topic"}
    decision = g.evaluate(make_retrieval([chunk], 0.20), "question")
    assert decision.should_refuse
    assert decision.failure_type == FailureType.LOW_CONFIDENCE


def test_high_confidence_ok():
    g = Guardrails(settings_override())
    chunk = {"content": "some retrieved text about a topic"}
    decision = g.evaluate(make_retrieval([chunk], 0.85), "question")
    assert not decision.should_refuse


def test_oversized_query_refuses():
    g = Guardrails(settings_override())
    chunk = {"content": "some text"}
    decision = g.evaluate(make_retrieval([chunk], 0.85), "x" * 2001)
    assert decision.should_refuse
    assert decision.failure_type == FailureType.INVALID_DOCUMENT