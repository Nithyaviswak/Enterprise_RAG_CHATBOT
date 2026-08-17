"""Unit tests for the deterministic hallucination detector."""

from app.rag.hallucination import HallucinationDetector, is_refusal_answer

CTX = [
    {"content": "The AI Research Agent uses LangGraph and Google Gemini for deep web research.", "source": "kb.txt"},
    {"content": "It achieves sub-3s latency with a 60% improvement over REST baselines.", "source": "kb.txt"},
]


def test_grounded_answer_passes():
    answer = "The AI Research Agent uses LangGraph and Google Gemini. It achieves sub-3s latency."
    report = HallucinationDetector().analyze(answer, CTX)
    assert report.is_grounded
    assert report.grounded_ratio >= 0.7
    assert report.risk_level in {"low", "medium"}


def test_unsupported_claim_flagged():
    answer = "The AI Research Agent uses LangGraph. It was funded by the Apollo Foundation in 2031."
    report = HallucinationDetector().analyze(answer, CTX)
    assert not report.is_grounded
    assert any("Apollo" in c or "apollo" in c for c in report.unsupported_claims)
    assert len(report.unsupported_claims) == 1


def test_empty_answer_is_high_risk():
    report = HallucinationDetector().analyze("", CTX)
    assert report.risk_level == "high"
    assert report.grounded_ratio == 0.0


def test_empty_context_flagged():
    report = HallucinationDetector().analyze("The agent uses LangGraph.", [])
    assert report.risk_level == "high"


def test_refusal_marker_detected():
    assert is_refusal_answer("I don't have enough information in the provided documents to answer this reliably.")
    assert not is_refusal_answer("The agent uses LangGraph.")


def test_combined_confidence_ranges():
    det = HallucinationDetector()
    report = det.analyze("The agent uses LangGraph and Google Gemini.", CTX)
    conf = det.combined_confidence(retrieval_confidence=0.8, report=report, answered=True)
    assert 0.0 <= conf["overall_confidence"] <= 1.0
    assert conf["confidence_level"] in {"low", "medium", "high"}