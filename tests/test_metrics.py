"""Unit tests for the evaluation metrics (deterministic)."""

from evaluation.metrics import Metrics, aggregate, normalize_contexts

CTX_GOOD = [
    {"content": "Nithyananda Chari R is pursuing a B.Tech in AI and Machine Learning with a CGPA of 8.44.", "source": "resume.txt"},
]
CTX_EMPTY = [{"content": "Weather forecast for coastal regions tomorrow.", "source": "other.txt"}]

METRICS = Metrics(embedding_service=None)


def test_normalize_contexts_strings():
    assert normalize_contexts(["hello"]) == [{"content": "hello"}]
    assert normalize_contexts([{"content": "hi"}]) == [{"content": "hi"}]


def test_grounded_answer_scores_well():
    m = METRICS.compute_sample(
        question="What CGPA does Nithyananda have?",
        reference_answer="A CGPA of 8.44, pursuing B.Tech in AI and Machine Learning.",
        contexts=CTX_GOOD,
        generated_answer="Nithyananda Chari R has a CGPA of 8.44 and is pursuing a B.Tech in AI and Machine Learning.",
        expected_in_corpus=True,
    )
    assert m["faithfulness"] >= 0.7
    assert m["context_recall"] >= 0.5
    assert not m["hallucination_flag"]


def test_unrelated_context_precision_low():
    m = METRICS.compute_sample(
        question="What CGPA does Nithyananda have?",
        reference_answer="A CGPA of 8.44.",
        contexts=CTX_EMPTY,
        generated_answer="The agent uses LangGraph.",
        expected_in_corpus=True,
    )
    assert m["hallucination_flag"]
    assert m["faithfulness"] < 0.5


def test_refusal_for_missing_doc_is_faithful():
    m = METRICS.compute_sample(
        question="What is the price of Bitcoin today?",
        reference_answer="",
        contexts=[],
        generated_answer="I don't have enough information in the provided documents to answer this reliably.",
        expected_in_corpus=False,
    )
    assert m["faithfulness"] == 1.0
    assert m["refused"]
    assert not m["hallucination_flag"]


def test_aggregate_basic():
    a = METRICS.compute_sample("q", "ref", CTX_GOOD, "Nithyananda has a CGPA of 8.44.", expected_in_corpus=True)
    b = METRICS.compute_sample("q", "", [], "", expected_in_corpus=False)
    summary = aggregate([a, b])
    assert summary["total_samples"] == 2
    assert 0.0 <= summary["faithfulness"] <= 1.0
    assert "hallucination_rate" in summary
    assert "retrieval_success_rate" in summary


def test_aggregate_empty():
    assert aggregate([]) == {}