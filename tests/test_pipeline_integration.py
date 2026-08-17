"""Integration test of the full RAG pipeline using hermetic fakes.

Runs the real RagPipeline orchestration (retrieval → guardrails → generation
→ hallucination → attribution) with a fake retriever and fake LLM so the flow
is exercised without any network access.
"""

import asyncio

import pytest

from app.rag.failures import FailureType
from app.rag.generator import GenerationResult
from app.rag.guardrails import Guardrails
from app.rag.hallucination import HallucinationDetector
from app.rag.pipeline import RagPipeline
from app.rag.retriever import RetrievalResult

GOOD_CHUNKS = [
    {
        "content": "The Autonomous AI Research Agent uses LangGraph and Google Gemini.",
        "source": "kb.txt",
        "metadata": {"page": 1},
        "score": 0.9,
    }
]
GOOD_CONFIDENCE = 0.85

ANSWER = "The Autonomous AI Research Agent uses LangGraph and Google Gemini."


class FakeGenerator:
    def __init__(self, answer=ANSWER):
        self.answer = answer

    async def generate(self, query, contexts, history=None):
        return GenerationResult(
            answer=self.answer,
            system_prompt="<grounded>",
            latency_ms=5.0,
            model="fake-llm",
        )


class FailingGenerator(FakeGenerator):
    async def generate(self, query, contexts, history=None):
        return GenerationResult(
            answer="",
            system_prompt="<grounded>",
            generation_failure=True,
            failure_reason="LLM down",
        )


class FakeRetriever:
    def __init__(self, chunks, confidence):
        self._chunks = chunks
        self._confidence = confidence
        self.retrieve_calls = 0

    def retrieve(self, query, top_k=None, where=None, use_ragflow=True, use_reranking=True):
        self.retrieve_calls += 1
        return RetrievalResult(
            chunks=[dict(c) for c in self._chunks],
            retrieval_confidence=self._confidence,
            methods_used=["fake"],
        )

    def get_context(self, chunks, max_chars=None):
        return chunks


def make_pipeline(retriever, generator):
    return RagPipeline(
        retriever=retriever,
        generator=generator,
        guardrails=Guardrails(),
        hallucination_detector=HallucinationDetector(),
    )


def test_pipeline_answers_and_attributes():
    retriever = FakeRetriever(GOOD_CHUNKS, GOOD_CONFIDENCE)
    pipeline = make_pipeline(retriever, FakeGenerator())
    result = asyncio.run(pipeline.run("What does the agent use?"))

    assert result.answered
    assert not result.refused
    assert result.failure_type is None
    assert ANSWER in result.answer
    assert result.contexts
    assert result.sources
    assert result.sources[0]["source"] == "kb.txt"
    assert result.sources[0]["page"] == 1
    assert result.confidence["overall_confidence"] > 0.5
    assert retriever.retrieve_calls == 1


def test_pipeline_refuses_on_empty_context():
    retriever = FakeRetriever([], 0.0)
    pipeline = make_pipeline(retriever, FakeGenerator())
    result = asyncio.run(pipeline.run("what is the capital of australia?"))

    assert not result.answered
    assert result.refused
    assert result.failure_type == FailureType.EMPTY_CONTEXT.value
    assert "enough information" in result.answer


def test_pipeline_refuses_on_low_confidence():
    retriever = FakeRetriever(GOOD_CHUNKS, 0.05)
    pipeline = make_pipeline(retriever, FakeGenerator())
    result = asyncio.run(pipeline.run("question?"))

    assert result.refused
    assert result.failure_type == FailureType.LOW_CONFIDENCE.value


def test_pipeline_handles_llm_failure():
    retriever = FakeRetriever(GOOD_CHUNKS, GOOD_CONFIDENCE)
    pipeline = make_pipeline(retriever, FailingGenerator())
    result = asyncio.run(pipeline.run("question?"))

    assert not result.answered
    assert result.refused
    assert result.failure_type == FailureType.LLM_FAILURE.value
    assert "enough information" in result.answer


def test_pipeline_debug_payload():
    retriever = FakeRetriever(GOOD_CHUNKS, GOOD_CONFIDENCE)
    pipeline = make_pipeline(retriever, FakeGenerator())
    result = asyncio.run(pipeline.run("What does the agent use?", debug=True))

    assert result.debug is not None
    assert result.debug["query"] == "What does the agent use?"
    assert result.debug["request_id"]
    assert result.debug["retrieved_documents"]
    assert result.debug["final_context"]
    assert result.debug["stage_times"] is not None


def test_pipeline_normal_run_no_debug():
    retriever = FakeRetriever(GOOD_CHUNKS, GOOD_CONFIDENCE)
    pipeline = make_pipeline(retriever, FakeGenerator())
    result = asyncio.run(pipeline.run("Where is the agent used?"))
    assert result.debug is None