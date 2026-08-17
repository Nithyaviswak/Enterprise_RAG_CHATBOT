"""Failure taxonomy and structured error types for the RAG pipeline.

Every guardrail/failure in the pipeline maps to one of these explicit
categories so failures are observable, debuggable and measurable.
"""

from enum import Enum
from typing import Optional


class FailureType(str, Enum):
    """Explicit failure categories used across the RAG pipeline."""

    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    EMPTY_CONTEXT = "EMPTY_CONTEXT"
    LLM_FAILURE = "LLM_FAILURE"
    TIMEOUT = "TIMEOUT"
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    PARSING_FAILURE = "PARSING_FAILURE"
    HALLUCINATION_RISK = "HALLUCINATION_RISK"


class PipelineError(Exception):
    """Base pipeline error carrying a failure category."""

    def __init__(self, failure_type: FailureType, message: str, **details):
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type.value,
            "message": self.message,
            "details": self.details,
        }


class RetrievalFailure(PipelineError):
    def __init__(self, message: str, **details):
        super().__init__(FailureType.RETRIEVAL_FAILURE, message, **details)


class LowConfidenceError(PipelineError):
    def __init__(self, message: str, **details):
        super().__init__(FailureType.LOW_CONFIDENCE, message, **details)


class EmptyContextError(PipelineError):
    def __init__(self, message: str = "No relevant context retrieved.", **details):
        super().__init__(FailureType.EMPTY_CONTEXT, message, **details)


class LLMFailure(PipelineError):
    def __init__(self, message: str, **details):
        super().__init__(FailureType.LLM_FAILURE, message, **details)


class TimeoutError_(PipelineError):
    def __init__(self, message: str, **details):
        super().__init__(FailureType.TIMEOUT, message, **details)


class InvalidDocumentError(PipelineError):
    def __init__(self, message: str, **details):
        super().__init__(FailureType.INVALID_DOCUMENT, message, **details)


class ParsingFailure(PipelineError):
    def __init__(self, message: str, **details):
        super().__init__(FailureType.PARSING_FAILURE, message, **details)


class HallucinationRisk(PipelineError):
    def __init__(self, message: str, **details):
        super().__init__(FailureType.HALLUCINATION_RISK, message, **details)


def describe_failure(failure_type: FailureType) -> str:
    """Return a short human-readable description of a failure category."""
    return {
        FailureType.RETRIEVAL_FAILURE: "The retrieval stage raised an unexpected error.",
        FailureType.LOW_CONFIDENCE: "Retrieval confidence was below the acceptable threshold.",
        FailureType.EMPTY_CONTEXT: "No relevant context could be retrieved for the query.",
        FailureType.LLM_FAILURE: "The language model call failed or timed out.",
        FailureType.TIMEOUT: "A pipeline stage exceeded its allowed time budget.",
        FailureType.INVALID_DOCUMENT: "The uploaded document is unsupported or empty.",
        FailureType.PARSING_FAILURE: "The document could not be parsed into text.",
        FailureType.HALLUCINATION_RISK: "The generated answer contains unsupported claims.",
    }.get(failure_type, "Unknown failure.")
