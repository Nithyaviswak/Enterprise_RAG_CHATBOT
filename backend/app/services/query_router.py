"""
Query Router — Intent Classification and Routing.

Routes user queries to appropriate handlers:
1. Small Talk → Gemini Direct (no RAG)
2. Knowledge Base Question → RAG pipeline
3. General LLM Question → Gemini Direct
4. Coding Question → Gemini Pro (special handling)
5. Document Question → RAG with document focus

This reduces unnecessary retrieval costs and improves response quality
by matching query type to optimal processing path.
"""

import logging
import json
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from app.services.gemini_service import GeminiService
from app.config import get_settings

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Query intent classification types."""
    SMALL_TALK = "small_talk"
    KNOWLEDGE_BASE = "knowledge_base"
    GENERAL_LLM = "general_llm"
    CODING = "coding"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """Result of query routing decision."""
    intent: QueryIntent
    confidence: float
    reasoning: str
    use_rag: bool
    use_gemini_pro: bool


class QueryRouter:
    """Routes queries to appropriate handlers based on intent classification."""

    # Classification prompt template
    CLASSIFICATION_PROMPT = """You are a query classifier for an Enterprise RAG system.

Classify the following user query into ONE of these categories:
- SMALL_TALK: Greetings, casual conversation, thank you, etc.
- KNOWLEDGE_BASE: Questions about documents, policies, procedures, knowledge in the system
- GENERAL_LLM: General questions that can be answered without specific documents
- CODING: Programming questions, technical implementation, code debugging
- DOCUMENT: Questions specifically about uploaded documents, files, PDFs

Query: "{query}"

Respond ONLY with a JSON object containing:
{{
    "intent": "category_name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Example response:
{{"intent": "KNOWLEDGE_BASE", "confidence": 0.92, "reasoning": "User is asking about company policy"}}"""

    # Keywords for fast classification (fallback when LLM is unavailable)
    KEYWORD_PATTERNS = {
        QueryIntent.SMALL_TALK: {
            "greetings": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            "casual": ["how are you", "what's up", "thanks", "thank you", "nice", "cool"],
            "farewell": ["bye", "goodbye", "see you", "later"],
        },
        QueryIntent.CODING: {
            "programming": ["code", "function", "class", "api", "python", "javascript", "java", "debug", "error", "bug"],
            "technical": ["implement", "algorithm", "database", "sql", "rest", "http", "frontend", "backend"],
        },
        QueryIntent.DOCUMENT: {
            "file_ref": ["document", "pdf", "file", "upload", "this document", "the file", "attached"],
            "page": ["page", "section", "chapter", "paragraph"],
        },
    }

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        """Initialize query router.

        Args:
            gemini_service: Gemini service for LLM-based classification
        """
        self.gemini_service = gemini_service
        self.settings = get_settings()

    async def route(self, query: str) -> RoutingDecision:
        """Route a query to the appropriate handler.

        Args:
            query: User's query string

        Returns:
            RoutingDecision with intent, confidence, and routing info
        """
        # First try fast keyword-based classification
        keyword_decision = self._keyword_classify(query)

        if keyword_decision.confidence >= 0.8:
            logger.info(f"Query routed via keywords: {keyword_decision.intent.value} (confidence: {keyword_decision.confidence})")
            return keyword_decision

        # Use LLM for more nuanced classification
        if self.gemini_service:
            try:
                llm_decision = await self._llm_classify(query)
                if llm_decision.confidence >= 0.5:
                    logger.info(f"Query routed via LLM: {llm_decision.intent.value} (confidence: {llm_decision.confidence})")
                    return llm_decision
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back to keywords: {e}")

        # Fallback to keyword classification
        return keyword_decision

    def _keyword_classify(self, query: str) -> RoutingDecision:
        """Fast keyword-based classification for low-latency routing.

        Args:
            query: User query

        Returns:
            RoutingDecision based on keyword matching
        """
        query_lower = query.lower()

        # Check Small Talk patterns
        for category, keywords in self.KEYWORD_PATTERNS[QueryIntent.SMALL_TALK].items():
            if any(kw in query_lower for kw in keywords):
                return RoutingDecision(
                    intent=QueryIntent.SMALL_TALK,
                    confidence=0.9,
                    reasoning=f"Detected {category} in query",
                    use_rag=False,
                    use_gemini_pro=False,
                )

        # Check Coding patterns
        for category, keywords in self.KEYWORD_PATTERNS[QueryIntent.CODING].items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches >= 2:
                return RoutingDecision(
                    intent=QueryIntent.CODING,
                    confidence=0.85,
                    reasoning=f"Detected {matches} coding keywords",
                    use_rag=False,
                    use_gemini_pro=True,  # Use Pro for coding tasks
                )

        # Check Document patterns
        for category, keywords in self.KEYWORD_PATTERNS[QueryIntent.DOCUMENT].items():
            if any(kw in query_lower for kw in keywords):
                return RoutingDecision(
                    intent=QueryIntent.DOCUMENT,
                    confidence=0.85,
                    reasoning=f"Detected document reference in query",
                    use_rag=True,
                    use_gemini_pro=False,
                )

        # Default to Knowledge Base (most common for RAG systems)
        return RoutingDecision(
            intent=QueryIntent.KNOWLEDGE_BASE,
            confidence=0.6,
            reasoning="Default classification - no specific pattern detected",
            use_rag=True,
            use_gemini_pro=False,
        )

    async def _llm_classify(self, query: str) -> RoutingDecision:
        """LLM-based classification for accurate routing.

        Args:
            query: User query

        Returns:
            RoutingDecision from LLM classification
        """
        prompt = self.CLASSIFICATION_PROMPT.format(query=query)

        # Use direct client call for JSON response
        try:
            from google import genai
            from google.genai import types
            from app.config import get_settings

            settings = get_settings()
            client = genai.Client(api_key=settings.gemini_api_key)

            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                ),
            )

            result = json.loads(response.text.strip())

            intent_str = result.get("intent", "UNKNOWN").upper()
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                intent = QueryIntent.UNKNOWN

            # Determine routing based on intent
            use_rag = intent in [QueryIntent.KNOWLEDGE_BASE, QueryIntent.DOCUMENT]
            use_gemini_pro = intent == QueryIntent.CODING

            return RoutingDecision(
                intent=intent,
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                use_rag=use_rag,
                use_gemini_pro=use_gemini_pro,
            )
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return RoutingDecision(
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                reasoning="Failed to classify",
                use_rag=True,  # Default to RAG
                use_gemini_pro=False,
            )

    def should_use_rag(self, query: str) -> bool:
        """Quick check if query should use RAG (synchronous).

        This is a fast path for cases where we need a quick decision
        without awaiting the full classification.

        Args:
            query: User query

        Returns:
            True if RAG should be used
        """
        # Fast negative checks
        query_lower = query.lower()

        # Negative patterns (no RAG needed)
        negative_patterns = [
            "hello", "hi", "hey", "thanks", "thank you",
            "bye", "goodbye", "how are you", "what's up",
        ]

        for pattern in negative_patterns:
            if pattern in query_lower:
                return False

        return True
