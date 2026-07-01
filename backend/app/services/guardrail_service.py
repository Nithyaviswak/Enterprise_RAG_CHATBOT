"""
Guardrail Service — Prompt Injection Defense.

Detects and blocks prompt injection attacks:
- "Ignore previous instructions"
- System prompt leakage attempts
- Data exfiltration attempts
- Jailbreak prompts

Blocks malicious queries before they reach the retrieval or generation stage.
"""

import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat level classification."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


@dataclass
class GuardrailResult:
    """Result of guardrail analysis."""
    is_safe: bool
    threat_level: ThreatLevel
    detected_patterns: list[str]
    blocked_reason: Optional[str]
    confidence: float


class GuardrailService:
    """Service for detecting and blocking prompt injection attacks."""

    # High-confidence blocking patterns
    BLOCK_PATTERNS = [
        # Direct instruction override attempts
        r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|prior|earlier)\s+(?:instructions?|commands?|directives?))",
        r"(?i)(?:disregard\s+(?:all\s+)?(?:previous|prior|earlier)\s+(?:instructions?|commands?))",
        r"(?i)(?:forget\s+(?:everything|all\s+your\s+)?(?:instructions?|training|rules))",

        # System prompt extraction
        r"(?i)(?:reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|directives?))",
        r"(?i)(?:show\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions?))",
        r"(?i)(?:what\s+(?:are|were)\s+your\s+(?:original\s+)?(?:instructions?|system\s+prompt))",
        r"(?i)(?:tell\s+(?:me\s+)?(?:about\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|directives?))",

        # Jailbreak attempts
        r"(?i)(?:do\s+anything\s+now)",
        r"(?i)(?:ignore\s+(?:all\s+)?(?:rules?|restrictions?|limitations?))",
        r"(?i)(?:jailbreak)",
        r"(?i)(?:developer\s+mode",
        r"(?i)(?: DAN\b)",  # "Do Anything Now"
        r"(?i)(?:roleplay\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered)",
        r"(?i)(?:bypass\s+(?:your\s+)?(?:safety|content\s+filter))",

        # Data exfiltration
        r"(?i)(?:extract\s+(?:all\s+)?(?:your\s+)?(?:training\s+)?(?:data|information))",
        r"(?i)(?:list\s+(?:all\s+)?(?:the\s+)?(?:files?|documents?|data)\s+you\s+(?:have|know\s+of))",
        r"(?i)(?:reveal\s+(?:your\s+)?(?:full|complete)\s+(?:system|architecture))",

        # Injection via formatting
        r"(?:^|\n)###\s*SYSTEM",
        r"(?:^|\n)###\s*INSTRUCTION",
        r"(?:^|\n)###\s*DIRECTIVE",
        r"<\|system\|>",
        r"\[INST\]",
        r"<<SYS>>",
    ]

    # Medium-confidence suspicious patterns
    SUSPICIOUS_PATTERNS = [
        # Subtle override attempts
        r"(?i)(?:instead\s+of\s+(?:what\s+you\s+were?\s+(?:told|asked|programmed)))",
        r"(?i)(?:new\s+(?:instruction|command|directive))",
        r"(?i)(?:override\s+(?:your\s+)?(?:safety|guidelines))",
        r"(?i)(?:pretend\s+(?:to\s+be|you\s+are))",
        r"(?i)(?:act\s+as\s+(?:if|though)",
        r"(?i)(?:you\s+are\s+(?:now|previously|originally))",

        # Prompt manipulation
        r"(?i)(?:in\s+the\s+following\s+(?:scenario|case|context))",
        r"(?i)(?:for\s+(?:educational|research)\s+purposes?)",
        r"(?i)(?:hypothetically\s+speaking",
        r"(?i)(?:just\s+(?:curious|imagine))",

        # Token manipulation
        r"(?i)(?:\\n{3,})",  # Multiple newlines (potential injection)
        r"(?i)(?:{.*}.*)",   # Template-like patterns
    ]

    # Patterns that might indicate accidentally leaked prompts
    LEAK_PATTERNS = [
        r"(?i)You\s+are\s+a\s+(?:helpful| benevolent).*assistant",
        r"(?i)system\s+instruction",
        r"(?i)base[d]?\s+on\s+(?:the\s+)?(?:following|above)\s+context",
        r"(?i)context:\s*",
    ]

    def __init__(self):
        """Initialize guardrail service."""
        # Compile regex patterns for efficiency
        self._block_patterns = [re.compile(p) for p in self.BLOCK_PATTERNS]
        self._suspicious_patterns = [re.compile(p) for p in self.SUSPICIOUS_PATTERNS]
        self._leak_patterns = [re.compile(p) for p in self.LEAK_PATTERNS]

        # Statistics
        self._total_checked = 0
        self._total_blocked = 0

    def check(self, text: str) -> GuardrailResult:
        """Check text for prompt injection attempts.

        Args:
            text: Text to analyze

        Returns:
            GuardrailResult with threat assessment
        """
        self._total_checked += 1
        detected_patterns = []

        # Check high-confidence blocking patterns
        for pattern in self._block_patterns:
            match = pattern.search(text)
            if match:
                detected_patterns.append(f"BLOCK: {match.group()[:50]}")
                self._total_blocked += 1
                return GuardrailResult(
                    is_safe=False,
                    threat_level=ThreatLevel.BLOCKED,
                    detected_patterns=detected_patterns,
                    blocked_reason=self._get_block_reason(match.group()),
                    confidence=0.95,
                )

        # Check suspicious patterns
        for pattern in self._suspicious_patterns:
            match = pattern.search(text)
            if match:
                detected_patterns.append(f"SUSPICIOUS: {match.group()[:50]}")

        # Check for potential prompt leaks (lower confidence)
        for pattern in self._leak_patterns:
            match = pattern.search(text)
            if match:
                detected_patterns.append(f"LEAK: {match.group()[:50]}")

        # Determine threat level based on pattern count
        if len(detected_patterns) >= 3:
            threat_level = ThreatLevel.HIGH
            is_safe = False
            confidence = 0.8
        elif len(detected_patterns) >= 1:
            threat_level = ThreatLevel.MEDIUM
            is_safe = True
            confidence = 0.6
        else:
            threat_level = ThreatLevel.SAFE
            is_safe = True
            confidence = 0.95

        return GuardrailResult(
            is_safe=is_safe,
            threat_level=threat_level,
            detected_patterns=detected_patterns,
            blocked_reason=None if is_safe else "High threat detected",
            confidence=confidence,
        )

    def check_with_llm(self, text: str) -> GuardrailResult:
        """Enhanced check using LLM for ambiguous cases.

        This is called when the pattern-based check returns MEDIUM threat.

        Args:
            text: Text to analyze

        Returns:
            GuardrailResult with LLM-enhanced assessment
        """
        # For now, return the pattern-based result
        # Can be extended with LLM-based classification
        return self.check(text)

    def _get_block_reason(self, matched_text: str) -> str:
        """Get human-readable block reason.

        Args:
            matched_text: The matched text that triggered blocking

        Returns:
            Reason string
        """
        matched_lower = matched_text.lower()

        if "ignore" in matched_lower and "previous" in matched_lower:
            return "Attempted to override previous instructions"
        elif "reveal" in matched_lower and "system" in matched_lower:
            return "Attempted to extract system prompt"
        elif "jailbreak" in matched_lower:
            return "Jailbreak attempt detected"
        elif "developer" in matched_lower and "mode" in matched_lower:
            return "Developer mode jailbreak detected"
        elif "dan" in matched_lower:
            return "DAN (Do Anything Now) jailbreak detected"
        elif "extract" in matched_lower or "exfiltrat" in matched_lower:
            return "Data exfiltration attempt detected"
        else:
            return "Malicious prompt pattern detected"

    def get_statistics(self) -> dict:
        """Get guardrail statistics.

        Returns:
            Dictionary with check and block counts
        """
        return {
            "total_checked": self._total_checked,
            "total_blocked": self._total_blocked,
            "block_rate": self._total_blocked / max(self._total_checked, 1),
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self._total_checked = 0
        self._total_blocked = 0
