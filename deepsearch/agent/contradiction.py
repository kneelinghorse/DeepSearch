"""Contradiction detection for Alpha-tier research.

This module implements B10.9: Contradiction Detection, which uses LLM-based
analysis to identify conflicting claims across research sources. This is
critical for Alpha-tier research where novel domains often have incomplete
or conflicting information.

Contradiction detection is controlled by depth_config.enable_contradiction_detection
and only runs when enabled (Alpha tier by default).

The detection process:
1. Extract key claims from research findings
2. Use LLM to identify potential contradictions between claims
3. Score each contradiction by confidence (how clearly they conflict)
4. High-confidence contradictions (>0.8) reduce overall research confidence
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Confidence threshold above which contradictions affect overall score
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Penalty applied to overall confidence per high-confidence contradiction
CONFIDENCE_PENALTY_PER_CONTRADICTION = 0.10


@dataclass
class Contradiction:
    """A detected contradiction between two claims.

    Attributes:
        claim_a: First claim text from the research.
        claim_b: Second claim that contradicts claim_a.
        topic: The subject area where the contradiction occurs.
        confidence: How confident we are this is a real contradiction (0.0-1.0).
        source_a: Optional URL or source identifier for claim_a.
        source_b: Optional URL or source identifier for claim_b.
        explanation: LLM-generated explanation of why these contradict.
    """

    claim_a: str
    claim_b: str
    topic: str
    confidence: float
    source_a: Optional[str] = None
    source_b: Optional[str] = None
    explanation: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate confidence is in valid range."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "topic": self.topic,
            "confidence": self.confidence,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contradiction":
        """Create from dictionary."""
        return cls(
            claim_a=str(data.get("claim_a", "")),
            claim_b=str(data.get("claim_b", "")),
            topic=str(data.get("topic", "unknown")),
            confidence=float(data.get("confidence", 0.5)),
            source_a=data.get("source_a"),
            source_b=data.get("source_b"),
            explanation=data.get("explanation"),
        )

    @property
    def is_high_confidence(self) -> bool:
        """Check if this contradiction meets the high-confidence threshold."""
        return self.confidence >= HIGH_CONFIDENCE_THRESHOLD


@dataclass
class ContradictionReport:
    """Report containing all detected contradictions.

    Attributes:
        contradictions: List of detected contradictions.
        sources_analyzed: Number of sources that were analyzed.
        claims_extracted: Number of claims extracted from findings.
        analysis_complete: Whether analysis completed successfully.
        error_message: Error message if analysis failed.
    """

    contradictions: List[Contradiction] = field(default_factory=list)
    sources_analyzed: int = 0
    claims_extracted: int = 0
    analysis_complete: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "contradictions": [c.to_dict() for c in self.contradictions],
            "sources_analyzed": self.sources_analyzed,
            "claims_extracted": self.claims_extracted,
            "analysis_complete": self.analysis_complete,
            "error_message": self.error_message,
            "has_contradictions": self.has_contradictions,
            "high_confidence_count": self.high_confidence_count,
            "confidence_penalty": self.confidence_penalty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContradictionReport":
        """Create from dictionary."""
        contradictions = [
            Contradiction.from_dict(c)
            for c in data.get("contradictions", [])
        ]
        return cls(
            contradictions=contradictions,
            sources_analyzed=int(data.get("sources_analyzed", 0)),
            claims_extracted=int(data.get("claims_extracted", 0)),
            analysis_complete=bool(data.get("analysis_complete", True)),
            error_message=data.get("error_message"),
        )

    @property
    def has_contradictions(self) -> bool:
        """Check if any contradictions were detected."""
        return len(self.contradictions) > 0

    @property
    def high_confidence_count(self) -> int:
        """Count of high-confidence contradictions."""
        return sum(1 for c in self.contradictions if c.is_high_confidence)

    @property
    def confidence_penalty(self) -> float:
        """Calculate the confidence penalty based on high-confidence contradictions.

        Each high-confidence contradiction applies CONFIDENCE_PENALTY_PER_CONTRADICTION
        (default 10%) penalty, capped at 50% total penalty.
        """
        penalty = self.high_confidence_count * CONFIDENCE_PENALTY_PER_CONTRADICTION
        return min(0.5, penalty)  # Cap at 50% penalty

    @classmethod
    def empty(cls) -> "ContradictionReport":
        """Create an empty report indicating no contradictions found."""
        return cls(
            contradictions=[],
            sources_analyzed=0,
            claims_extracted=0,
            analysis_complete=True,
            error_message=None,
        )

    @classmethod
    def error(cls, message: str) -> "ContradictionReport":
        """Create a report indicating an error occurred."""
        return cls(
            contradictions=[],
            sources_analyzed=0,
            claims_extracted=0,
            analysis_complete=False,
            error_message=message,
        )


def build_contradiction_detection_prompt(
    findings: Sequence[str],
    sources: Sequence[Dict[str, Any]],
    mission_objectives: Sequence[str],
) -> Tuple[str, str]:
    """Build the system and human prompts for contradiction detection.

    Args:
        findings: List of research findings to analyze.
        sources: List of source dictionaries with url/title/snippet.
        mission_objectives: Mission objectives for context.

    Returns:
        Tuple of (system_prompt, human_prompt).
    """
    system_prompt = """You are a research analyst specializing in identifying contradictions.
Your task is to find OBVIOUS contradictions in research findings - claims that directly
contradict each other on the same topic.

IMPORTANT: Only flag clear, substantive contradictions. Do NOT flag:
- Different perspectives on subjective topics
- Partial or incomplete information that doesn't directly conflict
- Nuanced differences that could both be true
- Information from different time periods that may both have been true

Output STRICT JSON matching this schema:
{
  "contradictions": [
    {
      "claim_a": "Exact text of first claim",
      "claim_b": "Exact text of contradicting claim",
      "topic": "Subject area (e.g., 'market size', 'technology capability')",
      "confidence": 0.0-1.0,
      "explanation": "Brief explanation of why these contradict"
    }
  ]
}

Confidence scoring:
- 0.9-1.0: Direct numerical/factual contradiction (e.g., "X is 10" vs "X is 100")
- 0.7-0.8: Clear logical contradiction (e.g., "X is required" vs "X is optional")
- 0.5-0.6: Potential contradiction requiring interpretation
- Below 0.5: Not a clear contradiction, don't include

If no contradictions are found, return: {"contradictions": []}"""

    # Format findings with indices for reference
    findings_section = "\n".join(
        f"[{i+1}] {finding}" for i, finding in enumerate(findings)
    )

    # Format sources for context
    source_section = "\n".join(
        f"- {src.get('title', 'Untitled')}: {src.get('url', 'no URL')}"
        for src in sources[:10]  # Limit to 10 sources
    )

    objectives_section = "\n".join(
        f"- {obj}" for obj in mission_objectives
    ) if mission_objectives else "- General research mission"

    human_prompt = f"""Analyze these research findings for contradictions.

**Mission Objectives:**
{objectives_section}

**Research Findings:**
{findings_section}

**Sources Consulted:**
{source_section}

Identify any obvious contradictions between findings. Focus on factual conflicts, not differences in opinion or perspective.

Respond with ONLY the JSON object. No markdown, no explanation outside JSON."""

    return system_prompt, human_prompt


def parse_contradiction_response(response: str) -> List[Contradiction]:
    """Parse LLM response into Contradiction objects.

    Handles various response formats including markdown code blocks
    and partial JSON.

    Args:
        response: Raw LLM response text.

    Returns:
        List of parsed Contradiction objects.
    """
    if not response or not response.strip():
        return []

    # Clean up response - remove markdown code blocks
    cleaned = response.strip()
    if cleaned.startswith("```"):
        # Remove opening code block (```json or just ```)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        # Remove closing code block
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # Try to find JSON object in response
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        logger.warning("No JSON object found in contradiction response")
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse contradiction JSON: %s", exc)
        return []

    contradictions: List[Contradiction] = []
    raw_contradictions = data.get("contradictions", [])

    if not isinstance(raw_contradictions, list):
        logger.warning("contradictions field is not a list")
        return []

    for item in raw_contradictions:
        if not isinstance(item, dict):
            continue

        # Validate required fields
        claim_a = item.get("claim_a", "")
        claim_b = item.get("claim_b", "")
        topic = item.get("topic", "unknown")
        confidence = item.get("confidence", 0.5)

        if not claim_a or not claim_b:
            logger.debug("Skipping contradiction with empty claims")
            continue

        # Filter out low-confidence contradictions
        try:
            conf_float = float(confidence)
        except (TypeError, ValueError):
            conf_float = 0.5

        if conf_float < 0.5:
            logger.debug("Skipping low-confidence contradiction: %.2f", conf_float)
            continue

        contradiction = Contradiction(
            claim_a=str(claim_a),
            claim_b=str(claim_b),
            topic=str(topic),
            confidence=conf_float,
            explanation=item.get("explanation"),
        )
        contradictions.append(contradiction)

    return contradictions


async def detect_contradictions(
    findings: Sequence[str],
    sources: Sequence[Dict[str, Any]],
    mission_objectives: Sequence[str],
    llm: Any,
) -> ContradictionReport:
    """Detect contradictions in research findings using LLM analysis.

    This is the main entry point for contradiction detection. It builds
    prompts, invokes the LLM, and parses the response into a structured
    report.

    Args:
        findings: List of research findings to analyze.
        sources: List of source dictionaries with url/title/snippet.
        mission_objectives: Mission objectives for context.
        llm: LangChain-compatible LLM instance.

    Returns:
        ContradictionReport with detected contradictions and metrics.
    """
    # Skip if insufficient data
    if not findings or len(findings) < 2:
        logger.debug("Insufficient findings for contradiction detection")
        return ContradictionReport(
            contradictions=[],
            sources_analyzed=len(sources),
            claims_extracted=len(findings),
            analysis_complete=True,
        )

    try:
        # Import message classes
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            # Fallback for testing
            class SystemMessage:
                def __init__(self, content: str) -> None:
                    self.content = content

            class HumanMessage:
                def __init__(self, content: str) -> None:
                    self.content = content

        system_prompt, human_prompt = build_contradiction_detection_prompt(
            findings=findings,
            sources=sources,
            mission_objectives=mission_objectives,
        )

        # Invoke LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(messages)
        else:
            # Fallback for sync invocation
            response = llm.invoke(messages)

        # Extract content from response
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        # Parse contradictions
        contradictions = parse_contradiction_response(content)

        return ContradictionReport(
            contradictions=contradictions,
            sources_analyzed=len(sources),
            claims_extracted=len(findings),
            analysis_complete=True,
        )

    except Exception as exc:
        logger.error("Contradiction detection failed: %s", exc)
        return ContradictionReport.error(str(exc))


def apply_contradiction_penalty(
    confidence_score: float,
    report: ContradictionReport,
) -> float:
    """Apply confidence penalty based on detected contradictions.

    High-confidence contradictions (>0.8) indicate unreliable information
    in the research, so we reduce the overall confidence score.

    Args:
        confidence_score: Current confidence score (0.0-1.0).
        report: ContradictionReport with detected contradictions.

    Returns:
        Adjusted confidence score after applying penalty.
    """
    if not report.has_contradictions:
        return confidence_score

    penalty = report.confidence_penalty
    if penalty > 0:
        logger.info(
            "Applying %.0f%% confidence penalty for %d high-confidence contradiction(s)",
            penalty * 100,
            report.high_confidence_count,
        )

    adjusted = max(0.0, confidence_score - penalty)
    return adjusted


__all__ = [
    "Contradiction",
    "ContradictionReport",
    "HIGH_CONFIDENCE_THRESHOLD",
    "CONFIDENCE_PENALTY_PER_CONTRADICTION",
    "build_contradiction_detection_prompt",
    "parse_contradiction_response",
    "detect_contradictions",
    "apply_contradiction_penalty",
]
