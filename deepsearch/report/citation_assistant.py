"""LLM-assisted citation placement for improved citation rates.

This module provides AI-powered citation suggestions for research findings
that couldn't be matched using keyword or semantic matching. It uses
local Ollama models to suggest which sources should cite which text segments.

Usage:
    assistant = CitationAssistant(llm_client)
    suggestions = assistant.suggest_citations(uncited_text, sources)

    # Or batch process for efficiency
    suggestions = batch_suggest_citations(uncited_findings, sources, llm_client)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CitationSuggestion:
    """A suggested citation from the LLM."""

    text_segment: str
    source: Mapping[str, object]
    confidence: float
    source_index: int


class CitationAssistant:
    """LLM-powered citation suggestion engine.

    Uses a local Ollama model to analyze uncited text and suggest
    which sources are most relevant for citation.

    Attributes:
        llm_client: Ollama client or compatible LLM interface
        min_confidence: Minimum confidence threshold for suggestions
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        min_confidence: float = 0.7,
    ):
        """Initialize the citation assistant.

        Args:
            llm_client: LLM client with generate() method
            min_confidence: Minimum confidence for suggestions (0-1)
        """
        self.llm = llm_client
        self.min_confidence = min_confidence

    @property
    def available(self) -> bool:
        """Check if LLM is available for suggestions."""
        return self.llm is not None

    def suggest_citations(
        self,
        uncited_text: str,
        sources: List[Mapping[str, object]],
    ) -> List[CitationSuggestion]:
        """Suggest which sources should cite which text segments.

        Args:
            uncited_text: Text without citations
            sources: Available source documents

        Returns:
            List of CitationSuggestion objects with source and confidence
        """
        if not self.llm or not sources:
            return []

        # Build source summary for LLM (limit to top 10)
        source_summary = self._build_source_summary(sources[:10])

        prompt = self._build_prompt(uncited_text, source_summary)

        try:
            response = self.llm.generate(prompt)
            return self._parse_suggestions(response, sources)
        except Exception as e:
            logger.warning("LLM citation suggestion failed: %s", e)
            return []

    def _build_source_summary(
        self,
        sources: List[Mapping[str, object]],
    ) -> str:
        """Build a concise summary of sources for the LLM prompt."""
        lines = []
        for i, source in enumerate(sources):
            title = str(source.get("title", "Unknown"))[:60]
            snippet = str(source.get("snippet", ""))[:100]
            lines.append(f"{i+1}. {title}: {snippet}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        uncited_text: str,
        source_summary: str,
    ) -> str:
        """Build the LLM prompt for citation suggestions."""
        return f"""Analyze this research text and suggest which source(s) should be cited.

TEXT:
{uncited_text}

AVAILABLE SOURCES:
{source_summary}

For each claim or factual statement in the text, suggest the most relevant source number.
Return a JSON array like this:
[{{"text": "the specific claim text", "source_num": 1, "confidence": 0.9}}]

Rules:
- Only suggest citations where confidence >= 0.7
- Match claims to the most semantically relevant source
- A source can be suggested multiple times if it supports multiple claims
- If no good match exists, don't suggest a citation
- Return ONLY the JSON array, no other text

JSON:"""

    def _parse_suggestions(
        self,
        response: str,
        sources: List[Mapping[str, object]],
    ) -> List[CitationSuggestion]:
        """Parse LLM response into citation suggestions."""
        # Extract JSON from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            logger.debug("No JSON array found in LLM response")
            return []

        try:
            suggestions = json.loads(json_match.group())
            results = []

            for s in suggestions:
                text = s.get("text", "")
                source_num = s.get("source_num", 0) - 1  # Convert to 0-indexed
                confidence = float(s.get("confidence", 0.0))

                if (
                    0 <= source_num < len(sources)
                    and confidence >= self.min_confidence
                    and text
                ):
                    results.append(
                        CitationSuggestion(
                            text_segment=text,
                            source=sources[source_num],
                            confidence=confidence,
                            source_index=source_num,
                        )
                    )

            logger.debug("Parsed %d citation suggestions from LLM", len(results))
            return results

        except json.JSONDecodeError as e:
            logger.debug("Failed to parse LLM JSON: %s", e)
            return []


def batch_suggest_citations(
    uncited_findings: List[str],
    sources: List[Mapping[str, object]],
    llm_client: Optional[Any] = None,
    batch_size: int = 5,
) -> Dict[int, Tuple[Mapping[str, object], float]]:
    """Batch process uncited findings for efficiency.

    Groups multiple findings into a single LLM call to reduce latency.

    Args:
        uncited_findings: List of uncited finding texts
        sources: Available source documents
        llm_client: LLM client for suggestions
        batch_size: Number of findings per batch (default 5)

    Returns:
        Dict mapping finding index to (source, confidence) tuple
    """
    if not llm_client or not uncited_findings or not sources:
        return {}

    assistant = CitationAssistant(llm_client)
    suggestions: Dict[int, Tuple[Mapping[str, object], float]] = {}

    # Process in batches
    for batch_start in range(0, len(uncited_findings), batch_size):
        batch_end = min(batch_start + batch_size, len(uncited_findings))
        batch = uncited_findings[batch_start:batch_end]

        # Number each finding in the batch for tracking
        batch_text = "\n\n".join(
            f"[FINDING {batch_start + j}]: {finding}"
            for j, finding in enumerate(batch)
        )

        results = assistant.suggest_citations(batch_text, sources)

        # Match suggestions back to original findings
        for suggestion in results:
            # Find which finding this suggestion belongs to
            for j, finding in enumerate(batch):
                global_idx = batch_start + j
                if (
                    global_idx not in suggestions
                    and _text_in_finding(suggestion.text_segment, finding)
                ):
                    suggestions[global_idx] = (
                        suggestion.source,
                        suggestion.confidence,
                    )
                    break

    logger.info(
        "Batch suggested citations for %d/%d uncited findings",
        len(suggestions),
        len(uncited_findings),
    )
    return suggestions


def _text_in_finding(text: str, finding: str) -> bool:
    """Check if suggestion text matches the finding (fuzzy)."""
    text_lower = text.lower().strip()
    finding_lower = finding.lower()

    # Direct substring match
    if text_lower in finding_lower:
        return True

    # Word overlap check (at least 50% of suggestion words in finding)
    text_words = set(text_lower.split())
    finding_words = set(finding_lower.split())
    if text_words:
        overlap = len(text_words & finding_words) / len(text_words)
        return overlap >= 0.5

    return False


class OllamaClient:
    """Simple Ollama client for citation suggestions.

    This is a lightweight wrapper for Ollama's generate API.
    For production use, consider using the full ollama-python package.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if Ollama service is running."""
        if self._available is None:
            self._available = self._check_availability()
        return self._available

    def _check_availability(self) -> bool:
        """Ping Ollama to check if it's running."""
        try:
            import httpx

            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt.

        Args:
            prompt: The prompt text
            **kwargs: Additional options (temperature, etc.)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If Ollama is not available
        """
        if not self.available:
            raise RuntimeError("Ollama is not available")

        import httpx

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("response", "")


def create_ollama_assistant(
    model: str = "llama3.2",
    min_confidence: float = 0.7,
) -> Optional[CitationAssistant]:
    """Create a CitationAssistant with Ollama backend.

    Returns None if Ollama is not available.

    Args:
        model: Ollama model name (default: llama3.2)
        min_confidence: Minimum confidence threshold

    Returns:
        CitationAssistant instance or None if unavailable
    """
    client = OllamaClient(model=model)
    if not client.available:
        logger.info("Ollama not available, LLM citation assist disabled")
        return None

    return CitationAssistant(llm_client=client, min_confidence=min_confidence)


__all__ = [
    "CitationAssistant",
    "CitationSuggestion",
    "OllamaClient",
    "batch_suggest_citations",
    "create_ollama_assistant",
]
