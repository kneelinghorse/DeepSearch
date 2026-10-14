"""Legacy citation rate calculation utilities (B3.2 baseline).

This module preserves the original regex-based citation detection for A/B
comparison with the new span-aware citation engine. The pattern-based approach
has known limitations:
- Misses inline Markdown hyperlinks [text](url)
- Undercounts reference-style links
- No sentence tokenization (relies on simple splitting)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

__all__ = ["LegacyCitationResult", "calculate_legacy_citation_rate"]


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_CITATION_PATTERN = re.compile(r"\[[^\]]+\]|\([^)]+?,\s*\d{4}\)")


@dataclass(slots=True)
class LegacyCitationResult:
    """Structured summary of citation coverage (legacy algorithm)."""

    citation_rate: float
    total_sentences: int
    cited_sentences: int
    meets_threshold: bool
    sentences_without_citations: List[str] = field(default_factory=list)


def calculate_legacy_citation_rate(
    report_text: str, *, threshold: float = 0.8
) -> LegacyCitationResult:
    """Legacy citation rate calculation using regex pattern matching.

    This function preserves the original Sprint 02 implementation for
    regression comparison. See citation_rate.py for the improved span-aware
    algorithm.

    Args:
        report_text: Markdown text produced by the research agent.
        threshold: Minimum acceptable fraction (0.0-1.0) required to pass.

    Returns:
        LegacyCitationResult describing citation coverage.
    """

    sentences = _tokenize_sentences(report_text)
    cited_sentences = [
        sentence for sentence in sentences if _CITATION_PATTERN.search(sentence)
    ]
    total = len(sentences)
    rate = (len(cited_sentences) / total) if total else 0.0
    uncited = [
        sentence for sentence in sentences if not _CITATION_PATTERN.search(sentence)
    ]
    return LegacyCitationResult(
        citation_rate=rate,
        total_sentences=total,
        cited_sentences=len(cited_sentences),
        meets_threshold=rate >= threshold,
        sentences_without_citations=uncited[:20],
    )


def _tokenize_sentences(report_text: str) -> List[str]:
    text = (report_text or "").replace("\n", " ").strip()
    if not text:
        return []
    candidates = _SENTENCE_SPLIT_PATTERN.split(text)
    sentences: List[str] = []
    for candidate in candidates:
        sentence = candidate.strip()
        if not sentence:
            continue
        if len(sentence) < 4:
            continue
        sentences.append(sentence)
    return sentences
