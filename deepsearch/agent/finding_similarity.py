"""Shared finding-similarity helpers for deduplication and clustering."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

_CITATION_TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]\(https?://[^\s)]+\)"
    r"|\[\d+\]"
    r"|\(https?://[^)\s]+\)"
    r"|https?://[^\s)]+"
)
_URL_PATTERN = re.compile(r"https?://[^\s)\]]+")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{1,}")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


def normalize_finding_text(text: str) -> str:
    """Collapse repeated whitespace in finding text."""

    return " ".join(text.split())


def unique_strings(values: Iterable[object]) -> List[str]:
    """Return unique non-empty string values in first-seen order."""

    items: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_finding_text(str(value or ""))
        if not text or text in seen:
            continue
        items.append(text)
        seen.add(text)
    return items


def extract_citation_tokens(text: str) -> List[str]:
    """Return citation-like tokens from text in first-seen order."""

    seen: set[str] = set()
    citations: List[str] = []
    for match in _CITATION_TOKEN_PATTERN.finditer(text):
        token = match.group(0).strip()
        if token and token not in seen:
            citations.append(token)
            seen.add(token)
    return citations


def extract_urls(text: str) -> List[str]:
    """Return URLs from text in first-seen order."""

    return unique_strings(match.group(0) for match in _URL_PATTERN.finditer(text))


def extract_urls_from_values(values: Iterable[object]) -> List[str]:
    """Return unique URLs extracted from a mixed collection of values."""

    urls: List[str] = []
    seen: set[str] = set()
    for value in values:
        for url in extract_urls(str(value or "")):
            if url in seen:
                continue
            urls.append(url)
            seen.add(url)
    return urls


def strip_citation_tokens(text: str) -> str:
    """Remove citation markers and bare URLs from text."""

    stripped = _CITATION_TOKEN_PATTERN.sub("", text)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = re.sub(r"([(\[])\s+", r"\1", stripped)
    stripped = re.sub(r"\s+([)\]])", r"\1", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped.strip()


def content_tokens(text: str) -> frozenset[str]:
    """Return normalized content tokens with stopwords removed."""

    return frozenset(
        token.lower()
        for token in _WORD_PATTERN.findall(text)
        if token.lower() not in _STOPWORDS
    )


def jaccard_similarity(
    left: Sequence[str] | frozenset[str],
    right: Sequence[str] | frozenset[str],
) -> float:
    """Return Jaccard similarity for two token collections."""

    left_set = frozenset(left)
    right_set = frozenset(right)
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


__all__ = [
    "content_tokens",
    "extract_citation_tokens",
    "extract_urls",
    "extract_urls_from_values",
    "jaccard_similarity",
    "normalize_finding_text",
    "strip_citation_tokens",
    "unique_strings",
]
