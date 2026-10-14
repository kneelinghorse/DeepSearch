"""Span-aware citation rate calculation utilities (B3.2).

This module provides accurate citation detection using markdown AST parsing
instead of simple regex patterns. It handles:
- Inline links: [text](url)
- Reference-style links: [text][ref] and [text]
- Autolinks: <https://example.com>
- Academic citations: (Author, Year)
- Numbered references: [1], [2]
- Bare URLs: https://example.com
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token

    _HAS_MARKDOWN_IT = True
except ImportError:
    _HAS_MARKDOWN_IT = False
    Token = Any  # type: ignore[misc, assignment]

__all__ = [
    "CitationRateResult",
    "CitationDetail",
    "calculate_citation_rate",
    "SpanAwareCitationParser",
]

# Patterns for non-link citation formats
_ACADEMIC_CITATION = re.compile(r"\([^)]+?,\s*\d{4}[a-z]?\)")  # (Author, 2024)
_NUMBERED_REF = re.compile(r"\[\d+\]")  # [1], [42]
_BARE_URL = re.compile(r"https?://[^\s<>\[\]()]+")  # Bare URLs in text

# Sentence tokenization (improved)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(slots=True)
class CitationDetail:
    """Per-sentence citation metadata."""

    sentence: str
    sentence_index: int
    has_citation: bool
    citation_count: int
    citation_types: List[str] = field(default_factory=list)
    citation_urls: List[str] = field(default_factory=list)


@dataclass(slots=True)
class CitationRateResult:
    """Structured summary of citation coverage within a report."""

    citation_rate: float
    total_sentences: int
    cited_sentences: int
    meets_threshold: bool
    sentences_without_citations: List[str] = field(default_factory=list)
    # Extended fields for B3.2
    sentence_details: List[CitationDetail] = field(default_factory=list)
    unique_sources: int = 0
    duplicate_citations: int = 0


class SpanAwareCitationParser:
    """AST-based citation parser using markdown-it-py.

    This parser walks the markdown AST to accurately detect:
    - Inline links [text](url)
    - Reference links [text][ref] and [text]
    - Autolinks <url>
    - Image links ![alt](url)

    It then combines with regex-based detection for:
    - Academic citations (Author, Year)
    - Numbered references [1]
    - Bare URLs
    """

    def __init__(self) -> None:
        if not _HAS_MARKDOWN_IT:
            raise ImportError("markdown-it-py is required for SpanAwareCitationParser")
        # Don't use linkify as it requires linkify-it-py
        # We detect bare URLs via regex instead
        self._md = MarkdownIt("commonmark")

    def extract_links(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Extract all links from markdown with position information.

        Returns:
            List of dicts with: url, text, type, line
        """
        tokens = self._md.parse(markdown_text)
        links: List[Dict[str, Any]] = []
        self._walk_tokens(tokens, links, 0)
        return links

    def _walk_tokens(
        self, tokens: List[Token], links: List[Dict[str, Any]], depth: int = 0
    ) -> None:
        """Recursively walk token tree to find links."""
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Handle inline tokens which contain the actual link nodes
            if token.type == "inline" and token.children:
                self._walk_tokens(token.children, links, depth + 1)
                i += 1
                continue

            if token.type == "link_open":
                href = token.attrGet("href") or ""
                title = ""
                # Collect inline content until link_close
                j = i + 1
                inline_text: List[str] = []
                while j < len(tokens) and tokens[j].type != "link_close":
                    if tokens[j].type == "text":
                        inline_text.append(tokens[j].content or "")
                    j += 1
                title = "".join(inline_text)
                links.append({
                    "url": href,
                    "text": title,
                    "type": "inline_link",
                    "line": token.map[0] if token.map else 0,
                })
                i = j + 1
                continue

            if token.type == "autolink":
                links.append({
                    "url": token.content or "",
                    "text": token.content or "",
                    "type": "autolink",
                    "line": token.map[0] if token.map else 0,
                })

            if token.type == "image":
                src = token.attrGet("src") or ""
                if src:
                    links.append({
                        "url": src,
                        "text": token.content or "",
                        "type": "image",
                        "line": token.map[0] if token.map else 0,
                    })

            # Recurse into children if they exist
            if token.children:
                self._walk_tokens(token.children, links, depth + 1)

            i += 1

    def analyze(self, markdown_text: str, threshold: float = 0.8) -> CitationRateResult:
        """Full citation analysis with sentence-level details.

        Args:
            markdown_text: Research report in markdown format.
            threshold: Minimum citation rate to pass (0.0-1.0).

        Returns:
            CitationRateResult with complete analysis.
        """
        if not markdown_text or not markdown_text.strip():
            return CitationRateResult(
                citation_rate=0.0,
                total_sentences=0,
                cited_sentences=0,
                meets_threshold=False,
            )

        # Extract AST-based links
        ast_links = self.extract_links(markdown_text)
        all_urls: Set[str] = {link["url"] for link in ast_links if link.get("url")}

        # Split into sentences
        sentences = self._tokenize_sentences(markdown_text)

        # Build sentence citation details
        details: List[CitationDetail] = []
        cited_count = 0
        uncited: List[str] = []
        duplicate_count = 0
        url_counts: Dict[str, int] = {}

        for idx, sentence in enumerate(sentences):
            citation_info = self._analyze_sentence(sentence, all_urls)
            detail = CitationDetail(
                sentence=sentence[:200] if len(sentence) > 200 else sentence,
                sentence_index=idx,
                has_citation=citation_info["has_citation"],
                citation_count=citation_info["count"],
                citation_types=citation_info["types"],
                citation_urls=citation_info["urls"],
            )
            details.append(detail)

            if citation_info["has_citation"]:
                cited_count += 1
                for url in citation_info["urls"]:
                    url_counts[url] = url_counts.get(url, 0) + 1
            else:
                if len(uncited) < 20:
                    uncited.append(sentence[:150] if len(sentence) > 150 else sentence)

        # Count duplicates (URLs used more than once)
        for count in url_counts.values():
            if count > 1:
                duplicate_count += count - 1

        total = len(sentences)
        rate = cited_count / total if total > 0 else 0.0

        return CitationRateResult(
            citation_rate=rate,
            total_sentences=total,
            cited_sentences=cited_count,
            meets_threshold=rate >= threshold,
            sentences_without_citations=uncited,
            sentence_details=details,
            unique_sources=len(all_urls),
            duplicate_citations=duplicate_count,
        )

    @staticmethod
    def _strip_boilerplate_sections(text: str) -> str:
        """Remove known report template sections that cannot contain citations.

        These sections (metadata header, telemetry tables, Sources Consulted, etc.)
        are generated by the report template, not the LLM synthesis. Counting them
        as uncited content artificially deflates the citation rate.
        """
        _BOILERPLATE_HEADERS = {
            "telemetry", "sources consulted", "mission context",
            "references", "related research", "quality gates",
        }
        lines = text.split("\n")
        result: List[str] = []
        skip = False
        for line in lines:
            stripped = line.strip()
            # Check if this is a ## section header
            if stripped.startswith("## "):
                section_name = stripped[3:].strip().lower()
                if section_name in _BOILERPLATE_HEADERS:
                    skip = True
                    continue
                else:
                    skip = False
            if skip:
                continue
            # Skip table rows (| col | col |)
            if stripped.startswith("|") and stripped.endswith("|"):
                continue
            # Skip horizontal rules
            if re.match(r"^-{3,}$", stripped):
                continue
            # Skip bold metadata lines (**Generated:** ..., **Research Loops:** ...)
            if re.match(r"^\*\*[^*]+:\*\*\s", stripped):
                continue
            result.append(line)
        return "\n".join(result)

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Improved sentence tokenization that handles markdown better."""
        # Strip report boilerplate before tokenizing
        text = self._strip_boilerplate_sections(text)

        # Remove code blocks before tokenizing
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`[^`]+`", " ", text)

        # Remove headers (they're not sentences to cite)
        text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)

        sentences: List[str] = []

        # Split by newlines first to preserve list items
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Handle list items - each is a separate sentence
            if line.startswith("-") or line.startswith("*") or re.match(r"^\d+\.", line):
                # Remove list marker
                cleaned = re.sub(r"^[-*]\s*", "", line)
                cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
                cleaned = cleaned.strip()
                if len(cleaned) >= 10:
                    sentences.append(cleaned)
                continue

            # For regular paragraphs, split by sentence boundaries
            parts = _SENTENCE_END.split(line)
            for part in parts:
                cleaned = " ".join(part.split()).strip()
                if len(cleaned) >= 10:
                    sentences.append(cleaned)

        return sentences

    def _analyze_sentence(
        self, sentence: str, known_urls: Set[str]
    ) -> Dict[str, Any]:
        """Analyze a single sentence for citations."""
        types: List[str] = []
        urls: List[str] = []
        count = 0

        # Check for inline markdown links [text](url)
        inline_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", sentence)
        for text, url in inline_links:
            types.append("inline_link")
            urls.append(url)
            count += 1

        # Check for reference links [text][ref] or [text]
        ref_links = re.findall(r"\[([^\]]+)\]\[[^\]]*\]|\[([^\]]+)\](?!\()", sentence)
        for match in ref_links:
            text = match[0] or match[1]
            if text and not text.isdigit():  # Exclude numbered refs handled below
                types.append("reference_link")
                count += 1

        # Check for numbered references [1], [2]
        numbered = _NUMBERED_REF.findall(sentence)
        for num in numbered:
            types.append("numbered_reference")
            count += 1

        # Check for academic citations (Author, Year)
        academic = _ACADEMIC_CITATION.findall(sentence)
        for cite in academic:
            types.append("academic_citation")
            count += 1

        # Check for bare URLs
        bare = _BARE_URL.findall(sentence)
        for url in bare:
            if url not in urls:  # Avoid double-counting
                types.append("bare_url")
                urls.append(url)
                count += 1

        # Check if sentence contains any known URL from AST
        for url in known_urls:
            if url in sentence and url not in urls:
                types.append("ast_detected")
                urls.append(url)
                count += 1

        return {
            "has_citation": count > 0,
            "count": count,
            "types": types,
            "urls": urls,
        }


# Module-level parser instance (lazy init)
_parser: Optional[SpanAwareCitationParser] = None


def _get_parser() -> SpanAwareCitationParser:
    """Get or create the module-level parser instance."""
    global _parser
    if _parser is None:
        _parser = SpanAwareCitationParser()
    return _parser


def calculate_citation_rate(
    report_text: str,
    *,
    threshold: float = 0.8,
    use_legacy: bool = False,
) -> CitationRateResult:
    """Compute the percentage of sentences that include citation markers.

    This function uses the span-aware AST parser by default (B3.2). Set
    use_legacy=True to use the original regex-based approach for comparison.

    Args:
        report_text: Markdown text produced by the research agent.
        threshold: Minimum acceptable fraction (0.0-1.0) required to pass.
        use_legacy: Use original regex-only detection (for A/B testing).

    Returns:
        CitationRateResult describing citation coverage and details.
    """
    if use_legacy:
        from deepsearch.validation.legacy_citations import (
            calculate_legacy_citation_rate,
            LegacyCitationResult,
        )
        legacy_result = calculate_legacy_citation_rate(report_text, threshold=threshold)
        # Convert to new result format
        return CitationRateResult(
            citation_rate=legacy_result.citation_rate,
            total_sentences=legacy_result.total_sentences,
            cited_sentences=legacy_result.cited_sentences,
            meets_threshold=legacy_result.meets_threshold,
            sentences_without_citations=legacy_result.sentences_without_citations,
        )

    try:
        parser = _get_parser()
        return parser.analyze(report_text, threshold=threshold)
    except ImportError:
        # Fallback if markdown-it-py not available
        from deepsearch.validation.legacy_citations import (
            calculate_legacy_citation_rate,
        )
        legacy_result = calculate_legacy_citation_rate(report_text, threshold=threshold)
        return CitationRateResult(
            citation_rate=legacy_result.citation_rate,
            total_sentences=legacy_result.total_sentences,
            cited_sentences=legacy_result.cited_sentences,
            meets_threshold=legacy_result.meets_threshold,
            sentences_without_citations=legacy_result.sentences_without_citations,
        )
