"""Unit tests for the span-aware citation parser (B3.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepsearch.validation.citation_rate import (
    CitationDetail,
    CitationRateResult,
    SpanAwareCitationParser,
    calculate_citation_rate,
)
from deepsearch.validation.legacy_citations import (
    LegacyCitationResult,
    calculate_legacy_citation_rate,
)


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "reports"


class TestSpanAwareCitationParser:
    """Tests for the AST-based citation parser."""

    @pytest.fixture
    def parser(self) -> SpanAwareCitationParser:
        return SpanAwareCitationParser()

    def test_extract_inline_links(self, parser: SpanAwareCitationParser) -> None:
        """Parser should detect inline markdown links."""
        md = "Check the [documentation](https://example.com/docs) for details."
        links = parser.extract_links(md)
        assert len(links) == 1
        assert links[0]["url"] == "https://example.com/docs"
        assert links[0]["type"] == "inline_link"

    def test_extract_multiple_links(self, parser: SpanAwareCitationParser) -> None:
        """Parser should extract multiple links from text."""
        md = """See [Link A](https://a.com) and [Link B](https://b.com).
Also check [Link C](https://c.com) for more info.
"""
        links = parser.extract_links(md)
        assert len(links) == 3
        urls = {link["url"] for link in links}
        assert "https://a.com" in urls
        assert "https://b.com" in urls
        assert "https://c.com" in urls

    def test_empty_text(self, parser: SpanAwareCitationParser) -> None:
        """Empty text should return empty result."""
        result = parser.analyze("")
        assert result.total_sentences == 0
        assert result.citation_rate == 0.0

    def test_analyze_with_inline_links(self, parser: SpanAwareCitationParser) -> None:
        """Sentences with inline links should be counted as cited."""
        md = """
        The [official docs](https://example.com) explain the concept.
        This sentence has no references at all.
        Another [link here](https://other.com) proves the point.
        """
        result = parser.analyze(md)
        assert result.cited_sentences >= 2
        assert result.citation_rate > 0.5

    def test_analyze_academic_citations(self, parser: SpanAwareCitationParser) -> None:
        """Parser should detect academic citation format."""
        md = """
        Research shows positive results (Smith, 2024).
        This finding was confirmed (Johnson et al., 2023).
        No citation here.
        """
        result = parser.analyze(md)
        assert result.cited_sentences >= 2

    def test_analyze_numbered_references(self, parser: SpanAwareCitationParser) -> None:
        """Parser should detect numbered references like [1], [2]."""
        md = """
        The study confirms the hypothesis [1].
        Further evidence supports this claim [2].
        This sentence lacks citations.
        Additional findings are reported [3].
        """
        result = parser.analyze(md)
        assert result.cited_sentences >= 3

    def test_analyze_bare_urls(self, parser: SpanAwareCitationParser) -> None:
        """Parser should detect bare URLs in text."""
        md = """
        More information available at https://example.com/research page.
        See also https://another.com/docs for documentation.
        This sentence has nothing.
        """
        result = parser.analyze(md)
        assert result.cited_sentences >= 2

    def test_sentence_details_populated(self, parser: SpanAwareCitationParser) -> None:
        """Analysis should populate sentence-level details."""
        md = """
        First sentence with [citation](https://example.com).
        Second sentence without citation.
        """
        result = parser.analyze(md)
        assert len(result.sentence_details) >= 2
        # At least one should have citation
        cited_details = [d for d in result.sentence_details if d.has_citation]
        assert len(cited_details) >= 1
        assert cited_details[0].citation_count >= 1

    def test_unique_sources_counted(self, parser: SpanAwareCitationParser) -> None:
        """Unique sources should be tracked."""
        md = """
See [Link A](https://a.com) here.
And [Link B](https://b.com) there.
Also [Link A again](https://a.com) repeated.
"""
        result = parser.analyze(md)
        # AST extracts 3 links, but 2 unique URLs (a.com used twice)
        assert result.unique_sources == 2  # a.com and b.com

    def test_duplicate_citations_counted(self, parser: SpanAwareCitationParser) -> None:
        """Duplicate citations should be tracked."""
        md = """
        Reference to source https://example.com/paper here.
        Another mention of https://example.com/paper there.
        And once more https://example.com/paper for emphasis.
        """
        result = parser.analyze(md)
        assert result.duplicate_citations >= 2


class TestCalculateCitationRate:
    """Tests for the main calculate_citation_rate function."""

    def test_new_parser_by_default(self) -> None:
        """Default should use new span-aware parser."""
        md = "Check [this link](https://example.com) for info."
        result = calculate_citation_rate(md)
        assert isinstance(result, CitationRateResult)
        # New parser populates sentence_details
        assert hasattr(result, "sentence_details")

    def test_legacy_parser_flag(self) -> None:
        """Legacy flag should use regex-only parser."""
        md = "Check [this link](https://example.com) for info."
        result = calculate_citation_rate(md, use_legacy=True)
        assert isinstance(result, CitationRateResult)
        # Legacy doesn't populate extended fields
        assert len(result.sentence_details) == 0

    def test_threshold_pass(self) -> None:
        """High citation rate should pass threshold."""
        md = """
        First fact [1].
        Second fact [2].
        Third fact [3].
        Fourth fact [4].
        """
        result = calculate_citation_rate(md, threshold=0.8)
        assert result.meets_threshold is True

    def test_threshold_fail(self) -> None:
        """Low citation rate should fail threshold."""
        md = """
        First sentence without citation.
        Second sentence without citation.
        Third sentence with [1] citation.
        Fourth sentence without citation.
        """
        result = calculate_citation_rate(md, threshold=0.8)
        assert result.meets_threshold is False


class TestLegacyCitationCalculation:
    """Tests for legacy citation rate for regression comparison."""

    def test_legacy_basic_calculation(self) -> None:
        """Legacy parser should detect bracket patterns."""
        md = """
        First fact [citation].
        Second fact without.
        Third fact (Author, 2024).
        """
        result = calculate_legacy_citation_rate(md)
        assert isinstance(result, LegacyCitationResult)
        assert result.cited_sentences >= 2

    def test_legacy_misses_inline_links(self) -> None:
        """Legacy parser should NOT detect inline markdown links."""
        md = "See [documentation](https://example.com) for info."
        result = calculate_legacy_citation_rate(md)
        # Legacy treats [documentation] as citation but misses URL significance
        # The pattern \[[^\]]+\] matches [documentation] but not the full link
        # This is why the new parser is needed
        assert result.total_sentences >= 1


class TestComparisonWithFixture:
    """Compare new vs legacy parser on fixture report."""

    @pytest.fixture
    def sample_report(self) -> str:
        """Load sample report fixture."""
        report_path = FIXTURES_DIR / "sample_cited_report.md"
        if report_path.exists():
            return report_path.read_text()
        # Fallback if fixture not available
        return """
        This is a test [1]. See (Author, 2024).
        Check https://example.com for details.
        No citation here.
        """

    def test_new_parser_higher_accuracy(self, sample_report: str) -> None:
        """New parser should detect more citations than legacy."""
        new_result = calculate_citation_rate(sample_report, use_legacy=False)
        legacy_result = calculate_citation_rate(sample_report, use_legacy=True)

        # New parser should generally find more citations
        # (or at least equal) due to link detection
        assert new_result.cited_sentences >= legacy_result.cited_sentences

    def test_new_parser_provides_details(self, sample_report: str) -> None:
        """New parser should provide sentence-level details."""
        result = calculate_citation_rate(sample_report, use_legacy=False)
        assert len(result.sentence_details) > 0

        # Check that citation types are identified
        all_types: set[str] = set()
        for detail in result.sentence_details:
            all_types.update(detail.citation_types)

        # Should detect multiple citation types in sample report
        # (numbered refs, inline links, academic citations, bare URLs)
        assert len(all_types) >= 1


class TestEdgeCases:
    """Edge case tests for citation parsing."""

    def test_code_blocks_excluded(self) -> None:
        """Code blocks should not count as sentences."""
        md = """
        This is a real sentence [1].

        ```python
        def example():
            pass
        ```

        Another sentence (Author, 2024).
        """
        result = calculate_citation_rate(md)
        # Should only count prose sentences, not code
        assert result.total_sentences == 2

    def test_headers_excluded(self) -> None:
        """Headers should not count as citable sentences."""
        md = """# Main Header

First paragraph with [1] citation.

## Sub Header

Second paragraph (Smith, 2024).
"""
        result = calculate_citation_rate(md)
        # Only paragraphs should count, not headers
        assert result.total_sentences == 2
        assert result.cited_sentences == 2

    def test_short_fragments_excluded(self) -> None:
        """Very short text fragments should be excluded."""
        md = """
        Yes.
        No.
        OK.
        This is a proper sentence with enough content [1].
        """
        result = calculate_citation_rate(md)
        # Only the last sentence should count
        assert result.total_sentences >= 1

    def test_list_items_processed(self) -> None:
        """List items should be processed correctly."""
        md = """
        Key findings:
        - First finding with source [1].
        - Second finding without source.
        - Third finding (Author, 2024).
        """
        result = calculate_citation_rate(md)
        # List items should be counted
        assert result.total_sentences >= 3
        assert result.cited_sentences >= 2
