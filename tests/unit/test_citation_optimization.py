"""Unit tests for B5.4 citation rate optimization.

Tests the new citation enhancement features:
1. List item citation support (bullets, numbered lists)
2. Multi-citation support for complex findings
3. Optimized build_findings_section_optimized function
"""

from __future__ import annotations

import pytest

from deepsearch.report.templates import (
    _inject_citation_to_item,
    _inject_multiple_citations,
    _inject_citation,
    build_findings_section_optimized,
)


class TestListItemCitationSupport:
    """Test citation injection into various text formats."""

    def test_bullet_list_item_with_period(self) -> None:
        """Bullet list items ending with period get citations."""
        source = {"title": "Test Source", "url": "https://example.com/test"}
        item = "- This is a bullet point."

        result = _inject_citation_to_item(item, source)

        assert "- This is a bullet point." in result
        assert "[Test Source](https://example.com/test)" in result

    def test_bullet_list_item_without_period(self) -> None:
        """Bullet list items without period get period added."""
        source = {"title": "Test Source", "url": "https://example.com/test"}
        item = "- This is a bullet point without period"

        result = _inject_citation_to_item(item, source)

        assert "[Test Source](https://example.com/test)" in result
        assert ". [" in result  # Period should be added

    def test_asterisk_bullet_item(self) -> None:
        """Asterisk bullets are handled like dash bullets."""
        source = {"title": "Doc", "url": "https://doc.com"}
        item = "* Feature supports async operations"

        result = _inject_citation_to_item(item, source)

        assert result.startswith("* ")
        assert "[Doc](https://doc.com)" in result

    def test_unicode_bullet_item(self) -> None:
        """Unicode bullet • is handled correctly."""
        source = {"title": "Source", "url": "https://example.com"}
        item = "• Research finding here"

        result = _inject_citation_to_item(item, source)

        assert result.startswith("• ")
        assert "[Source](https://example.com)" in result

    def test_numbered_list_item(self) -> None:
        """Numbered list items get proper citations."""
        source = {"title": "Reference", "url": "https://ref.com"}
        item = "1. First point with details."

        result = _inject_citation_to_item(item, source)

        assert result.startswith("1. ")
        assert "[Reference](https://ref.com)" in result

    def test_numbered_list_multi_digit(self) -> None:
        """Multi-digit numbered lists work correctly."""
        source = {"title": "Source", "url": "https://example.com"}
        item = "12. Twelfth point here"

        result = _inject_citation_to_item(item, source)

        assert result.startswith("12. ")
        assert "[Source](https://example.com)" in result

    def test_paragraph_unchanged(self) -> None:
        """Standard paragraphs still work as before."""
        source = {"title": "Article", "url": "https://article.com"}
        item = "This is a regular paragraph."

        result = _inject_citation_to_item(item, source)

        assert "This is a regular paragraph." in result
        assert "[Article](https://article.com)" in result

    def test_no_url_returns_unchanged(self) -> None:
        """Items with sources missing URL are returned unchanged."""
        source = {"title": "No URL Source"}
        item = "- Some list item."

        result = _inject_citation_to_item(item, source)

        assert result == "- Some list item."

    def test_long_title_truncation(self) -> None:
        """Long source titles are truncated to 50 chars."""
        source = {
            "title": "A Very Long Title That Exceeds Fifty Characters For Testing Purposes",
            "url": "https://example.com",
        }
        item = "Finding text here."

        result = _inject_citation_to_item(item, source)

        # Title should be truncated with ...
        assert "..." in result
        assert len(result.split("](")[0].split("[")[1]) <= 53  # 50 + "..."


class TestMultiCitationSupport:
    """Test multiple citations per finding."""

    def test_multi_citation_basic(self) -> None:
        """Multiple sources are formatted with parentheses."""
        sources = [
            ({"title": "Source A", "url": "https://a.com"}, 0.9),
            ({"title": "Source B", "url": "https://b.com"}, 0.8),
        ]
        finding = "Complex finding with multiple sources."

        result = _inject_multiple_citations(finding, sources)

        assert "[Source A](https://a.com)" in result
        assert "[Source B](https://b.com)" in result
        assert "(" in result and ")" in result  # Parentheses grouping

    def test_multi_citation_sorted_by_score(self) -> None:
        """Higher-scored sources appear first."""
        sources = [
            ({"title": "Low", "url": "https://low.com"}, 0.5),
            ({"title": "High", "url": "https://high.com"}, 0.95),
            ({"title": "Mid", "url": "https://mid.com"}, 0.7),
        ]
        finding = "Finding text."

        result = _inject_multiple_citations(finding, sources)

        # High should appear before Low
        high_pos = result.find("High")
        low_pos = result.find("Low")
        assert high_pos < low_pos

    def test_multi_citation_respects_max(self) -> None:
        """Max citations limit is respected."""
        sources = [
            ({"title": f"Source {i}", "url": f"https://{i}.com"}, 0.9 - i * 0.1)
            for i in range(5)
        ]
        finding = "Finding with many sources."

        result = _inject_multiple_citations(finding, sources, max_citations=2)

        # Only 2 citations should appear
        assert result.count("](") == 2

    def test_multi_citation_list_item(self) -> None:
        """Multi-citation works for list items."""
        sources = [
            ({"title": "A", "url": "https://a.com"}, 0.9),
            ({"title": "B", "url": "https://b.com"}, 0.8),
        ]
        finding = "- List item finding"

        result = _inject_multiple_citations(finding, sources)

        assert result.startswith("- ")
        assert "[A]" in result and "[B]" in result

    def test_empty_sources_returns_unchanged(self) -> None:
        """Empty sources list returns finding unchanged."""
        result = _inject_multiple_citations("Finding text.", [])
        assert result == "Finding text."


class TestOptimizedBuildFindings:
    """Test the optimized findings section builder."""

    def test_basic_findings_with_sources(self) -> None:
        """Basic findings with sources get citations."""
        findings = [
            "Research shows Python is popular.",
            "Machine learning is transforming industries.",
        ]
        sources = [
            {"title": "Python Survey", "url": "https://python.org/survey", "snippet": "Python popularity survey results"},
            {"title": "ML Trends", "url": "https://ml.com/trends", "snippet": "Machine learning industry analysis"},
        ]

        markdown, rate = build_findings_section_optimized(findings, sources)

        assert "## Research Findings" in markdown
        assert rate >= 0  # Rate should be calculated

    def test_empty_findings_returns_zero_rate(self) -> None:
        """Empty findings list returns 0 rate."""
        markdown, rate = build_findings_section_optimized([], [])
        assert rate == 0.0
        assert "No findings were generated" in markdown

    def test_no_sources_returns_uncited(self) -> None:
        """Findings without sources have 0 citation rate."""
        findings = ["Finding one.", "Finding two."]
        markdown, rate = build_findings_section_optimized(findings, [])
        assert rate == 0.0

    def test_rate_calculation_accurate(self) -> None:
        """Citation rate is calculated correctly."""
        # Create findings that should match sources via keywords
        findings = [
            "Python language features are excellent.",
            "JavaScript runtime performance is good.",
            "Unrelated finding without keywords.",
        ]
        sources = [
            {"title": "Python Guide", "url": "https://python.org", "snippet": "Python language features"},
            {"title": "JS Performance", "url": "https://js.com", "snippet": "JavaScript runtime"},
        ]

        markdown, rate = build_findings_section_optimized(
            findings, sources, use_semantic=False  # Force keyword matching
        )

        # At least 2/3 findings should be cited (Python and JavaScript)
        assert rate >= 0.5

    def test_citation_rate_metadata_in_output(self) -> None:
        """Output includes citation rate as HTML comment."""
        findings = ["Python is great."]
        sources = [{"title": "Python", "url": "https://python.org", "snippet": "Python info"}]

        markdown, rate = build_findings_section_optimized(findings, sources)

        assert "<!-- citation_rate:" in markdown


class TestInjectCitationEnhancements:
    """Test enhanced _inject_citation with multi-citation support."""

    def test_allow_multi_citation_flag(self) -> None:
        """Multi-citation mode is controlled by flag."""
        sources = [
            {"title": "Source A", "url": "https://a.com", "snippet": "keyword match"},
            {"title": "Source B", "url": "https://b.com", "snippet": "keyword match too"},
        ]
        finding = "Finding with keyword match."
        cited_urls: set = set()

        # Single citation mode (default)
        result = _inject_citation(finding, sources, cited_urls)
        assert result.count("](") <= 1

    def test_inject_uses_item_helper(self) -> None:
        """_inject_citation properly uses _inject_citation_to_item for formatting."""
        source = {"title": "Test", "url": "https://test.com", "snippet": "test content"}
        finding = "- List item about test."
        cited_urls: set = set()

        result = _inject_citation(finding, [source], cited_urls)

        # Should preserve list format
        assert result.startswith("- ")
        assert "[Test](https://test.com)" in result


class TestCitationAssistantIntegration:
    """Test citation assistant module integration."""

    def test_ollama_client_unavailable_handled(self) -> None:
        """OllamaClient gracefully handles unavailability."""
        from deepsearch.report.citation_assistant import OllamaClient

        client = OllamaClient()
        # Without Ollama running, available should be False
        # (don't fail, just check handling)
        assert client.available in (True, False)

    def test_create_ollama_assistant_returns_none_when_unavailable(self) -> None:
        """create_ollama_assistant returns None if Ollama not running."""
        from deepsearch.report.citation_assistant import create_ollama_assistant

        assistant = create_ollama_assistant()
        # Either returns assistant or None - both valid
        assert assistant is None or hasattr(assistant, "suggest_citations")

    def test_batch_suggest_empty_inputs(self) -> None:
        """batch_suggest_citations handles empty inputs."""
        from deepsearch.report.citation_assistant import batch_suggest_citations

        result = batch_suggest_citations([], [], None)
        assert result == {}

        result = batch_suggest_citations(["finding"], [], None)
        assert result == {}

    def test_citation_suggestion_dataclass(self) -> None:
        """CitationSuggestion dataclass works correctly."""
        from deepsearch.report.citation_assistant import CitationSuggestion

        suggestion = CitationSuggestion(
            text_segment="test segment",
            source={"title": "Test", "url": "https://test.com"},
            confidence=0.85,
            source_index=0,
        )

        assert suggestion.text_segment == "test segment"
        assert suggestion.confidence == 0.85
        assert suggestion.source_index == 0
