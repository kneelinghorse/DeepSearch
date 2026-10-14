"""Unit tests for B16.1 source content pipeline changes.

Validates that source content is preserved through the pipeline
instead of being truncated to 100 chars.
"""

from __future__ import annotations

from deepsearch.tools.exa_search import ExaResult, _title_from_url
from deepsearch.tools.google_search import (
    GoogleSearchResult,
    GroundingChunk,
    GroundingSupport,
)

# =============================================================================
# ExaResult.to_source() tests
# =============================================================================


class TestExaSourceContent:
    """Verify Exa preserves substantive source content."""

    def test_text_preserved_up_to_2000_chars(self):
        """to_source() keeps up to 2000 chars of article text."""
        long_text = "x" * 3000
        result = ExaResult(url="https://example.com", title="Test", text=long_text)
        source = result.to_source()
        assert len(source["snippet"]) == 2000

    def test_short_text_not_truncated(self):
        """Short text preserved in full."""
        result = ExaResult(url="https://example.com", title="Test", text="Short content")
        source = result.to_source()
        assert source["snippet"] == "Short content"

    def test_highlights_use_four(self):
        """to_source() uses up to 4 highlights."""
        highlights = ["h1", "h2", "h3", "h4", "h5"]
        result = ExaResult(
            url="https://example.com", title="Test", highlights=highlights
        )
        source = result.to_source()
        assert source["snippet"] == "h1 ... h2 ... h3 ... h4"

    def test_highlights_preferred_over_text(self):
        """Highlights take priority over raw text."""
        result = ExaResult(
            url="https://example.com",
            title="Test",
            text="raw text here",
            highlights=["highlight one"],
        )
        source = result.to_source()
        assert source["snippet"] == "highlight one"


class TestExaTitleFallback:
    """S52.2 / D-S51-B: Empty Exa titles must derive from URL, not literal placeholder.

    Real-shape evidence from S51-linkup-mm-01 (paid smoke 2026-04-25): Exa
    returned empty title for arxiv PDF endpoints such as
    ``https://arxiv.org/pdf/2501.09136``, causing the references list to render
    rows like ``[2] Exa Result - https://arxiv.org/pdf/2501.09136``. The same
    defect was visible in S25-linkup-mm-01 across 14 separate references —
    pre-existing pre-S40, surviving S50.2 because that change migrated source
    citation resolution but not source title resolution.
    """

    def test_arxiv_pdf_url_yields_arxiv_id_title_when_exa_omits_title(self):
        result = ExaResult(url="https://arxiv.org/pdf/2501.09136", title="")
        source = result.to_source()
        assert source["title"] == "arxiv:2501.09136"
        assert "Exa Result" not in source["title"]

    def test_arxiv_pdf_url_with_version_suffix_preserved(self):
        result = ExaResult(url="https://arxiv.org/pdf/2501.09136v4", title="")
        source = result.to_source()
        assert source["title"] == "arxiv:2501.09136v4"

    def test_arxiv_pdf_url_with_dot_pdf_extension_strips_extension(self):
        result = ExaResult(url="https://arxiv.org/pdf/2501.09136.pdf", title="")
        source = result.to_source()
        assert source["title"] == "arxiv:2501.09136"

    def test_arxiv_abs_url_yields_arxiv_id_title_when_exa_omits_title(self):
        result = ExaResult(url="https://arxiv.org/abs/2505.20096", title="")
        source = result.to_source()
        assert source["title"] == "arxiv:2505.20096"

    def test_arxiv_html_url_yields_arxiv_id_title(self):
        result = ExaResult(url="https://arxiv.org/html/2501.09136v4", title="")
        source = result.to_source()
        assert source["title"] == "arxiv:2501.09136v4"

    def test_non_arxiv_url_yields_hostname_title_when_exa_omits_title(self):
        result = ExaResult(
            url="https://huggingface.co/learn/cookbook/multiagent_rag_system",
            title="",
        )
        source = result.to_source()
        assert source["title"] == "Source from huggingface.co"
        assert "Exa Result" not in source["title"]

    def test_www_prefix_stripped_from_hostname_fallback(self):
        result = ExaResult(url="https://www.example.com/path", title="")
        source = result.to_source()
        assert source["title"] == "Source from example.com"

    def test_explicit_title_takes_precedence_over_url_derivation(self):
        result = ExaResult(
            url="https://arxiv.org/pdf/2501.09136",
            title="Agentic RAG Survey",
        )
        source = result.to_source()
        assert source["title"] == "Agentic RAG Survey"

    def test_whitespace_only_title_treated_as_empty(self):
        result = ExaResult(url="https://arxiv.org/pdf/2501.09136", title="   ")
        source = result.to_source()
        assert source["title"] == "arxiv:2501.09136"

    def test_title_from_url_helper_handles_edge_cases(self):
        """Direct unit coverage for the URL-derivation helper."""
        assert _title_from_url("") == ""
        assert _title_from_url("https://arxiv.org/pdf/2412.05838") == "arxiv:2412.05838"
        assert _title_from_url("HTTPS://ARXIV.ORG/ABS/2501.09136") == "arxiv:2501.09136"
        assert _title_from_url("https://example.io/page") == "Source from example.io"
        # Malformed URL with no host but a scheme still degrades gracefully.
        assert _title_from_url("not a url") == ""

    def test_no_exa_result_literal_emitted_for_typed_real_world_urls(self):
        """Anti-regression sweep: every URL shape from S51 evidence resolves cleanly."""
        s51_evidence_urls = [
            "https://arxiv.org/pdf/2501.09136",
            "https://arxiv.org/pdf/2603.07379",
            "https://arxiv.org/pdf/2503.13514",
            "https://arxiv.org/pdf/2501.15228",
            "https://link.springer.com/chapter/10.1007/978-3-032-05179-0_22",
            "https://huggingface.co/learn/cookbook/multiagent_rag_system",
        ]
        for url in s51_evidence_urls:
            source = ExaResult(url=url, title="").to_source()
            assert source["title"] != "Exa Result", (
                f"URL {url} fell back to 'Exa Result' literal"
            )
            assert source["title"], f"URL {url} produced empty title"


# =============================================================================
# GoogleSearchResult.to_enriched_sources() tests
# =============================================================================


class TestGoogleEnrichedSources:
    """Verify Google grounding produces real snippets."""

    def _make_result(
        self,
        text: str = "The API supports OAuth 2.0 and API key authentication.",
        sources: list[GroundingChunk] | None = None,
        supports: list[GroundingSupport] | None = None,
    ) -> GoogleSearchResult:
        if sources is None:
            sources = [GroundingChunk(uri="https://example.com", title="Docs")]
        if supports is None:
            supports = [GroundingSupport(start_index=0, end_index=len(text), chunk_indices=[0])]
        return GoogleSearchResult(
            text=text, sources=sources, supports=supports, queries_used=["test"]
        )

    def test_maps_supports_to_snippets(self):
        """Grounding supports are mapped to source snippets."""
        result = self._make_result()
        enriched = result.to_enriched_sources()
        assert len(enriched) == 1
        assert enriched[0]["snippet"] != ""
        assert "OAuth" in enriched[0]["snippet"]

    def test_fallback_to_full_text_when_no_supports(self):
        """Without supports, uses full response text as fallback."""
        result = self._make_result(supports=[])
        enriched = result.to_enriched_sources()
        assert len(enriched) == 1
        assert enriched[0]["snippet"] != ""
        assert "OAuth" in enriched[0]["snippet"]

    def test_multiple_supports_concatenated(self):
        """Multiple support spans for same source are joined."""
        text = "OAuth 2.0 is standard. API keys are simpler."
        result = self._make_result(
            text=text,
            supports=[
                GroundingSupport(start_index=0, end_index=22, chunk_indices=[0]),
                GroundingSupport(start_index=23, end_index=45, chunk_indices=[0]),
            ],
        )
        enriched = result.to_enriched_sources()
        assert "OAuth" in enriched[0]["snippet"]
        assert "API keys" in enriched[0]["snippet"]

    def test_max_snippet_len_respected(self):
        """Snippet is truncated to max_snippet_len."""
        long_text = "x" * 5000
        result = self._make_result(
            text=long_text,
            supports=[GroundingSupport(start_index=0, end_index=5000, chunk_indices=[0])],
        )
        enriched = result.to_enriched_sources(max_snippet_len=500)
        assert len(enriched[0]["snippet"]) <= 500

    def test_preserves_url_and_title(self):
        """Source metadata preserved in enriched output."""
        result = self._make_result()
        enriched = result.to_enriched_sources()
        assert enriched[0]["url"] == "https://example.com"
        assert enriched[0]["title"] == "Docs"

    def test_empty_text_empty_snippet(self):
        """Empty text with no supports produces empty snippet."""
        result = GoogleSearchResult(
            text="",
            sources=[GroundingChunk(uri="https://example.com", title="Docs")],
            supports=[],
            queries_used=["test"],
        )
        enriched = result.to_enriched_sources()
        assert enriched[0]["snippet"] == ""


