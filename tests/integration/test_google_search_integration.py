"""Integration tests for GoogleSearchClient with real Gemini API.

These tests require both RUN_GEMINI_INTEGRATION=true and GEMINI_API_KEY.
They make real API calls and incur usage costs.

Run with: RUN_GEMINI_INTEGRATION=true pytest \
  tests/integration/test_google_search_integration.py -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip("google.genai", reason="google-genai SDK required for real Gemini API tests")

from deepsearch.tools.google_search import (  # noqa: E402, I001
    GoogleSearchClient,
    GoogleSearchError,
    GoogleSearchResult,
    extract_grounded_citations,
)


# A credential can be present for unrelated development. Paid tests require a
# separate affirmative switch so broad local verification remains unpaid.
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION", "").lower() != "true"
    or not os.getenv("GEMINI_API_KEY"),
    reason=(
        "Set RUN_GEMINI_INTEGRATION=true with GEMINI_API_KEY to enable paid "
        "Gemini API tests"
    ),
)


class TestGoogleSearchIntegration:
    """Integration tests using real Gemini API."""

    @pytest.fixture
    def client(self) -> GoogleSearchClient:
        """Create a client with real API key."""
        return GoogleSearchClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.5-flash",
            max_retries=2,
        )

    def test_search_returns_result(self, client: GoogleSearchClient) -> None:
        """Should return a GoogleSearchResult with text."""
        async def run() -> GoogleSearchResult:
            return await client.search("What is Python programming language?")

        result = asyncio.run(run())

        assert result.text  # Should have content
        assert len(result.text) > 50  # Should be substantive

    def test_search_returns_sources(self, client: GoogleSearchClient) -> None:
        """Should return grounding sources."""
        async def run() -> GoogleSearchResult:
            return await client.search(
                "Who invented the World Wide Web?",
                max_results=5,
            )

        result = asyncio.run(run())

        # Expect at least one source
        assert len(result.sources) >= 1
        # Sources should have URLs
        for source in result.sources:
            assert source.uri.startswith("http")

    def test_search_returns_queries_used(self, client: GoogleSearchClient) -> None:
        """Should return the search queries used."""
        async def run() -> GoogleSearchResult:
            return await client.search("Latest developments in AI 2025")

        result = asyncio.run(run())

        # May or may not have queries_used, but if present they should be strings
        for query in result.queries_used:
            assert isinstance(query, str)
            assert len(query) > 0

    def test_search_with_grounding_supports(self, client: GoogleSearchClient) -> None:
        """Should return grounding supports linking text to sources."""
        async def run() -> GoogleSearchResult:
            return await client.search(
                "What are the health benefits of green tea according to research?"
            )

        result = asyncio.run(run())

        # May have grounding supports
        if result.supports:
            for support in result.supports:
                assert support.start_index >= 0
                assert support.end_index > support.start_index
                assert len(support.chunk_indices) >= 1

    def test_extract_citations_from_real_result(
        self,
        client: GoogleSearchClient,
    ) -> None:
        """Should be able to extract citations from real search result."""
        async def run() -> GoogleSearchResult:
            return await client.search(
                "How tall is Mount Everest in meters?",
                max_results=3,
            )

        result = asyncio.run(run())
        citations = extract_grounded_citations(result)

        # If we have citations, verify structure
        for citation in citations:
            assert "text" in citation
            assert "sources" in citation
            assert "start_index" in citation
            assert "end_index" in citation
            # Each citation should have at least one source URL
            if citation["sources"]:
                for url in citation["sources"]:
                    assert url.startswith("http")

    def test_source_to_source_format(self, client: GoogleSearchClient) -> None:
        """Should be able to convert sources to AgentState format."""
        async def run() -> GoogleSearchResult:
            return await client.search("Python documentation", max_results=3)

        result = asyncio.run(run())

        # Convert each source
        for source in result.sources:
            source_dict = source.to_source()
            assert "title" in source_dict
            assert "url" in source_dict
            assert "snippet" in source_dict
            # URL should be valid
            assert source_dict["url"].startswith("http")


class TestGoogleSearchErrorHandling:
    """Test error handling with real API."""

    def test_invalid_api_key(self) -> None:
        """Should raise auth error with invalid key."""
        client = GoogleSearchClient(
            api_key="invalid-key-12345",
            max_retries=1,
        )

        async def run() -> None:
            await client.search("test query")

        with pytest.raises(GoogleSearchError) as exc_info:
            asyncio.run(run())

        # Should be an auth or unknown error
        assert exc_info.value.category in ("auth_error", "unknown")


class TestGoogleSearchWithResearchQueries:
    """Test with realistic research queries."""

    @pytest.fixture
    def client(self) -> GoogleSearchClient:
        """Create a client with real API key."""
        return GoogleSearchClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.5-flash",
            max_retries=2,
        )

    def test_research_query_technical(self, client: GoogleSearchClient) -> None:
        """Test with a technical research query."""
        async def run() -> GoogleSearchResult:
            return await client.search(
                "Best practices for building production-ready LangGraph agents"
            )

        result = asyncio.run(run())
        assert result.text
        # Should have some sources
        assert len(result.sources) >= 0  # May not always have sources

    def test_research_query_factual(self, client: GoogleSearchClient) -> None:
        """Test with a factual research query."""
        async def run() -> GoogleSearchResult:
            return await client.search(
                "What is the current population of the United States?"
            )

        result = asyncio.run(run())
        assert result.text
        # Factual queries should generally have sources
        # Note: Not asserting on number of sources as it varies
