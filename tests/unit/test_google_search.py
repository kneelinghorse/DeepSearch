"""Unit tests for GoogleSearchTool and GoogleSearchClient."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepsearch.tools.google_search import (
    GoogleSearchClient,
    GoogleSearchError,
    GoogleSearchInput,
    GoogleSearchResult,
    GoogleSearchTool,
    GroundingChunk,
    GroundingSupport,
    extract_grounded_citations,
)


class TestGroundingChunk:
    """Tests for GroundingChunk dataclass."""

    def test_to_source_with_title_and_uri(self) -> None:
        chunk = GroundingChunk(uri="https://example.com", title="Example")
        source = chunk.to_source()
        assert source == {
            "title": "Example",
            "url": "https://example.com",
            "snippet": "",
        }

    def test_to_source_empty_title(self) -> None:
        chunk = GroundingChunk(uri="https://example.com", title="")
        source = chunk.to_source()
        assert source["title"] == "Google Search Result"

    def test_to_source_no_snippet(self) -> None:
        chunk = GroundingChunk(uri="https://example.com", title="Test")
        assert chunk.to_source()["snippet"] == ""


class TestGroundingSupport:
    """Tests for GroundingSupport dataclass."""

    def test_default_confidence_scores(self) -> None:
        support = GroundingSupport(
            start_index=0,
            end_index=10,
            chunk_indices=[0, 1],
        )
        assert support.confidence_scores == []

    def test_with_confidence_scores(self) -> None:
        support = GroundingSupport(
            start_index=0,
            end_index=10,
            chunk_indices=[0],
            confidence_scores=[0.95],
        )
        assert support.confidence_scores == [0.95]


class TestGoogleSearchResult:
    """Tests for GoogleSearchResult dataclass."""

    def test_empty_result(self) -> None:
        result = GoogleSearchResult(
            text="",
            sources=[],
            supports=[],
            queries_used=[],
        )
        assert result.text == ""
        assert result.sources == []

    def test_with_sources(self) -> None:
        result = GoogleSearchResult(
            text="Some response text",
            sources=[
                GroundingChunk(uri="https://example.com", title="Example"),
            ],
            supports=[],
            queries_used=["example query"],
        )
        assert len(result.sources) == 1
        assert result.queries_used == ["example query"]


class TestGoogleSearchInput:
    """Tests for GoogleSearchInput schema."""

    def test_valid_query(self) -> None:
        input_data = GoogleSearchInput(query="test query")
        assert input_data.query == "test query"

    def test_empty_query_accepts(self) -> None:
        # Pydantic accepts empty strings; validation happens in client
        input_data = GoogleSearchInput(query="")
        assert input_data.query == ""


class TestGoogleSearchClient:
    """Tests for GoogleSearchClient."""

    def test_missing_api_key_raises_error(self) -> None:
        """Should raise error when no API key is configured."""
        client = GoogleSearchClient(api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(GoogleSearchError) as exc_info:
                client._get_client()
            assert exc_info.value.category == "configuration"

    def test_empty_query_raises_error(self) -> None:
        """Should raise error for empty query."""
        client = GoogleSearchClient(api_key="test-key")

        async def run_test() -> None:
            with pytest.raises(GoogleSearchError) as exc_info:
                await client.search("")
            assert exc_info.value.category == "validation"

        asyncio.run(run_test())

    def test_whitespace_query_raises_error(self) -> None:
        """Should raise error for whitespace-only query."""
        client = GoogleSearchClient(api_key="test-key")

        async def run_test() -> None:
            with pytest.raises(GoogleSearchError) as exc_info:
                await client.search("   ")
            assert exc_info.value.category == "validation"

        asyncio.run(run_test())

    def test_parse_response_extracts_text(self) -> None:
        """Should extract text from response."""
        client = GoogleSearchClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.text = "Test response text"
        mock_response.candidates = []

        result = client._parse_response(mock_response, max_results=5)
        assert result.text == "Test response text"

    def test_parse_response_extracts_grounding_chunks(self) -> None:
        """Should extract grounding chunks from response."""
        client = GoogleSearchClient(api_key="test-key")

        # Create mock web chunk
        mock_web = MagicMock()
        mock_web.uri = "https://example.com"
        mock_web.title = "Example Site"

        mock_chunk = MagicMock()
        mock_chunk.web = mock_web

        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.web_search_queries = ["test query"]
        mock_grounding_metadata.grounding_chunks = [mock_chunk]
        mock_grounding_metadata.grounding_supports = []

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding_metadata
        mock_candidate.content = None

        mock_response = MagicMock()
        mock_response.text = "Response with grounding"
        mock_response.candidates = [mock_candidate]

        result = client._parse_response(mock_response, max_results=5)

        assert len(result.sources) == 1
        assert result.sources[0].uri == "https://example.com"
        assert result.sources[0].title == "Example Site"
        assert result.queries_used == ["test query"]

    def test_parse_response_extracts_grounding_supports(self) -> None:
        """Should extract grounding supports from response."""
        client = GoogleSearchClient(api_key="test-key")

        # Create mock segment
        mock_segment = MagicMock()
        mock_segment.start_index = 0
        mock_segment.end_index = 10

        mock_support = MagicMock()
        mock_support.segment = mock_segment
        mock_support.grounding_chunk_indices = [0]
        mock_support.confidence_scores = [0.9]

        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.web_search_queries = []
        mock_grounding_metadata.grounding_chunks = []
        mock_grounding_metadata.grounding_supports = [mock_support]

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding_metadata
        mock_candidate.content = None

        mock_response = MagicMock()
        mock_response.text = "0123456789"
        mock_response.candidates = [mock_candidate]

        result = client._parse_response(mock_response, max_results=5)

        assert len(result.supports) == 1
        assert result.supports[0].start_index == 0
        assert result.supports[0].end_index == 10
        assert result.supports[0].chunk_indices == [0]

    def test_translate_exception_rate_limit(self) -> None:
        """Should detect rate limit errors."""
        client = GoogleSearchClient(api_key="test-key")
        exc = Exception("Error 429: Quota exceeded")
        result = client._translate_exception(exc)
        assert result.category == "rate_limit"
        assert result.status_code == 429

    def test_translate_exception_auth_error(self) -> None:
        """Should detect auth errors."""
        client = GoogleSearchClient(api_key="test-key")
        exc = Exception("Error 401: Invalid API key")
        result = client._translate_exception(exc)
        assert result.category == "auth_error"

    def test_translate_exception_timeout(self) -> None:
        """Should detect timeout errors."""
        client = GoogleSearchClient(api_key="test-key")
        exc = Exception("Request timeout exceeded")
        result = client._translate_exception(exc)
        assert result.category == "timeout"

    def test_translate_exception_unknown(self) -> None:
        """Should handle unknown errors."""
        client = GoogleSearchClient(api_key="test-key")
        exc = ValueError("Something went wrong")
        result = client._translate_exception(exc)
        assert result.category == "unknown"


class TestGoogleSearchTool:
    """Tests for GoogleSearchTool LangChain wrapper."""

    def test_tool_attributes(self) -> None:
        """Should have correct tool metadata."""
        tool = GoogleSearchTool(api_key="test-key")
        assert tool.name == "web_search"
        assert "Google Search" in tool.description

    def test_args_schema(self) -> None:
        """Should expose correct args schema."""
        tool = GoogleSearchTool(api_key="test-key")
        assert tool.args_schema == GoogleSearchInput

    def test_default_model(self) -> None:
        """Should have default model configured."""
        tool = GoogleSearchTool(api_key="test-key")
        assert tool.model == "gemini-2.5-flash"

    @patch("deepsearch.tools.google_search.GoogleSearchClient.search")
    def test_arun_returns_sources(self, mock_search: AsyncMock) -> None:
        """Should return list of source dicts."""
        mock_search.return_value = GoogleSearchResult(
            text="Test response",
            sources=[
                GroundingChunk(uri="https://example.com", title="Example"),
                GroundingChunk(uri="https://test.com", title="Test"),
            ],
            supports=[],
            queries_used=["test query"],
        )

        tool = GoogleSearchTool(api_key="test-key")

        async def run_test() -> List[dict]:
            return await tool._arun("test query")

        result = asyncio.run(run_test())

        assert len(result) == 2
        assert result[0]["url"] == "https://example.com"
        assert result[0]["title"] == "Example"


class TestExtractGroundedCitations:
    """Tests for extract_grounded_citations helper."""

    def test_extracts_citations(self) -> None:
        """Should extract inline citations from result."""
        result = GoogleSearchResult(
            text="The sky is blue according to NASA.",
            sources=[
                GroundingChunk(uri="https://nasa.gov", title="NASA"),
            ],
            supports=[
                GroundingSupport(
                    start_index=0,
                    end_index=15,
                    chunk_indices=[0],
                    confidence_scores=[0.95],
                ),
            ],
            queries_used=["sky color"],
        )

        citations = extract_grounded_citations(result)

        assert len(citations) == 1
        assert citations[0]["text"] == "The sky is blue"
        assert citations[0]["sources"] == ["https://nasa.gov"]
        assert citations[0]["confidence"] == 0.95

    def test_skips_invalid_indices(self) -> None:
        """Should skip supports with invalid span indices."""
        result = GoogleSearchResult(
            text="Test",
            sources=[GroundingChunk(uri="https://example.com", title="Example")],
            supports=[
                GroundingSupport(
                    start_index=10,  # Invalid: start >= end
                    end_index=5,
                    chunk_indices=[0],
                ),
            ],
            queries_used=[],
        )

        citations = extract_grounded_citations(result)
        assert len(citations) == 0

    def test_skips_out_of_range_chunk_indices(self) -> None:
        """Should skip invalid chunk index references."""
        result = GoogleSearchResult(
            text="Test text",
            sources=[],  # No sources
            supports=[
                GroundingSupport(
                    start_index=0,
                    end_index=4,
                    chunk_indices=[999],  # Invalid index
                ),
            ],
            queries_used=[],
        )

        citations = extract_grounded_citations(result)
        assert len(citations) == 0

    def test_multiple_sources_per_citation(self) -> None:
        """Should handle multiple sources for one citation."""
        result = GoogleSearchResult(
            text="Fact supported by multiple sources.",
            sources=[
                GroundingChunk(uri="https://source1.com", title="Source 1"),
                GroundingChunk(uri="https://source2.com", title="Source 2"),
            ],
            supports=[
                GroundingSupport(
                    start_index=0,
                    end_index=35,
                    chunk_indices=[0, 1],
                ),
            ],
            queries_used=[],
        )

        citations = extract_grounded_citations(result)

        assert len(citations) == 1
        assert len(citations[0]["sources"]) == 2
        assert "https://source1.com" in citations[0]["sources"]
        assert "https://source2.com" in citations[0]["sources"]

    def test_no_confidence_score(self) -> None:
        """Should handle missing confidence scores."""
        result = GoogleSearchResult(
            text="Test",
            sources=[GroundingChunk(uri="https://example.com", title="Example")],
            supports=[
                GroundingSupport(
                    start_index=0,
                    end_index=4,
                    chunk_indices=[0],
                    confidence_scores=[],  # Empty
                ),
            ],
            queries_used=[],
        )

        citations = extract_grounded_citations(result)
        assert citations[0]["confidence"] is None


class TestGoogleSearchError:
    """Tests for GoogleSearchError exception."""

    def test_error_with_category(self) -> None:
        """Should store error category."""
        error = GoogleSearchError("Test error", category="rate_limit")
        assert error.category == "rate_limit"
        assert str(error) == "Test error"

    def test_error_with_status_code(self) -> None:
        """Should store status code."""
        error = GoogleSearchError(
            "HTTP error",
            category="http_error",
            status_code=500,
        )
        assert error.status_code == 500

    def test_default_category(self) -> None:
        """Should default to unknown category."""
        error = GoogleSearchError("Unknown error")
        assert error.category == "unknown"
        assert error.status_code is None
