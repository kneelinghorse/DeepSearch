"""Unit tests for OpenAlexClient (B24.4).

Key requirement: results must have publisher/DOI URLs, not openalex.org URLs.
Tests cover:
- Publisher URL extraction priority (DOI > landing_page_url > oa_url)
- openalex.org URLs are never returned
- Abstract reconstruction from inverted index
- Works without publisher URL are skipped
- JSON parse error → OpenAlexError(category="parse")
- Network failure → OpenAlexError(category="network")
- Empty query → OpenAlexError(category="validation")
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from deepsearch.tools.openalex_search import (
    OpenAlexClient,
    OpenAlexError,
    OpenAlexResult,
)


def _make_response(works: list[dict]) -> str:
    return json.dumps({"results": works, "meta": {"count": len(works)}})


def _make_work(
    title: str = "Test Work",
    doi: str | None = None,
    landing_page_url: str | None = None,
    oa_url: str | None = None,
    abstract_inverted_index: dict | None = None,
) -> dict:
    return {
        "id": "https://openalex.org/W123",
        "title": title,
        "doi": doi,
        "primary_location": {"landing_page_url": landing_page_url} if landing_page_url else {},
        "open_access": {"oa_url": oa_url} if oa_url else {},
        "abstract_inverted_index": abstract_inverted_index,
    }


class TestExtractPublisherUrl:
    def test_doi_takes_priority(self) -> None:
        work = _make_work(doi="https://doi.org/10.1234/test", landing_page_url="https://nature.com/paper")
        url = OpenAlexClient._extract_publisher_url(work)
        assert url == "https://doi.org/10.1234/test"

    def test_landing_page_used_when_no_doi(self) -> None:
        work = _make_work(landing_page_url="https://nature.com/paper123")
        url = OpenAlexClient._extract_publisher_url(work)
        assert url == "https://nature.com/paper123"

    def test_oa_url_as_fallback(self) -> None:
        work = _make_work(oa_url="https://arxiv.org/pdf/2301.00001")
        url = OpenAlexClient._extract_publisher_url(work)
        assert url == "https://arxiv.org/pdf/2301.00001"

    def test_none_when_no_publisher_url(self) -> None:
        work = _make_work()
        url = OpenAlexClient._extract_publisher_url(work)
        assert url is None

    def test_openalex_landing_page_rejected(self) -> None:
        work = _make_work(landing_page_url="https://openalex.org/W123456")
        url = OpenAlexClient._extract_publisher_url(work)
        assert url is None

    def test_openalex_oa_url_rejected(self) -> None:
        work = _make_work(oa_url="https://openalex.org/W123456")
        url = OpenAlexClient._extract_publisher_url(work)
        assert url is None

    def test_openalex_url_never_returned(self) -> None:
        work = _make_work(doi="https://doi.org/10.1234/x")
        url = OpenAlexClient._extract_publisher_url(work)
        if url:
            assert "openalex.org" not in url


class TestExtractSnippet:
    def test_reconstructs_from_inverted_index(self) -> None:
        # "Hello World" → {"Hello": [0], "World": [1]}
        inverted = {"Hello": [0], "World": [1]}
        snippet = OpenAlexClient._extract_snippet({"abstract_inverted_index": inverted})
        assert snippet == "Hello World"

    def test_none_inverted_index_returns_empty(self) -> None:
        snippet = OpenAlexClient._extract_snippet({"abstract_inverted_index": None})
        assert snippet == ""

    def test_missing_inverted_index_returns_empty(self) -> None:
        snippet = OpenAlexClient._extract_snippet({})
        assert snippet == ""

    def test_snippet_truncated_at_500(self) -> None:
        long_text = {f"word{i}": [i] for i in range(300)}
        snippet = OpenAlexClient._extract_snippet({"abstract_inverted_index": long_text})
        assert len(snippet) == 500


class TestParseResponse:
    def test_parses_doi_work(self) -> None:
        raw = _make_response([_make_work(title="Climate Study", doi="https://doi.org/10.1234/climate")])
        client = OpenAlexClient()
        results = client._parse_response(raw, 5)
        assert len(results) == 1
        assert results[0].url == "https://doi.org/10.1234/climate"
        assert results[0].title == "Climate Study"

    def test_skips_works_without_publisher_url(self) -> None:
        raw = _make_response([_make_work(title="No URL Work")])
        client = OpenAlexClient()
        results = client._parse_response(raw, 5)
        assert results == []

    def test_max_results_respected(self) -> None:
        works = [_make_work(doi=f"https://doi.org/10.1234/{i}", title=f"W{i}") for i in range(10)]
        raw = _make_response(works)
        client = OpenAlexClient()
        results = client._parse_response(raw, 3)
        assert len(results) == 3

    def test_json_parse_error_raises(self) -> None:
        client = OpenAlexClient()
        with pytest.raises(OpenAlexError) as exc_info:
            client._parse_response("{broken", 5)
        assert exc_info.value.category == "parse"

    def test_no_openalex_urls_in_results(self) -> None:
        raw = _make_response([_make_work(doi="https://doi.org/10.1234/x", title="T")])
        client = OpenAlexClient()
        results = client._parse_response(raw, 5)
        for r in results:
            assert "openalex.org" not in r.url


class TestOpenAlexClientAsync:
    def test_empty_query_raises_validation_error(self) -> None:
        client = OpenAlexClient()
        with pytest.raises(OpenAlexError) as exc_info:
            asyncio.run(client.search(""))
        assert exc_info.value.category == "validation"

    def test_network_failure_raises(self) -> None:
        client = OpenAlexClient()
        with patch.object(
            client, "_fetch", side_effect=OpenAlexError("timeout", category="network")
        ):
            with pytest.raises(OpenAlexError) as exc_info:
                asyncio.run(client.search("climate change"))
            assert exc_info.value.category == "network"

    def test_successful_search_returns_results(self) -> None:
        raw = _make_response([_make_work(title="AI Work", doi="https://doi.org/10.1234/ai")])
        client = OpenAlexClient()
        with patch.object(client, "_fetch", return_value=raw):
            results = asyncio.run(client.search("artificial intelligence"))
        assert len(results) == 1
        assert results[0].url == "https://doi.org/10.1234/ai"

    def test_to_source_has_no_openalex_url(self) -> None:
        raw = _make_response([_make_work(doi="https://doi.org/10.1234/x", title="X")])
        client = OpenAlexClient()
        with patch.object(client, "_fetch", return_value=raw):
            results = asyncio.run(client.search("test"))
        sources = [r.to_source() for r in results]
        assert all("openalex.org" not in s.get("url", "") for s in sources)
