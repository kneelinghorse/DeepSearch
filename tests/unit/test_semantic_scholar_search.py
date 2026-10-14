"""Unit tests for SemanticScholarClient (B24.4).

Key requirement: results must have publisher/DOI URLs, not semanticscholar.org URLs.
Tests cover:
- Publisher URL extraction priority (OA PDF > DOI > PubMed)
- Papers without publisher URL are skipped
- semanticscholar.org URLs are never returned
- JSON parse error → SemanticScholarError(category="parse")
- Network failure → SemanticScholarError(category="network")
- Empty query → SemanticScholarError(category="validation")
- max_results respected
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from deepsearch.tools.semantic_scholar_search import (
    SemanticScholarClient,
    SemanticScholarError,
    SemanticScholarResult,
)


def _make_response(papers: list[dict]) -> str:
    return json.dumps({"data": papers, "total": len(papers)})


def _make_paper(
    title: str = "Test Paper",
    doi: str | None = None,
    oa_pdf_url: str | None = None,
    pmid: str | None = None,
    abstract: str = "Test abstract.",
) -> dict:
    external_ids: dict = {}
    if doi:
        external_ids["DOI"] = doi
    if pmid:
        external_ids["PubMed"] = pmid
    return {
        "paperId": "abc123",
        "title": title,
        "abstract": abstract,
        "year": 2024,
        "externalIds": external_ids,
        "openAccessPdf": {"url": oa_pdf_url} if oa_pdf_url else None,
    }


class TestExtractPublisherUrl:
    def test_oa_pdf_takes_priority(self) -> None:
        paper = _make_paper(oa_pdf_url="https://arxiv.org/pdf/2301.00001", doi="10.1234/test")
        url = SemanticScholarClient._extract_publisher_url(paper)
        assert url == "https://arxiv.org/pdf/2301.00001"

    def test_doi_used_when_no_oa_pdf(self) -> None:
        paper = _make_paper(doi="10.1234/test.paper")
        url = SemanticScholarClient._extract_publisher_url(paper)
        assert url == "https://doi.org/10.1234/test.paper"

    def test_pubmed_used_as_fallback(self) -> None:
        paper = _make_paper(pmid="12345678")
        url = SemanticScholarClient._extract_publisher_url(paper)
        assert url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"

    def test_none_when_no_publisher_url(self) -> None:
        paper = _make_paper()  # no doi, no oa pdf, no pmid
        url = SemanticScholarClient._extract_publisher_url(paper)
        assert url is None

    def test_semanticscholar_url_never_returned(self) -> None:
        paper = _make_paper()
        url = SemanticScholarClient._extract_publisher_url(paper)
        # Should never return semanticscholar.org
        if url:
            assert "semanticscholar.org" not in url


class TestParseResponse:
    def test_parses_doi_paper(self) -> None:
        raw = _make_response([_make_paper(title="Quantum Study", doi="10.1234/quantum")])
        client = SemanticScholarClient()
        results = client._parse_response(raw, 5)
        assert len(results) == 1
        assert results[0].url == "https://doi.org/10.1234/quantum"
        assert results[0].title == "Quantum Study"

    def test_skips_papers_without_publisher_url(self) -> None:
        raw = _make_response([_make_paper(title="No URL Paper")])  # no doi/oa/pmid
        client = SemanticScholarClient()
        results = client._parse_response(raw, 5)
        assert results == []

    def test_max_results_respected(self) -> None:
        papers = [_make_paper(doi=f"10.1234/paper{i}", title=f"Paper {i}") for i in range(10)]
        raw = _make_response(papers)
        client = SemanticScholarClient()
        results = client._parse_response(raw, 3)
        assert len(results) == 3

    def test_json_parse_error_raises(self) -> None:
        client = SemanticScholarClient()
        with pytest.raises(SemanticScholarError) as exc_info:
            client._parse_response("not json {{{", 5)
        assert exc_info.value.category == "parse"

    def test_empty_data_array_returns_empty(self) -> None:
        raw = json.dumps({"data": [], "total": 0})
        client = SemanticScholarClient()
        results = client._parse_response(raw, 5)
        assert results == []

    def test_snippet_truncated(self) -> None:
        long_abstract = "word " * 200
        raw = _make_response([_make_paper(doi="10.1234/x", abstract=long_abstract)])
        client = SemanticScholarClient()
        results = client._parse_response(raw, 5)
        assert len(results[0].snippet) == 500

    def test_result_has_no_semanticscholar_url(self) -> None:
        raw = _make_response([_make_paper(doi="10.1234/test", title="Test")])
        client = SemanticScholarClient()
        results = client._parse_response(raw, 5)
        for r in results:
            assert "semanticscholar.org" not in r.url


class TestSemanticScholarClientAsync:
    def test_empty_query_raises_validation_error(self) -> None:
        client = SemanticScholarClient()
        with pytest.raises(SemanticScholarError) as exc_info:
            asyncio.run(client.search(""))
        assert exc_info.value.category == "validation"

    def test_network_failure_raises(self) -> None:
        client = SemanticScholarClient()
        with patch.object(
            client, "_fetch", side_effect=SemanticScholarError("timeout", category="network")
        ):
            with pytest.raises(SemanticScholarError) as exc_info:
                asyncio.run(client.search("quantum computing"))
            assert exc_info.value.category == "network"

    def test_successful_search_returns_results(self) -> None:
        raw = _make_response([_make_paper(title="AI Research", doi="10.1234/ai")])
        client = SemanticScholarClient()
        with patch.object(client, "_fetch", return_value=raw):
            results = asyncio.run(client.search("artificial intelligence"))
        assert len(results) == 1
        assert results[0].url == "https://doi.org/10.1234/ai"

    def test_to_source_has_url(self) -> None:
        raw = _make_response([_make_paper(doi="10.1234/x", title="X")])
        client = SemanticScholarClient()
        with patch.object(client, "_fetch", return_value=raw):
            results = asyncio.run(client.search("test"))
        sources = [r.to_source() for r in results]
        assert all("url" in s for s in sources)
        assert all("semanticscholar.org" not in s["url"] for s in sources)
