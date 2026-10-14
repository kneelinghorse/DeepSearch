"""Unit tests for ArxivSearchClient (B24.3).

Tests cover:
- Happy path: parse valid Atom XML response
- Version suffix stripping from arXiv IDs
- http → https URL normalization
- Empty results on empty feed
- Parse error handling → ArxivSearchError(category="parse")
- Network failure handling → ArxivSearchError(category="network")
- Empty query raises ArxivSearchError(category="validation")
- Snippet truncation to 500 chars
"""

import asyncio
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from deepsearch.tools.arxiv_search import ArxivResult, ArxivSearchClient, ArxivSearchError

_ATOM_NS = "http://www.w3.org/2005/Atom"


def _make_atom_feed(entries: list[dict]) -> str:
    """Build minimal Atom XML feed from a list of entry dicts."""
    root = ET.Element(f"{{{_ATOM_NS}}}feed")
    for e in entries:
        entry = ET.SubElement(root, f"{{{_ATOM_NS}}}entry")
        id_el = ET.SubElement(entry, f"{{{_ATOM_NS}}}id")
        id_el.text = e.get("id", "")
        title_el = ET.SubElement(entry, f"{{{_ATOM_NS}}}title")
        title_el.text = e.get("title", "")
        summary_el = ET.SubElement(entry, f"{{{_ATOM_NS}}}summary")
        summary_el.text = e.get("summary", "")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


class TestArxivResultToSource:
    def test_to_source_basic(self) -> None:
        r = ArxivResult(url="https://arxiv.org/abs/2301.00001", title="Test", snippet="Abstract.")
        s = r.to_source()
        assert s["url"] == "https://arxiv.org/abs/2301.00001"
        assert s["title"] == "Test"
        assert s["snippet"] == "Abstract."

    def test_to_source_empty_url_omitted(self) -> None:
        r = ArxivResult(url="", title="No URL", snippet="")
        s = r.to_source()
        assert "url" not in s

    def test_to_source_empty_snippet_omitted(self) -> None:
        r = ArxivResult(url="https://arxiv.org/abs/1234", title="T", snippet="")
        s = r.to_source()
        assert "snippet" not in s


class TestArxivParseResponse:
    def test_parses_single_entry(self) -> None:
        xml = _make_atom_feed([
            {
                "id": "http://arxiv.org/abs/2301.00001v2",
                "title": "Deep Learning Survey",
                "summary": "A survey of deep learning methods.",
            }
        ])
        client = ArxivSearchClient()
        results = client._parse_response(xml, 5)
        assert len(results) == 1
        assert results[0].url == "https://arxiv.org/abs/2301.00001"
        assert results[0].title == "Deep Learning Survey"
        assert "survey" in results[0].snippet.lower()

    def test_strips_version_suffix(self) -> None:
        xml = _make_atom_feed([{"id": "http://arxiv.org/abs/2211.99999v3", "title": "T", "summary": "S"}])
        client = ArxivSearchClient()
        results = client._parse_response(xml, 5)
        assert results[0].url == "https://arxiv.org/abs/2211.99999"

    def test_no_version_suffix_preserved(self) -> None:
        xml = _make_atom_feed([{"id": "http://arxiv.org/abs/2301.00001", "title": "T", "summary": "S"}])
        client = ArxivSearchClient()
        results = client._parse_response(xml, 5)
        assert results[0].url == "https://arxiv.org/abs/2301.00001"

    def test_http_to_https_normalization(self) -> None:
        xml = _make_atom_feed([{"id": "http://arxiv.org/abs/2301.00001v1", "title": "T", "summary": "S"}])
        client = ArxivSearchClient()
        results = client._parse_response(xml, 5)
        assert results[0].url.startswith("https://")

    def test_empty_feed_returns_empty(self) -> None:
        root = ET.Element(f"{{{_ATOM_NS}}}feed")
        xml = ET.tostring(root, encoding="unicode")
        client = ArxivSearchClient()
        results = client._parse_response(xml, 5)
        assert results == []

    def test_max_results_respected(self) -> None:
        entries = [{"id": f"http://arxiv.org/abs/230{i}.00001v1", "title": f"T{i}", "summary": "S"} for i in range(5)]
        xml = _make_atom_feed(entries)
        client = ArxivSearchClient()
        results = client._parse_response(xml, 3)
        assert len(results) == 3

    def test_snippet_truncated_at_500(self) -> None:
        long_summary = "x" * 600
        xml = _make_atom_feed([{"id": "http://arxiv.org/abs/2301.00001v1", "title": "T", "summary": long_summary}])
        client = ArxivSearchClient()
        results = client._parse_response(xml, 5)
        assert len(results[0].snippet) == 500

    def test_parse_error_raises(self) -> None:
        client = ArxivSearchClient()
        with pytest.raises(ArxivSearchError) as exc_info:
            client._parse_response("not xml at all <<<", 5)
        assert exc_info.value.category == "parse"

    def test_whitespace_normalized_in_title(self) -> None:
        xml = _make_atom_feed([{"id": "http://arxiv.org/abs/2301.00001v1", "title": "  A   B  C  ", "summary": "S"}])
        client = ArxivSearchClient()
        results = client._parse_response(xml, 5)
        assert results[0].title == "A B C"

    def test_entry_without_url_skipped(self) -> None:
        xml = _make_atom_feed([{"id": "", "title": "No URL", "summary": "S"}])
        client = ArxivSearchClient()
        results = client._parse_response(xml, 5)
        assert results == []


class TestArxivSearchClientAsync:
    def test_empty_query_raises_validation_error(self) -> None:
        client = ArxivSearchClient()
        with pytest.raises(ArxivSearchError) as exc_info:
            asyncio.run(client.search(""))
        assert exc_info.value.category == "validation"

    def test_whitespace_query_raises_validation_error(self) -> None:
        client = ArxivSearchClient()
        with pytest.raises(ArxivSearchError) as exc_info:
            asyncio.run(client.search("   "))
        assert exc_info.value.category == "validation"

    def test_network_failure_raises_arxiv_error(self) -> None:
        client = ArxivSearchClient()
        with patch.object(client, "_fetch", side_effect=ArxivSearchError("timeout", category="network")):
            with pytest.raises(ArxivSearchError) as exc_info:
                asyncio.run(client.search("quantum computing"))
            assert exc_info.value.category == "network"

    def test_successful_search_returns_results(self) -> None:
        xml = _make_atom_feed([
            {"id": "http://arxiv.org/abs/2301.00001v1", "title": "Quantum Computing", "summary": "Abstract."},
        ])
        client = ArxivSearchClient()
        with patch.object(client, "_fetch", return_value=xml):
            results = asyncio.run(client.search("quantum computing", max_results=5))
        assert len(results) == 1
        assert results[0].url == "https://arxiv.org/abs/2301.00001"

    def test_to_source_dicts_have_url(self) -> None:
        xml = _make_atom_feed([
            {"id": "http://arxiv.org/abs/2301.00001v1", "title": "T", "summary": "S"},
        ])
        client = ArxivSearchClient()
        with patch.object(client, "_fetch", return_value=xml):
            results = asyncio.run(client.search("test"))
        sources = [r.to_source() for r in results]
        assert all("url" in s for s in sources)
        assert all(s["url"].startswith("https://arxiv.org") for s in sources)
