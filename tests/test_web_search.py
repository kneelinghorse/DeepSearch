"""Tests for the Tavily web search integration layer."""

from __future__ import annotations

import types

import pytest

import deepsearch.tools.web_search as web_search
from deepsearch.tools.web_search import (
    TavilyResult,
    TavilySearchClient,
    TavilySearchTool,
    WebSearchError,
)


@pytest.mark.asyncio
async def test_tavily_search_client_returns_normalized_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client should call the Tavily API and return parsed results."""

    payload = {
        "results": [
            {"title": "LangGraph", "url": "https://example.com", "content": "Stateful agents"}
        ]
    }

    class DummyResponse:
        def raise_for_status(self) -> None:  # pragma: no cover - trivial guard
            return None

        def json(self) -> dict[str, object]:
            return payload

    class DummyHttpxClient:
        request_payload: dict[str, object] | None = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "DummyHttpxClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> DummyResponse:
            DummyHttpxClient.request_payload = json
            return DummyResponse()

    dummy_httpx = types.SimpleNamespace(
        AsyncClient=DummyHttpxClient,
        HTTPError=Exception,
        HTTPStatusError=Exception,
        TimeoutException=Exception,
        RequestError=Exception,
    )
    monkeypatch.setattr(web_search, "httpx", dummy_httpx)
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    client = TavilySearchClient()
    results = await client.search("LangGraph", max_results=1)

    assert len(results) == 1
    assert results[0].title == "LangGraph"
    assert DummyHttpxClient.request_payload is not None
    assert DummyHttpxClient.request_payload["query"] == "LangGraph"


@pytest.mark.asyncio
async def test_tavily_search_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing API keys should raise a WebSearchError."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    client = TavilySearchClient(api_key=None)

    with pytest.raises(WebSearchError):
        await client.search("missing credentials")


@pytest.mark.asyncio
async def test_tavily_search_tool_formats_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool wrapper should expose formatted source dictionaries."""

    class DummyClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def search(self, query: str, max_results: int) -> list[TavilyResult]:
            assert query == "LangGraph"
            assert max_results == 5
            return [TavilyResult(title="Doc", url="https://example.com", content="Snippet")]

    monkeypatch.setattr(web_search, "TavilySearchClient", DummyClient)

    tool = TavilySearchTool(api_key="key")
    sources = await tool._arun(query="LangGraph")

    assert sources == [
        {"title": "Doc", "url": "https://example.com", "snippet": "Snippet"}
    ]
