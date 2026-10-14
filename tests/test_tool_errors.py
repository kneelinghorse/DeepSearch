"""Tool-layer regression tests for error handling and circuit breaker behaviour."""

from __future__ import annotations

import types
from typing import Any, List

import pytest

from deepsearch.agent.resilience import CircuitBreaker
from deepsearch.tools import web_search
from deepsearch.tools.web_search import TavilySearchClient, WebSearchError


class DummyHTTPError(Exception):
    """Base HTTP error raised by the stub httpx client."""


class DummyHTTPStatusError(DummyHTTPError):
    """Status error carrying the failing response."""

    def __init__(self, response: "DummyResponse") -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


class DummyTimeoutError(DummyHTTPError):
    """Timeout placeholder."""


class DummyRequestError(DummyHTTPError):
    """Network error placeholder."""


class DummyResponse:
    """Minimal HTTP response stub."""

    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise DummyHTTPStatusError(self)

    def json(self) -> dict[str, Any]:
        return self._payload


class DummyAsyncClient:
    """AsyncClient replacement that pops responses from a queue."""

    queue: List[Any] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def post(self, url: str, json: dict[str, Any]) -> DummyResponse:
        if not DummyAsyncClient.queue:
            raise AssertionError("No stub response queued")
        item = DummyAsyncClient.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def stub_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace httpx with an in-memory stub and speed up sleeps."""

    DummyAsyncClient.queue = []
    dummy_httpx = types.SimpleNamespace(
        AsyncClient=DummyAsyncClient,
        HTTPError=DummyHTTPError,
        HTTPStatusError=DummyHTTPStatusError,
        TimeoutException=DummyTimeoutError,
        RequestError=DummyRequestError,
    )
    monkeypatch.setattr(web_search, "httpx", dummy_httpx)

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("deepsearch.agent.resilience.asyncio.sleep", fast_sleep)


@pytest.mark.asyncio
async def test_tavily_client_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client should retry transient errors with exponential backoff."""

    monkeypatch.setenv("TAVILY_API_KEY", "testing-key")
    DummyAsyncClient.queue = [
        DummyHTTPStatusError(DummyResponse({}, status_code=500)),
        DummyResponse({"results": [{"title": "Recovered", "url": "https://example.com"}]}),
    ]

    client = TavilySearchClient(
        api_key=None,
        base_url="https://api.example.com",
        max_retries=1,
        circuit_breaker=None,
    )

    results = await client.search("LangGraph")

    assert len(results) == 1
    assert results[0].title == "Recovered"


@pytest.mark.asyncio
async def test_tavily_client_surfaces_circuit_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Circuit breaker should short-circuit repeated failures."""

    monkeypatch.setenv("TAVILY_API_KEY", "testing-key")
    DummyAsyncClient.queue = [
        DummyHTTPStatusError(DummyResponse({}, status_code=503)),
    ]

    circuit = CircuitBreaker(
        failure_threshold=1,
        success_threshold=1,
        timeout_seconds=60,
        name="test",
    )
    client = TavilySearchClient(
        api_key=None,
        base_url="https://api.example.com",
        max_retries=0,
        circuit_breaker=circuit,
    )

    with pytest.raises(WebSearchError) as first_error:
        await client.search("LangGraph")
    assert first_error.value.category in {"service_error", "http_error"}

    with pytest.raises(WebSearchError) as second_error:
        await client.search("LangGraph")
    assert second_error.value.category == "circuit_open"
