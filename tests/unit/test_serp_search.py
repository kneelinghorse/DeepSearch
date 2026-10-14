"""Unit tests for SerpSearchClient, SerpSearchTool, and related models.

B14.1: Comprehensive coverage for deepsearch/tools/serp_search.py including:
- SerpResult model and to_source()
- SerpSearchInput schema validation
- SerpSearchError exception hierarchy
- SerpSearchClient initialization, API key resolution, search, parsing, retry, circuit breaker
- SerpSearchTool LangChain wrapper
- Response parsing for Google, Bing, DuckDuckGo engine types
- Error translation (rate limit, auth, timeout, unknown)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepsearch.agent.resilience import CircuitBreaker, CircuitBreakerOpenError, CircuitState
from deepsearch.tools.serp_search import (
    SerpResult,
    SerpSearchClient,
    SerpSearchError,
    SerpSearchInput,
    SerpSearchTool,
)


# ---------------------------------------------------------------------------
# SerpResult
# ---------------------------------------------------------------------------
class TestSerpResult:
    """Tests for SerpResult model."""

    def test_to_source_with_url_and_title(self) -> None:
        result = SerpResult(url="https://example.com", title="Example", snippet="A snippet")
        source = result.to_source()
        assert source == {
            "title": "Example",
            "url": "https://example.com",
            "snippet": "A snippet",
        }

    def test_to_source_empty_title_uses_default(self) -> None:
        result = SerpResult(url="https://example.com", title="", snippet="")
        source = result.to_source()
        assert source["title"] == "SerpAPI Result"

    def test_to_source_no_url_omits_url(self) -> None:
        result = SerpResult(url="", title="Test", snippet="snip")
        source = result.to_source()
        assert "url" not in source

    def test_to_source_no_snippet_omits_snippet(self) -> None:
        result = SerpResult(url="https://example.com", title="Test", snippet="")
        source = result.to_source()
        assert "snippet" not in source

    def test_default_fields(self) -> None:
        result = SerpResult()
        assert result.url == ""
        assert result.title == ""
        assert result.snippet == ""
        assert result.position is None
        assert result.source is None

    def test_position_and_source(self) -> None:
        result = SerpResult(
            url="https://example.com",
            title="Test",
            snippet="",
            position=3,
            source="example.com",
        )
        assert result.position == 3
        assert result.source == "example.com"


# ---------------------------------------------------------------------------
# SerpSearchInput
# ---------------------------------------------------------------------------
class TestSerpSearchInput:
    """Tests for SerpSearchInput schema."""

    def test_valid_query(self) -> None:
        inp = SerpSearchInput(query="python tutorials")
        assert inp.query == "python tutorials"

    def test_empty_query_accepted_by_schema(self) -> None:
        # Pydantic accepts empty; validation is in the client
        inp = SerpSearchInput(query="")
        assert inp.query == ""


# ---------------------------------------------------------------------------
# SerpSearchError
# ---------------------------------------------------------------------------
class TestSerpSearchError:
    """Tests for SerpSearchError exception."""

    def test_default_category(self) -> None:
        err = SerpSearchError("Something broke")
        assert err.category == "unknown"
        assert err.status_code is None
        assert str(err) == "Something broke"

    def test_custom_category_and_status_code(self) -> None:
        err = SerpSearchError(
            "Rate limited",
            category="rate_limit",
            status_code=429,
        )
        assert err.category == "rate_limit"
        assert err.status_code == 429

    def test_is_runtime_error(self) -> None:
        err = SerpSearchError("fail")
        assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# SerpSearchClient — API key resolution
# ---------------------------------------------------------------------------
class TestSerpSearchClientApiKey:
    """Tests for SerpSearchClient._get_api_key."""

    def test_explicit_api_key(self) -> None:
        client = SerpSearchClient(api_key="explicit-key")
        assert client._get_api_key() == "explicit-key"

    def test_env_api_key(self) -> None:
        client = SerpSearchClient(api_key=None)
        with patch.dict("os.environ", {"SERPAPI_API_KEY": "env-key"}):
            assert client._get_api_key() == "env-key"

    def test_missing_api_key_raises_error(self) -> None:
        client = SerpSearchClient(api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SerpSearchError) as exc_info:
                client._get_api_key()
            assert exc_info.value.category == "configuration"

    def test_explicit_key_overrides_env(self) -> None:
        client = SerpSearchClient(api_key="explicit")
        with patch.dict("os.environ", {"SERPAPI_API_KEY": "env"}):
            assert client._get_api_key() == "explicit"


# ---------------------------------------------------------------------------
# SerpSearchClient — search validation
# ---------------------------------------------------------------------------
class TestSerpSearchClientValidation:
    """Tests for SerpSearchClient input validation."""

    def test_empty_query_raises_error(self) -> None:
        client = SerpSearchClient(api_key="test-key")

        async def run() -> None:
            with pytest.raises(SerpSearchError) as exc_info:
                await client.search("")
            assert exc_info.value.category == "validation"

        asyncio.run(run())

    def test_whitespace_query_raises_error(self) -> None:
        client = SerpSearchClient(api_key="test-key")

        async def run() -> None:
            with pytest.raises(SerpSearchError) as exc_info:
                await client.search("   \t\n  ")
            assert exc_info.value.category == "validation"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# SerpSearchClient — _parse_response
# ---------------------------------------------------------------------------
class TestSerpSearchClientParseResponse:
    """Tests for SerpSearchClient._parse_response with various response shapes."""

    def _client(self, engine: str = "google") -> SerpSearchClient:
        return SerpSearchClient(api_key="test-key", engine=engine)

    # -- Google engine response -----------------------------------------------

    def test_google_dict_response(self) -> None:
        """Parse a typical Google organic_results dict response."""
        response: Dict[str, Any] = {
            "organic_results": [
                {
                    "link": "https://google.com/result1",
                    "title": "Google Result 1",
                    "snippet": "First result snippet",
                    "position": 1,
                    "source": "google.com",
                },
                {
                    "link": "https://google.com/result2",
                    "title": "Google Result 2",
                    "snippet": "Second result snippet",
                    "position": 2,
                },
            ],
        }
        client = self._client("google")
        results = client._parse_response(response, max_results=5, query="test")

        assert len(results) == 2
        assert results[0].url == "https://google.com/result1"
        assert results[0].title == "Google Result 1"
        assert results[0].snippet == "First result snippet"
        assert results[0].position == 1
        assert results[0].source == "google.com"

    # -- Bing engine response -------------------------------------------------

    def test_bing_dict_response(self) -> None:
        """Parse a Bing-style response with 'link' keys."""
        response: Dict[str, Any] = {
            "organic_results": [
                {
                    "link": "https://bing.com/r1",
                    "title": "Bing Result",
                    "snippet": "Bing snippet",
                    "position": 1,
                },
            ],
        }
        client = self._client("bing")
        results = client._parse_response(response, max_results=5, query="bing query")

        assert len(results) == 1
        assert results[0].url == "https://bing.com/r1"
        assert results[0].title == "Bing Result"

    # -- DuckDuckGo engine response -------------------------------------------

    def test_duckduckgo_dict_response(self) -> None:
        """Parse a DuckDuckGo-style response."""
        response: Dict[str, Any] = {
            "organic_results": [
                {
                    "link": "https://ddg.com/r1",
                    "title": "DDG Result",
                    "snippet": "DDG snippet",
                    "position": 1,
                },
            ],
        }
        client = self._client("duckduckgo")
        results = client._parse_response(response, max_results=5, query="ddg query")

        assert len(results) == 1
        assert results[0].url == "https://ddg.com/r1"

    # -- max_results limit ----------------------------------------------------

    def test_max_results_limits_output(self) -> None:
        response: Dict[str, Any] = {
            "organic_results": [
                {"link": f"https://example.com/{i}", "title": f"R{i}", "snippet": ""}
                for i in range(10)
            ],
        }
        client = self._client()
        results = client._parse_response(response, max_results=3, query="test")
        assert len(results) == 3

    # -- empty organic_results ------------------------------------------------

    def test_empty_organic_results(self) -> None:
        response: Dict[str, Any] = {"organic_results": []}
        client = self._client()
        results = client._parse_response(response, max_results=5, query="test")
        assert results == []

    # -- missing organic_results key ------------------------------------------

    def test_missing_organic_results_key(self) -> None:
        response: Dict[str, Any] = {"search_metadata": {"id": "123"}}
        client = self._client()
        results = client._parse_response(response, max_results=5, query="test")
        assert results == []

    # -- results with url key instead of link ---------------------------------

    def test_url_key_fallback(self) -> None:
        """Some engines return 'url' instead of 'link'."""
        response: Dict[str, Any] = {
            "organic_results": [
                {"url": "https://example.com/via-url", "title": "Via URL", "snippet": "test"},
            ],
        }
        client = self._client()
        results = client._parse_response(response, max_results=5, query="test")
        assert len(results) == 1
        assert results[0].url == "https://example.com/via-url"

    # -- results without any URL are skipped ----------------------------------

    def test_results_without_url_skipped(self) -> None:
        response: Dict[str, Any] = {
            "organic_results": [
                {"title": "No URL Result", "snippet": "test"},  # No link/url
                {"link": "https://valid.com", "title": "Valid", "snippet": "ok"},
            ],
        }
        client = self._client()
        results = client._parse_response(response, max_results=5, query="test")
        assert len(results) == 1
        assert results[0].url == "https://valid.com"

    # -- object-style response (non-dict items) --------------------------------

    def test_object_style_items(self) -> None:
        """Handle response items that are objects rather than dicts."""

        class FakeItem:
            link = "https://obj.com/r1"
            url = ""
            title = "Object Result"
            snippet = "obj snippet"
            position = 1
            source = "obj.com"

        response: Dict[str, Any] = {"organic_results": [FakeItem()]}
        client = self._client()
        results = client._parse_response(response, max_results=5, query="test")
        assert len(results) == 1
        assert results[0].url == "https://obj.com/r1"
        assert results[0].title == "Object Result"

    # -- None source field normalises to None ---------------------------------

    def test_none_source_normalised(self) -> None:
        response: Dict[str, Any] = {
            "organic_results": [
                {"link": "https://example.com", "title": "T", "snippet": "S", "source": ""},
            ],
        }
        client = self._client()
        results = client._parse_response(response, max_results=5, query="test")
        assert results[0].source is None

    # -- non-dict response with .get attribute --------------------------------

    def test_non_dict_response_with_get(self) -> None:
        """Handle response objects that have a .get method (like SerpAPI SDK objects)."""

        class FakeResponse:
            def get(self, key: str, default: Any = None) -> Any:
                if key == "organic_results":
                    return [
                        {"link": "https://fake.com/r1", "title": "Fake", "snippet": "snip"},
                    ]
                return default

        client = self._client()
        results = client._parse_response(FakeResponse(), max_results=5, query="test")
        assert len(results) == 1
        assert results[0].url == "https://fake.com/r1"


# ---------------------------------------------------------------------------
# SerpSearchClient — _translate_exception
# ---------------------------------------------------------------------------
class TestSerpSearchClientTranslateException:
    """Tests for SerpSearchClient._translate_exception."""

    def _client(self) -> SerpSearchClient:
        return SerpSearchClient(api_key="test-key")

    def test_rate_limit_429(self) -> None:
        err = self._client()._translate_exception(Exception("HTTP 429 Too Many Requests"))
        assert err.category == "rate_limit"
        assert err.status_code == 429

    def test_rate_limit_quota(self) -> None:
        err = self._client()._translate_exception(Exception("Quota exceeded"))
        assert err.category == "rate_limit"

    def test_rate_limit_rate_keyword(self) -> None:
        err = self._client()._translate_exception(Exception("Rate limit hit"))
        assert err.category == "rate_limit"

    def test_auth_error_401(self) -> None:
        err = self._client()._translate_exception(Exception("Error 401"))
        assert err.category == "auth_error"

    def test_auth_error_403(self) -> None:
        err = self._client()._translate_exception(Exception("Error 403: Forbidden"))
        assert err.category == "auth_error"

    def test_auth_error_invalid(self) -> None:
        err = self._client()._translate_exception(Exception("Invalid API key"))
        assert err.category == "auth_error"

    def test_timeout(self) -> None:
        err = self._client()._translate_exception(Exception("Connection timeout reached"))
        assert err.category == "timeout"

    def test_unknown_error(self) -> None:
        err = self._client()._translate_exception(ValueError("Random failure"))
        assert err.category == "unknown"
        assert "ValueError" in str(err)


# ---------------------------------------------------------------------------
# SerpSearchClient — search (integration with mock serpapi module)
# ---------------------------------------------------------------------------
class TestSerpSearchClientSearch:
    """Tests for SerpSearchClient.search with mocked serpapi module."""

    def _mock_serpapi(self, response: Dict[str, Any]) -> MagicMock:
        """Create a mock serpapi module that returns the given response."""
        mock_module = MagicMock()
        mock_module.search.return_value = response
        return mock_module

    def test_successful_search(self) -> None:
        """Full success path: query → serpapi.search → parsed results."""
        mock_module = self._mock_serpapi(
            {
                "organic_results": [
                    {
                        "link": "https://example.com/1",
                        "title": "Result 1",
                        "snippet": "Snippet 1",
                        "position": 1,
                    },
                    {
                        "link": "https://example.com/2",
                        "title": "Result 2",
                        "snippet": "Snippet 2",
                        "position": 2,
                    },
                ],
            }
        )

        client = SerpSearchClient(
            api_key="test-key",
            max_retries=0,
            circuit_breaker=None,
        )

        async def run() -> List[SerpResult]:
            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                return await client.search("test query", max_results=5)

        results = asyncio.run(run())

        assert len(results) == 2
        assert results[0].url == "https://example.com/1"
        assert results[0].title == "Result 1"
        mock_module.search.assert_called_once()

        # Verify params passed to serpapi.search
        call_args = mock_module.search.call_args[0][0]
        assert call_args["q"] == "test query"
        assert call_args["engine"] == "google"
        assert call_args["api_key"] == "test-key"
        assert call_args["num"] == 5

    def test_max_results_capped_at_10(self) -> None:
        """SerpAPI caps at 10 per page."""
        mock_module = self._mock_serpapi({"organic_results": []})

        client = SerpSearchClient(api_key="test-key", max_retries=0, circuit_breaker=None)

        async def run() -> List[SerpResult]:
            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                return await client.search("test", max_results=20)

        asyncio.run(run())
        call_args = mock_module.search.call_args[0][0]
        assert call_args["num"] == 10

    def test_custom_engine(self) -> None:
        """Should pass specified engine to serpapi."""
        mock_module = self._mock_serpapi({"organic_results": []})

        client = SerpSearchClient(
            api_key="test-key",
            engine="bing",
            max_retries=0,
            circuit_breaker=None,
        )

        async def run() -> List[SerpResult]:
            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                return await client.search("test")

        asyncio.run(run())
        call_args = mock_module.search.call_args[0][0]
        assert call_args["engine"] == "bing"

    def test_serpapi_not_installed_raises(self) -> None:
        """Should raise SerpSearchError when serpapi module is not installed."""
        client = SerpSearchClient(api_key="test-key", max_retries=0, circuit_breaker=None)

        async def run() -> None:
            with patch("deepsearch.tools.serp_search._serpapi_module", None):
                with pytest.raises(SerpSearchError) as exc_info:
                    await client.search("test")
                assert exc_info.value.category == "dependency"

        asyncio.run(run())

    def test_missing_api_key_during_search(self) -> None:
        """Should raise configuration error when API key is missing during search."""
        client = SerpSearchClient(api_key=None, max_retries=0, circuit_breaker=None)

        async def run() -> None:
            mock_module = MagicMock()
            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                with patch.dict("os.environ", {}, clear=True):
                    with pytest.raises(SerpSearchError) as exc_info:
                        await client.search("test")
                    assert exc_info.value.category == "configuration"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# SerpSearchClient — circuit breaker integration
# ---------------------------------------------------------------------------
class TestSerpSearchClientCircuitBreaker:
    """Tests for circuit breaker behavior in SerpSearchClient."""

    def test_circuit_breaker_opens_after_failures(self) -> None:
        """Circuit should open after failure_threshold consecutive failures."""
        cb = CircuitBreaker(
            failure_threshold=2,
            success_threshold=1,
            timeout_seconds=60,
            name="test_serp",
        )

        client = SerpSearchClient(
            api_key="test-key",
            max_retries=0,
            circuit_breaker=cb,
        )

        call_count = 0

        def fail_search(params: dict) -> dict:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated failure")

        mock_module = MagicMock()
        mock_module.search.side_effect = fail_search

        async def run() -> None:
            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                # First failure
                with pytest.raises(SerpSearchError):
                    await client.search("test1")
                assert cb.state == CircuitState.CLOSED

                # Second failure — should open the circuit
                with pytest.raises(SerpSearchError):
                    await client.search("test2")
                assert cb.state == CircuitState.OPEN

                # Third call — circuit is open, should raise immediately
                with pytest.raises(SerpSearchError) as exc_info:
                    await client.search("test3")
                assert exc_info.value.category == "circuit_open"

        asyncio.run(run())
        # Only 2 actual API calls should have been made (3rd was blocked by CB)
        assert call_count == 2

    def test_circuit_breaker_half_open_recovery(self) -> None:
        """Circuit should recover through half-open → closed."""
        # Use a controllable clock
        current_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        def clock() -> datetime:
            return current_time

        cb = CircuitBreaker(
            failure_threshold=1,
            success_threshold=1,
            timeout_seconds=10,
            name="test_serp_recovery",
            clock=clock,
        )

        client = SerpSearchClient(
            api_key="test-key",
            max_retries=0,
            circuit_breaker=cb,
        )

        call_count = 0

        def search_side_effect(params: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")
            return {"organic_results": [{"link": "https://ok.com", "title": "OK", "snippet": ""}]}

        mock_module = MagicMock()
        mock_module.search.side_effect = search_side_effect

        async def run() -> None:
            nonlocal current_time

            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                # 1) Fail → opens circuit
                with pytest.raises(SerpSearchError):
                    await client.search("trigger-open")
                assert cb.state == CircuitState.OPEN

                # 2) Advance time past timeout → allows half-open probe
                current_time += timedelta(seconds=15)

                # 3) Succeed in half-open → closes circuit
                results = await client.search("recovery-probe")
                assert cb.state == CircuitState.CLOSED
                assert len(results) == 1

        asyncio.run(run())

    def test_search_without_circuit_breaker(self) -> None:
        """Should work normally when circuit_breaker is None."""
        mock_module = MagicMock()
        mock_module.search.return_value = {
            "organic_results": [
                {"link": "https://example.com", "title": "Test", "snippet": "snip"},
            ],
        }

        client = SerpSearchClient(
            api_key="test-key",
            max_retries=0,
            circuit_breaker=None,
        )

        async def run() -> List[SerpResult]:
            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                return await client.search("test")

        results = asyncio.run(run())
        assert len(results) == 1


# ---------------------------------------------------------------------------
# SerpSearchClient — retry logic
# ---------------------------------------------------------------------------
class TestSerpSearchClientRetry:
    """Tests for retry_with_backoff integration in SerpSearchClient."""

    def test_retries_on_transient_failure(self) -> None:
        """Should retry and eventually succeed."""
        call_count = 0

        def search_side_effect(params: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient failure")
            return {"organic_results": [{"link": "https://ok.com", "title": "OK", "snippet": ""}]}

        mock_module = MagicMock()
        mock_module.search.side_effect = search_side_effect

        client = SerpSearchClient(
            api_key="test-key",
            max_retries=3,
            backoff_base=0.001,  # Near-instant for tests
            backoff_max=0.01,
            jitter_ms=0,
            circuit_breaker=None,
        )

        async def run() -> List[SerpResult]:
            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                return await client.search("test")

        results = asyncio.run(run())
        assert len(results) == 1
        assert call_count == 3  # 2 failures + 1 success

    def test_exhausts_retries_raises_error(self) -> None:
        """Should raise after max retries exhausted."""
        mock_module = MagicMock()
        mock_module.search.side_effect = RuntimeError("Persistent failure")

        client = SerpSearchClient(
            api_key="test-key",
            max_retries=2,
            backoff_base=0.001,
            backoff_max=0.01,
            jitter_ms=0,
            circuit_breaker=None,
        )

        async def run() -> None:
            with patch("deepsearch.tools.serp_search._serpapi_module", mock_module):
                with pytest.raises(SerpSearchError):
                    await client.search("test")

        asyncio.run(run())
        # Should have called search max_retries + 1 times (initial + retries)
        assert mock_module.search.call_count == 3


# ---------------------------------------------------------------------------
# SerpSearchTool
# ---------------------------------------------------------------------------
class TestSerpSearchTool:
    """Tests for SerpSearchTool LangChain wrapper."""

    def test_tool_name_and_description(self) -> None:
        tool = SerpSearchTool(api_key="test-key")
        assert tool.name == "web_search"
        assert "SerpAPI" in tool.description

    def test_args_schema(self) -> None:
        tool = SerpSearchTool(api_key="test-key")
        assert tool.args_schema == SerpSearchInput

    def test_default_config(self) -> None:
        tool = SerpSearchTool(api_key="test-key")
        assert tool.engine == "google"
        assert tool.max_results == 5
        assert tool.max_retries == 3

    def test_custom_config(self) -> None:
        tool = SerpSearchTool(
            api_key="custom-key",
            engine="bing",
            max_results=10,
            max_retries=5,
        )
        assert tool.api_key == "custom-key"
        assert tool.engine == "bing"
        assert tool.max_results == 10
        assert tool.max_retries == 5

    @patch("deepsearch.tools.serp_search.SerpSearchClient.search")
    def test_arun_returns_source_dicts(self, mock_search: AsyncMock) -> None:
        """_arun should return a list of source dicts via SerpResult.to_source()."""
        mock_search.return_value = [
            SerpResult(url="https://r1.com", title="Result 1", snippet="S1"),
            SerpResult(url="https://r2.com", title="Result 2", snippet="S2"),
        ]

        tool = SerpSearchTool(api_key="test-key")

        async def run() -> List[Dict[str, str]]:
            return await tool._arun("test query")

        result = asyncio.run(run())

        assert len(result) == 2
        assert result[0] == {"title": "Result 1", "url": "https://r1.com", "snippet": "S1"}
        assert result[1] == {"title": "Result 2", "url": "https://r2.com", "snippet": "S2"}

    @patch("deepsearch.tools.serp_search.SerpSearchClient.search")
    def test_arun_passes_max_results(self, mock_search: AsyncMock) -> None:
        """_arun should pass max_results to client.search."""
        mock_search.return_value = []
        tool = SerpSearchTool(api_key="test-key", max_results=8)

        async def run() -> List[Dict[str, str]]:
            return await tool._arun("test")

        asyncio.run(run())
        mock_search.assert_called_once_with(query="test", max_results=8)

    @patch("deepsearch.tools.serp_search.SerpSearchClient.search")
    def test_arun_propagates_errors(self, mock_search: AsyncMock) -> None:
        """_arun should propagate SerpSearchError from the client."""
        mock_search.side_effect = SerpSearchError("API failure", category="unknown")
        tool = SerpSearchTool(api_key="test-key")

        async def run() -> None:
            with pytest.raises(SerpSearchError):
                await tool._arun("test")

        asyncio.run(run())
