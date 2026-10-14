"""SerpAPI web search integration for triple-search source diversity.

B13.7: Adds SerpAPI as a third search engine alongside Google and Exa.
SerpAPI scrapes multiple search indexes (Google, Bing, DuckDuckGo, Yahoo),
providing access to results from different crawl pipelines for greater
source diversity in Alpha-tier missions.

Architecture follows the established exa_search.py pattern:
- SerpSearchClient: async dataclass client with retry + circuit breaker
- SerpResult: normalized result model with to_source() for AgentState
- SerpSearchTool: LangChain-compatible tool wrapper
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - serpapi may not be installed
    import serpapi as _serpapi_module
except ImportError:  # pragma: no cover
    _serpapi_module = None  # type: ignore[assignment]

try:  # pragma: no cover - langchain_core may not be available during unit tests
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover

    class BaseTool(BaseModel):
        """Fallback BaseTool implementation when LangChain is unavailable."""

        name: str = "tool"
        description: str = ""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            return self._run(*args, **kwargs)

        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            return await self._arun(*args, **kwargs)

        def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError


from deepsearch.agent.resilience import CircuitBreaker, CircuitBreakerOpenError, retry_with_backoff

logger = logging.getLogger(__name__)

_DEFAULT_SERP_CIRCUIT = CircuitBreaker(name="serp_search")


class SerpSearchError(RuntimeError):
    """Raised when SerpAPI search operations fail."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


class SerpResult(BaseModel):
    """Normalized SerpAPI search result."""

    url: str = Field(default="")
    title: str = Field(default="")
    snippet: str = Field(default="")
    position: Optional[int] = Field(default=None)
    source: Optional[str] = Field(default=None)

    def to_source(self) -> Dict[str, str]:
        """Return a dict representation suitable for AgentState storage."""
        payload: Dict[str, str] = {"title": self.title or "SerpAPI Result"}
        if self.url:
            payload["url"] = self.url
        if self.snippet:
            payload["snippet"] = self.snippet
        return payload


class SerpSearchInput(BaseModel):
    """Input schema for the SerpAPI search tool."""

    query: str = Field(..., description="Web search query text")


@dataclass(slots=True)
class SerpSearchClient:
    """Async SerpAPI wrapper with retry and circuit breaker support.

    Uses SerpAPI to access multiple search indexes (Google, Bing, etc.)
    for greater source diversity in Alpha-tier research missions.
    """

    api_key: Optional[str] = None
    engine: str = "google"  # "google", "bing", "duckduckgo", "yahoo"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = field(
        default_factory=lambda: _DEFAULT_SERP_CIRCUIT,
    )

    def _get_api_key(self) -> str:
        """Get configured SerpAPI key."""
        key = self.api_key or os.getenv("SERPAPI_API_KEY")
        if not key:
            raise SerpSearchError(
                "SERPAPI_API_KEY is not configured",
                category="configuration",
            )
        return key

    async def search(self, query: str, *, max_results: int = 5) -> List[SerpResult]:
        """Execute a SerpAPI search query and return normalized results.

        Args:
            query: The search query to execute
            max_results: Maximum number of results to return (default 5)

        Returns:
            List of SerpResult objects with url, title, snippet
        """
        query = (query or "").strip()
        if not query:
            raise SerpSearchError("Search query cannot be empty", category="validation")

        async def _perform_search() -> List[SerpResult]:
            if _serpapi_module is None:
                raise SerpSearchError(
                    "serpapi is not installed; pip install serpapi",
                    category="dependency",
                )
            api_key = self._get_api_key()

            params = {
                "q": query,
                "engine": self.engine,
                "api_key": api_key,
                "num": min(max_results, 10),  # SerpAPI caps at 10 per page
            }

            # Run sync call in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: _serpapi_module.search(params),
            )

            return self._parse_response(response, max_results, query)

        retriable: Tuple[type[BaseException], ...] = (Exception,)
        wrapped = retry_with_backoff(
            max_retries=self.max_retries,
            base_delay=self.backoff_base,
            max_delay=self.backoff_max,
            jitter_ms=self.jitter_ms,
            retriable_exceptions=retriable,
        )(_perform_search)

        try:
            if self.circuit_breaker:
                return await self.circuit_breaker.call(wrapped)
            return await wrapped()
        except CircuitBreakerOpenError as exc:
            raise SerpSearchError(
                "SerpAPI circuit breaker is open; search temporarily disabled",
                category="circuit_open",
            ) from exc
        except SerpSearchError:
            raise
        except Exception as exc:
            raise self._translate_exception(exc) from exc

    def _parse_response(self, response: Any, max_results: int, query: str) -> List[SerpResult]:
        """Parse SerpAPI response into normalized SerpResult objects."""
        results: List[SerpResult] = []

        # SerpAPI returns organic_results in the response dict
        if isinstance(response, dict):
            raw_results = response.get("organic_results", []) or []
        else:
            raw_results = getattr(response, "get", lambda k, d=None: d)("organic_results", []) or []

        for item in raw_results[:max_results]:
            if isinstance(item, dict):
                url = item.get("link", "") or item.get("url", "") or ""
                title = item.get("title", "") or ""
                snippet = item.get("snippet", "") or ""
                position = item.get("position")
                source = item.get("source", "")
            else:
                url = getattr(item, "link", "") or getattr(item, "url", "") or ""
                title = getattr(item, "title", "") or ""
                snippet = getattr(item, "snippet", "") or ""
                position = getattr(item, "position", None)
                source = getattr(item, "source", "") or ""

            if url:  # Only include results with valid URLs
                results.append(SerpResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    position=position,
                    source=source or None,
                ))

        logger.debug(
            "SerpAPI search returned %d results for query: %s",
            len(results),
            query[:50] if query else "unknown",
        )

        return results

    def _translate_exception(self, exc: Exception) -> SerpSearchError:
        """Translate various exceptions to SerpSearchError."""
        exc_type = type(exc).__name__
        exc_str = str(exc).lower()

        # Check for rate limiting
        if "429" in str(exc) or "rate" in exc_str or "quota" in exc_str:
            return SerpSearchError(
                "SerpAPI rate limit exceeded",
                category="rate_limit",
                status_code=429,
            )

        # Check for auth errors
        if "401" in str(exc) or "403" in str(exc) or "invalid" in exc_str:
            return SerpSearchError(
                f"SerpAPI authentication error: {exc}",
                category="auth_error",
            )

        # Check for timeout
        if "timeout" in exc_str:
            return SerpSearchError(
                "SerpAPI request timed out",
                category="timeout",
            )

        return SerpSearchError(
            f"SerpAPI search failed ({exc_type}): {exc}",
            category="unknown",
        )


class SerpSearchTool(BaseTool):
    """LangChain-compatible tool exposing SerpAPI search to the agent LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "web_search"
    description: str = "Search the web using SerpAPI for diverse source coverage"
    args_schema: ClassVar[type[SerpSearchInput]] = SerpSearchInput
    api_key: Optional[str] = None
    engine: str = "google"
    timeout: float = 60.0
    max_retries: int = 3
    max_results: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = Field(
        default_factory=lambda: _DEFAULT_SERP_CIRCUIT,
        exclude=True,
    )

    def _run(self, query: str) -> List[Dict[str, str]]:  # pragma: no cover - sync fallback
        return asyncio.run(self._arun(query=query))

    async def _arun(self, query: str) -> List[Dict[str, str]]:
        client = SerpSearchClient(
            api_key=self.api_key,
            engine=self.engine,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            backoff_max=self.backoff_max,
            jitter_ms=self.jitter_ms,
            circuit_breaker=self.circuit_breaker,
        )
        results = await client.search(query=query, max_results=self.max_results)
        return [item.to_source() for item in results]


__all__ = [
    "SerpResult",
    "SerpSearchClient",
    "SerpSearchError",
    "SerpSearchInput",
    "SerpSearchTool",
]
