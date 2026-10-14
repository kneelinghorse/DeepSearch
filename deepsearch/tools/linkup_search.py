"""Linkup web search client — replaces SerpAPI as the primary web search backend.

Uses the linkup-sdk Python package. No Google/Scholar scraping — Linkup provides
its own web crawl index. Lower cost than SerpAPI with no minimum spend tier.

API: https://api.linkup.so/v1/search
Auth: Bearer token via LINKUP_API_KEY
SDK: pip install linkup-sdk
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - linkup-sdk may not be installed
    from linkup import LinkupClient as _LinkupClient
except ImportError:  # pragma: no cover
    _LinkupClient = None  # type: ignore[assignment,misc]

try:  # pragma: no cover
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover

    class BaseTool(BaseModel):
        """Fallback BaseTool when LangChain is unavailable."""

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

_DEFAULT_LINKUP_CIRCUIT = CircuitBreaker(name="linkup_search")

# Search depth passed to Linkup API.
# "standard" is the right tradeoff: quality results without the cost of "deep".
_LINKUP_DEPTH = "standard"


class LinkupSearchError(RuntimeError):
    """Raised when Linkup search operations fail."""

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


class LinkupResult(BaseModel):
    """Normalized Linkup search result."""

    url: str = Field(default="")
    title: str = Field(default="")
    snippet: str = Field(default="")

    def to_source(self) -> Dict[str, str]:
        """Return a dict suitable for AgentState source storage."""
        payload: Dict[str, str] = {"title": self.title or "Linkup Result"}
        if self.url:
            payload["url"] = self.url
        if self.snippet:
            payload["snippet"] = self.snippet
        return payload


class LinkupSearchInput(BaseModel):
    """Input schema for the Linkup search tool."""

    query: str = Field(..., description="Web search query text")


@dataclass(slots=True)
class LinkupSearchClient:
    """Async Linkup search client with retry and circuit breaker support.

    Wraps the synchronous linkup-sdk in a thread executor so it is safe
    to call from async research loops without blocking the event loop.
    """

    api_key: Optional[str] = None
    depth: str = _LINKUP_DEPTH  # "fast" | "standard" | "deep"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = field(
        default_factory=lambda: _DEFAULT_LINKUP_CIRCUIT,
    )

    def _get_api_key(self) -> str:
        key = self.api_key or os.getenv("LINKUP_API_KEY")
        if not key:
            raise LinkupSearchError(
                "LINKUP_API_KEY is not configured",
                category="configuration",
            )
        return key

    async def search(self, query: str, *, max_results: int = 5) -> List[LinkupResult]:
        """Execute a Linkup search and return normalized results.

        Args:
            query: Natural language or keyword search query.
            max_results: Maximum number of results to return.

        Returns:
            List of LinkupResult with url, title, snippet.
        """
        query = (query or "").strip()
        if not query:
            raise LinkupSearchError("Search query cannot be empty", category="validation")

        async def _perform_search() -> List[LinkupResult]:
            if _LinkupClient is None:
                raise LinkupSearchError(
                    "linkup-sdk is not installed; pip install linkup-sdk",
                    category="dependency",
                )
            api_key = self._get_api_key()

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: _LinkupClient(api_key=api_key).search(
                    query=query,
                    depth=self.depth,
                    output_type="searchResults",
                    max_results=min(max_results, 20),
                    timeout=self.timeout,
                ),
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
            raise LinkupSearchError(
                "Linkup circuit breaker is open; search temporarily disabled",
                category="circuit_open",
            ) from exc
        except LinkupSearchError:
            raise
        except Exception as exc:
            raise self._translate_exception(exc) from exc

    def _parse_response(self, response: Any, max_results: int, query: str) -> List[LinkupResult]:
        """Parse Linkup SDK response into normalized LinkupResult objects."""
        results: List[LinkupResult] = []

        # response.results is a list of LinkupSearchTextResult / LinkupSearchImageResult
        raw_results = getattr(response, "results", None) or []

        for item in raw_results[:max_results]:
            # Skip image results — we want text sources only
            item_type = getattr(item, "type", "text")
            if item_type == "image":
                continue

            url = getattr(item, "url", "") or ""
            title = getattr(item, "name", "") or ""
            # SDK uses .content for the text snippet
            snippet = getattr(item, "content", "") or getattr(item, "snippet", "") or ""

            if url:
                results.append(LinkupResult(url=url, title=title, snippet=snippet))

        logger.debug(
            "Linkup search returned %d results for query: %s",
            len(results),
            query[:50] if query else "unknown",
        )
        return results

    def _translate_exception(self, exc: Exception) -> LinkupSearchError:
        """Translate raw exceptions to typed LinkupSearchError."""
        exc_str = str(exc).lower()
        exc_repr = str(exc)

        if "429" in exc_repr or "rate" in exc_str or "quota" in exc_str:
            return LinkupSearchError(
                "Linkup rate limit exceeded",
                category="rate_limit",
                status_code=429,
            )
        if (
            "401" in exc_repr
            or "403" in exc_repr
            or "unauthorized" in exc_str
            or "forbidden" in exc_str
        ):
            return LinkupSearchError(
                f"Linkup authentication error: {exc}",
                category="auth_error",
            )
        if "timeout" in exc_str:
            return LinkupSearchError(
                "Linkup request timed out",
                category="timeout",
            )
        return LinkupSearchError(
            f"Linkup search failed ({type(exc).__name__}): {exc}",
            category="unknown",
        )


class LinkupSearchTool(BaseTool):
    """LangChain-compatible tool exposing Linkup search to the agent LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "web_search"
    description: str = "Search the web using Linkup"
    args_schema: ClassVar[type[LinkupSearchInput]] = LinkupSearchInput
    api_key: Optional[str] = None
    depth: str = _LINKUP_DEPTH
    timeout: float = 60.0
    max_retries: int = 3
    max_results: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = Field(
        default_factory=lambda: _DEFAULT_LINKUP_CIRCUIT,
        exclude=True,
    )

    def _run(self, query: str) -> List[Dict[str, str]]:  # pragma: no cover
        return asyncio.run(self._arun(query=query))

    async def _arun(self, query: str) -> List[Dict[str, str]]:
        client = LinkupSearchClient(
            api_key=self.api_key,
            depth=self.depth,
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
    "LinkupResult",
    "LinkupSearchClient",
    "LinkupSearchError",
    "LinkupSearchInput",
    "LinkupSearchTool",
]
