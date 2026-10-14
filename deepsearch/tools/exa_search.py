"""Exa.ai web search integration.

This module provides ExaSearchClient and ExaSearchTool for web search
using Exa's meaning-based (embedding) search API, designed for AI applications.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - exa_py may not be installed
    from exa_py import Exa
except ImportError:  # pragma: no cover
    Exa = None  # type: ignore[assignment,misc]

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

_DEFAULT_EXA_CIRCUIT = CircuitBreaker(name="exa_search")

_ARXIV_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?arxiv\.org/(?:pdf|abs|html)/(?P<identifier>[\w./-]+?)(?:\.pdf)?$",
    flags=re.IGNORECASE,
)


def _title_from_url(url: str) -> str:
    """Derive a deterministic title from a URL when Exa returns no title.

    Why: Exa's content extractor returns an empty ``title`` for PDF and binary
    endpoints, most commonly ``arxiv.org/pdf/<id>``. Without a sensible
    fallback the source rendered as ``Exa Result - <url>`` in the references
    list (S52.2 / D-S51-B), a defect that survived since pre-S40 because
    earlier work migrated source citation resolution but not source title
    resolution. This helper prefers structured identifiers (``arxiv:<id>``)
    and otherwise returns a hostname-derived label so reference rows stay
    readable for any URL shape.
    """

    if not url:
        return ""
    candidate = url.strip()
    arxiv_match = _ARXIV_URL_PATTERN.match(candidate)
    if arxiv_match:
        identifier = arxiv_match.group("identifier").strip("/")
        if identifier:
            return f"arxiv:{identifier}"
    host = (urlparse(candidate).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host:
        return f"Source from {host}"
    return ""


class ExaSearchError(RuntimeError):
    """Raised when Exa search operations fail."""

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


class ExaResult(BaseModel):
    """Normalized Exa search result."""

    url: str = Field(default="")
    title: str = Field(default="")
    text: str = Field(default="")
    highlights: List[str] = Field(default_factory=list)
    published_date: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    score: Optional[float] = Field(default=None)

    def to_source(self) -> Dict[str, str]:
        """Return a dict representation suitable for AgentState storage."""
        # Use highlights if available, otherwise use text
        snippet = ""
        if self.highlights:
            snippet = " ... ".join(self.highlights[:4])
        elif self.text:
            snippet = self.text[:2000]

        title = (self.title or "").strip()
        if not title:
            title = _title_from_url(self.url) or "Exa Result"
        payload: Dict[str, str] = {"title": title}
        if self.url:
            payload["url"] = self.url
        if snippet:
            payload["snippet"] = snippet
        return payload


class ExaSearchInput(BaseModel):
    """Input schema for the Exa search tool."""

    query: str = Field(..., description="Web search query text")


@dataclass(slots=True)
class ExaSearchClient:
    """Async Exa API wrapper with retry and circuit breaker support.

    Uses Exa's meaning-based (embedding) search which is optimized for
    AI applications and provides better semantic matching than keyword search.
    """

    api_key: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    search_type: str = "auto"  # "auto", "neural", "fast", or "deep"
    include_text: bool = True  # Get full text content
    include_highlights: bool = True  # Get relevant highlights
    circuit_breaker: CircuitBreaker | None = field(
        default_factory=lambda: _DEFAULT_EXA_CIRCUIT,
    )

    def _get_client(self) -> Any:
        """Get configured Exa client."""
        if Exa is None:
            raise ExaSearchError(
                "exa_py is not installed; pip install exa_py",
                category="dependency",
            )
        key = self.api_key or os.getenv("EXA_API_KEY")
        if not key:
            raise ExaSearchError(
                "EXA_API_KEY is not configured",
                category="configuration",
            )
        return Exa(api_key=key)

    async def search(self, query: str, *, max_results: int = 5) -> List[ExaResult]:
        """Execute an Exa search query and return normalized results.

        Args:
            query: The search query to execute
            max_results: Maximum number of results to return (default 5)

        Returns:
            List of ExaResult objects with url, title, text, highlights
        """
        query = (query or "").strip()
        if not query:
            raise ExaSearchError("Search query cannot be empty", category="validation")

        async def _perform_search() -> List[ExaResult]:
            client = self._get_client()

            # Run sync call in thread pool to avoid blocking
            loop = asyncio.get_running_loop()

            if self.include_text or self.include_highlights:
                # Use search_and_contents for richer results
                response = await loop.run_in_executor(
                    None,
                    lambda: client.search_and_contents(
                        query,
                        num_results=max_results,
                        type=self.search_type,
                        text=self.include_text,
                        highlights=self.include_highlights,
                    )
                )
            else:
                # Basic search (links only)
                response = await loop.run_in_executor(
                    None,
                    lambda: client.search(
                        query,
                        num_results=max_results,
                        type=self.search_type,
                    )
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
            raise ExaSearchError(
                "Exa circuit breaker is open; search temporarily disabled",
                category="circuit_open",
            ) from exc
        except ExaSearchError:
            raise
        except Exception as exc:
            raise self._translate_exception(exc) from exc

    def _parse_response(self, response: Any, max_results: int, query: str) -> List[ExaResult]:
        """Parse Exa response into normalized ExaResult objects."""
        results: List[ExaResult] = []

        # Exa returns a SearchResponse with .results attribute
        raw_results = getattr(response, "results", []) or []

        for item in raw_results[:max_results]:
            result = ExaResult(
                url=getattr(item, "url", "") or "",
                title=getattr(item, "title", "") or "",
                text=getattr(item, "text", "") or "",
                highlights=list(getattr(item, "highlights", []) or []),
                published_date=getattr(item, "published_date", None),
                author=getattr(item, "author", None),
                score=getattr(item, "score", None),
            )
            if result.url:  # Only include results with valid URLs
                results.append(result)

        logger.debug(
            "Exa search returned %d results for query: %s",
            len(results),
            query[:50] if query else "unknown",
        )

        return results

    def _translate_exception(self, exc: Exception) -> ExaSearchError:
        """Translate various exceptions to ExaSearchError."""
        exc_type = type(exc).__name__
        exc_str = str(exc).lower()

        # Check for rate limiting
        if "429" in str(exc) or "rate" in exc_str or "quota" in exc_str:
            return ExaSearchError(
                "Exa API rate limit exceeded",
                category="rate_limit",
                status_code=429,
            )

        # Check for auth errors
        if "401" in str(exc) or "403" in str(exc) or "unauthorized" in exc_str or "invalid" in exc_str:
            return ExaSearchError(
                f"Exa API authentication error: {exc}",
                category="auth_error",
            )

        # Check for timeout
        if "timeout" in exc_str:
            return ExaSearchError(
                "Exa API request timed out",
                category="timeout",
            )

        return ExaSearchError(
            f"Exa search failed ({exc_type}): {exc}",
            category="unknown",
        )


class ExaSearchTool(BaseTool):
    """LangChain-compatible tool exposing Exa search to the agent LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "web_search"
    description: str = "Search the web using Exa's meaning-based search for research"
    args_schema: ClassVar[type[ExaSearchInput]] = ExaSearchInput
    api_key: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    max_results: int = 5
    search_type: str = "auto"
    include_text: bool = True
    include_highlights: bool = True
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = Field(
        default_factory=lambda: _DEFAULT_EXA_CIRCUIT,
        exclude=True,
    )

    def _run(self, query: str) -> List[Dict[str, str]]:  # pragma: no cover - sync fallback
        return asyncio.run(self._arun(query=query))

    async def _arun(self, query: str) -> List[Dict[str, str]]:
        client = ExaSearchClient(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            backoff_max=self.backoff_max,
            jitter_ms=self.jitter_ms,
            search_type=self.search_type,
            include_text=self.include_text,
            include_highlights=self.include_highlights,
            circuit_breaker=self.circuit_breaker,
        )
        results = await client.search(query=query, max_results=self.max_results)
        return [item.to_source() for item in results]


__all__ = [
    "ExaResult",
    "ExaSearchClient",
    "ExaSearchError",
    "ExaSearchInput",
    "ExaSearchTool",
]
