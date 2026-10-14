"""Tavily web search integration and LangChain tool wrapper."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - httpx may be absent in minimal environments.
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:  # pragma: no cover - langchain_core may not be available during unit tests.
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover

    class BaseTool(BaseModel):
        """Fallback BaseTool implementation when LangChain is unavailable."""

        name: str = "tool"
        description: str = ""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime guard
            return self._run(*args, **kwargs)

        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            return await self._arun(*args, **kwargs)

        def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime guard
            raise NotImplementedError

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime guard
            raise NotImplementedError


from deepsearch.agent.resilience import CircuitBreaker, CircuitBreakerOpenError, retry_with_backoff

_DEFAULT_TAVILY_CIRCUIT = CircuitBreaker(name="tavily_web_search")


class WebSearchError(RuntimeError):
    """Raised when web search operations fail."""

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


class TavilyResult(BaseModel):
    """Normalized Tavily search entry returned by the API wrapper."""

    title: str = Field(default="")
    url: str = Field(default="")
    content: str = Field(default="")
    score: Optional[float] = Field(default=None)

    def to_source(self) -> Dict[str, str]:
        """Return a dict representation suitable for AgentState storage."""

        snippet = (self.content or "").strip()
        payload: Dict[str, str] = {"title": self.title or "Tavily Result"}
        if self.url:
            payload["url"] = self.url
        if snippet:
            payload["snippet"] = snippet
        return payload


class TavilySearchInput(BaseModel):
    """Input schema for the Tavily search tool."""

    query: str = Field(..., description="Web search query text")


@dataclass(slots=True)
class TavilySearchClient:
    """Async Tavily API wrapper with retry and timeout support."""

    api_key: Optional[str] = None
    base_url: str = "https://api.tavily.com"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = field(
        default_factory=lambda: _DEFAULT_TAVILY_CIRCUIT,
    )

    async def search(self, query: str, *, max_results: int = 5) -> List[TavilyResult]:
        """Execute a Tavily search query and return normalized results."""

        key = self.api_key or os.getenv("TAVILY_API_KEY")
        if not key:
            raise WebSearchError("TAVILY_API_KEY is not configured", category="configuration")
        if httpx is None:
            raise WebSearchError(
                "httpx is not installed; install project dependencies",
                category="dependency",
            )
        query = (query or "").strip()
        if not query:
            raise WebSearchError("Search query cannot be empty", category="validation")

        payload: Dict[str, Any] = {
            "api_key": key,
            "query": query,
            "num_results": max_results,
        }
        url = f"{self.base_url.rstrip('/')}/search"

        async def _perform_request() -> List[TavilyResult]:
            if httpx is None:  # pragma: no cover - defensive
                raise WebSearchError(
                    "httpx is not installed; install project dependencies",
                    category="dependency",
                )
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            raw_results = data.get("results") or []
            return [TavilyResult(**item) for item in raw_results[:max_results]]

        retriable: Tuple[type[BaseException], ...]
        if httpx is None:  # pragma: no cover - defensive
            retriable = (Exception,)
        else:
            retriable = (httpx.HTTPError, ValueError)
        wrapped = retry_with_backoff(
            max_retries=self.max_retries,
            base_delay=self.backoff_base,
            max_delay=self.backoff_max,
            jitter_ms=self.jitter_ms,
            retriable_exceptions=retriable,
        )(_perform_request)
        try:
            if self.circuit_breaker:
                return await self.circuit_breaker.call(wrapped)
            return await wrapped()
        except CircuitBreakerOpenError as exc:
            raise WebSearchError(
                "Tavily circuit breaker is open; web search temporarily disabled",
                category="circuit_open",
            ) from exc
        except WebSearchError:
            raise
        except Exception as exc:
            raise self._translate_exception(exc) from exc

    def _translate_exception(self, exc: Exception) -> WebSearchError:
        if httpx is not None:
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status == 429:
                    return WebSearchError(
                        "Tavily rate limit exceeded",
                        category="rate_limit",
                        status_code=status,
                    )
                if status >= 500:
                    return WebSearchError(
                        f"Tavily service error (HTTP {status})",
                        category="service_error",
                        status_code=status,
                    )
                return WebSearchError(
                    f"Tavily HTTP error (HTTP {status})",
                    category="http_error",
                    status_code=status,
                )
            if isinstance(exc, httpx.TimeoutException):
                return WebSearchError("Tavily request timed out", category="timeout")
            if isinstance(exc, httpx.RequestError):
                return WebSearchError(f"Tavily network error: {exc}", category="network")
        if isinstance(exc, WebSearchError):
            return exc
        return WebSearchError(f"Tavily search failed: {exc}", category="unknown")


class TavilySearchTool(BaseTool):
    """LangChain-compatible tool exposing Tavily search to the agent LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "web_search"
    description: str = "Search the web for up-to-date research"
    args_schema: ClassVar[type[TavilySearchInput]] = TavilySearchInput
    api_key: Optional[str] = None
    base_url: str = "https://api.tavily.com"
    timeout: float = 60.0
    max_retries: int = 3
    max_results: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = Field(
        default_factory=lambda: _DEFAULT_TAVILY_CIRCUIT,
        exclude=True,
    )

    def _run(self, query: str) -> List[Dict[str, str]]:  # pragma: no cover - sync fallback
        return asyncio.run(self._arun(query=query))

    async def _arun(self, query: str) -> List[Dict[str, str]]:
        client = TavilySearchClient(
            api_key=self.api_key,
            base_url=self.base_url,
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
    "TavilyResult",
    "TavilySearchClient",
    "TavilySearchInput",
    "TavilySearchTool",
    "WebSearchError",
]
