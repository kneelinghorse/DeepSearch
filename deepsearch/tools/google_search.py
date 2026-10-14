"""Google Search grounding integration using native Gemini API.

This module provides GoogleSearchClient and GoogleSearchTool for web search
using Google's native grounding capabilities instead of third-party APIs.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - google-genai may not be installed
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

try:  # pragma: no cover - langchain_core may not be available during unit tests.
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

_DEFAULT_GOOGLE_CIRCUIT = CircuitBreaker(name="google_search")


class GoogleSearchError(RuntimeError):
    """Raised when Google Search operations fail."""

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


@dataclass(slots=True)
class GroundingChunk:
    """A source chunk from Google Search grounding."""

    uri: str
    title: str

    def to_source(self) -> Dict[str, str]:
        """Return a dict representation suitable for AgentState storage."""
        return {
            "title": self.title or "Google Search Result",
            "url": self.uri,
            "snippet": "",  # Google doesn't provide snippets in groundingChunks
        }


@dataclass(slots=True)
class GroundingSupport:
    """Maps text segments to source chunks."""

    start_index: int
    end_index: int
    chunk_indices: List[int]
    confidence_scores: List[float] = field(default_factory=list)


@dataclass(slots=True)
class GoogleSearchResult:
    """Complete result from a grounded Google Search query."""

    text: str
    sources: List[GroundingChunk]
    supports: List[GroundingSupport]
    queries_used: List[str]

    def to_enriched_sources(self, max_snippet_len: int = 1500) -> List[Dict[str, str]]:
        """Build source dicts with snippets extracted from grounding supports.

        Maps grounding supports (text span → chunk index) back to source dicts
        so each source gets real content instead of an empty snippet.
        """
        # Map chunk_index -> list of text spans from supports
        chunk_texts: Dict[int, List[str]] = {}
        for support in self.supports:
            text_span = self.text[support.start_index:support.end_index]
            if not text_span.strip():
                continue
            for idx in support.chunk_indices:
                chunk_texts.setdefault(idx, []).append(text_span.strip())

        enriched: List[Dict[str, str]] = []
        for i, source in enumerate(self.sources):
            spans = chunk_texts.get(i, [])
            snippet = " ".join(spans)[:max_snippet_len] if spans else ""
            # Fallback: use a portion of the full response text
            if not snippet and self.text:
                snippet = self.text[:max_snippet_len]
            enriched.append({
                "title": source.title or "Google Search Result",
                "url": source.uri,
                "snippet": snippet,
            })
        return enriched


class GoogleSearchInput(BaseModel):
    """Input schema for the Google Search tool."""

    query: str = Field(..., description="Web search query text")


@dataclass(slots=True)
class GoogleSearchClient:
    """Async Google Search client using native Gemini grounding.

    Uses the google-generativeai SDK to perform grounded searches via
    Gemini's google_search tool, extracting citations and source metadata.
    """

    api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = field(
        default_factory=lambda: _DEFAULT_GOOGLE_CIRCUIT,
    )

    def _get_client(self) -> Any:
        """Get configured genai client."""
        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise GoogleSearchError(
                "GEMINI_API_KEY is not configured",
                category="configuration",
            )
        if genai is None:
            raise GoogleSearchError(
                "google-generativeai is not installed; pip install google-generativeai",
                category="dependency",
            )
        return genai.Client(api_key=key)

    async def search(self, query: str, *, max_results: int = 5) -> GoogleSearchResult:
        """Execute a grounded Google Search query and return structured results.

        Args:
            query: The search query to execute
            max_results: Maximum number of source chunks to return (default 5)

        Returns:
            GoogleSearchResult with text, sources, supports, and queries_used
        """
        query = (query or "").strip()
        if not query:
            raise GoogleSearchError("Search query cannot be empty", category="validation")

        async def _perform_search() -> GoogleSearchResult:
            client = self._get_client()

            # Configure google_search as a tool with timeout
            grounding_tool = genai_types.Tool(
                google_search=genai_types.GoogleSearch()
            )
            # Convert timeout from seconds to milliseconds for HTTP options
            http_opts = genai_types.HttpOptions(timeout=int(self.timeout * 1000))
            config = genai_types.GenerateContentConfig(
                tools=[grounding_tool],
                http_options=http_opts,
            )

            # Craft a search-focused prompt
            search_prompt = f"""Search the web and provide factual, well-sourced information about: {query}

Focus on finding authoritative, recent sources. Cite specific facts with their sources."""

            # Run sync call in thread pool to avoid blocking
            # Timeout is configured in http_options within config above
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.model,
                    contents=search_prompt,
                    config=config,
                )
            )

            return self._parse_response(response, max_results)

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
            raise GoogleSearchError(
                "Google Search circuit breaker is open; search temporarily disabled",
                category="circuit_open",
            ) from exc
        except GoogleSearchError:
            raise
        except Exception as exc:
            raise self._translate_exception(exc) from exc

    def _parse_response(self, response: Any, max_results: int) -> GoogleSearchResult:
        """Parse Gemini response and extract grounding metadata."""
        text = ""
        sources: List[GroundingChunk] = []
        supports: List[GroundingSupport] = []
        queries_used: List[str] = []

        # Extract text content - handle None values at all levels
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = getattr(candidate.content, "parts", None) or []
                text = "".join(getattr(p, "text", "") or "" for p in parts)

        # Extract grounding metadata
        grounding_metadata = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)

        if grounding_metadata:
            # Extract search queries used (handle None)
            raw_queries = getattr(grounding_metadata, "web_search_queries", None) or []
            if raw_queries:
                queries_used = list(raw_queries)[:5]

            # Extract grounding chunks (sources) - handle None
            raw_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
            for chunk in raw_chunks[:max_results]:
                web = getattr(chunk, "web", None)
                if web:
                    uri = getattr(web, "uri", "") or ""
                    title = getattr(web, "title", "") or ""
                    if uri:
                        sources.append(GroundingChunk(uri=uri, title=title))

            # Extract grounding supports (text-to-source mappings) - handle None
            raw_supports = getattr(grounding_metadata, "grounding_supports", None) or []
            for support in raw_supports:
                segment = getattr(support, "segment", None)
                if segment:
                    start = getattr(segment, "start_index", 0) or 0
                    end = getattr(segment, "end_index", 0) or 0
                    chunk_indices = list(getattr(support, "grounding_chunk_indices", None) or [])
                    confidence = list(getattr(support, "confidence_scores", None) or [])
                    if chunk_indices:
                        supports.append(GroundingSupport(
                            start_index=start,
                            end_index=end,
                            chunk_indices=chunk_indices,
                            confidence_scores=confidence,
                        ))

        query_summary = queries_used[0][:50] if queries_used else "unknown"
        logger.debug(
            "Google Search returned %d sources, %d supports for query: %s",
            len(sources),
            len(supports),
            query_summary,
        )

        return GoogleSearchResult(
            text=text,
            sources=sources,
            supports=supports,
            queries_used=queries_used,
        )

    def _translate_exception(self, exc: Exception) -> GoogleSearchError:
        """Translate various exceptions to GoogleSearchError."""
        exc_type = type(exc).__name__

        # Check for rate limiting
        if "429" in str(exc) or "quota" in str(exc).lower():
            return GoogleSearchError(
                "Google API rate limit exceeded",
                category="rate_limit",
                status_code=429,
            )

        # Check for auth errors
        if "401" in str(exc) or "403" in str(exc) or "invalid" in str(exc).lower():
            return GoogleSearchError(
                f"Google API authentication error: {exc}",
                category="auth_error",
            )

        # Check for timeout
        if "timeout" in str(exc).lower():
            return GoogleSearchError(
                "Google API request timed out",
                category="timeout",
            )

        return GoogleSearchError(
            f"Google Search failed ({exc_type}): {exc}",
            category="unknown",
        )


class GoogleSearchTool(BaseTool):
    """LangChain-compatible tool exposing Google Search grounding to the agent LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "web_search"
    description: str = "Search the web using Google Search for up-to-date research"
    args_schema: ClassVar[type[GoogleSearchInput]] = GoogleSearchInput
    api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    timeout: float = 60.0
    max_retries: int = 3
    max_results: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter_ms: int = 100
    circuit_breaker: CircuitBreaker | None = Field(
        default_factory=lambda: _DEFAULT_GOOGLE_CIRCUIT,
        exclude=True,
    )

    def _run(self, query: str) -> List[Dict[str, str]]:  # pragma: no cover - sync fallback
        return asyncio.run(self._arun(query=query))

    async def _arun(self, query: str) -> List[Dict[str, str]]:
        client = GoogleSearchClient(
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            backoff_max=self.backoff_max,
            jitter_ms=self.jitter_ms,
            circuit_breaker=self.circuit_breaker,
        )
        result = await client.search(query=query, max_results=self.max_results)
        return [source.to_source() for source in result.sources]


def extract_grounded_citations(
    result: GoogleSearchResult,
) -> List[Dict[str, Any]]:
    """Extract inline citation mappings from a GoogleSearchResult.

    Returns a list of citation objects with text spans and source URLs,
    suitable for report generation with inline citations.
    """
    citations: List[Dict[str, Any]] = []

    for support in result.supports:
        if support.start_index >= support.end_index:
            continue

        text_span = result.text[support.start_index:support.end_index]
        source_urls: List[str] = []

        for idx in support.chunk_indices:
            if 0 <= idx < len(result.sources):
                source_urls.append(result.sources[idx].uri)

        if source_urls:
            citations.append({
                "text": text_span,
                "start_index": support.start_index,
                "end_index": support.end_index,
                "sources": source_urls,
                "confidence": (
                    support.confidence_scores[0]
                    if support.confidence_scores
                    else None
                ),
            })

    return citations


__all__ = [
    "GoogleSearchClient",
    "GoogleSearchError",
    "GoogleSearchInput",
    "GoogleSearchResult",
    "GoogleSearchTool",
    "GroundingChunk",
    "GroundingSupport",
    "extract_grounded_citations",
]
