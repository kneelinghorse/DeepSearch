"""OpenAlex native search client (B24.4).

Uses the free OpenAlex API. Returns publisher/DOI URLs — NOT openalex.org URLs —
so domain scoring in source_quality.py applies to the actual publisher.

API: https://api.openalex.org/works?search=...
Docs: https://docs.openalex.org/api-entities/works/search-works
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openalex.org/works"
_SNIPPET_MAX_CHARS = 500
_EMAIL_PARAM = "deepseearch@example.com"  # polite pool — improves rate limits


class OpenAlexError(RuntimeError):
    """Raised when OpenAlex API operations fail."""

    def __init__(self, message: str, *, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category


@dataclass
class OpenAlexResult:
    """Normalized OpenAlex search result with publisher URL."""

    url: str        # Publisher/DOI URL — NOT openalex.org
    title: str
    snippet: str    # Abstract excerpt (if available)

    def to_source(self) -> Dict[str, str]:
        """Return a dict suitable for AgentState source storage."""
        payload: Dict[str, str] = {"title": self.title or "Academic Paper"}
        if self.url:
            payload["url"] = self.url
        if self.snippet:
            payload["snippet"] = self.snippet
        return payload


@dataclass
class OpenAlexClient:
    """Async OpenAlex API client.

    Returns publisher/DOI URLs so HIGH_AUTHORITY_DOMAINS scoring applies.
    Works without a publisher URL are skipped.
    """

    api_key: Optional[str] = None  # Used as email for polite pool
    timeout: float = 20.0

    async def search(self, query: str, *, max_results: int = 5) -> List[OpenAlexResult]:
        """Search OpenAlex and return results with publisher URLs.

        Args:
            query: Research query.
            max_results: Maximum number of results to return.

        Returns:
            List of OpenAlexResult with publisher/DOI URLs.

        Raises:
            OpenAlexError: On network failure or API error.
        """
        query = (query or "").strip()
        if not query:
            raise OpenAlexError("Query cannot be empty", category="validation")

        # OpenAlex uses email for polite pool — better rate limits
        email = self.api_key or _EMAIL_PARAM
        params = urllib.parse.urlencode({
            "search": query,
            "per_page": str(min(max_results * 3, 50)),
            "select": "id,doi,title,primary_location,open_access,abstract_inverted_index",
            "mailto": email,
        })
        url = f"{_BASE_URL}?{params}"

        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, lambda: self._fetch(url))
        except OpenAlexError:
            raise
        except Exception as exc:
            raise OpenAlexError(str(exc), category="network") from exc

        return self._parse_response(raw, max_results)

    def _fetch(self, url: str) -> str:
        """Synchronous fetch — called in thread executor."""
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "DeepSearch/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            raise OpenAlexError(str(exc), category="network") from exc

    def _parse_response(self, raw: str, max_results: int) -> List[OpenAlexResult]:
        """Parse OpenAlex JSON response into results with publisher URLs."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenAlexError(f"JSON parse error: {exc}", category="parse") from exc

        works = data.get("results", [])
        if not isinstance(works, list):
            return []

        results: List[OpenAlexResult] = []
        for work in works:
            if len(results) >= max_results:
                break

            publisher_url = self._extract_publisher_url(work)
            if not publisher_url:
                continue

            title = (work.get("title") or "").strip()
            snippet = self._extract_snippet(work)

            results.append(OpenAlexResult(url=publisher_url, title=title, snippet=snippet))

        logger.debug("OpenAlex search returned %d results with publisher URLs", len(results))
        return results

    @staticmethod
    def _extract_publisher_url(work: Dict) -> Optional[str]:
        """Extract publisher/DOI URL from work metadata.

        Priority:
        1. doi (OpenAlex provides full DOI URL like https://doi.org/10.xxx/...)
        2. primary_location.landing_page_url
        3. open_access.oa_url

        Does NOT return openalex.org URLs.
        """
        # Priority 1: DOI (already a full URL in OpenAlex)
        doi = (work.get("doi") or "").strip()
        if doi and doi.startswith("http"):
            return doi

        # Priority 2: primary location landing page
        primary = work.get("primary_location") or {}
        landing = (primary.get("landing_page_url") or "").strip()
        if landing and landing.startswith("http") and "openalex.org" not in landing:
            return landing

        # Priority 3: OA URL
        oa = work.get("open_access") or {}
        oa_url = (oa.get("oa_url") or "").strip()
        if oa_url and oa_url.startswith("http") and "openalex.org" not in oa_url:
            return oa_url

        return None

    @staticmethod
    def _extract_snippet(work: Dict) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        inverted = work.get("abstract_inverted_index")
        if not inverted or not isinstance(inverted, dict):
            return ""

        # Inverted index: {word: [position1, position2, ...]}
        # Reconstruct ordered list of (position, word) pairs
        position_word: List[tuple[int, str]] = []
        for word, positions in inverted.items():
            for pos in positions:
                position_word.append((pos, word))

        position_word.sort(key=lambda x: x[0])
        abstract = " ".join(w for _, w in position_word)

        if len(abstract) > _SNIPPET_MAX_CHARS:
            abstract = abstract[:_SNIPPET_MAX_CHARS - 3] + "..."
        return abstract
