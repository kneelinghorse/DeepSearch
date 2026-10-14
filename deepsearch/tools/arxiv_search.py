"""arXiv native search client using the free arXiv Atom API (no key required).

B24.3: Provides an always-on academic search lane for Deep/Alpha tiers.
Results are arxiv.org URLs which score 0.9 in HIGH_AUTHORITY_DOMAINS.

API endpoint: https://export.arxiv.org/api/query
Returns Atom XML parsed with stdlib xml.etree.ElementTree.
No new pip dependencies — uses stdlib urllib + xml.etree + asyncio.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_SNIPPET_MAX_CHARS = 500


class ArxivSearchError(RuntimeError):
    """Raised when arXiv search operations fail."""

    def __init__(self, message: str, *, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category


@dataclass
class ArxivResult:
    """Normalized arXiv search result."""

    url: str
    title: str
    snippet: str  # abstract excerpt

    def to_source(self) -> Dict[str, str]:
        """Return a dict suitable for AgentState source storage."""
        payload: Dict[str, str] = {"title": self.title or "arXiv Paper"}
        if self.url:
            payload["url"] = self.url
        if self.snippet:
            payload["snippet"] = self.snippet
        return payload


@dataclass
class ArxivSearchClient:
    """Async arXiv search client using the free Atom API.

    No API key required. Rate limit: ~3 requests/second per arXiv policy.
    Non-fatal on network or parse errors — caller should catch ArxivSearchError.
    """

    timeout: float = 20.0
    # Retry configuration for 429 rate-limit responses.
    # arXiv allows ~3 req/sec; deep research runs fire many queries quickly.
    _rate_limit_retries: int = 2
    _rate_limit_backoff: float = 5.0  # seconds between retries on 429

    async def search(self, query: str, *, max_results: int = 5) -> List[ArxivResult]:
        """Search arXiv and return normalized results.

        Args:
            query: Natural language or keyword search query.
            max_results: Maximum number of results to return.

        Returns:
            List of ArxivResult with arxiv.org URLs, titles, and abstract snippets.

        Raises:
            ArxivSearchError: On network failure, XML parse error, or exhausted retries.
        """
        query = (query or "").strip()
        if not query:
            raise ArxivSearchError("Search query cannot be empty", category="validation")

        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "max_results": str(min(max_results, 20)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        url = f"{_ARXIV_API_URL}?{params}"

        loop = asyncio.get_running_loop()
        last_exc: ArxivSearchError | None = None
        for attempt in range(1 + self._rate_limit_retries):
            try:
                xml_text = await loop.run_in_executor(None, lambda: self._fetch(url))
                return self._parse_response(xml_text, max_results)
            except ArxivSearchError as exc:
                if exc.category == "rate_limit" and attempt < self._rate_limit_retries:
                    logger.warning(
                        "arXiv rate limit hit (attempt %d/%d) — backing off %.1fs",
                        attempt + 1,
                        1 + self._rate_limit_retries,
                        self._rate_limit_backoff,
                    )
                    await asyncio.sleep(self._rate_limit_backoff)
                    last_exc = exc
                    continue
                raise
            except Exception as exc:
                raise ArxivSearchError(str(exc), category="network") from exc

        # Exhausted retries — re-raise last rate-limit error
        assert last_exc is not None
        raise last_exc

    def _fetch(self, url: str) -> str:
        """Synchronous fetch — called in thread executor."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DeepSearch/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ArxivSearchError(
                    f"arXiv rate limit (429): {exc}", category="rate_limit"
                ) from exc
            raise ArxivSearchError(str(exc), category="network") from exc
        except Exception as exc:
            raise ArxivSearchError(str(exc), category="network") from exc

    def _parse_response(self, xml_text: str, max_results: int) -> List[ArxivResult]:
        """Parse arXiv Atom XML into normalized ArxivResult objects."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ArxivSearchError(f"XML parse error: {exc}", category="parse") from exc

        results: List[ArxivResult] = []
        for entry in root.findall(f"{{{_ATOM_NS}}}entry")[:max_results]:
            id_elem = entry.find(f"{{{_ATOM_NS}}}id")
            raw_url = (id_elem.text or "").strip() if id_elem is not None else ""
            # Strip version suffix (e.g. "...abs/2301.00001v2" → "...abs/2301.00001")
            if raw_url and "/" in raw_url:
                last_segment = raw_url.rsplit("/", 1)[-1]
                if last_segment and "v" in last_segment:
                    parts = last_segment.rsplit("v", 1)
                    if parts[1].isdigit():
                        raw_url = raw_url.rsplit("v", 1)[0]
            # Ensure https
            canonical_url = raw_url.replace("http://arxiv.org", "https://arxiv.org", 1)

            title_elem = entry.find(f"{{{_ATOM_NS}}}title")
            title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""

            summary_elem = entry.find(f"{{{_ATOM_NS}}}summary")
            summary = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""
            if len(summary) > _SNIPPET_MAX_CHARS:
                summary = summary[:_SNIPPET_MAX_CHARS - 3] + "..."

            if canonical_url:
                results.append(ArxivResult(url=canonical_url, title=title, snippet=summary))

        logger.debug("arXiv search returned %d results", len(results))
        return results
