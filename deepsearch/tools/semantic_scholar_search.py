"""Semantic Scholar native search client (B24.4).

Uses the Semantic Scholar Academic Graph API (no key required for basic use,
key improves rate limits). Returns publisher/DOI URLs — NOT semanticscholar.org
URLs — so domain scoring in source_quality.py applies to the actual publisher.

API: https://api.semanticscholar.org/graph/v1/paper/search
Docs: https://api.semanticscholar.org/api-docs/
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

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,externalIds,openAccessPdf,year"
_SNIPPET_MAX_CHARS = 500


class SemanticScholarError(RuntimeError):
    """Raised when Semantic Scholar API operations fail."""

    def __init__(self, message: str, *, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category


@dataclass
class SemanticScholarResult:
    """Normalized Semantic Scholar search result with publisher URL."""

    url: str        # Publisher/DOI URL — NOT semanticscholar.org
    title: str
    snippet: str    # Abstract excerpt
    year: Optional[int] = None

    def to_source(self) -> Dict[str, str]:
        """Return a dict suitable for AgentState source storage."""
        payload: Dict[str, str] = {"title": self.title or "Academic Paper"}
        if self.url:
            payload["url"] = self.url
        if self.snippet:
            payload["snippet"] = self.snippet
        return payload


@dataclass
class SemanticScholarClient:
    """Async Semantic Scholar API client.

    Returns publisher/DOI URLs so HIGH_AUTHORITY_DOMAINS scoring applies.
    Papers without resolvable publisher URLs are skipped.
    """

    api_key: Optional[str] = None
    timeout: float = 20.0

    async def search(self, query: str, *, max_results: int = 5) -> List[SemanticScholarResult]:
        """Search Semantic Scholar and return results with publisher URLs.

        Args:
            query: Research query.
            max_results: Maximum number of results to return.

        Returns:
            List of SemanticScholarResult with publisher/DOI URLs.

        Raises:
            SemanticScholarError: On network failure or API error.
        """
        query = (query or "").strip()
        if not query:
            raise SemanticScholarError("Query cannot be empty", category="validation")

        params = urllib.parse.urlencode({
            "query": query,
            "limit": str(min(max_results * 3, 50)),  # fetch extra since many lack DOIs
            "fields": _FIELDS,
        })
        url = f"{_BASE_URL}?{params}"

        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, lambda: self._fetch(url))
        except SemanticScholarError:
            raise
        except Exception as exc:
            raise SemanticScholarError(str(exc), category="network") from exc

        return self._parse_response(raw, max_results)

    def _fetch(self, url: str) -> str:
        """Synchronous fetch — called in thread executor."""
        headers: Dict[str, str] = {"User-Agent": "DeepSearch/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            raise SemanticScholarError(str(exc), category="network") from exc

    def _parse_response(self, raw: str, max_results: int) -> List[SemanticScholarResult]:
        """Parse Semantic Scholar JSON response into results with publisher URLs."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SemanticScholarError(f"JSON parse error: {exc}", category="parse") from exc

        papers = data.get("data", [])
        if not isinstance(papers, list):
            return []

        results: List[SemanticScholarResult] = []
        for paper in papers:
            if len(results) >= max_results:
                break

            publisher_url = self._extract_publisher_url(paper)
            if not publisher_url:
                # Skip papers without a resolvable publisher URL
                continue

            title = (paper.get("title") or "").strip()

            abstract = (paper.get("abstract") or "").strip()
            snippet = abstract[:_SNIPPET_MAX_CHARS - 3] + "..." if len(abstract) > _SNIPPET_MAX_CHARS else abstract

            year = paper.get("year")

            results.append(SemanticScholarResult(
                url=publisher_url,
                title=title,
                snippet=snippet,
                year=int(year) if year else None,
            ))

        logger.debug("Semantic Scholar search returned %d results with publisher URLs", len(results))
        return results

    @staticmethod
    def _extract_publisher_url(paper: Dict) -> Optional[str]:
        """Extract publisher/DOI URL from paper metadata.

        Priority:
        1. openAccessPdf.url (direct OA PDF — often arxiv.org, publisher site)
        2. DOI → https://doi.org/{doi}
        3. PubMed → https://pubmed.ncbi.nlm.nih.gov/{pmid}/

        Does NOT return semanticscholar.org URLs.
        """
        # Priority 1: open access PDF URL
        oa_pdf = paper.get("openAccessPdf") or {}
        oa_url = (oa_pdf.get("url") or "").strip()
        if oa_url and oa_url.startswith("http"):
            return oa_url

        # Priority 2: DOI
        external_ids = paper.get("externalIds") or {}
        doi = (external_ids.get("DOI") or "").strip()
        if doi:
            return f"https://doi.org/{doi}"

        # Priority 3: PubMed
        pmid = external_ids.get("PubMed") or external_ids.get("PMID")
        if pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return None
