"""URL parsing helpers carried forward from the pre-S75 ``source_quality.py``.

S75 m01 Phase D (Option A refactor): split the two consumers — ``resolve_source_url``
(used by ``report/templates.py`` + ``report/evidence.py``) and ``extract_domain``
(used by ``agent/slot_coverage.py``) — out of ``source_quality.py`` so the rest of
that module's LLM-as-judge admission stack could be deleted alongside the bypassed
LangGraph nodes. No I/O, no external state; pure URL parsing + Google-grounding
redirect resolution. S98 m02 adds ``arxiv_identity``, the paper identity shared by
the collection-membership boundaries (liveness classification and evidence
dependencies).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# Google grounding redirect URL prefix
_GOOGLE_REDIRECT_PREFIX = "vertexaisearch.cloud.google.com"

# Pattern matching a domain-like string (e.g. "example.com", "docs.python.org")
_DOMAIN_PATTERN = re.compile(
    r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?"
    r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)+$"
)


def _parse_domain_from_url(url: str) -> str:
    """Extract domain from a standard URL. Internal helper."""
    if not url:
        return ""
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if ":" in domain:
            domain = domain.split(":")[0]
        return domain
    except Exception:
        return ""


def _looks_like_domain(text: str) -> bool:
    """Check if a string looks like a domain name."""
    if not text:
        return False
    cleaned = text.strip().lower()
    # Remove trailing dots or slashes
    cleaned = cleaned.rstrip("/.")
    return bool(_DOMAIN_PATTERN.match(cleaned))


def _is_google_redirect(url: str) -> bool:
    """Check if URL is a Google grounding redirect."""
    domain = _parse_domain_from_url(url)
    return domain == _GOOGLE_REDIRECT_PREFIX


def extract_domain(url: str, *, source: Optional[Dict[str, Any]] = None) -> str:
    """Extract the actual domain from a URL, resolving redirects.

    For Google grounding redirect URLs (vertexaisearch.cloud.google.com),
    the actual destination domain is extracted from:
    1. source["display_url"] if present
    2. source["title"] if it looks like a domain name

    For all other URLs, standard URL parsing is used. If the source dict
    provides a display_url field, that takes priority over the raw URL.

    Args:
        url: The URL to extract domain from.
        source: Optional source dict with display_url/title metadata.

    Returns:
        Lowercase domain string, or empty string if extraction fails.
    """
    # Priority 1: display_url from source dict (Exa, SerpAPI provide this)
    if source:
        display_url = source.get("display_url", "")
        if display_url:
            resolved = _parse_domain_from_url(display_url)
            if resolved:
                return resolved

    # Priority 2: If Google redirect URL, try to resolve from title
    if _is_google_redirect(url) and source:
        title = source.get("title", "")
        if _looks_like_domain(title):
            return title.strip().lower().rstrip("/.")
        # Title might be "Page Title - domain.com" or similar
        # Try the last word after common separators
        for sep in [" - ", " | ", " – ", " — "]:
            if sep in title:
                candidate = title.rsplit(sep, 1)[-1].strip()
                if _looks_like_domain(candidate):
                    return candidate.lower().rstrip("/.")

    # Priority 3: Standard URL parsing
    if _is_google_redirect(url):
        # Can't resolve — return empty rather than the redirect domain
        return ""

    return _parse_domain_from_url(url)


def resolve_source_url(url: str, source: Optional[Dict[str, Any]] = None) -> str:
    """Resolve a source URL to its actual destination, bypassing redirects.

    For Google grounding redirect URLs (vertexaisearch.cloud.google.com),
    returns a usable URL from:
    1. source["display_url"] if present and valid
    2. A constructed https:// URL from the domain extracted from source title
    3. The original URL as fallback

    For non-redirect URLs, returns the original URL unchanged.

    Args:
        url: The raw URL to resolve.
        source: Optional source dict with display_url/title metadata.

    Returns:
        Resolved URL string.
    """
    if not _is_google_redirect(url):
        return url

    if source:
        # Priority 1: display_url from source dict
        display_url = source.get("display_url", "")
        if display_url:
            # Ensure it has a scheme
            if display_url.startswith("http"):
                return display_url
            return f"https://{display_url}"

        # Priority 2: Construct URL from domain extracted via title
        domain = extract_domain(url, source=source)
        if domain:
            return f"https://{domain}"

    # Fallback: return original (better than nothing)
    return url


# arXiv paper IDs: new-style YYMM.NNNN(N), or old-style archive[.SUBJECT]/YYMMNNN
# whose subject class arXiv does not need to resolve the paper.
_ARXIV_ID = (
    r"(?P<id>\d{4}\.\d{4,5}"
    r"|(?P<archive>[a-z]+(?:-[a-z]+)*)(?:\.[a-z]{2})?/(?P<number>\d{7}))"
)
_ARXIV_PAGE_RE = re.compile(r"/(?:abs|pdf|html)/" + _ARXIV_ID + r"(?:v\d+)?(?:\.pdf)?/?", re.I)
# DataCite prefix 10.48550 has exactly one client, arXiv (verified 2026-10-28).
_ARXIV_DOI_RE = re.compile(r"/10\.48550/arxiv\." + _ARXIV_ID + r"(?:v\d+)?/?", re.I)
_ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})
_DOI_HOSTS = frozenset({"doi.org", "dx.doi.org"})


def arxiv_identity(url: str) -> Optional[str]:
    """Return ``arxiv:<id>`` without its version for an arXiv paper URL, else None.

    Recognizes the abs, pdf and html pages and the arXiv DOI
    (``doi.org/10.48550/arXiv.<id>``, any case). This identity answers collection
    membership — "did the run collect this paper" — and, since S100 m02, whether
    one References entry's links name one paper. Readable bodies, probe caching
    and the References audit's body mapping keep exact URLs, because an abstract
    page is not the full text.
    """
    if not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if host in _ARXIV_HOSTS:
        match = _ARXIV_PAGE_RE.fullmatch(parsed.path)
    elif host in _DOI_HOSTS:
        match = _ARXIV_DOI_RE.fullmatch(parsed.path)
    else:
        return None
    if not match:
        return None
    if match.group("archive"):
        return f"arxiv:{match.group('archive').lower()}/{match.group('number')}"
    return f"arxiv:{match.group('id')}"


__all__ = ["arxiv_identity", "extract_domain", "resolve_source_url"]
