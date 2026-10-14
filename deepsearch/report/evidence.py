"""TraceLab Evidence generation from AgentState sources.

This module transforms the flat sources_found list from AgentState into
TraceLab-compatible Evidence objects for ingestion and auto-linking.

The Evidence objects produced here are compatible with TraceLab's
MissionProtocolComplete schema and enable evidence auto-linking during upload.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from deepsearch.agent.url_utils import resolve_source_url
from deepsearch.agent.state import AgentState


class Evidence(BaseModel):
    """TraceLab-compatible Evidence object.

    This model matches the TraceLab Evidence schema for mission protocol
    ingestion. Evidence auto-linking uses the summary field to match against
    document chunks via difflib.SequenceMatcher.
    """

    evidence_id: str = Field(description="Unique evidence identifier (e.g., EV-001)")
    source: str = Field(description="Source URL or identifier")
    summary: str = Field(description="Description of what this source provides")
    chunk_id: Optional[str] = Field(
        default=None,
        description="TraceLab chunk ID (auto-populated by TraceLab linking)",
    )
    insight_id: Optional[str] = Field(
        default=None,
        description="TraceLab insight ID (auto-populated by TraceLab linking)",
    )
    source_type: Optional[str] = Field(
        default=None,
        description="Type of source (e.g., web_search, interview, document)",
    )
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 to 1.0",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and filtering",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(exclude_none=True)


def generate_evidence_list(state: AgentState) -> List[Evidence]:
    """Transform sources_found into TraceLab Evidence objects.

    This function converts the flat sources_found list from AgentState into
    structured Evidence objects suitable for TraceLab ingestion. Each source
    maps to exactly one Evidence item.

    Args:
        state: AgentState containing sources_found list. Each source dict
               should have: url, title, snippet, retrieved_at (from nodes.py).

    Returns:
        List of Evidence objects ready for TraceLab ingestion.
        Empty list if no sources found (not an error).

    Example:
        >>> state = {"sources_found": [
        ...     {"url": "https://example.com", "title": "Article", "snippet": "..."}
        ... ]}
        >>> evidence = generate_evidence_list(state)
        >>> len(evidence)
        1
        >>> evidence[0].evidence_id
        'EV-001'
    """
    # S83 m06 (PD83-b) CARVE-OUT: this is the ONE retained reader of the
    # S78-stripped sources_found. generate_evidence_list is a tested generic
    # Evidence-builder; rather than gut it (+ its ~21-test suite) for a read
    # that is dead-in-production (no writer -> always []), it is kept as-is.
    # The never-reappears guards (test_s78 / test_s83_m06) exclude this file.
    sources = state.get("sources_found") or []
    if not sources:
        return []

    evidence_list: List[Evidence] = []
    seen_urls: Set[str] = set()
    evidence_index = 1

    for source in sources:
        if not isinstance(source, dict):
            continue

        raw_url = source.get("url", "")
        # Resolve Google grounding redirect URLs to actual destinations
        url = resolve_source_url(raw_url, source) if raw_url else raw_url

        # Deduplicate by URL - keep first occurrence
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        # Generate unique evidence_id
        evidence_id = f"EV-{evidence_index:03d}"
        evidence_index += 1

        # Build summary from snippet or title
        summary = _build_summary(source)

        # Determine source identifier (URL preferred, fallback to title)
        source_identifier = url or source.get("title", "Unknown source")

        # Create Evidence object
        evidence = Evidence(
            evidence_id=evidence_id,
            source=source_identifier,
            summary=summary,
            source_type="web_search",
            tags=_extract_tags(source),
        )
        evidence_list.append(evidence)

    return evidence_list


def _build_summary(source: Dict[str, Any]) -> str:
    """Build evidence summary from source data.

    Priority:
    1. Use snippet if present and non-trivial (>= 10 chars)
    2. Fall back to title with "Source: " prefix
    3. Default to "Web search result" if nothing available

    Args:
        source: Source dictionary with snippet, title fields.

    Returns:
        Non-empty summary string (TraceLab validation requires non-empty).
    """
    snippet = source.get("snippet", "")
    if snippet and len(snippet.strip()) >= 10:
        # Truncate long snippets to 500 chars for TraceLab compatibility
        return snippet[:500].strip()

    title = source.get("title", "")
    if title:
        return f"Source: {title}"

    return "Web search result"


def _extract_tags(source: Dict[str, Any]) -> List[str]:
    """Extract tags from source metadata.

    Generates tags based on:
    - Always includes "research" tag
    - Extracts domain from URL as a tag (without www. prefix)

    Args:
        source: Source dictionary with url field.

    Returns:
        List of tag strings.
    """
    tags = ["research"]

    # Add domain as tag (resolve redirects first)
    raw_url = source.get("url", "")
    url = resolve_source_url(raw_url, source) if raw_url else raw_url
    if url:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain:
            # Remove www. prefix for cleaner tags
            clean_domain = domain.replace("www.", "")
            if clean_domain:
                tags.append(clean_domain)

    for slot_id in source.get("target_slots") or []:
        text = str(slot_id).strip()
        if text and f"slot:{text}" not in tags:
            tags.append(f"slot:{text}")

    return tags


def deduplicate_by_url(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate sources by URL, keeping first occurrence.

    Utility function for pre-processing sources before evidence generation.

    Args:
        sources: List of source dictionaries with url field.

    Returns:
        Deduplicated list preserving original order.
    """
    seen: Set[str] = set()
    result: List[Dict[str, Any]] = []

    for source in sources:
        url = source.get("url", "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        result.append(source)

    return result


__all__ = [
    "Evidence",
    "generate_evidence_list",
    "deduplicate_by_url",
]
