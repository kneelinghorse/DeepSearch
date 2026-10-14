"""Markdown templating helpers used by the report generator."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from deepsearch.agent.url_utils import resolve_source_url
from deepsearch.mission.title_utils import normalize_mission_title
from .embeddings import SemanticMatcher, USE_SEMANTIC_MATCHING, DEFAULT_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def build_report_header(
    mission_id: str,
    title: str,
    *,
    loop_count: int,
    source_count: int,
    generated_at: datetime | None = None,
    research_depth: str | None = None,
) -> str:
    """Return the report title line.

    The truthful at-a-glance run-integrity surface lives in the
    ``_render_report_metadata_preamble`` block emitted by
    ``codeagent_finalize_node`` (PD22). Legacy ``loop_count`` /
    ``source_count`` / ``research_depth`` arguments are accepted for
    backward-compatible call sites but no longer rendered — the
    post-S64 path counts steps, not loops, and the legacy counters
    read 0 because the LangGraph stages they tracked are bypassed.
    """
    safe_title = normalize_mission_title(title, mission_id) or mission_id
    return f"# Research Report: {mission_id} - {safe_title}\n"


def build_findings_section(
    findings: Sequence[str],
    *,
    research_phases: Sequence[Mapping[str, object]] | None = None,
    sources: Sequence[Mapping[str, object]] | None = None,
    use_semantic_matching: bool = USE_SEMANTIC_MATCHING,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> str:
    """Format the research findings block, optionally grouped by phase.

    Args:
        findings: List of finding strings
        research_phases: Optional phase metadata for grouping
        sources: Optional list of sources for inline citation injection
        use_semantic_matching: Use embedding-based matching (default from env)
        similarity_threshold: Minimum similarity for citation (default 0.7)

    Returns:
        Formatted markdown findings section with inline citations
    """
    clean_findings = [item.strip() for item in findings if str(item).strip()]
    if not clean_findings and not research_phases:
        return "## Research Findings\n\nNo findings were generated during research.\n"

    sources_list = list(sources) if sources else []
    cited_urls: Set[str] = set()

    # Initialize semantic matcher if requested and sources available
    matcher: Optional[SemanticMatcher] = None
    if use_semantic_matching and sources_list:
        matcher = SemanticMatcher(similarity_threshold=similarity_threshold)
        embedded_count = matcher.precompute_sources(sources_list)
        if embedded_count > 0:
            logger.debug(
                "Semantic matching enabled: %d/%d sources embedded",
                embedded_count,
                len(sources_list),
            )
        else:
            logger.debug("Semantic matching unavailable, using keyword fallback")
            matcher = None

    lines: List[str] = ["## Research Findings", ""]
    if research_phases:
        remaining = list(clean_findings)
        for phase in _normalise_research_phases(research_phases):
            phase_matches, remaining = _partition_findings(remaining, phase["keywords"])
            lines.append(f"### {phase['title']}")
            if phase["description"]:
                lines.append(f"{phase['description']}\n")
            if not phase_matches:
                lines.append("No findings captured for this phase.\n")
            else:
                for idx, match in enumerate(phase_matches, 1):
                    cited_match = _inject_citation(
                        match, sources_list, cited_urls, matcher=matcher
                    )
                    lines.append(f"{idx}. {cited_match}")
                lines.append("")
        if remaining:
            lines.append("### Additional Insights")
            for idx, match in enumerate(remaining, 1):
                cited_match = _inject_citation(
                    match, sources_list, cited_urls, matcher=matcher
                )
                lines.append(f"{idx}. {cited_match}")
            lines.append("")
    else:
        lines.append("### Key Insights")
        lines.append("")
        for idx, finding in enumerate(clean_findings, 1):
            cited_finding = _inject_citation(
                finding, sources_list, cited_urls, matcher=matcher
            )
            lines.append(f"{idx}. {cited_finding}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_findings_section_optimized(
    findings: Sequence[str],
    sources: Sequence[Mapping[str, object]],
    *,
    use_semantic: bool = True,
    use_llm_assist: bool = False,
    allow_multi_citation: bool = True,
    target_rate: float = 0.8,
    llm_client: Optional[object] = None,
) -> Tuple[str, float]:
    """Build findings with optimized citation rate targeting 80%+.

    Uses a three-phase strategy:
    1. Semantic matching for all findings (primary)
    2. LLM-assist for remaining uncited (optional, requires llm_client)
    3. Multi-citation for complex findings (optional)

    Args:
        findings: List of finding strings
        sources: List of source dicts with title, url, snippet
        use_semantic: Enable semantic matching (default True)
        use_llm_assist: Enable LLM suggestions for uncited text (default False)
        allow_multi_citation: Allow multiple citations per finding (default True)
        target_rate: Target citation rate (default 0.8)
        llm_client: Optional LLM client for suggestions

    Returns:
        Tuple of (formatted markdown section, actual citation rate)
    """
    from .citation_assistant import batch_suggest_citations

    clean_findings = [item.strip() for item in findings if str(item).strip()]
    if not clean_findings:
        return (
            "## Research Findings\n\nNo findings were generated during research.\n",
            0.0,
        )

    sources_list = list(sources) if sources else []
    if not sources_list:
        lines = ["## Research Findings", "", "### Key Insights", ""]
        for idx, finding in enumerate(clean_findings, 1):
            lines.append(f"{idx}. {finding}")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n", 0.0

    # Phase 1: Initialize semantic matcher
    matcher: Optional[SemanticMatcher] = None
    if use_semantic:
        matcher = SemanticMatcher(similarity_threshold=0.65)  # Slightly lower for better coverage
        embedded_count = matcher.precompute_sources(sources_list)
        if embedded_count > 0:
            logger.debug("Semantic matching: %d sources embedded", embedded_count)
        else:
            matcher = None
            logger.debug("Semantic matching unavailable, using keyword fallback")

    cited_urls: Set[str] = set()
    cited_findings: List[str] = []
    uncited_indices: List[int] = []
    citation_count = 0

    # Phase 2: Apply semantic/keyword matching
    for i, finding in enumerate(clean_findings):
        cited = _inject_citation(
            finding,
            sources_list,
            cited_urls,
            matcher=matcher,
            allow_multi_citation=allow_multi_citation,
        )

        # Check if citation was added (finding changed)
        if cited != finding and cited != finding + ".":
            cited_findings.append(cited)
            citation_count += 1
        else:
            cited_findings.append(finding)
            uncited_indices.append(i)

    # Phase 3: LLM-assist for uncited findings
    if use_llm_assist and llm_client and uncited_indices:
        uncited_texts = [clean_findings[i] for i in uncited_indices]
        unused_sources = [s for s in sources_list if _stringify(s.get("url")) not in cited_urls]

        logger.debug(
            "LLM-assisting %d uncited findings with %d unused sources",
            len(uncited_texts),
            len(unused_sources),
        )

        if unused_sources:
            suggestions = batch_suggest_citations(
                uncited_texts, unused_sources, llm_client
            )

            for local_idx, (source, conf) in suggestions.items():
                global_idx = uncited_indices[local_idx]
                if global_idx < len(cited_findings):
                    cited_findings[global_idx] = _inject_citation_to_item(
                        clean_findings[global_idx], source
                    )
                    url = _stringify(source.get("url"))
                    if url:
                        cited_urls.add(url)
                    citation_count += 1

    # Calculate actual citation rate
    actual_rate = citation_count / len(clean_findings) if clean_findings else 0.0

    # Format output
    lines = ["## Research Findings", "", "### Key Insights", ""]
    for idx, finding in enumerate(cited_findings, 1):
        lines.append(f"{idx}. {finding}")
    lines.append("")

    # Add citation rate metadata as HTML comment
    lines.append(f"<!-- citation_rate: {actual_rate:.2%} ({citation_count}/{len(clean_findings)}) -->")

    markdown = "\n".join(lines).rstrip() + "\n"
    return markdown, actual_rate


def build_sources_section(sources: Sequence[Mapping[str, object]]) -> str:
    """Return a numbered list of sources with clickable citations."""

    normalised = [_normalise_source(entry) for entry in sources if entry]
    if not normalised:
        return "## Sources Consulted\n\nNo sources were recorded for this mission.\n"

    lines = ["## Sources Consulted", ""]
    for idx, source in enumerate(normalised, 1):
        entry = f"[{source['title']}]({source['url']})" if source["url"] else source["title"]
        lines.append(f"{idx}. {entry}  ")
        lines.append(f"   Retrieved: {source['retrieved']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_related_research_section(graph_context: Mapping[str, object]) -> str:
    """Return a related research section derived from graph context."""

    results = (
        graph_context.get("results")
        or graph_context.get("matches")
        or graph_context.get("related_research")
        or []
    )
    if not isinstance(results, Iterable):
        return ""
    entries = [item for item in results if isinstance(item, Mapping)]
    if not entries:
        return ""

    lines = ["## Related Research", "", "Graph-derived related missions:"]
    count = 0
    for entry in entries:
        mission_id = _stringify(entry.get("mission_id") or entry.get("missionId"))
        title = _stringify(entry.get("title") or entry.get("label"))
        score = (
            entry.get("rrf_score")
            or entry.get("rrfScore")
            or entry.get("similarity_score")
            or entry.get("similarity")
            or entry.get("score")
        )
        label = mission_id or title or "Related mission"
        if mission_id and title:
            label = f"{mission_id} — {title}"
        score_text = "n/a"
        if score is not None:
            try:
                score_text = f"{float(score):.2f}"
            except (TypeError, ValueError):
                score_text = str(score)
        lines.append(f"- {label} (score: {score_text})")
        count += 1
        if count >= 8:
            break
    if count == 0:
        return ""
    lines.extend(["", "_Source: PEDR graph search_"])
    return "\n".join(lines).strip() + "\n"


def build_mission_context_section(
    objectives: Sequence[str],
    *,
    context: Mapping[str, object],
) -> str:
    """Append mission objectives and background context."""

    lines = ["---", "", "**Mission Objectives:**"]
    clean_objectives = [item.strip() for item in objectives if str(item).strip()]
    if not clean_objectives:
        lines.append("- Not specified")
    else:
        for objective in clean_objectives:
            lines.append(f"- {objective}")
    lines.append("")

    background = _extract_background(context)
    if background:
        lines.append("**Research Context:**  ")
        lines.append(background)
    else:
        summary = _stringify(context.get("summary"))
        if summary:
            lines.append("**Research Context:**  ")
            lines.append(summary)
    return "\n".join(lines).strip() + "\n"


def _normalise_research_phases(
    data: Sequence[Mapping[str, object]]
) -> List[dict[str, List[str] | str]]:
    phases: List[dict[str, List[str] | str]] = []
    for entry in data:
        title = _stringify(entry.get("title")) or _stringify(entry.get("name")) or "Research Phase"
        description = _stringify(
            entry.get("summary")
            or entry.get("description")
            or entry.get("details")
            or entry.get("objective")
        )
        keywords: List[str] = []
        raw_keywords = entry.get("keywords") or entry.get("tasks") or []
        if isinstance(raw_keywords, str):
            raw_keywords = [raw_keywords]
        if isinstance(raw_keywords, Iterable):
            keywords.extend(str(item).strip().lower() for item in raw_keywords if str(item).strip())
        keywords.extend(token.strip().lower() for token in (title.lower(),))
        phases.append({"title": title, "description": description, "keywords": keywords})
    return phases


def _partition_findings(
    findings: List[str],
    keywords: Sequence[str],
) -> tuple[List[str], List[str]]:
    matches: List[str] = []
    remaining: List[str] = []
    lowered_keywords = [kw for kw in keywords if kw]
    for item in findings:
        if any(kw in item.lower() for kw in lowered_keywords if kw):
            matches.append(item)
        else:
            remaining.append(item)
    return matches, remaining


def _normalise_source(payload: Mapping[str, object]) -> dict[str, str]:
    raw_url = _stringify(payload.get("url"))
    # Resolve Google grounding redirect URLs to actual destinations
    resolved_url = resolve_source_url(raw_url, dict(payload)) if raw_url else ""
    return {
        "title": _stringify(payload.get("title")) or "Untitled Source",
        "url": resolved_url,
        "retrieved": _stringify(
            payload.get("retrieved_at")
            or payload.get("retrieved")
            or payload.get("accessed_at")
            or payload.get("timestamp")
        )
        or "Unknown",
    }


def _extract_background(context: Mapping[str, object]) -> str:
    background = context.get("background") or context.get("details")
    return _stringify(background)


def _stringify(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        value = datetime.now(tz=timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def _find_best_matching_source(
    finding: str,
    sources: Sequence[Mapping[str, object]],
    cited_urls: Set[str],
) -> Optional[Mapping[str, object]]:
    """Find the source most relevant to a finding.

    Tie-breaks on authority_score (B35.3): when two sources match a finding
    equally well by keyword overlap, the source with the higher authority
    (set by B35.1's admit_sources path) is selected as the citation anchor.
    Authority is a TIE-BREAKER, not a primary signal — keyword relevance
    still has to clear the same minimum bar (>= 2) to be cited at all.

    Args:
        finding: The finding text to match against
        sources: List of source dicts with title, url, snippet, authority_score
        cited_urls: Set of already-cited URLs to avoid repetition

    Returns:
        Best matching source dict or None if no good match found
    """
    finding_lower = finding.lower()
    finding_words = set(_extract_keywords(finding_lower))

    candidates: List[Tuple[int, float, int, Mapping[str, object]]] = []

    for index, source in enumerate(sources):
        url = _stringify(source.get("url"))
        if url in cited_urls:
            continue

        score = 0

        title = _stringify(source.get("title")).lower()
        title_words = set(_extract_keywords(title))
        common_title = title_words & finding_words
        score += len(common_title) * 2

        snippet = _stringify(source.get("snippet")).lower()
        snippet_words = set(_extract_keywords(snippet))
        common_snippet = snippet_words & finding_words
        score += len(common_snippet)

        if score < 2:
            continue

        candidates.append((score, _source_authority_score(source), index, source))

    if not candidates:
        return None

    # Sort by (keyword score desc, authority desc, original position asc).
    # Position last so ties are deterministic and stable.
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][3]


def _source_authority_score(source: Mapping[str, object]) -> float:
    """Read authority_score from a source dict; default 0.0 when absent.

    Sources admitted via deepsearch.agent.working_set.admit_sources carry
    authority_score (B35.1). Sources from legacy paths or test fixtures
    without the field default to 0.0, which means they will only be picked
    when no scored source ties them on keyword relevance.
    """

    raw = source.get("authority_score")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text, filtering stopwords."""
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "it", "its", "as", "if", "then", "than",
        "so", "no", "not", "only", "just", "also", "more", "most", "some",
        "any", "all", "each", "every", "both", "few", "many", "much", "such",
        "very", "too", "quite", "rather", "about", "into", "over", "under",
    }
    words = []
    for word in text.split():
        clean = "".join(c for c in word if c.isalnum())
        if clean and len(clean) > 2 and clean not in stopwords:
            words.append(clean)
    return words


def _inject_citation_to_item(
    item: str,
    source: Mapping[str, object],
) -> str:
    """Inject citation into any text item (paragraph or list item).

    Handles:
    - Bullet points (-, *, •)
    - Numbered lists (1., 2., etc.)
    - Standard paragraphs

    Args:
        item: The text item to annotate
        source: Source dict with title/url fields

    Returns:
        Item with citation appended
    """
    title = _stringify(source.get("title")) or "Source"
    url = _stringify(source.get("url"))

    if not url:
        return item

    display_title = title[:50] + "..." if len(title) > 50 else title
    citation = f"[{display_title}]({url})"

    text = item.rstrip()

    # Handle bullet list items (-, *, •)
    if text.startswith(("-", "*", "•")):
        prefix = text[0]
        content = text[1:].lstrip()
        if content.endswith((".", "!", "?")):
            return f"{prefix} {content} {citation}"
        else:
            return f"{prefix} {content}. {citation}"

    # Handle numbered lists (1., 2., etc.)
    numbered = re.match(r'^(\d+\.)\s*(.+)$', text)
    if numbered:
        num, content = numbered.groups()
        if content.endswith((".", "!", "?")):
            return f"{num} {content} {citation}"
        else:
            return f"{num} {content}. {citation}"

    # Standard paragraph
    if text.endswith((".", "!", "?")):
        return f"{text} {citation}"
    else:
        return f"{text}. {citation}"


def _inject_multiple_citations(
    finding: str,
    matched_sources: List[Tuple[Mapping[str, object], float]],
    max_citations: int = 3,
) -> str:
    """Inject multiple relevant citations into a finding.

    Args:
        finding: The finding text
        matched_sources: List of (source, score) tuples
        max_citations: Maximum citations to inject (default 3)

    Returns:
        Finding with multiple citations appended
    """
    if not matched_sources:
        return finding

    # Sort by score and take top N
    sorted_sources = sorted(matched_sources, key=lambda x: x[1], reverse=True)
    top_sources = sorted_sources[:max_citations]

    # Build citation string
    citations = []
    for source, score in top_sources:
        title = _stringify(source.get("title")) or "Source"
        url = _stringify(source.get("url"))
        if url:
            display_title = title[:30] + "..." if len(title) > 30 else title
            citations.append(f"[{display_title}]({url})")

    if not citations:
        return finding

    # Use parentheses for grouped multi-citation format
    citation_str = ", ".join(citations)

    text = finding.rstrip()

    # Handle list items
    if text.startswith(("-", "*", "•")):
        prefix = text[0]
        content = text[1:].lstrip()
        if content.endswith((".", "!", "?")):
            return f"{prefix} {content} ({citation_str})"
        else:
            return f"{prefix} {content}. ({citation_str})"

    numbered = re.match(r'^(\d+\.)\s*(.+)$', text)
    if numbered:
        num, content = numbered.groups()
        if content.endswith((".", "!", "?")):
            return f"{num} {content} ({citation_str})"
        else:
            return f"{num} {content}. ({citation_str})"

    # Standard paragraph
    if text.endswith((".", "!", "?")):
        return f"{text} ({citation_str})"
    else:
        return f"{text}. ({citation_str})"


def _inject_citation(
    finding: str,
    sources: Sequence[Mapping[str, object]],
    cited_urls: Set[str],
    matcher: Optional[SemanticMatcher] = None,
    allow_multi_citation: bool = False,
) -> str:
    """Inject inline citation into a finding.

    Args:
        finding: The finding text
        sources: List of {title, url, snippet} dicts
        cited_urls: Set of already-cited URLs (updated in-place)
        matcher: Optional SemanticMatcher for embedding-based matching
        allow_multi_citation: If True, inject multiple citations for complex findings

    Returns:
        Finding with inline [Title](url) citation appended if match found
    """
    if not sources:
        return finding

    # Use semantic matcher if available, otherwise keyword fallback
    if matcher:
        best_match, score = matcher.find_best_match(finding, sources, cited_urls)
        if best_match:
            logger.debug("Semantic match (score=%.2f): %s", score, finding[:50])

        # For multi-citation mode, find additional matches
        if allow_multi_citation and best_match:
            matched = [(best_match, score)]
            cited_urls.add(_stringify(best_match.get("url")))

            # Find up to 2 more matches
            for _ in range(2):
                additional, add_score = matcher.find_best_match(finding, sources, cited_urls)
                if additional and add_score >= 0.6:
                    matched.append((additional, add_score))
                    cited_urls.add(_stringify(additional.get("url")))
                else:
                    break

            if len(matched) > 1:
                return _inject_multiple_citations(finding, matched)
    else:
        best_match = _find_best_matching_source(finding, sources, cited_urls)

    if not best_match:
        return finding

    url = _stringify(best_match.get("url"))
    if not url:
        return finding

    cited_urls.add(url)

    return _inject_citation_to_item(finding, best_match)


__all__ = [
    "build_report_header",
    "build_findings_section",
    "build_findings_section_optimized",
    "build_sources_section",
    "build_related_research_section",
    "build_mission_context_section",
    "_inject_citation",
    "_inject_citation_to_item",
    "_inject_multiple_citations",
    "_find_best_matching_source",
    "_source_authority_score",
    "_extract_keywords",
]
