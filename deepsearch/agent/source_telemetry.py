"""Source identity helpers for source-utilization telemetry."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


def source_telemetry_id(
    source: Mapping[str, Any],
    *,
    fallback_index: int | None = None,
    prompt_position: int | None = None,
) -> str:
    """Return a stable, human-auditable source identifier for telemetry."""

    explicit_id = _first_text(
        source.get("source_id"),
        source.get("id"),
        source.get("uid"),
        source.get("canonical_id"),
    )
    if explicit_id:
        base = f"source_id:{explicit_id}"
    else:
        url = _normalize_url(_first_text(source.get("url"), source.get("link")))
        if url:
            base = f"url:{url}"
        else:
            title = _normalize_title(_first_text(source.get("title"), source.get("name")))
            if title:
                base = f"title:{title}"
            elif fallback_index is not None:
                base = f"index:{fallback_index}"
            else:
                base = "unknown"

    if prompt_position is not None:
        return f"prompt:{prompt_position}|{base}"
    return base


def source_telemetry_ids(sources: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return de-duplicated telemetry IDs in source order."""

    ids: list[str] = []
    seen: set[str] = set()
    for index, source in enumerate(sources):
        source_id = source_telemetry_id(source, fallback_index=index)
        if source_id in seen:
            continue
        ids.append(source_id)
        seen.add(source_id)
    return ids


def prompt_source_telemetry_ids(
    prompt_source_map: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return source IDs keyed to their final prompt source-map positions."""

    ids: list[str] = []
    for entry in prompt_source_map:
        source = entry.get("source_dict")
        if not isinstance(source, Mapping):
            continue
        try:
            position = int(entry.get("position") or 0)
        except (TypeError, ValueError):
            position = 0
        ids.append(
            source_telemetry_id(
                source,
                fallback_index=entry.get("original_index")
                if isinstance(entry.get("original_index"), int)
                else None,
                prompt_position=position or None,
            )
        )
    return ids


def rank_sources_by_authority_and_relevance(
    sources: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return ``sources`` ordered best-first by composite authority/relevance.

    Composite score mirrors the admission-time weighting in
    :mod:`deepsearch.agent.working_set` (authority_score * 0.5 + relevance_score
    * 0.5). Sort is stable so equal-score sources keep their incoming order.
    Best-first ordering means that when ``apply_source_token_budget`` trims
    from the tail (``sources[:-1]``) the dropped records are the lowest-ranked
    ones, not arbitrary insertion-order tail slices.
    """

    def _composite(entry: tuple[int, Mapping[str, Any]]) -> tuple[float, int]:
        index, source = entry
        try:
            authority = float(source.get("authority_score") or 0.0)
        except (TypeError, ValueError):
            authority = 0.0
        try:
            relevance = float(source.get("relevance_score") or 0.0)
        except (TypeError, ValueError):
            relevance = 0.0
        composite = authority * 0.5 + relevance * 0.5
        # Negate so higher composite sorts first; tiebreak on insertion order
        # so the sort stays stable for unscored pools.
        return (-composite, index)

    indexed = list(enumerate(sources))
    indexed.sort(key=_composite)
    return [source for _index, source in indexed]


def rank_with_per_entity_floor(
    sources: Sequence[Mapping[str, Any]],
    *,
    required_entities: Sequence[str],
    per_entity_floor: int,
) -> list[Mapping[str, Any]]:
    """Rank globally, but reserve a per-entity floor at the front.

    S60.2 — closes the OODS-FIGMA-HOST-01 funnel-collapse defect where
    high-volume vendor docs for one declared entity (e.g. AWS Lambda) crowd
    out the global authority+relevance ranking and prevent vendor docs for
    other declared entities (e.g. ``vercel.com``, ``fly.io``) from reaching
    extraction-display. The S59.3 telemetry shows the right data was
    retrieved into the admitted pool, then *discarded* at the 232→49
    displayed-source step before reaching the LLM.

    Mechanic: when ``required_entities`` is non-empty (N entities) and
    ``per_entity_floor > 0``, take the top ``per_entity_floor`` sources for
    each declared entity (by composite rank within that entity's bound
    pool) and place them at the front of the returned ordering. The
    remainder fills by global rank. A source bound to multiple declared
    entities counts toward each entity's floor but appears only once in
    output.

    Architectural shape is the inverse of
    ``project_pre_filter_defect_pattern.md``: pre-filter defects compete
    with LLM judgment by withholding data; this is a FLOOR that protects
    LLM judgment by guaranteeing each declared entity's vendor docs reach
    the prompt — never trims sources, only reorders them so the downstream
    ``apply_source_token_budget`` tail-trim drops lowest-rank sources first
    rather than crowding out under-represented entities.

    When ``required_entities`` is empty, blank-only, or
    ``per_entity_floor <= 0``, returns the same ordering as
    :func:`rank_sources_by_authority_and_relevance` (no behavior change).
    """

    ranked = rank_sources_by_authority_and_relevance(sources)

    if per_entity_floor <= 0:
        return ranked

    declared_keys: list[str] = []
    seen_keys: set[str] = set()
    for entity in required_entities:
        cleaned = str(entity or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        declared_keys.append(key)

    if not declared_keys:
        return ranked

    reserved_indices: list[int] = []
    reserved_set: set[int] = set()

    for entity_key in declared_keys:
        per_entity_added = 0
        for index, source in enumerate(ranked):
            if per_entity_added >= per_entity_floor:
                break
            matches = source.get("contract_entity_matches") or []
            if not _source_binds_entity(matches, entity_key):
                continue
            per_entity_added += 1
            if index in reserved_set:
                # Multi-bound source — counted toward this entity's floor
                # but already in the reserved list (preserves global rank
                # order across entities).
                continue
            reserved_set.add(index)
            reserved_indices.append(index)

    if not reserved_indices:
        return ranked

    reserved_indices.sort()
    reserved = [ranked[i] for i in reserved_indices]
    remainder = [source for index, source in enumerate(ranked) if index not in reserved_set]
    return reserved + remainder


def _source_binds_entity(matches: Any, entity_key: str) -> bool:
    if not isinstance(matches, (list, tuple)):
        return False
    return any(str(match or "").strip().lower() == entity_key for match in matches)


def select_displayed_research_sources(
    visible_sources: Sequence[Mapping[str, Any]],
    *,
    depth_config: Any = None,  # kept for signature compatibility; unused
    compact_mode: bool = False,
    max_display: int | None = None,
) -> list[Mapping[str, Any]]:
    """Return the source records actually rendered into the extraction prompt.

    On the normal path the full ranked pool is returned and the caller
    (``apply_source_token_budget``) handles size enforcement against
    ``max_tokens_per_loop``. ``compact_mode`` keeps the deliberate ≤ 4
    degraded-mode tuning. ``max_display`` is honored only when a positive
    int is passed explicitly — Sprint 58 removed all in-tree callers that
    used to pin 10/20 here, but the parameter remains for callers that need
    to override (e.g., tests, ad-hoc tooling).
    """

    del depth_config  # signature kept for back-compat; tier sizing belongs to token budget

    ranked = rank_sources_by_authority_and_relevance(visible_sources)

    if compact_mode:
        cap = 4
        if max_display is not None and int(max_display) > 0:
            cap = min(cap, int(max_display))
        return ranked[:cap]

    if max_display is not None and int(max_display) > 0:
        return ranked[: int(max_display)]

    return ranked


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = re.sub(r"/+$", "", parts.path or "")
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


__all__ = [
    "prompt_source_telemetry_ids",
    "rank_sources_by_authority_and_relevance",
    "rank_with_per_entity_floor",
    "select_displayed_research_sources",
    "source_telemetry_id",
    "source_telemetry_ids",
]
