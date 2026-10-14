"""Refinement helpers that guide multi-loop research strategies."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from deepsearch.mission.title_utils import normalize_mission_title

LOW_SCORE_THRESHOLD = 0.8
HIGH_PRIORITY_THRESHOLD = 0.5
MAX_TARGETED_QUERIES = 5
MAX_QUERY_LENGTH = 256
DEFAULT_STUCK_WINDOW = 3
MIN_IMPROVEMENT = 0.05
AUTHOR_YEAR_PATTERN = re.compile(r"(?:[A-Z][a-z]+\s+)+et\s+al\.?\s+\d{4}")
ACRONYM_PATTERN = re.compile(r"\b[A-Z0-9]{3,}(?:-[A-Z0-9]+)*\b")
CAMEL_CASE_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,}[a-z0-9]*|[A-Z][a-z0-9]+)(?:[A-Z][a-z0-9]+)+\b"
)
QUOTED_PHRASE_PATTERN = re.compile(r'"([^"\n]{3,})"')
PROPER_NOUN_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|of|the|and|for|Is)){1,5})\b"
)
ENUMERATED_ENTITY_INTRO_PATTERN = re.compile(
    r"\b(?:compare|assess|evaluate|examine|analyze|contrast|benchmark|survey)\b"
    r"(?P<rest>[^.!?;\n]+)",
    re.IGNORECASE,
)

_QUERY_INTENT_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwhen to use\b", re.IGNORECASE), "use cases"),
    (re.compile(r"\bgotcha(?:s)?\b", re.IGNORECASE), "operational gotchas"),
    (re.compile(r"\bbriefly describe\b", re.IGNORECASE), "overview"),
    (re.compile(r"\bdescribe\b", re.IGNORECASE), "overview"),
    (re.compile(r"\bcompare\b", re.IGNORECASE), "comparison"),
    (re.compile(r"\bevidence gap(?:s)?\b", re.IGNORECASE), "evidence gaps"),
    (re.compile(r"\barchitecture(?:s)?\b", re.IGNORECASE), "architecture"),
)
_ENTITY_TRUNCATION_WORDS = frozenset(
    {
        "in",
        "for",
        "with",
        "on",
        "at",
        "across",
        "among",
    }
)
_ENTITY_TRAILING_DESCRIPTORS = frozenset(
    {
        "architecture",
        "architectures",
        "backend",
        "backends",
        "bundler",
        "bundlers",
        "framework",
        "frameworks",
        "patterns",
        "pattern",
        "retriever",
        "retrievers",
    }
)
_OBJECTIVE_SENTENCE_STOPWORDS = frozenset(
    {
        "a",
        "all",
        "an",
        "and",
        "at",
        "concrete",
        "covered",
        "each",
        "evidence",
        "explain",
        "for",
        "in",
        "include",
        "includes",
        "least",
        "must",
        "note",
        "of",
        "one",
        "primary",
        "should",
        "the",
        "to",
        "used",
        "with",
    }
)
_OBJECTIVE_FRAME_STOPWORDS = frozenset(
    {
        "any",
        "existing",
        "foundational",
        "on",
        "or",
        "proposed",
        "works",
    }
)
_INSTRUCTION_LEAD_WORDS = frozenset(
    {
        "analyze",
        "assess",
        "benchmark",
        "briefly",
        "capture",
        "compare",
        "contrast",
        "describe",
        "document",
        "evaluate",
        "examine",
        "explain",
        "find",
        "identify",
        "include",
        "keep",
        "note",
        "produce",
        "study",
        "survey",
    }
)


def analyze_objective_gaps(
    reflection_scores: Dict[str, Any],
    *,
    threshold: float = LOW_SCORE_THRESHOLD,
    high_priority_threshold: float = HIGH_PRIORITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Return objectives that fall below ``threshold`` sorted by severity."""

    per_objective = reflection_scores.get("per_objective_scores") or []
    gaps: List[Dict[str, Any]] = []
    for index, entry in enumerate(per_objective, start=1):
        try:
            score = float(entry.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        objective_id = str(entry.get("objective_id") or f"OBJ{index:02d}").strip()
        reasoning = str(entry.get("reasoning") or "Additional research required")
        if score >= threshold:
            continue
        gap = {
            "objective_id": objective_id,
            "current_score": round(score, 2),
            "gap_size": round(1.0 - score, 2),
            "reasoning": reasoning,
            "priority": "high" if score < high_priority_threshold else "medium",
        }
        gaps.append(gap)
    gaps.sort(key=lambda item: (item["priority"] == "high", item["gap_size"]), reverse=True)
    return gaps


def prioritize_objectives(
    per_objective_scores: Sequence[Dict[str, Any]],
) -> List[Tuple[str, float, str]]:
    """Return tuples of ``(objective_id, score, status)`` sorted by score."""

    prioritized: List[Tuple[str, float, str]] = []
    for index, entry in enumerate(per_objective_scores, start=1):
        try:
            score = float(entry.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        objective_id = str(entry.get("objective_id") or f"OBJ{index:02d}").strip()
        status = (
            "complete"
            if score >= 0.9
            else ("focus" if score < LOW_SCORE_THRESHOLD else "monitor")
        )
        prioritized.append((objective_id, round(score, 2), status))
    prioritized.sort(key=lambda item: item[1])
    return prioritized


def generate_targeted_queries(
    gaps: Sequence[Dict[str, Any]],
    mission_context: Dict[str, Any] | None,
    *,
    previous_queries: Sequence[str] | None = None,
    objective_lookup: Sequence[str] | None = None,
    max_queries: int = MAX_TARGETED_QUERIES,
) -> List[str]:
    """Generate deduplicated research queries that focus on the provided gaps."""

    focus_hint = _extract_focus_hint(mission_context)
    preserved_terms = _extract_preserved_terms(mission_context, objective_lookup)
    seen = {_normalise_query(item) for item in _clean_queries(previous_queries)}
    queries: List[str] = []
    for gap in gaps:
        candidate = _build_query_from_gap(gap, focus_hint, objective_lookup)
        if not candidate:
            continue
        candidate = _append_preserved_terms(candidate, preserved_terms)
        normalized = _normalise_query(candidate)
        if normalized in seen:
            continue
        queries.append(candidate)
        seen.add(normalized)
        if len(queries) >= max_queries:
            break
    if not queries:
        # Build fallback from the first objective text, not internal labels
        fallback_topic = ""
        if objective_lookup:
            fallback_topic = objective_lookup[0]
        if not fallback_topic and focus_hint and not _is_internal_label(focus_hint):
            fallback_topic = focus_hint
        if fallback_topic:
            fallback_query = _append_preserved_terms(fallback_topic, preserved_terms)
            normalized = _normalise_query(fallback_query)
            if normalized not in seen:
                queries.append(fallback_query[:MAX_QUERY_LENGTH])
    return queries


def detect_stuck_objectives(
    score_history: Sequence[Dict[str, Any]],
    *,
    window: int = DEFAULT_STUCK_WINDOW,
    min_improvement: float = MIN_IMPROVEMENT,
) -> List[str]:
    """Return objective identifiers whose improvement stays below ``min_improvement``."""

    if not score_history:
        return []
    relevant = list(score_history)[-window:]
    per_objective: Dict[str, List[float]] = {}
    for entry in relevant:
        scores = entry.get("scores") or []
        for score_entry in scores:
            objective_id = str(score_entry.get("objective_id") or "").strip()
            if not objective_id:
                continue
            try:
                value = float(score_entry.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            per_objective.setdefault(objective_id, []).append(value)
    stuck: List[str] = []
    for objective_id, values in per_objective.items():
        # S18: Require 3+ loops of data before declaring stuck (was 2)
        if len(values) < 3:
            continue
        improvement = values[-1] - values[0]
        if improvement < min_improvement:
            stuck.append(objective_id)
    stuck.sort()
    return stuck


def _clean_queries(queries: Sequence[str] | None) -> Iterable[str]:
    for query in queries or []:
        if not query:
            continue
        yield query


def _unique_strings(values: Sequence[str] | None) -> List[str]:
    unique: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        normalized = _normalise_query(text)
        if not normalized or normalized in seen:
            continue
        unique.append(text)
        seen.add(normalized)
    return unique


def _extract_preserved_terms(
    mission_context: Dict[str, Any] | None,
    objective_lookup: Sequence[str] | None,
) -> List[str]:
    texts: List[str] = []
    context = mission_context or {}
    objective_text = context.get("objective")
    if isinstance(objective_text, str) and objective_text.strip():
        texts.append(objective_text.strip())
    for objective in objective_lookup or []:
        if isinstance(objective, str) and objective.strip():
            texts.append(objective.strip())

    preserved_terms: List[str] = []
    seen: set[str] = set()
    for extractor in (
        _extract_author_year_citations,
        _extract_acronyms,
        _extract_camel_case_terms,
        _extract_named_phrases,
    ):
        for text in texts:
            for term in extractor(text):
                normalized = _normalise_query(term)
                if not normalized or normalized in seen:
                    continue
                preserved_terms.append(term.strip())
                seen.add(normalized)
    return preserved_terms


def _extract_author_year_citations(text: str) -> List[str]:
    return [match.group(0).strip() for match in AUTHOR_YEAR_PATTERN.finditer(text or "")]


def _extract_acronyms(text: str) -> List[str]:
    return [match.group(0).strip() for match in ACRONYM_PATTERN.finditer(text or "")]


def _extract_camel_case_terms(text: str) -> List[str]:
    return [match.group(0).strip() for match in CAMEL_CASE_PATTERN.finditer(text or "")]


def _extract_named_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    seen: set[str] = set()
    for raw_phrase in QUOTED_PHRASE_PATTERN.findall(text or ""):
        phrase = raw_phrase.strip()
        if phrase and _normalise_query(phrase) not in seen:
            phrases.append(phrase)
            seen.add(_normalise_query(phrase))
    for match in PROPER_NOUN_PATTERN.finditer(text or ""):
        phrase = match.group(0).strip()
        if _looks_like_instruction_phrase(phrase):
            continue
        normalized = _normalise_query(phrase)
        if not normalized or normalized in seen:
            continue
        phrases.append(phrase)
        seen.add(normalized)
    return phrases


def _looks_like_instruction_phrase(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", str(text or ""))
    if not tokens:
        return False
    return tokens[0].lower() in _INSTRUCTION_LEAD_WORDS


def _extract_enumerated_entities(text: str) -> List[str]:
    lowered = (text or "").lower()
    if "column" in lowered or "table" in lowered:
        return []

    entities: List[str] = []
    for match in ENUMERATED_ENTITY_INTRO_PATTERN.finditer(text or ""):
        rest = match.group("rest")
        parts = [
            item.strip(" .`'\"()")
            for item in re.split(r",|\band\b", rest)
            if item.strip(" .`'\"()")
        ]
        if len(parts) < 2:
            continue
        for part in parts:
            normalized = _normalize_enumerated_entity(part)
            if normalized:
                entities.append(normalized)
    return _unique_strings(entities)


def _normalize_enumerated_entity(part: str) -> str:
    tokens = [token for token in str(part or "").strip().split() if token]
    if not tokens:
        return ""

    for index, token in enumerate(tokens):
        if index > 0 and token.lower() in _ENTITY_TRUNCATION_WORDS:
            tokens = tokens[:index]
            break

    while len(tokens) > 1 and tokens[-1].lower() in _ENTITY_TRAILING_DESCRIPTORS:
        tokens = tokens[:-1]

    if not tokens:
        return ""
    return " ".join(tokens)


def _extract_objective_subject_terms(text: str) -> List[str]:
    subjects: List[str] = []
    seen: set[str] = set()
    for extractor in (
        _extract_author_year_citations,
        _extract_acronyms,
        _extract_camel_case_terms,
        _extract_enumerated_entities,
        _extract_named_phrases,
    ):
        for term in extractor(text):
            normalized = _normalise_query(term)
            if not normalized or normalized in seen:
                continue
            subjects.append(term.strip())
            seen.add(normalized)
    return subjects


def _objective_query_terms(text: str) -> str:
    objective_text = str(text or "").strip()
    if not objective_text:
        return ""

    for pattern, replacement in _QUERY_INTENT_REWRITES:
        if pattern.search(objective_text):
            return replacement

    tokens = [
        token
        for token in re.findall(r"[A-Za-z]+", objective_text.lower())
        if token not in _OBJECTIVE_SENTENCE_STOPWORDS
        and token not in _INSTRUCTION_LEAD_WORDS
        and token not in _OBJECTIVE_FRAME_STOPWORDS
    ]
    if not tokens:
        return objective_text
    return " ".join(tokens[:8])


def _compose_objective_query(
    *,
    focus_hint: str,
    subject_terms: Sequence[str],
    objective_terms: str,
) -> str:
    parts: List[str] = []
    if focus_hint:
        parts.append(focus_hint)
    if subject_terms:
        parts.append(" ".join(str(item).strip() for item in subject_terms if str(item).strip()))
    if objective_terms:
        parts.append(objective_terms)

    query_parts: List[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = _normalise_query(part)
        if not normalized or normalized in seen:
            continue
        query_parts.append(part.strip())
        seen.add(normalized)
    return " ".join(query_parts).strip()[:MAX_QUERY_LENGTH]


def _append_preserved_terms(query: str, preserved_terms: Sequence[str]) -> str:
    if not preserved_terms:
        return query[:MAX_QUERY_LENGTH]

    combined = query.strip()
    for term in preserved_terms:
        if not term or _normalise_query(term) in _normalise_query(combined):
            continue
        candidate = f"{combined} {term}".strip()
        if len(candidate) > MAX_QUERY_LENGTH:
            break
        combined = candidate
    return combined[:MAX_QUERY_LENGTH]


def _extract_focus_hint(mission_context: Dict[str, Any] | None) -> str:
    context = mission_context or {}
    for key in ("focus", "summary", "title", "background"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            if key == "title":
                return normalize_mission_title(value, context.get("mission_id"))
            return value.strip()
    return ""


def _build_query_from_gap(
    gap: Dict[str, Any],
    focus_hint: str,
    objective_lookup: Sequence[str] | None,
) -> str:
    """Build a clean, keyword-focused search query from a gap analysis entry.

    Drops reasoning paragraphs (internal scoring commentary, not search terms).
    Only includes focus_hint if it adds topical context (not test names).
    """
    objective_id = str(gap.get("objective_id") or "").strip()
    descriptor = _objective_descriptor(objective_id, objective_lookup)
    # Only use focus_hint if it looks like a real topic, not an internal label
    # (skip test names, IDs, and strings that are mostly uppercase/dots)
    use_hint = ""
    if focus_hint and not _is_internal_label(focus_hint):
        use_hint = focus_hint
    subject_terms = _extract_objective_subject_terms(descriptor)
    objective_terms = _objective_query_terms(descriptor)
    if subject_terms or objective_terms != descriptor:
        query = _compose_objective_query(
            focus_hint=use_hint,
            subject_terms=subject_terms,
            objective_terms=objective_terms,
        )
        if query:
            return query
    pieces = [use_hint, descriptor]
    query = " ".join(part for part in pieces if part).strip()
    return query[:MAX_QUERY_LENGTH]


def _objective_descriptor(objective_id: str, objective_lookup: Sequence[str] | None) -> str:
    """Return clean objective text for use in search queries. No OBJ ID prefix."""
    if not objective_lookup:
        return "" if _looks_like_internal_objective_id(objective_id) else objective_id
    index = _objective_index(objective_id)
    if index is None or index >= len(objective_lookup):
        return "" if _looks_like_internal_objective_id(objective_id) else objective_id
    return objective_lookup[index]


def _objective_index(objective_id: str) -> int | None:
    digits = "".join(ch for ch in objective_id if ch.isdigit())
    if not digits:
        return None
    try:
        value = int(digits)
    except ValueError:
        return None
    return max(0, value - 1)


def _is_internal_label(text: str) -> bool:
    """Return True if the text looks like an internal label rather than a real topic.

    Catches things like test names ("Dual Search Baseline Tier Smoke Test"),
    mission IDs ("DS.TEST.BASELINE"), and strings with internal jargon.
    """
    lower = text.lower()
    internal_markers = ("smoke test", "baseline tier", "test run", "mission gap", "ds.test")
    for marker in internal_markers:
        if marker in lower:
            return True
    # Strings with lots of dots or all-caps segments are likely IDs
    if text.count(".") >= 2:
        return True
    return False


def _looks_like_internal_objective_id(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return bool(re.fullmatch(r"(?:\d+|obj-?\d+|objective-\d+|primary|sc_\d+)", normalized))


def _normalise_query(query: str | None) -> str:
    return " ".join(str(query or "").lower().split())


__all__ = [
    "analyze_objective_gaps",
    "generate_targeted_queries",
    "detect_stuck_objectives",
    "prioritize_objectives",
]
