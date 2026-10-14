"""Bounded, deterministic passage selection over an already collected body."""

from __future__ import annotations

import re
from typing import Any

MAX_PASSAGE_CHARS = 12_000
MAX_FIND_MATCHES = 5
MAX_FIND_QUERY_CHARS = 200


def validate_passage_request(start: int, length: int, query: str) -> None:
    """Reject invalid ranges before either a cache read or a network request."""
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("invalid_start")
    if (
        isinstance(length, bool) or not isinstance(length, int)
        or not 1 <= length <= MAX_PASSAGE_CHARS
    ):
        raise ValueError("invalid_length")
    if (
        not isinstance(query, str) or len(query) > MAX_FIND_QUERY_CHARS
        or (query and not query.strip())
    ):
        raise ValueError("invalid_query")


def select_source_passage(text: str, start: int, length: int, query: str) -> dict[str, Any]:
    """Read a range or find literal text, preserving Unicode character offsets.

    Find returns at most five match ranges plus context around the first match.
    Further matches can be read with start/length, or searched after the last
    match's end. No regex supplied by the model is executed. Search completeness
    describes the stored body only, never the original remote document.
    """
    validate_passage_request(start, length, query)
    if start > len(text):
        raise ValueError("range_out_of_bounds")
    matches: list[dict[str, int]] = []
    more_matches = False
    offset = start
    if query:
        for match in re.finditer(re.escape(query), text[start:], re.IGNORECASE):
            if len(matches) == MAX_FIND_MATCHES:
                more_matches = True
                break
            matches.append({"start": start + match.start(), "end": start + match.end()})
        if matches:
            # Context is centered on the first hit, not on the old source prefix.
            offset = max(start, matches[0]["start"] - min(300, length // 4))
    passage = text[offset:offset + length] if not query or matches else ""
    end = offset + len(passage)
    return {
        "text": passage,
        "range_start": offset,
        "range_end": end,
        "remaining_chars": len(text) - end,
        "next_start": end if end < len(text) else None,
        "matches": matches,
        "matches_truncated": more_matches,
        "next_match_start": matches[-1]["end"] if more_matches else None,
        "search_from": start if query else None,
        "search_complete": not more_matches if query else None,
        "text_chars_total": len(text),
        "text_chars_returned": len(passage),
        "text_truncated_for_tool": offset > 0 or end < len(text),
    }
