"""S68 PD8 — indexing-validation escape hatch.

Confirms the S67 ``CitationAnnotation`` schema parses an OpenAI-shape
``url_citation`` payload **byte-identically** under Python's codepoint /
end-exclusive slicing convention (S67 plan-doc decision #519). If a +1 / −1
mismatch surfaces, ship the trivial fix in s68-m01 (per the S67 escape
hatch). If no mismatch, this file stands as a regression guard against
future indexing drift.

OpenAI's documented ``url_citation`` shape — verified against Microsoft
Learn / Azure OpenAI mirror docs at S67 plan-doc time — is::

    {
        "type": "url_citation",
        "start_index": int,
        "end_index": int,
        "url": str,
        "title": Optional[str],
    }

where ``start_index`` and ``end_index`` are codepoint offsets into the
final message text and ``message_text[start_index:end_index]`` returns the
cited span (Python-slice, end-exclusive).

These tests use synthetic OpenAI-shape payloads (the actual SDK is not a
test dependency); the contract being tested is the *shape compatibility*,
not network behaviour.
"""

from __future__ import annotations

import pytest

from deepsearch.report.citation import CitationAnnotation


def _openai_payload(
    *,
    start: int,
    end: int,
    url: str,
    title: str | None = None,
) -> dict:
    """Construct a payload matching OpenAI's documented ``url_citation`` shape."""

    payload: dict = {
        "type": "url_citation",
        "start_index": start,
        "end_index": end,
        "url": url,
    }
    if title is not None:
        payload["title"] = title
    return payload


# ---------------------------------------------------------------------------
# 1. ASCII happy path — round-trip identity for indices and span extraction.
# ---------------------------------------------------------------------------


def test_ascii_happy_path_round_trips_indices():
    text = "Lambda's cold start is well-documented."
    cited_span = "Lambda's cold start is well-documented"
    start = text.index(cited_span)
    end = start + len(cited_span)

    payload = _openai_payload(
        start=start,
        end=end,
        url="https://docs.aws.amazon.com/lambda/latest/dg/foundation-arch.html",
        title="Lambda Foundation",
    )
    ann = CitationAnnotation(**payload)

    assert ann.start_index == start
    assert ann.end_index == end
    assert ann.url == payload["url"]
    assert ann.title == payload["title"]
    # End-exclusive Python slice returns the cited span verbatim.
    assert text[ann.start_index : ann.end_index] == cited_span


def test_ascii_to_dict_matches_openai_shape():
    payload = _openai_payload(
        start=0,
        end=10,
        url="https://example.com/x",
        title="X",
    )
    ann = CitationAnnotation(**payload)
    data = ann.to_dict()

    # to_dict() includes the type discriminator + indices + URL + title.
    # The S68 ``live`` field is omitted when None (default) per
    # ``model_dump(exclude_none=True)``.
    for key in ("type", "start_index", "end_index", "url", "title"):
        assert data[key] == payload[key]
    assert data["type"] == "url_citation"
    assert "live" not in data


# ---------------------------------------------------------------------------
# 2. Multi-byte (BMP + supplementary plane) — codepoint-vs-byte indexing
#    discrimination. Python str indexing is codepoint-based; UTF-8 byte
#    offsets would shift on emoji or CJK. OpenAI's contract is also
#    codepoint-based.
# ---------------------------------------------------------------------------


def test_emoji_before_citation_indices_are_codepoints_not_bytes():
    # "🔥" is a single codepoint (U+1F525) but 4 bytes in UTF-8.
    text = "🔥 Lambda is hot"
    cited = "Lambda is hot"
    start = text.index(cited)
    end = start + len(cited)
    assert start == 2  # one emoji codepoint + one space → 2 codepoints

    payload = _openai_payload(start=start, end=end, url="https://x")
    ann = CitationAnnotation(**payload)

    assert text[ann.start_index : ann.end_index] == cited
    # If indexing were byte-oriented this would be ≠ 2 (bytes-of-🔥 = 4).
    assert ann.start_index == 2


def test_cjk_before_citation_indices_are_codepoints():
    # Each CJK char in this string is a single codepoint, 3 bytes in UTF-8.
    text = "クラウド環境では Lambda's cold start matters"
    cited = "Lambda's cold start matters"
    start = text.index(cited)
    end = start + len(cited)

    payload = _openai_payload(start=start, end=end, url="https://x")
    ann = CitationAnnotation(**payload)
    assert text[ann.start_index : ann.end_index] == cited


# ---------------------------------------------------------------------------
# 3. Edge positions — citation at start / end of message; empty span.
# ---------------------------------------------------------------------------


def test_citation_at_start_of_message():
    text = "First claim about cold start"
    cited = "First claim"
    payload = _openai_payload(start=0, end=len(cited), url="https://a")
    ann = CitationAnnotation(**payload)
    assert text[ann.start_index : ann.end_index] == cited
    assert ann.start_index == 0


def test_citation_at_end_of_message():
    text = "trailing claim"
    payload = _openai_payload(start=0, end=len(text), url="https://a")
    ann = CitationAnnotation(**payload)
    assert text[ann.start_index : ann.end_index] == text
    assert ann.end_index == len(text)


def test_two_citations_share_end_index():
    """A claim with two citations: both annotations may share end_index.

    OpenAI's contract permits multiple ``url_citation`` annotations covering
    the same character span. The schema must accept identical end_index
    values across two annotations.
    """

    text = "Cold start and concurrency interact"
    end = len(text)
    a1 = CitationAnnotation(**_openai_payload(start=0, end=end, url="https://a"))
    a2 = CitationAnnotation(**_openai_payload(start=0, end=end, url="https://b"))
    assert a1.end_index == a2.end_index == end
    assert text[a1.start_index : a1.end_index] == text


def test_zero_length_span_accepted():
    """end_index == start_index is a marker insertion point — allowed."""

    payload = _openai_payload(start=5, end=5, url="https://a")
    ann = CitationAnnotation(**payload)
    assert ann.start_index == ann.end_index == 5


# ---------------------------------------------------------------------------
# 4. Round-trip — to_dict() output is itself a valid input for re-parsing.
#    Closes the indexing escape hatch by proving symmetry.
# ---------------------------------------------------------------------------


def test_to_dict_round_trip_preserves_indices():
    text = "Round-trip claim 🚀 about cold start"
    cited = "Round-trip claim"
    payload = _openai_payload(
        start=0, end=len(cited), url="https://x", title="X"
    )
    ann1 = CitationAnnotation(**payload)
    data = ann1.to_dict()
    ann2 = CitationAnnotation(**data)
    assert ann1.start_index == ann2.start_index
    assert ann1.end_index == ann2.end_index
    assert ann1.url == ann2.url
    assert ann1.title == ann2.title
    assert text[ann2.start_index : ann2.end_index] == cited


def test_invalid_indices_rejected():
    """Schema enforces start_index ≤ end_index."""

    with pytest.raises(Exception):
        CitationAnnotation(**_openai_payload(start=10, end=5, url="https://a"))
