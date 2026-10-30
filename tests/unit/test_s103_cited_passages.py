"""S103-m03: a cut cited source gets the source budget today's packer leaves unused.

Today's packer shows each cited source at most four 500-character passages under a
2,400-character cap, whether one claim cites it or 44, and in some runs leaves most of
the 160,000-character budget unused. When the baseline is unsqueezed (no row dropped,
every selector budget the cap), a cited source the baseline cut shows up to
``min(16, max(4, n))`` chunks for its ``n`` citations: its baseline chunks widened to
whole 600-character strides, then the selector's other chunks in rank order, touching
ranges as one passage. The remainder pays for widening first and otherwise scales every
source's extra chunks by one factor. Nothing changes for any other row or for any
visibility field, so nothing shown can authorize a deletion today's rules would not.
The extension commits all or nothing and can never become a critique error.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import pytest

from deepsearch.agent import critique
from deepsearch.agent.critique import (
    SOURCE_GROUNDING_MAX_CHARS,
    _build_critique_messages,
    _extend_cited_passages,
    _format_source_row,
    _GroundingVisibility,
    _prepare_sources_with_telemetry,
    run_critique,
)
from deepsearch.report.citation import CitationAnnotation
from tests.unit.test_critique import _annotations_raw, _mock_response

URL_A = "https://a.example/paper"
URL_B = "https://b.example/paper"
URL_C = "https://c.example/paper"


def strides(count: int) -> str:
    """One paragraph of 600-character strides, each led by its own topic word."""

    words = [f"topic{chr(97 + i % 26)}{chr(97 + i // 26)} " for i in range(count)]
    return "".join((word + "lorem " * 120)[:599] + " " for word in words).rstrip()


def source(url: str, body: str) -> dict[str, Any]:
    return {"url": url, "title": url, "body": body, "content_quality": "readable_body"}


def cited(draft: str, url: str, times: int) -> list[CitationAnnotation]:
    end = draft.index(".") + 1
    return [CitationAnnotation(type="url_citation", start_index=0, end_index=end, url=url,
                               live=True) for _ in range(times)]


DRAFT = "The claim names topicaa and topicca and topicea.\n"


class Packed:
    def __init__(self, sources: list[dict[str, Any]], citations: list[CitationAnnotation],
                 total_budget: int = critique.GROUNDING_TOTAL_BUDGET_CHARS) -> None:
        self.visibility = _GroundingVisibility()
        self.records: list[dict[str, Any]] = []
        self.citations = citations
        rows, self.telemetry = _prepare_sources_with_telemetry(
            sources, citations=citations, grounding_visibility=self.visibility,
            draft_markdown=DRAFT, row_records=self.records, total_budget=total_budget,
        )
        self.baseline_rows = rows
        self.baseline_telemetry = copy.deepcopy(self.telemetry)
        self.baseline_passages = copy.deepcopy(self.visibility.passages)
        self.baseline_text = dict(self.visibility.passage_text)
        self.rows = rows

    def extend(self) -> Packed:
        self.rows = _extend_cited_passages(
            self.baseline_rows, self.telemetry, self.records, citations=self.citations,
            grounding_visibility=self.visibility,
        )
        return self

    def passages(self, url: str) -> list[dict[str, Any]]:
        return [p for p in self.visibility.passages if p["url"] == url]

    def chunks_shown(self, url: str) -> int:
        return sum(p["end"] - p["start"] for p in self.passages(url)) // 600

    def unchanged(self) -> bool:
        return (self.rows is self.baseline_rows
                and self.visibility.passages == self.baseline_passages
                and self.visibility.passage_text == self.baseline_text
                and {k: v for k, v in self.telemetry.items() if not k.startswith("cited_extra")}
                == self.baseline_telemetry)


def block(rows: list[dict[str, Any]]) -> int:
    return sum(len(_format_source_row(i + 1, row)) for i, row in enumerate(rows)) + 2 * (
        len(rows) - 1)


def test_a_cut_cited_source_shows_whole_strides_and_one_chunk_per_citation() -> None:
    packed = Packed([source(URL_A, strides(20))], cited(DRAFT, URL_A, 8)).extend()
    telemetry = packed.telemetry
    assert (telemetry["cited_extra_unsqueezed"], telemetry["cited_extra_eligible"],
            telemetry["cited_extra_sources"], telemetry["cited_extra_baseline_passages"]) == (
        True, 1, 1, 4)
    # Eight citations: the four baseline chunks widened, then four more.
    assert telemetry["cited_extra_chunks"] == 4 and packed.chunks_shown(URL_A) == 8
    baseline = [(p["start"], p["end"]) for p in packed.baseline_passages]
    assert all(end - start == 500 for start, end in baseline)
    shown = packed.passages(URL_A)
    assert sorted(tuple(r) for p in shown for r in p["baseline_ranges"]) == baseline
    for start, end in baseline:
        assert any(p["start"] <= start and end <= p["end"] for p in shown)
    assert [p["evidence_id"] for p in shown] == [f"E1.{k + 1}" for k in range(len(shown))]
    assert telemetry["source_block_chars"] == block(packed.rows) <= telemetry["source_budget_chars"]
    assert telemetry["cited_extra_chars"] == (
        telemetry["source_block_chars"] - packed.baseline_telemetry["source_block_chars"])


@pytest.mark.parametrize(("citations", "chunks"), [(1, 4), (4, 4), (10, 10), (40, 16)])
def test_the_chunk_limit_is_one_per_citation_between_4_and_16(citations: int, chunks: int) -> None:
    packed = Packed([source(URL_A, strides(20))], cited(DRAFT, URL_A, citations)).extend()
    assert packed.chunks_shown(URL_A) == chunks
    assert packed.telemetry["cited_extra_chunks"] == chunks - 4


def test_a_source_shown_whole_and_an_uncited_source_are_never_extended() -> None:
    sources = [source(URL_A, strides(20)), source(URL_B, "A short body that fits whole."),
               source(URL_C, strides(20))]
    citations = [*cited(DRAFT, URL_A, 6), *cited(DRAFT, URL_B, 6)]
    packed = Packed(sources, citations).extend()
    assert packed.telemetry["cited_extra_eligible"] == 1
    assert packed.rows[1] is packed.baseline_rows[1] and packed.rows[2] is packed.baseline_rows[2]
    for url in (URL_B, URL_C):
        assert packed.passages(url) == [p for p in packed.baseline_passages if p["url"] == url]
        assert all("baseline_ranges" not in p for p in packed.passages(url))


def test_widening_stops_at_the_chunk_and_text_end_and_touching_ranges_merge() -> None:
    paragraphs = [f"topic{chr(97 + i)}a " + "lorem " * 100 for i in range(6)]
    body = "\n\n".join(p[:550] for p in paragraphs[:5]) + "\n\n" + paragraphs[5][:300]
    packed = Packed([source(URL_A, body)], cited(DRAFT, URL_A, 6)).extend()
    assert [len(m) for m in body.split("\n\n")] == [550] * 5 + [300]
    for passage in packed.passages(URL_A):
        text = packed.visibility.passage_text[passage["evidence_id"]]
        assert "\n" not in text and passage["end"] <= len(body)
        assert passage["end"] - passage["start"] in {550, 300}
    # In one paragraph, adjacent strides touch and are shown as one passage.
    long = Packed([source(URL_A, strides(20))], cited(DRAFT, URL_A, 16)).extend()
    shown = long.passages(URL_A)
    assert len(shown) < 16
    assert sum(len(p["baseline_ranges"]) for p in shown) == 4


def test_a_row_widened_to_its_whole_body_keeps_the_baseline_visibility() -> None:
    body = strides(5)
    packed = Packed([source(URL_A, body)], cited(DRAFT, URL_A, 5)).extend()
    (passage,) = packed.passages(URL_A)
    assert (passage["start"], passage["end"]) == (0, len(body))
    baseline = packed.baseline_telemetry
    for key in ("sources_truncated", "source_urls_fully_visible", "source_urls_incomplete",
                "grounding_context_complete"):
        assert packed.telemetry[key] == baseline[key]
    assert (baseline["sources_truncated"], baseline["grounding_context_complete"]) == (1, 0)
    assert packed.visibility.fully_visible_urls == set()
    assert packed.visibility.corpus_complete is False


def test_a_squeezed_baseline_gets_nothing() -> None:
    long = strides(20)
    # A row limited by the total: its selector budget is below the cap.
    limited = Packed([source(URL_A, long)], cited(DRAFT, URL_A, 8), total_budget=2000).extend()
    assert limited.records[0]["budget"] < SOURCE_GROUNDING_MAX_CHARS
    # A dropped row: the second source no longer fits.
    rows = [source(URL_A, long), source(URL_B, long)]
    first = Packed(rows[:1], cited(DRAFT, URL_A, 8))
    dropped = Packed(rows, cited(DRAFT, URL_A, 8),
                     total_budget=first.telemetry["source_block_chars"] + 60).extend()
    assert dropped.telemetry["sources_dropped"] == 1
    for packed in (limited, dropped):
        assert packed.telemetry["cited_extra_unsqueezed"] is False
        assert (packed.telemetry["cited_extra_chunks"], packed.telemetry["cited_extra_sources"],
                packed.telemetry["cited_extra_chars"]) == (0, 0, 0)
        assert packed.unchanged()
        assert all("baseline_ranges" not in p for p in packed.visibility.passages)


def test_a_remainder_too_small_to_widen_skips_and_changes_nothing() -> None:
    baseline = Packed([source(URL_A, strides(20))], cited(DRAFT, URL_A, 8))
    fixed = len(_format_source_row(1, {"url": URL_A, "title": URL_A, "snippet": ""}))
    tight = Packed([source(URL_A, strides(20))], cited(DRAFT, URL_A, 8),
                   total_budget=SOURCE_GROUNDING_MAX_CHARS + fixed).extend()
    assert tight.records[0]["budget"] == SOURCE_GROUNDING_MAX_CHARS
    assert tight.baseline_passages == baseline.baseline_passages
    assert tight.telemetry["cited_extra_unsqueezed"] is True
    assert tight.telemetry["cited_extra_remainder"] < 400
    assert (tight.telemetry["cited_extra_skipped"], tight.telemetry["cited_extra_fallback"]) == (
        "remainder_below_widening", False)
    assert tight.telemetry["cited_extra_chunks"] == 0 and tight.unchanged()


def test_a_short_remainder_scales_every_sources_extra_chunks_by_one_factor() -> None:
    sources = [source(URL_A, strides(20)), source(URL_B, strides(20))]
    citations = [*cited(DRAFT, URL_A, 16), *cited(DRAFT, URL_B, 8)]
    full = Packed(sources, citations).extend()
    assert (full.chunks_shown(URL_A), full.chunks_shown(URL_B)) == (16, 8)
    # Half of each source's extras (12 and 4) fit; the next whole-chunk step does not.
    budget = full.baseline_telemetry["source_block_chars"]
    half_demand = block(scaled_rows(Packed(sources, citations), {0: 6, 1: 2})) - budget
    short = Packed(sources, citations, total_budget=budget + half_demand + 1).extend()
    assert short.telemetry["cited_extra_unsqueezed"] is True
    assert (short.chunks_shown(URL_A), short.chunks_shown(URL_B)) == (4 + 6, 4 + 2)
    assert short.telemetry["cited_extra_chunks"] == 8
    assert short.telemetry["source_block_chars"] <= short.telemetry["source_budget_chars"]


def scaled_rows(packed: Packed, counts: dict[int, int]) -> list[dict[str, Any]]:
    """The rows with each source's first ``counts`` extra chunks, as the extension renders."""

    rows = list(packed.baseline_rows)
    for index, extra_chunks in counts.items():
        record = packed.records[index]
        shown = [(p["start"], p["end"]) for p in record["passages"]]
        limit = min(16, max(4, sum(c.url == record["url"] for c in packed.citations)))
        widened, extra = critique.cited_passage_ranges(
            record["text"], record["claims"], shown, limit=limit)
        text, _ = critique.render_passages(
            record["text"], [*widened, *extra[:extra_chunks]], shown,
            source_index=index + 1, url=record["url"])
        rows[index] = {**rows[index], "snippet": text}
    return rows


def _raise(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("injected")


@pytest.mark.parametrize("injection", [
    "cited_passage_ranges", "render_passages", "late_render_passages", "format_source_row",
    "invariant",
])
def test_any_failure_keeps_the_baseline_whole(monkeypatch, injection: str) -> None:
    packed = Packed([source(URL_A, strides(20)), source(URL_B, strides(20))],
                    [*cited(DRAFT, URL_A, 8), *cited(DRAFT, URL_B, 8)])
    real_render = critique.render_passages
    calls = {"n": 0}

    def late(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] > 4:
            raise RuntimeError("injected late")
        return real_render(*args, **kwargs)

    def drop(body, ranges, baseline, *, source_index, url):
        return real_render(body, list(ranges)[1:], baseline, source_index=source_index, url=url)

    patches = {
        "cited_passage_ranges": ("cited_passage_ranges", _raise),
        "render_passages": ("render_passages", _raise),
        "late_render_passages": ("render_passages", late),
        "format_source_row": ("_format_source_row", _raise),
        "invariant": ("render_passages", drop),
    }
    name, replacement = patches[injection]
    monkeypatch.setattr(critique, name, replacement)
    packed.extend()
    assert packed.telemetry["cited_extra_fallback"] is True
    assert packed.telemetry["cited_extra_chunks"] == packed.telemetry["cited_extra_sources"] == 0
    assert packed.unchanged()


def test_a_failed_extension_is_never_a_critique_error(monkeypatch) -> None:
    monkeypatch.setattr(critique, "cited_passage_ranges", _raise)
    sources = [source(URL_A, strides(20))]
    citations = cited(DRAFT, URL_A, 8)
    grounding: dict[str, Any] = {}
    messages = _build_critique_messages(
        DRAFT, sources, {"title": "t", "objective": "o"}, citations=citations,
        grounding_telemetry=grounding, grounding_visibility=_GroundingVisibility(),
    )
    assert grounding["cited_extra_fallback"] is True and messages[1]["content"]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw([], claim_count=1)))
    result = run_critique(draft_markdown=DRAFT, sources_collected=sources,
                          mission_context={"title": "t", "objective": "o"},
                          citations=citations, model=fake_model)
    assert result.error is None
    assert result.grounding_telemetry["cited_extra_fallback"] is True


def test_baseline_ranges_reach_the_assessment_evidence_entries() -> None:
    sources = [source(URL_A, strides(20))]
    citations = cited(DRAFT, URL_A, 8)
    visibility = _GroundingVisibility()
    grounding: dict[str, Any] = {}
    _build_critique_messages(DRAFT, sources, {"title": "t", "objective": "o"},
                             citations=citations, grounding_telemetry=grounding,
                             grounding_visibility=visibility)
    passage = next(p for p in visibility.passages if p.get("baseline_ranges"))
    quote = visibility.passage_text[passage["evidence_id"]][:40]
    raw = _annotations_raw([], claim_count=1).replace(
        '"annotations": []',
        '"annotations": [], "assessments": [{"anchor": "The claim names topicaa", '
        f'"verdict": "supported", "evidence_ids": ["{passage["evidence_id"]}"], '
        f'"evidence_quotes": {{"{passage["evidence_id"]}": "{quote}"}}}}]')
    fake_model = MagicMock(return_value=_mock_response(raw))
    result = run_critique(draft_markdown=DRAFT, sources_collected=sources,
                          mission_context={"title": "t", "objective": "o"},
                          citations=citations, model=fake_model)
    entries = [e for r in result.assessment["records"] for e in r["evidence"]]
    assert entries and entries[0]["baseline_ranges"] == passage["baseline_ranges"]
