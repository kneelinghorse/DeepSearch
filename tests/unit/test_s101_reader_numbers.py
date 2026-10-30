"""S101-m02: a kept inline citation shows its reader number; the word count holds.

Finalize strips every ``[cite:URL]`` marker before persistence, so a claim cited
only by a machine marker reached the reader with no citation. Rendering now puts
the number the report's own References section gives that URL in front of the
marker before it is stripped. A number the agent wrote is never changed, no
References entry is added, and the stripped draft keeps its word count, because
an authored word limit is terminal when exceeded. These tests pin the rule on
synthetic drafts; test_s101_private_reader_number_replays.py replays it on the
frozen corpus.
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_extraction import extract_citations
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import (
    GAP_NOTE_DEFAULT,
    CritiqueAnnotation,
    CritiqueResult,
    run_critique,
)
from deepsearch.mission.scope import AuthoredScope, ScopeAdmission
from deepsearch.report.citation import CitationAnnotation, rebase_citation_annotations
from deepsearch.report.scope_validation import (
    _WHITESPACE,
    count_report_words,
    validate_persisted_scope,
)
from tests.unit.test_critique import _annotations_raw, _mock_response

URL_A = "https://a.example/x"
URL_B = "https://b.example/y"
URL_Z = "https://z.example/q"
A, B, Z = (f"[cite:{url}]" for url in (URL_A, URL_B, URL_Z))
REFERENCES = f"\n\n## References\n[1] Alpha — {URL_A}\n[2] Beta — {URL_B}\n"


def render(body: str, references: str = REFERENCES) -> tuple[str, dict, dict]:
    return fnode._render_reader_numbers(body + references)


def reader_text(draft: str) -> str:
    """What the reader sees of the body: the stripped draft before References."""

    return extract_citations(draft)[0].split("\n\n## References")[0]


def skips(telemetry: dict) -> dict[str, int]:
    return {key: value for key, value in telemetry.items()
            if key.startswith("skipped_") and value}


def test_whitespace_and_line_breaks_come_from_the_word_counter() -> None:
    members = {c for c in map(chr, range(sys.maxunicode + 1)) if _WHITESPACE.fullmatch(c)}
    breaks = {c for c in members if fnode._is_line_break(c)}
    assert breaks == set("\n\x0b\x0c\r\x1c\x1d\x1e\x85\u2028\u2029")
    assert {" ", "\t", "\u00a0", "\u202f", "\u001f"} <= members - breaks
    assert all(fnode._is_spacing(c) for c in members - breaks)
    assert not any(fnode._is_report_whitespace(c) for c in ("\u200b", "a", "[", "."))


SHAPES = [
    ("mid-sentence", f"The claim {A} and more.", "The claim[1] and more.", {}),
    ("before punctuation", f"The claim {A}.", "The claim [1].", {}),
    ("chain", f"The claim {A} {B} and more.", "The claim[1][2] and more.", {}),
    ("partial chain, unrendered first", f"The claim {Z} {B} and", "The claim[2] and",
     {"skipped_not_in_references": 1}),
    ("partial chain, unrendered last", f"The claim {A} {Z} and", "The claim[1] and",
     {"skipped_not_in_references": 1}),
    ("line end", f"The claim {A}\nNext line.", "The claim[1]\nNext line.", {}),
    ("line start", f"First line.\n{A} next", "First line.\n next", {"skipped_line_start": 1}),
    ("text start", f"{A} opens the text", " opens the text", {"skipped_line_start": 1}),
    ("after a sentence end", f"The claim. {A} {B} Next", "The claim.   Next",
     {"skipped_sentence_boundary": 2}),
    ("after a question", f"Is it? {A} Next", "Is it?  Next", {"skipped_sentence_boundary": 1}),
    ("before a closing quote", f'"The claim {A}" he said', '"The claim [1]" he said', {}),
    ("before a closing parenthesis", f"(The claim {A}) more", "(The claim [1]) more", {}),
    ("table cell", f"| The claim {A} | x |", "| The claim[1] | x |", {}),
    ("no-break space before", f"The claim\u00a0{A} and", "The claim[1] and", {}),
    ("U+202F inside a group", f"The claim {A}\u202f{B} and", "The claim[1][2] and", {}),
    ("U+001F before", f"The claim\u001f{A} and", "The claim[1] and", {}),
    ("U+2028 before", f"The claim\u2028{A} and", "The claim\u2028 and",
     {"skipped_line_start": 1}),
    ("period and no-break space", f"The claim.\u00a0{A} and", "The claim.\u00a0 and",
     {"skipped_sentence_boundary": 1}),
    ("line end then line start", f"The claim {A}\n{B} next", "The claim[1]\n next",
     {"skipped_line_start": 1}),
    ("agent number on the first marker", f"The claim [1]{A} {B} and", "The claim [1]  and",
     {"skipped_conflicting_number": 1}),
    # The group's inner spacing alone separated the claim from the period.
    ("inner spacing only, then text", f"The claim{A} {B}.", "The claim [1][2].", {}),
    # A number opening a line would turn a list item or table row into prose.
    ("line start, then text", f"First line.\n{A}- item", "First line.\n- item",
     {"skipped_line_start": 1}),
    ("text start, then text", f"{A}, opens", ", opens", {"skipped_line_start": 1}),
    # The second marker repeats the first's claim and URL after its number.
    ("repeat after its own number", f"The claim {A} [1] {A} and", "The claim  [1]  and", {}),
]


@pytest.mark.parametrize(("name", "body", "expected", "skipped"), SHAPES,
                         ids=[shape[0] for shape in SHAPES])
def test_each_shape_keeps_its_word_count(name: str, body: str, expected: str,
                                         skipped: dict) -> None:
    draft = body + REFERENCES
    rendered, telemetry, keys = render(body)
    assert reader_text(rendered) == expected
    assert count_report_words(extract_citations(rendered)[0]) == count_report_words(
        extract_citations(draft)[0]) == telemetry["words_before"] == telemetry["words_after"]
    assert skips(telemetry) == skipped and telemetry["error"] is False
    assert telemetry["rendered"] == len(keys) == expected.count("[1]") + expected.count("[2]") - (
        body.count("[1]") + body.count("[2]"))
    assert (telemetry["draft_chars_before"], telemetry["draft_chars_after"]) == (
        len(draft), len(rendered))


def test_random_drafts_keep_their_word_count_and_every_rendered_number_maps() -> None:
    tokens = ["claim", "word", ".", ",", "!", "?", '"', ")", "(", "|", "`", "\\", " ", " ", "  ",
              "\t", "\u00a0", "\u202f", "\u001f", "\n", "\u2028", "\x0c", A, A, B, Z, "[1]", "[2]"]
    generator = random.Random(101)
    rendered_total = 0
    for _ in range(3000):
        body = "".join(generator.choice(tokens) for _ in range(generator.randint(1, 12)))
        draft = body + REFERENCES
        rendered, telemetry, keys = render(body)
        assert telemetry["error"] is False, repr(body)
        before, _ = extract_citations(draft)
        after, annotations = extract_citations(rendered)
        assert count_report_words(after) == count_report_words(before), repr(body)
        # Every rendered number names its citation's URL at the audit's boundary.
        entries = fnode._parse_references_entries(after, expected_urls={
            fnode._reference_url_identity(a.url) for a in annotations})
        primary = {(a.end_index, a.url): a for a in annotations if a._extraction_mode == "primary"}
        for (end, url), number in keys.items():
            # No token is a superscript, so the drafts never read one (S103 m02).
            assert number in fnode._boundary_numbers(
                after, primary[(end, url)], superscripts=False), repr(body)
            assert fnode._reference_url_identity(url) in entries.numbered_urls[number]
        rendered_total += len(keys)
    assert rendered_total > 500  # the generator exercises rendering, not only skips


def test_a_scoped_draft_at_its_maximum_still_passes_validation_after_rendering() -> None:
    body = " ".join(f"Claim {n} holds {A} and claim {n} also holds {B} here." for n in range(40))
    draft = body + REFERENCES
    words = count_report_words(extract_citations(draft)[0])
    admission = ScopeAdmission(AuthoredScope(max_words=words))
    rendered, telemetry, keys = render(body)
    assert len(keys) == 80
    result = validate_persisted_scope(
        admission=admission, pre_render=draft, persisted=extract_citations(rendered)[0],
        observed_sources=[], cited_urls=[URL_A, URL_B],
    )
    assert result["persisted"]["words"] == words and result["violations"] == []


# --- negative controls -------------------------------------------------------


def test_a_url_listed_under_two_numbers_is_skipped() -> None:
    references = f"\n\n## References\n[1] Alpha — {URL_A}\n[2] Alpha again — {URL_A}\n"
    rendered, telemetry, _ = render(f"The claim {A} and", references)
    assert rendered == f"The claim {A} and" + references
    assert skips(telemetry) == {"skipped_multiple_numbers": 1}


def test_without_a_references_section_each_citation_is_counted_once() -> None:
    rendered, telemetry, _ = render(f"The claim {A} {A} and {B}.", "\n")
    assert rendered == f"The claim {A} {A} and {B}.\n"
    assert skips(telemetry) == {"skipped_not_in_references": 2}


def test_a_draft_without_inline_markers_is_not_extracted(monkeypatch) -> None:
    draft = "Numbered claim [1] only.\n" + REFERENCES

    def boom(*args, **kwargs):
        raise AssertionError("extraction ran on a draft with nothing to render")

    monkeypatch.setattr(fnode, "extract_citations", boom)
    rendered, telemetry, keys = fnode._render_reader_numbers(draft)
    assert (rendered, keys, telemetry["error"]) == (draft, {}, False)
    assert telemetry["words_before"] == telemetry["words_after"] == count_report_words(draft)


def test_a_url_absent_from_references_is_skipped() -> None:
    rendered, telemetry, _ = render(f"The claim {Z} and")
    assert rendered == f"The claim {Z} and" + REFERENCES
    assert skips(telemetry) == {"skipped_not_in_references": 1}


def test_a_number_the_agent_wrote_is_never_changed() -> None:
    # The S96 conflict: the agent's [1] names another URL than its marker.
    references = f"\n\n## References\n[1] Beta — {URL_B}\n[2] Alpha — {URL_A}\n"
    body = f"Claim [1][cite:{URL_A}]."
    rendered, telemetry, _ = render(body, references)
    assert rendered == body + references
    assert skips(telemetry) == {"skipped_conflicting_number": 1}


def test_a_group_followed_by_a_link_opening_is_skipped() -> None:
    rendered, telemetry, _ = render(f"The claim {A} {B}(see) and")
    assert rendered == f"The claim {A} {B}(see) and" + REFERENCES
    assert skips(telemetry) == {"skipped_link_syntax": 2}


@pytest.mark.parametrize("references", [
    # The extractor resolves [1] to another spelling, so its number would not
    # merge into this citation.
    "\n\n## References\n[1] Alpha — https://A.example/x\n",
    # Two entries name different sources under one number: a collision.
    f"\n\n## References\n[1] Alpha — {URL_A}\n[1] Beta — {URL_B}\n",
])
def test_parser_disagreement_is_skipped(references: str) -> None:
    rendered, telemetry, _ = render(f"The claim {A} and", references)
    assert rendered == f"The claim {A} and" + references
    assert skips(telemetry) == {"skipped_parser_disagreement": 1}


def test_an_escaped_number_would_be_no_reader_number() -> None:
    rendered, telemetry, _ = render(f"The path C:\\ {A} and")
    assert reader_text(rendered) == "The path C:\\  and"
    assert skips(telemetry) == {"skipped_parser_disagreement": 1}


def test_code_spans_and_the_references_section_are_untouched() -> None:
    references = f"\n\n## References\n[1] Alpha — {URL_A} {A}\n[2] Beta — {URL_B}\n"
    body = f"Inline `code {A}` and a block:\n\n    indented {B}\n\nText."
    rendered, telemetry, keys = render(body, references)
    assert rendered == body + references
    assert (telemetry["rendered"], skips(telemetry), keys) == (0, {}, {})


def test_www_and_apex_urls_do_not_match() -> None:
    rendered, telemetry, _ = render("The claim [cite:https://www.a.example/x] and")
    assert reader_text(rendered) == "The claim  and"
    assert skips(telemetry) == {"skipped_not_in_references": 1}


def test_an_unsafe_url_is_left_alone_uncounted() -> None:
    url = "https://user:password@a.example/x"
    references = f"\n\n## References\n[1] Alpha — {url}\n"
    rendered, telemetry, _ = render(f"The claim [cite:{url}] and", references)
    assert rendered == f"The claim [cite:{url}] and" + references
    assert (telemetry["rendered"], skips(telemetry)) == (0, {})


def test_a_repeated_marker_of_one_citation_renders_once() -> None:
    rendered, telemetry, keys = render(f"The claim {A} {A} and")
    assert reader_text(rendered) == "The claim[1] and"
    assert telemetry["rendered"] == len(keys) == 1


def test_a_renderer_exception_returns_the_draft_unchanged(monkeypatch) -> None:
    draft = f"The claim {A} and" + REFERENCES
    real = fnode._parse_references_entries

    def boom(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("synthetic parser fault")

    monkeypatch.setattr(fnode, "_parse_references_entries", boom)
    rendered, telemetry, keys = fnode._render_reader_numbers(draft)
    assert (rendered, keys) == (draft, {})
    assert telemetry["error"] is True and telemetry["rendered"] == 0
    words = count_report_words(extract_citations(draft)[0])
    assert telemetry["words_before"] == telemetry["words_after"] == words


# --- fates ---------------------------------------------------------------------


def test_the_critique_reports_its_applied_edit_spans() -> None:
    draft = "Supported claim stays.\nThe cold-start penalty is exactly 9.4 seconds.\nEnd.\n"
    anchor = "The cold-start penalty is exactly 9.4 seconds."
    model = MagicMock(return_value=_mock_response(_annotations_raw(
        [{"anchor": anchor, "verdict": "unsupported", "note": "no source"}])))
    result = run_critique(draft, [], {"title": "t", "objective": "o"}, model=model)
    start = draft.index(anchor)
    assert result.gap_notes_inserted == 1
    assert result.applied_edit_spans == [(start, start + len(anchor))]
    assert result.output_markdown == draft.replace(anchor, GAP_NOTE_DEFAULT)


def flagged(url: str, start: int, end: int, number: int | None) -> CitationAnnotation:
    annotation = CitationAnnotation(start_index=start, end_index=end, url=url, live=True)
    annotation._extraction_mode = "primary"
    annotation._reader_number_rendered = number
    return annotation


def test_a_persisted_citation_without_its_number_is_unmapped() -> None:
    report = "The claim [1] holds. Another claim holds.\n\n## References\n[1] Alpha — " + URL_A
    mapped = flagged(URL_A, 0, report.index("[1]") + 3, 1)
    unmapped = flagged(URL_A, 21, report.index("holds.\n") + 5, 1)
    fates = fnode._reader_number_fates(
        kept=[mapped, unmapped], persisted=[mapped, unmapped], dropped=[], quarantined=[],
        applied_edit_spans=[], critique_input=report, report_content=report,
        expected_urls={URL_A},
    )
    assert fates == {"mapped": 1, "unmapped": 1, "quarantined": 0,
                     "lost_with_critique_edit": 0, "lost_at_rebase": 0}


def test_a_loss_outside_any_critique_edit_is_lost_at_rebase() -> None:
    kept = [flagged(URL_A, 0, 10, 1), flagged(URL_B, 20, 30, 2), flagged(URL_Z, 40, 50, None)]
    dropped = [{"annotation_index": index, "reason": "source_span_missing_from_persisted_segment"}
               for index in range(3)]
    fates = fnode._reader_number_fates(
        kept=kept, persisted=[], dropped=dropped, quarantined=[flagged(URL_Z, 0, 5, 3)],
        applied_edit_spans=[(25, 35)],
        critique_input="".join(chr(0x100 + i) for i in range(60)), report_content="",
        expected_urls=set(),
    )
    # The unflagged citation is in no fate, however it was lost.
    assert fates == {"mapped": 0, "unmapped": 0, "quarantined": 1,
                     "lost_with_critique_edit": 1, "lost_at_rebase": 1}


def test_a_loss_the_critique_caused_through_repeated_text_is_its_loss() -> None:
    # The rebase binds cited text in document order: once the edit removes the
    # first "Fact X holds [1]", the edited citation takes the second occurrence
    # and the untouched one is the citation dropped.
    text = "Fact X holds [1]. Other claim is here. Fact X holds [1]. End."
    end = text.index("[1]") + 3
    second = text.index("Fact X", 1)
    kept = [flagged(URL_A, 0, end, 1), flagged(URL_A, second, second + end, 1)]
    edit = (0, text.index(" Fact X", 1))
    persisted_text = GAP_NOTE_DEFAULT + text[edit[1]:]
    outcome = rebase_citation_annotations(
        reference_markdown=text, persisted_segment_markdown=persisted_text,
        persisted_markdown=persisted_text, annotations=kept,
    )
    assert [record["annotation_index"] for record in outcome.dropped] == [1]
    fates = fnode._reader_number_fates(
        kept=kept, persisted=outcome.annotations, dropped=outcome.dropped, quarantined=[],
        applied_edit_spans=[edit], critique_input=text, report_content=persisted_text,
        expected_urls={URL_A},
    )
    assert (fates["lost_with_critique_edit"], fates["lost_at_rebase"]) == (1, 0)


@pytest.mark.parametrize("body", [
    f"Intro.\n\n- {A} Alpha source\n- {B} Beta source\n",
    f"Intro.\n\n## {A} Alpha heading\n",
    f"Intro.\n\n1) {A} Alpha item\n",
    f"Intro.\n\n* {A} Alpha item\n",
    f"Intro.\n\n- [ ] {A} Alpha task\n",
    f"Intro.\n\n> - {A} Alpha quoted item\n",
    f"Intro.\n\n-{A} Alpha item\n",
], ids=["bullets", "heading", "ordered", "star", "task", "quoted list", "bullet glued"])
def test_a_group_after_line_leading_block_syntax_is_skipped(body: str) -> None:
    rendered, telemetry, _ = render(body)
    assert rendered == body + REFERENCES
    assert set(skips(telemetry)) == {"skipped_line_start"}


def test_a_list_item_whose_citation_follows_its_text_keeps_its_bullet() -> None:
    rendered, telemetry, _ = render(f"Intro.\n\n- Alpha {A} holds\n## Beta {B} heading\n")
    assert reader_text(rendered) == "Intro.\n\n- Alpha[1] holds\n## Beta[2] heading\n"
    assert telemetry["rendered"] == 2


def test_a_marker_ending_the_draft_inside_references_is_untouched() -> None:
    draft = f"Alpha holds {A} and Beta holds {B} too.\n\n## References\n[1] Alpha — {URL_A}\n" \
            f"[2] Beta — {URL_B} {B}"
    rendered, telemetry, _ = fnode._render_reader_numbers(draft)
    assert rendered.endswith(f"[2] Beta — {URL_B} {B}")
    assert reader_text(rendered) == "Alpha holds[1] and Beta holds[2] too."
    assert telemetry["rendered"] == 2


@pytest.mark.parametrize("body", [
    f"See https://a.example/x {A} for details.",
    f"(see https://a.example/x {A} more)",
    f"Read www.a.example/x{A}.",
])
def test_a_number_never_joins_a_bare_url(body: str) -> None:
    rendered, telemetry, _ = render(body)
    assert rendered == body + REFERENCES
    assert skips(telemetry) == {"skipped_link_syntax": 1}


URL_INVENTED = "https://invented.example/z"
URL_UNLISTED = "https://unlisted.example/q"
EDITED = "Beta claim was later removed by the critique"


def _synthesis() -> SynthesisResult:
    body = (
        f"Alpha claim stands on collected evidence [cite:{URL_A}] and more words here.\n"
        f"{EDITED} [cite:{URL_B}] and more words.\n"
        f"Gamma claim cites an invented page [cite:{URL_INVENTED}] and more words.\n"
        f"Delta claim ends in a partial group [cite:{URL_A}][cite:{URL_UNLISTED}] and more.\n"
    )
    references = (f"[1] Alpha — {URL_A}\n[2] Beta — {URL_B}\n[3] Invented — {URL_INVENTED}\n")
    return SynthesisResult(
        output_markdown=f"# Report\n\n{body}\n## References\n\n{references}",
        step_count=5, self_terminated=True, hit_max_steps=False, runtime_seconds=10.0,
        token_usage={"input": 100, "output": 100, "total": 200},
        tool_call_summary={"total_calls": 3},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0, step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=[{"url": url, "title": url, "snippet": "x"}
                           for url in (URL_A, URL_B, URL_UNLISTED)],
        error=None,
    )


async def _liveness(citations, sources_collected, budget_seconds=45.0, scope_admission=None):
    for annotation in citations:
        annotation.live = annotation.url != URL_INVENTED
    return list(citations), LivenessTelemetry(
        checked=len(citations), live=len(citations) - 1, dead=0, orphan=1, unknown=0,
        hallucinated_url_count=1, distinct_urls_checked=4,
    )


def _critique(draft_markdown, sources_collected, mission_context, *, citations=None,
              scope_admission=None) -> CritiqueResult:
    """Replace the Beta claim, number included, with a gap note."""

    start = draft_markdown.index(EDITED)
    end = draft_markdown.index(" and more words.", start)
    return CritiqueResult(
        output_markdown=draft_markdown[:start] + GAP_NOTE_DEFAULT + draft_markdown[end:],
        claim_count=4, gap_notes_inserted=1, runtime_seconds=0.0,
        token_usage={"input": 1, "output": 1, "total": 2}, token_usage_complete=True,
        annotations=[CritiqueAnnotation(anchor=draft_markdown[start:end], verdict="unsupported",
                                        citation_urls=(URL_B,), applied=True,
                                        reason="applied_gap_note")],
        applied_edit_spans=[(start, end)],
    )


def test_finalize_renders_numbers_and_counts_where_each_one_ended_up(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    captured: dict = {}
    state = {
        "mission_id": "S101-M02-FATES", "mission_objectives": ["c1"],
        # No deliverables: a coverage shortfall would start a recovery pass.
        "mission_context": {"title": "Fates", "objective": "o", "deliverables": []},
        "sources_found": [], "telemetry": {},
    }

    def _capture_protocol(protocol, **kwargs):
        captured["citations"] = list(protocol.citations)
        return str(tmp_path / "protocol.json")

    def _capture_report(content, **kwargs):
        captured["report"] = content
        return str(tmp_path / "report.md")

    async def run():
        with patch.object(fnode, "run_codeagent_synthesis", return_value=_synthesis()), \
                patch.object(fnode, "run_critique", side_effect=_critique), \
                patch.object(fnode, "check_citations_liveness", new=_liveness), \
                patch.object(fnode, "save_protocol", side_effect=_capture_protocol), \
                patch.object(fnode, "save_report", side_effect=_capture_report):
            return await fnode.codeagent_finalize_node(state)

    telemetry = asyncio.run(run())["synthesis_telemetry"]
    report = captured["report"]
    assert "collected evidence[1] and more words here." in report
    assert "partial group[1] and more." in report
    assert "[cite:" not in report and EDITED not in report
    rendering = telemetry["reader_number_rendering"]
    assert {key: value for key, value in rendering.items() if key != "fates"} == {
        "version": 1, "rendered": 4, "skipped_not_in_references": 1,
        "skipped_multiple_numbers": 0, "skipped_conflicting_number": 0,
        "skipped_link_syntax": 0, "skipped_parser_disagreement": 0,
        "skipped_sentence_boundary": 0, "skipped_line_start": 0,
        "words_before": rendering["words_before"], "words_after": rendering["words_before"],
        "draft_chars_before": rendering["draft_chars_before"],
        # Four glued groups: each adds a three-character number and drops a space.
        "draft_chars_after": rendering["draft_chars_before"] + 4 * (3 - 1), "error": False,
    }
    # Alpha and Delta map; the invented URL was numbered, then quarantined; the
    # Beta claim left with the critique's edit. The unlisted neighbour, which
    # now sits after Delta's [1], is kept but flagged in no fate.
    assert rendering["fates"] == {"mapped": 2, "unmapped": 0, "quarantined": 1,
                                  "lost_with_critique_edit": 1, "lost_at_rebase": 0}
    assert sum(rendering["fates"].values()) == rendering["rendered"]
    assert URL_UNLISTED in {c.url for c in captured["citations"]}
    assert telemetry["citation_consistency"]["unmapped_protocol_citations"] == 1
    # The pass record keeps its per-pass counts without fates.
    first_pass = telemetry["recovery"]["first_pass"]["reader_number_rendering"]
    assert first_pass == {key: value for key, value in rendering.items() if key != "fates"}


def test_a_fates_failure_is_recorded_and_never_costs_the_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))

    def boom(**kwargs):
        raise ValueError("Exceeds the limit (4300 digits) for integer string conversion")

    monkeypatch.setattr(fnode, "_reader_number_fates", boom)
    state = {
        "mission_id": "S101-M02-FATES-GUARD", "mission_objectives": ["c1"],
        "mission_context": {"title": "Fates", "objective": "o", "deliverables": []},
        "sources_found": [], "telemetry": {},
    }

    async def run():
        with patch.object(fnode, "run_codeagent_synthesis", return_value=_synthesis()), \
                patch.object(fnode, "check_citations_liveness", new=_liveness):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    rendering = result["synthesis_telemetry"]["reader_number_rendering"]
    assert (rendering["rendered"], rendering["fates"], rendering["fates_error"]) == (4, None, True)
    assert "collected evidence[1] and more words here." in Path(result["report_path"]).read_text()
