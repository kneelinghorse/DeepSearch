"""S102-m02: an assessment record keeps the critic's verdict, so clean means no finding.

A version 2 record keeps ``model_verdict`` beside the validated ``verdict``, and
the items dropped before becoming records are counted by their verdict. A
sampled claim the critic left unverified is a coverage fact, counted in
``quality_record.claim_assessment``, not a warning. A contradiction or an
internal inconsistency the critic reported warns even when its quote or anchor
failed validation or the item was dropped, and an unreadable verdict warns as
unknown. Version 1 records cannot show any of this, so they keep today's
``critique_unassessed_claims_N``. New assessment code falls back to the
version 1 builder rather than fail the run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent import critique
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_finalize_node import _critique_telemetry_dict
from deepsearch.agent.outcome import MissionOutcome, evaluate_outcome_gate
from deepsearch.agent.quality_series import build_quality_record
from tests.unit.test_codeagent_finalize_node import _state, _synthesis_with_sources
from tests.unit.test_s96_quality_signals import URL, healthy

QUOTE = "An incomplete response can violate the schema."
BODY = QUOTE + " The cache expires after ten minutes."
SUPPORTED = "The schema can be violated by an incomplete response."
LOST = "The cache expires after exactly ten minutes in every region."
CHOSEN = "The cache policy is documented for every deployment region."
RANGE = "Costs across both settings range from 0.18 to 1.49 overall."
OTHER = "Varied-cost games run from 0.24 to 1.49 instead."
REPEATED = "This sentence repeats verbatim."
LONG = "A long statement that keeps going " + "and going " * 210 + "until it ends."
DRAFT = "\n".join([
    "# Findings", "", SUPPORTED, LOST, CHOSEN, "", "## Ranges", "", RANGE, OTHER,
    REPEATED, REPEATED, LONG, "",
])
CRITIQUE_REASONS = ("critique_",)


def item(anchor: Any, verdict: Any, quote: str | None = QUOTE, **extra: Any) -> dict:
    row: dict[str, Any] = {"anchor": anchor, "verdict": verdict, "note": "reason"}
    if anchor is None:
        del row["anchor"]
    if quote is not None:
        row.update(evidence_ids=["E1.1"], evidence_quotes={"E1.1": quote})
    return {**row, **extra}


def raw_output(items: list[dict]) -> str:
    return critique.CRITIQUE_OUTPUT_DELIMITER + "\n" + json.dumps(
        {"claim_count": len(items), "annotations": [], "assessments": items})


def assess(items: list[dict], *, draft: str = DRAFT) -> dict[str, Any]:
    sources = [{"url": URL, "body": BODY, "content_quality": "readable_body",
                "body_truncated": False}]
    result = critique.run_critique(draft, sources, {}, model=lambda **kw: raw_output(items))
    assert result.error is None and result.output_markdown == draft
    return result.assessment


def gate(assessment: dict[str, Any]) -> Any:
    telemetry = healthy()
    telemetry["critique_telemetry"]["assessment"] = assessment
    return evaluate_outcome_gate(telemetry, report_word_count=500)


def critique_reasons(assessment: dict[str, Any]) -> list[str]:
    return [r for r in gate(assessment).quality_gate_failures if r.startswith(CRITIQUE_REASONS)]


def claims(assessment: dict[str, Any]) -> dict[str, Any]:
    return build_quality_record({"critique_telemetry": {"assessment": assessment}}, 500)[
        "claim_assessment"]


def record(assessment: dict[str, Any], anchor: str) -> dict[str, Any]:
    (row,) = [r for r in assessment["records"] if r["anchor"] == anchor]
    return row


def test_unverified_claims_that_are_not_findings_leave_the_run_clean() -> None:
    assessment = assess([
        item(SUPPORTED, "supported"),
        item(LOST, "supported", quote="A quotation the passage does not hold."),
        item(CHOSEN, "unassessed", quote=None),
        item(RANGE, "declared_context", quote=None),
    ])
    assert assessment["version"] == 2
    assert [(r["verdict"], r["model_verdict"], r["reason"]) for r in assessment["records"]] == [
        ("supported", "supported", "traceable_model_judgment"),
        ("unassessed", "supported", "no_visible_quoted_evidence"),
        ("unassessed", "unassessed", "located"),
        ("unassessed", "declared_context", "no_visible_quoted_evidence"),
    ]
    result = gate(assessment)
    assert (result.delivery_quality, result.quality_gate_failures) == ("clean", [])
    # The coverage fact stays in the quality record.
    assert claims(assessment) == {
        "version": 2, "status": "partial", "assessed": 1, "supported": 1, "contradicted": 0,
        "unassessed": 3, "declared_context": 0, "report_claim_count": None,
        "coverage_ratio": None, "accuracy_pass": None,
        "unverified_contradicted": 0, "inconsistent": 0, "unknown_verdicts": 0,
    }
    record_v2 = build_quality_record(
        {"critique_telemetry": {"assessment": assessment}}, 500)
    assert record_v2["verification"]["scope"] == "sampled"
    assert record_v2["schema_version"] == 6


def test_a_demoted_contradiction_warns_without_degrading() -> None:
    items = [item(SUPPORTED, "supported"),
             item(LOST, "contradicted", quote="A quotation the passage does not hold.")]
    assessment = assess(items)
    assert (record(assessment, LOST)["verdict"], record(assessment, LOST)["model_verdict"]) == (
        "unassessed", "contradicted")
    result = gate(assessment)
    assert critique_reasons(assessment) == ["critique_unverified_contradiction_1"]
    assert result.delivery_quality == "usable_with_warning"
    # Decision 1001's pattern reads verified contradictions only.
    assert assessment["contradicted"] == 0
    # Today's builder stores the same item as an unassessed claim and nothing more.
    legacy = critique._assessment_record_v1(
        raw_output(items), DRAFT, _visibility(), 0)
    assert (legacy["contradicted"], legacy["unassessed"]) == (0, 1)
    assert critique_reasons(legacy) == ["critique_unassessed_claims_1"]


@pytest.mark.parametrize("dropped", [
    item(SUPPORTED, "contradicted"),  # repeats the first item's anchor
    item(None, "contradicted"),
])
def test_a_contradiction_dropped_before_it_became_a_record_still_warns(dropped: dict) -> None:
    items = [item(SUPPORTED, "supported"), dropped]
    assessment = assess(items)
    assert (assessment["emitted"], assessment["reported"], assessment["dropped"]) == (2, 1, 1)
    assert assessment["dropped_by_model_verdict"] == {"contradicted": 1}
    assert critique_reasons(assessment) == ["critique_unverified_contradiction_1"]
    # Today's builder ends with contradicted 0 and no warning at all.
    legacy = critique._assessment_record_v1(raw_output(items), DRAFT, _visibility(), 0)
    assert (legacy["contradicted"], legacy["dropped"]) == (0, 1)
    assert critique_reasons(legacy) == []


def test_a_verified_inconsistency_warns_and_is_not_assessed() -> None:
    assessment = assess([
        item(SUPPORTED, "supported"),
        item(RANGE, "inconsistent", quote=None, conflict_anchor=OTHER),
    ])
    row = record(assessment, RANGE)
    assert (row["verdict"], row["model_verdict"], row["reason"], row["conflict_anchor"]) == (
        "inconsistent", "inconsistent", "traceable_internal_conflict", OTHER)
    assert "quote_failure" not in row
    assert (assessment["assessed"], assessment["inconsistent"], assessment["unassessed"]) == (
        1, 1, 0)
    assert critique_reasons(assessment) == ["critique_internal_inconsistency_1"]


@pytest.mark.parametrize(("conflict_anchor", "reason"), [
    ("A statement the draft never makes.", "conflict_anchor_not_found"),
    (None, "conflict_anchor_not_found"),
    (REPEATED, "conflict_anchor_ambiguous"),
    ("0.24", "conflict_anchor_too_short"),
    (LONG + " And more.", "conflict_anchor_too_long"),
    ("## Ranges", "conflict_anchor_structure_overlap"),
    ("range from 0.18 to 1.49", "conflict_anchor_overlaps_anchor"),
])
def test_an_inconsistency_whose_other_statement_fails_still_warns(
    conflict_anchor: str | None, reason: str,
) -> None:
    extra = {} if conflict_anchor is None else {"conflict_anchor": conflict_anchor}
    assessment = assess([item(SUPPORTED, "supported"),
                         item(RANGE, "inconsistent", quote=None, **extra)])
    row = record(assessment, RANGE)
    assert (row["verdict"], row["model_verdict"], row["reason"]) == (
        "unassessed", "inconsistent", reason)
    assert "quote_failure" not in row
    assert critique_reasons(assessment) == ["critique_internal_inconsistency_1"]


def test_an_inconsistency_whose_other_statement_is_not_visible_still_warns() -> None:
    visibility = _visibility(draft_chars_visible=DRAFT.index(OTHER) + 4)
    raw = raw_output([item(SUPPORTED, "supported"),
                      item(RANGE, "inconsistent", quote=None, conflict_anchor=OTHER)])
    assessment = critique._assessment_record(raw, DRAFT, visibility, 0)
    assert record(assessment, RANGE)["reason"] == "conflict_anchor_not_visible"
    assert critique_reasons(assessment) == ["critique_internal_inconsistency_1"]


@pytest.mark.parametrize("unreadable", [
    item(LOST, "plausible"),
    item(LOST, ["supported"]),
    item(LOST, None, quote=None),
])
def test_an_unreadable_verdict_is_an_unknown_verdict(unreadable: dict) -> None:
    assessment = assess([item(SUPPORTED, "supported"), unreadable])
    row = record(assessment, LOST)
    assert (row["verdict"], row["model_verdict"], row["reason"]) == (
        "unassessed", "invalid", "invalid_verdict")
    assert critique_reasons(assessment) == ["critique_unknown_verdicts_1"]


def test_an_unreadable_dropped_verdict_is_unknown() -> None:
    assessment = assess([item(SUPPORTED, "supported"), item(SUPPORTED, "maybe"), "garbage"])
    assert assessment["dropped_by_model_verdict"] == {"invalid": 2}
    assert critique_reasons(assessment) == ["critique_unknown_verdicts_2"]


def _version_2(**fields: Any) -> dict[str, Any]:
    supported = {"anchor": SUPPORTED, "verdict": "supported", "model_verdict": "supported",
                 "reason": "traceable_model_judgment", "evidence": [], "note": ""}
    return {"version": 2, "status": "partial", "assessed": 1, "supported": 1,
            "contradicted": 0, "inconsistent": 0, "unassessed": 0, "declared_context": 0,
            "reported": 1, "emitted": 1, "dropped": 0, "dropped_by_model_verdict": {},
            "records": [supported], **fields}


def test_a_record_without_its_model_verdict_is_unknown() -> None:
    chosen = {"anchor": CHOSEN, "verdict": "unassessed", "reason": "located", "evidence": []}
    assessment = _version_2(unassessed=1, reported=2, emitted=2,
                            records=[*_version_2()["records"], chosen])
    assert claims(assessment)["unknown_verdicts"] == 1
    assert critique_reasons(assessment) == ["critique_unknown_verdicts_1"]


@pytest.mark.parametrize(("fields", "provenance"), [
    # Missing records: every unassessed record is unknown.
    ({"records": None, "unassessed": 2, "reported": 3, "emitted": 3}, (0, 0, 2)),
    # Missing drop counts: every dropped item is unknown.
    ({"dropped_by_model_verdict": None, "emitted": 3, "dropped": 2}, (0, 0, 2)),
    # Without emitted, the drop total is the retained dropped field.
    ({"emitted": None, "dropped": 2, "dropped_by_model_verdict": {"contradicted": 1}},
     (1, 0, 1)),
    ({"emitted": 3, "dropped": 2, "dropped_by_model_verdict": {"inconsistent": 2}},
     (0, 2, 0)),
    # Malformed input counts as unknown and never as safe.
    ({"records": ["garbage"], "unassessed": 1}, (0, 0, 1)),
    ({"records": "garbage", "unassessed": 1}, (0, 0, 1)),
    ({"records": [{"verdict": "unassessed", "model_verdict": ["unassessed"]}],
      "unassessed": 1}, (0, 0, 1)),
    ({"emitted": 3, "dropped": 2, "dropped_by_model_verdict": {"contradicted": "2"}},
     (0, 0, 2)),
    ({"emitted": 3, "dropped": 2, "dropped_by_model_verdict": {"contradicted": True}},
     (0, 0, 2)),
    ({"emitted": 3, "dropped": 2, "dropped_by_model_verdict": [["contradicted", 2]]},
     (0, 0, 2)),
    ({"emitted": "3", "reported": None, "dropped": "x"}, (0, 0, 0)),
    ({"emitted": None, "dropped": float("inf")}, (0, 0, 0)),
    # A verified inconsistency survives malformed records: the builder's own
    # count when the records are unreadable, its verdict when its model verdict
    # is missing, and an unknown verdict when the record itself is unreadable.
    ({"records": None, "inconsistent": 1}, (0, 1, 0)),
    ({"records": [{"verdict": "inconsistent", "reason": "traceable_internal_conflict"}],
      "inconsistent": 1}, (0, 1, 0)),
    ({"records": ["garbage"], "inconsistent": 1}, (0, 0, 1)),
    ({"records": [{"model_verdict": "inconsistent"}], "inconsistent": 1}, (0, 0, 1)),
])
def test_missing_or_malformed_provenance_is_unknown_and_never_raises(
    fields: dict[str, Any], provenance: tuple[int, int, int],
) -> None:
    assessment = _version_2(**fields)
    found = claims(assessment)
    assert (found["unverified_contradicted"], found["inconsistent"],
            found["unknown_verdicts"]) == provenance
    gate(assessment)  # never raises


def test_a_heading_overlap_is_a_structure_overlap_that_keeps_its_verdict() -> None:
    anchor = "## Ranges\n\n" + RANGE
    assessment = assess([item(SUPPORTED, "supported"), item(anchor, "supported")])
    row = record(assessment, anchor)
    assert (row["verdict"], row["model_verdict"], row["reason"]) == (
        "unassessed", "supported", "structure_overlap")
    # Version 1 kept the truthy "located", indistinguishable from the critic's choice.
    legacy = critique._assessment_record_v1(
        raw_output([item(SUPPORTED, "supported"), item(anchor, "supported")]), DRAFT,
        _visibility(), 0)
    assert record(legacy, anchor)["reason"] == "located"
    assert critique_reasons(assessment) == []


def test_no_assessed_claim_still_warns() -> None:
    assessment = assess([item(CHOSEN, "unassessed", quote=None)])
    assert critique_reasons(assessment) == ["critique_assessment_unassessed"]


def test_every_emitted_item_becomes_a_record_or_a_counted_drop() -> None:
    items = [item(SUPPORTED, "supported"), item(SUPPORTED, "contradicted"), 7,
             *(item(f"{CHOSEN} {n}", "unassessed", quote=None) for n in range(199)),
             item(OTHER, "inconsistent", quote=None, conflict_anchor=RANGE)]
    assessment = assess(items)
    assert assessment["emitted"] == len(items) == 203
    assert assessment["reported"] == len(assessment["records"]) == 198
    assert assessment["reported"] + assessment["dropped"] == assessment["emitted"]
    # Past the cap of 200 items, the last three are dropped and still counted.
    assert assessment["dropped_by_model_verdict"] == {
        "contradicted": 1, "invalid": 1, "unassessed": 2, "inconsistent": 1}
    assert critique_reasons(assessment) == [
        "critique_unknown_verdicts_1", "critique_unverified_contradiction_1",
        "critique_internal_inconsistency_1"]


def test_a_builder_fault_records_version_1_and_warns_without_failing_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected")

    monkeypatch.setattr(critique, "_assessment_record_v2", broken)
    sources = [{"url": URL, "body": BODY, "content_quality": "readable_body",
                "body_truncated": False}]
    items = [item(SUPPORTED, "supported"), item(LOST, "unassessed", quote=None)]
    result = critique.run_critique(DRAFT, sources, {}, model=lambda **kw: raw_output(items))
    assert result.error is None
    assert (result.assessment["version"], result.assessment["provenance_error"]) == (1, True)
    telemetry = healthy()
    telemetry["critique_telemetry"] = _critique_telemetry_dict(result)
    outcome = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert outcome.outcome is MissionOutcome.COMPLETE
    assert outcome.delivery_quality == "usable_with_warning"
    assert [r for r in outcome.quality_gate_failures if r.startswith(CRITIQUE_REASONS)] == [
        "critique_unassessed_claims_1", "critique_assessment_provenance_error"]
    assert claims(result.assessment)["provenance_error"] is True


def test_declared_context_anchors_are_committed_only_when_the_builder_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fact = "All workflows were disabled on 2026-09-15."
    draft = DRAFT + fact + "\n"
    items = [item(fact, "declared_context", quote=None, evidence_ids=["C1"],
                  evidence_quotes={"C1": fact}),
             item(RANGE, "inconsistent", quote=None, conflict_anchor=OTHER)]

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected")

    monkeypatch.setattr(critique, "_conflict_anchor_reason", broken)
    visibility = _visibility(author_context=json.dumps({"background": fact}))
    with pytest.raises(RuntimeError):
        critique._assessment_record_v2(raw_output(items), draft, visibility, 0)
    assert visibility.declared_context_anchors == set()
    # The fallback sees exactly today's state and records today's anchors.
    fallback = critique._assessment_record(raw_output(items), draft, visibility, 0)
    today = _visibility(author_context=json.dumps({"background": fact}))
    critique._assessment_record_v1(raw_output(items), draft, today, 0)
    assert fallback["provenance_error"] is True
    assert visibility.declared_context_anchors == today.declared_context_anchors == {fact}
    monkeypatch.undo()
    success = _visibility(author_context=json.dumps({"background": fact}))
    critique._assessment_record(raw_output(items), draft, success, 0)
    assert success.declared_context_anchors == {fact}


@pytest.mark.parametrize(("assessment", "reasons"), [
    # A version 1 record set keeps today's warning whatever its reasons.
    ({"version": 1, "status": "partial", "assessed": 1, "supported": 1, "contradicted": 0,
      "unassessed": 2, "records": [
          {"verdict": "unassessed", "reason": "located"},
          {"verdict": "unassessed", "reason": "no_visible_quoted_evidence"}]},
     ["critique_unassessed_claims_2"]),
    # Counts without records keep it.
    ({"version": 1, "status": "partial", "assessed": 1, "supported": 1, "contradicted": 0,
      "unassessed": 3}, ["critique_unassessed_claims_3"]),
    ({"version": 1, "status": "partial", "assessed": 1, "supported": 1, "contradicted": 0,
      "unassessed": 0}, []),
    (None, ["critique_assessment_unknown"]),
])
def test_version_1_and_missing_assessments_keep_todays_rule(
    assessment: dict[str, Any] | None, reasons: list[str],
) -> None:
    telemetry = healthy()
    if assessment is None:
        del telemetry["critique_telemetry"]["assessment"]
    else:
        telemetry["critique_telemetry"]["assessment"] = assessment
    result = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert [r for r in result.quality_gate_failures if r.startswith(CRITIQUE_REASONS)] == reasons


def test_finalize_reaches_clean_when_the_critic_only_left_a_claim_unassessed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    markdown = (f"# Report\n\n## Findings\n\n{QUOTE} [1] [cite:{URL}]\n\n{CHOSEN} [1] "
                f"[cite:{URL}]\n\n## References\n\n[1] Schema source — {URL}\n")
    sources = [{"url": URL, "title": "Schema source", "body": BODY,
                "body_source": "source_fetch", "content_quality": "readable_body",
                "body_truncated": False}]
    synthesis = _synthesis_with_sources(sources=sources, markdown=markdown)
    synthesis.tool_outcomes = {"by_tool": {}, "records": [], "records_truncated": 0}
    items = [item(QUOTE, "supported"), item(CHOSEN, "unassessed", quote=None)]

    def run_critique(*args: Any, **kwargs: Any) -> Any:
        return real_critique(*args, **{**kwargs, "model": lambda **kw: raw_output(items)})

    async def keep_live(citations: list, _sources: Any, budget_seconds: float = 45.0, *,
                        scope_admission: Any = None) -> Any:
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=len(citations), live=len(citations), distinct_urls_checked=1)

    real_critique = critique.run_critique

    async def run() -> dict[str, Any]:
        with patch.object(fnode, "run_codeagent_synthesis", return_value=synthesis), \
                patch.object(fnode, "run_critique", side_effect=run_critique), \
                patch.object(fnode, "check_citations_liveness", new=keep_live):
            return await fnode.codeagent_finalize_node(_state(mission_id="CLEAN-SAMPLE"))

    result = asyncio.run(run())
    assert (result["delivery_quality"], result["quality_gate_failures"]) == ("clean", [])
    telemetry = result["synthesis_telemetry"]
    assessment = telemetry["critique_telemetry"]["assessment"]
    assert (assessment["version"], assessment["assessed"], assessment["unassessed"]) == (2, 1, 1)
    claim_assessment = build_quality_record(telemetry, telemetry["report_word_count"])[
        "claim_assessment"]
    assert {k: claim_assessment[k] for k in (
        "supported", "unassessed", "unverified_contradicted", "inconsistent",
        "unknown_verdicts")} == {
        "supported": 1, "unassessed": 1, "unverified_contradicted": 0, "inconsistent": 0,
        "unknown_verdicts": 0}


def _visibility(**fields: Any) -> Any:
    visibility = critique._GroundingVisibility(**fields)
    visibility.passages.append({
        "evidence_id": "E1.1", "source_id": "s", "url": URL, "body_sha256": "h",
        "start": 0, "end": len(BODY), "body_chars": len(BODY), "kind": "source_passage"})
    visibility.passage_text["E1.1"] = BODY
    if visibility.author_context:
        visibility.passages.append({"evidence_id": "C1", "kind": "declared_context",
                                    "chars": len(visibility.author_context),
                                    "truncated": False})
    if visibility.draft_chars_visible is None:
        visibility.draft_chars_visible = len(DRAFT) + 100
    return visibility
