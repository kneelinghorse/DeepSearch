"""Truthful critique views and verdict provenance, without model-quality claims."""

from __future__ import annotations

import json

import pytest

from deepsearch.agent import critique
from deepsearch.agent.codeagent_finalize_node import _critique_telemetry_dict
from deepsearch.agent.outcome import evaluate_outcome_gate
from deepsearch.agent.quality_series import build_quality_record

CLAIM = "The schema is guaranteed in every response without exception."
URL = "https://example.org/structured-output"


def _raw(assessments: list, annotations: list | None = None, count: int = 0) -> str:
    return (
        critique.CRITIQUE_OUTPUT_DELIMITER
        + "\n"
        + json.dumps(
            {
                "claim_count": count,
                "annotations": annotations or [],
                "assessments": assessments,
            }
        )
    )


def _sources() -> list[dict]:
    return [
        {
            "url": URL,
            "body": "An incomplete response can violate the schema.",
            "content_quality": "readable_body",
            "body_truncated": False,
        }
    ]




@pytest.mark.parametrize("verdict", ["supported", "contradicted"])
def test_traceable_judgments_are_counted_without_an_accuracy_pass(verdict: str) -> None:
    raw = _raw(
        [
            {
                "anchor": CLAIM,
                "verdict": verdict,
                "evidence_ids": ["E1.1"],
                "evidence_quotes": {"E1.1": "An incomplete response can violate the schema."},
            }
        ]
    )
    result = critique.run_critique(CLAIM, _sources(), {}, model=lambda **kw: raw)
    assert result.output_markdown == CLAIM and result.gap_notes_inserted == 0
    assert result.assessment[verdict] == result.assessment["assessed"] == 1
    assert result.assessment["status"] == "partial"
    assert result.assessment["coverage_ratio"] is None
    assert result.assessment["accuracy_pass"] is None
    evidence = result.assessment["records"][0]["evidence"][0]
    assert evidence["quote_start"] == 0 and evidence["quote_end"] == len(_sources()[0]["body"])
    telemetry = {"critique_telemetry": _critique_telemetry_dict(result)}
    assert build_quality_record(telemetry, 100)["claim_assessment"]["assessed"] == 1
    gate = evaluate_outcome_gate(telemetry, report_word_count=20)
    assert "critique_assessment_partial" not in gate.quality_gate_failures
    assert build_quality_record(telemetry, 100)["verification"]["scope"] == "sampled"
    if verdict == "contradicted":
        assert "critique_contradicted_1" in gate.quality_gate_failures


@pytest.mark.parametrize(
    "item",
    [
        {
            "anchor": CLAIM,
            "verdict": "supported",
            "evidence_ids": ["invented"],
            "evidence_quotes": {},
        },
        {
            "anchor": CLAIM,
            "verdict": "contradicted",
            "evidence_ids": ["E1.1"],
            "evidence_quotes": {"E1.1": "Fabricated supporting quotation."},
        },
        {"anchor": CLAIM, "verdict": ["supported"]},
        {"anchor": "A claim absent from the report.", "verdict": "supported"},
    ],
)
def test_untraceable_judgments_are_unassessed_and_never_mutate_artifact(item: dict) -> None:
    result = critique.run_critique(
        CLAIM, _sources(), {}, model=lambda **kw: _raw([item], count=100)
    )
    assert result.error is None
    assert result.output_markdown == CLAIM
    assert result.assessment["assessed"] == 0 and result.assessment["unassessed"] == 1
    assert result.assessment["declared_claim_count"] == 100
    assert result.assessment["report_claim_count"] is None


def test_zero_claims_and_missing_assessment_are_explicitly_unassessed() -> None:
    result = critique.run_critique(CLAIM, _sources(), {}, model=lambda **kw: _raw([]))
    assert result.assessment["status"] == "unassessed" and result.assessment["assessed"] == 0
    assert result.assessment["unreported_claims"] == "unknown"
    telemetry = {"critique_telemetry": _critique_telemetry_dict(result)}
    assert (
        "critique_assessment_unassessed"
        in evaluate_outcome_gate(telemetry, report_word_count=20).quality_gate_failures
    )


def test_tl_eng_declared_context_is_not_deleted_for_lacking_public_verification() -> None:
    author_fact = "All workflows were disabled on 2026-09-15."
    public_claim = "The provider guarantees infinitely many free CI minutes."
    draft = author_fact + "\n" + public_claim
    context = {"background": author_fact}
    raw = _raw(
        [
            {
                "anchor": author_fact,
                "verdict": "declared_context",
                "evidence_ids": ["C1"],
                "evidence_quotes": {"C1": author_fact},
            }
        ],
        [
            {"anchor": author_fact, "verdict": "unsupported"},
            {"anchor": public_claim, "verdict": "unsupported"},
        ],
        count=2,
    )
    result = critique.run_critique(draft, _sources(), context, model=lambda **kw: raw)
    assert result.annotations[0].reason == "preserved_declared_context"
    assert result.annotations[1].applied  # Declared context does not exempt public assertions.
    assert author_fact in result.output_markdown
    assert result.assessment["declared_context"] == 1 and result.assessment["assessed"] == 0


def test_selected_passage_cannot_authorize_an_unsupported_deletion() -> None:
    sources = [{"url": URL, "body": "Schema statement.\n\n" * 4000}]
    result = critique.run_critique(
        CLAIM,
        sources,
        {},
        model=lambda **kw: _raw(
            [],
            [{"anchor": CLAIM, "verdict": "unsupported"}],
            count=1,
        ),
    )
    assert result.output_markdown == CLAIM
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.assessment["grounding_context_complete"] is False


def test_supported_in_annotation_slot_is_not_coerced_into_a_deletion() -> None:
    result = critique.run_critique(
        CLAIM,
        _sources(),
        {},
        model=lambda **kw: _raw(
            [],
            [{"anchor": CLAIM, "verdict": "supported"}],
            count=1,
        ),
    )
    assert result.output_markdown == CLAIM
    assert result.annotations[0].reason == "invalid_verdict"


def test_author_context_and_source_views_share_one_hard_context_budget() -> None:
    sources = [{"url": f"https://example.org/{i}", "body": "body " * 1000} for i in range(100)]
    telemetry: dict = {}
    messages = critique._build_critique_messages(
        "claim " * 60000,
        sources,
        {"background": "private context " * 3000},
        grounding_telemetry=telemetry,
    )
    assert len(messages[1]["content"]) <= critique.CRITIQUE_CONTEXT_MAX_CHARS
    assert telemetry["author_context_chars_included"] == 8000
    assert telemetry["author_context_chars_dropped"] > 0
    assert telemetry["draft_chars_dropped"] > 0
