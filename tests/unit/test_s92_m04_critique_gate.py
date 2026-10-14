"""S92 m04 critique-ratio denominator-floor outcome invariants."""

from __future__ import annotations

from typing import Any

import pytest

from deepsearch.agent.critique_policy import CRITIQUE_RATIO_MIN_APPLIED
from deepsearch.agent.outcome import (
    DELIVERY_QUALITY_DEGRADED,
    DELIVERY_QUALITY_FAILED,
    DELIVERY_QUALITY_USABLE_WITH_WARNING,
    GATE_REASON_CRITIQUE_ERROR,
    GATE_REASON_MAX_STEPS_NO_FINAL,
    GATE_REASON_OUTPUT_TRUNCATED,
    GATE_REASON_ZERO_CITATIONS,
    MissionOutcome,
    evaluate_outcome_gate,
)


def _telemetry(
    *,
    applied: int,
    claim_count: int,
    skipped: int = 0,
    critique_error: str | None = None,
    citations: int = 4,
    hit_max_steps: bool = False,
    self_terminated: bool = True,
    output_truncated: bool = False,
    max_steps_fallback_complete: bool = False,
    recovered_from_scratchpad: bool = False,
) -> dict[str, Any]:
    """Build complete quality telemetry without unrelated warning signals."""

    return {
        "hit_max_steps": hit_max_steps,
        "self_terminated": self_terminated,
        "output_truncated": output_truncated,
        "max_steps_fallback_complete": max_steps_fallback_complete,
        "recovered_from_scratchpad": recovered_from_scratchpad,
        "step_count": 8,
        "citation_telemetry": {
            "citations_extracted": citations,
            "distinct_urls_cited": citations,
        },
        "deliverable_coverage": {
            "shortfall_score": 0.0,
            "satisfied": ["required section"],
            "missing": [],
        },
        "liveness_telemetry": {
            "checked": citations,
            "live": citations,
            "dead": 0,
            "orphan": 0,
            "unknown": 0,
            "hallucinated_url_count": 0,
        },
        "critique_telemetry": {
            "assessment": {
                "version": 1, "status": "partial", "assessed": 1, "supported": 1,
                "contradicted": 0, "unassessed": 0,
            },
            "claim_count": claim_count,
            "gap_notes_inserted": applied,
            "error": critique_error,
            "annotation_summary": {
                "total": applied + skipped,
                "applied": applied,
                "skipped": skipped,
                "dropped": 0,
                "by_reason": {"applied_gap_note": applied} if applied else {},
                "reconciled": True,
            },
            "grounding": {
                "cited_dropped": 0,
                "draft_chars_dropped": 0,
                "sources_dropped": 0,
                "sources_truncated": 0,
                "liveness_chars_dropped": 0,
            },
        },
        "critique_skipped": bool(critique_error),
        "tool_outcomes": {
            "by_tool": {
                "web_search": {
                    "calls": 1,
                    "ok": 1,
                    "error": 0,
                    "results_total": citations,
                    "circuit_open": 0,
                    "by_category": {},
                }
            },
            "records": [],
            "records_truncated": 0,
        },
        "recovery_passes": 0,
        "recovery": {
            "passes": 0,
            "status": "not_triggered",
            "selected_artifact": "first_pass",
            "decision": {"triggered": False},
        },
        "retrieval_state": {
            "discovered": citations,
            "fetched_readable": citations,
            "fetch_attempts": citations,
            "probes": 0,
            "grounding_sources": citations,
        },
    }


def test_one_of_two_applied_annotations_warns_but_does_not_hard_fail() -> None:
    """A noisy 50% ratio below recovery's floor must remain reviewable."""

    assert CRITIQUE_RATIO_MIN_APPLIED == 3
    result = evaluate_outcome_gate(
        _telemetry(applied=1, skipped=1, claim_count=2),
        report_word_count=3_771,
    )

    assert result.outcome is MissionOutcome.COMPLETE
    assert result.routing_reason == ""
    assert result.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING
    assert result.quality_gate_failures == ["critique_applied_1"]


def test_r2_shaped_model_declared_ratio_degrades_but_cannot_destroy_report() -> None:
    """An unstable model-declared 8/9 ratio is advisory, not terminal proof."""

    result = evaluate_outcome_gate(
        _telemetry(applied=8, claim_count=9, citations=146),
        report_word_count=3_979,
    )

    assert result.outcome is MissionOutcome.COMPLETE
    assert result.routing_reason == ""
    assert result.delivery_quality == DELIVERY_QUALITY_DEGRADED
    assert result.quality_gate_failures == [
        "critique_applied_ratio_0.889_count_8", "critique_applied_8"
    ]


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"critique_error": "backend error"}, GATE_REASON_CRITIQUE_ERROR),
        ({"citations": 0}, GATE_REASON_ZERO_CITATIONS),
        ({"output_truncated": True}, GATE_REASON_OUTPUT_TRUNCATED),
    ],
)
def test_low_denominator_floor_does_not_mask_independent_catastrophes(
    overrides: dict[str, Any],
    expected_reason: str,
) -> None:
    """The floor narrows only ratio damage; independent hard gates stay loud."""

    result = evaluate_outcome_gate(
        _telemetry(applied=1, skipped=1, claim_count=2, **overrides),
        report_word_count=1_000,
    )

    assert result.outcome is MissionOutcome.VALIDATION_FAILED
    assert result.routing_reason == expected_reason
    assert result.delivery_quality == DELIVERY_QUALITY_FAILED


def test_valid_max_steps_fallback_warns_without_terminal_failure() -> None:
    """A cited, untruncated fallback is reviewable terminal evidence."""

    result = evaluate_outcome_gate(
        _telemetry(
            applied=0,
            claim_count=4,
            hit_max_steps=True,
            self_terminated=False,
            max_steps_fallback_complete=True,
        ),
        report_word_count=1_000,
    )

    assert result.outcome is MissionOutcome.COMPLETE
    assert result.routing_reason == ""
    assert result.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING
    assert result.quality_gate_failures == [GATE_REASON_MAX_STEPS_NO_FINAL]


@pytest.mark.parametrize(
    "telemetry_override",
    (
        {"max_steps_fallback_complete": False},
        {
            "max_steps_fallback_complete": True,
            "recovered_from_scratchpad": True,
        },
    ),
    ids=("finish_reason_unknown", "scratchpad_salvage"),
)
def test_unproven_cited_max_steps_output_remains_validation_failed(
    telemetry_override: dict[str, bool],
) -> None:
    """Citations alone cannot prove an emergency fallback is complete."""

    result = evaluate_outcome_gate(
        _telemetry(
            applied=0,
            claim_count=4,
            hit_max_steps=True,
            self_terminated=False,
            **telemetry_override,
        ),
        report_word_count=1_000,
    )

    assert result.outcome is MissionOutcome.VALIDATION_FAILED
    assert result.routing_reason == GATE_REASON_MAX_STEPS_NO_FINAL
    assert result.delivery_quality == DELIVERY_QUALITY_FAILED
