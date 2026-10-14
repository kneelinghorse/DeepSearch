"""S87 m04 quality truth, calibrated delivery, and CAT reconciliation."""

from __future__ import annotations

from typing import Any

import pytest

from deepsearch.agent.outcome import (
    CRITIQUE_DEGRADED_RATIO,
    CRITIQUE_FAILED_RATIO,
    DELIVERY_QUALITY_CLEAN,
    DELIVERY_QUALITY_DEGRADED,
    DELIVERY_QUALITY_FAILED,
    DELIVERY_QUALITY_USABLE_WITH_WARNING,
    MissionOutcome,
    evaluate_outcome_gate,
)
from deepsearch.agent.quality_series import (
    QUALITY_RECORD_SCHEMA_VERSION,
    build_quality_record,
)
from deepsearch.eval import benchmark as benchmark_mod

# Real S87 CAT artifacts (provenance: tests/fixtures/PROVENANCE.md).


def _modern_telemetry(
    *,
    applied: int = 0,
    claim_count: int = 20,
    critique_error: str | None = None,
    critique_skipped: bool = False,
    liveness_unknown: int = 0,
    retrieval_errors: dict[str, int] | None = None,
    recovery_outcome: str = "not_triggered",
    recovery_passes: int = 0,
    context_truncated: bool = False,
) -> dict[str, Any]:
    skipped_annotations = 0
    critique_summary = {
        "total": applied + skipped_annotations,
        "applied": applied,
        "skipped": skipped_annotations,
        "dropped": 0,
        "by_reason": {"applied_gap_note": applied} if applied else {},
        "reconciled": True,
    }
    by_tool = {
        tool: {
            "calls": max(1, count),
            "ok": 0,
            "error": count,
            "results_total": 0,
            "circuit_open": 0,
            "by_category": {"backend": count},
        }
        for tool, count in (retrieval_errors or {}).items()
    }
    if not by_tool:
        by_tool = {
            "web_search": {
                "calls": 1,
                "ok": 1,
                "error": 0,
                "results_total": 3,
                "circuit_open": 0,
                "by_category": {},
            }
        }
    grounding = {
        "draft_chars_dropped": 10 if context_truncated else 0,
        "sources_dropped": 0,
        "sources_truncated": 0,
        "liveness_chars_dropped": 0,
    }
    return {
        "hit_max_steps": False,
        "self_terminated": True,
        "step_count": 8,
        "citation_telemetry": {
            "citations_extracted": 3,
            "distinct_urls_cited": 3,
        },
        "deliverable_coverage": {
            "shortfall_score": 0.0,
            "satisfied": ["required section"],
            "missing": [],
        },
        "liveness_telemetry": {
            "checked": 3,
            "live": 3 - liveness_unknown,
            "dead": 0,
            "orphan": 0,
            "unknown": liveness_unknown,
            "hallucinated_url_count": 0,
        },
        "critique_telemetry": {
            "claim_count": claim_count,
            "gap_notes_inserted": applied,
            "error": critique_error,
            "annotation_summary": critique_summary,
            "grounding": grounding,
        },
        "critique_skipped": critique_skipped,
        "tool_outcomes": {
            "by_tool": by_tool,
            "records": [],
            "records_truncated": 0,
        },
        "recovery_passes": recovery_passes,
        "recovery": {
            "passes": recovery_passes,
            "status": recovery_outcome,
            "selected_artifact": (
                "recovery" if recovery_outcome == "completed" else "first_pass"
            ),
            "decision": {"triggered": recovery_outcome != "not_triggered"},
        },
        "retrieval_state": {
            "discovered": 3,
            "fetched_readable": 2,
            "fetch_attempts": 2,
            "probes": 1,
            "grounding_sources": 3,
        },
    }


def test_quality_record_carries_all_reconciled_quality_facts() -> None:
    telemetry = _modern_telemetry(
        applied=3,
        liveness_unknown=1,
        retrieval_errors={"source_fetch": 2},
        recovery_outcome="completed",
        recovery_passes=1,
        context_truncated=True,
    )
    record = build_quality_record(telemetry, 1200)

    # v3 (S90): context_truncation gained critique_cited_dropped +
    # evidence_loss_occurred. Additive — every v2 field kept its meaning.
    assert record["schema_version"] == QUALITY_RECORD_SCHEMA_VERSION == 5
    assert record["critique"] == {
        "status": "ok",
        "applied": 3,
        "skipped": 0,
        "dropped": 0,
        "errors": 0,
        "pass_skipped": 0,
        "claim_count": 20,
        "applied_ratio": 0.15,
        "legacy_gap_notes_inserted": 3,
    }
    assert record["liveness"]["unknown"] == 1
    assert record["retrieval_failures"] == {
        "available": True,
        "total": 2,
        "by_tool": {"source_fetch": 2},
    }
    assert record["recovery"] == {
        "passes": 1,
        "outcome": "completed",
        "selected_artifact": "recovery",
        "triggered": True,
    }
    assert record["context_truncation"]["occurred"] is True
    assert record["retrieval_state"] == {
        "available": True,
        "discovered": 3,
        "fetched_readable": 2,
        "grounding_sources": 3,
        "fetch_attempts": 2,
        "probes": 1,
    }


def test_calibrated_delivery_tiers_use_exact_applied_ratio() -> None:
    assert CRITIQUE_DEGRADED_RATIO == 0.20
    assert CRITIQUE_FAILED_RATIO == 0.50

    assessed = _modern_telemetry()
    assessed["critique_telemetry"]["assessment"] = {
        "version": 1, "status": "partial", "assessed": 1, "supported": 1,
        "contradicted": 0, "unassessed": 0,
    }
    clean = evaluate_outcome_gate(assessed, report_word_count=1000)
    warning = evaluate_outcome_gate(
        _modern_telemetry(applied=3, claim_count=20),
        report_word_count=1000,
    )
    degraded = evaluate_outcome_gate(
        _modern_telemetry(applied=4, claim_count=20),
        report_word_count=1000,
    )
    high_ratio = evaluate_outcome_gate(
        _modern_telemetry(applied=10, claim_count=20),
        report_word_count=1000,
    )

    assert clean.delivery_quality == DELIVERY_QUALITY_CLEAN
    assert warning.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING
    assert degraded.delivery_quality == DELIVERY_QUALITY_DEGRADED
    assert high_ratio.delivery_quality == DELIVERY_QUALITY_DEGRADED
    assert high_ratio.outcome is MissionOutcome.COMPLETE
    assert "critique_applied_ratio_0.500_count_10" in high_ratio.quality_gate_failures
    assert "critique_assessment_unknown" in high_ratio.quality_gate_failures


@pytest.mark.parametrize(
    ("telemetry", "expected"),
    [
        (
            _modern_telemetry(critique_error="backend error", critique_skipped=True),
            DELIVERY_QUALITY_FAILED,
        ),
        (_modern_telemetry(liveness_unknown=1), DELIVERY_QUALITY_USABLE_WITH_WARNING),
        (
            _modern_telemetry(retrieval_errors={"web_search": 1}),
            DELIVERY_QUALITY_USABLE_WITH_WARNING,
        ),
        (
            _modern_telemetry(recovery_outcome="completed", recovery_passes=1),
            DELIVERY_QUALITY_USABLE_WITH_WARNING,
        ),
        (
            _modern_telemetry(recovery_outcome="failed", recovery_passes=1),
            DELIVERY_QUALITY_DEGRADED,
        ),
        (
            _modern_telemetry(context_truncated=True),
            DELIVERY_QUALITY_USABLE_WITH_WARNING,
        ),
    ],
)
def test_gate_consumes_each_quality_truth_surface(
    telemetry: dict[str, Any],
    expected: str,
) -> None:
    result = evaluate_outcome_gate(telemetry, report_word_count=1000)
    assert result.delivery_quality == expected
    assert result.quality_gate_failures


def _protocol_with_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "mission_id": "S87-M04",
        "citations": [{"url": "https://primary.example", "live": True}],
        "report_metadata": {
            "references": 1,
            "sources_collected": 1,
            "forensic": {
                "citation_telemetry": {"citations_extracted": 1},
                "quality_record": record,
            },
        },
        "research_statement": {"success_metrics": ["supported result"]},
    }


def test_benchmark_t2_uses_exact_damage_not_declared_unavailability() -> None:
    record = build_quality_record(_modern_telemetry(applied=2), 1000)
    body = (
        "# Report\n\nSupported result.\n"
        + "\n".join(
            "[Data unavailable in primary sources]" for _ in range(20)
        )
    )
    report = benchmark_mod.benchmark("S87-M04", _protocol_with_record(record), body)

    assert report.critique_gap_count == 2
    assert report.declared_unavailable_count == 20
    assert report.per_mission_thresholds["structured_critique_gaps_lt_10"][
        "pass"
    ] is True


@pytest.mark.parametrize("status", ["unknown", "error", "legacy_unstructured"])
def test_benchmark_t2_fails_closed_when_exact_damage_is_missing(status: str) -> None:
    record = build_quality_record(_modern_telemetry(), 1000)
    record["critique"]["status"] = status
    record["critique"]["applied"] = None
    report = benchmark_mod.benchmark(
        "S87-M04",
        _protocol_with_record(record),
        "# Report\n\nSupported result.",
    )

    assert report.critique_gap_count is None
    threshold = report.per_mission_thresholds["structured_critique_gaps_lt_10"]
    assert threshold["pass"] is False
    assert threshold["value"] is None
