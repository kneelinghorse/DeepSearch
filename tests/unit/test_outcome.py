"""Unit tests for final mission outcome helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from deepsearch.agent.outcome import (
    MissionOutcome,
    map_reason_to_outcome,
    resolve_final_outcome_payload,
)


def test_map_reason_to_outcome_treats_structural_validation_failure_as_complete() -> None:
    """Structural validation failures should ship as completed reports."""

    outcome = map_reason_to_outcome(
        "structural_validation_failed",
        {"passed": True, "failed_gates": []},
    )

    assert outcome is MissionOutcome.COMPLETE


def test_map_reason_to_outcome_treats_coverage_failure_as_complete() -> None:
    """Coverage failures should remain telemetry, not terminal outcomes."""

    outcome = map_reason_to_outcome(
        "max_loops_reached_coverage_unmet",
        {"passed": False, "failed_gates": ["entity"]},
    )

    assert outcome is MissionOutcome.COMPLETE


def test_map_reason_to_outcome_treats_circuit_breaker_as_blocked() -> None:
    """Circuit breakers are still operational blockers."""

    outcome = map_reason_to_outcome(
        "circuit_breaker_active",
        {"passed": False, "failed_gates": ["entity"]},
    )

    assert outcome is MissionOutcome.BLOCKED


def test_resolve_final_outcome_payload_includes_structural_validation_failures() -> None:
    """Final outcome payload should preserve structural validator failures."""

    payload = resolve_final_outcome_payload(
        {
            "routing_reason": "structural_validation_failed",
            "coverage_gates": {"passed": True, "failed_gates": []},
            "research_loop_count": 2,
            "synthesis_telemetry": {
                "structural_validation_failed_schemas": [
                    "classification_matrix",
                    "recommendation_section",
                ]
            },
        },
        completed_at=datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert payload["outcome"] == "complete"
    assert payload["quality_gate_failures"] == ["synthesis_quality"]
    assert payload["structural_validation_failures"] == [
        "classification_matrix",
        "recommendation_section",
    ]


# ---------------------------------------------------------------------------
# S41.2 — full ValidationReport surfacing in final_outcome
# ---------------------------------------------------------------------------


def test_resolve_final_outcome_payload_surfaces_full_validation_report() -> None:
    """S41.2 — the full validator list (passing + failing) must appear in
    ``final_outcome.validation_report`` so post-run audit can distinguish
    "validator fired and passed" from "validator didn't run on this build"
    without shell access to Railway.
    """

    full_report = {
        "passed": False,
        "failed_schema_ids": ["deliverable-schema-1"],
        "results": [
            {
                "validator_id": "validate_table_structure",
                "schema_id": "deliverable-schema-1",
                "passed": True,
                "detail": "Matched table with 6 rows and required columns.",
                "remediation_hint": "",
            },
            {
                "validator_id": "validate_citation_title_coherence",
                "schema_id": "deliverable-schema-1",
                "passed": False,
                "detail": "Citation-title keyword overlap below the 30% floor for 3 of 10 ...",
                "remediation_hint": "Rewrite the affected paragraphs...",
            },
        ],
    }

    payload = resolve_final_outcome_payload(
        {
            "routing_reason": "structural_validation_failed",
            "coverage_gates": {"passed": True, "failed_gates": []},
            "research_loop_count": 4,
            "synthesis_telemetry": {
                "structural_validation": full_report,
                "structural_validation_failed_schemas": ["deliverable-schema-1"],
            },
        },
        completed_at=datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc),
    )

    # The full list is present under validation_report — including the
    # passing validator entry, not only the failing one.
    assert "validation_report" in payload
    surfaced = payload["validation_report"]
    assert surfaced is not None
    validator_ids = {r["validator_id"] for r in surfaced["results"]}
    assert "validate_table_structure" in validator_ids  # passing validator visible
    assert "validate_citation_title_coherence" in validator_ids  # failing validator visible
    # Existing structural_validation_failures list shape is unchanged.
    assert payload["structural_validation_failures"] == ["deliverable-schema-1"]


def test_resolve_final_outcome_payload_omits_validation_report_when_absent() -> None:
    """When synthesis_telemetry has no structural_validation key, the
    validation_report field is simply absent from the payload (additive, not
    breaking — downstream consumers that never read it keep working)."""

    payload = resolve_final_outcome_payload(
        {
            "routing_reason": "converged_with_coverage",
            "coverage_gates": {"passed": True, "failed_gates": []},
            "research_loop_count": 2,
            "synthesis_telemetry": {},
        },
        completed_at=datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "validation_report" not in payload
    assert payload["outcome"] == "complete"


# ---------------------------------------------------------------------------
# S41.4 — entity_slot_admission_audit surfacing in final_outcome
# ---------------------------------------------------------------------------


def test_resolve_final_outcome_payload_surfaces_entity_slot_admission_audit() -> None:
    """S41.4 — per-slot audit must be surfaced in ``final_outcome`` so the
    Phase A scoring script can verify whether S40.1's coherence gate bound
    or stripped specific sources, without shell access to the Railway
    container. Load-bearing fields: ``coherence_score``, ``relevance_source``
    (distinguishes embedding vs tfidf_fallback — critical for S41.5 §4.5
    scoring), and ``bound``."""

    audit = [
        {
            "slot_id": "slot-entity-semantic-entropy",
            "entity": "semantic entropy",
            "source_url": "https://doi.org/10.1038/s41586-024-07421-0",
            "source_title": "Detecting hallucinations using semantic entropy",
            "coherence_score": 0.7821,
            "relevance_source": "embedding",
            "floor_used": 0.4,
            "bound": True,
            "loop": 1,
        },
        {
            "slot_id": "slot-entity-ccs",
            "entity": "CCS",
            "source_url": "https://en.wikipedia.org/wiki/CCS",
            "source_title": "CCS",
            "coherence_score": 0.1234,
            "relevance_source": "tfidf_fallback",
            "floor_used": 0.4,
            "bound": False,
            "loop": 2,
        },
    ]

    payload = resolve_final_outcome_payload(
        {
            "routing_reason": "converged_with_coverage",
            "coverage_gates": {"passed": True, "failed_gates": []},
            "research_loop_count": 3,
            "entity_slot_admission_audit": audit,
        },
        completed_at=datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "entity_slot_admission_audit" in payload
    surfaced = payload["entity_slot_admission_audit"]
    assert len(surfaced) == 2
    # Load-bearing fields preserved.
    first = surfaced[0]
    assert first["entity"] == "semantic entropy"
    assert first["bound"] is True
    assert first["relevance_source"] == "embedding"
    assert first["coherence_score"] == 0.7821
    assert first["floor_used"] == 0.4
    second = surfaced[1]
    assert second["bound"] is False
    assert second["relevance_source"] == "tfidf_fallback"


def test_resolve_final_outcome_payload_omits_entity_slot_audit_when_empty() -> None:
    """Additive — absent or empty audit is simply not surfaced."""

    payload_no_audit = resolve_final_outcome_payload(
        {
            "routing_reason": "converged_with_coverage",
            "coverage_gates": {"passed": True, "failed_gates": []},
            "research_loop_count": 1,
        },
        completed_at=datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert "entity_slot_admission_audit" not in payload_no_audit

    payload_empty_audit = resolve_final_outcome_payload(
        {
            "routing_reason": "converged_with_coverage",
            "coverage_gates": {"passed": True, "failed_gates": []},
            "research_loop_count": 1,
            "entity_slot_admission_audit": [],
        },
        completed_at=datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert "entity_slot_admission_audit" not in payload_empty_audit
