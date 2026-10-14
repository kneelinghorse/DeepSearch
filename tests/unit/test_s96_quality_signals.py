"""S96 quality semantics distinguish checking scope from detected defects."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from deepsearch.agent.outcome import evaluate_outcome_gate
from deepsearch.agent.quality_series import build_evidence_dependencies, build_quality_record

URL = "https://example.org/source"


def healthy() -> dict[str, Any]:
    outcomes = {"by_tool": {}, "records": [], "records_truncated": 0}
    return {
        "self_terminated": True,
        "citation_telemetry": {"citations_extracted": 1, "distinct_urls_cited": 1},
        "deliverable_coverage": {"shortfall_score": 0, "satisfied": ["Report"], "missing": []},
        "liveness_telemetry": {"checked": 1, "live": 1, "dead": 0, "orphan": 0, "unknown": 0},
        "critique_telemetry": {
            "claim_count": 1, "error": None,
            "annotation_summary": {"applied": 0, "skipped": 0, "dropped": 0, "reconciled": True},
            "assessment": {"version": 1, "status": "partial", "assessed": 1, "supported": 1,
                           "unassessed": 0, "contradicted": 0},
        },
        "tool_outcomes": outcomes,
        "evidence_dependencies": build_evidence_dependencies([URL], [source()], outcomes),
        "recovery": {"status": "not_triggered", "passes": 0},
    }


def source(**updates: Any) -> dict[str, Any]:
    return {"url": URL, "body_source": "source_fetch", "content_quality": "readable_body",
            "body": "A retained readable source body.", **updates}


def test_supported_sample_can_be_clean_without_claiming_report_accuracy() -> None:
    telemetry = healthy()
    gate = evaluate_outcome_gate(telemetry, report_word_count=500)
    record = build_quality_record(telemetry, 500)
    assert gate.delivery_quality == "clean" and gate.quality_gate_failures == []
    assert record["verification"]["scope"] == "sampled"
    assert record["verification"]["report_wide_accuracy"] == "unknown"
    assert record["claim_assessment"]["status"] == "partial"
    assert record["claim_assessment"]["accuracy_pass"] is None
    assert record["claim_assessment"]["coverage_ratio"] is None


@pytest.mark.parametrize(("assessment", "reason", "tier"), [
    ({"unassessed": 2}, "critique_unassessed_claims_2", "usable_with_warning"),
    ({"assessed": 0, "supported": 0}, "critique_assessment_unassessed", "usable_with_warning"),
    ({"contradicted": 1, "supported": 0}, "critique_contradicted_1", "degraded"),
])
def test_real_assessment_limits_remain_actionable(
    assessment: dict[str, int], reason: str, tier: str,
) -> None:
    telemetry = healthy()
    telemetry["critique_telemetry"]["assessment"].update(assessment)
    result = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert result.delivery_quality == tier and reason in result.quality_gate_failures


def test_recovered_discovery_cache_and_controller_failures_are_operational() -> None:
    telemetry = healthy()
    outcomes = {"by_tool": {"source_fetch": {"error": 3}, "web_search": {"error": 2}},
                "records_truncated": 0, "records": [
        {"tool": "source_fetch", "url": URL, "status": "error", "network_requested": True},
        {"tool": "source_fetch", "url": URL, "status": "error", "network_requested": False,
         "error_category": "assembly_draft_required"},
        {"tool": "source_fetch", "url": URL, "status": "error", "network_requested": False,
         "error_category": "source_not_cached"},
    ]}
    telemetry["tool_outcomes"] = outcomes
    telemetry["evidence_dependencies"] = build_evidence_dependencies([URL], [source()], outcomes)
    assert evaluate_outcome_gate(telemetry, report_word_count=500).delivery_quality == "clean"
    quality = build_quality_record(telemetry, 500)
    assert quality["retrieval_failures"]["total"] == 5
    assert quality["evidence_dependencies"]["recovered_cited_fetch_failures"] == 1
    assert quality["evidence_dependencies"]["local_fetch_refusals"] == 2


def test_unresolved_cited_failure_and_snippet_only_evidence_are_visible() -> None:
    telemetry = healthy()
    outcomes = {"records_truncated": 0, "records": [
        {"tool": "source_fetch", "url": URL, "status": "error", "network_requested": True},
    ]}
    telemetry["evidence_dependencies"] = build_evidence_dependencies(
        [URL], [source(body_source="snippet", content_quality="search_snippet")], outcomes)
    gate = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert gate.delivery_quality == "usable_with_warning"
    assert "unresolved_cited_fetch_failures_1" in gate.quality_gate_failures
    assert "cited_without_readable_body_1" in gate.quality_gate_failures
    assert telemetry["evidence_dependencies"]["missing_cited_sources"] == 0


def test_local_refusal_is_not_a_network_failure_but_missing_evidence_still_warns() -> None:
    telemetry = healthy()
    outcomes = {"records_truncated": 0, "records": [
        {"tool": "source_fetch", "url": URL, "status": "error", "network_requested": False},
    ]}
    telemetry["evidence_dependencies"] = build_evidence_dependencies([URL], [], outcomes)
    gate = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert "missing_cited_sources_1" in gate.quality_gate_failures
    assert "unresolved_cited_fetch_failures_1" not in gate.quality_gate_failures


def test_missing_legacy_diagnostics_never_become_stronger_certainty() -> None:
    telemetry = healthy()
    del telemetry["evidence_dependencies"]
    del telemetry["critique_telemetry"]["assessment"]
    telemetry["tool_outcomes"] = {"by_tool": {"source_fetch": {"error": 1}}}
    result = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert "retrieval_failure_impact_unknown" in result.quality_gate_failures
    assert "critique_assessment_unknown" in result.quality_gate_failures
    quality = build_quality_record(telemetry, 500)
    assert quality["verification"]["scope"] == "unassessed_or_unknown"
    assert quality["evidence_dependencies"]["available"] is False


@pytest.mark.parametrize("changes", [
    {"hallucinated_quarantined": 1}, {"citations_unmappable": 1}, {"markers_quarantined": 1},
    {"citation_url_projection": {"dropped_count": 1}},
    {"citation_consistency": {"reference_number_collisions": 1, "consistent": False}},
    {"citation_consistency": {"unmapped_protocol_citations": 1, "consistent": False}},
    {"citation_consistency": {"consistent": False}},
])
def test_concrete_citation_defects_cannot_become_clean(changes: dict[str, Any]) -> None:
    telemetry = healthy() | changes
    result = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert result.delivery_quality == "usable_with_warning" and result.quality_gate_failures


def test_accounting_surplus_is_disclosed_without_a_false_mapping_defect() -> None:
    telemetry = healthy()
    telemetry["citation_consistency"] = {
        "consistent": False, "unmapped_protocol_citations": 0, "reference_number_collisions": 0,
        "references_only": 1, "unused_reference_numbers": [2],
    }
    assert evaluate_outcome_gate(telemetry, report_word_count=500).delivery_quality == "clean"
    surface = build_quality_record(telemetry, 500)["citation_surface"]
    assert surface["accounting_only"] and surface["unused_reference_numbers"] == [2]
    telemetry["citation_consistency"]["unresolved_reader_markers"] = 1
    assert evaluate_outcome_gate(telemetry, report_word_count=500).delivery_quality != "clean"


def test_truncated_evidence_and_missing_attribution_remain_warnings() -> None:
    telemetry = healthy()
    telemetry["evidence_dependencies"] = build_evidence_dependencies(
        [URL], [source(body_truncated=True)], {"records": [], "records_truncated": 2})
    result = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert "truncated_cited_bodies_1" in result.quality_gate_failures
    assert "retrieval_dependency_attribution_unknown" in result.quality_gate_failures


@pytest.mark.parametrize("changes", [
    {"output_truncated": True},
    {"generation_error_recovered": True, "generation_recovery_result": "unknown_finish"},
])
def test_unproven_artifacts_still_fail_quality_but_remain_reviewable(
    changes: dict[str, Any],
) -> None:
    result = evaluate_outcome_gate(healthy() | changes, report_word_count=500)
    assert result.outcome.value == "validation_failed" and result.delivery_quality == "failed"


def test_material_coverage_gap_keeps_other_actionable_reasons() -> None:
    telemetry = deepcopy(healthy())
    telemetry["deliverable_coverage"] = {"shortfall_score": 1, "missing": ["Required section"]}
    telemetry["critique_telemetry"]["assessment"]["unassessed"] = 1
    result = evaluate_outcome_gate(telemetry, report_word_count=500)
    assert result.delivery_quality == "degraded" and result.coverage_gate_failures
    assert "critique_unassessed_claims_1" in result.quality_gate_failures
