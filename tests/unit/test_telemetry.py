"""Unit tests for mission telemetry normalization helpers."""

from __future__ import annotations

from deepsearch.agent.telemetry import normalize_final_telemetry


def test_normalize_final_telemetry_projects_reflection_state_history() -> None:
    """Final telemetry should carry state-level reflection history into JSONL payloads."""

    state = {
        "telemetry": {
            "api_calls": {"llm_reflection": "2", "llm_research": 1},
            "token_usage": {"prompt_tokens": "120", "completion_tokens": 45},
            "coverage_gate_trace": [{"loop": 1, "passed": False, "failed_gates": ["slot"]}],
        },
        "objective_score_history": [
            {
                "loop": 1,
                "scores": [{"objective_id": "objective-0", "score": 0.62}],
            }
        ],
        "convergence_history": [0.55, "0.62", None, "bad"],
    }

    telemetry = normalize_final_telemetry(state)

    assert telemetry["api_calls"] == {"llm_reflection": 2, "llm_research": 1}
    assert telemetry["token_usage"] == {"prompt_tokens": 120, "completion_tokens": 45}
    assert telemetry["coverage_gate_trace"][0]["failed_gates"] == ["slot"]
    assert telemetry["objective_score_history"][0]["scores"][0]["score"] == 0.62
    assert telemetry["convergence_history"] == [0.55, 0.62]


def test_normalize_final_telemetry_projects_structural_validation_failures() -> None:
    """Structural validation failures should be promoted into persisted telemetry."""

    state = {
        "synthesis_telemetry": {
            "structural_validation_failed_schemas": [
                "classification_matrix",
                "recommendation_section",
            ]
        }
    }

    telemetry = normalize_final_telemetry(state)

    assert telemetry["structural_validation_failures"] == [
        "classification_matrix",
        "recommendation_section",
    ]


def test_normalize_final_telemetry_projects_slot_and_finalize_diagnostics() -> None:
    """Final telemetry should carry slot/finalize diagnostics for smoke triage."""

    state = {
        "slot_coverage": {
            "slots": [
                {
                    "slot_id": "slot-deliverable-schema-1",
                    "kind": "deliverable_schema",
                    "status": "unaddressed",
                    "source_count": 0,
                    "target": {"schema_id": "deliverable-schema-1", "format": "table"},
                    "reasons": [],
                    "domains": [],
                }
            ]
        },
        "synthesis_telemetry": {
            "report_render_path": "template_fallback",
            "report_render_reason": "no_api_key",
            "finalize_contract_present": True,
            "finalize_has_deliverable_schemas": True,
            "finalize_deliverable_schema_count": 1,
            "finalize_deliverable_schema_ids": ["deliverable-schema-1"],
        },
    }

    telemetry = normalize_final_telemetry(state)

    assert telemetry["slot_diagnostics"]["failed_slot_ids"] == ["slot-deliverable-schema-1"]
    assert telemetry["slot_diagnostics"]["failed_slot_kinds"] == ["deliverable_schema"]
    assert telemetry["finalize_diagnostics"] == {
        "report_render_path": "template_fallback",
        "report_render_reason": "no_api_key",
        "contract_present": True,
        "has_deliverable_schemas": True,
        "deliverable_schema_ids": ["deliverable-schema-1"],
        "deliverable_schema_count": 1,
    }
