"""Unit tests for AgentState TypedDict and empty_state() helper.

Tests cover:
- empty_state initialization (defaults + overrides + isolation)
- JSON serialization (checkpoint compatibility)
- Removed bug-patch fields are absent
- Post-S76 boundary research_depth string passthrough (no in-process dispatch)
"""

import json

import pytest

from deepsearch.agent.state import empty_state


class TestEmptyStateDefaults:
    """empty_state() returns correct defaults for all fields."""

    def test_required_fields_present(self) -> None:
        state = empty_state(mission_id="T.1", mission_objectives=["obj1"])
        assert state["mission_id"] == "T.1"
        assert state["research_loop_count"] == 0
        assert state["max_loops"] == 3  # post-S76 default
        assert state["min_loops"] == 2
        assert state["mission_objectives"] == ["obj1"]
        assert state["deliverable_format"] == "markdown"

    def test_empty_collections(self) -> None:
        state = empty_state(mission_id="T.2")
        assert state["messages"] == []
        # sources_found stripped at S78 m02 (PD71 conservative). Field is
        # absent from empty_state output; readers across runner.py / report/
        # use state.get(...) and degrade gracefully on absence.
        assert "sources_found" not in state
        assert state["finding_records"] == []
        assert state["concept_clusters"] == []
        assert state["findings"] == []
        assert state["previous_queries"] == []
        assert state["convergence_history"] == []
        assert state["objective_score_history"] == []
        assert state["refinement_log"] == []
        assert state["error_history"] == []

    def test_boolean_defaults(self) -> None:
        state = empty_state(mission_id="T.3")
        assert state["circuit_breaker_active"] is False
        assert state["degraded_mode"] is False
        assert state["pedr_checked"] is False
        assert state["pedr_reused"] is False
        assert state["pedr_reviewed"] is False

    def test_none_defaults(self) -> None:
        state = empty_state(mission_id="T.4")
        assert state["project_id"] is None
        assert state["pedr_result"] is None
        assert state["report_path"] is None
        assert state["protocol_path"] is None
        assert state["report_markdown"] is None
        assert state["protocol_payload"] is None
        assert state["contradiction_report"] is None
        assert state["mission_outcome"] == ""
        assert state["routing_reason"] == ""
        assert state["coverage_gates"]["passed"] is False

    def test_reflection_result_default(self) -> None:
        state = empty_state(mission_id="T.5")
        assert state["reflection_result"]["decision"] == "continue"
        assert state["reflection_result"]["evaluated_at_loop"] == 0

    def test_reflection_scores_default(self) -> None:
        state = empty_state(mission_id="T.6")
        assert state["reflection_scores"]["overall_coverage_score"] == 0.0
        assert state["reflection_scores"]["is_mission_complete"] is False
        assert state["reflection_scores"]["converged"] is False

    def test_telemetry_default(self) -> None:
        state = empty_state(mission_id="T.7")
        assert state["telemetry"]["estimated_cost_usd"] == 0.0

    def test_degradation_policy_default(self) -> None:
        state = empty_state(mission_id="T.8")
        assert state["degradation_policy"] == "normal"


class TestEmptyStateDepthLabel:
    """empty_state passes research_depth label through as a normalised string.

    Post-S76 the depth.py dispatch module is gone; research_depth is a boundary
    string carrier only — no enum, no dispatch on tier values.
    """

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("baseline", "baseline"),
            ("BASELINE", "baseline"),
            ("deep", "deep"),
            ("DEEP", "deep"),
            ("alpha", "alpha"),
            ("DeEp", "deep"),
            ("invalid", "baseline"),
            ("", "baseline"),
        ],
    )
    def test_depth_label_normalization(self, raw: str, expected: str) -> None:
        state = empty_state(mission_id="T.LBL", research_depth=raw)
        assert state["research_depth"] == expected

    def test_default_depth_is_baseline(self) -> None:
        state = empty_state(mission_id="T.DEF")
        assert state["research_depth"] == "baseline"
        assert state["max_loops"] == 3
        assert state["min_loops"] == 2

    def test_depth_config_is_empty_post_s76(self) -> None:
        state = empty_state(mission_id="T.CFG", research_depth="alpha")
        assert state["depth_config"] == {}


class TestEmptyStateOverrides:
    """Explicit parameter overrides work correctly."""

    def test_max_loops_override(self) -> None:
        state = empty_state(mission_id="T.OV1", max_loops=10)
        assert state["max_loops"] == 10
        assert state["min_loops"] == 2

    def test_min_loops_override(self) -> None:
        state = empty_state(mission_id="T.OV2", min_loops=1)
        assert state["min_loops"] == 1

    def test_both_loop_overrides(self) -> None:
        state = empty_state(mission_id="T.OV3", max_loops=8, min_loops=4)
        assert state["max_loops"] == 8
        assert state["min_loops"] == 4

    def test_project_id(self) -> None:
        state = empty_state(mission_id="T.PID", project_id="proj-123")
        assert state["project_id"] == "proj-123"

    def test_mission_context(self) -> None:
        state = empty_state(mission_id="T.CTX", mission_context={"scope": "test"})
        assert state["mission_context"] == {"scope": "test"}

    def test_deliverable_format(self) -> None:
        state = empty_state(mission_id="T.FMT", deliverable_format="json")
        assert state["deliverable_format"] == "json"


class TestEmptyStateIsolation:
    """Mutable collections are not shared between instances."""

    def test_lists_are_independent(self) -> None:
        a = empty_state(mission_id="A", mission_objectives=["A"])
        b = empty_state(mission_id="B", mission_objectives=["B"])

        # sources_found stripped at S78 m02 — isolation check uses the
        # remaining accumulator collections.
        a["finding_records"].append({"summary": "Leaked finding"})
        a["findings"].append("Leaked finding")
        a["convergence_history"].append(0.99)

        assert b["finding_records"] == []
        assert b["findings"] == []
        assert b["convergence_history"] == []

    def test_objectives_are_independent(self) -> None:
        a = empty_state(mission_id="A", mission_objectives=["A"])
        b = empty_state(mission_id="B", mission_objectives=["B"])

        a["mission_objectives"].append("Extra")
        assert b["mission_objectives"] == ["B"]


class TestEmptyStateSerialization:
    """AgentState must be JSON-serializable for checkpointing."""

    def test_json_roundtrip(self) -> None:
        state = empty_state(mission_id="T.JSON", mission_objectives=["Serialize"])
        state["report_markdown"] = "# Report\n"
        state["protocol_payload"] = {
            "mission_id": "T.JSON",
            "created_at": "2026-08-29T12:00:00Z",
        }
        encoded = json.dumps(state)
        decoded = json.loads(encoded)
        assert decoded["mission_id"] == "T.JSON"
        assert decoded["mission_objectives"] == ["Serialize"]
        assert decoded["research_loop_count"] == 0
        assert decoded["report_markdown"] == "# Report\n"
        assert decoded["protocol_payload"] == {
            "mission_id": "T.JSON",
            "created_at": "2026-08-29T12:00:00Z",
        }


class TestRemovedBugPatchFields:
    """Fields removed in S17 rebuild should not be present."""

    def test_no_consecutive_reflection_failures(self) -> None:
        state = empty_state(mission_id="T.RM1")
        assert "consecutive_reflection_failures" not in state

    def test_no_early_termination(self) -> None:
        state = empty_state(mission_id="T.RM2")
        assert "early_termination" not in state

    def test_no_termination_reason(self) -> None:
        state = empty_state(mission_id="T.RM3")
        assert "termination_reason" not in state
