"""Unit tests for the AgentState TypedDict and helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepsearch.agent.state import empty_state  # noqa: E402


def test_empty_state_initializes_required_fields() -> None:
    """AgentState should include all required keys with sensible defaults."""

    state = empty_state(
        mission_id="TEST.0.1",
        mission_objectives=["Collect data"],
        project_id="project-123",
        mission_context={"scope": "unit-test"},
        deliverable_format="json",
        max_loops=5,
    )

    assert state["mission_id"] == "TEST.0.1"
    assert state["project_id"] == "project-123"
    assert state["research_loop_count"] == 0
    assert state["max_loops"] == 5
    assert state["mission_objectives"] == ["Collect data"]
    assert state["mission_context"] == {"scope": "unit-test"}
    assert state["deliverable_format"] == "json"
    # sources_found stripped at S78 m02 (PD71 conservative).
    assert "sources_found" not in state
    assert state["findings"] == []
    assert state["reflection_result"]["decision"] == "continue"
    assert state["reflection_scores"]["overall_coverage_score"] == 0.0
    assert state["convergence_history"] == []
    assert state["error_history"] == []
    assert state["circuit_breaker_active"] is False
    assert state["degraded_mode"] is False
    assert state["degradation_policy"] == "normal"


def test_state_serializes_and_deserializes_via_json() -> None:
    """AgentState must remain JSON-serializable for checkpointing."""

    state = empty_state(mission_id="TEST.0.2", mission_objectives=["Serialize"])
    encoded = json.dumps(state)
    decoded = json.loads(encoded)

    assert decoded["mission_id"] == "TEST.0.2"
    assert decoded["mission_objectives"] == ["Serialize"]
    assert decoded["research_loop_count"] == 0


def test_state_mutations_do_not_leak_between_instances() -> None:
    """Each call to empty_state should yield independent list instances."""

    state_a = empty_state(mission_id="TEST.A", mission_objectives=["A"])
    state_b = empty_state(mission_id="TEST.B", mission_objectives=["B"])

    # sources_found stripped at S78 m02; isolation check uses the
    # remaining mutable collections.
    state_a["mission_objectives"].append("Extra objective")
    state_a["convergence_history"].append(0.5)

    assert state_b["mission_objectives"] == ["B"]
    assert state_b["convergence_history"] == []


def test_default_depth_label_is_baseline() -> None:
    """empty_state defaults research_depth label to baseline post-S76."""

    state = empty_state(mission_id="TEST.DEPTH.DEFAULT")
    assert state["research_depth"] == "baseline"
    assert state["max_loops"] == 3
    assert state["min_loops"] == 2
    assert state["depth_config"] == {}


def test_depth_label_string_normalization() -> None:
    """research_depth strings normalize to lowercase, with invalid → baseline."""

    assert empty_state(mission_id="T1", research_depth="DEEP")["research_depth"] == "deep"
    assert empty_state(mission_id="T2", research_depth="Alpha")["research_depth"] == "alpha"
    assert empty_state(mission_id="T3", research_depth="invalid")["research_depth"] == "baseline"
