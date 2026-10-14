"""Tests for Mission Protocol loader utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch.mission.loader import (  # noqa: E402
    MissionLoadError,
    _resolve_research_depth,
    load_mission_from_yaml,
)
from deepsearch.mission.schema import MissionYAML  # noqa: E402

FIXTURE_DIR = Path("tests/fixtures/missions")


def _fixture_path(name: str) -> Path:
    return FIXTURE_DIR / name


def test_load_mission_from_yaml_returns_agent_state() -> None:
    """Valid YAML documents should become populated AgentState objects."""

    state = load_mission_from_yaml(_fixture_path("valid_mission.yaml"))

    assert state["mission_id"] == "TEST.1"
    assert state["project_id"] == "TEST"
    assert state["mission_objectives"] == [
        "Document API endpoints",
        "Capture authentication details",
    ]
    assert state["deliverable_format"] == "markdown"
    assert state["max_loops"] == 5
    assert state["mission_context"]["background"].startswith("This mission validates")
    assert state["mission_context"]["success_criteria"] == [
        "Document API endpoints",
        "Capture authentication details",
    ]
    assert "research_phases" in state["mission_context"]


def test_load_mission_from_yaml_minimal_defaults() -> None:
    """Minimal missions should still populate defaults."""

    state = load_mission_from_yaml(_fixture_path("minimal_mission.yaml"))

    assert state["mission_id"] == "MIN.1"
    assert state["mission_objectives"] == ["Return baseline agent state"]
    assert state["deliverable_format"] == "markdown"
    assert state["max_loops"] == 3
    assert state["mission_context"]["objective"] == "Ensure defaults work"


def test_load_mission_from_yaml_missing_required_field_raises() -> None:
    """Invalid mission YAML should raise MissionLoadError with context."""

    with pytest.raises(MissionLoadError) as exc:
        load_mission_from_yaml(_fixture_path("invalid_mission.yaml"))

    assert "objective" in str(exc.value)


def test_load_mission_from_yaml_invalid_yaml() -> None:
    """Malformed YAML should raise MissionLoadError."""

    with pytest.raises(MissionLoadError):
        load_mission_from_yaml(_fixture_path("bad_yaml.yaml"))


def test_load_mission_alpha_depth() -> None:
    """Alpha depth label is parsed and surfaced as a normalised string."""

    state = load_mission_from_yaml(_fixture_path("depth_alpha.yaml"))

    assert state["research_depth"] == "alpha"
    assert state["depth_config"] == {}


def test_load_mission_deep_depth_case_insensitive() -> None:
    """Deep depth label is parsed case-insensitively via the 'depth' alias."""

    state = load_mission_from_yaml(_fixture_path("depth_deep.yaml"))

    assert state["research_depth"] == "deep"
    assert state["depth_config"] == {}


def test_load_mission_invalid_depth_defaults_baseline(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown depth values fall back to 'baseline' with a warning."""

    with caplog.at_level(logging.WARNING):
        state = load_mission_from_yaml(_fixture_path("depth_invalid.yaml"))

    assert state["research_depth"] == "baseline"
    assert state["depth_config"] == {}
    assert any("Unknown research depth 'ultra'" in record.message for record in caplog.records)


def test_load_mission_depth_from_metadata() -> None:
    """Depth label is readable from metadata as a fallback."""

    state = load_mission_from_yaml(_fixture_path("depth_in_metadata.yaml"))

    assert state["research_depth"] == "deep"


def test_load_mission_no_depth_defaults_baseline() -> None:
    """Missions without depth default to 'baseline'."""

    state = load_mission_from_yaml(_fixture_path("minimal_mission.yaml"))

    assert state["research_depth"] == "baseline"


def test_resolve_research_depth_direct() -> None:
    """_resolve_research_depth returns normalised label strings."""

    mission = MissionYAML(
        mission_id="TEST.1",
        title="Test",
        objective="Test objective",
        success_criteria=["criterion"],
        research_depth="alpha",
    )
    assert _resolve_research_depth(mission) == "alpha"

    mission = MissionYAML(
        mission_id="TEST.2",
        title="Test",
        objective="Test objective",
        success_criteria=["criterion"],
        research_depth="DEEP",
    )
    assert _resolve_research_depth(mission) == "deep"


def test_resolve_research_depth_metadata_fallback() -> None:
    """_resolve_research_depth falls back to metadata when schema field is empty."""

    mission = MissionYAML(
        mission_id="TEST.3",
        title="Test",
        objective="Test objective",
        success_criteria=["criterion"],
        metadata={"researchDepth": "alpha"},
    )
    assert _resolve_research_depth(mission) == "alpha"
