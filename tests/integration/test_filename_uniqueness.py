"""S80 m04 (PD109) — re-runs of the same mission_id slug land in distinct files.

User-observed pre-S80 (S79 review session, 2026-05-02): "I can't find the
previous run of this mission now." TraceLab's missions table preserves both
re-run protocols at distinct UUIDs (different DB rows) but the local
``cmos/research/`` directory only retains the most-recent run because the
filename pattern was ``{mission_id}_{title_slug}.md`` — pure slug, no run
discriminator.

Fix at S80 m04: thread the TraceLab system-assigned ``MissionRow.id`` UUID
through ``empty_state`` (`state["mission_uuid"]`) into
``save_report`` / ``save_protocol``. The persistence layer appends an
8-hex-char disambiguator suffix derived from the UUID (or 4 bytes of
OS-randomness for the local path with no TraceLab context).

The integration tests below exercise the full
``empty_state -> save_report / save_protocol`` chain on two simulated
re-runs and assert the on-disk filenames diverge.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from deepsearch.agent.state import empty_state
from deepsearch.report.generator import (
    REPORT_DIRECTORY_ENV,
    save_protocol,
    save_report,
)


@pytest.fixture
def report_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(REPORT_DIRECTORY_ENV, str(tmp_path))
    return tmp_path


def _stub_protocol():
    class _StubProtocol:
        def model_dump_json(self, indent: int = 2) -> str:
            return "{}\n"

    return _StubProtocol()


def test_two_simulated_reruns_with_uuids_produce_distinct_filenames(report_dir):
    """Same mission_id + same title + DIFFERENT UUIDs → DIFFERENT filenames."""
    mission_id = "PLUGIN-BACKEND-AUTH-01"
    title = "Backend Identity Model"

    # Two TraceLab re-runs of the same mission_id slug — UUIDs differ.
    uuid_first = "93201955-71e8-47e1-8124-81244e892924"
    uuid_second = "f7c1a2b3-0e2d-4f6a-9d8e-c1d2e3f4a5b6"

    path_first = save_report(
        "first run content",
        mission_id=mission_id,
        title=title,
        mission_uuid=uuid_first,
    )
    path_second = save_report(
        "second run content",
        mission_id=mission_id,
        title=title,
        mission_uuid=uuid_second,
    )

    assert path_first != path_second, (
        "S80 m04 PD109 — two re-runs of the same mission_id with distinct "
        "TraceLab UUIDs must NOT collide on disk. The pre-S80 filename "
        "pattern silently overwrote the prior run's `.md`."
    )

    # Both files survive on disk.
    assert Path(path_first).exists()
    assert Path(path_second).exists()
    assert Path(path_first).read_text() == "first run content"
    assert Path(path_second).read_text() == "second run content"

    # The 8-hex-char suffix is deterministic per UUID.
    assert "93201955" in Path(path_first).name
    assert "f7c1a2b3" in Path(path_second).name


def test_protocol_filename_carries_same_uuid_disambiguator(report_dir):
    """Markdown report and protocol JSON share an identical run-tag suffix."""
    mission_id = "S80-PROTOCOL-TEST"
    title = "Protocol Pairing"
    uuid_value = "abcd1234-5678-9abc-def0-123456789abc"

    report_path = save_report(
        "report content",
        mission_id=mission_id,
        title=title,
        mission_uuid=uuid_value,
    )
    protocol_path = save_protocol(
        _stub_protocol(),
        mission_id=mission_id,
        title=title,
        mission_uuid=uuid_value,
    )

    # Both filenames carry the same 8-char disambiguator (`abcd1234`).
    assert "abcd1234" in Path(report_path).name
    assert "abcd1234" in Path(protocol_path).name


def test_local_run_with_no_uuid_falls_back_to_random_suffix(report_dir):
    """``mission_uuid=None`` → random 4-byte hex suffix, no collision."""
    mission_id = "LOCAL-NO-UUID"
    title = "Local Run"

    path_a = save_report(
        "first run",
        mission_id=mission_id,
        title=title,
        mission_uuid=None,
    )
    path_b = save_report(
        "second run",
        mission_id=mission_id,
        title=title,
        mission_uuid=None,
    )

    # Random suffixes are extremely unlikely to collide (1 in 2^32).
    assert path_a != path_b, (
        "S80 m04 PD109 — the no-UUID fallback must produce a random "
        "8-hex-char suffix so locally-run missions also avoid silent "
        "overwrite."
    )

    # Both files survive.
    assert Path(path_a).exists()
    assert Path(path_b).exists()


def test_empty_state_propagates_mission_uuid_into_state_field():
    """End-to-end through empty_state — the kwarg lands on the state key."""
    state = empty_state(
        mission_id="S80-M04-PROPAGATION",
        mission_uuid="11111111-2222-3333-4444-555555555555",
    )
    assert state.get("mission_uuid") == "11111111-2222-3333-4444-555555555555"


def test_empty_state_omits_mission_uuid_when_not_provided():
    """Default kwarg is None — the state field reflects the absence."""
    state = empty_state(mission_id="S80-M04-NO-UUID")
    # TypedDict total=False allows the key to be present-with-None.
    assert state.get("mission_uuid") is None
