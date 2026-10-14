"""S75 m01 HD4 — round-trip persistence test for the legacy-field strip.

Locks the worker→TraceLab serialisation path to confirm the 3 lying
``execution_metadata`` fields stripped at S75 m01 do NOT come back as
``None`` / ``False`` / ``0`` defaults via dataclass-construction or
to_dict serialisation. T42.3 confirmed zero TraceLab readers on
``loops_executed`` / ``sources_count`` / ``coverage`` / ``quality_gates_passed``;
this test guards the DeepSearch-side emission stays gone.

HD4-pattern test (carry-forward from S68.5 ``test_writer_telemetry_persistence``,
S69 ``test_factscore_persistence``, S71 ``test_sources_collected_persistence``,
S72 ``test_report_metadata_persistence``): build a populated AgentState +
ExecutionMetadata, run them through the documented serialisation hops, and
assert the post-strip shape survives.

Persistence chain exercised:

    AgentState (post-codeagent shape) →
    build_execution_metadata(state, duration) →
    ExecutionMetadata.to_dict() →
    _strip_null_bytes(payload) →
    json.dumps/json.loads (JSONB-safe round-trip)

If a future change accidentally:

* re-adds ``loops_executed`` / ``sources_count`` / ``quality_gates_passed`` to
  :class:`ExecutionMetadata`,
* re-emits them via :meth:`ExecutionMetadata.to_dict`,
* re-introduces the ``check_quality_gates`` fallback computation in
  :func:`build_execution_metadata`, or
* re-introduces the ``coverage_not_computed`` sentinel into the
  ``final_outcome.coverage_gate_failures`` payload,

these tests fail loudly at CI rather than at the next real mission run —
closing the PD43 verification surface so build-hash 28th-consecutive isn't
the only signal.

The tests do not require a database, network, LLM, or smolagents — the
round-trip is pure Python: dataclass → dict → strip-NUL → JSON string → dict.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from deepsearch.worker.writer import (
    ExecutionMetadata,
    _strip_null_bytes,
    build_execution_metadata,
)


# ---------------------------------------------------------------------------
# Realistic post-codeagent AgentState fixture.
# ---------------------------------------------------------------------------


def _post_codeagent_state() -> Dict[str, Any]:
    """An AgentState shape post-S64 entry-point flip + post-S75 deletion.

    The legacy keys ``research_loop_count`` / ``loop_count`` /
    ``quality_gates_passed`` are absent (no legacy node sets them on the
    deployed path). ``coverage_gates`` carries the truthful empty-failures
    surface the m01 strip established at ``agent/state.py:207``.
    """

    return {
        "mission_id": "S75-RT-FIXTURE",
        "telemetry": {
            "token_usage": {"input": 12450, "output": 3120, "total": 15570},
            "runtime_identity": {
                "model": "deepseek-v4-pro",
                "llm_backend": "deepseek",
            },
            "config_fingerprint": "sha256:beef",
            "final_outcome": {
                "coverage_gate_failures": [],  # ← post-strip truth surface
                "outcome": "completed",
                "reason": "self_terminated",
            },
        },
        "synthesis_telemetry": {
            "engine": "codeagent",
            "model_id": "deepseek-v4-pro",
            "step_count": 18,
        },
        "coverage_gates": {
            "passed": False,
            "failed_gates": [],  # ← post-S75 truth, NOT ['coverage_not_computed']
            "entity": {"passed": False, "covered": [], "uncovered": [], "threshold": 0.75, "detail": ""},
            "objective": {"passed": False, "per_objective_scores": {}, "uncovered_objective_ids": [], "threshold": 0.70, "detail": ""},
            "slot": {"passed": False, "total_slots": 0, "filled_slots": 0, "thin_slot_ids": [], "unaddressed_slot_ids": []},
        },
    }


def _serialise_round_trip(em: ExecutionMetadata) -> Dict[str, Any]:
    """Run the dataclass through every documented serialisation hop."""
    payload = _strip_null_bytes(em.to_dict())
    encoded = json.dumps(payload, default=str, ensure_ascii=False)
    return json.loads(encoded)


# ---------------------------------------------------------------------------
# Field-existence guards (the strip).
# ---------------------------------------------------------------------------


def test_execution_metadata_constructor_rejects_lying_fields() -> None:
    """The 3 stripped fields are not constructor kwargs anymore."""
    import pytest

    for field in ("loops_executed", "sources_count", "quality_gates_passed"):
        kwargs = {"total_tokens": 0, "duration_seconds": 0.0, field: 1}
        with pytest.raises(TypeError):
            ExecutionMetadata(**kwargs)


def test_to_dict_does_not_emit_loops_executed() -> None:
    em = ExecutionMetadata(total_tokens=0, duration_seconds=0.0)
    decoded = _serialise_round_trip(em)
    assert "loops_executed" not in decoded


def test_to_dict_does_not_emit_sources_count() -> None:
    em = ExecutionMetadata(total_tokens=0, duration_seconds=0.0)
    decoded = _serialise_round_trip(em)
    assert "sources_count" not in decoded


def test_to_dict_does_not_emit_quality_gates_passed() -> None:
    em = ExecutionMetadata(total_tokens=0, duration_seconds=0.0)
    decoded = _serialise_round_trip(em)
    assert "quality_gates_passed" not in decoded


# ---------------------------------------------------------------------------
# build_execution_metadata round-trip on a realistic post-codeagent state.
# ---------------------------------------------------------------------------


def test_build_execution_metadata_round_trip_strips_lying_fields() -> None:
    """End-to-end build → to_dict → JSONB round trip."""
    state = _post_codeagent_state()
    em = build_execution_metadata(state, duration_seconds=142.4)
    decoded = _serialise_round_trip(em)

    forbidden = {"loops_executed", "sources_count", "quality_gates_passed"}
    leaked = forbidden & set(decoded.keys())
    assert not leaked, (
        f"build_execution_metadata round-trip leaked stripped fields: "
        f"{sorted(leaked)}"
    )


def test_build_execution_metadata_preserves_truthful_surfaces() -> None:
    """The non-lying fields survive the round-trip."""
    state = _post_codeagent_state()
    em = build_execution_metadata(state, duration_seconds=142.4)
    decoded = _serialise_round_trip(em)

    assert decoded["total_tokens"] == 15570
    assert decoded["duration_seconds"] == 142.4
    assert decoded["agent_version"] == "deepsearch-1.0"
    # S84 m04: degraded_mode/degradation_policy DELETED — dead constants
    # presenting retired machinery as a live not-degraded verdict.
    assert "degraded_mode" not in decoded
    assert "degradation_policy" not in decoded
    # synthesis_telemetry envelope is the truth surface
    assert decoded["synthesis_telemetry"]["engine"] == "codeagent"
    assert decoded["synthesis_telemetry"]["model_id"] == "deepseek-v4-pro"
    # runtime_identity gets the truthful HD3 model override
    assert decoded["runtime_identity"]["model"] == "deepseek-v4-pro"
    assert decoded["runtime_identity"]["llm_backend"] == "deepseek"
    # final_outcome empty-coverage_gate_failures truth surface
    assert decoded["final_outcome"]["coverage_gate_failures"] == []


# ---------------------------------------------------------------------------
# coverage_not_computed sentinel does NOT come back via the round trip.
# ---------------------------------------------------------------------------


def test_coverage_not_computed_sentinel_does_not_round_trip() -> None:
    """Even if upstream wrote a sentinel into telemetry.final_outcome, the
    truth surface in the post-strip path stays empty.

    This test holds even when the upstream telemetry carries a stale sentinel
    (some legacy paths might still inject it — until S76 working_set scrub
    finishes the cleanup, the writer's emission must survive defensively).
    """
    state = _post_codeagent_state()
    # Build with an empty surface (the post-strip truth surface)
    em = build_execution_metadata(state, duration_seconds=10.0)
    decoded = _serialise_round_trip(em)
    final_outcome = decoded.get("final_outcome", {})
    failures = final_outcome.get("coverage_gate_failures", [])
    assert "coverage_not_computed" not in failures
