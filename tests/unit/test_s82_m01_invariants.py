"""S82 m01 (PD136) RD2 invariants — the tiered outcome gate.

Pins ``deepsearch.agent.outcome.evaluate_outcome_gate`` against the FIVE
preserved real telemetry shapes from the Concordance R001–R004 + cross-store
triage (PD135). These are the exact ``synthesis_telemetry`` envelopes recorded
on TraceLab/Postgres for those missions — snapshotted here so the gate's
decisions are verified with ZERO new dispatches (Coordination Hard Rule 3).

The keystone defect (PD135): the pipeline COMPUTED every quality signal —
shortfall_score, dead/orphan counts, citations_extracted, hit_max_steps — and
ENFORCED none of them. ``codeagent_finalize_node`` hardcoded
``mission_outcome = COMPLETE`` and the only real gate hid behind
``runner.py validate=True`` which the worker never passes. This gate makes the
signals load-bearing on the production worker path.

Tier policy (PD136, user-approved 2026-05-28), with ANY deliverable shortfall
DEGRADING rather than hard-failing (DSV-045, 2026-06-15 — extends the original
single-deliverable carve-out to all shortfalls; see the module comment in
outcome.py for the rationale: completeness is a recall-biased heuristic, not a
catastrophic kill switch):

    cross-store 62eab1b0 -> VALIDATION_FAILED  (0 citations on 789w; max-steps also observed)
    R003        e788deba -> COMPLETE/degraded  (shortfall 1.0 across 5 deliverables; orphan 12)
    R001        9f9c96bb -> COMPLETE/degraded  (single-deliverable shortfall; dead 20%)
    R002        280c064f -> COMPLETE/degraded  (single-deliverable shortfall; dead 19%)
    R004        770b0794 -> COMPLETE/clean      (shortfall 0, dead 5.7% < 10%, orphan 0)

If business logic ever changes a threshold or a tier boundary, these fixtures
fail loudly — that is the point (RD2 layered guard).
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.outcome import (
    DELIVERY_QUALITY_DEGRADED,
    DELIVERY_QUALITY_FAILED,
    DELIVERY_QUALITY_USABLE_WITH_WARNING,
    HARD_SHORTFALL,
    MIN_DELIVERABLES_FOR_HARD_SHORTFALL,
    MIN_REPORT_WORDS,
    SOFT_DEAD_RATE,
    SOFT_ORPHAN,
    MissionOutcome,
    evaluate_outcome_gate,
    resolve_final_outcome_payload,
)

# ---------------------------------------------------------------------------
# The 5 preserved real telemetry shapes (verbatim numeric values pulled from
# TraceLab execution_metadata.synthesis_telemetry; no new dispatch).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateFixture:
    name: str
    uuid: str
    synthesis_telemetry: dict[str, Any]
    report_word_count: int
    expected_outcome: MissionOutcome
    expected_delivery_quality: str


def _liveness(
    *, checked: int, live: int, dead: int, orphan: int, unknown: int = 0
) -> dict[str, Any]:
    """LivenessTelemetry.to_dict() shape (citation_liveness.py:110-123)."""
    return {
        "checked": checked,
        "live": live,
        "dead": dead,
        "orphan": orphan,
        "unknown": unknown,
        "hallucinated_url_count": orphan,
        "distinct_urls_checked": checked,
        "wall_clock_ms": 0,
        "cache_hits": 0,
        "budget_exhausted": False,
    }


FIXTURES: list[GateFixture] = [
    GateFixture(
        name="cross-store",
        uuid="62eab1b0-dbef-4c68-a30c-1c8348999072",
        synthesis_telemetry={
            "hit_max_steps": True,
            "self_terminated": False,
            "step_count": 41,
            "citation_telemetry": {
                "citations_extracted": 0,
                "orphan_marker_count": 0,
                "distinct_urls_cited": 0,
            },
            "deliverable_coverage": {
                "shortfall_score": 0.625,
                "satisfied": ["a", "b", "c"],  # 3 satisfied
                "missing": ["d", "e", "f", "g", "h"],  # 5 missing -> 8 total
            },
            "liveness_telemetry": _liveness(checked=0, live=0, dead=0, orphan=0),
        },
        report_word_count=789,
        expected_outcome=MissionOutcome.VALIDATION_FAILED,
        expected_delivery_quality=DELIVERY_QUALITY_FAILED,
    ),
    GateFixture(
        name="R001",
        uuid="9f9c96bb-a176-4901-8971-d5c89dd1a946",
        synthesis_telemetry={
            "hit_max_steps": False,
            "self_terminated": True,
            "step_count": 26,
            "citation_telemetry": {
                "citations_extracted": 64,
                "orphan_marker_count": 4,
                "distinct_urls_cited": 59,
            },
            "deliverable_coverage": {
                "shortfall_score": 1.0,
                "satisfied": [],
                "missing": ["design systems compilation index"],  # single deliverable
            },
            "liveness_telemetry": _liveness(checked=64, live=43, dead=13, orphan=3),
        },
        report_word_count=79714,
        expected_outcome=MissionOutcome.COMPLETE,
        expected_delivery_quality=DELIVERY_QUALITY_DEGRADED,
    ),
    GateFixture(
        name="R002",
        uuid="280c064f-a917-4a0b-a07a-505e226fabff",
        synthesis_telemetry={
            "hit_max_steps": False,
            "self_terminated": True,
            "step_count": 11,
            "citation_telemetry": {
                "citations_extracted": 82,
                "orphan_marker_count": 4,
                "distinct_urls_cited": 69,
            },
            "deliverable_coverage": {
                "shortfall_score": 1.0,
                "satisfied": [],
                "missing": ["vendor -> SDK-family ownership map"],  # single deliverable
            },
            "liveness_telemetry": _liveness(checked=82, live=49, dead=16, orphan=4),
        },
        report_word_count=5891,
        expected_outcome=MissionOutcome.COMPLETE,
        expected_delivery_quality=DELIVERY_QUALITY_DEGRADED,
    ),
    GateFixture(
        name="R003",
        uuid="e788deba-5af7-459f-8962-0882e2985349",
        synthesis_telemetry={
            "hit_max_steps": False,
            "self_terminated": True,
            "step_count": 40,
            "citation_telemetry": {
                "citations_extracted": 113,
                "orphan_marker_count": 26,
                "distinct_urls_cited": 90,
            },
            "deliverable_coverage": {
                "shortfall_score": 1.0,
                "satisfied": [],
                "missing": ["a", "b", "c", "d", "e"],  # 5 deliverables, all missing
            },
            "liveness_telemetry": _liveness(checked=113, live=72, dead=6, orphan=12),
        },
        report_word_count=1756,
        # DSV-045 (2026-06-15): a broad multi-deliverable shortfall now DEGRADES
        # rather than catastrophically fails — completeness is a heuristic, not a
        # kill switch (113 citations, self-terminated, 1756w; orphan 12 flags it).
        expected_outcome=MissionOutcome.COMPLETE,
        expected_delivery_quality=DELIVERY_QUALITY_DEGRADED,
    ),
    GateFixture(
        name="R004",
        uuid="770b0794-8088-4cbf-afcf-c4433b78e9b8",
        synthesis_telemetry={
            "hit_max_steps": False,
            "self_terminated": True,
            "step_count": 25,
            "citation_telemetry": {
                "citations_extracted": 87,
                "orphan_marker_count": 10,
                "distinct_urls_cited": 41,
            },
            "deliverable_coverage": {
                "shortfall_score": 0.0,
                "satisfied": ["triple-section state-of-art survey"],
                "missing": [],
            },
            "liveness_telemetry": _liveness(checked=87, live=36, dead=5, orphan=0),
        },
        report_word_count=5847,
        expected_outcome=MissionOutcome.COMPLETE,
        # S87 m04: preserved telemetry predates structured critique, retrieval-
        # failure, and recovery facts, so it is usable but cannot be clean.
        expected_delivery_quality=DELIVERY_QUALITY_USABLE_WITH_WARNING,
    ),
]


@pytest.mark.parametrize("fx", FIXTURES, ids=[f.name for f in FIXTURES])
def test_gate_decision_matches_preserved_telemetry(fx: GateFixture) -> None:
    """The gate must reproduce the agreed decision for each real fixture."""

    result = evaluate_outcome_gate(
        fx.synthesis_telemetry, report_word_count=fx.report_word_count
    )

    assert result.outcome is fx.expected_outcome, (
        f"{fx.name} ({fx.uuid}): expected {fx.expected_outcome} "
        f"got {result.outcome}"
    )
    assert result.delivery_quality == fx.expected_delivery_quality, (
        f"{fx.name}: expected delivery_quality={fx.expected_delivery_quality} "
        f"got {result.delivery_quality}"
    )


def _fx(name: str) -> GateFixture:
    return next(f for f in FIXTURES if f.name == name)


def test_catastrophic_fixtures_populate_failure_lists() -> None:
    """VALIDATION_FAILED fixtures must carry a routing_reason + populated lists.

    Only unambiguous signals remain catastrophic (DSV-045): cross-store fails via
    zero-citations + max-steps-no-final. R003 (a pure shortfall) now DEGRADES.
    """

    for name in ("cross-store",):
        fx = _fx(name)
        result = evaluate_outcome_gate(
            fx.synthesis_telemetry, report_word_count=fx.report_word_count
        )
        assert result.outcome is MissionOutcome.VALIDATION_FAILED
        assert result.routing_reason, f"{name}: catastrophic verdict needs a reason"
        # At least one failure list must be non-empty to explain the verdict.
        assert result.quality_gate_failures or result.coverage_gate_failures


def test_degraded_fixtures_complete_with_populated_failures() -> None:
    """R001/R002 ship COMPLETE but flagged degraded with explanatory lists."""

    for name in ("R001", "R002"):
        fx = _fx(name)
        result = evaluate_outcome_gate(
            fx.synthesis_telemetry, report_word_count=fx.report_word_count
        )
        assert result.outcome is MissionOutcome.COMPLETE
        assert result.delivery_quality == DELIVERY_QUALITY_DEGRADED
        assert result.routing_reason == ""  # degraded does not set an error reason
        assert result.coverage_gate_failures, f"{name}: shortfall must be listed"
        assert result.quality_gate_failures, f"{name}: dead-rate must be listed"


def test_legacy_healthy_fixture_is_usable_with_explicit_unknowns() -> None:
    """R004 lacks S87 facts, so missing validation cannot become clean."""

    fx = _fx("R004")
    result = evaluate_outcome_gate(
        fx.synthesis_telemetry, report_word_count=fx.report_word_count
    )
    assert result.outcome is MissionOutcome.COMPLETE
    assert result.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING
    assert "critique_legacy_unstructured" not in result.quality_gate_failures
    assert "critique_unknown" in result.quality_gate_failures
    assert result.coverage_gate_failures == []


# ---------------------------------------------------------------------------
# Tier-boundary guards — the load-bearing decision logic, isolated.
# ---------------------------------------------------------------------------


def test_single_deliverable_shortfall_degrades_not_fails() -> None:
    """A 1-of-1 shortfall=1.0 with a substantive cited report DEGRADES.

    This is the user-approved (2026-05-28) single-deliverable policy: the
    binary 1.0 is dominated by completeness_check substring artifacts (m04
    fixes that), and the report is downstream-usable — flag, do not suppress.
    """

    telemetry = {
        "hit_max_steps": False,
        "self_terminated": True,
        "citation_telemetry": {"citations_extracted": 40},
        "deliverable_coverage": {
            "shortfall_score": 1.0,
            "satisfied": [],
            "missing": ["the one deliverable"],  # count == 1
        },
        "liveness_telemetry": _liveness(checked=40, live=40, dead=0, orphan=0),
    }
    result = evaluate_outcome_gate(telemetry, report_word_count=5000)
    assert result.outcome is MissionOutcome.COMPLETE
    assert result.delivery_quality == DELIVERY_QUALITY_DEGRADED


def test_multi_deliverable_shortfall_degrades_not_fails() -> None:
    """DSV-045 (2026-06-15): a shortfall across >= 2 deliverables now DEGRADES,
    not catastrophically fails. completeness_check is a recall-biased heuristic
    whose nested/misnested edge cases cannot be resolved from structure alone, so
    a coverage shortfall must not be the SOLE catastrophic trigger — a live
    3.5k-word, well-cited report was failed 0/4 by a completeness false-MISSING.
    A genuinely empty run still hard-fails via the zero-citations trigger."""

    telemetry = {
        "hit_max_steps": False,
        "self_terminated": True,
        "citation_telemetry": {"citations_extracted": 40},
        "deliverable_coverage": {
            "shortfall_score": 1.0,
            "satisfied": [],
            "missing": ["one", "two"],  # broad multi-deliverable shortfall
        },
        "liveness_telemetry": _liveness(checked=40, live=40, dead=0, orphan=0),
    }
    result = evaluate_outcome_gate(telemetry, report_word_count=5000)
    assert result.outcome is MissionOutcome.COMPLETE
    assert result.delivery_quality == DELIVERY_QUALITY_DEGRADED
    assert result.coverage_gate_failures  # still flagged, not suppressed


def test_zero_citations_requires_nontrivial_word_floor() -> None:
    """0 citations only hard-fails when the report clears MIN_REPORT_WORDS."""

    base = {
        "hit_max_steps": False,
        "self_terminated": True,
        "citation_telemetry": {"citations_extracted": 0},
        "deliverable_coverage": {"shortfall_score": 0.0, "satisfied": ["x"], "missing": []},
        "liveness_telemetry": _liveness(checked=0, live=0, dead=0, orphan=0),
    }
    # Non-trivial 0-citation report -> catastrophic.
    big = evaluate_outcome_gate(base, report_word_count=MIN_REPORT_WORDS)
    assert big.outcome is MissionOutcome.VALIDATION_FAILED
    # A terse note below the floor is NOT hard-failed on the citations trigger.
    small = evaluate_outcome_gate(base, report_word_count=MIN_REPORT_WORDS - 1)
    assert small.outcome is MissionOutcome.COMPLETE


def test_dead_rate_boundary_degrades() -> None:
    """dead-rate >= SOFT_DEAD_RATE degrades; just under stays clean."""

    def telemetry(dead: int, checked: int) -> dict[str, Any]:
        return {
            "hit_max_steps": False,
            "self_terminated": True,
            "citation_telemetry": {"citations_extracted": checked},
            "deliverable_coverage": {"shortfall_score": 0.0, "satisfied": ["x"], "missing": []},
            "liveness_telemetry": _liveness(
                checked=checked, live=checked - dead, dead=dead, orphan=0
            ),
        }

    # 10/100 == 0.10 == SOFT_DEAD_RATE -> degraded.
    at = evaluate_outcome_gate(telemetry(dead=10, checked=100), report_word_count=1000)
    assert at.delivery_quality == DELIVERY_QUALITY_DEGRADED
    # 9/100 < SOFT_DEAD_RATE, but absent S87 validation facts prevent clean.
    under = evaluate_outcome_gate(telemetry(dead=9, checked=100), report_word_count=1000)
    assert under.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING


def test_gate_never_raises_on_empty_or_malformed_telemetry() -> None:
    """Defensive: malformed signals never raise or silently become clean."""

    empty = evaluate_outcome_gate({}, report_word_count=0)
    assert empty.outcome is MissionOutcome.COMPLETE
    assert empty.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING
    junk = {
        "hit_max_steps": "yes",
        "citation_telemetry": "nope",
        "deliverable_coverage": None,
        "liveness_telemetry": [1, 2, 3],
    }
    result = evaluate_outcome_gate(junk, report_word_count=10)
    assert result.outcome in (MissionOutcome.COMPLETE, MissionOutcome.VALIDATION_FAILED)


# ---------------------------------------------------------------------------
# Threshold-constant guards — pin the conservative starting values (PD136).
# ---------------------------------------------------------------------------


def test_threshold_constants_are_pinned() -> None:
    assert HARD_SHORTFALL == 0.60
    assert MIN_DELIVERABLES_FOR_HARD_SHORTFALL == 2
    assert SOFT_DEAD_RATE == 0.10
    assert SOFT_ORPHAN == 3
    assert MIN_REPORT_WORDS == 200


# ---------------------------------------------------------------------------
# Wiring guards — the gate is actually load-bearing on the finalize path.
# ---------------------------------------------------------------------------


def test_finalize_node_invokes_gate_and_drops_hardcode() -> None:
    """Static guard: the line-343 COMPLETE hardcode is gone; the gate is called."""

    src = inspect.getsource(fnode)
    assert "evaluate_outcome_gate(" in src, "finalize node must call the gate"
    # The old unconditional pattern must not survive.
    assert "mission_outcome = MissionOutcome.COMPLETE" not in src, (
        "the unconditional COMPLETE hardcode must be replaced by the gate"
    )


def _state() -> dict[str, Any]:
    return {
        "mission_id": "S82-M01-WIRING",
        "mission_objectives": ["c1", "c2"],
        "mission_context": {
            "title": "Wiring Mission",
            "objective": "verify gate wiring",
            "deliverables": ["Comparison Matrix"],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _catastrophic_synthesis() -> SynthesisResult:
    """Uncited partial output produced after exhausting the action budget."""

    return SynthesisResult(
        output_markdown="# Report\n\nPartial draft, agent ran out of steps mid-assembly.",
        step_count=40,
        self_terminated=False,
        hit_max_steps=True,
        runtime_seconds=42.0,
        token_usage={"input": 1000, "output": 2000, "total": 3000},
        tool_call_summary={"total_calls": 7, "by_tool": {"web_search": 3}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        error=None,
    )


def _critique_passthrough() -> CritiqueResult:
    return CritiqueResult(
        output_markdown="# Report\n\nPartial draft, agent ran out of steps mid-assembly.",
        claim_count=1,
        gap_notes_inserted=0,
        runtime_seconds=1.0,
        token_usage={"input": 100, "output": 50, "total": 150},
        error=None,
    )


def test_finalize_node_emits_validation_failed_direct_call(
    tmp_path,
    monkeypatch,
) -> None:
    """An uncited max-step partial remains reviewable but validation-failed.

    S83 m02 rename: this invokes ``codeagent_finalize_node`` directly — it does
    NOT compile or ``.ainvoke`` the real ``StateGraph``, so it does not cross the
    node-return channel-merge boundary that silently dropped the gate channels
    (the S42.2 class). The "end_to_end" name was misleading; the real-graph
    round-trip that crosses that boundary lives in
    ``test_s83_m02_invariants.py::test_gate_channels_survive_real_graph_roundtrip``.
    """

    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))

    async def run() -> dict[str, Any]:
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_catastrophic_synthesis(),
        ), patch.object(fnode, "run_critique", return_value=_critique_passthrough()):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert result["delivery_quality"] == DELIVERY_QUALITY_FAILED
    assert result["routing_reason"]
    assert isinstance(result["quality_gate_failures"], list)
    assert any("hit_max_steps" in f for f in result["quality_gate_failures"])
    # The report is still produced (the gate FLAGS, it does not suppress).
    assert "report_path" in result
    assert Path(result["report_path"]).exists()


def test_gate_state_keys_are_declared_agentstate_channels() -> None:
    """Regression guard for the S42.2 LangGraph pitfall caught by the post-deploy
    cross-store re-dispatch (62eab1b0): the keys the gate writes on the finalize
    state-update MUST be declared AgentState channels AND initialised in
    empty_state, or LangGraph silently drops them on node-return merge and the
    worker's resolve_final_outcome_payload reads empty state — so final_outcome
    shipped delivery_quality=absent + empty failure lists despite the gate
    computing 'degraded'."""
    from deepsearch.agent.state import AgentState, empty_state

    gate_channels = {
        "delivery_quality",
        "quality_gate_failures",
        "coverage_gate_failures",
    }
    declared = set(AgentState.__annotations__)
    assert gate_channels <= declared, (
        f"gate channels not declared in AgentState (LangGraph will drop them): "
        f"{gate_channels - declared}"
    )
    st = empty_state(mission_id="GUARD")
    for key in gate_channels:
        assert key in st, f"empty_state must initialise {key}"


def test_gate_fields_thread_into_final_outcome_payload() -> None:
    """The seam the worker depends on: gate state keys -> FinalOutcomePayload.

    resolve_final_outcome_payload runs on the worker path (worker/main.py:337)
    and its output is persisted into execution_metadata.final_outcome, visible
    via tracelab_mission get. It must surface the gate's outcome,
    delivery_quality, and populated failure lists that finalize_node wrote onto
    state — not the legacy always-empty fallback.
    """

    # Mimic the finalize-node state update for a VALIDATION_FAILED verdict.
    state = {
        "mission_outcome": MissionOutcome.VALIDATION_FAILED.value,
        "routing_reason": "hit_max_steps_without_final_answer",
        "delivery_quality": DELIVERY_QUALITY_FAILED,
        "quality_gate_failures": ["hit_max_steps_without_final_answer"],
        "coverage_gate_failures": ["shortfall_1.000_across_5_deliverables"],
        "research_loop_count": 1,
    }
    payload = resolve_final_outcome_payload(
        state, completed_at=datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    )

    assert payload["outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert payload["delivery_quality"] == DELIVERY_QUALITY_FAILED
    assert payload["quality_gate_failures"] == ["hit_max_steps_without_final_answer"]
    assert payload["coverage_gate_failures"] == ["shortfall_1.000_across_5_deliverables"]


def test_degraded_complete_payload_threads_delivery_quality() -> None:
    """A COMPLETE+degraded verdict surfaces delivery_quality + lists too."""

    state = {
        "mission_outcome": MissionOutcome.COMPLETE.value,
        "routing_reason": "",
        "delivery_quality": DELIVERY_QUALITY_DEGRADED,
        "quality_gate_failures": ["dead_link_rate_0.203"],
        "coverage_gate_failures": ["deliverable_shortfall_1.000"],
        "research_loop_count": 2,
    }
    payload = resolve_final_outcome_payload(
        state, completed_at=datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    )

    assert payload["outcome"] == MissionOutcome.COMPLETE.value
    assert payload["delivery_quality"] == DELIVERY_QUALITY_DEGRADED
    assert payload["quality_gate_failures"] == ["dead_link_rate_0.203"]
    assert payload["coverage_gate_failures"] == ["deliverable_shortfall_1.000"]


def test_legacy_payload_without_gate_keys_omits_delivery_quality() -> None:
    """Additive contract: states with no gate keys keep the legacy shape."""

    payload = resolve_final_outcome_payload(
        {
            "routing_reason": "converged_with_coverage",
            "coverage_gates": {"passed": True, "failed_gates": []},
            "research_loop_count": 1,
        },
        completed_at=datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert payload["outcome"] == "complete"
    assert "delivery_quality" not in payload


# ---------------------------------------------------------------------------
# S88 — a final-answer generation cut at max_tokens (finish_reason == "length")
# is an unambiguous incompleteness: it must route to VALIDATION_FAILED, unlike a
# cited and untruncated max-step fallback. It also
# carries a coverage shortfall, so a clean recovery still lands COMPLETE; only an
# unrecovered truncation ships VALIDATION_FAILED (a reviewable artifact).
# ---------------------------------------------------------------------------

_CLEAN_COMPLETE_TELEMETRY = {
    "hit_max_steps": False,
    "self_terminated": True,
    "step_count": 20,
    "citation_telemetry": {
        "citations_extracted": 50,
        "orphan_marker_count": 0,
        "distinct_urls_cited": 45,
    },
    "deliverable_coverage": {"shortfall_score": 0.0, "satisfied": ["a", "b"], "missing": []},
    "liveness_telemetry": _liveness(checked=50, live=48, dead=1, orphan=0),
}


def test_output_truncated_routes_to_validation_failed() -> None:
    telemetry = {**_CLEAN_COMPLETE_TELEMETRY, "output_truncated": True}
    result = evaluate_outcome_gate(telemetry, report_word_count=5000)
    assert result.outcome is MissionOutcome.VALIDATION_FAILED
    assert result.routing_reason == "final_answer_truncated_at_max_tokens"
    assert result.delivery_quality == DELIVERY_QUALITY_FAILED
    assert "final_answer_truncated_at_max_tokens" in result.quality_gate_failures


def test_untruncated_same_shape_is_not_failed_by_the_new_trigger() -> None:
    """Backward-compat: the identical envelope WITHOUT output_truncated is a
    COMPLETE report — the new trigger must not fire on absence/False."""
    result = evaluate_outcome_gate(
        dict(_CLEAN_COMPLETE_TELEMETRY), report_word_count=5000
    )
    assert result.outcome is MissionOutcome.COMPLETE
    assert result.routing_reason != "final_answer_truncated_at_max_tokens"
    result_false = evaluate_outcome_gate(
        {**_CLEAN_COMPLETE_TELEMETRY, "output_truncated": False},
        report_word_count=5000,
    )
    assert result_false.outcome is MissionOutcome.COMPLETE


def test_generation_error_recovery_is_complete_only_with_explicit_warning() -> None:
    telemetry = {
        **_CLEAN_COMPLETE_TELEMETRY,
        "generation_failure_error": "safe-generation-failure",
        "synthesis_error": None,
        "generation_error_recovered": True,
        "generation_recovery_attempted": True,
        "generation_recovery_result": "success",
        "generation_recovery_finish_reason": "stop",
    }

    result = evaluate_outcome_gate(telemetry, report_word_count=5000)

    assert result.outcome is MissionOutcome.COMPLETE
    assert result.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING
    assert (
        "generation_error_recovered_from_paid_memory"
        in result.quality_gate_failures
    )


def test_generation_error_recovery_with_zero_citations_never_completes() -> None:
    telemetry = {
        **_CLEAN_COMPLETE_TELEMETRY,
        "citation_telemetry": {
            "citations_extracted": 0,
            "orphan_marker_count": 0,
            "distinct_urls_cited": 0,
        },
        "generation_failure_error": "safe-generation-failure",
        "synthesis_error": None,
        "generation_error_recovered": True,
        "generation_recovery_attempted": True,
        "generation_recovery_result": "success",
        "generation_recovery_finish_reason": "stop",
    }

    result = evaluate_outcome_gate(telemetry, report_word_count=50)

    assert result.outcome is MissionOutcome.VALIDATION_FAILED
    assert result.routing_reason == "zero_citations_on_nontrivial_report"


def test_generation_error_recovery_without_provider_stop_never_completes() -> None:
    telemetry = {
        **_CLEAN_COMPLETE_TELEMETRY,
        "generation_failure_error": "safe-generation-failure",
        "synthesis_error": None,
        "generation_error_recovered": True,
        "generation_recovery_attempted": True,
        "generation_recovery_result": "success",
        "generation_recovery_finish_reason": "",
    }

    result = evaluate_outcome_gate(telemetry, report_word_count=5000)

    assert result.outcome is MissionOutcome.VALIDATION_FAILED
    assert result.routing_reason == "generation_recovery_unproven"


def test_generation_failure_partial_with_retained_error_never_completes() -> None:
    telemetry = {
        **_CLEAN_COMPLETE_TELEMETRY,
        "generation_failure_error": "safe-generation-failure",
        "synthesis_error": "safe-generation-failure",
        "generation_error_recovered": False,
    }

    result = evaluate_outcome_gate(telemetry, report_word_count=5000)

    assert result.outcome is MissionOutcome.VALIDATION_FAILED
    assert result.routing_reason == "generation_failure_unrecovered"
    assert "generation_failure_unrecovered" in result.quality_gate_failures


def test_generation_failure_replaced_by_bounded_pass_completes_with_warning() -> None:
    telemetry = {
        **_CLEAN_COMPLETE_TELEMETRY,
        "generation_failure_error": "safe-generation-failure",
        "synthesis_error": None,
        "generation_error_recovered": False,
        "recovery": {
            "status": "completed",
            "selected_artifact": "recovery",
        },
    }

    result = evaluate_outcome_gate(telemetry, report_word_count=5000)

    assert result.outcome is MissionOutcome.COMPLETE
    assert result.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING
    assert (
        "generation_failure_recovered_by_bounded_pass"
        in result.quality_gate_failures
    )


def test_generation_recovered_flag_without_full_proof_never_completes() -> None:
    telemetry = {
        **_CLEAN_COMPLETE_TELEMETRY,
        "generation_error_recovered": True,
        "generation_recovery_finish_reason": "stop",
    }

    result = evaluate_outcome_gate(telemetry, report_word_count=5000)

    assert result.outcome is MissionOutcome.VALIDATION_FAILED
    assert result.routing_reason == "generation_recovery_unproven"


def test_generation_failure_without_selected_replacement_never_completes() -> None:
    telemetry = {
        **_CLEAN_COMPLETE_TELEMETRY,
        "generation_failure_error": "safe-generation-failure",
        "synthesis_error": None,
        "generation_error_recovered": False,
        "recovery": {
            "status": "rejected_regression",
            "selected_artifact": "first_pass",
        },
    }

    result = evaluate_outcome_gate(telemetry, report_word_count=5000)

    assert result.outcome is MissionOutcome.VALIDATION_FAILED
    assert result.routing_reason == "generation_recovery_unproven"
