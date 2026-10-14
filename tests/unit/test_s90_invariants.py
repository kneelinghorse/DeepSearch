"""S90 carry-forward repairs.

Three changes, verified here:

(a) #957 — ``validation_context_truncated`` fires only on real evidence loss.
    The warning previously keyed off ``context_truncation.occurred``, which is
    True whenever ANY packer truncation happened — including the uncited-tail
    eviction the grounding packer is designed to perform. It downgrades
    delivery_quality from 'clean' to 'usable_with_warning', so on a
    source-rich mission it fired every time and systematically understated
    quality. Empirical anchor: mission 7cc63118 (pedr-retrieval-evaluation-R001)
    landed cited_dropped=0 with sources_dropped=127 out of 201 available, at
    159,988 of a 160,000-char budget — a perfectly grounded critique that the
    old gate flagged.

(b) #952 — an exhausted ``queued`` row is dead-lettered instead of stranded.
    The claim predicate caps attempts on 'queued' rows too, while the sweep
    only covered 'in_progress'. A row re-queued at or past max_attempts was
    therefore never claimed AND never dead-lettered: an invisible zombie.

(c) #959 — a transient, zero-token synthesis failure returns the mission to
    'queued' so the lease's remaining attempts become reachable, instead of
    landing terminally in 'blocked'.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from deepsearch.agent.outcome import (
    DELIVERY_QUALITY_CLEAN,
    DELIVERY_QUALITY_USABLE_WITH_WARNING,
    MissionOutcome,
    evaluate_outcome_gate,
)
from deepsearch.agent.quality_series import build_quality_record
from deepsearch.worker.lease import LeaseAttempt, LeasePolicy
from deepsearch.worker.main import DeepSearchWorker
from deepsearch.worker.poller import MissionPoller
from deepsearch.worker.writer import ResultWriter

# ---------------------------------------------------------------------------
# (a) #957 — evidence-loss gating for validation_context_truncated.
# ---------------------------------------------------------------------------


def _telemetry(*, grounding: dict[str, Any]) -> dict[str, Any]:
    """A clean, warning-free telemetry payload with a configurable grounding block."""
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
            "live": 3,
            "dead": 0,
            "orphan": 0,
            "unknown": 0,
            "hallucinated_url_count": 0,
        },
        "critique_telemetry": {
            "assessment": {
                "version": 1, "status": "partial", "assessed": 1, "supported": 1,
                "contradicted": 0, "unassessed": 0,
            },
            "claim_count": 20,
            "gap_notes_inserted": 0,
            "error": None,
            "annotation_summary": {
                "total": 0,
                "applied": 0,
                "skipped": 0,
                "dropped": 0,
                "by_reason": {},
                "reconciled": True,
            },
            "grounding": grounding,
        },
        "critique_skipped": False,
        "tool_outcomes": {
            "by_tool": {
                "web_search": {
                    "calls": 1,
                    "ok": 1,
                    "error": 0,
                    "results_total": 3,
                    "circuit_open": 0,
                    "by_category": {},
                }
            },
            "records": [],
            "records_truncated": 0,
        },
        "recovery_passes": 0,
        "recovery": {
            "passes": 0,
            "status": "not_triggered",
            "selected_artifact": "first_pass",
            "decision": {"triggered": False},
        },
        "retrieval_state": {
            "discovered": 3,
            "fetched_readable": 2,
            "fetch_attempts": 2,
            "probes": 1,
            "grounding_sources": 3,
        },
    }


# The real numbers from mission 7cc63118 — a fully-grounded critique whose
# uncited corroborators were evicted by the budget, exactly as designed.
_R001_GROUNDING = {
    "cited_available": 19,
    "cited_included": 19,
    "cited_dropped": 0,
    "cited_live_available": 19,
    "cited_live_included": 19,
    "sources_available": 201,
    "sources_included": 74,
    "sources_dropped": 127,
    "sources_truncated": 21,
    "draft_chars_dropped": 0,
    "liveness_chars_dropped": 0,
}


def test_uncited_tail_eviction_is_not_a_quality_warning() -> None:
    """127 uncited sources evicted from a packed budget must stay 'clean'.

    This is the packer working: cited-first ordering, hard char budget,
    overflow drops uncited. Flagging it told operators nothing and cost the
    mission its clean rating.
    """
    result = evaluate_outcome_gate(
        _telemetry(grounding=dict(_R001_GROUNDING)), report_word_count=1200
    )

    assert result.delivery_quality == DELIVERY_QUALITY_CLEAN
    assert "validation_context_truncated" not in result.quality_gate_failures


def test_dropping_a_cited_source_still_warns() -> None:
    """cited_dropped > 0 means critique judged a claim against evidence it never saw."""
    grounding = dict(_R001_GROUNDING, cited_included=17, cited_dropped=2)

    result = evaluate_outcome_gate(
        _telemetry(grounding=grounding), report_word_count=1200
    )

    assert result.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING
    assert "validation_context_truncated" in result.quality_gate_failures


def test_dropping_draft_or_liveness_context_still_warns() -> None:
    """A truncated draft or liveness table is a real critique blind spot."""
    for field in ("draft_chars_dropped", "liveness_chars_dropped"):
        grounding = dict(_R001_GROUNDING, **{field: 500})
        result = evaluate_outcome_gate(
            _telemetry(grounding=grounding), report_word_count=1200
        )
        assert "validation_context_truncated" in result.quality_gate_failures, field


def test_quality_record_exposes_cited_dropped_and_evidence_loss() -> None:
    """The gate can only distinguish the two cases if the record carries them.

    ``occurred`` is retained unchanged so the historical series stays
    comparable across the semantic change.
    """
    record = build_quality_record(
        _telemetry(grounding=dict(_R001_GROUNDING)), 1200
    )
    truncation = record["context_truncation"]

    assert truncation["critique_cited_dropped"] == 0
    assert truncation["critique_sources_dropped"] == 127
    assert truncation["evidence_loss_occurred"] is False
    # Unchanged legacy field: any truncation at all still sets it.
    assert truncation["occurred"] is True

    lossy = build_quality_record(
        _telemetry(grounding=dict(_R001_GROUNDING, cited_dropped=2)), 1200
    )
    assert lossy["context_truncation"]["evidence_loss_occurred"] is True


# ---------------------------------------------------------------------------
# (b) #952 — the zombie-queued trap.
# ---------------------------------------------------------------------------


def test_dead_letter_sweep_covers_exhausted_queued_rows() -> None:
    """An exhausted 'queued' row must be reachable by the sweep.

    The claim predicate caps attempts on 'queued' as well as 'in_progress'
    (poller.CLAIM_MISSION_SQL), so without this a re-queued row at max_attempts
    is claimed by nobody and swept by nobody — it simply disappears.
    """
    sql = " ".join(MissionPoller.DEAD_LETTER_EXHAUSTED_SQL.split())

    assert "status = 'queued'" in sql, (
        "sweep must cover queued rows or #959's requeue can strand a mission"
    )
    assert "status = 'in_progress'" in sql, "in_progress coverage must be retained"
    assert "deepsearch_attempt_count, 0) >= $1" in sql


def test_claim_still_caps_attempts_on_queued_rows() -> None:
    """The cap stays — it is what bounds #959's automatic requeue.

    Superseding note: next-step #952 originally preferred exempting 'queued'
    from the attempt cap, treating a re-queue as a human reset. That predates
    #959. With the worker able to requeue itself, an exemption would make the
    requeue unbounded, so the cap is kept and the sweep was widened instead.
    """
    sql = " ".join(MissionPoller.CLAIM_MISSION_SQL.split())

    assert "COALESCE(deepsearch_attempt_count, 0) < $4" in sql


# ---------------------------------------------------------------------------
# (c) #959 — transient requeue.
# ---------------------------------------------------------------------------


def _lease(attempt_count: int) -> LeaseAttempt:
    now = datetime(2026, 8, 13, 12, 0, 0)
    return LeaseAttempt(
        owner="worker-1",
        token="tok",
        leased_at=now,
        heartbeat_at=now,
        expires_at=now,
        attempt_count=attempt_count,
    )


def _worker(max_attempts: int | None = 3) -> Any:
    """A DeepSearchWorker shell carrying only what the predicate reads."""
    worker = DeepSearchWorker.__new__(DeepSearchWorker)
    worker.lease_policy = LeasePolicy(
        lease_seconds=120, heartbeat_seconds=30, max_attempts=max_attempts
    )
    return worker


_TRANSIENT_STATE = {"synthesis_telemetry": {"synthesis_error_transient": True}}


def test_transient_blocked_failure_with_attempts_left_is_requeued() -> None:
    assert (
        _worker()._should_requeue_transient(
            MissionOutcome.BLOCKED, _TRANSIENT_STATE, _lease(1)
        )
        is True
    )


def test_exhausted_attempts_are_not_requeued() -> None:
    """Requeuing at the cap strands the row until the sweep — noise, not recovery."""
    assert (
        _worker(max_attempts=3)._should_requeue_transient(
            MissionOutcome.BLOCKED, _TRANSIENT_STATE, _lease(3)
        )
        is False
    )


def test_non_transient_blocked_failure_is_not_requeued() -> None:
    for telemetry in (
        {"synthesis_telemetry": {"synthesis_error_transient": False}},
        {"synthesis_telemetry": {}},
        {},
    ):
        assert (
            _worker()._should_requeue_transient(
                MissionOutcome.BLOCKED, telemetry, _lease(1)
            )
            is False
        ), telemetry


def test_validation_failed_is_never_requeued() -> None:
    """It produced a reviewable artifact — re-running would discard it."""
    assert (
        _worker()._should_requeue_transient(
            MissionOutcome.VALIDATION_FAILED, _TRANSIENT_STATE, _lease(1)
        )
        is False
    )


def test_unusable_lease_policy_refuses_to_requeue() -> None:
    """Without a known ceiling the requeue is unbounded — fail terminal instead."""
    for bad in (None, 0, -1):
        assert (
            _worker(max_attempts=bad)._should_requeue_transient(
                MissionOutcome.BLOCKED, _TRANSIENT_STATE, _lease(1)
            )
            is False
        ), bad


def test_requeue_sql_preserves_attempt_count_and_is_lease_guarded() -> None:
    """The attempt count is the only thing bounding the requeue loop."""
    sql = " ".join(ResultWriter.MARK_REQUEUED_SQL.split())

    assert "status = 'queued'" in sql
    assert "deepsearch_attempt_count =" not in sql.split("WHERE")[0], (
        "requeue must NOT reset the attempt count — that would unbound the loop"
    )
    assert "AND deepsearch_attempt_count = $5" in sql, "lease-guarded on attempt"
    assert "AND deepsearch_lease_token = $4" in sql
    assert "completed_at" not in sql, "a requeue is not a completion"
