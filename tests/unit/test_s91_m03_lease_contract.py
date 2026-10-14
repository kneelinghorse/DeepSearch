"""Sprint 91 m03 frozen TraceLab lease-v2 contract invariants."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deepsearch.worker.lease import (
    LEASE_COLUMN_TYPES,
    LEASE_CONTRACT_REVISION,
    LEASE_CONTRACT_VERSION,
    LEASE_INDEX_SPECS,
    LEASE_NON_NULL_DEFAULTS,
    LEASE_NULLABLE_COLUMNS,
    LeaseAttempt,
    LeaseIndexSpec,
    LeasePolicy,
)
from deepsearch.worker.poller import MissionPoller
from deepsearch.worker.writer import ResultWriter


def _compact_sql(sql: str) -> str:
    return " ".join(sql.split())


def _lease() -> LeaseAttempt:
    claimed_at = datetime(2026, 8, 14, 6, 0, tzinfo=timezone.utc)
    return LeaseAttempt(
        owner="worker-s91",
        token="opaque-s91-token",
        leased_at=claimed_at,
        heartbeat_at=claimed_at,
        expires_at=claimed_at + timedelta(seconds=120),
        attempt_count=2,
    )


def test_contract_revision_advances_beyond_the_stale_s87_contract() -> None:
    """Readiness and execution telemetry must identify the same new policy."""

    assert LEASE_CONTRACT_REVISION == 2
    assert LEASE_CONTRACT_VERSION == "tracelab-missions-lease-v2"


def test_migration_039_column_constraints_are_machine_readable() -> None:
    """Matching types alone cannot prove terminal reset and first claim are safe."""

    assert LEASE_NULLABLE_COLUMNS == {
        "deepsearch_lease_owner",
        "deepsearch_lease_token",
        "deepsearch_leased_at",
        "deepsearch_heartbeat_at",
        "deepsearch_lease_expires_at",
        "deepsearch_result_key",
    }
    assert LEASE_NON_NULL_DEFAULTS == {"deepsearch_attempt_count": "0"}
    assert set(LEASE_COLUMN_TYPES) == (
        set(LEASE_NULLABLE_COLUMNS) | set(LEASE_NON_NULL_DEFAULTS)
    )
    assert LEASE_COLUMN_TYPES["deepsearch_attempt_count"] == {"integer"}


def test_migration_039_indexes_pin_global_claim_order() -> None:
    """A project-leading claim index would silently reintroduce scoped claims."""

    assert LEASE_INDEX_SPECS == {
        "missions_deepsearch_lease_token_active_uq": LeaseIndexSpec(
            ("deepsearch_lease_token",),
            True,
            "deepsearch_lease_token IS NOT NULL",
        ),
        "missions_deepsearch_result_key_uq": LeaseIndexSpec(
            ("deepsearch_result_key",),
            True,
            "deepsearch_result_key IS NOT NULL",
        ),
        "missions_deepsearch_claim_scan_idx": LeaseIndexSpec(
            ("status", "deepsearch_lease_expires_at", "queued_at"),
            False,
            None,
        ),
    }
    claim_columns = LEASE_INDEX_SPECS[
        "missions_deepsearch_claim_scan_idx"
    ].columns
    assert claim_columns[0] == "status"
    assert "project_id" not in claim_columns


def test_claim_is_global_non_orphan_and_terminal_rows_are_not_resubmitted() -> None:
    """One execution stays fenced to one non-orphan mission row."""

    sql = _compact_sql(MissionPoller.CLAIM_MISSION_SQL)
    assert "WHERE project_id IS NOT NULL" in sql
    assert "project_id = $" not in sql
    assert "TRACELAB_PROJECT_ID" not in sql
    assert "status = 'queued'" in sql
    assert "status = 'in_progress'" in sql
    for terminal_status in ("completed", "validation_failed", "blocked"):
        assert f"status = '{terminal_status}'" not in sql


def test_terminal_writes_clear_token_and_expiry_and_store_result_key() -> None:
    """A terminal row cannot retain an active ownership proof."""

    terminal_sql = (
        ResultWriter.MARK_COMPLETED_SQL,
        ResultWriter.MARK_VALIDATION_FAILED_SQL,
        ResultWriter.MARK_BLOCKED_SQL,
    )
    for sql in terminal_sql:
        terminal_update = sql[sql.index("UPDATE missions") :]
        update_clause = terminal_update.split("WHERE", 1)[0]
        assert "deepsearch_lease_token = NULL" in update_clause
        assert "deepsearch_lease_expires_at = NULL" in update_clause
        assert "deepsearch_result_key = $" in update_clause


def test_nonterminal_requeue_never_consumes_a_terminal_result_key() -> None:
    """TraceLab may materialize a result key, so a retry transition cannot own one."""

    sql = _compact_sql(ResultWriter.MARK_REQUEUED_SQL)
    update_clause = sql.split("WHERE", 1)[0]
    assert "status = 'queued'" in update_clause
    assert "deepsearch_result_key = NULL" in update_clause
    assert "deepsearch_result_key = $" not in sql
    assert "deepsearch_attempt_count = $5" in sql
    assert "error_message = $2" in sql
    assert "deepsearch_lease_owner IS NULL" in sql
    assert "deepsearch_lease_token IS NULL" in sql
    assert "deepsearch_lease_expires_at IS NULL" in sql


@pytest.mark.asyncio
async def test_requeue_runtime_does_not_pass_a_result_key() -> None:
    """The writer call shape must agree with the nonterminal SQL contract."""

    fetchval = AsyncMock(return_value="mission-uuid")
    writer = ResultWriter(SimpleNamespace(fetchval=fetchval))  # type: ignore[arg-type]
    lease = _lease()
    from deepsearch.worker.writer import ExecutionMetadata

    metadata = ExecutionMetadata(total_tokens=0, duration_seconds=1, agent_version="test")

    assert await writer.mark_requeued(
        "mission-uuid",
        "transient upstream outage",
        lease=lease,
        execution_metadata=metadata,
    )

    fetchval.assert_awaited_once_with(
        ResultWriter.MARK_REQUEUED_SQL,
        "mission-uuid",
        "transient upstream outage",
        lease.owner,
        lease.token,
        lease.attempt_count,
        json.dumps(metadata.to_dict(), ensure_ascii=False),
    )


def test_attempt_audit_labels_immutable_timestamps_as_claim_time() -> None:
    """An initial expiry must not masquerade as the final heartbeat snapshot."""

    lease = _lease()
    audit = lease.audit_dict("mission-uuid")
    serialized = json.dumps(audit, sort_keys=True)

    assert audit["contract_version"] == LEASE_CONTRACT_VERSION
    assert audit["contract_revision"] == LEASE_CONTRACT_REVISION
    assert audit["snapshot_phase"] == "claim"
    assert audit["claim_leased_at"] == lease.leased_at.isoformat()
    assert audit["claim_heartbeat_at"] == lease.heartbeat_at.isoformat()
    assert audit["claim_expires_at"] == lease.expires_at.isoformat()
    assert "leased_at" not in audit
    assert "heartbeat_at" not in audit
    assert "expires_at" not in audit
    assert lease.token not in serialized
    assert len(audit["token_fingerprint"]) == 12


@pytest.mark.asyncio
async def test_heartbeat_audit_reports_last_database_confirmed_extension() -> None:
    """Terminal telemetry must not relabel the original claim as the final beat."""

    first = datetime(2026, 8, 14, 6, 0, 30, tzinfo=timezone.utc)
    second = first + timedelta(seconds=30)
    fetchval = AsyncMock(side_effect=[first, second])
    poller = MissionPoller(
        SimpleNamespace(fetchval=fetchval),  # type: ignore[arg-type]
        lease_owner="worker-s91",
        lease_policy=LeasePolicy(120, 30, 3),
    )
    lease = _lease()

    assert await poller.heartbeat("mission-uuid", lease) is True
    assert await poller.heartbeat("mission-uuid", lease) is True

    audit = poller.heartbeat_audit("mission-uuid", lease)
    assert audit == {
        "snapshot_phase": "pre_terminal",
        "heartbeat_count": 2,
        "last_confirmed_heartbeat_at": second.isoformat(),
        "last_confirmed_expires_at": (
            second + timedelta(seconds=120)
        ).isoformat(),
    }
    assert "RETURNING deepsearch_heartbeat_at" in MissionPoller.HEARTBEAT_SQL

    poller.clear_heartbeat_audit("mission-uuid", lease)
    assert poller.heartbeat_audit("mission-uuid", lease)["heartbeat_count"] == 0


def test_attempt_accounting_survives_requeue_and_every_terminal_write() -> None:
    """A successful later lease cannot erase the compiler/failed-call spend before it."""
    sql = _compact_sql(ResultWriter.MARK_REQUEUED_SQL)
    mutation, replay = sql.split("UNION ALL", 1)
    assert "jsonb_build_array($6::jsonb)" in mutation
    assert "jsonb_build_array" not in replay  # Idempotent replay never appends twice.
    for terminal in (ResultWriter.MARK_COMPLETED_SQL, ResultWriter.MARK_VALIDATION_FAILED_SQL,
                     ResultWriter.MARK_BLOCKED_SQL):
        assert "execution_metadata->'prior_execution_attempts'" in terminal
