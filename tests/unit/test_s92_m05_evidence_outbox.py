"""Durability contract for the migration-043 Evidence Ledger outbox."""

from __future__ import annotations

import json
from collections.abc import Iterable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deepsearch.tracelab.evidence_ledger import EvidenceLedgerDeliveryResult
from deepsearch.worker.evidence_outbox import (
    EVIDENCE_OUTBOX_CLAIM_LIMIT,
    EVIDENCE_OUTBOX_MAX_ATTEMPTS,
    EvidenceOutboxClaim,
    EvidenceOutboxDispatcher,
    EvidenceOutboxError,
    EvidenceOutboxRepository,
    evidence_outbox_backoff_seconds,
)
from deepsearch.worker.writer import ResultWriter

MISSION_UUID = "11111111-2222-4333-8444-555555555555"


def _claim(*, attempt: int = 1, ordinal: int = 0) -> EvidenceOutboxClaim:
    return EvidenceOutboxClaim(
        mission_uuid=MISSION_UUID,
        deepsearch_job_id=f"job-{ordinal}",
        lease_token=f"22222222-3333-4444-8555-{ordinal:012d}",
        delivery_attempt_count=attempt,
    )


def _result(
    *,
    success: bool,
    retryable: bool,
    category: str,
    status: int | None,
) -> EvidenceLedgerDeliveryResult:
    return EvidenceLedgerDeliveryResult(
        success=success,
        attempts=1,
        retryable=retryable,
        category=category,  # type: ignore[arg-type]
        final_status_code=status,
    )


class _Transport:
    max_attempts = 1

    def __init__(self, results: Iterable[EvidenceLedgerDeliveryResult]) -> None:
        self.results = list(results)
        self.triggers = []

    async def deliver(self, trigger):
        self.triggers.append(trigger)
        return self.results.pop(0)


class _Repository:
    def __init__(self, claims: list[EvidenceOutboxClaim]) -> None:
        self.claims = claims
        self.acknowledged = []
        self.retried = []
        self.dead = []
        self.claim_calls = []

    async def sweep_exhausted(self) -> int:
        return 0

    async def claim_due(self, *, limit: int, lease_seconds: float):
        self.claim_calls.append((limit, lease_seconds))
        claims, self.claims = self.claims, []
        return claims

    async def state_counts(self) -> dict[str, int]:
        return {
            "pending": len(self.retried),
            "leased": 0,
            "acked": len(self.acknowledged),
            "dead_letter": len(self.dead),
            "invalid_exhausted_pending": 0,
        }

    async def acknowledge(self, claim, *, status_code: int) -> bool:
        self.acknowledged.append((claim, status_code))
        return True

    async def release_retry(
        self,
        claim,
        *,
        delay_seconds: float,
        status_code: int | None,
        error_code: str,
    ) -> str:
        self.retried.append(
            (claim, delay_seconds, status_code, error_code)
        )
        return (
            "dead_letter"
            if claim.delivery_attempt_count >= EVIDENCE_OUTBOX_MAX_ATTEMPTS
            else "pending"
        )

    async def dead_letter(
        self,
        claim,
        *,
        status_code: int | None,
        error_code: str,
    ) -> bool:
        self.dead.append((claim, status_code, error_code))
        return True


class TestTerminalEnrollmentSql:
    """The terminal result and delivery intent must share one SQL statement."""

    @pytest.mark.parametrize(
        ("sql", "status", "job_placeholder", "key_placeholder", "attempt_placeholder"),
        (
            (
                ResultWriter.MARK_COMPLETED_SQL,
                "completed",
                "$10",
                "$9",
                "$8",
            ),
            (
                ResultWriter.MARK_VALIDATION_FAILED_SQL,
                "validation_failed",
                "$11",
                "$10",
                "$9",
            ),
        ),
    )
    def test_eligible_terminal_updates_enroll_exact_immutable_snapshot(
        self,
        sql: str,
        status: str,
        job_placeholder: str,
        key_placeholder: str,
        attempt_placeholder: str,
    ) -> None:
        """A terminal transition is unreachable when reused identity drifts."""

        assert "WITH eligible AS MATERIALIZED" in sql
        assert "FOR UPDATE" in sql
        assert "INSERT INTO deepsearch_evidence_outbox" in sql
        assert "ON CONFLICT (mission_id, deepsearch_job_id) DO NOTHING" in sql
        assert f"existing.deepsearch_job_id = {job_placeholder}" in sql
        assert f"existing.deepsearch_result_key = {key_placeholder}" in sql
        assert f"existing.mission_attempt_count = {attempt_placeholder}" in sql
        assert f"existing.terminal_status = '{status}'" in sql
        assert "existing.schema_version = 1" in sql
        assert f"mission.deepsearch_attempt_count = {attempt_placeholder}" in sql
        assert sql.index("INSERT INTO deepsearch_evidence_outbox") < sql.index(
            "UPDATE missions AS mission"
        )

    def test_blocked_terminal_path_never_enrolls_evidence(self) -> None:
        """Only completed and validation-failed persisted results are eligible."""

        assert "deepsearch_evidence_outbox" not in ResultWriter.MARK_BLOCKED_SQL

    def test_idempotent_branch_cannot_backfill_a_historical_terminal_row(self) -> None:
        """The read-only retry requires an outbox row; it never creates one."""

        sql = ResultWriter.MARK_COMPLETED_SQL
        assert sql.count("INSERT INTO deepsearch_evidence_outbox") == 1
        assert "FROM eligible\n            ON CONFLICT" in sql
        final_branch = sql.rsplit("UNION ALL", maxsplit=1)[-1]
        assert "JOIN deepsearch_evidence_outbox AS existing" in final_branch


class TestEvidenceOutboxRepository:
    @pytest.mark.asyncio
    async def test_claim_maps_database_identity_and_never_exceeds_25(self) -> None:
        """The SQL limit and Python boundary independently cap contention."""

        db = SimpleNamespace(
            fetchall=AsyncMock(
                return_value=[
                    {
                        "mission_id": MISSION_UUID,
                        "deepsearch_job_id": "job-1",
                        "lease_token": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
                        "delivery_attempt_count": 2,
                    }
                ]
            )
        )
        repository = EvidenceOutboxRepository(db)  # type: ignore[arg-type]

        claims = await repository.claim_due(limit=25, lease_seconds=60)

        assert claims == [
            EvidenceOutboxClaim(
                mission_uuid=MISSION_UUID,
                deepsearch_job_id="job-1",
                lease_token="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
                delivery_attempt_count=2,
            )
        ]
        db.fetchall.assert_awaited_once_with(
            EvidenceOutboxRepository.CLAIM_DUE_SQL,
            25,
            60.0,
        )
        assert "LIMIT LEAST($1::integer, 25)" in repository.CLAIM_DUE_SQL
        assert "FOR UPDATE SKIP LOCKED" in repository.CLAIM_DUE_SQL
        assert "gen_random_uuid()" in repository.CLAIM_DUE_SQL
        assert "delivery_attempt_count < 12" in repository.CLAIM_DUE_SQL
        assert "state = 'pending'" in repository.CLAIM_DUE_SQL
        assert "state = 'leased'" in repository.CLAIM_DUE_SQL

    @pytest.mark.asyncio
    async def test_claim_refuses_an_invalid_bound_before_database_access(self) -> None:
        db = SimpleNamespace(fetchall=AsyncMock())
        repository = EvidenceOutboxRepository(db)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="between 1 and 25"):
            await repository.claim_due(limit=26)

        db.fetchall.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_claim_refuses_corrupt_attempt_count(self) -> None:
        """A catalog violation must fail loudly instead of over-delivering."""

        db = SimpleNamespace(
            fetchall=AsyncMock(
                return_value=[
                    {
                        "mission_id": MISSION_UUID,
                        "deepsearch_job_id": "job-1",
                        "lease_token": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
                        "delivery_attempt_count": 13,
                    }
                ]
            )
        )
        repository = EvidenceOutboxRepository(db)  # type: ignore[arg-type]

        with pytest.raises(EvidenceOutboxError, match="1..12"):
            await repository.claim_due()

    @pytest.mark.asyncio
    async def test_ack_is_token_fenced_and_accepts_only_2xx(self) -> None:
        db = SimpleNamespace(fetchval=AsyncMock(return_value="acked"))
        repository = EvidenceOutboxRepository(db)  # type: ignore[arg-type]
        claim = _claim()

        assert await repository.acknowledge(claim, status_code=201)
        db.fetchval.assert_awaited_once_with(
            EvidenceOutboxRepository.ACK_SQL,
            claim.mission_uuid,
            claim.deepsearch_job_id,
            claim.lease_token,
            201,
        )
        assert "state = 'leased'" in repository.ACK_SQL
        assert "lease_token = $3" in repository.ACK_SQL

        with pytest.raises(ValueError, match="2xx"):
            await repository.acknowledge(claim, status_code=409)

    @pytest.mark.asyncio
    async def test_retry_release_dead_letters_the_twelfth_attempt_in_sql(self) -> None:
        """A crash or caller mistake cannot create a thirteenth delivery."""

        db = SimpleNamespace(fetchval=AsyncMock(return_value="dead_letter"))
        repository = EvidenceOutboxRepository(db)  # type: ignore[arg-type]
        claim = _claim(attempt=12)

        state = await repository.release_retry(
            claim,
            delay_seconds=300,
            status_code=503,
            error_code="server_error",
        )

        assert state == "dead_letter"
        assert "delivery_attempt_count >= 12" in repository.RELEASE_RETRY_SQL
        assert "state = 'leased'" in repository.RELEASE_RETRY_SQL
        assert "lease_token = $3" in repository.RELEASE_RETRY_SQL

    @pytest.mark.asyncio
    async def test_expired_twelfth_claim_is_swept_after_a_process_crash(self) -> None:
        """A crash on the final lease cannot strand an exhausted row forever."""

        db = SimpleNamespace(fetchval=AsyncMock(return_value=1))
        repository = EvidenceOutboxRepository(db)  # type: ignore[arg-type]

        assert await repository.sweep_exhausted() == 1
        assert "delivery_attempt_count >= 12" in repository.SWEEP_EXHAUSTED_SQL
        assert "lease_expires_at <= NOW()" in repository.SWEEP_EXHAUSTED_SQL
        assert "LIMIT 25" in repository.SWEEP_EXHAUSTED_SQL
        assert "FOR UPDATE SKIP LOCKED" in repository.SWEEP_EXHAUSTED_SQL
        assert "outbox.lease_token = candidates.lease_token" in (
            repository.SWEEP_EXHAUSTED_SQL
        )
        assert "state = 'pending'" not in repository.SWEEP_EXHAUSTED_SQL
        assert "last_error_code = 'attempts_exhausted'" in (
            repository.SWEEP_EXHAUSTED_SQL
        )

    @pytest.mark.asyncio
    async def test_state_counts_are_bounded_to_identity_free_aggregates(self) -> None:
        db = SimpleNamespace(
            fetchall=AsyncMock(
                return_value=[
                    {
                        "state": "pending",
                        "row_count": 3,
                        "invalid_exhausted_pending": 0,
                    },
                    {
                        "state": "dead_letter",
                        "row_count": 1,
                        "invalid_exhausted_pending": 0,
                    },
                ]
            )
        )
        repository = EvidenceOutboxRepository(db)  # type: ignore[arg-type]

        assert await repository.state_counts() == {
            "pending": 3,
            "leased": 0,
            "acked": 0,
            "dead_letter": 1,
            "invalid_exhausted_pending": 0,
        }
        db.fetchall.assert_awaited_once_with(repository.STATE_COUNTS_SQL)
        assert "mission_id" not in repository.STATE_COUNTS_SQL
        assert "deepsearch_job_id" not in repository.STATE_COUNTS_SQL


class TestEvidenceOutboxDispatcher:
    @pytest.mark.asyncio
    async def test_one_bounded_batch_acknowledges_201_and_exact_replay_200(self) -> None:
        """Both canonical initial capture and replay are durable acknowledgements."""

        repository = _Repository([_claim(ordinal=1), _claim(ordinal=2)])
        transport = _Transport(
            [
                _result(
                    success=True,
                    retryable=False,
                    category="accepted",
                    status=201,
                ),
                _result(
                    success=True,
                    retryable=False,
                    category="accepted",
                    status=200,
                ),
            ]
        )
        dispatcher = EvidenceOutboxDispatcher(  # type: ignore[arg-type]
            repository,
            transport,
        )

        assert await dispatcher.dispatch_once() == 2

        assert [status for _claim_value, status in repository.acknowledged] == [
            201,
            200,
        ]
        assert dispatcher.stats["acknowledged"] == 2
        assert dispatcher.stats["claimed"] == 2
        assert dispatcher.stats["acked"] == 2
        assert dispatcher.stats["overflowed"] == 0
        assert dispatcher.stats["discarded"] == 0
        assert repository.claim_calls == [(EVIDENCE_OUTBOX_CLAIM_LIMIT, 60.0)]
        assert [trigger.deepsearch_job_id for trigger in transport.triggers] == [
            "job-1",
            "job-2",
        ]

    @pytest.mark.asyncio
    async def test_retryable_failure_releases_with_bounded_backoff(self) -> None:
        """Network/429/5xx failures return to pending without losing identity."""

        repository = _Repository([_claim(attempt=4)])
        transport = _Transport(
            [
                _result(
                    success=False,
                    retryable=True,
                    category="server_error",
                    status=503,
                )
            ]
        )
        dispatcher = EvidenceOutboxDispatcher(  # type: ignore[arg-type]
            repository,
            transport,
        )

        await dispatcher.dispatch_once()

        assert len(repository.retried) == 1
        _claim_value, delay, status, error = repository.retried[0]
        assert (delay, status, error) == (8.0, 503, "server_error")
        assert dispatcher.stats["retry_scheduled"] == 1
        assert dispatcher.stats["dead_lettered"] == 0

    @pytest.mark.asyncio
    async def test_retryable_twelfth_failure_is_dead_lettered(
        self,
        caplog,
    ) -> None:
        repository = _Repository([_claim(attempt=12)])
        transport = _Transport(
            [
                _result(
                    success=False,
                    retryable=True,
                    category="rate_limited",
                    status=429,
                )
            ]
        )
        dispatcher = EvidenceOutboxDispatcher(  # type: ignore[arg-type]
            repository,
            transport,
        )

        await dispatcher.dispatch_once()

        assert dispatcher.stats["dead_lettered"] == 1
        assert dispatcher.stats["retry_scheduled"] == 0
        assert "category=attempts_exhausted" in caplog.text
        assert "attempt=12" in caplog.text
        assert MISSION_UUID not in caplog.text
        assert "job-0" not in caplog.text

    @pytest.mark.asyncio
    async def test_deterministic_409_dead_letters_without_retry(
        self,
        caplog,
    ) -> None:
        """Persisted identity drift requires operator attention, not replay churn."""

        repository = _Repository([_claim(attempt=1)])
        transport = _Transport(
            [
                _result(
                    success=False,
                    retryable=False,
                    category="identity_mismatch",
                    status=409,
                )
            ]
        )
        dispatcher = EvidenceOutboxDispatcher(  # type: ignore[arg-type]
            repository,
            transport,
        )

        await dispatcher.dispatch_once()

        assert len(repository.dead) == 1
        assert not repository.retried
        assert dispatcher.stats["dead_lettered"] == 1
        assert "category=identity_mismatch" in caplog.text
        assert MISSION_UUID not in caplog.text
        assert "job-0" not in caplog.text

    def test_durable_dispatcher_rejects_nested_http_retries(self) -> None:
        """Twelve outbox attempts must mean at most twelve HTTP attempts."""

        repository = _Repository([])
        transport = _Transport([])
        transport.max_attempts = 4

        with pytest.raises(ValueError, match="one HTTP attempt"):
            EvidenceOutboxDispatcher(  # type: ignore[arg-type]
                repository,
                transport,
            )

    def test_durable_dispatcher_rejects_transport_without_attempt_contract(
        self,
    ) -> None:
        """Missing metadata cannot silently weaken the one-wire-call bound."""

        repository = _Repository([])
        transport = SimpleNamespace(deliver=AsyncMock())

        with pytest.raises(ValueError, match="one HTTP attempt"):
            EvidenceOutboxDispatcher(  # type: ignore[arg-type]
                repository,
                transport,
            )

    @pytest.mark.asyncio
    async def test_background_dispatch_exception_logs_only_category(
        self,
        caplog,
    ) -> None:
        """The long-running loop survives without logging exception content."""

        dispatcher = EvidenceOutboxDispatcher(  # type: ignore[arg-type]
            _Repository([]),
            _Transport([]),
            poll_seconds=0.01,
        )

        async def explode() -> int:
            dispatcher._stop_event.set()
            raise RuntimeError(f"sensitive failure for {MISSION_UUID} job-0")

        dispatcher.dispatch_once = explode  # type: ignore[method-assign]
        await dispatcher._run()

        assert dispatcher.stats["dispatch_errors"] == 1
        assert "category=RuntimeError" in caplog.text
        assert "sensitive failure" not in caplog.text
        assert MISSION_UUID not in caplog.text
        assert "job-0" not in caplog.text

    @pytest.mark.asyncio
    async def test_health_stats_never_expose_mission_job_token_or_payload(self) -> None:
        """Operational telemetry reports category and cardinality, not identity."""

        claim = _claim(ordinal=7)
        repository = _Repository([claim])
        transport = _Transport(
            [
                _result(
                    success=False,
                    retryable=False,
                    category="authorization",
                    status=403,
                )
            ]
        )
        dispatcher = EvidenceOutboxDispatcher(  # type: ignore[arg-type]
            repository,
            transport,
        )

        await dispatcher.dispatch_once()
        serialized = json.dumps(dispatcher.stats, sort_keys=True)

        assert MISSION_UUID not in serialized
        assert claim.deepsearch_job_id not in serialized
        assert claim.lease_token not in serialized
        assert "authorization" in serialized
        assert '"status_code": 403' in serialized


@pytest.mark.parametrize(
    ("attempt", "expected"),
    ((1, 1.0), (2, 2.0), (4, 8.0), (10, 300.0), (12, 300.0)),
)
def test_backoff_is_exponential_and_bounded(attempt: int, expected: float) -> None:
    assert evidence_outbox_backoff_seconds(attempt) == expected
