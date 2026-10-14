"""Durable, leased delivery of terminal Evidence Ledger triggers.

The outbox stores only the identity of an authoritative persisted result. It
never stores report text, citations, credentials, or a caller-authored evidence
payload. PostgreSQL owns concurrency through bounded ``SKIP LOCKED`` claims;
all outcome transitions are fenced by the exact row identity and lease token.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from deepsearch.tracelab.evidence_ledger import (
    EvidenceLedgerDeliveryResult,
    EvidenceLedgerTrigger,
)

from .database import DatabaseClient

logger = logging.getLogger(__name__)

EVIDENCE_OUTBOX_CLAIM_LIMIT = 25
EVIDENCE_OUTBOX_MAX_ATTEMPTS = 12
DEFAULT_EVIDENCE_OUTBOX_LEASE_SECONDS = 60.0
DEFAULT_EVIDENCE_OUTBOX_POLL_SECONDS = 5.0
MAX_EVIDENCE_OUTBOX_BACKOFF_SECONDS = 300.0


class EvidenceOutboxError(RuntimeError):
    """The durable outbox returned an invalid or unusable state."""


class EvidenceLedgerTransport(Protocol):
    """One-shot transport used by the durable dispatcher."""

    max_attempts: int

    async def deliver(
        self,
        trigger: EvidenceLedgerTrigger,
    ) -> EvidenceLedgerDeliveryResult:
        """Deliver exactly one HTTP attempt for a claimed outbox row."""


@dataclass(frozen=True, slots=True)
class EvidenceOutboxClaim:
    """One token-fenced delivery claim returned by PostgreSQL."""

    mission_uuid: str
    deepsearch_job_id: str
    lease_token: str
    delivery_attempt_count: int


def evidence_outbox_backoff_seconds(delivery_attempt_count: int) -> float:
    """Return deterministic exponential backoff capped at five minutes."""

    if (
        isinstance(delivery_attempt_count, bool)
        or not isinstance(delivery_attempt_count, int)
        or delivery_attempt_count < 1
    ):
        raise ValueError("delivery_attempt_count must be a positive integer")
    return min(
        MAX_EVIDENCE_OUTBOX_BACKOFF_SECONDS,
        float(2 ** min(delivery_attempt_count - 1, 9)),
    )


class EvidenceOutboxRepository:
    """PostgreSQL state transitions for the TraceLab migration-043 outbox."""

    CLAIM_DUE_SQL = """
        WITH candidates AS MATERIALIZED (
            SELECT mission_id, deepsearch_job_id
            FROM deepsearch_evidence_outbox
            WHERE delivery_attempt_count < 12
              AND (
                  (state = 'pending' AND next_attempt_at <= NOW())
                  OR (
                      state = 'leased'
                      AND lease_expires_at IS NOT NULL
                      AND lease_expires_at <= NOW()
                  )
              )
            ORDER BY
                next_attempt_at ASC,
                created_at ASC,
                mission_id ASC,
                deepsearch_job_id ASC
            LIMIT LEAST($1::integer, 25)
            FOR UPDATE SKIP LOCKED
        )
        UPDATE deepsearch_evidence_outbox AS outbox
        SET
            state = 'leased',
            delivery_attempt_count = outbox.delivery_attempt_count + 1,
            lease_token = gen_random_uuid(),
            lease_expires_at = NOW() + ($2 * INTERVAL '1 second'),
            next_attempt_at = NOW() + ($2 * INTERVAL '1 second'),
            acked_at = NULL,
            updated_at = NOW()
        FROM candidates
        WHERE outbox.mission_id = candidates.mission_id
          AND outbox.deepsearch_job_id = candidates.deepsearch_job_id
        RETURNING
            outbox.mission_id,
            outbox.deepsearch_job_id,
            outbox.lease_token,
            outbox.delivery_attempt_count
    """

    ACK_SQL = """
        UPDATE deepsearch_evidence_outbox
        SET
            state = 'acked',
            lease_token = NULL,
            lease_expires_at = NULL,
            acked_at = NOW(),
            last_http_status = $4,
            last_error_code = NULL,
            updated_at = NOW()
        WHERE mission_id = $1
          AND deepsearch_job_id = $2
          AND state = 'leased'
          AND lease_token = $3
        RETURNING state
    """

    RELEASE_RETRY_SQL = """
        UPDATE deepsearch_evidence_outbox
        SET
            state = CASE
                WHEN delivery_attempt_count >= 12 THEN 'dead_letter'
                ELSE 'pending'
            END,
            next_attempt_at = CASE
                WHEN delivery_attempt_count >= 12 THEN NOW()
                ELSE NOW() + ($4 * INTERVAL '1 second')
            END,
            lease_token = NULL,
            lease_expires_at = NULL,
            acked_at = NULL,
            last_http_status = $5,
            last_error_code = CASE
                WHEN delivery_attempt_count >= 12 THEN 'attempts_exhausted'
                ELSE $6
            END,
            updated_at = NOW()
        WHERE mission_id = $1
          AND deepsearch_job_id = $2
          AND state = 'leased'
          AND lease_token = $3
        RETURNING state
    """

    DEAD_LETTER_SQL = """
        UPDATE deepsearch_evidence_outbox
        SET
            state = 'dead_letter',
            next_attempt_at = NOW(),
            lease_token = NULL,
            lease_expires_at = NULL,
            acked_at = NULL,
            last_http_status = $4,
            last_error_code = $5,
            updated_at = NOW()
        WHERE mission_id = $1
          AND deepsearch_job_id = $2
          AND state = 'leased'
          AND lease_token = $3
        RETURNING state
    """

    SWEEP_EXHAUSTED_SQL = """
        WITH candidates AS MATERIALIZED (
            SELECT mission_id, deepsearch_job_id, lease_token
            FROM deepsearch_evidence_outbox
            WHERE state = 'leased'
              AND delivery_attempt_count >= 12
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= NOW()
            ORDER BY
                lease_expires_at ASC,
                created_at ASC,
                mission_id ASC,
                deepsearch_job_id ASC
            LIMIT 25
            FOR UPDATE SKIP LOCKED
        ),
        exhausted AS (
            UPDATE deepsearch_evidence_outbox AS outbox
            SET
                state = 'dead_letter',
                next_attempt_at = NOW(),
                lease_token = NULL,
                lease_expires_at = NULL,
                acked_at = NULL,
                last_error_code = 'attempts_exhausted',
                updated_at = NOW()
            FROM candidates
            WHERE outbox.mission_id = candidates.mission_id
              AND outbox.deepsearch_job_id = candidates.deepsearch_job_id
              AND outbox.state = 'leased'
              AND outbox.lease_token = candidates.lease_token
            RETURNING 1
        )
        SELECT COUNT(*) FROM exhausted
    """

    STATE_COUNTS_SQL = """
        SELECT
            state,
            COUNT(*) AS row_count,
            COUNT(*) FILTER (
                WHERE state = 'pending' AND delivery_attempt_count >= 12
            ) AS invalid_exhausted_pending
        FROM deepsearch_evidence_outbox
        GROUP BY state
    """

    def __init__(self, db_client: DatabaseClient) -> None:
        self.db_client = db_client

    async def claim_due(
        self,
        *,
        limit: int = EVIDENCE_OUTBOX_CLAIM_LIMIT,
        lease_seconds: float = DEFAULT_EVIDENCE_OUTBOX_LEASE_SECONDS,
    ) -> list[EvidenceOutboxClaim]:
        """Claim at most 25 due or expired rows with unique database tokens."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 25:
            raise ValueError("limit must be an integer between 1 and 25")
        _validate_positive_seconds(lease_seconds, name="lease_seconds")
        rows = await self.db_client.fetchall(
            self.CLAIM_DUE_SQL,
            limit,
            float(lease_seconds),
        )
        return [_claim_from_row(row) for row in rows]

    async def acknowledge(
        self,
        claim: EvidenceOutboxClaim,
        *,
        status_code: int,
    ) -> bool:
        """Acknowledge one successful 2xx delivery under its exact token."""

        if (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or not 200 <= status_code < 300
        ):
            raise ValueError("acknowledgement requires a 2xx status code")
        state = await self.db_client.fetchval(
            self.ACK_SQL,
            claim.mission_uuid,
            claim.deepsearch_job_id,
            claim.lease_token,
            status_code,
        )
        return bool(state == "acked")

    async def release_retry(
        self,
        claim: EvidenceOutboxClaim,
        *,
        delay_seconds: float,
        status_code: int | None,
        error_code: str,
    ) -> str | None:
        """Release a retryable failure or dead-letter its twelfth attempt."""

        _validate_nonnegative_seconds(delay_seconds, name="delay_seconds")
        _validate_status_code(status_code)
        _validate_error_code(error_code)
        state = await self.db_client.fetchval(
            self.RELEASE_RETRY_SQL,
            claim.mission_uuid,
            claim.deepsearch_job_id,
            claim.lease_token,
            float(delay_seconds),
            status_code,
            error_code,
        )
        return state if state in {"pending", "dead_letter"} else None

    async def dead_letter(
        self,
        claim: EvidenceOutboxClaim,
        *,
        status_code: int | None,
        error_code: str,
    ) -> bool:
        """Permanently release one deterministic failure under its token."""

        _validate_status_code(status_code)
        _validate_error_code(error_code)
        state = await self.db_client.fetchval(
            self.DEAD_LETTER_SQL,
            claim.mission_uuid,
            claim.deepsearch_job_id,
            claim.lease_token,
            status_code,
            error_code,
        )
        return bool(state == "dead_letter")

    async def sweep_exhausted(self) -> int:
        """Dead-letter a crashed twelfth claim after its lease expires."""

        count = await self.db_client.fetchval(self.SWEEP_EXHAUSTED_SQL)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EvidenceOutboxError("exhausted outbox sweep returned an invalid count")
        return int(count)

    async def state_counts(self) -> dict[str, int]:
        """Return a bounded, identity-free aggregate snapshot for health."""

        counts = {
            "pending": 0,
            "leased": 0,
            "acked": 0,
            "dead_letter": 0,
            "invalid_exhausted_pending": 0,
        }
        rows = await self.db_client.fetchall(self.STATE_COUNTS_SQL)
        for row in rows:
            try:
                state = row["state"]
                count = row["row_count"]
                invalid_exhausted_pending = row["invalid_exhausted_pending"]
            except (KeyError, TypeError) as exc:
                raise EvidenceOutboxError(
                    "outbox state-count row is missing required fields"
                ) from exc
            if state not in counts:
                raise EvidenceOutboxError("outbox state-count row has an unknown state")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise EvidenceOutboxError("outbox state-count row has an invalid count")
            counts[state] = count
            if (
                isinstance(invalid_exhausted_pending, bool)
                or not isinstance(invalid_exhausted_pending, int)
                or invalid_exhausted_pending < 0
            ):
                raise EvidenceOutboxError(
                    "outbox state-count row has an invalid corruption count"
                )
            counts["invalid_exhausted_pending"] += invalid_exhausted_pending
        return counts


class EvidenceOutboxDispatcher:
    """Continuously deliver durable outbox claims without exposing identity."""

    def __init__(
        self,
        repository: EvidenceOutboxRepository,
        transport: EvidenceLedgerTransport,
        *,
        claim_limit: int = EVIDENCE_OUTBOX_CLAIM_LIMIT,
        lease_seconds: float = DEFAULT_EVIDENCE_OUTBOX_LEASE_SECONDS,
        poll_seconds: float = DEFAULT_EVIDENCE_OUTBOX_POLL_SECONDS,
    ) -> None:
        if (
            isinstance(claim_limit, bool)
            or not isinstance(claim_limit, int)
            or not 1 <= claim_limit <= EVIDENCE_OUTBOX_CLAIM_LIMIT
        ):
            raise ValueError("claim_limit must be an integer between 1 and 25")
        _validate_positive_seconds(lease_seconds, name="lease_seconds")
        _validate_positive_seconds(poll_seconds, name="poll_seconds")
        if getattr(transport, "max_attempts", None) != 1:
            raise ValueError("durable Evidence Ledger transport must make one HTTP attempt")
        self.repository = repository
        self.transport = transport
        self.claim_limit = claim_limit
        self.lease_seconds = float(lease_seconds)
        self.poll_seconds = float(poll_seconds)
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stats: dict[str, Any] = {
            "claimed": 0,
            "acknowledged": 0,
            "retry_scheduled": 0,
            "dead_lettered": 0,
            "exhausted_swept": 0,
            "lease_conflicts": 0,
            "dispatch_errors": 0,
            "last_delivery": None,
            "pending": 0,
            "leased": 0,
            "acked": 0,
            "dead_letter": 0,
            "invalid_exhausted_pending": 0,
            # A durable table has no bounded process buffer to overflow or
            # discard. Keep the explicit zeroes so health consumers can prove
            # that no delivery identity was silently evicted.
            "overflowed": 0,
            "discarded": 0,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Return secret- and identity-free operational counters."""

        payload = dict(self._stats)
        last = payload.get("last_delivery")
        if isinstance(last, Mapping):
            payload["last_delivery"] = dict(last)
        payload["running"] = bool(self._task and not self._task.done())
        return payload

    async def start(self) -> None:
        """Start an immediate recovery sweep followed by the idle loop."""

        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(
            self._run(),
            name="deepsearch-evidence-outbox",
        )

    async def stop(self) -> None:
        """Stop after any bounded in-flight delivery batch finishes."""

        self._stop_event.set()
        self._wake_event.set()
        task = self._task
        if task is not None:
            await task
        self._task = None

    def wake(self) -> None:
        """Wake the dispatcher after a terminal transaction commits."""

        self._wake_event.set()

    async def dispatch_once(self) -> int:
        """Sweep exhausted leases, claim one bounded batch, and deliver it."""

        swept = await self.repository.sweep_exhausted()
        self._stats["exhausted_swept"] += swept
        self._stats["dead_lettered"] += swept
        if swept:
            logger.error(
                "Evidence outbox exhausted leases dead-lettered "
                "(count=%d, category=attempts_exhausted)",
                swept,
            )
        claims = await self.repository.claim_due(
            limit=self.claim_limit,
            lease_seconds=self.lease_seconds,
        )
        self._stats["claimed"] += len(claims)
        if claims:
            await asyncio.gather(*(self._deliver_claim(claim) for claim in claims))
        state_counts = await self.repository.state_counts()
        self._stats.update(state_counts)
        invalid_pending = state_counts["invalid_exhausted_pending"]
        if invalid_pending:
            logger.error(
                "Evidence outbox invariant violation "
                "(category=exhausted_pending, count=%d)",
                invalid_pending,
            )
        return len(claims)

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            claimed = 0
            try:
                claimed = await self.dispatch_once()
            except Exception as exc:  # noqa: BLE001 - expired leases recover
                self._stats["dispatch_errors"] += 1
                logger.error(
                    "Evidence outbox dispatch failed (category=%s)",
                    type(exc).__name__,
                )
            if self._stop_event.is_set():
                break
            if claimed == self.claim_limit:
                continue
            await self._wait_until_due_or_woken()

    async def _wait_until_due_or_woken(self) -> None:
        if self._wake_event.is_set():
            self._wake_event.clear()
            return
        try:
            await asyncio.wait_for(
                self._wake_event.wait(),
                timeout=self.poll_seconds,
            )
        except asyncio.TimeoutError:
            pass
        finally:
            self._wake_event.clear()

    async def _deliver_claim(self, claim: EvidenceOutboxClaim) -> None:
        result = await self.transport.deliver(
            EvidenceLedgerTrigger(
                mission_uuid=claim.mission_uuid,
                deepsearch_job_id=claim.deepsearch_job_id,
            )
        )
        if result.success:
            status_code = result.final_status_code
            if status_code is None:
                raise EvidenceOutboxError("successful delivery omitted its HTTP status")
            applied = await self.repository.acknowledge(
                claim,
                status_code=status_code,
            )
            state = "acked" if applied else "lease_conflict"
            if applied:
                self._stats["acknowledged"] += 1
            else:
                self._stats["lease_conflicts"] += 1
            self._record_last_delivery(result, state, claim.delivery_attempt_count)
            return

        if result.retryable:
            retry_state = await self.repository.release_retry(
                claim,
                delay_seconds=evidence_outbox_backoff_seconds(
                    claim.delivery_attempt_count
                ),
                status_code=result.final_status_code,
                error_code=result.category,
            )
            if retry_state == "pending":
                self._stats["retry_scheduled"] += 1
            elif retry_state == "dead_letter":
                self._stats["dead_lettered"] += 1
                logger.error(
                    "Evidence outbox delivery dead-lettered "
                    "(category=attempts_exhausted, status=%s, attempt=%d)",
                    result.final_status_code,
                    claim.delivery_attempt_count,
                )
            else:
                self._stats["lease_conflicts"] += 1
                retry_state = "lease_conflict"
            self._record_last_delivery(
                result,
                retry_state,
                claim.delivery_attempt_count,
            )
            return

        applied = await self.repository.dead_letter(
            claim,
            status_code=result.final_status_code,
            error_code=result.category,
        )
        state = "dead_letter" if applied else "lease_conflict"
        if applied:
            self._stats["dead_lettered"] += 1
            logger.error(
                "Evidence outbox delivery dead-lettered "
                "(category=%s, status=%s, attempt=%d)",
                result.category,
                result.final_status_code,
                claim.delivery_attempt_count,
            )
        else:
            self._stats["lease_conflicts"] += 1
        self._record_last_delivery(result, state, claim.delivery_attempt_count)

    def _record_last_delivery(
        self,
        result: EvidenceLedgerDeliveryResult,
        state: str,
        attempt_number: int,
    ) -> None:
        self._stats["last_delivery"] = {
            "category": result.category,
            "status_code": result.final_status_code,
            "retryable": result.retryable,
            "state": state,
            "attempt_number": attempt_number,
        }


def _claim_from_row(row: Mapping[str, Any]) -> EvidenceOutboxClaim:
    try:
        mission_uuid = str(row["mission_id"])
        deepsearch_job_id = str(row["deepsearch_job_id"])
        lease_token = str(row["lease_token"])
        attempt_count = row["delivery_attempt_count"]
    except (KeyError, TypeError) as exc:
        raise EvidenceOutboxError("outbox claim row is missing required fields") from exc
    if not mission_uuid or not deepsearch_job_id or not lease_token:
        raise EvidenceOutboxError("outbox claim identity fields must be nonempty")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or not 1 <= attempt_count <= EVIDENCE_OUTBOX_MAX_ATTEMPTS
    ):
        raise EvidenceOutboxError("outbox claim attempt is outside the 1..12 contract")
    return EvidenceOutboxClaim(
        mission_uuid=mission_uuid,
        deepsearch_job_id=deepsearch_job_id,
        lease_token=lease_token,
        delivery_attempt_count=attempt_count,
    )


def _validate_positive_seconds(value: float, *, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")


def _validate_nonnegative_seconds(value: float, *, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")


def _validate_status_code(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 599:
        raise ValueError("status_code must be null or an integer from 100 through 599")


def _validate_error_code(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 100
    ):
        raise ValueError("error_code must be a nonempty string of at most 100 characters")


__all__ = [
    "DEFAULT_EVIDENCE_OUTBOX_LEASE_SECONDS",
    "DEFAULT_EVIDENCE_OUTBOX_POLL_SECONDS",
    "EVIDENCE_OUTBOX_CLAIM_LIMIT",
    "EVIDENCE_OUTBOX_MAX_ATTEMPTS",
    "EvidenceOutboxClaim",
    "EvidenceOutboxDispatcher",
    "EvidenceOutboxError",
    "EvidenceOutboxRepository",
    "evidence_outbox_backoff_seconds",
]
