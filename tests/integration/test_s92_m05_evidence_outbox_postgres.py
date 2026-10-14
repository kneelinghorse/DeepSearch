"""Real-PostgreSQL durability gate for the S92 Evidence Ledger outbox.

These tests deliberately use PostgreSQL 15 rather than SQL mocks.  The gate
proves the transaction, locking, and lease-fencing properties that protect a
persisted mission result from being lost or delivered under the wrong identity.
It never contacts TraceLab or a model provider.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from uuid import UUID, uuid4

import pytest

from deepsearch.agent.outcome import MissionOutcome
from deepsearch.tracelab.evidence_ledger import (
    EvidenceLedgerDeliveryResult,
    EvidenceLedgerTrigger,
    _classify_status,
)
from deepsearch.worker.database import DatabaseClient, DatabaseConfig
from deepsearch.worker.evidence_outbox import (
    EvidenceOutboxClaim,
    EvidenceOutboxDispatcher,
    EvidenceOutboxRepository,
)
from deepsearch.worker.lease import LeaseAttempt
from deepsearch.worker.writer import (
    ExecutionMetadata,
    ResultWriter,
    StaleLeaseError,
    WriterError,
)

_POSTGRES_IMAGE = "postgres:15"
_POSTGRES_DATABASE = "deepsearch_s92"

_SCHEMA_SQL = """
    DROP TABLE IF EXISTS deepsearch_evidence_outbox;
    DROP TABLE IF EXISTS missions;
    DROP FUNCTION IF EXISTS reject_completed_terminal_update();
    DROP FUNCTION IF EXISTS reject_validation_failed_terminal_update();
    DROP FUNCTION IF EXISTS slow_pending_outbox_claim();

    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    CREATE TABLE missions (
        id UUID PRIMARY KEY,
        status VARCHAR(32) NOT NULL,
        completed_at TIMESTAMP WITHOUT TIME ZONE,
        result_markdown TEXT,
        result_protocol JSONB,
        execution_metadata JSONB,
        error_message TEXT,
        updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        deepsearch_job_id VARCHAR(100),
        deepsearch_lease_owner TEXT,
        deepsearch_lease_token TEXT,
        deepsearch_leased_at TIMESTAMP WITH TIME ZONE,
        deepsearch_heartbeat_at TIMESTAMP WITH TIME ZONE,
        deepsearch_lease_expires_at TIMESTAMP WITH TIME ZONE,
        deepsearch_attempt_count INTEGER NOT NULL DEFAULT 0,
        deepsearch_result_key TEXT
    );

    CREATE TABLE deepsearch_evidence_outbox (
        mission_id UUID NOT NULL,
        deepsearch_job_id VARCHAR(100) NOT NULL,
        deepsearch_result_key TEXT NOT NULL,
        mission_attempt_count INTEGER NOT NULL,
        terminal_status VARCHAR(32) NOT NULL,
        schema_version SMALLINT NOT NULL DEFAULT 1,
        state VARCHAR(16) NOT NULL DEFAULT 'pending',
        delivery_attempt_count INTEGER NOT NULL DEFAULT 0,
        next_attempt_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        lease_token UUID,
        lease_expires_at TIMESTAMP WITH TIME ZONE,
        acked_at TIMESTAMP WITH TIME ZONE,
        last_http_status SMALLINT,
        last_error_code VARCHAR(100),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        CONSTRAINT pk_deepsearch_evidence_outbox
            PRIMARY KEY (mission_id, deepsearch_job_id),
        CONSTRAINT fk_deepsearch_evidence_outbox_mission
            FOREIGN KEY (mission_id) REFERENCES missions (id) ON DELETE CASCADE,
        CONSTRAINT ck_deepsearch_evidence_outbox_nonempty_job
            CHECK (LENGTH(TRIM(deepsearch_job_id)) > 0),
        CONSTRAINT ck_deepsearch_evidence_outbox_nonempty_result_key
            CHECK (LENGTH(TRIM(deepsearch_result_key)) > 0),
        CONSTRAINT ck_deepsearch_evidence_outbox_positive_attempt
            CHECK (mission_attempt_count > 0),
        CONSTRAINT ck_deepsearch_evidence_outbox_terminal_status
            CHECK (terminal_status IN ('completed', 'validation_failed')),
        CONSTRAINT ck_deepsearch_evidence_outbox_schema_version
            CHECK (schema_version = 1),
        CONSTRAINT ck_deepsearch_evidence_outbox_state
            CHECK (state IN ('pending', 'leased', 'acked', 'dead_letter')),
        CONSTRAINT ck_deepsearch_evidence_outbox_delivery_attempts
            CHECK (delivery_attempt_count >= 0),
        CONSTRAINT ck_deepsearch_evidence_outbox_http_status
            CHECK (
                last_http_status IS NULL
                OR last_http_status BETWEEN 100 AND 599
            ),
        CONSTRAINT ck_deepsearch_evidence_outbox_state_coherence
            CHECK (
                (
                    state = 'leased'
                    AND lease_token IS NOT NULL
                    AND lease_expires_at IS NOT NULL
                    AND next_attempt_at = lease_expires_at
                    AND acked_at IS NULL
                )
                OR (
                    state = 'acked'
                    AND lease_token IS NULL
                    AND lease_expires_at IS NULL
                    AND acked_at IS NOT NULL
                )
                OR (
                    state IN ('pending', 'dead_letter')
                    AND lease_token IS NULL
                    AND lease_expires_at IS NULL
                    AND acked_at IS NULL
                )
            )
    );

    CREATE INDEX ix_deepsearch_evidence_outbox_delivery
        ON deepsearch_evidence_outbox (state, next_attempt_at, created_at);
"""

_OUTBOX_COLUMNS = (
    "mission_id",
    "deepsearch_job_id",
    "deepsearch_result_key",
    "mission_attempt_count",
    "terminal_status",
    "schema_version",
    "state",
    "delivery_attempt_count",
    "next_attempt_at",
    "lease_token",
    "lease_expires_at",
    "acked_at",
    "last_http_status",
    "last_error_code",
    "created_at",
    "updated_at",
)


@pytest.fixture(scope="module")
def postgres_15_url() -> Iterator[str]:
    """Start a private PostgreSQL 15 container and fail loudly if unavailable."""

    docker = shutil.which("docker")
    assert docker is not None, "S92 PostgreSQL gate requires the Docker CLI"

    container_name = f"deepsearch-s92-pg-{uuid4().hex[:12]}"
    password = f"fixture-{uuid4().hex}"
    start = subprocess.run(
        [
            docker,
            "run",
            "--detach",
            "--rm",
            "--pull=never",
            "--name",
            container_name,
            "--env",
            "POSTGRES_USER=postgres",
            "--env",
            f"POSTGRES_PASSWORD={password}",
            "--env",
            f"POSTGRES_DB={_POSTGRES_DATABASE}",
            "--publish",
            "127.0.0.1::5432",
            _POSTGRES_IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert start.returncode == 0, (
        "S92 PostgreSQL 15 container failed to start "
        f"(exit={start.returncode}, stderr={start.stderr.strip()!r})"
    )

    try:
        published_port = subprocess.run(
            [docker, "port", container_name, "5432/tcp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert published_port.returncode == 0, (
            "S92 PostgreSQL gate could not resolve its private published port"
        )
        first_binding = published_port.stdout.strip().splitlines()[0]
        host_port = int(first_binding.rsplit(":", maxsplit=1)[1])
        database_url = (
            "postgresql://postgres:"
            f"{quote(password, safe='')}@127.0.0.1:{host_port}/{_POSTGRES_DATABASE}"
        )

        # ``pg_isready`` inside the official image can observe the temporary
        # Unix-socket server used by docker-entrypoint.sh during first-time
        # database initialization.  That server is intentionally restarted
        # before the final TCP postmaster comes online, so yielding on the
        # socket probe creates a false-green race.  Probe the exact published
        # TCP endpoint with asyncpg instead and require one stable postmaster
        # for a continuous window before any test pool is allowed to connect.
        import asyncpg

        async def probe_final_server() -> tuple[str, str, bool, bool]:
            connection = await asyncpg.connect(database_url, timeout=2)
            try:
                row = await connection.fetchrow(
                    """
                    SELECT
                        current_setting('server_version_num') AS version_num,
                        pg_postmaster_start_time()::text AS postmaster_started_at,
                        inet_server_addr() IS NOT NULL AS reached_over_tcp,
                        NOT pg_is_in_recovery() AS primary_ready
                    """
                )
                assert row is not None
                return (
                    str(row["version_num"]),
                    str(row["postmaster_started_at"]),
                    bool(row["reached_over_tcp"]),
                    bool(row["primary_ready"]),
                )
            finally:
                await connection.close(timeout=2)

        deadline = time.monotonic() + 45.0
        stable_since: float | None = None
        stable_postmaster: str | None = None
        last_failure = "not_ready"
        while time.monotonic() < deadline:
            try:
                version, postmaster_started_at, reached_over_tcp, primary_ready = (
                    asyncio.run(probe_final_server())
                )
            except Exception as exc:  # noqa: BLE001 - readiness retries are bounded
                stable_since = None
                stable_postmaster = None
                last_failure = type(exc).__name__
            else:
                ready = (
                    version.startswith("15")
                    and reached_over_tcp
                    and primary_ready
                )
                if not ready:
                    stable_since = None
                    stable_postmaster = None
                    last_failure = "capability_mismatch"
                elif stable_postmaster != postmaster_started_at:
                    stable_postmaster = postmaster_started_at
                    stable_since = time.monotonic()
                elif (
                    stable_since is not None
                    and time.monotonic() - stable_since >= 2.0
                ):
                    break
            time.sleep(0.2)
        else:
            raise AssertionError(
                "S92 PostgreSQL 15 final TCP server did not remain ready "
                f"within 45 seconds (last_failure={last_failure})"
            )

        yield database_url
    finally:
        subprocess.run(
            [docker, "rm", "--force", container_name],
            capture_output=True,
            text=True,
            timeout=15,
        )


async def _connect(url: str) -> DatabaseClient:
    client = DatabaseClient(
        DatabaseConfig(
            url=url,
            min_pool_size=1,
            max_pool_size=6,
            retry_attempts=1,
        )
    )
    await client.connect()
    version = await client.fetchval("SHOW server_version_num")
    assert isinstance(version, str) and version.startswith("15"), version
    return client


async def _reset_schema(client: DatabaseClient) -> None:
    await client.execute(_SCHEMA_SQL)


def _lease(*, attempt_count: int = 1) -> LeaseAttempt:
    now = datetime.now(timezone.utc)
    return LeaseAttempt(
        owner="s92-postgres-gate",
        token=str(uuid4()),
        leased_at=now,
        heartbeat_at=now,
        expires_at=now + timedelta(minutes=5),
        attempt_count=attempt_count,
    )


async def _insert_in_progress_mission(
    client: DatabaseClient,
    mission_id: UUID,
    lease: LeaseAttempt,
) -> None:
    await client.execute(
        """
        INSERT INTO missions (
            id,
            status,
            updated_at,
            deepsearch_lease_owner,
            deepsearch_lease_token,
            deepsearch_leased_at,
            deepsearch_heartbeat_at,
            deepsearch_lease_expires_at,
            deepsearch_attempt_count
        )
        VALUES (
            $1,
            'in_progress',
            NOW() AT TIME ZONE 'UTC',
            $2,
            $3,
            $4,
            $5,
            $6,
            $7
        )
        """,
        mission_id,
        lease.owner,
        lease.token,
        lease.leased_at,
        lease.heartbeat_at,
        lease.expires_at,
        lease.attempt_count,
    )


async def _seed_pending_outbox(client: DatabaseClient, count: int) -> None:
    await client.execute(
        """
        WITH inserted_missions AS (
            INSERT INTO missions (id, status, completed_at, updated_at)
            SELECT
                gen_random_uuid(),
                'completed',
                NOW() AT TIME ZONE 'UTC',
                NOW() AT TIME ZONE 'UTC'
            FROM generate_series(1, $1::integer)
            RETURNING id
        )
        INSERT INTO deepsearch_evidence_outbox (
            mission_id,
            deepsearch_job_id,
            deepsearch_result_key,
            mission_attempt_count,
            terminal_status,
            schema_version
        )
        SELECT
            id,
            'job-pending',
            'result-' || id::text,
            1,
            'completed',
            1
        FROM inserted_missions
        """,
        count,
    )


def _metadata() -> ExecutionMetadata:
    return ExecutionMetadata(
        total_tokens=17,
        duration_seconds=1.25,
        runtime_identity={"build_hash": "s92-postgres-gate"},
    )


class _OneShotTransport:
    max_attempts = 1

    def __init__(self, result: EvidenceLedgerDeliveryResult) -> None:
        self.result = result
        self.triggers: list[EvidenceLedgerTrigger] = []

    async def deliver(
        self,
        trigger: EvidenceLedgerTrigger,
    ) -> EvidenceLedgerDeliveryResult:
        self.triggers.append(trigger)
        return self.result


def _status_result(status_code: int) -> EvidenceLedgerDeliveryResult:
    category, retryable = _classify_status(status_code)
    return EvidenceLedgerDeliveryResult(
        success=category == "accepted",
        attempts=1,
        retryable=retryable,
        category=category,
        final_status_code=status_code,
    )


@pytest.mark.asyncio
async def test_terminal_update_and_metadata_enrollment_are_one_atomic_statement(
    postgres_15_url: str,
) -> None:
    """A failure after enrollment rolls both writes back; exact replay is inert."""

    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        writer = ResultWriter(client)
        rollback_mission = uuid4()
        rollback_lease = _lease()
        await _insert_in_progress_mission(client, rollback_mission, rollback_lease)

        await client.execute(
            """
            CREATE FUNCTION reject_completed_terminal_update()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.status = 'completed' THEN
                    RAISE EXCEPTION 'forced terminal rollback';
                END IF;
                RETURN NEW;
            END;
            $$;

            CREATE TRIGGER reject_completed_terminal_update
            BEFORE UPDATE ON missions
            FOR EACH ROW
            EXECUTE FUNCTION reject_completed_terminal_update();
            """
        )

        with pytest.raises(WriterError, match="forced terminal rollback"):
            await writer.mark_completed(
                str(rollback_mission),
                "# atomic result",
                {"schema_version": 1},
                _metadata(),
                rollback_lease,
                "job-atomic",
            )

        rolled_back = await client.fetchone(
            """
            SELECT
                status,
                result_markdown,
                deepsearch_result_key,
                deepsearch_job_id,
                deepsearch_lease_token
            FROM missions
            WHERE id = $1
            """,
            rollback_mission,
        )
        assert rolled_back is not None
        assert dict(rolled_back) == {
            "status": "in_progress",
            "result_markdown": None,
            "deepsearch_result_key": None,
            "deepsearch_job_id": None,
            "deepsearch_lease_token": rollback_lease.token,
        }
        assert (
            await client.fetchval(
                "SELECT COUNT(*) FROM deepsearch_evidence_outbox WHERE mission_id = $1",
                rollback_mission,
            )
            == 0
        )

        await client.execute(
            """
            DROP TRIGGER reject_completed_terminal_update ON missions;
            DROP FUNCTION reject_completed_terminal_update();
            """
        )
        assert await writer.mark_completed(
            str(rollback_mission),
            "# atomic result",
            {"schema_version": 1},
            _metadata(),
            rollback_lease,
            "job-atomic",
        )

        enrolled = await client.fetchone(
            """
            SELECT
                mission_id::text,
                deepsearch_job_id,
                deepsearch_result_key,
                mission_attempt_count,
                terminal_status,
                schema_version,
                state,
                delivery_attempt_count,
                lease_token,
                lease_expires_at,
                acked_at,
                last_http_status,
                last_error_code
            FROM deepsearch_evidence_outbox
            WHERE mission_id = $1
            """,
            rollback_mission,
        )
        assert enrolled is not None
        assert dict(enrolled) == {
            "mission_id": str(rollback_mission),
            "deepsearch_job_id": "job-atomic",
            "deepsearch_result_key": rollback_lease.persistence_key(
                str(rollback_mission)
            ),
            "mission_attempt_count": 1,
            "terminal_status": "completed",
            "schema_version": 1,
            "state": "pending",
            "delivery_attempt_count": 0,
            "lease_token": None,
            "lease_expires_at": None,
            "acked_at": None,
            "last_http_status": None,
            "last_error_code": None,
        }

        columns = await client.fetchall(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'deepsearch_evidence_outbox'
            ORDER BY ordinal_position
            """
        )
        assert tuple(row["column_name"] for row in columns) == _OUTBOX_COLUMNS
        assert not {
            "result_markdown",
            "result_protocol",
            "execution_metadata",
            "error_message",
        }.intersection(_OUTBOX_COLUMNS)

        # The exact tuple is idempotent after the lease fields are cleared.
        assert await writer.mark_completed(
            str(rollback_mission),
            "# atomic result",
            {"schema_version": 1},
            _metadata(),
            rollback_lease,
            "job-atomic",
        )
        assert (
            await client.fetchval(
                "SELECT COUNT(*) FROM deepsearch_evidence_outbox WHERE mission_id = $1",
                rollback_mission,
            )
            == 1
        )

        # A reused mission/job identity with a different immutable snapshot
        # cannot transition the active mission or overwrite the enrolled row.
        drift_mission = uuid4()
        drift_lease = _lease()
        await _insert_in_progress_mission(client, drift_mission, drift_lease)
        await client.execute(
            """
            INSERT INTO deepsearch_evidence_outbox (
                mission_id,
                deepsearch_job_id,
                deepsearch_result_key,
                mission_attempt_count,
                terminal_status,
                schema_version
            )
            VALUES ($1, 'job-drift', 'different-result-key', 1, 'completed', 1)
            """,
            drift_mission,
        )

        with pytest.raises(StaleLeaseError, match="conflicting result"):
            await writer.mark_completed(
                str(drift_mission),
                "# must not persist",
                {"schema_version": 1},
                _metadata(),
                drift_lease,
                "job-drift",
            )

        drift_state = await client.fetchone(
            """
            SELECT
                mission.status,
                mission.result_markdown,
                mission.deepsearch_result_key,
                mission.deepsearch_lease_token,
                outbox.deepsearch_result_key AS enrolled_result_key
            FROM missions AS mission
            JOIN deepsearch_evidence_outbox AS outbox
              ON outbox.mission_id = mission.id
             AND outbox.deepsearch_job_id = 'job-drift'
            WHERE mission.id = $1
            """,
            drift_mission,
        )
        assert drift_state is not None
        assert dict(drift_state) == {
            "status": "in_progress",
            "result_markdown": None,
            "deepsearch_result_key": None,
            "deepsearch_lease_token": drift_lease.token,
            "enrolled_result_key": "different-result-key",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_only_evidence_bearing_terminal_verdicts_enroll(
    postgres_15_url: str,
) -> None:
    """Validation failures enroll atomically; blocked verdicts never do."""

    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        writer = ResultWriter(client)

        reviewable_mission = uuid4()
        reviewable_lease = _lease()
        await _insert_in_progress_mission(
            client,
            reviewable_mission,
            reviewable_lease,
        )
        assert await writer.mark_failed(
            str(reviewable_mission),
            "reviewable validation verdict",
            outcome=MissionOutcome.VALIDATION_FAILED,
            lease=reviewable_lease,
            job_id="job-validation-failed",
            result_markdown="# reviewable artifact",
            result_protocol={"schema_version": 1},
            execution_metadata=_metadata(),
        )

        reviewable = await client.fetchone(
            """
            SELECT
                mission.status,
                mission.deepsearch_lease_token,
                mission.deepsearch_result_key = outbox.deepsearch_result_key
                    AS result_identity_matches,
                mission.deepsearch_job_id = outbox.deepsearch_job_id
                    AS job_identity_matches,
                outbox.terminal_status,
                outbox.state,
                outbox.mission_attempt_count,
                outbox.schema_version
            FROM missions AS mission
            JOIN deepsearch_evidence_outbox AS outbox
              ON outbox.mission_id = mission.id
            WHERE mission.id = $1
            """,
            reviewable_mission,
        )
        assert reviewable is not None
        assert dict(reviewable) == {
            "status": "validation_failed",
            "deepsearch_lease_token": None,
            "result_identity_matches": True,
            "job_identity_matches": True,
            "terminal_status": "validation_failed",
            "state": "pending",
            "mission_attempt_count": 1,
            "schema_version": 1,
        }

        blocked_mission = uuid4()
        blocked_lease = _lease()
        await _insert_in_progress_mission(client, blocked_mission, blocked_lease)
        assert await writer.mark_failed(
            str(blocked_mission),
            "non-evidence-bearing blocked verdict",
            outcome=MissionOutcome.BLOCKED,
            lease=blocked_lease,
            job_id="job-blocked",
        )

        blocked = await client.fetchone(
            """
            SELECT
                status,
                deepsearch_lease_token,
                deepsearch_result_key IS NOT NULL AS result_key_present
            FROM missions
            WHERE id = $1
            """,
            blocked_mission,
        )
        assert blocked is not None
        assert dict(blocked) == {
            "status": "blocked",
            "deepsearch_lease_token": None,
            "result_key_present": True,
        }
        assert await client.fetchval(
            "SELECT COUNT(*) FROM deepsearch_evidence_outbox WHERE mission_id = $1",
            blocked_mission,
        ) == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_validation_failed_terminal_and_outbox_roll_back_together(
    postgres_15_url: str,
) -> None:
    """A validation_failed trigger error cannot leave orphan enrollment."""

    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        writer = ResultWriter(client)
        mission_id = uuid4()
        lease = _lease()
        await _insert_in_progress_mission(client, mission_id, lease)
        await client.execute(
            """
            CREATE FUNCTION reject_validation_failed_terminal_update()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.status = 'validation_failed' THEN
                    RAISE EXCEPTION 'forced validation terminal rollback';
                END IF;
                RETURN NEW;
            END;
            $$;

            CREATE TRIGGER reject_validation_failed_terminal_update
            BEFORE UPDATE ON missions
            FOR EACH ROW
            EXECUTE FUNCTION reject_validation_failed_terminal_update();
            """
        )

        with pytest.raises(WriterError, match="forced validation terminal rollback"):
            await writer.mark_failed(
                str(mission_id),
                "reviewable validation verdict",
                outcome=MissionOutcome.VALIDATION_FAILED,
                lease=lease,
                job_id="job-validation-rollback",
                result_markdown="# reviewable artifact",
                result_protocol={"schema_version": 1},
                execution_metadata=_metadata(),
            )

        mission = await client.fetchone(
            """
            SELECT
                status,
                result_markdown,
                deepsearch_result_key,
                deepsearch_job_id,
                deepsearch_lease_token
            FROM missions
            WHERE id = $1
            """,
            mission_id,
        )
        assert mission is not None
        assert dict(mission) == {
            "status": "in_progress",
            "result_markdown": None,
            "deepsearch_result_key": None,
            "deepsearch_job_id": None,
            "deepsearch_lease_token": lease.token,
        }
        assert await client.fetchval(
            "SELECT COUNT(*) FROM deepsearch_evidence_outbox WHERE mission_id = $1",
            mission_id,
        ) == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_two_concurrent_claimers_take_disjoint_bounded_batches(
    postgres_15_url: str,
) -> None:
    """PostgreSQL SKIP LOCKED gives two workers disjoint batches of at most 25."""

    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        await _seed_pending_outbox(client, 60)
        await client.execute(
            """
            CREATE FUNCTION slow_pending_outbox_claim()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF OLD.state = 'pending' AND NEW.state = 'leased' THEN
                    PERFORM pg_sleep(0.01);
                END IF;
                RETURN NEW;
            END;
            $$;

            CREATE TRIGGER slow_pending_outbox_claim
            BEFORE UPDATE ON deepsearch_evidence_outbox
            FOR EACH ROW
            EXECUTE FUNCTION slow_pending_outbox_claim();
            """
        )
        repository_a = EvidenceOutboxRepository(client)
        repository_b = EvidenceOutboxRepository(client)

        claims_a, claims_b = await asyncio.gather(
            repository_a.claim_due(limit=25, lease_seconds=60),
            repository_b.claim_due(limit=25, lease_seconds=60),
        )

        assert len(claims_a) == 25
        assert len(claims_b) == 25
        identities_a = {
            (claim.mission_uuid, claim.deepsearch_job_id) for claim in claims_a
        }
        identities_b = {
            (claim.mission_uuid, claim.deepsearch_job_id) for claim in claims_b
        }
        assert identities_a.isdisjoint(identities_b)
        assert len(identities_a | identities_b) == 50
        assert len(
            {claim.lease_token for claim in [*claims_a, *claims_b]}
        ) == 50

        counts = await client.fetchone(
            """
            SELECT
                COUNT(*) FILTER (WHERE state = 'leased') AS leased,
                COUNT(*) FILTER (WHERE state = 'pending') AS pending,
                MIN(delivery_attempt_count) FILTER (WHERE state = 'leased')
                    AS min_attempt,
                MAX(delivery_attempt_count) FILTER (WHERE state = 'leased')
                    AS max_attempt
            FROM deepsearch_evidence_outbox
            """
        )
        assert counts is not None
        assert dict(counts) == {
            "leased": 50,
            "pending": 10,
            "min_attempt": 1,
            "max_attempt": 1,
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_lost_initial_ack_restarts_as_exact_replay_then_acks_once(
    postgres_15_url: str,
) -> None:
    """A lost 201 ack replays the same tuple as 200 and then becomes inert."""

    first_client = await _connect(postgres_15_url)
    await _reset_schema(first_client)
    await _seed_pending_outbox(first_client, 1)
    first_repository = EvidenceOutboxRepository(first_client)
    first_claims = await first_repository.claim_due(limit=1, lease_seconds=60)
    assert len(first_claims) == 1
    first_claim = first_claims[0]

    # TraceLab commits the first projection and returns 201/captured.  Model a
    # process crash (or lost response) before the local token-fenced ack: the
    # outbox row must remain leased and recoverable, not be deleted or cloned.
    initial_transport = _OneShotTransport(_status_result(201))
    initial_trigger = EvidenceLedgerTrigger(
        mission_uuid=first_claim.mission_uuid,
        deepsearch_job_id=first_claim.deepsearch_job_id,
    )
    initial_result = await initial_transport.deliver(initial_trigger)
    assert initial_result.success
    assert initial_result.final_status_code == 201
    assert len(initial_transport.triggers) == 1
    assert await first_client.fetchval(
        "SELECT COUNT(*) FROM deepsearch_evidence_outbox"
    ) == 1

    wrong_claim = EvidenceOutboxClaim(
        mission_uuid=first_claim.mission_uuid,
        deepsearch_job_id=first_claim.deepsearch_job_id,
        lease_token=str(uuid4()),
        delivery_attempt_count=first_claim.delivery_attempt_count,
    )

    assert not await first_repository.acknowledge(wrong_claim, status_code=200)
    assert (
        await first_repository.release_retry(
            wrong_claim,
            delay_seconds=0,
            status_code=503,
            error_code="server_error",
        )
        is None
    )
    fenced_row = await first_client.fetchone(
        """
        SELECT state, lease_token::text, delivery_attempt_count
        FROM deepsearch_evidence_outbox
        """
    )
    assert fenced_row is not None
    assert dict(fenced_row) == {
        "state": "leased",
        "lease_token": first_claim.lease_token,
        "delivery_attempt_count": 1,
    }

    await first_client.execute(
        """
        UPDATE deepsearch_evidence_outbox
        SET
            lease_expires_at = NOW() - INTERVAL '1 second',
            next_attempt_at = NOW() - INTERVAL '1 second'
        """
    )
    await first_client.close()

    # A fresh pool/repository models a restarted worker with no process memory.
    restarted_client = await _connect(postgres_15_url)
    try:
        restarted_repository = EvidenceOutboxRepository(restarted_client)
        reclaimed = await restarted_repository.claim_due(limit=1, lease_seconds=60)
        assert len(reclaimed) == 1
        current_claim = reclaimed[0]
        assert current_claim.delivery_attempt_count == 2
        assert current_claim.lease_token != first_claim.lease_token

        assert not await restarted_repository.acknowledge(
            first_claim,
            status_code=200,
        )
        assert (
            await restarted_repository.release_retry(
                first_claim,
                delay_seconds=0,
                status_code=503,
                error_code="server_error",
            )
            is None
        )

        replay_transport = _OneShotTransport(_status_result(200))
        dispatcher = EvidenceOutboxDispatcher(
            restarted_repository,
            replay_transport,
        )
        await dispatcher._deliver_claim(current_claim)

        acked = await restarted_client.fetchone(
            """
            SELECT
                state,
                lease_token,
                lease_expires_at,
                acked_at IS NOT NULL AS acked_at_present,
                last_http_status,
                delivery_attempt_count
            FROM deepsearch_evidence_outbox
            """
        )
        assert acked is not None
        assert dict(acked) == {
            "state": "acked",
            "lease_token": None,
            "lease_expires_at": None,
            "acked_at_present": True,
            "last_http_status": 200,
            "delivery_attempt_count": 2,
        }
        assert replay_transport.triggers == initial_transport.triggers
        assert len(replay_transport.triggers) == 1
        assert await restarted_client.fetchval(
            "SELECT COUNT(*) FROM deepsearch_evidence_outbox"
        ) == 1
        assert await dispatcher.dispatch_once() == 0
        assert len(replay_transport.triggers) == 1
        assert await restarted_client.fetchval(
            "SELECT COUNT(*) FROM deepsearch_evidence_outbox"
        ) == 1
    finally:
        await restarted_client.close()


@pytest.mark.asyncio
async def test_dispatcher_crash_after_201_reclaims_exact_tuple_and_acks_replay(
    postgres_15_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process loss between HTTP 201 and ACK preserves one replayable row."""

    first_client = await _connect(postgres_15_url)
    initial_transport = _OneShotTransport(_status_result(201))
    first_lease_token = ""
    try:
        await _reset_schema(first_client)
        await _seed_pending_outbox(first_client, 1)
        first_repository = EvidenceOutboxRepository(first_client)

        async def crash_before_ack(
            _claim: EvidenceOutboxClaim,
            *,
            status_code: int,
        ) -> bool:
            assert status_code == 201
            raise RuntimeError("fault injection after transport before ACK")

        monkeypatch.setattr(first_repository, "acknowledge", crash_before_ack)
        first_dispatcher = EvidenceOutboxDispatcher(
            first_repository,
            initial_transport,
            claim_limit=1,
        )

        with pytest.raises(RuntimeError, match="before ACK"):
            await first_dispatcher.dispatch_once()

        assert len(initial_transport.triggers) == 1
        stranded = await first_client.fetchone(
            """
            SELECT
                state,
                lease_token::text AS lease_token,
                delivery_attempt_count,
                last_http_status,
                (SELECT COUNT(*) FROM deepsearch_evidence_outbox) AS cardinality
            FROM deepsearch_evidence_outbox
            """
        )
        assert stranded is not None
        first_lease_token = str(stranded["lease_token"])
        assert dict(stranded) == {
            "state": "leased",
            "lease_token": first_lease_token,
            "delivery_attempt_count": 1,
            "last_http_status": None,
            "cardinality": 1,
        }
        await first_client.execute(
            """
            UPDATE deepsearch_evidence_outbox
            SET
                lease_expires_at = NOW() - INTERVAL '1 second',
                next_attempt_at = NOW() - INTERVAL '1 second'
            """
        )
    finally:
        await first_client.close()

    restarted_client = await _connect(postgres_15_url)
    try:
        replay_transport = _OneShotTransport(_status_result(200))
        restarted_repository = EvidenceOutboxRepository(restarted_client)
        restarted_dispatcher = EvidenceOutboxDispatcher(
            restarted_repository,
            replay_transport,
            claim_limit=1,
        )

        assert await restarted_dispatcher.dispatch_once() == 1
        assert replay_transport.triggers == initial_transport.triggers
        assert len(replay_transport.triggers) == 1

        acked = await restarted_client.fetchone(
            """
            SELECT
                state,
                lease_token,
                lease_expires_at,
                acked_at IS NOT NULL AS acked_at_present,
                last_http_status,
                delivery_attempt_count,
                (SELECT COUNT(*) FROM deepsearch_evidence_outbox) AS cardinality
            FROM deepsearch_evidence_outbox
            """
        )
        assert acked is not None
        assert dict(acked) == {
            "state": "acked",
            "lease_token": None,
            "lease_expires_at": None,
            "acked_at_present": True,
            "last_http_status": 200,
            "delivery_attempt_count": 2,
            "cardinality": 1,
        }
        assert restarted_dispatcher.stats["acknowledged"] == 1
        assert restarted_dispatcher.stats["lease_conflicts"] == 0
        assert await restarted_dispatcher.dispatch_once() == 0
        assert len(replay_transport.triggers) == 1
    finally:
        await restarted_client.close()


@pytest.mark.asyncio
async def test_stale_token_permanent_dead_letter_is_a_noop_in_postgres(
    postgres_15_url: str,
) -> None:
    """A stale worker cannot permanently discard another worker's lease."""

    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        await _seed_pending_outbox(client, 1)
        repository = EvidenceOutboxRepository(client)
        claims = await repository.claim_due(limit=1, lease_seconds=60)
        assert len(claims) == 1
        active_claim = claims[0]
        stale_claim = EvidenceOutboxClaim(
            mission_uuid=active_claim.mission_uuid,
            deepsearch_job_id=active_claim.deepsearch_job_id,
            lease_token=str(uuid4()),
            delivery_attempt_count=active_claim.delivery_attempt_count,
        )

        assert not await repository.dead_letter(
            stale_claim,
            status_code=409,
            error_code="identity_mismatch",
        )

        persisted = await client.fetchone(
            """
            SELECT
                state,
                lease_token::text AS lease_token,
                lease_expires_at IS NOT NULL AS lease_expires_at_present,
                last_http_status,
                last_error_code,
                delivery_attempt_count
            FROM deepsearch_evidence_outbox
            """
        )
        assert persisted is not None
        assert dict(persisted) == {
            "state": "leased",
            "lease_token": active_claim.lease_token,
            "lease_expires_at_present": True,
            "last_http_status": None,
            "last_error_code": None,
            "delivery_attempt_count": 1,
        }
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    (
        (400, "client_rejected"),
        (401, "authentication"),
        (403, "authorization"),
        (404, "client_rejected"),
        (409, "identity_mismatch"),
        (422, "client_rejected"),
    ),
)
@pytest.mark.asyncio
async def test_deterministic_4xx_is_dead_lettered_in_postgres(
    postgres_15_url: str,
    status_code: int,
    expected_category: str,
) -> None:
    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        await _seed_pending_outbox(client, 1)
        repository = EvidenceOutboxRepository(client)
        transport = _OneShotTransport(_status_result(status_code))
        dispatcher = EvidenceOutboxDispatcher(
            repository,
            transport,
            claim_limit=1,
        )

        assert await dispatcher.dispatch_once() == 1

        persisted = await client.fetchone(
            """
            SELECT
                state,
                lease_token,
                lease_expires_at,
                acked_at,
                last_http_status,
                last_error_code,
                delivery_attempt_count
            FROM deepsearch_evidence_outbox
            """
        )
        assert persisted is not None
        assert dict(persisted) == {
            "state": "dead_letter",
            "lease_token": None,
            "lease_expires_at": None,
            "acked_at": None,
            "last_http_status": status_code,
            "last_error_code": expected_category,
            "delivery_attempt_count": 1,
        }
        assert len(transport.triggers) == 1
        assert dispatcher.stats["dead_lettered"] == 1
        assert dispatcher.stats["retry_scheduled"] == 0
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    (
        (429, "rate_limited"),
        (500, "server_error"),
        (503, "server_error"),
        (None, "transport_error"),
    ),
)
@pytest.mark.asyncio
async def test_retryable_delivery_returns_to_pending_in_postgres(
    postgres_15_url: str,
    status_code: int | None,
    expected_category: str,
) -> None:
    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        await _seed_pending_outbox(client, 1)
        repository = EvidenceOutboxRepository(client)
        result = (
            _status_result(status_code)
            if status_code is not None
            else EvidenceLedgerDeliveryResult(
                success=False,
                attempts=1,
                retryable=True,
                category="transport_error",
                final_status_code=None,
            )
        )
        transport = _OneShotTransport(result)
        dispatcher = EvidenceOutboxDispatcher(
            repository,
            transport,
            claim_limit=1,
        )

        assert await dispatcher.dispatch_once() == 1

        persisted = await client.fetchone(
            """
            SELECT
                state,
                lease_token,
                lease_expires_at,
                acked_at,
                last_http_status,
                last_error_code,
                delivery_attempt_count,
                next_attempt_at > created_at AS retry_scheduled_after_creation
            FROM deepsearch_evidence_outbox
            """
        )
        assert persisted is not None
        assert dict(persisted) == {
            "state": "pending",
            "lease_token": None,
            "lease_expires_at": None,
            "acked_at": None,
            "last_http_status": status_code,
            "last_error_code": expected_category,
            "delivery_attempt_count": 1,
            "retry_scheduled_after_creation": True,
        }
        assert len(transport.triggers) == 1
        assert dispatcher.stats["retry_scheduled"] == 1
        assert dispatcher.stats["dead_lettered"] == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_retryable_twelfth_attempt_dead_letters_as_exhausted_in_postgres(
    postgres_15_url: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The final wire attempt cannot return to pending or reach attempt 13."""

    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        await _seed_pending_outbox(client, 1)
        await client.execute(
            """
            UPDATE deepsearch_evidence_outbox
            SET delivery_attempt_count = 11,
                next_attempt_at = NOW() - INTERVAL '1 second'
            """
        )
        repository = EvidenceOutboxRepository(client)
        transport = _OneShotTransport(_status_result(503))
        dispatcher = EvidenceOutboxDispatcher(
            repository,
            transport,
            claim_limit=1,
        )

        assert await dispatcher.dispatch_once() == 1

        persisted = await client.fetchone(
            """
            SELECT
                state,
                lease_token,
                lease_expires_at,
                last_http_status,
                last_error_code,
                delivery_attempt_count
            FROM deepsearch_evidence_outbox
            """
        )
        assert persisted is not None
        assert dict(persisted) == {
            "state": "dead_letter",
            "lease_token": None,
            "lease_expires_at": None,
            "last_http_status": 503,
            "last_error_code": "attempts_exhausted",
            "delivery_attempt_count": 12,
        }
        assert dispatcher.stats["dead_lettered"] == 1
        assert dispatcher.stats["retry_scheduled"] == 0
        assert "category=attempts_exhausted" in caplog.text
        assert "attempt=12" in caplog.text
        assert transport.triggers[0].mission_uuid not in caplog.text
        assert transport.triggers[0].deepsearch_job_id not in caplog.text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_expired_final_lease_sweep_is_bounded_and_token_fenced_in_postgres(
    postgres_15_url: str,
) -> None:
    """A crashed final attempt sweeps at most 25 rows per transaction."""

    client = await _connect(postgres_15_url)
    try:
        await _reset_schema(client)
        await _seed_pending_outbox(client, 30)
        await client.execute(
            """
            UPDATE deepsearch_evidence_outbox
            SET delivery_attempt_count = 11,
                next_attempt_at = NOW() - INTERVAL '1 second'
            """
        )
        repository = EvidenceOutboxRepository(client)
        first_claims = await repository.claim_due(limit=25, lease_seconds=60)
        second_claims = await repository.claim_due(limit=25, lease_seconds=60)
        assert len(first_claims) == 25
        assert len(second_claims) == 5
        assert {
            claim.delivery_attempt_count for claim in [*first_claims, *second_claims]
        } == {12}
        assert len(
            {claim.lease_token for claim in [*first_claims, *second_claims]}
        ) == 30
        await client.execute(
            """
            UPDATE deepsearch_evidence_outbox
            SET
                lease_expires_at = NOW() - INTERVAL '1 second',
                next_attempt_at = NOW() - INTERVAL '1 second'
            WHERE state = 'leased'
            """
        )

        assert await repository.sweep_exhausted() == 25
        after_first = await client.fetchone(
            """
            SELECT
                COUNT(*) FILTER (WHERE state = 'dead_letter') AS dead_letter,
                COUNT(*) FILTER (WHERE state = 'leased') AS leased
            FROM deepsearch_evidence_outbox
            """
        )
        assert after_first is not None
        assert dict(after_first) == {"dead_letter": 25, "leased": 5}

        assert await repository.sweep_exhausted() == 5
        final = await client.fetchone(
            """
            SELECT
                COUNT(*) AS cardinality,
                COUNT(*) FILTER (
                    WHERE state = 'dead_letter'
                      AND delivery_attempt_count = 12
                      AND lease_token IS NULL
                      AND lease_expires_at IS NULL
                      AND last_error_code = 'attempts_exhausted'
                ) AS exhausted
            FROM deepsearch_evidence_outbox
            """
        )
        assert final is not None
        assert dict(final) == {"cardinality": 30, "exhausted": 30}
        assert await repository.sweep_exhausted() == 0
    finally:
        await client.close()
