"""Unit tests for the Mission Poller module."""

import asyncio
import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepsearch.worker.lease import LeaseAttempt, LeasePolicy
from deepsearch.worker.poller import (
    MissionPoller,
    MissionRow,
    PollerError,
)

_LEASE_TIME = datetime(2026, 7, 18, 4, 0, 0)
_LEASE_POLICY = LeasePolicy(
    lease_seconds=120,
    heartbeat_seconds=30,
    max_attempts=3,
)


def _lease_attempt() -> LeaseAttempt:
    return LeaseAttempt(
        owner="worker-a",
        token="opaque-token-a",
        leased_at=_LEASE_TIME,
        heartbeat_at=_LEASE_TIME,
        expires_at=_LEASE_TIME,
        attempt_count=1,
    )


def _lease_fields() -> dict[str, object]:
    lease = _lease_attempt()
    return {
        "deepsearch_lease_owner": lease.owner,
        "deepsearch_lease_token": lease.token,
        "deepsearch_leased_at": lease.leased_at,
        "deepsearch_heartbeat_at": lease.heartbeat_at,
        "deepsearch_lease_expires_at": lease.expires_at,
        "deepsearch_attempt_count": lease.attempt_count,
    }


def _make_poller(db_client: Any, **kwargs: Any) -> MissionPoller:
    return MissionPoller(
        db_client,
        lease_owner="worker-a",
        lease_policy=_LEASE_POLICY,
        **kwargs,
    )


class TestMissionRow:
    """Tests for MissionRow dataclass (TraceLab schema)."""

    def test_mission_row_basic(self):
        """Test basic MissionRow creation."""
        row = MissionRow(
            id="test-123",
            mission_id="B17.1",
            title="Test Mission",
            objective="Research Python vs Node.js",
            context={"background": "test"},
            status="queued",
            queued_at=datetime(2025, 1, 1, 12, 0, 0),
            lease=_lease_attempt(),
        )
        assert row.id == "test-123"
        assert row.mission_id == "B17.1"
        assert row.title == "Test Mission"
        assert row.objective == "Research Python vs Node.js"
        assert row.status == "queued"
        assert row.success_criteria is None
        assert row.research_phases is None

    def test_mission_row_with_optional_fields(self):
        """Test MissionRow with all optional fields."""
        row = MissionRow(
            id="test-456",
            mission_id="B17.2",
            title="Full Mission",
            objective="Analyze performance",
            context={"background": "full test"},
            status="in_progress",
            queued_at=datetime(2025, 1, 1, 12, 0, 0),
            lease=_lease_attempt(),
            success_criteria=["criterion1", "criterion2"],
            research_phases={"max_loops": 5},
        )
        assert row.success_criteria == ["criterion1", "criterion2"]
        assert row.research_phases == {"max_loops": 5}


class TestMissionPollerInit:
    """Tests for MissionPoller initialization."""

    def test_default_poll_interval(self):
        """Test default poll interval is 5 seconds."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client)
        assert poller.poll_interval == 5.0

    def test_custom_poll_interval(self):
        """Test custom poll interval."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client, poll_interval=10.0)
        assert poller.poll_interval == 10.0

    def test_poll_interval_from_env(self):
        """Test poll interval from environment variable."""
        mock_client = MagicMock()
        with patch.dict(os.environ, {"WORKER_POLL_INTERVAL": "15"}):
            poller = _make_poller(mock_client)
            assert poller.poll_interval == 15.0

    def test_invalid_env_poll_interval_uses_default(self):
        """Test invalid env value falls back to default."""
        mock_client = MagicMock()
        with patch.dict(os.environ, {"WORKER_POLL_INTERVAL": "invalid"}):
            poller = _make_poller(mock_client)
            assert poller.poll_interval == 5.0

    def test_jitter_enabled_by_default(self):
        """Test jitter is enabled by default."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client)
        assert poller.enable_jitter is True

    def test_jitter_can_be_disabled(self):
        """Test jitter can be disabled."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client, enable_jitter=False)
        assert poller.enable_jitter is False


class TestMissionPollerJitter:
    """Tests for poll interval jitter calculation."""

    def test_sleep_duration_without_jitter(self):
        """Test sleep duration is exact when jitter disabled."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client, poll_interval=5.0, enable_jitter=False)

        for _ in range(10):
            duration = poller._calculate_sleep_duration()
            assert duration == 5.0

    def test_sleep_duration_with_jitter(self):
        """Test sleep duration varies with jitter enabled."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client, poll_interval=5.0, enable_jitter=True)

        durations = [poller._calculate_sleep_duration() for _ in range(100)]

        # Should have some variation
        assert min(durations) != max(durations)

        # Should be within 10% of base interval
        for d in durations:
            assert 4.5 <= d <= 5.5  # 5 +/- 0.5 (10%)

    def test_sleep_duration_minimum(self):
        """Test sleep duration has a minimum of 0.1s."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client, poll_interval=0.05, enable_jitter=True)

        for _ in range(10):
            duration = poller._calculate_sleep_duration()
            assert duration >= 0.1


class TestParseMissionRow:
    """Tests for _parse_mission_row method (TraceLab schema)."""

    def test_parse_basic_record(self):
        """Test parsing a basic database record."""
        mock_record = {
            **_lease_fields(),
            "id": "mission-123",
            "mission_id": "B17.1",
            "title": "Test Mission",
            "objective": "Research Python vs Node.js",
            "context": {"background": "test context"},
            "status": "in_progress",
            "queued_at": datetime(2025, 1, 1, 12, 0, 0),
            "success_criteria": None,
            "research_phases": None,
        }

        mission = MissionPoller._parse_mission_row(mock_record)

        assert mission.id == "mission-123"
        assert mission.mission_id == "B17.1"
        assert mission.title == "Test Mission"
        assert mission.objective == "Research Python vs Node.js"
        assert mission.context == {"background": "test context"}
        assert mission.status == "in_progress"

    def test_parse_success_criteria_from_json_string(self):
        """Test parsing success_criteria from JSON string."""
        mock_record = {
            **_lease_fields(),
            "id": "mission-123",
            "mission_id": "B17.1",
            "title": "Test Mission",
            "objective": "Research something",
            "context": {},
            "status": "queued",
            "queued_at": datetime(2025, 1, 1),
            "success_criteria": '["criterion1", "criterion2"]',
            "research_phases": None,
        }

        mission = MissionPoller._parse_mission_row(mock_record)
        assert mission.success_criteria == ["criterion1", "criterion2"]

    def test_parse_research_phases_from_json_string(self):
        """Test parsing research_phases from JSON string."""
        mock_record = {
            **_lease_fields(),
            "id": "mission-123",
            "mission_id": "B17.1",
            "title": "Test Mission",
            "objective": "Research something",
            "context": {},
            "status": "queued",
            "queued_at": datetime(2025, 1, 1),
            "success_criteria": None,
            "research_phases": '{"max_loops": 5}',
        }

        mission = MissionPoller._parse_mission_row(mock_record)
        assert mission.research_phases == {"max_loops": 5}

    def test_parse_context_from_json_string(self):
        """Test parsing context from JSON string."""
        mock_record = {
            **_lease_fields(),
            "id": "mission-123",
            "mission_id": "B17.1",
            "title": "Test Mission",
            "objective": "Research something",
            "context": '{"background": "from json"}',
            "status": "queued",
            "queued_at": datetime(2025, 1, 1),
            "success_criteria": None,
            "research_phases": None,
        }

        mission = MissionPoller._parse_mission_row(mock_record)
        assert mission.context == {"background": "from json"}

    def test_parse_context_none(self):
        """Test parsing with None context."""
        mock_record = {
            **_lease_fields(),
            "id": "mission-123",
            "mission_id": "B17.1",
            "title": "Test Mission",
            "objective": "Research something",
            "context": None,
            "status": "queued",
            "queued_at": datetime(2025, 1, 1),
            "success_criteria": None,
            "research_phases": None,
        }

        mission = MissionPoller._parse_mission_row(mock_record)
        assert mission.context == {}

    def test_parse_with_all_fields(self):
        """Test parsing with all optional fields populated."""
        mock_record = {
            **_lease_fields(),
            "id": "mission-123",
            "mission_id": "B17.1",
            "title": "Test Mission",
            "objective": "Research something",
            "context": {"background": "test"},
            "status": "queued",
            "queued_at": datetime(2025, 1, 1),
            "success_criteria": ["crit1", "crit2"],
            "research_phases": {"max_loops": 5},
        }

        mission = MissionPoller._parse_mission_row(mock_record)
        assert mission.success_criteria == ["crit1", "crit2"]
        assert mission.research_phases == {"max_loops": 5}

    def test_parse_with_missing_mission_id(self):
        """Test parsing falls back to truncated id when mission_id is None."""
        mock_record = {
            **_lease_fields(),
            "id": "abcdefgh-1234-5678",
            "mission_id": None,
            "title": "Test Mission",
            "objective": "Research something",
            "context": {},
            "status": "queued",
            "queued_at": datetime(2025, 1, 1),
            "success_criteria": None,
            "research_phases": None,
        }

        mission = MissionPoller._parse_mission_row(mock_record)
        assert mission.mission_id == "abcdefgh"


def create_mock_connection(fetchrow_result=None):
    """Create a properly mocked asyncpg connection.

    In asyncpg, conn.transaction() is a sync method that returns an async context manager.
    conn.fetchrow() and conn.execute() are async methods.
    """
    mock_conn = MagicMock()

    # Setup async transaction context manager (sync method returning async cm)
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction.return_value = mock_tx

    # Setup async methods
    mock_conn.fetchrow = AsyncMock(return_value=fetchrow_result)
    mock_conn.execute = AsyncMock(return_value=None)

    return mock_conn


class TestPollOnce:
    """Tests for poll_once method."""

    @pytest.mark.asyncio
    async def test_poll_once_no_missions(self):
        """Test poll_once returns None when no missions queued."""
        mock_client = AsyncMock()
        mock_conn = create_mock_connection(fetchrow_result=None)

        # Setup connection acquire to return mock connection
        mock_client.acquire.return_value = mock_conn
        mock_client.release = AsyncMock()

        poller = _make_poller(mock_client)
        result = await poller.poll_once()

        assert result is None
        assert poller._poll_count == 1
        assert poller._missions_claimed == 0

    @pytest.mark.asyncio
    async def test_poll_once_claims_mission(self):
        """Test poll_once successfully claims a mission."""
        mock_record = {
            **_lease_fields(),
            "id": "mission-abc",
            "mission_id": "B17.1",
            "title": "Found Mission",
            "objective": "research something",
            "context": {"background": "test"},
            "status": "in_progress",
            "queued_at": datetime(2025, 1, 1, 10, 0, 0),
            "success_criteria": None,
            "research_phases": {"max_loops": 3},
        }

        mock_client = AsyncMock()
        mock_conn = create_mock_connection(fetchrow_result=mock_record)

        mock_client.acquire.return_value = mock_conn
        mock_client.release = AsyncMock()

        poller = _make_poller(mock_client)
        result = await poller.poll_once()

        assert result is not None
        assert result.id == "mission-abc"
        assert result.title == "Found Mission"
        assert poller._missions_claimed == 1

        # Retry exhaustion is swept before the single atomic claim statement.
        mock_conn.execute.assert_called_once()
        assert mock_conn.execute.call_args.args[0] == MissionPoller.DEAD_LETTER_EXHAUSTED_SQL
        assert mock_conn.execute.call_args.args[1:] == (3, 30)
        claim_args = mock_conn.fetchrow.call_args.args
        assert claim_args[0] == MissionPoller.CLAIM_MISSION_SQL
        # No project parameter: claims span every project (RBAC scopes access,
        # not the worker). Owner is now the first bind parameter.
        assert claim_args[1] == "worker-a"
        assert isinstance(claim_args[2], str) and claim_args[2]
        assert claim_args[3:] == (120, 3, 30)

    @pytest.mark.asyncio
    async def test_poll_once_handles_error(self):
        """Test poll_once raises PollerError on database failure."""
        mock_client = AsyncMock()
        mock_client.acquire.side_effect = Exception("Connection failed")

        poller = _make_poller(mock_client)

        with pytest.raises(PollerError) as exc_info:
            await poller.poll_once()

        assert "Connection failed" in str(exc_info.value)


class TestPollerStats:
    """Tests for poller statistics."""

    def test_initial_stats(self):
        """Test initial stats are zero."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client)

        stats = poller.stats
        assert stats["running"] is False
        assert stats["poll_count"] == 0
        assert stats["missions_claimed"] == 0
        assert stats["poll_interval"] == 5.0
        assert stats["jitter_enabled"] is True

    @pytest.mark.asyncio
    async def test_stats_after_polls(self):
        """Test stats update after polling."""
        mock_client = AsyncMock()
        mock_conn = create_mock_connection(fetchrow_result=None)

        mock_client.acquire.return_value = mock_conn
        mock_client.release = AsyncMock()

        poller = _make_poller(mock_client)

        await poller.poll_once()
        await poller.poll_once()
        await poller.poll_once()

        stats = poller.stats
        assert stats["poll_count"] == 3
        assert stats["missions_claimed"] == 0


class TestPollerLifecycle:
    """Tests for poller start/stop lifecycle."""

    def test_is_running_initially_false(self):
        """Test poller is not running initially."""
        mock_client = MagicMock()
        poller = _make_poller(mock_client)
        assert poller.is_running is False

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test starting and stopping the poller."""
        mock_client = AsyncMock()
        mock_conn = create_mock_connection(fetchrow_result=None)

        mock_client.acquire.return_value = mock_conn
        mock_client.release = AsyncMock()

        poller = _make_poller(mock_client, poll_interval=0.1, enable_jitter=False)

        # Start in background
        start_task = asyncio.create_task(poller.start())

        # Wait a bit for it to start polling
        await asyncio.sleep(0.15)
        assert poller.is_running is True
        assert poller._poll_count >= 1

        # Stop the poller
        await poller.stop()

        # Wait for task to complete
        await asyncio.wait_for(start_task, timeout=1.0)

        assert poller.is_running is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test poller as async context manager."""
        mock_client = AsyncMock()
        mock_conn = create_mock_connection(fetchrow_result=None)

        mock_client.acquire.return_value = mock_conn
        mock_client.release = AsyncMock()

        async with _make_poller(mock_client) as poller:
            result = await poller.poll_once()
            assert result is None
            assert poller._poll_count == 1


class TestPollerCallback:
    """Tests for mission callback handling."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_mission(self):
        """Test on_mission callback is invoked when mission claimed."""
        # First poll returns a mission, second returns None
        mission_record = {
            **_lease_fields(),
            "id": "mission-xyz",
            "mission_id": "B17.1",
            "title": "Callback Test",
            "objective": "test callback",
            "context": {},
            "status": "queued",
            "queued_at": datetime(2025, 1, 1),
            "success_criteria": None,
            "research_phases": None,
        }

        mock_client = AsyncMock()
        mock_conn = create_mock_connection(fetchrow_result=None)
        # Override to return mission first, then None
        mock_conn.fetchrow.side_effect = [mission_record, None, None, None, None]

        mock_client.acquire.return_value = mock_conn
        mock_client.release = AsyncMock()

        received_missions = []

        async def capture_mission(mission: MissionRow) -> None:
            received_missions.append(mission)

        poller = _make_poller(
            mock_client,
            poll_interval=0.05,
            on_mission=capture_mission,
            enable_jitter=False
        )

        # Run poller briefly
        start_task = asyncio.create_task(poller.start())
        await asyncio.sleep(0.1)
        await poller.stop()
        await asyncio.wait_for(start_task, timeout=1.0)

        assert len(received_missions) == 1
        assert received_missions[0].id == "mission-xyz"

    @pytest.mark.asyncio
    async def test_callback_error_does_not_stop_polling(self):
        """Test that callback errors don't stop the polling loop."""
        # Return missions that will cause callback to be called
        mission_record = {
            **_lease_fields(),
            "id": "mission-err",
            "mission_id": "B17.2",
            "title": "Error Test",
            "objective": "test error handling",
            "context": {},
            "status": "queued",
            "queued_at": datetime(2025, 1, 1),
            "success_criteria": None,
            "research_phases": None,
        }

        mock_client = AsyncMock()
        mock_conn = create_mock_connection(fetchrow_result=None)
        mock_conn.fetchrow.side_effect = [mission_record, None, None, None]

        mock_client.acquire.return_value = mock_conn
        mock_client.release = AsyncMock()

        callback_calls = []

        async def failing_callback(mission: MissionRow) -> None:
            callback_calls.append(mission.id)
            raise ValueError("Callback intentionally failed")

        poller = _make_poller(
            mock_client,
            poll_interval=0.05,
            on_mission=failing_callback,
            enable_jitter=False
        )

        start_task = asyncio.create_task(poller.start())
        await asyncio.sleep(0.15)  # Let it poll a few times
        await poller.stop()
        await asyncio.wait_for(start_task, timeout=1.0)

        # Callback was called despite error
        assert len(callback_calls) == 1
        # Poller continued after error (multiple polls happened)
        assert poller._poll_count >= 2


class TestPollerSqlQueries:
    """Tests to verify correct SQL queries are used."""

    def test_claim_query_uses_for_update_skip_locked(self):
        """Verify claim query uses FOR UPDATE SKIP LOCKED."""
        assert "FOR UPDATE SKIP LOCKED" in MissionPoller.CLAIM_MISSION_SQL

    def test_claim_query_orders_by_queued_at(self):
        """Verify claim query orders by queued_at ASC."""
        assert "ORDER BY queued_at ASC" in MissionPoller.CLAIM_MISSION_SQL

    def test_claim_query_limits_to_one(self):
        """Verify claim query limits to 1 result."""
        assert "LIMIT 1" in MissionPoller.CLAIM_MISSION_SQL

    def test_claim_query_filters_queued_status(self):
        """Verify claim query filters for queued status."""
        assert "status = 'queued'" in MissionPoller.CLAIM_MISSION_SQL

    def test_claim_query_sets_in_progress_and_ownership_atomically(self):
        """The one claim statement must write status, token, expiry, and attempt."""
        sql = MissionPoller.CLAIM_MISSION_SQL
        assert "UPDATE missions AS mission" in sql
        assert "status = 'in_progress'" in sql
        assert "deepsearch_lease_owner = $1" in sql
        assert "deepsearch_lease_token = $2" in sql
        assert "deepsearch_lease_expires_at" in sql
        assert "deepsearch_attempt_count" in sql
        assert "RETURNING" in sql

    def test_claim_query_preserves_first_started_at(self):
        """Recovered attempts keep the original first-start timestamp."""
        assert "started_at = COALESCE(mission.started_at, NOW())" in (
            MissionPoller.CLAIM_MISSION_SQL
        )

    def test_expired_recovery_and_active_lease_protection_are_explicit(self):
        sql = MissionPoller.CLAIM_MISSION_SQL
        assert "deepsearch_lease_token IS NULL" in sql
        assert "deepsearch_lease_expires_at IS NOT NULL" in sql
        assert "NOW() - ($5 * INTERVAL '1 second')" in sql
        assert "COALESCE(deepsearch_attempt_count, 0) < $4" in sql
        assert "status = 'completed'" not in sql

    def test_dead_letter_requires_expiry_for_tokened_active_rows(self):
        sql = MissionPoller.DEAD_LETTER_EXHAUSTED_SQL
        assert "deepsearch_lease_token IS NULL" in sql
        assert "deepsearch_lease_expires_at IS NOT NULL" in sql
        assert "NOW() - ($2 * INTERVAL '1 second')" in sql

    def test_explicit_release_clears_ownership_to_bypass_crash_grace(self):
        sql = MissionPoller.RELEASE_SQL
        assert "deepsearch_lease_owner = NULL" in sql
        assert "deepsearch_lease_token = NULL" in sql
        assert "deepsearch_lease_expires_at = NULL" in sql

    def test_claim_query_is_not_scoped_to_one_project(self):
        """Missions are authored across every project; RBAC scopes access.

        A project-equality filter here silently strands every mission outside
        the pinned project - claimed by nothing, logged by nothing.
        """
        for sql in (
            MissionPoller.CLAIM_MISSION_SQL,
            MissionPoller.DEAD_LETTER_EXHAUSTED_SQL,
        ):
            assert "project_id = $" not in sql
            assert "project_id = " not in sql

    def test_claim_query_refuses_rows_that_would_orphan_a_report(self):
        """The result is written back onto this row, so it must have a project."""
        assert "project_id IS NOT NULL" in MissionPoller.CLAIM_MISSION_SQL
