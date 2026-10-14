"""Unit tests for DeepSearch worker main module."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepsearch.agent.execution_control import (
    ExecutionCancelledError,
    raise_if_execution_cancelled,
)
from deepsearch.agent.outcome import MissionOutcome
from deepsearch.report.protocol import MissionProtocolComplete
from deepsearch.tracelab import (
    CompletionReceipt,
    ReceiptDeliveryResult,
    TraceLabAuth,
    reset_default_auth,
)
from deepsearch.worker.converter import ConversionError
from deepsearch.worker.database import DatabaseConfig, DatabaseError
from deepsearch.worker.execution import IsolatedExecutionError, ProcessTerminationError
from deepsearch.worker.lease import ExecutionFencePolicy, LeaseAttempt
from deepsearch.worker.main import DeepSearchWorker, LeaseLostError, WorkerError
from deepsearch.worker.poller import MissionRow
from deepsearch.worker.preflight import PreflightCheck, WorkerPreflightReport
from deepsearch.worker.writer import (
    ExecutionMetadata,
    WriterError,
    identity_sha256,
    protocol_identity_sha256,
)

pytest.importorskip("aiohttp")


# Test fixtures

@pytest.fixture(autouse=True)
def isolate_tracelab_transports(monkeypatch):
    """Unit worker tests must never inherit live TraceLab credentials."""

    for name in (
        "TRACELAB_BASE_URL",
        "TRACELAB_API_KEY",
        "TRACELAB_DEEPSEARCH_SERVICE_SECRET",
        "TRACELAB_USERNAME",
        "TRACELAB_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "deepsearch.worker.main.TraceLabAuth",
        lambda: TraceLabAuth(
            base_url="",
            api_key="",
            username="",
            password="",
        ),
    )
    reset_default_auth()
    yield
    reset_default_auth()

@pytest.fixture
def mock_db_config():
    """Create a mock database configuration."""
    return DatabaseConfig(
        url="postgresql://test:test@localhost:5432/test",
        min_pool_size=1,
        max_pool_size=5,
    )


@pytest.fixture
def sample_mission_row():
    """Create a sample MissionRow for testing."""
    return MissionRow(
        id="test-mission-001",
        mission_id="test-mission-001",
        title="Test Research Mission",
        objective="Research objective 1",
        context={"background": "Test background", "focus": "Test focus"},
        status="queued",
        queued_at=datetime.now(timezone.utc),
        lease=LeaseAttempt(
            owner="worker-test",
            token="opaque-test-token",
            leased_at=datetime.now(timezone.utc),
            heartbeat_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            attempt_count=1,
        ),
        success_criteria=["Research objective 2"],
        research_phases={"max_loops": 3},
        research_depth="baseline",
    )


def _mock_settings_obj(**overrides):
    """Return a serializable DeepSearchSettings-like test double."""

    defaults = {
        "llm_backend": "gemini",
        "llm_request_timeout_seconds": 300.0,
        "checkpoint_db_path": "/tmp/test.db",
        "generate_thread_id": MagicMock(return_value="thread-001"),
        "search_backend": "google",
        "dual_search": False,
        "triple_search": False,
        "tavily_dual_search": False,
        "use_v2_prompts": True,
        "use_new_reflection_parser": True,
        "use_new_citation_engine": True,
        "pedr_enable_graph": False,
        "gemini_model": "gemini-2.5-pro",
        "deepseek_model": "deepseek-chat",
        "minimax_model": "MiniMax-M2.7",
        "ollama_model": "llama3",
    }
    defaults.update(overrides)
    extraction_model = defaults.pop("extraction_model", "gemini-2.5-pro")
    reflection_model = defaults.pop("reflection_model", extraction_model)
    settings = SimpleNamespace(**defaults)
    settings.build_llm_config = MagicMock(return_value=SimpleNamespace(model=extraction_model))
    settings.build_reflection_llm_config = MagicMock(
        return_value=SimpleNamespace(model=reflection_model)
    )
    return settings


def _ready_preflight() -> WorkerPreflightReport:
    return WorkerPreflightReport(
        checked_at=datetime.now(timezone.utc),
        checks={"all": PreflightCheck(True, "ready")},
    )


# DeepSearchWorker tests

class TestDeepSearchWorkerInit:
    """Tests for DeepSearchWorker initialization."""

    def test_init_with_config(self, mock_db_config):
        """Test initialization with explicit config."""
        worker = DeepSearchWorker(
            db_config=mock_db_config,
            poll_interval=10.0,
            llm_backend="gemini",
            model="gemini-2.5-pro",
        )

        assert worker.db_config == mock_db_config
        assert worker.poll_interval == 10.0
        assert worker.llm_backend == "gemini"
        assert worker.model == "gemini-2.5-pro"
        assert worker._shutdown_requested is False
        assert worker._current_mission_id is None

    def test_init_from_environment(self, monkeypatch):
        """Test initialization from environment variables."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test@localhost/db")
        monkeypatch.setenv("WORKER_POLL_INTERVAL", "15.0")

        worker = DeepSearchWorker()

        assert worker.poll_interval == 15.0
        assert worker.db_config.url == "postgresql://test@localhost/db"

    def test_init_default_poll_interval(self, mock_db_config, monkeypatch):
        """Test default poll interval when env var not set."""
        monkeypatch.delenv("WORKER_POLL_INTERVAL", raising=False)

        worker = DeepSearchWorker(db_config=mock_db_config)

        assert worker.poll_interval == 5.0

    @pytest.mark.parametrize(
        ("backend", "expected_model"),
        [
            ("deepseek", "deepseek-chat"),
            ("minimax", "MiniMax-M2.7"),
        ],
    )
    def test_init_uses_backend_default_model(
        self,
        mock_db_config,
        backend: str,
        expected_model: str,
    ) -> None:
        """Non-Gemini backends should resolve their own default models."""
        with patch(
            "deepsearch.worker.main.DeepSearchSettings.load",
            return_value=_mock_settings_obj(llm_backend=backend),
        ):
            worker = DeepSearchWorker(db_config=mock_db_config, llm_backend=backend)

        assert worker.llm_backend == backend
        assert worker.model == expected_model

    def test_stats_initial_state(self, mock_db_config):
        """Test initial statistics state."""
        worker = DeepSearchWorker(db_config=mock_db_config)
        stats = worker.stats

        assert stats["missions_processed"] == 0
        assert stats["missions_completed"] == 0
        assert stats["missions_failed"] == 0
        assert stats["current_mission_id"] is None
        assert stats["shutdown_requested"] is False

    def test_is_processing_initially_false(self, mock_db_config):
        """Test is_processing property initially False."""
        worker = DeepSearchWorker(db_config=mock_db_config)
        assert worker.is_processing is False


class TestDeepSearchWorkerRun:
    """Tests for DeepSearchWorker.run() method."""

    @pytest.mark.asyncio
    async def test_run_fails_without_database(self, mock_db_config):
        """Test that run fails if database connection fails."""
        # Disable health server to avoid port binding in tests
        worker = DeepSearchWorker(db_config=mock_db_config, enable_health_server=False)

        with patch.object(worker, "_db_client") as mock_client:
            mock_client.connect = AsyncMock(side_effect=DatabaseError("Connection failed"))

            with patch("deepsearch.worker.main.DatabaseClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.connect = AsyncMock(side_effect=DatabaseError("Connection failed"))
                mock_instance.close = AsyncMock()

                with pytest.raises(WorkerError, match="Database connection failed"):
                    await worker.run()

    @pytest.mark.asyncio
    async def test_cleanup_called_on_shutdown(self, mock_db_config):
        """Test that cleanup is called when worker shuts down."""
        # Disable health server to avoid port binding in tests
        worker = DeepSearchWorker(db_config=mock_db_config, enable_health_server=False)

        mock_db = AsyncMock()
        mock_poller = AsyncMock()
        mock_poller.start = AsyncMock(return_value=None)
        mock_poller.stop = AsyncMock()

        with patch("deepsearch.worker.main.DatabaseClient", return_value=mock_db):
            with patch("deepsearch.worker.main.MissionPoller", return_value=mock_poller):
                with patch("deepsearch.worker.main.ResultWriter"):
                    with patch(
                        "deepsearch.worker.main.run_worker_preflight",
                        new=AsyncMock(return_value=_ready_preflight()),
                    ):
                        worker.project_id = "11111111-2222-4333-8444-555555555555"
                        await worker.run()

        mock_db.close.assert_called_once()
        mock_poller.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_durable_dispatcher_brackets_poller_and_database(
        self,
        mock_db_config,
    ) -> None:
        """Recovery starts before claims and stops before its database closes."""

        events: list[str] = []
        mock_db = AsyncMock()
        mock_db.connect.side_effect = lambda: events.append("database_connected")
        mock_db.close.side_effect = lambda: events.append("database_closed")
        mock_poller = AsyncMock()
        mock_poller.start.side_effect = lambda: events.append("poller_started")
        mock_poller.stop.side_effect = lambda: events.append("poller_stopped")
        mock_dispatcher = AsyncMock()
        mock_dispatcher.start.side_effect = lambda: events.append("dispatcher_started")
        mock_dispatcher.stop.side_effect = lambda: events.append("dispatcher_stopped")
        mock_repository = MagicMock()
        worker = DeepSearchWorker(
            db_config=mock_db_config,
            enable_health_server=False,
        )

        with (
            patch("deepsearch.worker.main.DatabaseClient", return_value=mock_db),
            patch(
                "deepsearch.worker.main.MissionPoller",
                return_value=mock_poller,
            ),
            patch("deepsearch.worker.main.ResultWriter"),
            patch(
                "deepsearch.worker.main.EvidenceOutboxRepository",
                return_value=mock_repository,
            ) as repository_class,
            patch(
                "deepsearch.worker.main.EvidenceOutboxDispatcher",
                return_value=mock_dispatcher,
            ) as dispatcher_class,
            patch(
                "deepsearch.worker.main.run_worker_preflight",
                new=AsyncMock(return_value=_ready_preflight()),
            ),
        ):
            await worker.run()

        repository_class.assert_called_once_with(mock_db)
        dispatcher_class.assert_called_once_with(
            mock_repository,
            worker._evidence_ledger_client,
        )
        assert worker._evidence_ledger_client.auth is worker._tracelab_auth
        assert worker._evidence_ledger_client.max_attempts == 1
        assert events == [
            "database_connected",
            "dispatcher_started",
            "poller_started",
            "poller_stopped",
            "dispatcher_stopped",
            "database_closed",
        ]


class TestWorkerLeaseLifecycle:
    """Mission execution must continuously prove and safely release ownership."""

    @pytest.mark.asyncio
    async def test_transport_retries_wait_until_claim_is_resolved(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """Old receipt/log retries cannot consume or orphan an active lease."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        poller = MagicMock()
        poller.release = AsyncMock(return_value=True)
        poller.heartbeat_audit.return_value = {
            "heartbeat_count": 0,
            "last_confirmed_heartbeat_at": None,
            "last_confirmed_expires_at": None,
        }
        worker._poller = poller
        writer = MagicMock()
        writer.mark_completed = AsyncMock(return_value=True)
        writer.mark_requeued = AsyncMock(return_value=True)
        writer.mark_failed = AsyncMock(return_value=True)
        worker._writer = writer
        heartbeat_started = asyncio.Event()
        heartbeat_stopped = asyncio.Event()

        async def _heartbeat(_row) -> None:
            heartbeat_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                heartbeat_stopped.set()
                raise

        async def _retry_transports(**_kwargs) -> None:
            assert heartbeat_started.is_set()
            assert heartbeat_stopped.is_set()
            writer.mark_failed.assert_awaited_once()

        with (
            patch.object(worker, "_heartbeat_lease", side_effect=_heartbeat),
            patch.object(
                worker,
                "_retry_pending_transports",
                side_effect=_retry_transports,
            ),
            patch.object(
                worker,
                "_execute_research",
                new=AsyncMock(side_effect=RuntimeError("research crashed")),
            ),
        ):
            await worker._handle_mission(sample_mission_row)

        assert heartbeat_started.is_set()
        assert heartbeat_stopped.is_set()
        writer.mark_failed.assert_awaited_once()
        metadata = writer.mark_failed.call_args.kwargs["execution_metadata"]
        assert metadata.lease_attempt["contract_version"] == (
            "tracelab-missions-lease-v2"
        )
        assert metadata.lease_attempt["heartbeat_runtime"] == {
            "heartbeat_count": 0,
            "last_confirmed_heartbeat_at": None,
            "last_confirmed_expires_at": None,
        }
        assert metadata.lease_attempt["terminal_reset"] == {
            "deepsearch_lease_token": "cleared_atomically",
            "deepsearch_lease_expires_at": "cleared_atomically",
        }

    @pytest.mark.asyncio
    async def test_heartbeat_task_fails_loud_when_ownership_is_lost(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        worker = DeepSearchWorker(db_config=mock_db_config)
        poller = MagicMock()
        poller.heartbeat = AsyncMock(return_value=False)
        worker._poller = poller

        with patch("deepsearch.worker.main.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(LeaseLostError, match="lease attempt 1 lost"):
                await worker._heartbeat_lease(sample_mission_row)

        poller.heartbeat.assert_awaited_once_with(
            sample_mission_row.id,
            sample_mission_row.lease,
        )

    @pytest.mark.asyncio
    async def test_heartbeat_io_timeout_fails_closed_as_lease_loss(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """Pool acquisition or reconnect cannot make ownership detection infinite."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        worker.execution_fence_policy = ExecutionFencePolicy(
            heartbeat_timeout_seconds=0.01,
            cancellation_grace_seconds=0.1,
            terminate_grace_seconds=0.1,
            kill_grace_seconds=0.1,
            safety_margin_seconds=1.0,
            provider_limits=worker.execution_fence_policy.provider_limits,
        )
        poller = MagicMock()

        async def never_confirms(*_args) -> bool:
            await asyncio.Event().wait()
            return True

        poller.heartbeat = AsyncMock(side_effect=never_confirms)
        worker._poller = poller

        with patch("deepsearch.worker.main.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(LeaseLostError, match="could not prove ownership"):
                await worker._heartbeat_lease(sample_mission_row)

        poller.heartbeat.assert_awaited_once_with(
            sample_mission_row.id,
            sample_mission_row.lease,
        )

    @pytest.mark.asyncio
    async def test_heartbeat_deadline_uses_monotonic_request_anchor_not_db_clock(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """DB/worker wall-clock skew cannot move the child past reclaim."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        poller = MagicMock()
        poller.heartbeat = AsyncMock(return_value=True)
        poller.heartbeat_audit.return_value = {
            "heartbeat_count": 1,
            "last_confirmed_heartbeat_at": "2099-01-01T00:00:00+00:00",
            "last_confirmed_expires_at": "2099-01-01T00:02:00+00:00",
        }
        worker._poller = poller
        before = time.monotonic()

        anchor = await worker._confirm_lease_ownership(sample_mission_row)

        after = time.monotonic()
        key = (
            sample_mission_row.id,
            sample_mission_row.lease.token,
            sample_mission_row.lease.attempt_count,
        )
        hard_deadline = worker._lease_deadlines_monotonic[key]
        assert before <= anchor <= after
        assert before + 140.0 <= hard_deadline <= after + 140.0

    @pytest.mark.asyncio
    async def test_production_executor_contains_the_entire_research_workflow(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """Contract, graph, recovery, and artifacts cross one spawn boundary."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        expected_state = {"mission_id": sample_mission_row.id}
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=expected_state)
        worker._mission_executor = executor

        result = await worker._execute_research_with_lease_watch(
            expected_state,  # type: ignore[arg-type]
            sample_mission_row,
            "job-fenced",
            None,
        )

        assert result is expected_state
        target, args = executor.execute.await_args.args
        assert target.__name__ == "_run_isolated_research"
        assert args == (
            expected_state,
            sample_mission_row.id,
            "job-fenced",
            worker.llm_backend,
            worker.model,
        )
        execute_kwargs = executor.execute.await_args.kwargs
        assert execute_kwargs["job_id"] == "job-fenced"
        assert execute_kwargs["heartbeat_task"] is None
        assert execute_kwargs["hard_deadline_monotonic"] > time.monotonic()

    @pytest.mark.asyncio
    async def test_immediate_heartbeat_loss_prevents_child_start_and_writes(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """A stale claim is rejected before the isolated target can begin."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        poller = MagicMock()
        poller.heartbeat = AsyncMock(return_value=False)
        poller.release = AsyncMock(return_value=False)
        poller.heartbeat_audit.return_value = {
            "heartbeat_count": 0,
            "last_confirmed_heartbeat_at": None,
            "last_confirmed_expires_at": None,
        }
        worker._poller = poller
        writer = MagicMock()
        writer.mark_completed = AsyncMock()
        writer.mark_requeued = AsyncMock()
        writer.mark_failed = AsyncMock()
        worker._writer = writer
        executor = MagicMock()
        executor.execute = AsyncMock()
        executor.is_active.return_value = False
        worker._mission_executor = executor

        with patch.object(worker, "_retry_pending_transports", new=AsyncMock()):
            with pytest.raises(LeaseLostError, match="lease attempt 1 lost"):
                await worker._handle_mission(sample_mission_row)

        executor.execute.assert_not_awaited()
        writer.mark_completed.assert_not_awaited()
        writer.mark_requeued.assert_not_awaited()
        writer.mark_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deadline_channel_failure_is_lease_loss_with_full_cleanup(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """A broken child watchdog cannot bypass release or handler cleanup."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        poller = MagicMock()
        poller.heartbeat = AsyncMock(return_value=True)
        poller.heartbeat_audit.return_value = {
            "heartbeat_count": 1,
            "last_confirmed_heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "last_confirmed_expires_at": (
                datetime.now(timezone.utc) + timedelta(seconds=120)
            ).isoformat(),
        }
        poller.release = AsyncMock(return_value=True)
        worker._poller = poller
        writer = MagicMock()
        writer.mark_completed = AsyncMock()
        writer.mark_requeued = AsyncMock()
        writer.mark_failed = AsyncMock()
        worker._writer = writer
        executor = MagicMock()
        executor.extend_deadline.side_effect = IsolatedExecutionError(
            "watchdog pipe closed"
        )
        executor.is_active.return_value = False
        executor.execute = AsyncMock()
        worker._mission_executor = executor
        deepsearch_logger = logging.getLogger("deepsearch")
        handlers_before = list(deepsearch_logger.handlers)

        with patch.object(worker, "_retry_pending_transports", new=AsyncMock()):
            with pytest.raises(LeaseLostError, match="deadline watchdog"):
                await worker._handle_mission(sample_mission_row)

        executor.execute.assert_not_awaited()
        executor.cleanup.assert_called_once()
        poller.release.assert_awaited_once_with(
            sample_mission_row.id,
            sample_mission_row.lease,
        )
        writer.mark_completed.assert_not_awaited()
        writer.mark_requeued.assert_not_awaited()
        writer.mark_failed.assert_not_awaited()
        assert deepsearch_logger.handlers == handlers_before
        assert worker._current_mission_id is None
        assert worker._current_job_id is None

    @pytest.mark.asyncio
    async def test_live_delivery_is_confirmed_watched_and_quiesced_before_release(
        self, mock_db_config, sample_mission_row,
    ) -> None:
        """Transport never races a successor or consumes the heartbeat event loop."""
        worker = DeepSearchWorker(db_config=mock_db_config)
        handler = MagicMock()
        handler.level = logging.NOTSET
        handler.get_telemetry.return_value = {"pending_entries": 0}
        poller = MagicMock()
        watch_started = asyncio.Event()
        order = []
        async def confirm(row):
            order.append("confirmed")
        async def heartbeat(row):
            watch_started.set()
            await asyncio.Future()
        def activate():
            assert watch_started.is_set()
            assert order == ["confirmed"]
            order.append("delivery")
        async def research(*args):
            assert order == ["confirmed", "delivery"]
            raise LeaseLostError("controlled loss")
        async def release(*args):
            handler.stop_delivery.assert_called()
            handler.discard.assert_called_once()
            order.append("released")
            return True
        handler.start_live_delivery.side_effect = activate
        poller.release = AsyncMock(side_effect=release)
        worker._poller = poller
        worker._mission_executor = MagicMock()
        worker._mission_executor.is_active.return_value = False
        worker._writer = MagicMock()
        with (
            patch("deepsearch.worker.main.TracelabLogHandler", return_value=handler),
            patch.object(worker, "_confirm_lease_ownership", side_effect=confirm),
            patch.object(worker, "_heartbeat_lease", side_effect=heartbeat),
            patch.object(worker, "_execute_research_with_lease_watch", side_effect=research),
            patch.object(worker, "_retry_pending_transports", new=AsyncMock()),
        ):
            with pytest.raises(LeaseLostError, match="controlled loss"):
                await worker._handle_mission(sample_mission_row)
        assert order == ["confirmed", "delivery", "released"]
        worker._writer.mark_completed.assert_not_called()
        worker._writer.mark_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_transient_requeue_discards_attempt_logs_before_retry(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """A queued successor cannot overlap attempt-A log delivery."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        writer = MagicMock()
        writer.mark_completed = AsyncMock()
        writer.mark_requeued = AsyncMock()
        writer.mark_failed = AsyncMock()
        worker._writer = writer
        log_handler = MagicMock()
        log_handler.level = logging.NOTSET
        log_handler.get_telemetry.return_value = {
            "pending_entries": 0,
            "discarded_entries": 1,
        }
        async def requeue_after_quiescence(**kwargs):
            log_handler.discard.assert_called_once_with()
        writer.mark_requeued.side_effect = requeue_after_quiescence
        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": MissionOutcome.BLOCKED.value,
            "routing_reason": "transient provider outage",
            "synthesis_used": True,
            "report_markdown": None,
            "protocol_payload": None,
            "synthesis_telemetry": {
                "synthesis_error_transient": True,
                "report_word_count": 0,
            },
        }
        protocol = MagicMock(spec=MissionProtocolComplete)
        protocol.model_dump.return_value = {"title": "retryable"}

        with (
            patch(
                "deepsearch.worker.main.TracelabLogHandler",
                return_value=log_handler,
            ) as handler_factory,
            patch.object(
                worker,
                "_execute_research",
                new=AsyncMock(return_value=final_state),
            ),
            patch(
                "deepsearch.worker.main.normalize_final_telemetry",
                return_value={},
            ),
            patch(
                "deepsearch.worker.main.resolve_final_outcome_payload",
                return_value={
                    "outcome": MissionOutcome.BLOCKED.value,
                    "routing_reason": "transient provider outage",
                },
            ),
            patch(
                "deepsearch.worker.main.generate_markdown_report",
                return_value="# Retry later",
            ) as legacy_report,
            patch(
                "deepsearch.worker.main.generate_mission_protocol",
                return_value=protocol,
            ) as legacy_protocol,
            patch("deepsearch.worker.main.assert_result_identity"),
            patch.object(worker, "_retry_pending_transports", new=AsyncMock()),
        ):
            await worker._handle_mission(sample_mission_row)

        handler_factory.assert_called_once_with(
            mission_id=sample_mission_row.id,
            auth=worker._tracelab_auth,
            lease=sample_mission_row.lease,
        )
        writer.mark_requeued.assert_awaited_once()
        writer.mark_completed.assert_not_awaited()
        writer.mark_failed.assert_not_awaited()
        legacy_report.assert_not_called()
        legacy_protocol.assert_not_called()
        assert log_handler.discard.call_count == 2  # Before requeue and idempotent cleanup.
        log_handler.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_unproven_child_termination_never_writes_or_releases(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """A live paid child quarantines the attempt and stops the worker."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        poller = MagicMock()
        poller.heartbeat = AsyncMock(return_value=True)
        poller.heartbeat_audit.return_value = {
            "heartbeat_count": 1,
            "last_confirmed_heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "last_confirmed_expires_at": (
                datetime.now(timezone.utc) + timedelta(seconds=120)
            ).isoformat(),
        }
        poller.release = AsyncMock(return_value=True)
        poller.stop = AsyncMock()
        worker._poller = poller
        writer = MagicMock()
        writer.mark_completed = AsyncMock()
        writer.mark_requeued = AsyncMock()
        writer.mark_failed = AsyncMock()
        worker._writer = writer
        executor = MagicMock()
        executor.is_active.return_value = True
        executor.extend_deadline.return_value = None
        worker._mission_executor = executor

        with (
            patch.object(
                worker,
                "_execute_research_with_lease_watch",
                new=AsyncMock(
                    side_effect=ProcessTerminationError("child still alive")
                ),
            ),
            patch.object(worker, "_retry_pending_transports", new=AsyncMock()),
        ):
            with pytest.raises(ProcessTerminationError, match="child still alive"):
                await worker._handle_mission(sample_mission_row)

        writer.mark_completed.assert_not_awaited()
        writer.mark_requeued.assert_not_awaited()
        writer.mark_failed.assert_not_awaited()
        poller.release.assert_not_awaited()
        poller.stop.assert_awaited_once()
        assert worker._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_lease_loss_stops_cooperative_sync_work_before_terminal_write(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """A reclaimed row stops executor work at its next cancellation boundary."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        poller = MagicMock()
        poller.release = AsyncMock(return_value=False)
        worker._poller = poller
        writer = MagicMock()
        writer.mark_completed = AsyncMock(return_value=True)
        writer.mark_requeued = AsyncMock(return_value=True)
        writer.mark_failed = AsyncMock(return_value=True)
        worker._writer = writer
        research_started = threading.Event()
        research_stopped = threading.Event()

        async def _heartbeat(_row) -> None:
            while not research_started.is_set():
                await asyncio.sleep(0)
            raise LeaseLostError("lease reclaimed")

        def _sync_paid_runner() -> None:
            research_started.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    raise_if_execution_cancelled()
                except ExecutionCancelledError:
                    research_stopped.set()
                    raise
                time.sleep(0.001)
            raise AssertionError("executor thread did not observe lease cancellation")

        async def _research(*_args) -> None:
            await asyncio.to_thread(_sync_paid_runner)

        with (
            patch.object(worker, "_heartbeat_lease", side_effect=_heartbeat),
            patch.object(worker, "_execute_research", side_effect=_research),
            patch.object(
                worker,
                "_retry_pending_transports",
                new=AsyncMock(),
            ),
        ):
            with pytest.raises(LeaseLostError, match="lease reclaimed"):
                await worker._handle_mission(sample_mission_row)

        assert research_stopped.is_set()
        writer.mark_completed.assert_not_awaited()
        writer.mark_requeued.assert_not_awaited()
        writer.mark_failed.assert_not_awaited()
        poller.release.assert_awaited_once_with(
            sample_mission_row.id,
            sample_mission_row.lease,
        )

    def test_retained_log_backlog_is_process_bounded(
        self,
        mock_db_config,
        monkeypatch,
    ) -> None:
        """A sustained TraceLab outage cannot retain one queue per mission forever."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        monkeypatch.setattr("deepsearch.worker.main.MAX_PENDING_LOG_HANDLERS", 2)
        first, second, newest = MagicMock(), MagicMock(), MagicMock()
        first.get_telemetry.return_value = {"pending_entries": 10}

        worker._retain_log_handler(first)
        worker._retain_log_handler(second)
        worker._retain_log_handler(newest)

        assert worker._pending_log_handlers == [second, newest]
        assert worker._stats["log_backlog_overflowed_handlers"] == 1
        assert worker._stats["log_backlog_overflowed_entries"] == 10

    def test_completion_receipt_backlog_is_process_bounded(
        self,
        mock_db_config,
        monkeypatch,
    ) -> None:
        """Notifier outages retain only a bounded set of recent recovery hints."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        monkeypatch.setattr(
            "deepsearch.worker.main.MAX_PENDING_COMPLETION_RECEIPTS",
            2,
        )
        first = CompletionReceipt(job_id="job-1", mission_id="mission-1")
        second = CompletionReceipt(job_id="job-2", mission_id="mission-2")
        newest = CompletionReceipt(job_id="job-3", mission_id="mission-3")

        worker._retain_completion_receipt(first, True)
        worker._retain_completion_receipt(second, True)
        worker._retain_completion_receipt(newest, True)

        assert list(worker._pending_completion_receipts) == [second, newest]
        assert worker._stats["completion_receipt_backlog_overflowed"] == 1

    @pytest.mark.asyncio
    async def test_completion_receipt_retry_is_bounded_and_round_robin(
        self,
        mock_db_config,
        monkeypatch,
    ) -> None:
        """One mission fence cannot retry an outage backlog without a breadth cap."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        monkeypatch.setattr("deepsearch.worker.main.MAX_RECEIPT_RETRIES_PER_FENCE", 1)
        first = CompletionReceipt(job_id="job-1", mission_id="mission-1")
        second = CompletionReceipt(job_id="job-2", mission_id="mission-2")
        worker._pending_completion_receipts = {first: True, second: True}

        with patch.object(
            worker,
            "_deliver_completion_receipt",
            new=AsyncMock(return_value=False),
        ) as deliver:
            await worker._retry_pending_transports()

        deliver.assert_awaited_once_with(
            "mission-1",
            "job-1",
            completion_outcome=True,
            enqueue_on_failure=False,
        )
        assert list(worker._pending_completion_receipts) == [second, first]

    @pytest.mark.asyncio
    async def test_current_mission_receipt_is_not_retried_in_same_fence(
        self,
        mock_db_config,
    ) -> None:
        """The initial bounded delivery owns this fence; backlog retry starts later."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        current = CompletionReceipt(job_id="job-current", mission_id="mission-current")
        older = CompletionReceipt(job_id="job-older", mission_id="mission-older")
        worker._pending_completion_receipts = {current: True, older: True}

        with patch.object(
            worker,
            "_deliver_completion_receipt",
            new=AsyncMock(return_value=False),
        ) as deliver:
            await worker._retry_pending_transports(
                excluded_receipt_job_id="job-current"
            )

        deliver.assert_awaited_once_with(
            "mission-older",
            "job-older",
            completion_outcome=True,
            enqueue_on_failure=False,
        )

    @pytest.mark.asyncio
    async def test_unpersisted_terminal_state_releases_lease_for_recovery(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        worker = DeepSearchWorker(db_config=mock_db_config)
        poller = MagicMock()
        poller.heartbeat = AsyncMock(return_value=True)
        poller.release = AsyncMock(return_value=True)
        worker._poller = poller
        writer = MagicMock()
        writer.mark_failed = AsyncMock(side_effect=WriterError("write rejected"))
        worker._writer = writer

        with patch.object(
            worker,
            "_execute_research",
            new=AsyncMock(side_effect=RuntimeError("research crashed")),
        ):
            await worker._handle_mission(sample_mission_row)

        poller.release.assert_awaited_once_with(
            sample_mission_row.id,
            sample_mission_row.lease,
        )


class TestDeepSearchWorkerHandleMission:
    """Tests for mission handling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("receipt_success", "receipt_status", "receipt_retryable"),
        [
            (True, 200, False),
            (False, 503, True),
            (False, 401, False),
        ],
    )
    async def test_handle_mission_success(
        self,
        mock_db_config,
        sample_mission_row,
        receipt_success,
        receipt_status,
        receipt_retryable,
    ):
        """Persistence keeps completion metrics while receipt ACK stays separate."""

        receipt_client = MagicMock()
        receipt_client.deliver = AsyncMock(
            return_value=ReceiptDeliveryResult(
                success=receipt_success,
                attempts=1,
                retryable=receipt_retryable,
                final_status_code=receipt_status,
                error=(
                    None
                    if receipt_success
                    else f"TraceLab returned HTTP {receipt_status}"
                ),
            )
        )
        worker = DeepSearchWorker(
            db_config=mock_db_config,
            receipt_client=receipt_client,
        )
        dispatcher = MagicMock()
        worker._evidence_outbox_dispatcher = dispatcher

        # Mock dependencies
        mock_final_state = {
            "mission_id": sample_mission_row.id,
            "research_loop_count": 2,
            "sources_found": [{"url": "https://example.com"}],
            "findings": ["Finding 1"],
            "loop_count": 2,
            "total_tokens": 1000,
            "coverage_gates": {
                "passed": True,
                "failed_gates": [],
                "entity": {"passed": True, "covered": ["x"], "uncovered": []},
                "objective": {
                    "passed": True,
                    "per_objective_scores": {"objective-0": 0.8},
                    "uncovered_objective_ids": [],
                },
                "slot": {
                    "passed": True,
                    "total_slots": 1,
                    "filled_slots": 1,
                    "thin_slot_ids": [],
                    "unaddressed_slot_ids": [],
                },
            },
            "routing_reason": "converged_with_coverage",
            "mission_outcome": "complete",
        }

        mock_writer = AsyncMock()
        worker._writer = mock_writer

        with patch.object(worker, "_execute_research", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_final_state

            with patch("deepsearch.worker.main.generate_markdown_report") as mock_report:
                mock_report.return_value = "# Test Report"

                with patch("deepsearch.worker.main.generate_mission_protocol") as mock_protocol:
                    mock_proto_instance = MagicMock(spec=MissionProtocolComplete)
                    mock_proto_instance.model_dump.return_value = {"title": "Test"}
                    mock_protocol.return_value = mock_proto_instance

                    with patch("deepsearch.worker.main.build_execution_metadata") as mock_meta:
                        mock_meta.return_value = ExecutionMetadata(
                            total_tokens=0,
                            duration_seconds=0.0,
                        )

                        await worker._handle_mission(sample_mission_row)

        assert worker._stats["missions_processed"] == 1
        assert worker._stats["missions_completed"] == 1
        assert worker._stats["missions_failed"] == 0
        assert worker._stats["terminal_rows_persisted"] == 1
        assert worker._stats["completed_rows_persisted"] == 1
        mock_writer.mark_completed.assert_called_once()
        mock_writer.mark_failed.assert_not_called()
        dispatcher.wake.assert_called_once_with()
        persisted_job_id = mock_writer.mark_completed.call_args.kwargs["job_id"]
        assert persisted_job_id
        assert len(worker._pending_completion_receipts) == int(receipt_retryable)
        receipt_client.deliver.assert_awaited_once()
        delivered_receipt = receipt_client.deliver.await_args.args[0]
        assert delivered_receipt.job_id == persisted_job_id
        assert delivered_receipt.mission_id == sample_mission_row.id

    @pytest.mark.asyncio
    async def test_validation_failed_commit_wakes_durable_evidence_delivery(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """Reviewable failed artifacts enroll and wake only after persistence."""

        receipt_client = MagicMock()
        receipt_client.deliver = AsyncMock(
            return_value=ReceiptDeliveryResult(
                success=True,
                attempts=1,
                retryable=False,
                final_status_code=200,
            )
        )
        worker = DeepSearchWorker(
            db_config=mock_db_config,
            receipt_client=receipt_client,
        )
        writer = AsyncMock()
        worker._writer = writer
        dispatcher = MagicMock()
        worker._evidence_outbox_dispatcher = dispatcher
        final_state = {
            "mission_id": sample_mission_row.id,
            "synthesis_telemetry": {},
            "mission_outcome": MissionOutcome.VALIDATION_FAILED.value,
            "routing_reason": "reviewable validation failure",
        }
        protocol = MagicMock(spec=MissionProtocolComplete)
        protocol.model_dump.return_value = {"title": "Test"}

        with (
            patch.object(
                worker,
                "_execute_research",
                new=AsyncMock(return_value=final_state),
            ),
            patch(
                "deepsearch.worker.main.resolve_final_outcome_payload",
                return_value={
                    "outcome": MissionOutcome.VALIDATION_FAILED.value,
                    "routing_reason": "reviewable validation failure",
                },
            ),
            patch(
                "deepsearch.worker.main.generate_markdown_report",
                return_value="# Reviewable report",
            ),
            patch(
                "deepsearch.worker.main.generate_mission_protocol",
                return_value=protocol,
            ),
            patch(
                "deepsearch.worker.main.build_execution_metadata",
                return_value=ExecutionMetadata(
                    total_tokens=0,
                    duration_seconds=0.0,
                ),
            ),
        ):
            await worker._handle_mission(sample_mission_row)

        writer.mark_completed.assert_not_awaited()
        writer.mark_failed.assert_awaited_once()
        assert writer.mark_failed.await_args.kwargs["outcome"] is (
            MissionOutcome.VALIDATION_FAILED
        )
        dispatcher.wake.assert_called_once_with()
        receipt_client.deliver.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_mission_prefers_finalize_emitted_artifacts(
        self,
        mock_db_config,
        sample_mission_row,
        tmp_path,
    ) -> None:
        """Worker should upload the already-rendered finalize artifacts when present."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        report_path = tmp_path / "final_report.md"
        protocol_path = tmp_path / "final_protocol.json"
        report_path.write_text("# Finalized report\n", encoding="utf-8")
        protocol_path.write_text(json.dumps({"title": "Finalized protocol"}), encoding="utf-8")

        mock_final_state = {
            "mission_id": sample_mission_row.id,
            "research_loop_count": 2,
            "sources_found": [{"url": "https://example.com"}],
            "findings": ["Finding 1"],
            "loop_count": 2,
            "total_tokens": 1000,
            "report_path": str(report_path),
            "protocol_path": str(protocol_path),
            "coverage_gates": {
                "passed": True,
                "failed_gates": [],
                "entity": {"passed": True, "covered": ["x"], "uncovered": []},
                "objective": {
                    "passed": True,
                    "per_objective_scores": {"objective-0": 0.8},
                    "uncovered_objective_ids": [],
                },
                "slot": {
                    "passed": True,
                    "total_slots": 1,
                    "filled_slots": 1,
                    "thin_slot_ids": [],
                    "unaddressed_slot_ids": [],
                },
            },
            "routing_reason": "converged_with_coverage",
            "mission_outcome": "complete",
        }

        mock_writer = AsyncMock()
        worker._writer = mock_writer

        with patch.object(worker, "_execute_research", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_final_state

            with patch("deepsearch.worker.main.generate_markdown_report") as mock_report:
                with patch("deepsearch.worker.main.generate_mission_protocol") as mock_protocol:
                    with patch("deepsearch.worker.main.build_execution_metadata") as mock_meta:
                        mock_meta.return_value = ExecutionMetadata(
                            total_tokens=0,
                            duration_seconds=0.0,
                        )

                        await worker._handle_mission(sample_mission_row)

        mock_report.assert_not_called()
        mock_protocol.assert_not_called()
        mock_writer.mark_completed.assert_called_once()
        assert (
            mock_writer.mark_completed.call_args.kwargs["result_markdown"]
            == "# Finalized report\n"
        )
        assert mock_writer.mark_completed.call_args.kwargs["result_protocol"] == {
            "title": "Finalized protocol"
        }

    @pytest.mark.asyncio
    async def test_modern_run_falls_back_to_exact_inline_artifacts(
        self,
        mock_db_config,
        sample_mission_row,
        tmp_path,
    ) -> None:
        """Missing/corrupt files cannot destroy an exact finalize-emitted pair."""

        report = (
            "# Research Report: test-mission-001 - Test Research Mission\n\n"
            "Reviewable synthesized result.\n"
        )
        protocol = {
            "mission_id": sample_mission_row.mission_id,
            "citations": [],
            "title": sample_mission_row.title,
        }
        missing_report = tmp_path / "missing-report.md"
        corrupt_protocol = tmp_path / "corrupt-protocol.json"
        corrupt_protocol.write_text("{not-json", encoding="utf-8")
        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": MissionOutcome.COMPLETE.value,
            "routing_reason": "",
            "synthesis_used": True,
            "report_path": str(missing_report),
            "protocol_path": str(corrupt_protocol),
            "report_markdown": report,
            "protocol_payload": protocol,
            "synthesis_telemetry": {
                "result_identity": {
                    "algo": "sha256-utf8-nl",
                    "mission_id": sample_mission_row.mission_id,
                    "title": sample_mission_row.title,
                    "report_sha256": identity_sha256(report),
                    "report_len": len(report),
                    "span_reference_len": len(report),
                    "protocol_sha256": protocol_identity_sha256(protocol),
                }
            },
        }
        worker = DeepSearchWorker(db_config=mock_db_config)
        writer = AsyncMock()
        worker._writer = writer

        with patch.object(
            worker,
            "_execute_research",
            new=AsyncMock(return_value=final_state),
        ):
            await worker._handle_mission(sample_mission_row)

        writer.mark_failed.assert_not_awaited()
        writer.mark_completed.assert_awaited_once()
        assert writer.mark_completed.await_args.kwargs["result_markdown"] == report
        assert writer.mark_completed.await_args.kwargs["result_protocol"] == protocol

    @pytest.mark.asyncio
    async def test_invalid_utf8_staging_falls_back_to_exact_inline_pair(
        self,
        mock_db_config,
        sample_mission_row,
        tmp_path,
    ) -> None:
        """A decode failure is a staging miss, not loss of the inline pair."""

        report = (
            "# Research Report: test-mission-001 - Test Research Mission\n\n"
            "Reviewable synthesized result.\n"
        )
        protocol = {
            "mission_id": sample_mission_row.mission_id,
            "citations": [],
            "sources_collected": [],
        }
        report_path = tmp_path / "invalid-report.md"
        protocol_path = tmp_path / "invalid-protocol.json"
        report_path.write_bytes(b"\xff\xfe")
        protocol_path.write_bytes(b"\xff\xfe")
        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": MissionOutcome.COMPLETE.value,
            "routing_reason": "",
            "synthesis_used": True,
            "report_path": str(report_path),
            "protocol_path": str(protocol_path),
            "report_markdown": report,
            "protocol_payload": protocol,
            "synthesis_telemetry": {
                "result_identity": {
                    "algo": "sha256-utf8-nl",
                    "mission_id": sample_mission_row.mission_id,
                    "title": sample_mission_row.title,
                    "report_sha256": identity_sha256(report),
                    "report_len": len(report),
                    "span_reference_len": len(report),
                    "protocol_sha256": protocol_identity_sha256(protocol),
                }
            },
        }
        worker = DeepSearchWorker(db_config=mock_db_config)
        writer = AsyncMock()
        worker._writer = writer

        with patch.object(
            worker,
            "_execute_research",
            new=AsyncMock(return_value=final_state),
        ):
            await worker._handle_mission(sample_mission_row)

        writer.mark_failed.assert_not_awaited()
        writer.mark_completed.assert_awaited_once()
        assert writer.mark_completed.await_args.kwargs["result_markdown"] == report
        assert writer.mark_completed.await_args.kwargs["result_protocol"] == protocol

    @pytest.mark.asyncio
    async def test_worker_prefers_agreeing_file_artifacts(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """The inline pair is fallback evidence, not a new file replacement."""

        class FileReport(str):
            pass

        file_report = FileReport(
            "# Research Report: test-mission-001 - Test Research Mission\n\nBody.\n"
        )
        inline_report = str(file_report)
        file_protocol = {
            "mission_id": sample_mission_row.mission_id,
            "citations": [],
        }
        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": MissionOutcome.COMPLETE.value,
            "routing_reason": "",
            "synthesis_used": True,
            "report_path": "/attempt/report.md",
            "protocol_path": "/attempt/protocol.json",
            "report_markdown": inline_report,
            "protocol_payload": dict(file_protocol),
            "synthesis_telemetry": {},
        }
        worker = DeepSearchWorker(db_config=mock_db_config)
        writer = AsyncMock()
        worker._writer = writer

        with (
            patch.object(
                worker,
                "_execute_research",
                new=AsyncMock(return_value=final_state),
            ),
            patch(
                "deepsearch.worker.main._read_emitted_text_artifact",
                return_value=file_report,
            ),
            patch(
                "deepsearch.worker.main._read_emitted_json_artifact",
                return_value=file_protocol,
            ),
        ):
            await worker._handle_mission(sample_mission_row)

        writer.mark_completed.assert_awaited_once()
        assert writer.mark_completed.await_args.kwargs["result_markdown"] is file_report
        assert writer.mark_completed.await_args.kwargs["result_protocol"] is file_protocol

    @pytest.mark.asyncio
    @pytest.mark.parametrize("artifact", ["report", "protocol"])
    @pytest.mark.parametrize(
        "requested_outcome",
        [MissionOutcome.COMPLETE, MissionOutcome.VALIDATION_FAILED],
    )
    async def test_file_inline_divergence_blocks_without_content_diagnostic(
        self,
        mock_db_config,
        sample_mission_row,
        tmp_path,
        artifact,
        requested_outcome,
    ) -> None:
        report = (
            "# Research Report: test-mission-001 - Test Research Mission\n\nBody.\n"
        )
        protocol = {
            "mission_id": sample_mission_row.mission_id,
            "citations": [],
        }
        report_path = tmp_path / "report.md"
        protocol_path = tmp_path / "protocol.json"
        report_path.write_text(report, encoding="utf-8")
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        secret = "https://credential.example/private-token"
        inline_report = report if artifact == "protocol" else f"{report}{secret}\n"
        inline_protocol = (
            {**protocol, "summary": secret}
            if artifact == "protocol"
            else dict(protocol)
        )
        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": requested_outcome.value,
            "routing_reason": (
                "reviewable validation failure"
                if requested_outcome is MissionOutcome.VALIDATION_FAILED
                else ""
            ),
            "synthesis_used": True,
            "report_path": str(report_path),
            "protocol_path": str(protocol_path),
            "report_markdown": inline_report,
            "protocol_payload": inline_protocol,
            "synthesis_telemetry": {},
        }
        worker = DeepSearchWorker(db_config=mock_db_config)
        writer = AsyncMock()
        worker._writer = writer
        dispatcher = MagicMock()
        worker._evidence_outbox_dispatcher = dispatcher

        with patch.object(
            worker,
            "_execute_research",
            new=AsyncMock(return_value=final_state),
        ):
            await worker._handle_mission(sample_mission_row)

        writer.mark_completed.assert_not_awaited()
        writer.mark_failed.assert_awaited_once()
        kwargs = writer.mark_failed.await_args.kwargs
        assert kwargs["outcome"] is MissionOutcome.BLOCKED
        diagnostic = kwargs["execution_metadata"].identity_violation
        assert diagnostic["violations"] == [f"{artifact}_artifact_divergence"]
        assert secret not in json.dumps(diagnostic, sort_keys=True)
        dispatcher.wake.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_failed_inline_pair_enrolls_and_wakes_outbox(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        report = (
            "# Research Report: test-mission-001 - Test Research Mission\n\n"
            "Reviewable but flagged result.\n"
        )
        protocol = {
            "mission_id": sample_mission_row.mission_id,
            "citations": [],
        }
        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": MissionOutcome.VALIDATION_FAILED.value,
            "routing_reason": "reviewable validation failure",
            "synthesis_used": True,
            "report_path": None,
            "protocol_path": None,
            "report_markdown": report,
            "protocol_payload": protocol,
            "synthesis_telemetry": {
                "result_identity": {
                    "algo": "sha256-utf8-nl",
                    "mission_id": sample_mission_row.mission_id,
                    "title": sample_mission_row.title,
                    "report_sha256": identity_sha256(report),
                    "report_len": len(report),
                    "span_reference_len": len(report),
                    "protocol_sha256": protocol_identity_sha256(protocol),
                }
            },
        }
        worker = DeepSearchWorker(db_config=mock_db_config)
        writer = AsyncMock()
        worker._writer = writer
        dispatcher = MagicMock()
        worker._evidence_outbox_dispatcher = dispatcher

        with patch.object(
            worker,
            "_execute_research",
            new=AsyncMock(return_value=final_state),
        ):
            await worker._handle_mission(sample_mission_row)

        writer.mark_completed.assert_not_awaited()
        writer.mark_failed.assert_awaited_once()
        kwargs = writer.mark_failed.await_args.kwargs
        assert kwargs["outcome"] is MissionOutcome.VALIDATION_FAILED
        assert kwargs["result_markdown"] == report
        assert kwargs["result_protocol"] == protocol
        dispatcher.wake.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("protocol_state", ["missing", "corrupt"])
    async def test_modern_run_requires_exact_protocol_artifact(
        self,
        mock_db_config,
        sample_mission_row,
        tmp_path,
        protocol_state,
    ) -> None:
        """Missing/corrupt modern protocols block without outbox enrollment."""

        worker = DeepSearchWorker(db_config=mock_db_config)
        report_path = tmp_path / "final_report.md"
        report_path.write_text("# Finalized report\n", encoding="utf-8")
        protocol_path = tmp_path / "final_protocol.json"
        if protocol_state == "corrupt":
            protocol_path.write_text("{not-json", encoding="utf-8")

        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": MissionOutcome.COMPLETE.value,
            "routing_reason": "",
            "synthesis_used": True,
            "synthesis_telemetry": {},
            "report_path": str(report_path),
            "protocol_path": str(protocol_path),
        }
        writer = AsyncMock()
        worker._writer = writer
        dispatcher = MagicMock()
        worker._evidence_outbox_dispatcher = dispatcher

        with (
            patch.object(
                worker,
                "_execute_research",
                new=AsyncMock(return_value=final_state),
            ),
            patch("deepsearch.worker.main.generate_mission_protocol") as regenerate,
        ):
            await worker._handle_mission(sample_mission_row)

        regenerate.assert_not_called()
        writer.mark_completed.assert_not_awaited()
        writer.mark_failed.assert_awaited_once()
        assert "outcome" not in writer.mark_failed.await_args.kwargs
        assert writer.mark_failed.await_args.args[1] == (
            '{"category":"internal","exception_class":"RequiredArtifactError"}'
        )
        assert writer.mark_failed.await_args.kwargs["result_markdown"] == (
            "# Finalized report\n"
        )
        assert writer.mark_failed.await_args.kwargs["result_protocol"] is None
        dispatcher.wake.assert_not_called()

    @pytest.mark.asyncio
    async def test_modern_missing_report_preserves_protocol_forensics(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """The modern pair fence preserves a surviving protocol on BLOCKED."""

        protocol = {
            "mission_id": sample_mission_row.mission_id,
            "citations": [],
            "sources_collected": [],
        }
        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": MissionOutcome.COMPLETE.value,
            "routing_reason": "",
            "synthesis_used": True,
            "report_path": None,
            "protocol_path": None,
            "report_markdown": None,
            "protocol_payload": protocol,
            "synthesis_telemetry": {},
        }
        worker = DeepSearchWorker(db_config=mock_db_config)
        writer = AsyncMock()
        worker._writer = writer

        with patch.object(
            worker,
            "_execute_research",
            new=AsyncMock(return_value=final_state),
        ):
            await worker._handle_mission(sample_mission_row)

        writer.mark_completed.assert_not_awaited()
        writer.mark_failed.assert_awaited_once()
        assert writer.mark_failed.await_args.kwargs["result_markdown"] is None
        assert writer.mark_failed.await_args.kwargs["result_protocol"] == protocol

    @pytest.mark.asyncio
    async def test_blocked_modern_run_never_manufactures_legacy_artifacts(
        self,
        mock_db_config,
        sample_mission_row,
    ) -> None:
        """A real CodeAgent failure persists its reason, not a legacy husk."""

        final_state = {
            "mission_id": sample_mission_row.id,
            "mission_outcome": MissionOutcome.BLOCKED.value,
            "routing_reason": "codeagent_synthesis_failed",
            "synthesis_used": True,
            "report_path": None,
            "protocol_path": None,
            "report_markdown": None,
            "protocol_payload": None,
            "synthesis_telemetry": {
                "sources_collected": 232,
                "generation_failure_step": 21,
            },
        }
        worker = DeepSearchWorker(db_config=mock_db_config)
        writer = AsyncMock()
        worker._writer = writer

        with (
            patch.object(
                worker,
                "_execute_research",
                new=AsyncMock(return_value=final_state),
            ),
            patch("deepsearch.worker.main.generate_markdown_report") as legacy_report,
            patch("deepsearch.worker.main.generate_mission_protocol") as legacy_protocol,
        ):
            await worker._handle_mission(sample_mission_row)

        legacy_report.assert_not_called()
        legacy_protocol.assert_not_called()
        writer.mark_completed.assert_not_awaited()
        writer.mark_failed.assert_awaited_once()
        kwargs = writer.mark_failed.await_args.kwargs
        assert kwargs["outcome"] is MissionOutcome.BLOCKED
        assert kwargs["error_message"] == "codeagent_synthesis_failed"
        assert kwargs["result_markdown"] is None
        assert kwargs["result_protocol"] is None

    @pytest.mark.asyncio
    async def test_handle_mission_conversion_error(self, mock_db_config, sample_mission_row):
        """Test handling of conversion errors."""
        worker = DeepSearchWorker(db_config=mock_db_config)
        mock_writer = AsyncMock()
        worker._writer = mock_writer
        dispatcher = MagicMock()
        worker._evidence_outbox_dispatcher = dispatcher

        with patch("deepsearch.worker.main.convert_row_to_state") as mock_convert:
            mock_convert.side_effect = ConversionError("Invalid mission")

            await worker._handle_mission(sample_mission_row)

        assert worker._stats["missions_failed"] == 1
        mock_writer.mark_failed.assert_called_once()
        dispatcher.wake.assert_not_called()
        call_args = mock_writer.mark_failed.call_args
        assert call_args[0][1] == (
            '{"category":"conversion","exception_class":"ConversionError"}'
        )
        assert call_args.kwargs["execution_metadata"].lease_attempt[
            "terminal_reset"
        ] == {
            "deepsearch_lease_token": "cleared_atomically",
            "deepsearch_lease_expires_at": "cleared_atomically",
        }

    @pytest.mark.asyncio
    async def test_handle_mission_execution_error(self, mock_db_config, sample_mission_row):
        """Test handling of execution errors."""
        worker = DeepSearchWorker(db_config=mock_db_config)
        mock_writer = AsyncMock()
        worker._writer = mock_writer

        with patch("deepsearch.worker.main.convert_row_to_state") as mock_convert:
            mock_convert.return_value = {"mission_id": sample_mission_row.id}

            with patch.object(worker, "_execute_research", new_callable=AsyncMock) as mock_exec:
                mock_exec.side_effect = RuntimeError("Research failed")

                await worker._handle_mission(sample_mission_row)

        assert worker._stats["missions_failed"] == 1
        mock_writer.mark_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_mission_sets_current_ids(self, mock_db_config, sample_mission_row):
        """Test that current mission/job IDs are set during processing."""
        worker = DeepSearchWorker(db_config=mock_db_config)
        mock_writer = AsyncMock()
        worker._writer = mock_writer

        captured_mission_id = None
        captured_job_id = None

        async def capture_ids(*args, **kwargs):
            nonlocal captured_mission_id, captured_job_id
            captured_mission_id = worker._current_mission_id
            captured_job_id = worker._current_job_id
            return {"mission_id": sample_mission_row.id}

        with patch("deepsearch.worker.main.convert_row_to_state", return_value={}):
            with patch.object(worker, "_execute_research", side_effect=capture_ids):
                with patch("deepsearch.worker.main.generate_markdown_report", return_value=""):
                    with patch("deepsearch.worker.main.generate_mission_protocol") as mock_p:
                        mock_p.return_value = MagicMock(
                            spec=MissionProtocolComplete,
                            model_dump=MagicMock(return_value={}),
                        )
                        with patch("deepsearch.worker.main.build_execution_metadata") as mock_m:
                            mock_m.return_value = ExecutionMetadata(
                                total_tokens=0,
                                duration_seconds=0.0,
                            )
                            await worker._handle_mission(sample_mission_row)

        assert captured_mission_id == sample_mission_row.id
        assert captured_job_id is not None
        # After completion, IDs should be cleared
        assert worker._current_mission_id is None
        assert worker._current_job_id is None


class TestDeepSearchWorkerExecuteResearch:
    """Tests for research execution."""

    @pytest.mark.asyncio
    async def test_execute_research_configures_environment(self, mock_db_config, monkeypatch):
        """Test that LLM environment is configured correctly."""
        monkeypatch.delenv("WORKER_DISABLE_CHECKPOINTING", raising=False)
        worker = DeepSearchWorker(
            db_config=mock_db_config,
            llm_backend="gemini",
            model="gemini-2.5-pro",
        )

        state = {
            "mission_id": "test-001",
            "mission_objectives": ["Test"],
        }

        with patch("deepsearch.worker.main.DeepSearchSettings") as mock_settings:
            mock_settings.load.return_value = _mock_settings_obj()

            with patch("deepsearch.worker.main.create_checkpointer") as mock_cp:
                # Mock the async context manager
                mock_cp_instance = AsyncMock()
                mock_cp_instance.__aenter__ = AsyncMock(return_value=None)
                mock_cp_instance.__aexit__ = AsyncMock(return_value=None)
                mock_cp.return_value = mock_cp_instance

                with patch("deepsearch.worker.main.agent_graph") as mock_graph:
                    mock_workflow = MagicMock()
                    mock_workflow.ainvoke = AsyncMock(return_value=state)
                    mock_graph.build_workflow.return_value = mock_workflow

                    with patch("deepsearch.worker.main.TelemetryCollector"):
                        with patch("deepsearch.worker.main.TelemetryWriter"):
                            with patch("deepsearch.worker.main.record_finding_records_from_state"):
                                with patch("deepsearch.worker.main.extract_sources_from_state"):
                                    await worker._execute_research(
                                        state, "test-001", "job-001"
                                    )

        assert os.environ.get("LLM_BACKEND") == "gemini"
        assert os.environ.get("GEMINI_MODEL") == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_execute_research_scopes_thread_id_per_job(
        self,
        mock_db_config,
        monkeypatch,
    ) -> None:
        """Worker executions should use a fresh thread ID for each claimed job."""
        monkeypatch.delenv("WORKER_DISABLE_CHECKPOINTING", raising=False)

        worker = DeepSearchWorker(
            db_config=mock_db_config,
            llm_backend="gemini",
            model="gemini-2.5-pro",
        )
        state = {
            "mission_id": "test-001",
            "mission_objectives": ["Test"],
        }

        thread_ids: list[str] = []

        def _thread_id(
            mission_id: str,
            *,
            execution_token: str | None = None,
        ) -> str:
            suffix = execution_token or "base"
            return f"thread-{mission_id}-{suffix}"

        settings = _mock_settings_obj()
        settings.generate_thread_id = MagicMock(side_effect=_thread_id)

        def fake_build_workflow(*, checkpointer=None):
            workflow = MagicMock()

            async def ainvoke(_state, config=None):
                thread_ids.append(config["configurable"]["thread_id"])
                return dict(_state)

            workflow.ainvoke = AsyncMock(side_effect=ainvoke)
            return workflow

        with patch("deepsearch.worker.main.DeepSearchSettings") as mock_settings:
            mock_settings.load.return_value = settings

            with patch("deepsearch.worker.main.create_checkpointer") as mock_cp:
                mock_cp_instance = AsyncMock()
                mock_cp_instance.__aenter__ = AsyncMock(return_value=None)
                mock_cp_instance.__aexit__ = AsyncMock(return_value=None)
                mock_cp.return_value = mock_cp_instance

                with patch("deepsearch.worker.main.agent_graph") as mock_graph:
                    mock_graph.build_workflow.side_effect = fake_build_workflow

                    with patch("deepsearch.worker.main.TelemetryCollector"):
                        with patch("deepsearch.worker.main.TelemetryWriter"):
                            with patch("deepsearch.worker.main.record_finding_records_from_state"):
                                with patch("deepsearch.worker.main.extract_sources_from_state"):
                                    await worker._execute_research(
                                        dict(state),
                                        "test-001",
                                        "job-001",
                                    )
                                    await worker._execute_research(
                                        dict(state),
                                        "test-001",
                                        "job-002",
                                    )

        assert thread_ids == [
            "thread-test-001-job-001",
            "thread-test-001-job-002",
        ]
        assert thread_ids[0] != thread_ids[1]
        assert settings.generate_thread_id.call_args_list[0].kwargs == {
            "execution_token": "job-001"
        }
        assert settings.generate_thread_id.call_args_list[1].kwargs == {
            "execution_token": "job-002"
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("backend", "model", "env_var"),
        [
            ("deepseek", "deepseek-chat", "DEEPSEEK_MODEL"),
            ("minimax", "MiniMax-M2.7", "MINIMAX_MODEL"),
        ],
    )
    async def test_execute_research_configures_openai_compatible_environment(
        self,
        mock_db_config,
        monkeypatch,
        backend: str,
        model: str,
        env_var: str,
    ) -> None:
        """OpenAI-compatible backends should export the matching model env var."""
        monkeypatch.delenv("WORKER_DISABLE_CHECKPOINTING", raising=False)
        worker = DeepSearchWorker(
            db_config=mock_db_config,
            llm_backend=backend,
            model=model,
        )

        state = {
            "mission_id": "test-001",
            "mission_objectives": ["Test"],
        }

        with patch("deepsearch.worker.main.DeepSearchSettings") as mock_settings:
            mock_settings.load.return_value = _mock_settings_obj()

            with patch("deepsearch.worker.main.create_checkpointer") as mock_cp:
                mock_cp_instance = AsyncMock()
                mock_cp_instance.__aenter__ = AsyncMock(return_value=None)
                mock_cp_instance.__aexit__ = AsyncMock(return_value=None)
                mock_cp.return_value = mock_cp_instance

                with patch("deepsearch.worker.main.agent_graph") as mock_graph:
                    mock_workflow = MagicMock()
                    mock_workflow.ainvoke = AsyncMock(return_value=state)
                    mock_graph.build_workflow.return_value = mock_workflow

                    with patch("deepsearch.worker.main.TelemetryCollector"):
                        with patch("deepsearch.worker.main.TelemetryWriter"):
                            with patch("deepsearch.worker.main.record_finding_records_from_state"):
                                with patch("deepsearch.worker.main.extract_sources_from_state"):
                                    await worker._execute_research(
                                        state, "test-001", "job-001"
                                    )

        assert os.environ.get("LLM_BACKEND") == backend
        assert os.environ.get(env_var) == model

    @pytest.mark.asyncio
    async def test_execute_research_persists_reflection_history_in_telemetry(
        self,
        mock_db_config,
        monkeypatch,
    ) -> None:
        """Worker telemetry persistence should include reflection history from final state."""
        monkeypatch.delenv("WORKER_DISABLE_CHECKPOINTING", raising=False)

        worker = DeepSearchWorker(
            db_config=mock_db_config,
            llm_backend="gemini",
            model="gemini-2.5-pro",
        )
        state = {
            "mission_id": "test-001",
            "mission_objectives": ["Test"],
        }
        final_state = {
            "mission_id": "test-001",
            "mission_objectives": ["Test"],
            "telemetry": {
                "coverage_gate_trace": [
                    {"loop": 1, "passed": False, "failed_gates": ["slot"]}
                ],
            },
            "objective_score_history": [
                {
                    "loop": 1,
                    "scores": [{"objective_id": "objective-0", "score": 0.62}],
                }
            ],
            "convergence_history": [0.55, 0.62],
        }

        mock_collector = MagicMock()
        mock_collector.finalize.return_value = MagicMock()

        with patch("deepsearch.worker.main.DeepSearchSettings") as mock_settings:
            mock_settings.load.return_value = _mock_settings_obj()

            with patch("deepsearch.worker.main.create_checkpointer") as mock_cp:
                mock_cp_instance = AsyncMock()
                mock_cp_instance.__aenter__ = AsyncMock(return_value=None)
                mock_cp_instance.__aexit__ = AsyncMock(return_value=None)
                mock_cp.return_value = mock_cp_instance

                with patch("deepsearch.worker.main.agent_graph") as mock_graph:
                    mock_workflow = MagicMock()
                    mock_workflow.ainvoke = AsyncMock(return_value=final_state)
                    mock_graph.build_workflow.return_value = mock_workflow

                    with patch(
                        "deepsearch.worker.main.TelemetryCollector",
                        return_value=mock_collector,
                    ):
                        with patch("deepsearch.worker.main.TelemetryWriter") as mock_writer_cls:
                            mock_writer = mock_writer_cls.return_value
                            mock_writer.write.return_value = "/tmp/mission-test-001.jsonl"

                            with patch("deepsearch.worker.main.record_finding_records_from_state"):
                                with patch("deepsearch.worker.main.extract_sources_from_state"):
                                    result = await worker._execute_research(
                                        state,
                                        "test-001",
                                        "job-001",
                                    )

        merge_payload = mock_collector.merge_state_telemetry.call_args.args[0]

        assert merge_payload["coverage_gate_trace"][0]["failed_gates"] == ["slot"]
        assert merge_payload["objective_score_history"][0]["scores"][0]["score"] == 0.62
        assert merge_payload["convergence_history"] == [0.55, 0.62]
        assert result["telemetry"]["telemetry_artifact_path"] == "/tmp/mission-test-001.jsonl"

    @pytest.mark.asyncio
    async def test_execute_research_continues_without_checkpointer_on_setup_failure(
        self,
        mock_db_config,
        monkeypatch,
    ):
        """Worker should continue even when checkpointer setup fails."""
        monkeypatch.delenv("WORKER_DISABLE_CHECKPOINTING", raising=False)

        worker = DeepSearchWorker(db_config=mock_db_config)
        state = {
            "mission_id": "test-001",
            "mission_objectives": ["Test"],
        }

        with patch("deepsearch.worker.main.DeepSearchSettings") as mock_settings:
            mock_settings.load.return_value = _mock_settings_obj()

            with patch("deepsearch.worker.main.create_checkpointer") as mock_cp:
                mock_cp_instance = AsyncMock()
                mock_cp_instance.__aenter__ = AsyncMock(
                    side_effect=AttributeError("'Connection' object has no attribute 'is_alive'")
                )
                mock_cp_instance.__aexit__ = AsyncMock(return_value=None)
                mock_cp.return_value = mock_cp_instance

                with patch("deepsearch.worker.main.agent_graph") as mock_graph:
                    mock_workflow = MagicMock()
                    mock_workflow.ainvoke = AsyncMock(return_value=state)
                    mock_graph.build_workflow.return_value = mock_workflow

                    with patch("deepsearch.worker.main.TelemetryCollector"):
                        with patch("deepsearch.worker.main.TelemetryWriter"):
                            with patch("deepsearch.worker.main.record_finding_records_from_state"):
                                with patch("deepsearch.worker.main.extract_sources_from_state"):
                                    result = await worker._execute_research(
                                        state, "test-001", "job-001"
                                    )

        assert result == state
        mock_graph.build_workflow.assert_called_once_with(checkpointer=None)

    @pytest.mark.asyncio
    async def test_execute_research_never_retries_after_is_alive_execution_error(
        self,
        mock_db_config,
        monkeypatch,
    ):
        """A paid invocation cannot be restarted after retrieval audit begins.

        The first graph call may already have made source-fetch/liveness calls.
        A retry from the original state would omit those attempts from the
        terminal ledger, so an in-execution compatibility error must fail loud.
        """
        monkeypatch.delenv("WORKER_DISABLE_CHECKPOINTING", raising=False)

        worker = DeepSearchWorker(db_config=mock_db_config)
        state = {
            "mission_id": "test-001",
            "mission_objectives": ["Test"],
        }

        with patch("deepsearch.worker.main.DeepSearchSettings") as mock_settings:
            mock_settings.load.return_value = _mock_settings_obj()

            with patch("deepsearch.worker.main.create_checkpointer") as mock_cp:
                mock_cp_instance = AsyncMock()
                mock_cp_instance.__aenter__ = AsyncMock(return_value=object())
                mock_cp_instance.__aexit__ = AsyncMock(return_value=None)
                mock_cp.return_value = mock_cp_instance

                captured_checkpointers = []
                retrieval_attempts = []

                def fake_build_workflow(*, checkpointer=None):
                    captured_checkpointers.append(checkpointer)
                    workflow = MagicMock()
                    async def fail_after_retrieval(_state, config=None):
                        retrieval_attempts.extend(
                            [
                                {
                                    "tool": "source_fetch",
                                    "status": "error",
                                },
                                {
                                    "tool": "url_liveness",
                                    "status": "error",
                                },
                            ]
                        )
                        raise AttributeError(
                            "'Connection' object has no attribute 'is_alive'"
                        )

                    workflow.ainvoke = AsyncMock(side_effect=fail_after_retrieval)
                    return workflow

                with patch("deepsearch.worker.main.agent_graph") as mock_graph:
                    mock_graph.build_workflow.side_effect = fake_build_workflow

                    with patch("deepsearch.worker.main.TelemetryCollector"):
                        with patch("deepsearch.worker.main.TelemetryWriter"):
                            with patch("deepsearch.worker.main.record_finding_records_from_state"):
                                with patch("deepsearch.worker.main.extract_sources_from_state"):
                                    with pytest.raises(AttributeError, match="is_alive"):
                                        await worker._execute_research(
                                            state, "test-001", "job-001"
                                        )

        assert retrieval_attempts == [
            {"tool": "source_fetch", "status": "error"},
            {"tool": "url_liveness", "status": "error"},
        ]
        assert len(captured_checkpointers) == 1
        assert captured_checkpointers[0] is not None
        mock_cp_instance.__aexit__.assert_awaited_once()


class TestSignalHandling:
    """Tests for signal handler registration."""

    def test_register_signal_handlers(self, mock_db_config):
        """Test signal handlers are registered."""
        worker = DeepSearchWorker(db_config=mock_db_config)

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop_instance = MagicMock()
            mock_loop.return_value = mock_loop_instance

            worker._register_signal_handlers()

            # Verify signal handlers were registered
            calls = mock_loop_instance.add_signal_handler.call_args_list
            signals_registered = [call[0][0] for call in calls]

            assert signal.SIGTERM in signals_registered
            assert signal.SIGINT in signals_registered


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_json_formatter_basic(self):
        """Test basic JSON log formatting."""
        from deepsearch.worker.__main__ import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_json_formatter_with_exception(self):
        """Exception info is projected without its message or traceback."""
        from deepsearch.worker.__main__ import JSONFormatter

        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "Unhandled exception"
        assert parsed["category"] == "validation"
        assert parsed["exception_class"] == "ValueError"
        assert "Test error" not in result
        assert "Traceback" not in result


class TestTextFormatter:
    """Tests for text log formatter."""

    def test_text_formatter_format(self):
        """Test text log formatting."""
        from deepsearch.worker.__main__ import TextFormatter

        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "[INFO]" in result
        assert "test.logger" in result
        assert "Test message" in result


class TestConfigureLogging:
    """Tests for logging configuration."""

    def test_configure_logging_json_format(self, monkeypatch):
        """Test JSON logging configuration."""
        from deepsearch.worker.__main__ import JSONFormatter, configure_logging

        monkeypatch.setenv("WORKER_LOG_FORMAT", "json")
        monkeypatch.setenv("WORKER_LOG_LEVEL", "DEBUG")

        configure_logging()

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_configure_logging_text_format(self, monkeypatch):
        """Test text logging configuration."""
        from deepsearch.worker.__main__ import TextFormatter, configure_logging

        monkeypatch.setenv("WORKER_LOG_FORMAT", "text")
        monkeypatch.setenv("WORKER_LOG_LEVEL", "INFO")

        configure_logging()

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, TextFormatter)


class TestMainEntryPoint:
    """Tests for main() entry point."""

    @pytest.mark.asyncio
    async def test_main_requires_database_url(self, monkeypatch):
        """Test that main fails without DATABASE_URL."""
        from deepsearch.worker.__main__ import main

        monkeypatch.delenv("DATABASE_URL", raising=False)

        result = await main()

        assert result == 1

    @pytest.mark.asyncio
    async def test_main_runs_worker(self, monkeypatch):
        """Test that main runs the worker."""
        from deepsearch.worker.__main__ import main

        monkeypatch.setenv("DATABASE_URL", "postgresql://test@localhost/db")

        with patch("deepsearch.worker.main.DeepSearchWorker") as MockWorker:
            mock_instance = MockWorker.return_value
            mock_instance.run = AsyncMock()

            result = await main()

        assert result == 0
        mock_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_handles_keyboard_interrupt(self, monkeypatch):
        """Test graceful handling of KeyboardInterrupt."""
        from deepsearch.worker.__main__ import main

        monkeypatch.setenv("DATABASE_URL", "postgresql://test@localhost/db")

        with patch("deepsearch.worker.main.DeepSearchWorker") as MockWorker:
            mock_instance = MockWorker.return_value
            mock_instance.run = AsyncMock(side_effect=KeyboardInterrupt())

            result = await main()

        assert result == 0

    @pytest.mark.asyncio
    async def test_main_handles_exception(self, monkeypatch):
        """Test error handling in main."""
        from deepsearch.worker.__main__ import main

        monkeypatch.setenv("DATABASE_URL", "postgresql://test@localhost/db")

        with patch("deepsearch.worker.main.DeepSearchWorker") as MockWorker:
            mock_instance = MockWorker.return_value
            mock_instance.run = AsyncMock(side_effect=RuntimeError("Fatal error"))

            result = await main()

        assert result == 1
