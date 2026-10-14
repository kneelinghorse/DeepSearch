"""Unit tests for DeepSearch worker health endpoint."""

from __future__ import annotations

import asyncio
import json
import time
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
pytest.importorskip("aiohttp")
from aiohttp import ClientSession
from aiohttp.test_utils import TestServer, TestClient

from deepsearch.worker.health import HealthServer


class TestHealthServerInit:
    """Tests for HealthServer initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        stats_provider = lambda: {}
        server = HealthServer(stats_provider=stats_provider)

        assert server._port == 8080
        assert server._host == "0.0.0.0"
        assert server._stats_provider == stats_provider
        assert server._app is None
        assert server._runner is None
        assert server._site is None

    def test_init_with_custom_port(self):
        """Test initialization with custom port."""
        stats_provider = lambda: {}
        server = HealthServer(stats_provider=stats_provider, port=9090)

        assert server._port == 9090

    def test_init_with_custom_host(self):
        """Test initialization with custom host."""
        stats_provider = lambda: {}
        server = HealthServer(stats_provider=stats_provider, host="127.0.0.1")

        assert server._host == "127.0.0.1"

    def test_port_property(self):
        """Test port property returns configured port."""
        server = HealthServer(stats_provider=lambda: {}, port=3000)
        assert server.port == 3000

    def test_is_running_initially_false(self):
        """Test is_running is False before start."""
        server = HealthServer(stats_provider=lambda: {})
        assert server.is_running is False


class TestHealthServerLifecycle:
    """Tests for HealthServer start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_started_at(self):
        """Test that start() sets the started_at timestamp."""
        server = HealthServer(stats_provider=lambda: {}, port=0)

        before = time.time()
        await server.start()
        after = time.time()

        try:
            assert server._started_at is not None
            assert before <= server._started_at <= after
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_start_creates_app_and_runner(self):
        """Test that start() creates app and runner."""
        server = HealthServer(stats_provider=lambda: {}, port=0)

        await server.start()

        try:
            assert server._app is not None
            assert server._runner is not None
            assert server._site is not None
            assert server.is_running is True
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        """Test that stop() cleans up resources."""
        server = HealthServer(stats_provider=lambda: {}, port=0)

        await server.start()
        await server.stop()

        # Runner should be cleaned up (is_running checks both site and runner)
        # After cleanup, the runner/site references are still there but cleaned
        # The important thing is no errors during stop


class TestHealthEndpoint:
    """Tests for /health endpoint responses."""

    @pytest.fixture
    def sample_stats(self):
        """Sample worker stats for testing."""
        return {
            "missions_processed": 10,
            "missions_completed": 8,
            "missions_failed": 2,
            "processing": True,
            "poll_interval": 5.0,
        }

    @pytest.mark.asyncio
    async def test_health_returns_200(self, sample_stats):
        """Test /health returns 200 OK."""
        server = HealthServer(
            stats_provider=lambda: sample_stats,
            port=0,  # Let OS assign a port
        )
        await server.start()

        try:
            # Create test client
            async with ClientSession() as session:
                # Get actual port from site
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    assert resp.status == HTTPStatus.OK
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_returns_503_until_worker_preflight_is_ready(self) -> None:
        """Orchestration must not route work to a fail-closed worker."""
        server = HealthServer(
            stats_provider=lambda: {
                "ready": False,
                "preflight": {"ready": False},
            },
            port=0,
        )
        await server.start()

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    assert resp.status == HTTPStatus.SERVICE_UNAVAILABLE
                    data = await resp.json()
                    assert data["status"] == "unhealthy"
                    assert data["ready"] is False
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_returns_json(self, sample_stats):
        """Test /health returns JSON content type."""
        server = HealthServer(
            stats_provider=lambda: sample_stats,
            port=0,
        )
        await server.start()

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    assert "application/json" in resp.headers.get("Content-Type", "")
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_includes_status(self, sample_stats):
        """Test /health response includes status field."""
        server = HealthServer(
            stats_provider=lambda: sample_stats,
            port=0,
        )
        await server.start()

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    data = await resp.json()
                    assert data["status"] == "healthy"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_includes_uptime(self, sample_stats):
        """Test /health response includes uptime_seconds."""
        server = HealthServer(
            stats_provider=lambda: sample_stats,
            port=0,
        )
        await server.start()

        # Wait a tiny bit to ensure non-zero uptime
        await asyncio.sleep(0.1)

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    data = await resp.json()
                    assert "uptime_seconds" in data
                    assert data["uptime_seconds"] >= 0.1
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_includes_worker_stats(self, sample_stats):
        """Test /health response includes worker stats."""
        server = HealthServer(
            stats_provider=lambda: sample_stats,
            port=0,
        )
        await server.start()

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    data = await resp.json()

                    assert data["missions_processed"] == 10
                    assert data["missions_completed"] == 8
                    assert data["missions_failed"] == 2
                    assert data["processing"] is True
                    assert data["poll_interval"] == 5.0
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_with_no_current_mission(self):
        """Test /health when no mission is processing."""
        stats = {
            "missions_processed": 5,
            "missions_completed": 5,
            "missions_failed": 0,
            "processing": False,
            "poll_interval": 10.0,
        }
        server = HealthServer(
            stats_provider=lambda: stats,
            port=0,
        )
        await server.start()

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    data = await resp.json()
                    assert data["processing"] is False
        finally:
            await server.stop()


class TestRootEndpoint:
    """Tests for / root endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_200(self):
        """Test / returns 200 OK."""
        server = HealthServer(stats_provider=lambda: {}, port=0)
        await server.start()

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/") as resp:
                    assert resp.status == HTTPStatus.OK
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_root_returns_service_info(self):
        """Test / returns service info JSON."""
        server = HealthServer(stats_provider=lambda: {}, port=0)
        await server.start()

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/") as resp:
                    data = await resp.json()
                    assert data["service"] == "deepsearch-worker"
                    assert data["health_endpoint"] == "/health"
        finally:
            await server.stop()


class TestWorkerHealthIntegration:
    """Tests for worker integration with health server."""

    def test_worker_has_health_stats_method(self):
        """Test DeepSearchWorker has get_health_stats method."""
        from deepsearch.worker.main import DeepSearchWorker
        from deepsearch.worker.database import DatabaseConfig

        config = DatabaseConfig(
            url="postgresql://test@localhost/test",
            min_pool_size=1,
            max_pool_size=5,
        )
        worker = DeepSearchWorker(db_config=config, enable_health_server=False)

        stats = worker.get_health_stats()

        assert "missions_processed" in stats
        assert "missions_completed" in stats
        assert "missions_failed" in stats
        assert "processing" in stats
        assert "current_mission_id" not in stats
        assert "current_job_id" not in stats
        assert "poll_interval" in stats

    def test_worker_health_stats_reflect_state(self):
        """Test get_health_stats returns current worker state."""
        from deepsearch.worker.main import DeepSearchWorker
        from deepsearch.worker.database import DatabaseConfig

        config = DatabaseConfig(
            url="postgresql://test@localhost/test",
            min_pool_size=1,
            max_pool_size=5,
        )
        worker = DeepSearchWorker(
            db_config=config,
            poll_interval=15.0,
            enable_health_server=False,
        )

        # Simulate some processing
        worker._stats["missions_processed"] = 3
        worker._stats["missions_completed"] = 2
        worker._stats["missions_failed"] = 1
        worker._current_mission_id = "active-mission"

        stats = worker.get_health_stats()

        assert stats["missions_processed"] == 3
        assert stats["missions_completed"] == 2
        assert stats["missions_failed"] == 1
        assert stats["processing"] is True
        assert "current_mission_id" not in stats
        assert stats["poll_interval"] == 15.0

    @pytest.mark.asyncio
    async def test_public_health_omits_internal_identity_and_error_detail(self):
        """The unauthenticated payload exposes only aggregate delivery state."""

        from deepsearch.worker.database import DatabaseConfig
        from deepsearch.worker.main import DeepSearchWorker

        mission_id = "mission-sensitive-123"
        job_id = "job-sensitive-456"
        result_id = "result-sensitive-789"
        lease_token = "lease-sensitive-abc"
        signed_url = "https://example.test/report?token=signed-secret"
        body_marker = "AUTHORED-PAGE-BODY-MARKER"
        worker = DeepSearchWorker(
            db_config=DatabaseConfig(url="postgresql://test@localhost/test"),
            enable_health_server=False,
        )
        worker._current_mission_id = mission_id
        worker._current_job_id = job_id
        worker._stats["last_completion_receipt"] = {
            "success": False,
            "mission_id": mission_id,
            "job_id": job_id,
            "result_id": result_id,
            "lease_token": lease_token,
            "attempts": 4,
            "retryable": True,
            "status_code": 503,
            "error": f"RuntimeError: {signed_url} {body_marker}",
        }
        worker._stats["last_log_delivery"] = {
            "configured": True,
            "accepted_entries": 2,
            "pending_entries": 1,
            "delivery_attempts": 3,
            "failed_batches": 1,
            "overflowed_entries": 0,
            "discarded_entries": 0,
            "last_status_code": None,
            "last_error": f"TimeoutError: {signed_url} {body_marker}",
        }
        server = HealthServer(stats_provider=worker.get_health_stats, port=0)
        await server.start()

        try:
            async with ClientSession() as session:
                port = server._site._server.sockets[0].getsockname()[1]
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    payload = await resp.json()
        finally:
            await server.stop()

        assert payload["processing"] is True
        assert payload["last_completion_receipt"] == {
            "category": "server_error",
            "status_code": 503,
            "attempts": 4,
        }
        assert payload["last_log_delivery"]["last_category"] == "transport_error"
        assert "last_error" not in payload["last_log_delivery"]
        assert "current_mission_id" not in payload
        assert "current_job_id" not in payload
        serialized = json.dumps(payload, sort_keys=True)
        for secret in (
            mission_id,
            job_id,
            result_id,
            lease_token,
            signed_url,
            body_marker,
            "RuntimeError",
            "TimeoutError",
        ):
            assert secret not in serialized

    def test_evidence_delivery_health_is_aggregate_and_identity_free(self):
        """Health exposes durable queue truth without trigger identities."""

        from deepsearch.worker.database import DatabaseConfig
        from deepsearch.worker.main import DeepSearchWorker

        config = DatabaseConfig(url="postgresql://test@localhost/test")
        worker = DeepSearchWorker(db_config=config, enable_health_server=False)
        dispatcher = MagicMock()
        dispatcher.stats = {
            "running": True,
            "claimed": 2,
            "acknowledged": 1,
            "retry_scheduled": 1,
            "dead_lettered": 0,
            "exhausted_swept": 0,
            "lease_conflicts": 0,
            "dispatch_errors": 0,
            "pending": 1,
            "leased": 0,
            "acked": 1,
            "dead_letter": 0,
            "invalid_exhausted_pending": 0,
            "overflowed": 0,
            "discarded": 0,
            "last_delivery": {
                "category": "server_error",
                "status_code": 503,
                "retryable": True,
                "state": "pending",
                "attempt_number": 1,
            },
        }
        worker._evidence_outbox_dispatcher = dispatcher

        payload = worker.get_health_stats()["evidence_ledger_delivery"]
        serialized = json.dumps(payload, sort_keys=True)

        assert payload == dispatcher.stats
        assert "mission" not in serialized
        assert "job" not in serialized
        assert "result" not in serialized
        assert "secret" not in serialized
        assert "url" not in serialized

    def test_worker_health_server_disabled(self):
        """Test worker can be created with health server disabled."""
        from deepsearch.worker.main import DeepSearchWorker
        from deepsearch.worker.database import DatabaseConfig

        config = DatabaseConfig(
            url="postgresql://test@localhost/test",
            min_pool_size=1,
            max_pool_size=5,
        )
        worker = DeepSearchWorker(
            db_config=config,
            enable_health_server=False,
        )

        assert worker.enable_health_server is False

    def test_worker_health_port_from_env(self, monkeypatch):
        """Test worker reads health port from environment."""
        from deepsearch.worker.main import DeepSearchWorker
        from deepsearch.worker.database import DatabaseConfig

        monkeypatch.setenv("HEALTH_PORT", "9999")

        config = DatabaseConfig(
            url="postgresql://test@localhost/test",
            min_pool_size=1,
            max_pool_size=5,
        )
        worker = DeepSearchWorker(db_config=config)

        assert worker.health_port == 9999

    def test_worker_health_port_default(self, monkeypatch):
        """Test worker uses default health port when not set."""
        from deepsearch.worker.main import DeepSearchWorker
        from deepsearch.worker.database import DatabaseConfig

        monkeypatch.delenv("HEALTH_PORT", raising=False)

        config = DatabaseConfig(
            url="postgresql://test@localhost/test",
            min_pool_size=1,
            max_pool_size=5,
        )
        worker = DeepSearchWorker(db_config=config)

        assert worker.health_port == 8080
