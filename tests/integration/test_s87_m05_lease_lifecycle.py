"""S87 M05 durable lease lifecycle through the real poller API."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from deepsearch.worker.execution import MissionProcessExecutor
from deepsearch.worker.lease import ExecutionFencePolicy, LeasePolicy
from deepsearch.worker.main import LeaseLostError
from deepsearch.worker.poller import MissionPoller, MissionRow

_PROJECT_ID = "11111111-2222-4333-8444-555555555555"
_POLICY = LeasePolicy(
    lease_seconds=120,
    heartbeat_seconds=30,
    max_attempts=2,
    reclaim_grace_seconds=30,
)


class _WithholdingState:
    def __init__(self) -> None:
        self.request_started = threading.Event()
        self.request_finished = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.active_requests = 0
        self.max_active_requests = 0
        self.events: list[str] = []

    def start_request(self) -> None:
        with self.lock:
            self.active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests,
                self.active_requests,
            )
            self.events.append("attempt_a_request_started")
        self.request_started.set()

    def finish_request(self) -> None:
        with self.lock:
            self.active_requests -= 1
            self.events.append("attempt_a_request_finished")
        self.request_finished.set()

    def record_second_claim(self) -> None:
        with self.lock:
            self.events.append("attempt_b_claimed")


class _WithholdingHandler(BaseHTTPRequestHandler):
    server: "_WithholdingServer"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.server.state.start_request()
        self.connection.settimeout(0.1)
        try:
            while not self.server.state.release.is_set():
                try:
                    probe = self.connection.recv(1, socket.MSG_PEEK)
                except socket.timeout:
                    continue
                except (ConnectionResetError, OSError):
                    break
                if probe == b"":
                    break
        finally:
            self.server.state.finish_request()

    def log_message(self, _format: str, *args: object) -> None:
        del args


class _WithholdingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: _WithholdingState) -> None:
        super().__init__(("127.0.0.1", 0), _WithholdingHandler)
        self.state = state


def _run_blocked_codeagent_request(api_base: str) -> None:
    """Enter the real smolagents/OpenAI transport, then write a sentinel."""

    from deepsearch.agent.codeagent_runner import (
        build_default_model,
        run_codeagent_synthesis,
    )

    model = build_default_model(
        api_key="unpaid-loopback-key",
        request_timeout_s=60.0,
    )
    # Redirect only this test client; production credentials cannot target proxies.
    model.client.base_url = api_base
    run_codeagent_synthesis(
        system_instructions="Return a one-sentence answer.",
        user_directive="Say hello without using tools.",
        model=model,
        max_steps=1,
    )
    sandbox = Path(os.environ["DEEPSEARCH_REPORT_DIR"]).parent
    (sandbox / "post-provider-sentinel").write_text("unsafe", encoding="utf-8")


class _Transaction(AbstractAsyncContextManager[None]):
    def __init__(self, database: "_LeaseDatabase") -> None:
        self.database = database

    async def __aenter__(self) -> None:
        await self.database.lock.acquire()

    async def __aexit__(self, *args: object) -> None:
        self.database.lock.release()


class _Connection:
    def __init__(self, database: "_LeaseDatabase") -> None:
        self.database = database

    def transaction(self) -> _Transaction:
        return _Transaction(self.database)

    async def execute(
        self,
        query: str,
        max_attempts: int,
        reclaim_grace_seconds: int,
    ) -> str:
        assert query == MissionPoller.DEAD_LETTER_EXHAUSTED_SQL
        row = self.database.row
        if (
            row["project_id"] is not None
            and row["status"] in {"queued", "in_progress"}
            and (
                row["status"] == "queued"
                or self.database.reclaimable(reclaim_grace_seconds)
            )
            and int(row["deepsearch_attempt_count"] or 0) >= max_attempts
        ):
            row["status"] = "blocked"
            row["error_message"] = "DeepSearch lease attempts exhausted"
            row["deepsearch_lease_token"] = None
            row["deepsearch_lease_expires_at"] = None
        return "UPDATE 1"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        assert query == MissionPoller.CLAIM_MISSION_SQL
        owner, token, lease_seconds, max_attempts, reclaim_grace_seconds = args
        row = self.database.row
        eligible = row["status"] == "queued" or (
            row["status"] == "in_progress"
            and self.database.reclaimable(reclaim_grace_seconds)
        )
        if (
            row["project_id"] is None
            or not eligible
            or int(row["deepsearch_attempt_count"] or 0) >= max_attempts
        ):
            return None
        row["status"] = "in_progress"
        row["started_at"] = row.get("started_at") or self.database.now
        row["deepsearch_lease_owner"] = owner
        row["deepsearch_lease_token"] = token
        row["deepsearch_leased_at"] = self.database.now
        row["deepsearch_heartbeat_at"] = self.database.now
        row["deepsearch_lease_expires_at"] = self.database.now + timedelta(
            seconds=lease_seconds
        )
        row["deepsearch_attempt_count"] = (
            int(row["deepsearch_attempt_count"] or 0) + 1
        )
        return dict(row)


class _LeaseDatabase:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.now = datetime(2026, 7, 18, 4, 0, tzinfo=timezone.utc)
        self.row: dict[str, Any] = {
            "id": "mission-uuid",
            "project_id": _PROJECT_ID,
            "mission_id": "S87-LIFECYCLE",
            "title": "Lease lifecycle fixture",
            "objective": "Prove one durable owner",
            "context": {},
            "status": "queued",
            "queued_at": self.now,
            "success_criteria": [],
            "deliverables": [],
            "research_phases": {},
            "required_entities": None,
            "excluded_entities": None,
            "deepsearch_lease_owner": None,
            "deepsearch_lease_token": None,
            "deepsearch_leased_at": None,
            "deepsearch_heartbeat_at": None,
            "deepsearch_lease_expires_at": None,
            "deepsearch_attempt_count": 0,
        }

    @property
    def expired(self) -> bool:
        expires_at = self.row["deepsearch_lease_expires_at"]
        return expires_at is None or expires_at <= self.now

    def reclaimable(self, grace_seconds: int) -> bool:
        if self.row["deepsearch_lease_token"] is None:
            return True
        expires_at = self.row["deepsearch_lease_expires_at"]
        return bool(
            expires_at is not None
            and expires_at <= self.now - timedelta(seconds=grace_seconds)
        )

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)

    async def acquire(self) -> _Connection:
        return _Connection(self)

    async def release(self, connection: _Connection) -> None:
        del connection

    async def fetchval(self, query: str, *args: Any) -> str | None:
        async with self.lock:
            mission_id, owner, token, attempt_count = args[:4]
            row = self.row
            owned = (
                row["id"] == mission_id
                and row["status"] == "in_progress"
                and row["deepsearch_lease_owner"] == owner
                and row["deepsearch_lease_token"] == token
                and row["deepsearch_attempt_count"] == attempt_count
            )
            if query == MissionPoller.HEARTBEAT_SQL:
                if not owned or self.expired:
                    return None
                lease_seconds = args[4]
                row["deepsearch_heartbeat_at"] = self.now
                row["deepsearch_lease_expires_at"] = self.now + timedelta(
                    seconds=lease_seconds
                )
                return str(row["id"])
            assert query == MissionPoller.RELEASE_SQL
            if not owned:
                return None
            row["deepsearch_lease_owner"] = None
            row["deepsearch_lease_token"] = None
            row["deepsearch_leased_at"] = None
            row["deepsearch_heartbeat_at"] = None
            row["deepsearch_lease_expires_at"] = None
            return str(row["id"])


def _poller(database: _LeaseDatabase, owner: str) -> MissionPoller:
    return MissionPoller(
        database,  # type: ignore[arg-type]
        lease_owner=owner,
        lease_policy=_POLICY,
        enable_jitter=False,
    )


@pytest.mark.asyncio
async def test_concurrent_workers_have_exactly_one_claim_winner() -> None:
    database = _LeaseDatabase()
    first, second = await asyncio.gather(
        _poller(database, "worker-a").poll_once(),
        _poller(database, "worker-b").poll_once(),
    )

    winners = [row for row in (first, second) if isinstance(row, MissionRow)]
    assert len(winners) == 1
    assert winners[0].lease.owner in {"worker-a", "worker-b"}
    assert winners[0].lease.attempt_count == 1
    assert database.row["status"] == "in_progress"


@pytest.mark.asyncio
async def test_active_lease_cannot_be_stolen_and_heartbeat_extends_it() -> None:
    database = _LeaseDatabase()
    owner = _poller(database, "worker-a")
    contender = _poller(database, "worker-b")
    claimed = await owner.poll_once()
    assert claimed is not None
    first_expiry = database.row["deepsearch_lease_expires_at"]

    database.advance(30)
    assert await contender.poll_once() is None
    assert await owner.heartbeat(claimed.id, claimed.lease) is True
    assert database.row["deepsearch_lease_expires_at"] > first_expiry


@pytest.mark.asyncio
async def test_crashed_claim_recovers_after_expiry_and_rejects_old_owner() -> None:
    database = _LeaseDatabase()
    first_poller = _poller(database, "worker-a")
    first = await first_poller.poll_once()
    assert first is not None

    # Simulate a crash: no heartbeat and no graceful release.
    database.advance(121)
    assert await _poller(database, "worker-b").poll_once() is None
    database.advance(29)
    second = await _poller(database, "worker-b").poll_once()

    assert second is not None
    assert second.lease.owner == "worker-b"
    assert second.lease.token != first.lease.token
    assert second.lease.attempt_count == 2
    assert await first_poller.heartbeat(first.id, first.lease) is False


@pytest.mark.asyncio
async def test_tokened_row_with_null_expiry_fails_closed() -> None:
    """Malformed active ownership cannot bypass the lease-duration invariant."""

    database = _LeaseDatabase()
    first = await _poller(database, "worker-a").poll_once()
    assert first is not None
    database.row["deepsearch_lease_expires_at"] = None
    database.advance(10_000)

    assert await _poller(database, "worker-b").poll_once() is None
    database.row["deepsearch_attempt_count"] = _POLICY.max_attempts
    assert await _poller(database, "worker-c").poll_once() is None
    assert database.row["status"] == "in_progress"
    assert database.row["deepsearch_lease_token"] == first.lease.token


@pytest.mark.asyncio
async def test_graceful_release_allows_immediate_recovery() -> None:
    database = _LeaseDatabase()
    first_poller = _poller(database, "worker-a")
    first = await first_poller.poll_once()
    assert first is not None

    assert await first_poller.release(first.id, first.lease) is True
    recovered = await _poller(database, "worker-b").poll_once()

    assert recovered is not None
    assert recovered.lease.attempt_count == 2
    assert recovered.lease.owner == "worker-b"


@pytest.mark.asyncio
async def test_exhausted_attempt_is_dead_lettered_and_never_replayed() -> None:
    database = _LeaseDatabase()
    first_poller = _poller(database, "worker-a")
    first = await first_poller.poll_once()
    assert first is not None
    assert await first_poller.release(first.id, first.lease) is True
    second_poller = _poller(database, "worker-b")
    second = await second_poller.poll_once()
    assert second is not None
    assert await second_poller.release(second.id, second.lease) is True

    assert await _poller(database, "worker-c").poll_once() is None
    assert database.row["status"] == "blocked"
    assert database.row["error_message"] == "DeepSearch lease attempts exhausted"
    assert await _poller(database, "worker-d").poll_once() is None


@pytest.mark.asyncio
async def test_completed_row_is_never_replayed_even_after_lease_expiry() -> None:
    database = _LeaseDatabase()
    claimed = await _poller(database, "worker-a").poll_once()
    assert claimed is not None
    database.row["status"] = "completed"
    database.advance(600)

    assert await _poller(database, "worker-b").poll_once() is None
    assert database.row["deepsearch_attempt_count"] == 1


@pytest.mark.asyncio
async def test_any_project_is_claimable() -> None:
    """Missions are authored across every TraceLab project.

    Access is scoped by RBAC, not by the worker. A worker that only claims one
    project silently strands every mission authored elsewhere - no claim, no
    error, no log line. That outage is invisible, which is what makes it bad.
    """
    database = _LeaseDatabase()
    database.row["project_id"] = "99999999-8888-4777-8666-555555555555"

    claimed = await _poller(database, "worker-a").poll_once()

    assert claimed is not None
    assert database.row["status"] == "in_progress"


@pytest.mark.asyncio
async def test_row_without_a_project_is_never_claimed() -> None:
    """The result is written back onto this row, so it must carry a project.

    Claiming a project-less row would strand its report outside every project.
    """
    database = _LeaseDatabase()
    database.row["project_id"] = None

    assert await _poller(database, "worker-a").poll_once() is None
    assert database.row["status"] == "queued"
    assert database.row["deepsearch_attempt_count"] == 0


@pytest.mark.asyncio
async def test_blocked_provider_is_dead_before_reclaim_grace_allows_attempt_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """A real blocked transport must close before another lease can be claimed."""

    # The real CodeAgent tool bundle requires a configured Linkup key even
    # though this loopback test never invokes the tool or reaches the network.
    monkeypatch.setenv("LINKUP_API_KEY", "unpaid-loopback-key")
    state = _WithholdingState()
    server = _WithholdingServer(state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    def close_server() -> None:
        state.release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)
        assert not server_thread.is_alive()

    request.addfinalizer(close_server)
    database = _LeaseDatabase()
    first_poller = _poller(database, "worker-a")
    first = await first_poller.poll_once()
    assert first is not None

    fence = ExecutionFencePolicy(
        heartbeat_timeout_seconds=1.0,
        cancellation_grace_seconds=0.1,
        terminate_grace_seconds=1.0,
        kill_grace_seconds=1.0,
        safety_margin_seconds=1.0,
        provider_limits={
            "contract": {
                "backend": "deepseek",
                "request_timeout_seconds": 60.0,
                "client_max_retries": 0,
                "configured_max_retries": 3,
                "wrapper_max_retries": 2,
                "wrapper_attempts": 3,
                "hard_bound_mode": "spawn-process-v1",
            },
            "codeagent": {
                "backend": "deepseek-openai-compatible",
                "request_timeout_seconds": 60.0,
                "client_max_retries": 0,
                "smolagents_rate_limit_retry": False,
                "synthesis_attempts": 3,
                "max_step_fallback_attempts": 1,
                "recovery_passes": 1,
                "hard_bound_mode": "spawn-process-v1",
            },
            "tools": {
                "configured_request_timeout_seconds": 60.0,
                "configured_max_retries": 3,
                "linkup_request_timeout_seconds": 60.0,
                "linkup_max_retries": 3,
                "source_fetch_request_timeout_seconds": 15.0,
                "source_fetch_max_redirects": 5,
                "url_liveness_request_timeout_seconds": 5.0,
                "url_liveness_max_requests": 3,
                "scholar_backend_request_timeout_seconds": 20.0,
                "scholar_max_backends": 2,
                "hard_bound_mode": "spawn-process-v1",
            },
        },
    )
    executor = MissionProcessExecutor(fence, attempt_root=tmp_path)
    lose_lease = asyncio.Event()

    async def heartbeat() -> None:
        await lose_lease.wait()
        if not await first_poller.heartbeat(first.id, first.lease):
            raise LeaseLostError("attempt A lease expired")

    heartbeat_task = asyncio.create_task(heartbeat())
    job_id = "production-shaped-overlap"
    api_base = f"http://127.0.0.1:{server.server_port}/v1"
    execution_task = asyncio.create_task(
        executor.execute(
            _run_blocked_codeagent_request,
            (api_base,),
            job_id=job_id,
            hard_deadline_monotonic=time.monotonic() + 90.0,
            heartbeat_task=heartbeat_task,
        )
    )

    try:
        request_started_task = asyncio.create_task(
            asyncio.to_thread(state.request_started.wait, 60.0)
        )
        started_or_exited, _ = await asyncio.wait(
            {request_started_task, execution_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if execution_task in started_or_exited:
            await execution_task
        assert await request_started_task
        child_pid = executor.child_pid(job_id)
        sandbox = executor.sandbox_path(job_id)
        assert child_pid is not None
        assert sandbox is not None

        # Expiry alone is not reclaim eligibility: attempt A still has the
        # entire grace window in which to prove local provider termination.
        database.advance(121)
        assert await _poller(database, "worker-b").poll_once() is None

        lose_lease.set()
        with pytest.raises(LeaseLostError, match="attempt A lease expired"):
            await asyncio.wait_for(
                execution_task,
                timeout=float(fence.shutdown_bound_seconds or 0.0) + 3.0,
            )

        assert executor.is_active(job_id) is False
        assert await asyncio.to_thread(state.request_finished.wait, 5.0)
        assert state.active_requests == 0
        assert state.max_active_requests == 1
        assert not (sandbox / "post-provider-sentinel").exists()

        database.advance(29)
        second = await _poller(database, "worker-b").poll_once()
        assert second is not None
        state.record_second_claim()
        assert second.lease.attempt_count == 2
        assert state.events.index("attempt_a_request_finished") < state.events.index(
            "attempt_b_claimed"
        )
    finally:
        if executor.is_active(job_id):
            await executor.stop(job_id)
        for task in (execution_task, heartbeat_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(execution_task, heartbeat_task, return_exceptions=True)
        executor.cleanup(job_id)
