"""S91 m07 hard execution-fence and startup arithmetic invariants."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.execution_control import (
    ExecutionCancelledError,
    bind_execution_cancel_event,
)
from deepsearch.mission import contract as contract_module
from deepsearch.report.generator import save_report
from deepsearch.worker.database import DatabaseConfig
from deepsearch.worker.execution import IsolatedExecutionError, MissionProcessExecutor
from deepsearch.worker.lease import ExecutionFencePolicy, LeasePolicy
from deepsearch.worker.main import DeepSearchWorker


def _provider_limits() -> dict[str, dict[str, object]]:
    return {
        "contract": {
            "backend": "deepseek",
            "request_timeout_seconds": 300.0,
            "client_max_retries": 0,
            "configured_max_retries": 3,
            "wrapper_max_retries": 2,
            "wrapper_attempts": 3,
            "hard_bound_mode": "spawn-process-v1",
        },
        "codeagent": {
            "backend": "deepseek-openai-compatible",
            "request_timeout_seconds": 300.0,
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
    }


def _policy() -> ExecutionFencePolicy:
    return ExecutionFencePolicy(
        heartbeat_timeout_seconds=10.0,
        cancellation_grace_seconds=0.1,
        terminate_grace_seconds=0.5,
        kill_grace_seconds=0.5,
        safety_margin_seconds=10.0,
        provider_limits=_provider_limits(),
    )


async def _write_report_in_child() -> dict[str, str]:
    """Mirror the production async spawn target while emitting one artifact."""

    path = save_report("# Isolated\n", "S91-M07", mission_uuid="12345678")
    return {"report_path": path}


async def _return_inline_artifacts_in_child() -> dict[str, object]:
    """Return realistic inline artifacts through the spawned-process pipe."""

    report = "# Inline report\n\n" + ("report-body " * 20_000)
    protocol = {
        "mission_id": "S91-M07-INLINE",
        "citations": [],
        "sources_collected": [
            {
                "url": "https://example.test/source",
                "body": "source-body " * 40_000,
            }
        ],
    }
    return {
        "report_markdown": report,
        "protocol_payload": protocol,
    }


def _block_without_polling_cancellation() -> None:
    sandbox = Path(os.environ["DEEPSEARCH_REPORT_DIR"]).parent
    (sandbox / "request-started").write_text("started", encoding="utf-8")
    time.sleep(30.0)
    (sandbox / "post-loss-side-effect").write_text("unsafe", encoding="utf-8")


def _ignore_sigterm_and_block() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sandbox = Path(os.environ["DEEPSEARCH_REPORT_DIR"]).parent
    (sandbox / "request-started").write_text("started", encoding="utf-8")
    time.sleep(30.0)


def _write_forbidden_sentinel(path_text: str) -> None:
    Path(path_text).write_text("unsafe", encoding="utf-8")


async def _wait_for_child_file(
    executor: MissionProcessExecutor,
    job_id: str,
    name: str,
    *,
    timeout: float = 60.0,
) -> Path:
    """Wait generously for a cold spawned interpreter, not for shutdown."""

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        sandbox = executor.sandbox_path(job_id)
        if sandbox is not None and (sandbox / name).exists():
            return sandbox
        await asyncio.sleep(0.01)
    raise AssertionError(f"isolated child did not create {name}")


def test_safe_defaults_publish_exact_overlap_equation() -> None:
    lease = LeasePolicy(120, 30, 3, reclaim_grace_seconds=30)

    snapshot = _policy().evaluate(lease)

    assert snapshot["equation"] == "H + D + U + M < L + G"
    assert snapshot["required_seconds"] == 51.1
    assert snapshot["available_seconds"] == 150.0
    assert snapshot["slack_seconds"] == pytest.approx(98.9)
    assert snapshot["safe"] is True
    assert snapshot["provider_limits"]["codeagent"]["client_max_retries"] == 0


def test_overlap_equation_fails_closed_at_the_sql_equality_edge() -> None:
    lease = LeasePolicy(30, 10, 3, reclaim_grace_seconds=0)
    policy = ExecutionFencePolicy(
        heartbeat_timeout_seconds=5.0,
        cancellation_grace_seconds=5.0,
        terminate_grace_seconds=5.0,
        kill_grace_seconds=0.1,
        safety_margin_seconds=4.9,
        provider_limits=_provider_limits(),
    )

    snapshot = policy.evaluate(lease)

    assert snapshot["required_seconds"] == 30.0
    assert snapshot["available_seconds"] == 30.0
    assert snapshot["safe"] is False
    assert snapshot["violations"] == [
        "heartbeat detection plus process shutdown and safety margin must be "
        "strictly less than lease duration plus reclaim grace"
    ]


def test_subsecond_safety_margin_fails_closed_before_deadline_rounding() -> None:
    policy = ExecutionFencePolicy(
        heartbeat_timeout_seconds=10.0,
        cancellation_grace_seconds=1.0,
        terminate_grace_seconds=1.0,
        kill_grace_seconds=1.0,
        safety_margin_seconds=5e-324,
        provider_limits=_provider_limits(),
    )

    snapshot = policy.evaluate(LeasePolicy(120, 30, 3, reclaim_grace_seconds=30))

    assert snapshot["safe"] is False
    assert snapshot["required_seconds"] is None
    assert snapshot["violations"] == [
        "WORKER_LEASE_SAFETY_MARGIN_SECONDS must be a finite number >= 1"
    ]


@pytest.mark.parametrize(
    "lease",
    [
        LeasePolicy(float("nan"), 30, 3),  # type: ignore[arg-type]
        LeasePolicy(120, float("nan"), 3),  # type: ignore[arg-type]
        LeasePolicy(120, 30, 3.0),  # type: ignore[arg-type]
    ],
)
def test_noninteger_lease_overrides_fail_closed(lease: LeasePolicy) -> None:
    snapshot = _policy().evaluate(lease)

    assert lease.valid is False
    assert snapshot["safe"] is False
    assert snapshot["required_seconds"] is None


def test_incomplete_or_unknown_provider_snapshot_fails_closed() -> None:
    incomplete = ExecutionFencePolicy(
        heartbeat_timeout_seconds=10.0,
        cancellation_grace_seconds=1.0,
        terminate_grace_seconds=1.0,
        kill_grace_seconds=1.0,
        safety_margin_seconds=1.0,
        provider_limits={"anything": {"unbounded": float("nan")}},
    )

    snapshot = incomplete.evaluate(LeasePolicy(120, 30, 3))

    assert snapshot["safe"] is False
    assert any("missing provider limit groups" in item for item in snapshot["violations"])
    assert any("unknown provider limit groups" in item for item in snapshot["violations"])


def test_worker_provider_snapshot_is_derived_from_runtime_constants() -> None:
    worker = DeepSearchWorker(
        db_config=DatabaseConfig(url="postgresql://test@localhost/deepsearch"),
        llm_backend="deepseek",
        enable_health_server=False,
    )

    limits = worker.execution_fence_policy.provider_limits
    assert set(limits) == {"contract", "codeagent", "tools"}
    assert limits["contract"]["backend"] == "deepseek"
    assert limits["contract"]["client_max_retries"] == 0
    assert limits["codeagent"]["request_timeout_seconds"] == (
        runner.DEFAULT_REQUEST_TIMEOUT_S
    )
    assert limits["codeagent"]["client_max_retries"] == 0
    assert limits["codeagent"]["smolagents_rate_limit_retry"] is False
    assert all(
        group["hard_bound_mode"] == "spawn-process-v1"
        for group in limits.values()
    )


def test_contract_compilation_cannot_persist_after_fence_loss(
    tmp_path: Path,
) -> None:
    """Enrichment returning after lease loss is stopped before contract I/O."""

    cancel_event = threading.Event()

    def compile_then_lose(*args, **kwargs):
        del args, kwargs
        cancel_event.set()
        return object()

    with bind_execution_cancel_event(cancel_event), patch.object(
        contract_module,
        "compile_contract_from_state",
        side_effect=compile_then_lose,
    ), patch.object(contract_module, "persist_contract") as persist:
        with pytest.raises(ExecutionCancelledError):
            contract_module.prepare_execution_contract(
                {},
                origin="tracelab",
                output_dir=tmp_path,
            )

    persist.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_lease_loss_after_max_step_run_preempts_scratchpad_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No DeepSearch fallback may start after a stale CodeAgent run returns."""

    cancel_event = threading.Event()
    monkeypatch.setenv("LINKUP_API_KEY", "unpaid-test-key")

    class FakeCodeAgent:
        def __init__(self, **_kwargs: object) -> None:
            class ActionStep:
                is_final_answer = False
                token_usage = None
                timing = None
                tool_calls = None

            self.memory = SimpleNamespace(
                steps=[ActionStep() for _ in range(runner.DEFAULT_MAX_STEPS)]
            )

        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            cancel_event.set()
            return SimpleNamespace(output="", token_usage=None)

    fake_smolagents = ModuleType("smolagents")
    fake_smolagents.CodeAgent = FakeCodeAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "smolagents", fake_smolagents)
    monkeypatch.setattr(
        runner,
        "build_codeagent_tools",
        lambda **_kwargs: SimpleNamespace(
            tools=[],
            scratchpad={"report_draft": "Draft [cite:https://example.test]"},
        ),
    )

    with patch.object(
        runner,
        "_recover_stranded_draft",
        wraps=runner._recover_stranded_draft,
    ) as recovery:
        with pytest.raises(ExecutionCancelledError):
            runner.run_codeagent_synthesis(
                system_instructions="sys",
                user_directive="do it",
                model=object(),
                cancel_event=cancel_event,
            )

    recovery.assert_not_called()


@pytest.mark.asyncio
async def test_successful_child_artifacts_are_attempt_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "inherited-output"
    for name in (
        "DEEPSEARCH_DATA_ROOT", "DEEPSEARCH_REPORT_DIR", "DEEPSEARCH_TELEMETRY_DIR",
        "DEEPSEARCH_CONTRACT_DIR", "CHECKPOINT_DB_PATH",
    ):
        monkeypatch.setenv(name, str(outside))
    executor = MissionProcessExecutor(_policy(), attempt_root=tmp_path)
    job_id = "successful-attempt"
    try:
        result = await executor.execute(
            _write_report_in_child,
            (),
            job_id=job_id,
            hard_deadline_monotonic=time.monotonic() + 60.0,
        )

        sandbox = executor.sandbox_path(job_id)
        assert sandbox is not None
        assert Path(result["report_path"]).resolve().is_relative_to(sandbox)
        assert not executor.is_active(job_id)
        assert not outside.exists()
    finally:
        if executor.is_active(job_id):
            await executor.stop(job_id)
        executor.cleanup(job_id)
    assert not sandbox.exists()


@pytest.mark.asyncio
async def test_inline_artifacts_survive_spawn_process_serialization(
    tmp_path: Path,
) -> None:
    """The fallback pair crosses the same process fence as production state."""

    executor = MissionProcessExecutor(_policy(), attempt_root=tmp_path)
    job_id = "inline-artifact-attempt"
    try:
        result = await executor.execute(
            _return_inline_artifacts_in_child,
            (),
            job_id=job_id,
            hard_deadline_monotonic=time.monotonic() + 60.0,
        )

        expected = await _return_inline_artifacts_in_child()
        assert result["report_markdown"] == expected["report_markdown"]
        assert result["protocol_payload"] == expected["protocol_payload"]
    finally:
        if executor.is_active(job_id):
            await executor.stop(job_id)
        executor.cleanup(job_id)


@pytest.mark.asyncio
async def test_noncooperative_child_is_dead_before_control_returns(
    tmp_path: Path,
) -> None:
    """A blocked call that never polls the event cannot outlive lease loss."""

    executor = MissionProcessExecutor(_policy(), attempt_root=tmp_path)
    job_id = "blocked-attempt"
    lose_lease = asyncio.Event()

    async def heartbeat() -> None:
        await lose_lease.wait()
        raise RuntimeError("lease ownership lost")

    heartbeat_task = asyncio.create_task(heartbeat())
    execution_task = asyncio.create_task(
        executor.execute(
            _block_without_polling_cancellation,
            (),
            job_id=job_id,
            hard_deadline_monotonic=time.monotonic() + 60.0,
            heartbeat_task=heartbeat_task,
        )
    )

    try:
        sandbox = await _wait_for_child_file(executor, job_id, "request-started")
        child_pid = executor.child_pid(job_id)

        lose_lease.set()
        with pytest.raises(RuntimeError, match="lease ownership lost"):
            await asyncio.wait_for(
                execution_task,
                timeout=float(_policy().shutdown_bound_seconds or 0.0) + 2.0,
            )

        assert child_pid is not None
        assert executor.is_active(job_id) is False
        assert not (sandbox / "post-loss-side-effect").exists()
    finally:
        if not execution_task.done():
            execution_task.cancel()
        await asyncio.gather(execution_task, heartbeat_task, return_exceptions=True)
        if executor.is_active(job_id):
            await executor.stop(job_id)
        executor.cleanup(job_id)


@pytest.mark.asyncio
async def test_child_self_deadline_does_not_depend_on_parent_event_loop(
    tmp_path: Path,
) -> None:
    """Deadline refresh is honored and expiry kills during a parent-loop stall."""

    executor = MissionProcessExecutor(_policy(), attempt_root=tmp_path)
    job_id = "self-deadline"
    execution_task = asyncio.create_task(
        executor.execute(
            _block_without_polling_cancellation,
            (),
            job_id=job_id,
            hard_deadline_monotonic=time.monotonic() + 60.0,
        )
    )
    try:
        sandbox = await _wait_for_child_file(executor, job_id, "request-started")

        original_deadline = time.monotonic() + 0.5
        executor.extend_deadline(job_id, original_deadline)
        extended_deadline = time.monotonic() + 1.5
        executor.extend_deadline(job_id, extended_deadline)

        # Deliberately block the parent's asyncio loop past the first deadline.
        # The child must honor the refresh, remain alive, then self-kill without
        # relying on this loop to schedule a cancellation callback.
        time.sleep(0.8)
        assert executor.is_active(job_id) is True
        time.sleep(1.0)

        with pytest.raises(IsolatedExecutionError, match="without a terminal result"):
            await asyncio.wait_for(
                execution_task,
                timeout=3.0,
            )

        assert executor.is_active(job_id) is False
        assert not (sandbox / "post-loss-side-effect").exists()
    finally:
        if not execution_task.done():
            execution_task.cancel()
        await asyncio.gather(execution_task, return_exceptions=True)
        if executor.is_active(job_id):
            await executor.stop(job_id)
        executor.cleanup(job_id)


@pytest.mark.asyncio
async def test_expired_deadline_never_enters_target(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "expired-deadline-target"
    executor = MissionProcessExecutor(_policy(), attempt_root=tmp_path)
    job_id = "expired-deadline"

    try:
        with pytest.raises(IsolatedExecutionError, match="without a terminal result"):
            await executor.execute(
                _write_forbidden_sentinel,
                (str(sentinel),),
                job_id=job_id,
                hard_deadline_monotonic=time.monotonic() - 1.0,
            )

        assert not sentinel.exists()
        assert executor.is_active(job_id) is False
    finally:
        if executor.is_active(job_id):
            await executor.stop(job_id)
        executor.cleanup(job_id)


@pytest.mark.asyncio
async def test_sigterm_ignoring_child_reaches_sigkill_before_return(
    tmp_path: Path,
) -> None:
    """The advertised shutdown bound includes a proven SIGKILL escalation."""

    policy = _policy()
    executor = MissionProcessExecutor(policy, attempt_root=tmp_path)
    job_id = "sigkill-escalation"
    lose_lease = asyncio.Event()

    async def heartbeat() -> None:
        await lose_lease.wait()
        raise RuntimeError("lease ownership lost")

    heartbeat_task = asyncio.create_task(heartbeat())
    execution_task = asyncio.create_task(
        executor.execute(
            _ignore_sigterm_and_block,
            (),
            job_id=job_id,
            hard_deadline_monotonic=time.monotonic() + 60.0,
            heartbeat_task=heartbeat_task,
        )
    )
    try:
        await _wait_for_child_file(executor, job_id, "request-started")

        lose_lease.set()
        with pytest.raises(RuntimeError, match="lease ownership lost"):
            await asyncio.wait_for(
                execution_task,
                timeout=float(policy.shutdown_bound_seconds or 0.0) + 2.0,
            )

        assert executor.is_active(job_id) is False
    finally:
        if not execution_task.done():
            execution_task.cancel()
        await asyncio.gather(execution_task, heartbeat_task, return_exceptions=True)
        if executor.is_active(job_id):
            await executor.stop(job_id)
        executor.cleanup(job_id)


@pytest.mark.asyncio
async def test_already_failed_heartbeat_never_starts_a_child(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "must-not-run"
    executor = MissionProcessExecutor(_policy(), attempt_root=tmp_path)

    async def heartbeat() -> None:
        raise RuntimeError("claim already lost")

    heartbeat_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="claim already lost"):
        await executor.execute(
            _write_forbidden_sentinel,
            (str(sentinel),),
            job_id="prelost-heartbeat",
            hard_deadline_monotonic=time.monotonic() + 10.0,
            heartbeat_task=heartbeat_task,
        )

    assert not sentinel.exists()
    assert executor.sandbox_path("prelost-heartbeat") is None
