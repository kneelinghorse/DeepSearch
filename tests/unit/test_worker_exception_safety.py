"""Worker boundaries must never export exception-controlled text."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from deepsearch.error_safety import safe_exception_text
from deepsearch.tracelab import CompletionReceiptClient, TraceLabAuth
from deepsearch.worker.__main__ import JSONFormatter, TextFormatter, main
from deepsearch.worker.database import DatabaseConfig
from deepsearch.worker.execution import (
    MissionProcessExecutor,
    RemoteExecutionError,
    _child_error_message,
    _forward_log,
    _PipeLogHandler,
)
from deepsearch.worker.lease import ExecutionFencePolicy, LeaseAttempt
from deepsearch.worker.main import DeepSearchWorker
from deepsearch.worker.poller import MissionRow

HOSTILE_MARKER = (
    "HOSTILE-MISSION-CONTENT body={'secret':'hunter2'} "
    "https://alice:password@example.test/provider?token=super-secret"
)
HOSTILE_FRAGMENTS = (
    "HOSTILE-MISSION-CONTENT",
    "hunter2",
    "alice:password",
    "token=super-secret",
)


def _provider_limits() -> dict[str, dict[str, object]]:
    return {
        "contract": {
            "backend": "deepseek",
            "request_timeout_seconds": 1.0,
            "client_max_retries": 0,
            "configured_max_retries": 0,
            "wrapper_max_retries": 0,
            "wrapper_attempts": 1,
            "hard_bound_mode": "spawn-process-v1",
        },
        "codeagent": {
            "backend": "deepseek-openai-compatible",
            "request_timeout_seconds": 1.0,
            "client_max_retries": 0,
            "smolagents_rate_limit_retry": False,
            "synthesis_attempts": 1,
            "max_step_fallback_attempts": 1,
            "recovery_passes": 1,
            "hard_bound_mode": "spawn-process-v1",
        },
        "tools": {
            "configured_request_timeout_seconds": 1.0,
            "configured_max_retries": 0,
            "linkup_request_timeout_seconds": 1.0,
            "linkup_max_retries": 0,
            "source_fetch_request_timeout_seconds": 1.0,
            "source_fetch_max_redirects": 1,
            "url_liveness_request_timeout_seconds": 1.0,
            "url_liveness_max_requests": 1,
            "scholar_backend_request_timeout_seconds": 1.0,
            "scholar_max_backends": 1,
            "hard_bound_mode": "spawn-process-v1",
        },
    }


def _policy() -> ExecutionFencePolicy:
    return ExecutionFencePolicy(
        heartbeat_timeout_seconds=1.0,
        cancellation_grace_seconds=0.1,
        terminate_grace_seconds=0.5,
        kill_grace_seconds=0.5,
        safety_margin_seconds=1.0,
        provider_limits=_provider_limits(),
    )


def _raise_hostile_child_error() -> None:
    raise ConnectionError(HOSTILE_MARKER)


def _hostile_exc_info() -> tuple[type[BaseException], BaseException, object]:
    try:
        raise ConnectionError(HOSTILE_MARKER)
    except ConnectionError:
        return sys.exc_info()  # type: ignore[return-value]


def _assert_hostile_absent(text: str) -> None:
    for fragment in HOSTILE_FRAGMENTS:
        assert fragment not in text
    assert "Traceback (most recent call last)" not in text


def _sample_row() -> MissionRow:
    now = datetime.now(timezone.utc)
    return MissionRow(
        id="safe-boundary-mission",
        mission_id="safe-boundary-mission",
        title="Authored title",
        objective="Authored objective",
        context={"background": "Authored context"},
        status="queued",
        queued_at=now,
        lease=LeaseAttempt(
            owner="worker-safe-boundary",
            token="opaque-lease-token",
            leased_at=now,
            heartbeat_at=now,
            expires_at=now,
            attempt_count=1,
        ),
        success_criteria=["Authored criterion"],
        research_phases={"max_loops": 1},
        research_depth="baseline",
    )


def test_child_error_envelope_contains_only_finite_category_and_class() -> None:
    """Provider bodies and URLs must not cross the child IPC boundary."""

    message = _child_error_message(ConnectionError(HOSTILE_MARKER))

    assert message == {
        "kind": "error",
        "category": "transport",
        "exception_class": "ConnectionError",
    }
    _assert_hostile_absent(json.dumps(message, sort_keys=True))


def test_child_exception_log_is_sanitized_before_parent_reemits(caplog) -> None:
    """An ``exc_info`` record cannot smuggle a body through the log pipe."""

    class RecordingConnection:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def send(self, message: dict[str, object]) -> None:
            self.messages.append(message)

    connection = RecordingConnection()
    handler = _PipeLogHandler(connection)
    record = logging.LogRecord(
        name="deepsearch.worker.child",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=f"Provider failed: {HOSTILE_MARKER}",
        args=(),
        exc_info=_hostile_exc_info(),
    )

    handler.emit(record)

    assert len(connection.messages) == 1
    message = connection.messages[0]
    assert message["message"] == "Unhandled exception"
    assert message["category"] == "transport"
    assert message["exception_class"] == "ConnectionError"
    _assert_hostile_absent(json.dumps(message, sort_keys=True))

    argument_record = logging.LogRecord(
        name="deepsearch.worker.child",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Provider failed: %s",
        args=(ConnectionError(HOSTILE_MARKER),),
        exc_info=None,
    )
    handler.emit(argument_record)
    argument_message = connection.messages[1]
    assert argument_message["message"] == (
        "Provider failed: "
        '{"category":"transport","exception_class":"ConnectionError"}'
    )
    _assert_hostile_absent(json.dumps(argument_message, sort_keys=True))

    with caplog.at_level(logging.ERROR, logger="deepsearch.worker.child"):
        _forward_log(message)
    rendered = "\n".join(item.getMessage() for item in caplog.records)
    assert '"category":"transport"' in rendered
    assert '"exception_class":"ConnectionError"' in rendered
    _assert_hostile_absent(rendered)


@pytest.mark.asyncio
async def test_isolated_child_failure_reaches_parent_without_raw_text(
    tmp_path: Path,
    caplog,
) -> None:
    """The real spawned-child path logs and raises only the safe projection."""

    executor = MissionProcessExecutor(_policy(), attempt_root=tmp_path)
    job_id = "hostile-child-error"
    try:
        with (
            caplog.at_level(logging.ERROR, logger="deepsearch.worker.execution"),
            pytest.raises(RemoteExecutionError) as exc_info,
        ):
            await executor.execute(
                _raise_hostile_child_error,
                (),
                job_id=job_id,
                hard_deadline_monotonic=time.monotonic() + 30.0,
            )

        assert exc_info.value.category == "transport"
        assert exc_info.value.remote_type == "ConnectionError"
        assert str(exc_info.value) == (
            '{"category":"transport","exception_class":"ConnectionError"}'
        )
        rendered = "\n".join(item.getMessage() for item in caplog.records)
        assert str(exc_info.value) in rendered
        _assert_hostile_absent(rendered + str(exc_info.value))
    finally:
        if executor.is_active(job_id):
            await executor.stop(job_id)
        executor.cleanup(job_id)


@pytest.mark.asyncio
async def test_mission_failure_persists_and_logs_only_safe_error(
    monkeypatch,
    caplog,
) -> None:
    """Mission-authored/provider text must not reach Railway, TraceLab, or SQL."""

    for name in (
        "TRACELAB_BASE_URL",
        "TRACELAB_API_KEY",
        "TRACELAB_DEEPSEARCH_SERVICE_SECRET",
        "TRACELAB_USERNAME",
        "TRACELAB_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    worker = DeepSearchWorker(
        db_config=DatabaseConfig(url="postgresql://test:test@localhost/test"),
        enable_health_server=False,
        receipt_client=CompletionReceiptClient(base_url="", secret=""),
        tracelab_auth=TraceLabAuth(
            base_url="",
            api_key="",
            username="",
            password="",
        ),
    )
    writer = AsyncMock()
    worker._writer = writer

    with (
        patch(
            "deepsearch.worker.main.convert_row_to_state",
            return_value={"mission_id": "safe-boundary-mission"},
        ),
        patch.object(
            worker,
            "_execute_research",
            new=AsyncMock(side_effect=ConnectionError(HOSTILE_MARKER)),
        ),
        caplog.at_level(logging.ERROR),
    ):
        await worker._handle_mission(_sample_row())

    writer.mark_failed.assert_awaited_once()
    persisted_error = writer.mark_failed.await_args.args[1]
    assert persisted_error == (
        '{"category":"transport","exception_class":"ConnectionError"}'
    )
    rendered = "\n".join(item.getMessage() for item in caplog.records)
    assert persisted_error in rendered
    assert all(item.exc_info is None for item in caplog.records)
    _assert_hostile_absent(rendered + persisted_error)


@pytest.mark.parametrize("formatter", [JSONFormatter(), TextFormatter()])
def test_console_formatters_replace_exc_info_with_safe_projection(formatter) -> None:
    """Console formatters must never append exception text or a traceback."""

    record = logging.LogRecord(
        name="deepsearch.worker.cli",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=f"Fatal provider response: {HOSTILE_MARKER}",
        args=(),
        exc_info=_hostile_exc_info(),
    )

    rendered = formatter.format(record)

    assert "Unhandled exception" in rendered
    assert "transport" in rendered
    assert "ConnectionError" in rendered
    _assert_hostile_absent(rendered)


@pytest.mark.asyncio
async def test_cli_main_logs_only_safe_category_and_exception_class(
    monkeypatch,
    capsys,
) -> None:
    """A fatal worker exception cannot leak its body through Railway JSON logs."""

    monkeypatch.setenv("WORKER_LOG_FORMAT", "json")
    monkeypatch.setenv("WORKER_LOG_LEVEL", "INFO")
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    try:
        with patch("deepsearch.worker.main.DeepSearchWorker") as worker_class:
            worker_class.return_value.run = AsyncMock(
                side_effect=ConnectionError(HOSTILE_MARKER)
            )
            exit_code = await main()

        output = capsys.readouterr().out
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            if handler not in original_handlers:
                handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)

    assert exit_code == 1
    _assert_hostile_absent(output)
    entries = [json.loads(line) for line in output.splitlines() if line.strip()]
    failure = next(entry for entry in entries if entry["level"] == "ERROR")
    assert failure["message"] == (
        "Worker failed: "
        '{"category":"transport","exception_class":"ConnectionError"}'
    )
    assert "exception" not in failure
    assert safe_exception_text(ConnectionError(HOSTILE_MARKER)) in failure["message"]
