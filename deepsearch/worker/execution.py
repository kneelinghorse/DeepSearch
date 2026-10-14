"""Killable per-mission execution boundary for lease-owned work."""

from __future__ import annotations

import asyncio
import ctypes
import inspect
import logging
import math
import multiprocessing as multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from deepsearch.agent.execution_control import (
    bind_execution_cancel_event,
    raise_if_execution_cancelled,
)
from deepsearch.error_safety import (
    SafeErrorDetails,
    safe_error_from_parts,
    safe_exception_details,
    safe_exception_text,
)
from deepsearch.tracelab.log_projection import project_log_fields

from .lease import ExecutionFencePolicy

logger = logging.getLogger(__name__)

DEFAULT_ATTEMPT_ROOT = Path("checkpoints/worker-attempts")
_TERMINAL_MESSAGE_KINDS = frozenset({"result", "error"})


class IsolatedExecutionError(RuntimeError):
    """Raised when isolated mission execution cannot return a valid result."""


class ProcessTerminationError(IsolatedExecutionError):
    """Raised when the worker cannot prove that an isolated child is dead."""


class RemoteExecutionError(IsolatedExecutionError):
    """Normalized exception raised by the isolated child process."""

    def __init__(self, details: SafeErrorDetails) -> None:
        safe_details = safe_error_from_parts(
            details.category,
            details.exception_class,
            default_category="execution",
        )
        super().__init__(safe_details.to_text())
        self.category = safe_details.category
        self.remote_type = safe_details.exception_class


@dataclass(slots=True)
class _ProcessHandle:
    job_id: str
    process: Any
    result_connection: Any
    control_connection: Any
    cancel_event: Any
    sandbox: Path
    stop_attempted: bool = False


class _PipeLogHandler(logging.Handler):
    """Forward already-formatted child logs through the drained result pipe."""

    def __init__(self, connection: Any) -> None:
        super().__init__()
        self._connection = connection

    def emit(self, record: logging.LogRecord) -> None:
        try:
            error_details = None
            if record.exc_info and isinstance(record.exc_info[1], BaseException):
                error_details = safe_exception_details(record.exc_info[1])
            self._connection.send(
                {
                    "kind": "log",
                    "name": record.name,
                    "level": record.levelno,
                    "message": (
                        "Unhandled exception"
                        if error_details is not None
                        else _safe_child_log_message(record)
                    ),
                    "created": record.created,
                    "observation": project_log_fields(vars(record)),
                    **(error_details.to_dict() if error_details is not None else {}),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            return


def _safe_child_log_message(record: logging.LogRecord) -> str:
    """Format a child log without stringifying exception arguments."""

    if isinstance(record.msg, BaseException):
        return safe_exception_text(record.msg)
    if not record.args:
        return str(record.msg)
    projected = copy(record)
    if isinstance(record.args, dict):
        projected.args = {
            key: (
                safe_exception_text(value)
                if isinstance(value, BaseException)
                else value
            )
            for key, value in record.args.items()
        }
    else:
        projected.args = tuple(
            safe_exception_text(value) if isinstance(value, BaseException) else value
            for value in record.args
        )
    return projected.getMessage()


def _install_linux_parent_death_signal(expected_parent_pid: int) -> None:
    """Ask Linux to kill the child if its worker parent disappears."""

    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(1, signal.SIGKILL, 0, 0, 0)
    except (AttributeError, OSError):
        return
    if result != 0:
        return
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _watch_parent(
    control_connection: Any,
    hard_deadline_monotonic: float,
    armed: threading.Event,
) -> None:
    """Exit on parent loss or before this lease can become reclaimable."""

    try:
        deadline = hard_deadline_monotonic
        if deadline <= time.monotonic():
            os._exit(73)
        armed.set()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not control_connection.poll(remaining):
                os._exit(73)
            message = control_connection.recv()
            if message == "stop":
                return
            if (
                isinstance(message, tuple)
                and len(message) == 2
                and message[0] == "deadline"
                and isinstance(message[1], (int, float))
                and not isinstance(message[1], bool)
                and math.isfinite(float(message[1]))
            ):
                deadline = float(message[1])
                continue
            os._exit(74)
    except (EOFError, OSError):
        os._exit(72)


def _configure_attempt_environment(sandbox: Path) -> None:
    """Confine every worker-created mission artifact to one attempt directory."""

    os.environ["DEEPSEARCH_CONTRACT_DIR"] = str(sandbox / "contracts")
    os.environ["DEEPSEARCH_REPORT_DIR"] = str(sandbox / "reports")
    os.environ["DEEPSEARCH_TELEMETRY_DIR"] = str(sandbox / "telemetry")
    os.environ["CHECKPOINT_DB_PATH"] = str(sandbox / "checkpoints" / "deepsearch.db")


async def _await_result(awaitable: Awaitable[Any]) -> Any:
    """Normalize an arbitrary awaitable into an ``asyncio.run`` coroutine."""

    return await awaitable


def _child_error_message(exc: BaseException) -> dict[str, str]:
    """Return the complete secret-safe child error envelope."""

    error_details = safe_exception_details(exc)
    return {"kind": "error", **error_details.to_dict()}


def _child_entry(
    result_connection: Any,
    control_connection: Any,
    cancel_event: Any,
    expected_parent_pid: int,
    sandbox_text: str,
    hard_deadline_monotonic: float,
    target: Callable[..., Any],
    args: Sequence[Any],
) -> None:
    """Arm containment before invoking any mission/provider code."""

    _install_linux_parent_death_signal(expected_parent_pid)
    watchdog_armed = threading.Event()
    watchdog = threading.Thread(
        target=_watch_parent,
        args=(control_connection, hard_deadline_monotonic, watchdog_armed),
        name="deepsearch-parent-watchdog",
        daemon=True,
    )
    watchdog.start()
    if not watchdog_armed.wait(timeout=1.0):
        os._exit(75)
    if hard_deadline_monotonic <= time.monotonic():
        os._exit(73)

    sandbox = Path(sandbox_text)
    _configure_attempt_environment(sandbox)
    child_handler = _PipeLogHandler(result_connection)
    child_logger = logging.getLogger("deepsearch")
    child_logger.addHandler(child_handler)

    try:
        result_connection.send({"kind": "ready", "pid": os.getpid()})
        with bind_execution_cancel_event(cancel_event):
            raise_if_execution_cancelled(cancel_event)
            result = target(*args)
            if inspect.isawaitable(result):
                result = asyncio.run(_await_result(result))
            raise_if_execution_cancelled(cancel_event)
        result_connection.send({"kind": "result", "value": result})
    except BaseException as exc:  # noqa: BLE001 - cross-process normalization
        try:
            result_connection.send(_child_error_message(exc))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        child_logger.removeHandler(child_handler)
        child_handler.close()
        result_connection.close()


def _forward_log(message: Mapping[str, Any]) -> None:
    """Re-emit a child log so the parent's TraceLab handler sees it."""

    name = str(message.get("name") or "deepsearch.worker.child")
    level = int(message.get("level") or logging.INFO)
    text = str(message.get("message") or "")
    if "category" in message or "exception_class" in message:
        error_details = safe_error_from_parts(
            message.get("category"),
            message.get("exception_class"),
        )
        text = f"{text}: {error_details.to_text()}"
    record = logging.LogRecord(name, level, "", 0, text, (), None)
    created = message.get("created")
    if isinstance(created, (int, float)):
        record.created = float(created)
    observation = message.get("observation")
    if isinstance(observation, Mapping):
        for key, value in project_log_fields(observation).items():
            setattr(record, key, value)
    logging.getLogger(name).handle(record)


def _receive_terminal_message(connection: Any) -> dict[str, Any]:
    """Continuously drain logs and return the child's single terminal message."""

    try:
        while True:
            message = connection.recv()
            if not isinstance(message, Mapping):
                raise IsolatedExecutionError("isolated child sent a malformed message")
            kind = str(message.get("kind") or "")
            if kind == "log":
                _forward_log(message)
                continue
            if kind == "ready":
                continue
            if kind in _TERMINAL_MESSAGE_KINDS:
                return dict(message)
            raise IsolatedExecutionError(
                f"isolated child sent unknown message kind {kind!r}"
            )
    except (EOFError, OSError) as exc:
        raise IsolatedExecutionError(
            "isolated child exited without a terminal result"
        ) from exc


class MissionProcessExecutor:
    """Run one mission in a fresh spawn process and prove it dead on exit."""

    def __init__(
        self,
        policy: ExecutionFencePolicy,
        *,
        attempt_root: Path | str | None = None,
    ) -> None:
        if not policy.valid:
            raise ValueError("MissionProcessExecutor requires a valid fence policy")
        self.policy = policy
        root_value = attempt_root
        if root_value is None:
            root_value = os.getenv(
                "WORKER_ATTEMPT_ROOT",
                str(DEFAULT_ATTEMPT_ROOT),
            )
        root = Path(root_value)
        self.attempt_root = root.expanduser().resolve()
        self._context = multiprocessing.get_context("spawn")
        self._handles: dict[str, _ProcessHandle] = {}

    def _start(
        self,
        target: Callable[..., Any],
        args: Sequence[Any],
        job_id: str,
        hard_deadline_monotonic: float,
    ) -> _ProcessHandle:
        if job_id in self._handles:
            raise IsolatedExecutionError(f"job {job_id} already has an execution child")
        if not math.isfinite(hard_deadline_monotonic):
            raise IsolatedExecutionError("mission hard deadline must be finite")
        self.attempt_root.mkdir(parents=True, exist_ok=True)
        sandbox = Path(
            tempfile.mkdtemp(prefix=f"{job_id[:12]}-", dir=self.attempt_root)
        ).resolve()
        result_receiver, result_sender = self._context.Pipe(duplex=False)
        control_receiver, control_sender = self._context.Pipe(duplex=False)
        cancel_event = self._context.Event()
        process = self._context.Process(
            target=_child_entry,
            args=(
                result_sender,
                control_receiver,
                cancel_event,
                os.getpid(),
                str(sandbox),
                hard_deadline_monotonic,
                target,
                tuple(args),
            ),
            name=f"deepsearch-mission-{job_id[:12]}",
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            result_receiver.close()
            result_sender.close()
            control_receiver.close()
            control_sender.close()
            shutil.rmtree(sandbox, ignore_errors=True)
            raise
        result_sender.close()
        control_receiver.close()
        handle = _ProcessHandle(
            job_id=job_id,
            process=process,
            result_connection=result_receiver,
            control_connection=control_sender,
            cancel_event=cancel_event,
            sandbox=sandbox,
        )
        self._handles[job_id] = handle
        return handle

    async def execute(
        self,
        target: Callable[..., Any],
        args: Sequence[Any],
        *,
        job_id: str,
        hard_deadline_monotonic: float,
        heartbeat_task: asyncio.Task[None] | None = None,
    ) -> Any:
        """Execute ``target`` until it returns or the ownership watcher fails."""

        if heartbeat_task is not None and heartbeat_task.done():
            heartbeat_error = await self._heartbeat_error(heartbeat_task)
            raise heartbeat_error
        handle = self._start(
            target,
            args,
            job_id,
            hard_deadline_monotonic,
        )
        result_task = asyncio.create_task(
            asyncio.to_thread(
                _receive_terminal_message,
                handle.result_connection,
            )
        )
        waiters: set[asyncio.Task[Any]] = {result_task}
        if heartbeat_task is not None:
            waiters.add(heartbeat_task)
        shutdown_started = False
        try:
            done, _ = await asyncio.wait(
                waiters,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task is not None and heartbeat_task in done:
                heartbeat_error = await self._heartbeat_error(heartbeat_task)
                shutdown_started = True
                await self.stop(job_id)
                await self._settle_receiver(result_task)
                raise heartbeat_error

            message = await result_task
            shutdown_started = True
            await self._stop_after_terminal_message(handle)
            if message["kind"] == "error":
                error_details = safe_error_from_parts(
                    message.get("category"),
                    message.get("exception_class"),
                    default_category="execution",
                )
                logger.error(
                    "Isolated execution failed: %s",
                    error_details.to_text(),
                )
                raise RemoteExecutionError(error_details)
            value = message.get("value")
            self._validate_artifact_paths(value, handle.sandbox)
            return value
        except asyncio.CancelledError:
            shutdown_started = True
            await asyncio.shield(self.stop(job_id))
            await self._settle_receiver(result_task)
            raise
        except BaseException:
            if handle.process.is_alive() and not shutdown_started:
                await self.stop(job_id)
            raise
        finally:
            if not result_task.done() and not handle.process.is_alive():
                await self._settle_receiver(result_task)

    @staticmethod
    async def _heartbeat_error(task: asyncio.Task[None]) -> BaseException:
        try:
            await task
        except BaseException as exc:  # preserve the exact ownership-loss cause
            return exc
        return IsolatedExecutionError("heartbeat watcher stopped during mission execution")

    @staticmethod
    async def _settle_receiver(task: asyncio.Task[dict[str, Any]]) -> None:
        try:
            await task
        except (asyncio.CancelledError, IsolatedExecutionError):
            pass

    async def _join(self, handle: _ProcessHandle, timeout: float) -> bool:
        await asyncio.to_thread(handle.process.join, timeout)
        return not handle.process.is_alive()

    async def _stop_after_terminal_message(self, handle: _ProcessHandle) -> None:
        self._begin_stop(handle)
        timeout = float(self.policy.cancellation_grace_seconds or 0.0)
        if await self._join(handle, timeout):
            return
        await self._terminate(handle, signal_cancel=False)

    async def stop(self, job_id: str) -> None:
        """Cooperatively cancel, then TERM/KILL and confirm process death."""

        handle = self._handles.get(job_id)
        if handle is None or not handle.process.is_alive():
            if handle is not None:
                await asyncio.to_thread(handle.process.join, 0)
            return
        self._begin_stop(handle)
        handle.cancel_event.set()
        grace = float(self.policy.cancellation_grace_seconds or 0.0)
        if await self._join(handle, grace):
            return
        await self._terminate(handle, signal_cancel=True)

    async def _terminate(
        self,
        handle: _ProcessHandle,
        *,
        signal_cancel: bool,
    ) -> None:
        if signal_cancel:
            handle.cancel_event.set()
        handle.process.terminate()
        terminate_grace = float(self.policy.terminate_grace_seconds or 0.0)
        if await self._join(handle, terminate_grace):
            return
        handle.process.kill()
        kill_grace = float(self.policy.kill_grace_seconds or 0.0)
        if await self._join(handle, kill_grace):
            return
        raise ProcessTerminationError(
            f"isolated child {handle.process.pid} remained alive after hard kill"
        )

    @staticmethod
    def _begin_stop(handle: _ProcessHandle) -> None:
        if handle.stop_attempted:
            raise ProcessTerminationError(
                f"isolated child {handle.process.pid} already exhausted its stop bound"
            )
        handle.stop_attempted = True

    def extend_deadline(self, job_id: str, hard_deadline_monotonic: float) -> None:
        """Refresh a live child's self-termination deadline after a DB heartbeat."""

        if not math.isfinite(hard_deadline_monotonic):
            raise IsolatedExecutionError("mission hard deadline must be finite")
        handle = self._handles.get(job_id)
        if handle is None or not handle.process.is_alive():
            return
        try:
            handle.control_connection.send(("deadline", hard_deadline_monotonic))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise IsolatedExecutionError(
                "isolated child deadline watchdog became unavailable"
            ) from exc

    def is_active(self, job_id: str) -> bool:
        """Return whether an attempt still has a live isolated child."""

        handle = self._handles.get(job_id)
        return bool(handle is not None and handle.process.is_alive())

    def child_pid(self, job_id: str) -> int | None:
        """Return the child PID for diagnostics and overlap tests."""

        handle = self._handles.get(job_id)
        return None if handle is None else handle.process.pid

    def sandbox_path(self, job_id: str) -> Path | None:
        """Return the attempt-local artifact root, if it has been started."""

        handle = self._handles.get(job_id)
        return None if handle is None else handle.sandbox

    def cleanup(self, job_id: str) -> None:
        """Remove a dead attempt's isolated artifacts and process handles."""

        handle = self._handles.get(job_id)
        if handle is None:
            return
        if handle.process.is_alive():
            raise ProcessTerminationError(
                f"refusing to clean live isolated child {handle.process.pid}"
            )
        if handle.sandbox.parent != self.attempt_root:
            raise IsolatedExecutionError("attempt sandbox escaped configured root")
        if handle.sandbox.exists():
            shutil.rmtree(handle.sandbox)
        handle.result_connection.close()
        handle.control_connection.close()
        self._handles.pop(job_id, None)

    async def close(self) -> None:
        """Stop and clean every retained child during worker shutdown."""

        for job_id in tuple(self._handles):
            await self.stop(job_id)
            self.cleanup(job_id)

    @staticmethod
    def _validate_artifact_paths(value: Any, sandbox: Path) -> None:
        if not isinstance(value, Mapping):
            return
        paths: list[Any] = [
            value.get("mission_contract_path"),
            value.get("report_path"),
            value.get("protocol_path"),
        ]
        telemetry = value.get("telemetry")
        if isinstance(telemetry, Mapping):
            paths.append(telemetry.get("telemetry_artifact_path"))
        for path_value in paths:
            if not isinstance(path_value, str) or not path_value.strip():
                continue
            resolved = Path(path_value).expanduser().resolve()
            if not resolved.is_relative_to(sandbox):
                raise IsolatedExecutionError(
                    f"isolated artifact path escaped attempt sandbox: {resolved.name}"
                )


__all__ = [
    "DEFAULT_ATTEMPT_ROOT",
    "IsolatedExecutionError",
    "MissionProcessExecutor",
    "ProcessTerminationError",
    "RemoteExecutionError",
]
