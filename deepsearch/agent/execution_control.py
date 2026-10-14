"""Task-local cooperative cancellation for one mission execution."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Protocol


class CancellationEvent(Protocol):
    """Small interface shared by threading and multiprocessing events."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


_execution_cancel_event: ContextVar[CancellationEvent | None] = ContextVar(
    "deepsearch_execution_cancel_event",
    default=None,
)


class ExecutionCancelledError(RuntimeError):
    """Raised when a mission must stop after losing its execution fence."""


@contextmanager
def bind_execution_cancel_event(event: CancellationEvent) -> Iterator[None]:
    """Bind ``event`` to work spawned for the current mission task."""

    token = _execution_cancel_event.set(event)
    try:
        yield
    finally:
        _execution_cancel_event.reset(token)


def current_execution_cancel_event() -> CancellationEvent | None:
    """Return the cancellation event bound to the current mission, if any."""

    return _execution_cancel_event.get()


def raise_if_execution_cancelled(event: CancellationEvent | None = None) -> None:
    """Fail before more work when the active mission fence was lost."""

    resolved = event if event is not None else current_execution_cancel_event()
    if resolved is not None and resolved.is_set():
        raise ExecutionCancelledError("mission execution cancelled after lease loss")


def wait_for_execution_cancel(
    seconds: float,
    event: CancellationEvent | None = None,
) -> None:
    """Wait for retry backoff while remaining responsive to cancellation."""

    resolved = event if event is not None else current_execution_cancel_event()
    if resolved is None:
        time.sleep(seconds)
        return
    if resolved.wait(seconds):
        raise ExecutionCancelledError("mission execution cancelled during retry backoff")


__all__ = [
    "CancellationEvent",
    "ExecutionCancelledError",
    "bind_execution_cancel_event",
    "current_execution_cancel_event",
    "raise_if_execution_cancelled",
    "wait_for_execution_cancel",
]
