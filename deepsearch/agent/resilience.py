"""Resilience primitives shared across DeepSearch agent components."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def _utc_now() -> str:
    """Return a truncated ISO-8601 timestamp in UTC."""

    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class DegradationPolicy(str, Enum):
    """Available fallback modes when the agent detects sustained failures."""

    NORMAL = "normal"  # Standard execution path
    SKIP_WEB_SEARCH = "skip_web_search"  # Continue without invoking external tools
    CONSERVATIVE_LLM = "conservative_llm"  # Use cached prompts / simplified instructions
    FINALIZE_EARLY = "finalize_early"  # Stop gathering new info and finalize immediately


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(RuntimeError):
    """Raised when attempting to use a circuit that is currently open."""


class CircuitBreaker:
    """Simple circuit breaker for async tool / LLM calls."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60,
        name: str = "service",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.success_threshold = max(1, success_threshold)
        self.timeout = timedelta(seconds=max(1, timeout_seconds))
        self.name = name
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None

    @property
    def state(self) -> CircuitState:
        return self._state

    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def call(
        self,
        func: Callable[P, Awaitable[T] | T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute ``func`` through the circuit breaker."""

        self._ensure_call_allowed()
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
        except Exception as exc:
            self._record_failure(exc)
            raise
        self._record_success()
        return result  # type: ignore[return-value]

    def _ensure_call_allowed(self) -> None:
        if self._state == CircuitState.OPEN:
            if self._can_enter_half_open():
                logger.debug("Circuit %s transitioning to HALF_OPEN", self.name)
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
            else:
                raise CircuitBreakerOpenError(f"Circuit '{self.name}' is open; refusing calls")

    def _can_enter_half_open(self) -> bool:
        if self._last_failure_time is None:
            return True
        return self._clock() - self._last_failure_time >= self.timeout

    def _record_failure(self, error: Exception) -> None:
        now = self._clock()
        self._failure_count += 1
        self._last_failure_time = now
        self._success_count = 0
        if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.warning("Circuit %s opened after failure: %s", self.name, error)
            self._state = CircuitState.OPEN
        else:
            logger.debug(
                "Circuit %s failure %s/%s: %s",
                self.name,
                self._failure_count,
                self.failure_threshold,
                error,
            )

    def _record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                logger.info("Circuit %s closed after successful probe", self.name)
                self._reset()
                return
        else:
            self._failure_count = 0
        if self._state == CircuitState.CLOSED:
            return
        if self._state == CircuitState.OPEN and self._can_enter_half_open():
            logger.info("Circuit %s ready for half-open probe", self.name)
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0

    def _reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


class ErrorRecord(TypedDict, total=False):
    """Structured error record stored in AgentState.error_history."""

    timestamp: str
    component: str
    error_type: str
    message: str
    retry_count: int
    recovered: bool
    metadata: Dict[str, Any]


class ErrorTracker:
    """Manage stateful error histories."""

    def __init__(
        self,
        entries: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_entries: int = 50,
    ) -> None:
        self._max_entries = max(1, max_entries)
        self._entries: List[ErrorRecord] = []
        if entries:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                record: ErrorRecord = {
                    "timestamp": str(entry.get("timestamp") or _utc_now()),
                    "component": str(entry.get("component") or "unknown"),
                    "error_type": str(entry.get("error_type") or "unknown"),
                    "message": str(entry.get("message") or ""),
                    "retry_count": int(entry.get("retry_count") or 0),
                    "recovered": bool(entry.get("recovered") or False),
                }
                metadata = entry.get("metadata")
                if isinstance(metadata, dict):
                    record["metadata"] = dict(metadata)
                self._entries.append(record)

    @classmethod
    def from_state(cls, state: Dict[str, Any], *, max_entries: int = 50) -> "ErrorTracker":
        return cls(state.get("error_history"), max_entries=max_entries)

    def record(
        self,
        *,
        component: str,
        error_type: str,
        message: str,
        retry_count: int = 0,
        recovered: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ErrorRecord:
        entry: ErrorRecord = {
            "timestamp": _utc_now(),
            "component": component,
            "error_type": error_type,
            "message": message,
            "retry_count": retry_count,
            "recovered": recovered,
        }
        if metadata:
            entry["metadata"] = metadata
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        return entry

    def mark_recovered(self, component: str) -> None:
        for entry in reversed(self._entries):
            if entry.get("component") == component and not entry.get("recovered"):
                entry["recovered"] = True
                break

    def to_list(self) -> List[ErrorRecord]:
        return [dict(entry) for entry in self._entries]


def retry_with_backoff(
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_ms: int = 100,
    retriable_exceptions: Tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[T] | T]], Callable[P, Awaitable[T] | T]]:
    """Retry decorator with exponential backoff + jitter."""

    def decorator(func: Callable[P, Awaitable[T] | T]) -> Callable[P, Awaitable[T] | T]:
        is_async = asyncio.iscoroutinefunction(func)

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = base_delay
            attempts = max(0, max_retries)
            last_error: BaseException | None = None
            for attempt in range(attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if inspect.isawaitable(result):
                        return await result  # type: ignore[return-value]
                    return result  # type: ignore[return-value]
                except retriable_exceptions as exc:
                    last_error = exc
                    if attempt == attempts:
                        logger.error("%s failed after %s retries: %s", func.__name__, attempts, exc)
                        raise
                    jitter = random.uniform(0, max(jitter_ms, 0) / 1000)
                    total_delay = min(delay, max_delay) + jitter
                    logger.warning(
                        "%s attempt %s/%s failed: %s. Retrying in %.2fs",
                        func.__name__,
                        attempt + 1,
                        attempts + 1,
                        exc,
                        total_delay,
                    )
                    await asyncio.sleep(total_delay)
                    delay = min(delay * 2, max_delay)
            assert last_error is not None
            raise last_error

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = base_delay
            attempts = max(0, max_retries)
            for attempt in range(attempts + 1):
                try:
                    return func(*args, **kwargs)  # type: ignore[return-value]
                except retriable_exceptions as exc:
                    if attempt == attempts:
                        logger.error("%s failed after %s retries: %s", func.__name__, attempts, exc)
                        raise
                    jitter = random.uniform(0, max(jitter_ms, 0) / 1000)
                    total_delay = min(delay, max_delay) + jitter
                    logger.warning(
                        "%s attempt %s/%s failed: %s. Retrying in %.2fs",
                        func.__name__,
                        attempt + 1,
                        attempts + 1,
                        exc,
                        total_delay,
                    )
                    time.sleep(total_delay)
                    delay = min(delay * 2, max_delay)
            raise RuntimeError("unreachable")

        return async_wrapper if is_async else sync_wrapper

    return decorator


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "DegradationPolicy",
    "ErrorRecord",
    "ErrorTracker",
    "retry_with_backoff",
]
