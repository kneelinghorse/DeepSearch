"""Acknowledged TraceLab mission-log transport.

The handler buffers records in emission order and removes a batch only after
TraceLab acknowledges the exact attempt and event identities. A hard capacity
retains the most recent entries and reports every overflow. Delivery failures
never interrupt research, but they remain visible through transport telemetry and a logger that
is deliberately outside the ``deepsearch`` hierarchy to avoid recursion.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any
from uuid import uuid4

from deepsearch.error_safety import SAFE_ERROR_CATEGORIES
from deepsearch.worker.lease import LeaseAttempt

from .auth import TraceLabAuth, get_default_auth
from .log_projection import _SAFE_LEVELS, _SAFE_PHASES, project_log_fields

_transport_logger = logging.getLogger("tracelab.transport")

FLUSH_THRESHOLD = 10
FLUSH_INTERVAL_SECONDS = 5.0
MAX_BUFFER_SIZE = 200
MAX_PENDING_ENTRIES = 2_000
CONTRACT_VERSION = "tracelab-mission-logs-v2"
MAX_DELIVERY_ATTEMPTS = 3
RETRY_BACKOFF_BASE_SECONDS = 1.0


class TracelabLogHandler(logging.Handler):
    """Push ordered, acknowledged log batches to one TraceLab mission.

    ``max_buffer`` is retained as a constructor name for compatibility, but it
    now bounds one HTTP batch. The pending queue is retained until a successful
    response unless its explicit capacity is exceeded.
    """

    def __init__(
        self,
        mission_id: str,
        base_url: str | None = None,
        flush_threshold: int = FLUSH_THRESHOLD,
        flush_interval: float = FLUSH_INTERVAL_SECONDS,
        max_buffer: int = MAX_BUFFER_SIZE,
        max_pending_entries: int = MAX_PENDING_ENTRIES,
        timeout: float = 10.0,
        max_attempts: int = MAX_DELIVERY_ATTEMPTS,
        backoff_base: float = RETRY_BACKOFF_BASE_SECONDS,
        auth: TraceLabAuth | None = None,
        *,
        lease: LeaseAttempt,
    ) -> None:
        super().__init__()
        self.mission_id = mission_id
        self._lease = lease
        self._sequence = 0
        # Credentials and destination are one trust unit. When auth is
        # injected, its verified origin wins over both explicit and env URLs.
        resolved_base_url = (
            auth.base_url
            if auth is not None
            else base_url
            if base_url is not None
            else os.getenv("TRACELAB_BASE_URL", "")
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.flush_interval = max(0.1, flush_interval)
        self.max_batch_size = min(MAX_BUFFER_SIZE, max(1, max_buffer))
        self.max_pending_entries = max(1, max_pending_entries)
        self.flush_threshold = min(
            max(1, flush_threshold),
            self.max_pending_entries,
        )
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.backoff_base = max(0.0, backoff_base)
        self._auth = auth
        self.auto_delivery = False
        self._negotiated = False
        self._stop_delivery = threading.Event()

        self._buffer: list[dict[str, Any]] = []
        self._inflight_entry_ids: set[int] = set()
        self._buffer_lock = threading.Lock()
        self._delivery_lock = threading.Lock()
        self._phase = "running"
        self._timer: threading.Timer | None = None
        self._delivery_scheduled = False
        self._closed = False

        self._accepted_entries = 0
        self._delivery_attempts = 0
        self._failed_batches = 0
        self._overflowed_entries = 0
        self._discarded_entries = 0
        self._last_status_code: int | None = None
        self._last_error: str | None = None

    def set_phase(self, phase: str) -> None:
        """Set the source phase applied to subsequent log records."""

        normalized = str(phase or "running").strip().lower()
        self._phase = normalized if normalized in _SAFE_PHASES else "running"

    def emit(self, record: logging.LogRecord) -> None:
        """Append one canonical entry and schedule a threshold flush."""

        try:
            with self._buffer_lock:
                if self._closed or not self.base_url:
                    return
                self._sequence += 1
                fields = project_log_fields(vars(record))
                phase = fields.get("tracelab_phase", self._phase)
                entry: dict[str, Any] = {
                    "event_id": str(uuid4()),
                    "sequence": self._sequence,
                    "logged_at": self._format_ts(record.created),
                    "level": self._safe_level(record),
                    "source": f"deepsearch:{phase}",
                    "message": self._project_message(record),
                }
                overflowed = len(self._buffer) >= self.max_pending_entries
                if overflowed:
                    self._overflowed_entries += 1
                    evict_index = next(
                        (
                            index
                            for index, pending in enumerate(self._buffer)
                            if id(pending) not in self._inflight_entry_ids
                        ),
                        None,
                    )
                    if evict_index is not None:
                        del self._buffer[evict_index]
                        overflow_detail = (
                            "log queue capacity exceeded; oldest pending entry "
                            "was evicted"
                        )
                    else:
                        overflow_detail = (
                            "log queue capacity exceeded; newest entry was not "
                            "buffered while all entries were in flight"
                        )
                else:
                    evict_index = None
                    overflow_detail = ""
                if not overflowed or evict_index is not None:
                    self._buffer.append(entry)
                should_flush = (
                    self.auto_delivery and len(self._buffer) >= self.flush_threshold
                )
            if overflowed:
                self._record_failure(overflow_detail)
            if should_flush:
                self._drain_async()
        except Exception as exc:  # noqa: BLE001 - logging must not break research
            self._record_failure(f"buffering failed: {type(exc).__name__}")

    def flush(self) -> None:
        """Synchronously deliver queued entries, retaining any failed batch."""

        if not self.base_url:
            return
        self._deliver_buffer()

    def close(self) -> None:
        """Stop interval delivery and make one final acknowledged flush."""

        with self._buffer_lock:
            if self._closed:
                return
            self._closed = True
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()
        self.flush()
        super().close()

    def stop_delivery(self) -> None:
        """Stop scheduling immediately; callers must then quiesce with discard."""

        self._stop_delivery.set()
        with self._buffer_lock:
            self._closed = True
            self.auto_delivery = False
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()

    def discard(self) -> None:
        """Quiesce in-flight HTTP before the worker releases this attempt."""

        self.stop_delivery()
        with self._delivery_lock:
            with self._buffer_lock:
                self._discarded_entries += len(self._buffer)
                self._buffer.clear()
                self._inflight_entry_ids.clear()
        super().close()

    def start_live_delivery(self) -> bool:
        """Negotiate only after the worker confirms ownership and starts its watch."""

        with self._delivery_lock:
            ready = self._negotiate()
        with self._buffer_lock:
            if not ready or self._closed or self._stop_delivery.is_set():
                return False
            self.auto_delivery = True
        self._start_timer()
        self._drain_async()
        return True

    def get_telemetry(self) -> dict[str, Any]:
        """Return redacted delivery state for worker health reporting."""

        with self._buffer_lock:
            pending_entries = len(self._buffer)
        return {
            "configured": bool(self.base_url),
            "accepted_entries": self._accepted_entries,
            "pending_entries": pending_entries,
            "delivery_attempts": self._delivery_attempts,
            "failed_batches": self._failed_batches,
            "overflowed_entries": self._overflowed_entries,
            "discarded_entries": self._discarded_entries,
            "last_status_code": self._last_status_code,
            "last_error": self._last_error,
        }

    def _start_timer(self) -> None:
        with self._buffer_lock:
            if self._closed or not self.auto_delivery or self._stop_delivery.is_set():
                return
            self._timer = threading.Timer(self.flush_interval, self._timer_flush)
            self._timer.daemon = True
            self._timer.start()

    def _timer_flush(self) -> None:
        self._drain_async()
        if not self._closed:
            self._start_timer()

    def _drain_async(self) -> None:
        """Schedule at most one background delivery worker."""

        with self._buffer_lock:
            if (self._closed or self._stop_delivery.is_set()
                    or not self._buffer or self._delivery_scheduled):
                return
            self._delivery_scheduled = True

        thread = threading.Thread(target=self._background_flush, daemon=True)
        thread.start()

    def _background_flush(self) -> None:
        try:
            self._deliver_buffer()
        finally:
            with self._buffer_lock:
                self._delivery_scheduled = False

    def _deliver_buffer(self) -> None:
        """Deliver batches serially and remove only acknowledged prefixes."""

        with self._delivery_lock:
            while not self._stop_delivery.is_set():
                with self._buffer_lock:
                    entries = self._buffer[: self.max_batch_size]
                    self._inflight_entry_ids = {id(entry) for entry in entries}
                if not entries:
                    with self._buffer_lock:
                        self._inflight_entry_ids.clear()
                    return
                try:
                    acknowledged = self._post_entries(entries)
                finally:
                    with self._buffer_lock:
                        inflight_ids = set(self._inflight_entry_ids)
                        self._inflight_entry_ids.clear()
                if not acknowledged:
                    return
                with self._buffer_lock:
                    self._buffer[:] = [
                        entry
                        for entry in self._buffer
                        if id(entry) not in inflight_ids
                    ]
                self._accepted_entries += len(entries)

    def _negotiate(self) -> bool:
        """Require the deployed receiver contract; never probe legacy writes."""

        if self._stop_delivery.is_set() or not self.base_url:
            return False
        if self._negotiated:
            return True
        try:
            import httpx

            auth = self._auth or get_default_auth()
            auth.require_service_role()
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/api/v1/missions/{self.mission_id}/logs/capabilities",
                    headers=auth.get_headers(),
                )
            capabilities = response.json() if response.status_code == 200 else None
            self._negotiated = isinstance(capabilities, dict) and all((
                capabilities.get("contract_version") == CONTRACT_VERSION,
                type(capabilities.get("max_batch_entries")) is int
                and capabilities.get("max_batch_entries") == MAX_BUFFER_SIZE,
                capabilities.get("max_message_chars") == 2048,
                capabilities.get("final_flush") == "terminal_result_key",
                capabilities.get("legacy") in {"terminal_only_until_LOG-2_cutover", "retired"},
            ))
            if not self._negotiated:
                self._record_failure("TraceLab log capability negotiation failed")
        except Exception as exc:  # noqa: BLE001 - logging cannot break research
            self._record_failure(f"{type(exc).__name__} during log negotiation")
        return self._negotiated and not self._stop_delivery.is_set()

    def _valid_ack(self, response: Any, entries: list[dict[str, Any]]) -> bool:
        try:
            ack = response.json()
            return isinstance(ack, dict) and all((
                ack.get("contract_version") == CONTRACT_VERSION,
                ack.get("mission_id") == self.mission_id,
                type(ack.get("attempt_count")) is int
                and ack.get("attempt_count") == self._lease.attempt_count,
                type(ack.get("accepted")) is int and ack.get("accepted") >= 0,
                type(ack.get("replayed")) is int and ack.get("replayed") >= 0,
                ack.get("accepted", -1) + ack.get("replayed", -1) == len(entries),
                (response.status_code == 200 and ack.get("accepted") == 0)
                or (response.status_code == 201 and ack.get("accepted", 0) > 0),
                ack.get("event_ids") == [entry["event_id"] for entry in entries],
            ))
        except (ValueError, TypeError):
            return False

    def _post_entries(self, entries: list[dict[str, Any]]) -> bool:
        """Retry immutable IDs and proof; only a complete v2 ACK dequeues."""

        if not self._negotiate():
            return False
        try:
            import httpx

            auth = self._auth or get_default_auth()
            url = f"{self.base_url}/api/v1/missions/{self.mission_id}/logs/v2"
            payload = {
                "contract_version": CONTRACT_VERSION,
                "mission_id": self.mission_id,
                "attempt_count": self._lease.attempt_count,
                "lease_owner": self._lease.owner,
                "lease_token": self._lease.token,
                "logs": entries,
            }
            for attempt in range(1, self.max_attempts + 1):
                if self._stop_delivery.is_set():
                    return False
                self._delivery_attempts += 1
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        response = client.post(url, json=payload, headers=auth.get_headers())
                    self._last_status_code = response.status_code
                    if response.status_code in {200, 201}:
                        if self._valid_ack(response, entries):
                            self._last_error = None
                            return True
                        self._last_error = "Invalid TraceLab log acknowledgement"
                    else:
                        self._last_error = f"TraceLab returned HTTP {response.status_code}"
                        if response.status_code == 409:
                            # This immutable attempt can never be relabeled for a successor.
                            self.stop_delivery()
                            with self._buffer_lock:
                                self._discarded_entries += len(self._buffer)
                                self._buffer.clear()
                        if response.status_code != 429 and response.status_code < 500:
                            break
                except (httpx.TimeoutException, httpx.RequestError) as exc:
                    self._last_status_code = None
                    self._last_error = f"{type(exc).__name__} during log delivery"
                if attempt < self.max_attempts and self._stop_delivery.wait(
                    self.backoff_base * (2 ** (attempt - 1))
                ):
                    return False
        except Exception as exc:  # noqa: BLE001 - retain queue and expose failure
            self._last_error = f"{type(exc).__name__} during log delivery"

        self._failed_batches += 1
        self._record_failure(self._last_error or "unknown log delivery failure")
        return False

    def _record_failure(self, detail: str) -> None:
        self._last_error = detail
        category = "queue_overflow" if detail.startswith("log queue capacity") else "delivery"
        _transport_logger.error("TraceLab log delivery failed category=%s", category)

    def _project_message(self, record: logging.LogRecord) -> str:
        """Return an identity-free allowlisted mission-log event.

        Never format ``record.msg``/``args`` or inspect ``exc_info``: those
        surfaces may contain mission-authored text, job identities, signed
        URLs, response bodies, or tracebacks. Only fixed enums and bounded
        numeric telemetry may cross the TraceLab log boundary.
        """

        fields = project_log_fields(vars(record))
        category = fields.get("tracelab_category", self._safe_level(record).lower())
        event = fields.get("tracelab_event")
        if event is None:
            if category in {"error", "critical"} or category in SAFE_ERROR_CATEGORIES:
                event = "mission_error"
            elif category == "warning":
                event = "mission_warning"
            else:
                event = "mission_progress"
        projected: dict[str, object] = {
            "category": category,
            "event": event,
            "phase": fields.get("tracelab_phase", self._phase),
        }
        for field in ("count", "duration_seconds"):
            if f"tracelab_{field}" in fields:
                projected[field] = fields[f"tracelab_{field}"]
        return json.dumps(projected, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _safe_level(record: logging.LogRecord) -> str:
        level = str(record.levelname or "INFO").upper()
        return level if level in _SAFE_LEVELS else "INFO"

    @staticmethod
    def _format_ts(created: float) -> str:
        """Format a Unix timestamp as ISO 8601 UTC with milliseconds."""

        dt = time.gmtime(created)
        ms = int((created % 1) * 1000)
        return (
            f"{dt.tm_year:04d}-{dt.tm_mon:02d}-{dt.tm_mday:02d}"
            f"T{dt.tm_hour:02d}:{dt.tm_min:02d}:{dt.tm_sec:02d}"
            f".{ms:03d}Z"
        )


__all__ = ["TracelabLogHandler"]
