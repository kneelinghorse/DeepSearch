"""S91 m05 service-auth and acknowledged mission-log contract tests."""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import httpx
import pytest

from deepsearch.tracelab.auth import TraceLabAuth, TraceLabAuthError
from deepsearch.tracelab.log_handler import TracelabLogHandler
from deepsearch.tracelab.uploader import TraceLabUploader
from deepsearch.worker.lease import LeaseAttempt


def _lease() -> LeaseAttempt:
    now = datetime.now(timezone.utc)
    return LeaseAttempt("worker-fixture", "lease-fixture", now, now, now, 1)


class _AuthStub:
    def __init__(self, base_url: str = "https://tracelab.example") -> None:
        self.base_url = base_url
        self.role_checks = 0

    def require_service_role(self) -> None:
        self.role_checks += 1

    def get_headers(self) -> dict[str, str]:
        return {"X-API-Key": "tl_service_test"}


class _Response:
    def __init__(self, status_code: int, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.body = body or {}

    def json(self) -> dict[str, Any]:
        return self.body


class _Client:
    def __init__(self, posts: list[dict[str, Any]], statuses: list[int]) -> None:
        self._posts = posts
        self._statuses = statuses

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, url: str, *, headers: dict[str, str]) -> _Response:
        assert url.endswith("/logs/capabilities")
        return _Response(200, {"contract_version": "tracelab-mission-logs-v2",
                               "max_batch_entries": 200, "max_message_chars": 2048,
                               "final_flush": "terminal_result_key",
                               "legacy": "terminal_only_until_LOG-2_cutover"})

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> _Response:
        self._posts.append({"url": url, "json": json, "headers": headers})
        return _Response(self._statuses.pop(0), {
            "contract_version": json["contract_version"], "mission_id": json["mission_id"],
            "attempt_count": json["attempt_count"], "accepted": len(json["logs"]),
            "replayed": 0, "event_ids": [entry["event_id"] for entry in json["logs"]],
        })


def _record(
    message: str,
    *,
    created: float = 1_723_654_496.123,
    event: str | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="deepsearch.agent",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.created = created
    if event is not None:
        record.tracelab_event = event
    return record


def _safe_message(
    *,
    phase: str = "running",
    category: str = "info",
    event: str = "mission_progress",
    **numeric: int | float,
) -> str:
    return json.dumps(
        {"category": category, "event": event, "phase": phase, **numeric},
        separators=(",", ":"),
        sort_keys=True,
    )


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    posts: list[dict[str, Any]],
    statuses: list[int],
) -> None:
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **_kwargs: _Client(posts, statuses),
    )


def _handler(auth: _AuthStub, **kwargs: Any) -> TracelabLogHandler:
    return TracelabLogHandler(
        lease=_lease(),
        mission_id="11111111-2222-4333-8444-555555555555",
        base_url="https://tracelab.example",
        flush_threshold=100,
        flush_interval=3600,
        auth=auth,  # type: ignore[arg-type]
        **kwargs,
    )


def test_api_key_is_preferred_over_jwt_credentials() -> None:
    """A provisioned service key must avoid the human login path entirely."""

    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="tl_service_key",
        username="fallback@example.com",
        password="fallback-password",
    )

    assert auth.get_headers() == {"X-API-Key": "tl_service_key"}
    assert auth.uses_api_key is True


def test_external_plain_http_never_exposes_service_credentials() -> None:
    """Only loopback development may use HTTP for TraceLab credentials."""

    external = TraceLabAuth(
        base_url="http://tracelab.example",
        api_key="tl_service_key",
    )
    local = TraceLabAuth(
        base_url="http://127.0.0.1:8000",
        api_key="tl_service_key",
    )
    with pytest.raises(TraceLabAuthError, match="require HTTPS"):
        external.get_headers()
    for deceptive_url in (
        "http://127.example.com",
        "http://127.0.0.1.evil.test",
    ):
        deceptive = TraceLabAuth(
            base_url=deceptive_url,
            api_key="tl_service_key",
        )
        with pytest.raises(TraceLabAuthError, match="require HTTPS"):
            deceptive.get_headers()
    assert local.get_headers() == {"X-API-Key": "tl_service_key"}


def test_injected_auth_owns_the_log_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified principal cannot be paired with a different env destination."""

    monkeypatch.setenv("TRACELAB_BASE_URL", "https://wrong.example")
    handler = TracelabLogHandler(
        lease=_lease(),
        mission_id="mission-row-uuid",
        auth=_AuthStub(base_url="https://verified.example"),  # type: ignore[arg-type]
    )

    assert handler.base_url == "https://verified.example"
    handler.base_url = ""
    handler.close()


def test_injected_auth_rejects_explicit_destination_override() -> None:
    """An explicit URL cannot redirect a verified credential to another host."""

    handler = TracelabLogHandler(
        lease=_lease(),
        mission_id="mission-row-uuid",
        base_url="https://attacker.example",
        auth=_AuthStub(base_url="https://verified.example"),  # type: ignore[arg-type]
    )

    assert handler.base_url == "https://verified.example"
    handler.base_url = ""
    handler.close()

    auth = TraceLabAuth(
        base_url="https://verified.example",
        api_key="tl_service_key",
    )
    uploader = TraceLabUploader(
        base_url="https://attacker.example",
        auth=auth,
    )
    assert uploader.base_url == "https://verified.example"


def test_service_role_validation_uses_auth_me_and_caches_success() -> None:
    """Log authorization is proven from TraceLab's live role, not key shape."""

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"role": "service"}
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="tl_service_key",
    )

    with patch("httpx.get", return_value=response) as get:
        auth.require_service_role()
        auth.require_service_role()

    get.assert_called_once_with(
        "https://tracelab.example/api/v1/auth/me",
        headers={"X-API-Key": "tl_service_key"},
        timeout=30.0,
    )


def test_service_role_validation_rejects_human_principal() -> None:
    """A valid human credential still cannot be used to append mission logs."""

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"role": "owner"}
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="tl_owner_key",
    )

    with patch("httpx.get", return_value=response):
        with pytest.raises(TraceLabAuthError, match="not 'service'"):
            auth.require_service_role()


def test_service_role_validation_rejects_non_object_json() -> None:
    """Malformed auth/me JSON must become a typed unhealthy result."""

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = ["service"]
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="tl_service_key",
    )

    with patch("httpx.get", return_value=response):
        with pytest.raises(TraceLabAuthError, match="non-object JSON"):
            auth.require_service_role()


def test_jwt_fallback_sends_tracelab_email_field() -> None:
    """TraceLab's login schema names the identity field ``email``."""

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "access_token": "token",  # secret-scan: allow -- authentication fixture
        "expires_in": 3600,
    }
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="",
        username="service@example.com",
        password="password",
    )

    with patch("httpx.post", return_value=response) as post:
        assert auth.get_headers() == {"Authorization": "Bearer token"}

    post.assert_called_once_with(
        "https://tracelab.example/api/v1/auth/login",
        json={
            "email": "service@example.com",
            "password": "password",  # secret-scan: allow -- authentication fixture
        },
        timeout=30.0,
    )


def test_acknowledged_batch_uses_canonical_wire_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[dict[str, Any]] = []
    statuses = [201]
    _install_client(monkeypatch, posts, statuses)
    auth = _AuthStub()
    handler = _handler(auth)
    try:
        handler.set_phase("critique")
        handler.emit(_record("checking citations"))
        handler.flush()

        event_id = posts[0]["json"]["logs"][0]["event_id"]
        UUID(event_id)
        assert posts == [
            {
                "url": (
                    "https://tracelab.example/api/v1/missions/"
                    "11111111-2222-4333-8444-555555555555/logs/v2"
                ),
                "json": {
                    "contract_version": "tracelab-mission-logs-v2",
                    "mission_id": handler.mission_id,
                    "attempt_count": 1,
                    "lease_owner": "worker-fixture",
                    "lease_token": "lease-fixture",
                    "logs": [
                        {
                            "event_id": event_id,
                            "sequence": 1,
                            "logged_at": "2024-08-14T16:54:56.122Z",
                            "level": "INFO",
                            "source": "deepsearch:critique",
                            "message": _safe_message(phase="critique"),
                        }
                    ]
                },
                "headers": {"X-API-Key": "tl_service_test"},
            }
        ]
        assert handler.get_telemetry()["pending_entries"] == 0
        assert handler.get_telemetry()["accepted_entries"] == 1
        assert auth.role_checks == 1
    finally:
        handler.close()


def test_outbound_batch_projects_only_allowlisted_identity_free_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authored text, identities, URLs, args, and tracebacks never leave."""

    posts: list[dict[str, Any]] = []
    _install_client(monkeypatch, posts, [201])
    handler = _handler(_AuthStub())
    mission_marker = "mission-sensitive-92"
    job_marker = "job-sensitive-92"
    title_marker = "AUTHORED-TITLE-SENSITIVE-92"
    body_marker = "AUTHORED-BODY-SENSITIVE-92"
    signed_url = (
        "https://api-user:"
        "api-password@signed.example/report?token=QUERY-SENSITIVE-92"
    )
    try:
        raise RuntimeError(f"{signed_url} {body_marker}")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="deepsearch.worker.main",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="mission=%s job=%s title=%s url=%s body=%s",
        args=(mission_marker, job_marker, title_marker, signed_url, body_marker),
        exc_info=exc_info,
    )
    record.stack_info = f"traceback {signed_url} {body_marker}"
    record.mission_id = mission_marker
    record.job_id = job_marker
    record.title = title_marker
    record.diagnostic = {"url": signed_url, "body": body_marker}
    record.tracelab_event = title_marker
    record.tracelab_category = body_marker
    record.tracelab_count = 7
    record.tracelab_duration_seconds = 1.25

    try:
        handler.set_phase("error")
        handler.emit(record)
        handler.flush()

        payload = posts[0]["json"]
        entry = payload["logs"][0]
        assert entry == {
            "event_id": entry["event_id"],
            "sequence": 1,
            "logged_at": handler._format_ts(record.created),
            "level": "ERROR",
            "source": "deepsearch:error",
            "message": _safe_message(
                phase="error",
                category="error",
                event="mission_error",
                count=7,
                duration_seconds=1.25,
            ),
        }
        serialized = json.dumps(payload, sort_keys=True)
        for forbidden in (
            mission_marker,
            job_marker,
            title_marker,
            body_marker,
            signed_url,
            "api-user",
            "api-password",
            "QUERY-SENSITIVE-92",
            "RuntimeError",
            "traceback",
        ):
            assert forbidden not in serialized
    finally:
        handler.close()


def test_failed_batch_is_retained_until_later_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-2xx response cannot silently delete the only copy of a batch."""

    posts: list[dict[str, Any]] = []
    statuses = [503, 201]
    _install_client(monkeypatch, posts, statuses)
    handler = _handler(_AuthStub(), max_attempts=1)
    try:
        handler.emit(_record("retain me"))
        handler.flush()

        failed = handler.get_telemetry()
        assert failed["pending_entries"] == 1
        assert failed["failed_batches"] == 1
        assert failed["last_status_code"] == 503

        handler.flush()
        recovered = handler.get_telemetry()
        assert recovered["pending_entries"] == 0
        assert recovered["accepted_entries"] == 1
        assert [post["json"]["logs"][0]["message"] for post in posts] == [
            _safe_message(),
            _safe_message(),
        ]
    finally:
        handler.close()


def test_batch_limit_preserves_order_without_dropping_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[dict[str, Any]] = []
    statuses = [201, 201]
    _install_client(monkeypatch, posts, statuses)
    handler = _handler(_AuthStub(), max_buffer=1)
    try:
        handler.emit(_record("first", event="phase_started"))
        handler.emit(_record("second", event="phase_completed"))
        handler.flush()

        assert [post["json"]["logs"][0]["message"] for post in posts] == [
            _safe_message(event="phase_started"),
            _safe_message(event="phase_completed"),
        ]
        assert handler.get_telemetry()["pending_entries"] == 0
    finally:
        handler.close()


def test_queue_capacity_preserves_newest_order_and_reports_overflow() -> None:
    """A TraceLab outage cannot grow mission-log memory without a bound."""

    handler = _handler(
        _AuthStub(),
        max_buffer=1,
        max_pending_entries=2,
    )
    try:
        with patch.object(handler, "_drain_async") as drain:
            handler.emit(_record("first", event="phase_started"))
            handler.emit(_record("second", event="mission_progress"))
            handler.emit(_record("overflow", event="phase_completed"))

        telemetry = handler.get_telemetry()
        assert telemetry["pending_entries"] == 2
        assert telemetry["overflowed_entries"] == 1
        assert telemetry["last_error"] == (
            "log queue capacity exceeded; oldest pending entry was evicted"
        )
        assert [entry["message"] for entry in handler._buffer] == [
            _safe_message(event="mission_progress"),
            _safe_message(event="phase_completed"),
        ]
        assert drain.call_count == 0  # Explicit activation requires ownership proof.
        assert [entry["sequence"] for entry in handler._buffer] == [2, 3]
    finally:
        handler.base_url = ""
        handler.close()


def test_overflow_never_deletes_an_unsent_entry_after_inflight_ack() -> None:
    """An ACK removes its exact batch even when concurrent emit evicts another row."""

    handler = _handler(
        _AuthStub(),
        max_buffer=1,
        max_pending_entries=2,
    )
    post_started = threading.Event()
    release_post = threading.Event()
    posted_messages: list[list[str]] = []

    def _post(entries: list[dict[str, Any]]) -> bool:
        posted_messages.append([entry["message"] for entry in entries])
        if len(posted_messages) == 1:
            post_started.set()
            assert release_post.wait(timeout=2.0)
            return True
        return False

    try:
        with patch.object(handler, "_drain_async"):
            handler.emit(_record("first", event="phase_started"))
            handler.emit(_record("second", event="mission_warning"))
        with patch.object(handler, "_post_entries", side_effect=_post):
            flusher = threading.Thread(target=handler.flush)
            flusher.start()
            assert post_started.wait(timeout=2.0)
            with patch.object(handler, "_drain_async"):
                handler.emit(_record("newest", event="phase_completed"))
            release_post.set()
            flusher.join(timeout=2.0)

        assert not flusher.is_alive()
        assert posted_messages == [
            [_safe_message(event="phase_started")],
            [_safe_message(event="phase_completed")],
        ]
        assert [entry["message"] for entry in handler._buffer] == [
            _safe_message(event="phase_completed")
        ]
        assert handler.get_telemetry()["overflowed_entries"] == 1
    finally:
        handler.base_url = ""
        handler.close()


def test_close_waits_for_inflight_emit_then_drains_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close must not report an empty queue while an emitter can still append."""

    posts: list[dict[str, Any]] = []
    statuses = [201]
    _install_client(monkeypatch, posts, statuses)
    handler = _handler(_AuthStub())
    formatting_started = threading.Event()
    release_formatting = threading.Event()
    original_format = handler._format_ts

    def _blocking_format(created: float) -> str:
        formatting_started.set()
        assert release_formatting.wait(timeout=2.0)
        return original_format(created)

    monkeypatch.setattr(handler, "_format_ts", _blocking_format)
    emitter = threading.Thread(target=handler.emit, args=(_record("in flight"),))
    emitter.start()
    assert formatting_started.wait(timeout=2.0)
    closer = threading.Thread(target=handler.close)
    closer.start()
    release_formatting.set()
    emitter.join(timeout=2.0)
    closer.join(timeout=2.0)

    assert not emitter.is_alive()
    assert not closer.is_alive()
    assert posts[0]["json"]["logs"][0]["message"] == _safe_message()
    assert handler.get_telemetry()["pending_entries"] == 0


def test_nonretryable_status_is_visible_and_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[dict[str, Any]] = []
    statuses = [403, 403]
    _install_client(monkeypatch, posts, statuses)
    handler = _handler(_AuthStub(), max_attempts=3)
    handler.emit(_record("unauthorized batch"))
    handler.flush()

    telemetry = handler.get_telemetry()
    assert len(posts) == 1
    assert telemetry["pending_entries"] == 1
    assert telemetry["last_status_code"] == 403
    assert telemetry["last_error"] == "TraceLab returned HTTP 403"

    # Prevent test teardown from making a second delivery attempt for the
    # intentionally retained batch.
    handler.base_url = ""
    handler.close()


def test_worker_buffer_can_discard_without_any_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unfinished lease drops buffered logs before successor recovery."""

    posts: list[dict[str, Any]] = []
    _install_client(monkeypatch, posts, [200])
    handler = TracelabLogHandler(
        lease=_lease(),
        mission_id="mission-row-uuid",
        base_url="https://tracelab.example",
        flush_threshold=1,
        flush_interval=0.1,
        auth=_AuthStub(),  # type: ignore[arg-type]
    )

    handler.emit(_record("attempt-local only"))
    handler.discard()

    telemetry = handler.get_telemetry()
    assert posts == []
    assert telemetry["pending_entries"] == 0
    assert telemetry["discarded_entries"] == 1
