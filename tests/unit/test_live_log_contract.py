"""Live observations need receiver proof and complete acknowledgements, not 2xx alone."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from deepsearch.tracelab.log_handler import TracelabLogHandler
from deepsearch.worker.lease import LeaseAttempt

WIRE = json.loads((Path(__file__).parents[1] / "fixtures/mission-logs-v2/batch.json").read_text())
CAPABILITIES = {
    "contract_version": "tracelab-mission-logs-v2", "max_batch_entries": 200,
    "max_message_chars": 2048, "final_flush": "terminal_result_key",
    "legacy": "terminal_only_until_LOG-2_cutover",
}


class ServiceAuth:
    base_url = "https://receiver.example"

    def require_service_role(self) -> None:
        pass

    def get_headers(self) -> dict[str, str]:
        return {"X-API-Key": "controlled-service-fixture"}


def owned_handler() -> TracelabLogHandler:
    now = datetime.now(timezone.utc)
    lease = LeaseAttempt(
        WIRE["lease_owner"], WIRE["lease_token"], now, now, now, WIRE["attempt_count"]
    )
    return TracelabLogHandler(WIRE["mission_id"], lease=lease, auth=ServiceAuth(),
                             flush_interval=3600, flush_threshold=100, backoff_base=0)


def observe(handler: TracelabLogHandler) -> None:
    record = logging.LogRecord("deepsearch.agent", logging.INFO, "", 0,
                               "Private prompt must never cross the boundary", (), None)
    record.tracelab_event = "phase_started"
    handler.emit(record)


def install_transport(monkeypatch: pytest.MonkeyPatch, serve) -> None:
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs)
    )


def acknowledgement(body: dict, accepted: int = 1, replayed: int = 0) -> dict:
    return {"contract_version": body["contract_version"], "mission_id": body["mission_id"],
            "attempt_count": body["attempt_count"], "accepted": accepted, "replayed": replayed,
            "event_ids": [row["event_id"] for row in body["logs"]]}


def test_lost_ack_reuses_the_exact_attempt_event_and_never_falls_back(monkeypatch):
    posted = []
    def serve(request):
        if request.method == "GET":
            return httpx.Response(200, json=CAPABILITIES)
        assert request.url.path.endswith("/logs/v2")
        body = json.loads(request.content)
        posted.append(body)
        if len(posted) == 1:
            raise httpx.ReadTimeout("lost response after receiver commit", request=request)
        return httpx.Response(200, json=acknowledgement(body, 0, 1))
    install_transport(monkeypatch, serve)
    handler = owned_handler()
    try:
        observe(handler)
        assert handler.start_live_delivery()
        handler.flush()
        assert len(posted) == 2 and posted[0] == posted[1]
        assert posted[0]["attempt_count"] == WIRE["attempt_count"]
        assert posted[0]["lease_token"] == WIRE["lease_token"]
        assert posted[0]["logs"][0]["sequence"] == 1
        assert "Private prompt" not in json.dumps(posted)
        assert handler.get_telemetry()["pending_entries"] == 0
        assert WIRE["lease_token"] not in json.dumps(handler.get_telemetry())
    finally:
        handler.discard()


@pytest.mark.parametrize("capabilities", [{}, {**CAPABILITIES, "contract_version": "old"}])
def test_old_or_unversioned_receiver_cannot_enable_delivery(monkeypatch, capabilities):
    requests = []
    def serve(request):
        requests.append(request.method)
        return httpx.Response(200, json=capabilities)
    install_transport(monkeypatch, serve)
    handler = owned_handler()
    try:
        observe(handler)
        assert not handler.start_live_delivery()
        handler.flush()
        assert set(requests) == {"GET"}
        assert handler.get_telemetry()["pending_entries"] == 1
    finally:
        handler.discard()


@pytest.mark.parametrize("change", ["marker", "mission", "attempt", "count", "ids", "bool_count"])
def test_incomplete_or_mismatched_2xx_never_dequeues_observations(monkeypatch, change):
    def serve(request):
        if request.method == "GET":
            return httpx.Response(200, json=CAPABILITIES)
        body = json.loads(request.content)
        ack = acknowledgement(body)
        key, value = {"marker": ("contract_version", "old"), "mission": ("mission_id", "other"),
                      "attempt": ("attempt_count", 2), "count": ("accepted", 0),
                      "ids": ("event_ids", []), "bool_count": ("accepted", True)}[change]
        ack[key] = value
        return httpx.Response(201, json=ack)
    install_transport(monkeypatch, serve)
    handler = owned_handler()
    try:
        observe(handler)
        handler.flush()
        assert handler.get_telemetry()["pending_entries"] == 1
        assert handler.get_telemetry()["accepted_entries"] == 0
    finally:
        handler.discard()


def test_conflict_stops_this_attempt_and_does_not_relabel(monkeypatch):
    posts = []
    def serve(request):
        if request.method == "GET":
            return httpx.Response(200, json=CAPABILITIES)
        posts.append(json.loads(request.content))
        return httpx.Response(409, json={"detail": "ownership lost"})
    install_transport(monkeypatch, serve)
    handler = owned_handler()
    observe(handler)
    handler.flush()
    observe(handler)
    handler.flush()
    handler.discard()
    assert len(posts) == 1
    assert handler.get_telemetry()["discarded_entries"] == 1
    assert handler.get_telemetry()["pending_entries"] == 0


def test_discard_waits_for_inflight_ack_but_stops_new_requests(monkeypatch):
    import threading
    started, release, discarded = threading.Event(), threading.Event(), threading.Event()
    posts = []
    def serve(request):
        if request.method == "GET":
            return httpx.Response(200, json=CAPABILITIES)
        body = json.loads(request.content)
        posts.append(body)
        started.set()
        assert release.wait(2)
        return httpx.Response(201, json=acknowledgement(body))
    install_transport(monkeypatch, serve)
    handler = owned_handler()
    handler.max_batch_size = 1
    observe(handler)
    observe(handler)
    assert handler.start_live_delivery()
    assert started.wait(2)
    handler.stop_delivery()
    def discard():
        handler.discard()
        discarded.set()
    cleanup = threading.Thread(target=discard)
    cleanup.start()
    assert not discarded.wait(0.05), "Lease release must wait for the in-flight write"
    release.set()
    cleanup.join(2)
    assert discarded.is_set()
    assert len(posts) == 1
    assert handler.get_telemetry()["accepted_entries"] == 1
    assert handler.get_telemetry()["discarded_entries"] == 1


def test_stop_interrupts_backoff_without_waiting_for_the_retry_budget(monkeypatch):
    import threading
    posted = threading.Event()
    requests = []
    def serve(request):
        if request.method == "GET":
            return httpx.Response(200, json=CAPABILITIES)
        requests.append(request)
        posted.set()
        return httpx.Response(503)
    install_transport(monkeypatch, serve)
    handler = owned_handler()
    handler.backoff_base = 60
    observe(handler)
    assert handler.start_live_delivery()
    assert posted.wait(2)
    cleanup = threading.Thread(target=handler.discard)
    cleanup.start()
    cleanup.join(2)
    assert not cleanup.is_alive()
    assert len(requests) == 1


def test_child_pipe_preserves_only_allowlisted_observations(monkeypatch):
    from deepsearch.worker.execution import _forward_log, _PipeLogHandler
    messages = []
    class Connection:
        def send(self, message):
            messages.append(message)
    child = _PipeLogHandler(Connection())
    record = logging.LogRecord("deepsearch.child", logging.INFO, "", 0, "internal text", (), None)
    record.tracelab_event = "phase_started"
    record.tracelab_phase = "critique"
    record.tracelab_count = 3
    record.tracelab_duration_seconds = float("nan")
    record.tracelab_category = "private-title"
    record.tracelab_secret = "must not forward"
    child.emit(record)
    assert messages[0]["observation"] == {
        "tracelab_event": "phase_started", "tracelab_phase": "critique", "tracelab_count": 3,
    }
    messages[0]["observation"]["tracelab_secret"] = "injected"
    parent = owned_handler()
    logger = logging.getLogger("deepsearch.child")
    monkeypatch.setattr(logger, "handle", parent.emit)
    try:
        _forward_log(messages[0])
        entry = parent._buffer[0]
        assert entry["source"] == "deepsearch:critique"
        assert json.loads(entry["message"]) == {
            "category": "info", "event": "phase_started", "phase": "critique", "count": 3,
        }
    finally:
        parent.discard()
