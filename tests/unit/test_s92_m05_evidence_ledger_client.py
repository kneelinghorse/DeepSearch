"""S92 m05 trigger-only TraceLab Evidence Ledger client invariants."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

import httpx
import pytest

from deepsearch.tracelab import (
    EvidenceLedgerClient,
    EvidenceLedgerDeliveryResult,
    EvidenceLedgerTrigger,
)
from deepsearch.tracelab.auth import TraceLabAuth, TraceLabAuthError

_MISSION_UUID = "11111111-2222-4333-8444-555555555555"


class _AuthStub:
    def __init__(
        self,
        *,
        base_url: str = "https://tracelab.example",
        configured: bool = True,
        uses_api_key: bool = True,
        role_outcomes: list[Exception | None] | None = None,
        api_key: str = "tl_service_test",
    ) -> None:
        self.base_url = base_url
        self.is_configured = configured
        self.uses_api_key = uses_api_key
        self.role_outcomes = list(role_outcomes or [])
        self.api_key = api_key
        self.role_checks = 0

    def require_service_role(self) -> None:
        self.role_checks += 1
        if self.role_outcomes:
            outcome = self.role_outcomes.pop(0)
            if outcome is not None:
                raise outcome

    def get_headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    @property
    def text(self) -> str:
        raise AssertionError("Evidence Ledger client must not read response text")

    def json(self) -> dict[str, Any]:
        raise AssertionError("Evidence Ledger client must not read response JSON")


class _AsyncClient:
    def __init__(
        self,
        requests: list[dict[str, Any]],
        outcomes: list[int | Exception],
    ) -> None:
        self._requests = requests
        self._outcomes = outcomes

    async def __aenter__(self) -> _AsyncClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
    ) -> _Response:
        self._requests.append({"url": url, "content": content, "headers": headers})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[dict[str, Any]],
    outcomes: list[int | Exception],
) -> list[dict[str, Any]]:
    constructor_calls: list[dict[str, Any]] = []

    def _client(**kwargs: Any) -> _AsyncClient:
        constructor_calls.append(kwargs)
        return _AsyncClient(requests, outcomes)

    monkeypatch.setattr(httpx, "AsyncClient", _client)
    return constructor_calls


def _trigger(
    *,
    mission_uuid: str = _MISSION_UUID,
    job_id: str = "job-123",
) -> EvidenceLedgerTrigger:
    return EvidenceLedgerTrigger(
        mission_uuid=mission_uuid,
        deepsearch_job_id=job_id,
    )


@pytest.mark.asyncio
async def test_trigger_body_and_route_are_frozen_and_minimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    constructor_calls = _install_client(monkeypatch, requests, [201])
    auth = _AuthStub()
    monkeypatch.setenv("TRACELAB_DEEPSEARCH_SERVICE_SECRET", "hmac-must-not-be-used")
    client = EvidenceLedgerClient(auth=auth)  # type: ignore[arg-type]

    result = await client.deliver(
        _trigger(mission_uuid="11111111-2222-4333-8444-555555555555")
    )

    assert result == EvidenceLedgerDeliveryResult(
        success=True,
        attempts=1,
        retryable=False,
        category="accepted",
        final_status_code=201,
    )
    assert constructor_calls == [{"timeout": 10.0, "follow_redirects": False}]
    assert auth.role_checks == 1
    assert requests == [
        {
            "url": (
                "https://tracelab.example/api/v1/missions/"
                "11111111-2222-4333-8444-555555555555/evidence"
            ),
            "content": b'{"schema_version":1,"deepsearch_job_id":"job-123"}',
            "headers": {
                "X-API-Key": "tl_service_test",
                "Content-Type": "application/json",
            },
        }
    ]
    decoded = json.loads(requests[0]["content"])
    assert set(decoded) == {"schema_version", "deepsearch_job_id"}
    assert all(
        forbidden not in requests[0]["content"]
        for forbidden in (
            b"result_protocol",
            b"result_markdown",
            b"citation",
            b"evidence",
            b"mission_id",
            b"status",
        )
    )
    assert "X-DeepSearch-Signature" not in requests[0]["headers"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mission_uuid",
    [
        "{aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee}",
        "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE",
        "aaaaaaaabbbb4ccc8dddeeeeeeeeeeee",
        " aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee ",
    ],
)
async def test_noncanonical_uuid_input_is_rejected_without_normalization(
    monkeypatch: pytest.MonkeyPatch,
    mission_uuid: str,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    auth = _AuthStub()
    client = EvidenceLedgerClient(auth=auth)  # type: ignore[arg-type]

    result = await client.deliver(_trigger(mission_uuid=mission_uuid))

    assert result.category == "invalid_mission_uuid"
    assert result.attempts == 0
    assert auth.role_checks == 0
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 201, 202, 204, 299])
async def test_any_2xx_is_an_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [status_code])
    client = EvidenceLedgerClient(auth=_AuthStub())  # type: ignore[arg-type]

    result = await client.deliver(_trigger())

    assert result.success is True
    assert result.retryable is False
    assert result.category == "accepted"
    assert result.final_status_code == status_code
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_429_and_5xx_retry_with_bounded_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [429, 503, 204])
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("deepsearch.tracelab.evidence_ledger.asyncio.sleep", _sleep)
    client = EvidenceLedgerClient(
        auth=_AuthStub(),  # type: ignore[arg-type]
        max_attempts=4,
        backoff_base=0.25,
    )

    result = await client.deliver(_trigger())

    assert result.success is True
    assert result.attempts == 3
    assert len(requests) == 3
    assert sleeps == [0.25, 0.5]


@pytest.mark.asyncio
async def test_transport_errors_retry_and_remain_redacted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[dict[str, Any]] = []
    request = httpx.Request("POST", "https://tracelab.example")
    raw_detail = "offline-with-sensitive-network-detail"
    _install_client(
        monkeypatch,
        requests,
        [httpx.ConnectError(raw_detail, request=request)] * 2,
    )

    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("deepsearch.tracelab.evidence_ledger.asyncio.sleep", _sleep)
    client = EvidenceLedgerClient(
        auth=_AuthStub(api_key="secret-api-key"),  # type: ignore[arg-type]
        max_attempts=2,
    )
    trigger = _trigger(job_id="sensitive-job-id")

    with caplog.at_level(logging.ERROR, logger="tracelab.evidence_ledger"):
        result = await client.deliver(trigger)

    assert result == EvidenceLedgerDeliveryResult(
        success=False,
        attempts=2,
        retryable=True,
        category="transport_error",
        final_status_code=None,
    )
    assert len(requests) == 2
    serialized = json.dumps(asdict(result), sort_keys=True) + caplog.text
    assert raw_detail not in serialized
    assert "secret-api-key" not in serialized
    assert "sensitive-job-id" not in serialized
    assert _MISSION_UUID not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (401, "authentication"),
        (403, "authorization"),
        (409, "identity_mismatch"),
        (400, "client_rejected"),
        (404, "client_rejected"),
        (422, "client_rejected"),
        (307, "redirect_rejected"),
    ],
)
async def test_permanent_responses_are_never_retried(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    category: str,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [status_code, 200])
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("deepsearch.tracelab.evidence_ledger.asyncio.sleep", _sleep)
    client = EvidenceLedgerClient(
        auth=_AuthStub(),  # type: ignore[arg-type]
        max_attempts=4,
    )

    result = await client.deliver(_trigger())

    assert result.success is False
    assert result.attempts == 1
    assert result.retryable is False
    assert result.category == category
    assert result.final_status_code == status_code
    assert len(requests) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_exhausted_server_failures_stay_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [502, 503])

    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("deepsearch.tracelab.evidence_ledger.asyncio.sleep", _sleep)
    client = EvidenceLedgerClient(
        auth=_AuthStub(),  # type: ignore[arg-type]
        max_attempts=2,
    )

    result = await client.deliver(_trigger())

    assert result == EvidenceLedgerDeliveryResult(
        success=False,
        attempts=2,
        retryable=True,
        category="server_error",
        final_status_code=503,
    )


@pytest.mark.asyncio
async def test_service_role_transport_failure_uses_the_same_retry_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "https://tracelab.example/api/v1/auth/me")
    cause = httpx.ConnectError("raw-auth-network-detail", request=request)
    auth_error = TraceLabAuthError("sanitized auth request failed")
    auth_error.__cause__ = cause
    auth = _AuthStub(role_outcomes=[auth_error, None])
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("deepsearch.tracelab.evidence_ledger.asyncio.sleep", _sleep)
    client = EvidenceLedgerClient(
        auth=auth,  # type: ignore[arg-type]
        backoff_base=0.5,
    )

    result = await client.deliver(_trigger())

    assert result.success is True
    assert result.attempts == 2
    assert auth.role_checks == 2
    assert len(requests) == 1
    assert sleeps == [0.5]


@pytest.mark.asyncio
async def test_human_role_is_rejected_before_ledger_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _AuthStub(
        role_outcomes=[
            TraceLabAuthError(
                "TraceLab credential is not service role",
                status_code=403,
            )
        ]
    )
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    client = EvidenceLedgerClient(auth=auth)  # type: ignore[arg-type]

    result = await client.deliver(_trigger())

    assert result.category == "authorization"
    assert result.retryable is False
    assert result.attempts == 1
    assert auth.role_checks == 1
    assert requests == []


@pytest.mark.asyncio
async def test_jwt_fallback_is_rejected_without_login_or_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="",
        username="human@example.com",
        password="human-password",
    )
    role_check = pytest.MonkeyPatch()
    role_check.setattr(auth, "require_service_role", lambda: None)
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    client = EvidenceLedgerClient(auth=auth)

    try:
        result = await client.deliver(_trigger())
    finally:
        role_check.undo()

    assert result == EvidenceLedgerDeliveryResult(
        success=False,
        attempts=0,
        retryable=False,
        category="api_key_required",
    )
    assert requests == []


@pytest.mark.asyncio
async def test_api_key_is_preferred_without_touching_jwt_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="tl_service_key",
        username="fallback@example.com",
        password="fallback-password",
    )
    monkeypatch.setattr(auth, "require_service_role", lambda: None)
    login = pytest.fail
    monkeypatch.setattr(auth, "_login", login)
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])

    result = await EvidenceLedgerClient(auth=auth).deliver(_trigger())

    assert result.success is True
    assert requests[0]["headers"] == {
        "X-API-Key": "tl_service_key",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://tracelab.example",
        "http://127.example.com",
        "http://127.0.0.1.evil.example",
        "https://user:password@tracelab.example",
        "https://tracelab.example/api",
        "https://tracelab.example?next=https://attacker.example",
        "https://tracelab.example#fragment",
        "https://tracelab.example:invalid",
    ],
)
async def test_invalid_or_non_origin_destinations_make_zero_auth_or_post_attempts(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
) -> None:
    auth = _AuthStub(base_url=base_url)
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    client = EvidenceLedgerClient(auth=auth)  # type: ignore[arg-type]

    result = await client.deliver(_trigger())

    assert result.category == "invalid_destination"
    assert result.attempts == 0
    assert result.retryable is False
    assert auth.role_checks == 0
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000",
        "http://service.localhost:8000",
        "http://127.0.0.1:8000",
        "http://[::1]:8000",
    ],
)
async def test_explicit_loopback_origins_retain_development_policy(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [204])
    client = EvidenceLedgerClient(
        auth=_AuthStub(base_url=base_url),  # type: ignore[arg-type]
    )

    result = await client.deliver(_trigger())

    assert result.success is True
    assert requests[0]["url"].startswith(f"{base_url}/api/v1/missions/")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "category"),
    [
        (_trigger(mission_uuid="not-a-uuid"), "invalid_mission_uuid"),
        (_trigger(mission_uuid="../../auth/me"), "invalid_mission_uuid"),
        (_trigger(job_id=""), "invalid_job_id"),
        (_trigger(job_id=" surrounded "), "invalid_job_id"),
        (_trigger(job_id="x" * 101), "invalid_job_id"),
    ],
)
async def test_invalid_identities_make_zero_auth_or_post_attempts(
    monkeypatch: pytest.MonkeyPatch,
    trigger: EvidenceLedgerTrigger,
    category: str,
) -> None:
    auth = _AuthStub()
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    client = EvidenceLedgerClient(auth=auth)  # type: ignore[arg-type]

    result = await client.deliver(trigger)

    assert result.category == category
    assert result.attempts == 0
    assert auth.role_checks == 0
    assert requests == []


@pytest.mark.asyncio
async def test_missing_auth_configuration_is_visible_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = _AuthStub(configured=False)
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    client = EvidenceLedgerClient(auth=auth)  # type: ignore[arg-type]

    result = await client.deliver(_trigger())

    assert result == EvidenceLedgerDeliveryResult(
        success=False,
        attempts=0,
        retryable=False,
        category="not_configured",
    )
    assert client.is_configured is False
    assert auth.role_checks == 0
    assert requests == []


def test_client_has_no_independent_destination_or_endpoint_override() -> None:
    with pytest.raises(TypeError, match="base_url"):
        EvidenceLedgerClient(  # type: ignore[call-arg]
            auth=_AuthStub(),  # type: ignore[arg-type]
            base_url="https://attacker.example",
        )


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("inf"), float("-inf"), float("nan"), 30.01, True],
)
def test_timeout_must_be_positive_finite_and_bounded(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        EvidenceLedgerClient(
            auth=_AuthStub(),  # type: ignore[arg-type]
            timeout=timeout,
        )


@pytest.mark.parametrize("max_attempts", [0, -1, 5, 1_000_000, True, 1.5])
def test_attempt_count_must_be_an_integer_within_the_hard_cap(
    max_attempts: Any,
) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        EvidenceLedgerClient(
            auth=_AuthStub(),  # type: ignore[arg-type]
            max_attempts=max_attempts,
        )


@pytest.mark.parametrize(
    "backoff_base",
    [-0.01, float("inf"), float("-inf"), float("nan"), 5.01, 1_000_000, True],
)
def test_backoff_must_be_nonnegative_finite_and_bounded(
    backoff_base: float,
) -> None:
    with pytest.raises(ValueError, match="backoff_base"):
        EvidenceLedgerClient(
            auth=_AuthStub(),  # type: ignore[arg-type]
            backoff_base=backoff_base,
        )


def test_maximum_transport_bounds_are_accepted_exactly() -> None:
    client = EvidenceLedgerClient(
        auth=_AuthStub(),  # type: ignore[arg-type]
        timeout=30,
        max_attempts=4,
        backoff_base=5,
    )

    assert client.timeout == 30.0
    assert client.max_attempts == 4
    assert client.backoff_base == 5.0


@pytest.mark.asyncio
async def test_unexpected_failure_is_redacted_and_never_raised(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_detail = "unexpected-sensitive-detail"
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [ValueError(raw_detail)])
    client = EvidenceLedgerClient(auth=_AuthStub())  # type: ignore[arg-type]

    with caplog.at_level(logging.ERROR, logger="tracelab.evidence_ledger"):
        result = await client.deliver(_trigger(job_id="private-job-marker"))

    assert result.category == "internal_error"
    assert result.retryable is False
    assert result.attempts == 1
    serialized = json.dumps(asdict(result), sort_keys=True) + caplog.text
    assert raw_detail not in serialized
    assert "private-job-marker" not in serialized
    assert _MISSION_UUID not in serialized
