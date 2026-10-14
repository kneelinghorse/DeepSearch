"""S91 m04 signed minimal completion-receipt transport tests."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import httpx
import pytest

from deepsearch.tracelab.receipt import CompletionReceipt, CompletionReceiptClient


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


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
) -> None:
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **_kwargs: _AsyncClient(requests, outcomes),
    )


@pytest.mark.asyncio
async def test_receipt_is_minimal_and_signature_covers_exact_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    client = CompletionReceiptClient(
        base_url="https://tracelab.example",
        secret="shared-secret",
    )

    result = await client.deliver(
        CompletionReceipt(job_id="job-123", mission_id="S91-TRACE-001")
    )

    assert result.success is True
    assert result.attempts == 1
    assert result.retryable is False
    assert requests[0]["content"] == (
        b'{"job_id":"job-123","mission_id":"S91-TRACE-001","status":"complete"}'
    )
    assert b"result_markdown" not in requests[0]["content"]
    digest = hmac.new(
        b"shared-secret",
        requests[0]["content"],
        hashlib.sha256,
    ).hexdigest()
    assert requests[0]["headers"] == {
        "Content-Type": "application/json",
        "X-DeepSearch-Signature": f"sha256={digest}",
    }
    assert requests[0]["url"] == "https://tracelab.example/api/v1/webhooks/deepsearch"


@pytest.mark.asyncio
async def test_429_and_5xx_are_retried_until_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [429, 503, 200])
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("deepsearch.tracelab.receipt.asyncio.sleep", _sleep)
    client = CompletionReceiptClient(
        base_url="https://tracelab.example",
        secret="shared-secret",
        max_attempts=4,
        backoff_base=0.25,
    )

    result = await client.deliver(CompletionReceipt(job_id="job", mission_id="mission"))

    assert result.success is True
    assert result.attempts == 3
    assert len(requests) == 3
    assert sleeps == [0.25, 0.5]


@pytest.mark.asyncio
async def test_nonretryable_4xx_stops_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [401])
    client = CompletionReceiptClient(
        base_url="https://tracelab.example",
        secret="shared-secret",
    )

    result = await client.deliver(CompletionReceipt(job_id="job", mission_id="mission"))

    assert result.success is False
    assert result.attempts == 1
    assert result.retryable is False
    assert result.final_status_code == 401
    assert result.error == "TraceLab returned HTTP 401"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_transport_exception_is_bounded_and_returned_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    request = httpx.Request("POST", "https://tracelab.example")
    _install_client(
        monkeypatch,
        requests,
        [httpx.ConnectError("offline", request=request)] * 2,
    )

    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("deepsearch.tracelab.receipt.asyncio.sleep", _sleep)
    client = CompletionReceiptClient(
        base_url="https://tracelab.example",
        secret="shared-secret",
        max_attempts=2,
    )

    result = await client.deliver(CompletionReceipt(job_id="job", mission_id="mission"))

    assert result.success is False
    assert result.attempts == 2
    assert result.retryable is True
    assert result.final_status_code is None
    assert result.error == "ConnectError during receipt delivery"


@pytest.mark.asyncio
async def test_missing_configuration_is_visible_without_network_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    client = CompletionReceiptClient(base_url="", secret="")

    result = await client.deliver(CompletionReceipt(job_id="job", mission_id="mission"))

    assert result.success is False
    assert result.attempts == 0
    assert result.retryable is False
    assert result.error == "TraceLab completion receipt transport is not configured"
    assert requests == []


@pytest.mark.asyncio
async def test_external_plain_http_receipt_is_rejected_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signing secret must never cross an external plaintext connection."""

    requests: list[dict[str, Any]] = []
    _install_client(monkeypatch, requests, [200])
    client = CompletionReceiptClient(
        base_url="http://tracelab.example",
        secret="shared-secret",
    )

    result = await client.deliver(CompletionReceipt(job_id="job", mission_id="mission"))

    assert result.success is False
    assert result.attempts == 0
    assert result.retryable is False
    assert result.error == (
        "TraceLab completion receipts require HTTPS outside localhost"
    )
    assert requests == []
