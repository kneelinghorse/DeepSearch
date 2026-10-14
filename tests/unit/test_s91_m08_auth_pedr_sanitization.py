"""Regression tests for credential-safe TraceLab and PEDR diagnostics."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict

import httpx
import pytest

from deepsearch.tracelab.auth import TraceLabAuth, TraceLabAuthError
from deepsearch.tracelab.pedr import PEDRClient

PASSWORD_SENTINEL = "S91_M08_FAKE_PASSWORD_SENTINEL_DO_NOT_PERSIST"


class _ApiKeyAuth:
    """Minimal configured auth that lets tests reach PEDR response handling."""

    base_url = "https://tracelab.example"
    is_configured = True

    def get_headers(self) -> Dict[str, str]:
        return {"X-API-Key": "fake-test-key"}


def _response_with_password_body(url: str) -> httpx.Response:
    """Build a 422 whose body resembles a server-side credential echo."""

    request = httpx.Request("POST", url)
    return httpx.Response(
        422,
        json={"detail": f"rejected password={PASSWORD_SENTINEL}"},
        request=request,
    )


def test_login_422_keeps_status_without_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auth failures can be persisted, so a credential-echoing body must be absent."""

    login_url = "https://tracelab.example/api/v1/auth/login"
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *_args, **_kwargs: _response_with_password_body(login_url),
    )
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="",
        username="service@example.com",
        password="fake-password",
    )

    with pytest.raises(TraceLabAuthError) as exc_info:
        auth.get_headers()

    assert exc_info.value.status_code == 422
    assert str(exc_info.value) == "Login failed with status 422"
    assert PASSWORD_SENTINEL not in str(exc_info.value)


def test_login_transport_error_keeps_class_without_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport exception messages may contain request data and must not escape."""

    login_url = "https://tracelab.example/api/v1/auth/login"
    request = httpx.Request("POST", login_url)

    def raise_connect_error(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.ConnectError(
            f"upstream echoed password={PASSWORD_SENTINEL}",
            request=request,
        )

    monkeypatch.setattr(httpx, "post", raise_connect_error)
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="",
        username="service@example.com",
        password="fake-password",
    )

    with pytest.raises(TraceLabAuthError) as exc_info:
        auth.get_headers()

    assert str(exc_info.value) == "Login request failed (ConnectError)"
    assert PASSWORD_SENTINEL not in str(exc_info.value)


def test_pedr_auth_fallback_excludes_422_password_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Graceful degradation must not copy an auth response body into mission state."""

    login_url = "https://tracelab.example/api/v1/auth/login"
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *_args, **_kwargs: _response_with_password_body(login_url),
    )
    caplog.set_level(logging.WARNING, logger="deepsearch.tracelab.pedr")
    auth = TraceLabAuth(
        base_url="https://tracelab.example",
        api_key="",
        username="service@example.com",
        password="fake-password",
    )

    result = PEDRClient(auth=auth).preflight("safe research query")
    fallback_payload = repr(asdict(result))

    assert result.error == "Authentication failed (TraceLabAuthError, HTTP 422)"
    assert PASSWORD_SENTINEL not in fallback_payload
    assert PASSWORD_SENTINEL not in caplog.text


def test_pedr_http_fallback_keeps_class_and_status_without_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PEDR endpoint bodies must stay out of fallback metadata and warning logs."""

    search_url = "https://tracelab.example/api/v1/pedr/search"
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *_args, **_kwargs: _response_with_password_body(search_url),
    )
    caplog.set_level(logging.WARNING, logger="deepsearch.tracelab.pedr")

    result = PEDRClient(auth=_ApiKeyAuth()).search("safe research query")  # type: ignore[arg-type]
    fallback_payload = result.model_dump_json()

    assert result.metadata is not None
    metadata = result.metadata.model_dump()
    assert metadata["status_code"] == 422
    assert metadata["error"] == (
        "HTTP failure (HTTPStatusError, HTTP 422)"
    )
    assert PASSWORD_SENTINEL not in fallback_payload
    assert PASSWORD_SENTINEL not in caplog.text


def test_pedr_transport_fallback_keeps_class_without_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PEDR fallback state needs the failure class, never its arbitrary message."""

    request = httpx.Request(
        "POST",
        "https://tracelab.example/api/v1/pedr/preflight",
    )

    def raise_connect_error(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.ConnectError(
            f"upstream echoed password={PASSWORD_SENTINEL}",
            request=request,
        )

    monkeypatch.setattr(httpx, "post", raise_connect_error)
    caplog.set_level(logging.WARNING, logger="deepsearch.tracelab.pedr")

    result = PEDRClient(auth=_ApiKeyAuth()).preflight(  # type: ignore[arg-type]
        "safe research query"
    )
    fallback_payload = repr(asdict(result))

    assert result.error == "Request failure (ConnectError)"
    assert PASSWORD_SENTINEL not in fallback_payload
    assert PASSWORD_SENTINEL not in caplog.text
