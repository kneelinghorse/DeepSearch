"""Deterministic tests for the safe readable-content fetch boundary."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import httpx
import pytest

import deepsearch.agent.tools.source_fetch as source_fetch_mod
from deepsearch.agent.tools.source_fetch import MAX_RESPONSE_BYTES, source_fetch


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        body: bytes | str = b"",
        url: str,
        headers: dict[str, str] | None = None,
        chunks: Iterable[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self._body = body.encode() if isinstance(body, str) else body
        self._chunks = list(chunks) if chunks is not None else [self._body]

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def iter_bytes(self) -> Iterable[bytes]:
        yield from self._chunks


class _FakeClient:
    def __init__(
        self,
        responses: list[_FakeResponse],
        calls: list[str],
        exc: Exception | None,
    ) -> None:
        self._responses = responses
        self._calls = calls
        self._exc = exc

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def stream(self, method: str, url: str) -> _FakeResponse:
        assert method == "GET"
        self._calls.append(url)
        if self._exc is not None:
            raise self._exc
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_fetch_mod,
        "_resolve_host_addresses",
        lambda host, port: {"93.184.216.34"},
    )


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    *responses: _FakeResponse,
    exc: Exception | None = None,
) -> list[str]:
    calls: list[str] = []

    def factory(**kwargs: Any) -> _FakeClient:
        assert kwargs["follow_redirects"] is False
        return _FakeClient(list(responses), calls, exc)

    monkeypatch.setattr(source_fetch_mod.httpx, "Client", factory)
    return calls


def test_html_success_extracts_visible_text_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        "<html><head><title>Hidden</title></head><body><h1>Page</h1>"
        "<p>Readable evidence.</p><script>secret()</script></body></html>"
    )
    response = _FakeResponse(
        status_code=200,
        body=body,
        url="https://docs.example.com/page",
    )
    _install_client(monkeypatch, response)

    out = source_fetch("https://docs.example.com/page")

    assert out["status"] == "ok"
    assert out["text"] == "Page\nReadable evidence."
    assert "Hidden" not in out["text"]
    assert "secret" not in out["text"]
    assert out["content_type"] == "text/html"
    assert out["bytes_read"] == len(body.encode())
    assert out["text_truncated"] is False


def test_plain_text_success(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=200,
        body="Primary source text",
        url="https://example.com/data.txt",
        headers={"content-type": "text/plain"},
    )
    _install_client(monkeypatch, response)

    out = source_fetch("https://example.com/data.txt")

    assert out["status"] == "ok"
    assert out["text"] == "Primary source text"


def test_timeout_returns_timeout_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_client(
        monkeypatch,
        exc=httpx.TimeoutException("slow"),
    )

    out = source_fetch("https://slow.example.com", timeout_s=1.0)

    assert out["status"] == "timeout"
    assert out["error"] == "timeout"
    assert calls == ["https://slow.example.com"]


def test_connect_failure_is_dead_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_client(
        monkeypatch,
        exc=httpx.ConnectError("dns or connect failed"),
    )

    out = source_fetch("https://gone.example.com")

    assert out["status"] == "dead"
    assert out["error"] == "connect"
    assert calls == ["https://gone.example.com"]


def test_unexpected_transport_failure_is_explicit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch, exc=RuntimeError("unexpected"))

    out = source_fetch("https://example.com")

    assert out["status"] == "error"
    assert out["error"] == "unexpected"


def test_unexpected_transport_log_omits_url_and_exception_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_url = (
        "https://api-user:"
        "api-password@signed-host.example/path?token=query-secret"
    )

    def fail_client(**kwargs: Any) -> None:
        del kwargs
        raise RuntimeError("exception-body-secret")

    monkeypatch.setattr(source_fetch_mod.httpx, "Client", fail_client)

    with caplog.at_level(logging.WARNING, logger=source_fetch_mod.__name__):
        out = source_fetch(sensitive_url)

    assert out["status"] == "error"
    assert out["error"] == "unexpected"
    assert "category=unexpected exception=RuntimeError" in caplog.text
    for forbidden in (
        "signed-host.example",
        "api-user",
        "api-password",
        "query-secret",
        "exception-body-secret",
    ):
        assert forbidden not in caplog.text


def test_extraction_log_omits_url_and_exception_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_url = "https://signed-host.example/path?token=query-secret"
    response = _FakeResponse(
        status_code=200,
        body="private response body",
        url=sensitive_url,
    )
    _install_client(monkeypatch, response)

    def fail_extraction(body: bytes, content_type: str, headers: Any) -> str:
        del body, content_type, headers
        raise RuntimeError("exception-body-secret")

    monkeypatch.setattr(source_fetch_mod, "_extract_readable_text", fail_extraction)

    with caplog.at_level(logging.INFO, logger=source_fetch_mod.__name__):
        out = source_fetch(sensitive_url)

    assert out["status"] == "error"
    assert out["error"] == "extraction_failed"
    assert "category=extraction_failed exception=RuntimeError" in caplog.text
    for forbidden in (
        "signed-host.example",
        "query-secret",
        "exception-body-secret",
        "private response body",
    ):
        assert forbidden not in caplog.text


@pytest.mark.parametrize(
    ("status_code", "status", "error"),
    [
        (401, "blocked", "http_blocked"),
        (403, "blocked", "http_blocked"),
        (429, "blocked", "http_blocked"),
        (404, "dead", "http_dead"),
        (410, "dead", "http_dead"),
        (503, "error", "http_error"),
    ],
)
def test_http_failures_are_single_attempt(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    status: str,
    error: str,
) -> None:
    response = _FakeResponse(
        status_code=status_code,
        url="https://example.com/page",
    )
    calls = _install_client(monkeypatch, response)

    out = source_fetch("https://example.com/page")

    assert (out["status"], out["status_code"], out["error"]) == (
        status,
        status_code,
        error,
    )
    assert calls == ["https://example.com/page"]


def test_declared_oversized_body_is_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status_code=200,
        body="not read",
        url="https://example.com/huge",
        headers={
            "content-type": "text/plain",
            "content-length": str(MAX_RESPONSE_BYTES + 1),
        },
    )
    _install_client(monkeypatch, response)

    out = source_fetch("https://example.com/huge")

    assert out["status"] == "error"
    assert out["error"] == "body_too_large"
    assert out["text"] == ""


def test_streamed_oversized_body_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=200,
        url="https://example.com/huge",
        headers={"content-type": "text/plain"},
        chunks=[b"x" * MAX_RESPONSE_BYTES, b"y"],
    )
    _install_client(monkeypatch, response)

    out = source_fetch("https://example.com/huge")

    assert out["status"] == "error"
    assert out["error"] == "body_too_large"
    assert out["bytes_read"] == MAX_RESPONSE_BYTES + 1


def test_binary_content_type_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=200,
        body=b"%PDF",
        url="https://example.com/file.pdf",
        headers={"content-type": "application/pdf"},
    )
    _install_client(monkeypatch, response)

    out = source_fetch("https://example.com/file.pdf")

    assert out["status"] == "error"
    assert out["error"] == "unsupported_content_type"


def test_empty_html_reports_extraction_or_spa_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status_code=200,
        body="<html><body><div id='app'></div><script>boot()</script></body></html>",
        url="https://example.com/app",
    )
    _install_client(monkeypatch, response)

    out = source_fetch("https://example.com/app")

    assert out["status"] == "error"
    assert out["error"] == "extraction_failed_or_spa_shell"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://[::1]/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://224.0.0.1/data",
        "http://metadata.google.internal/computeMetadata/v1/",
        "file:///etc/passwd",
        "https://user:secret@example.com/private",
        "https://example.com:99999/private",
    ],
)
def test_non_public_and_metadata_targets_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    def resolver(host: str, port: int) -> set[str]:
        del port
        return {host}

    monkeypatch.setattr(source_fetch_mod, "_resolve_host_addresses", resolver)

    out = source_fetch(url)

    assert out["status"] == "blocked"
    assert out["status_code"] == 0


def test_public_hostname_resolving_private_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_fetch_mod,
        "_resolve_host_addresses",
        lambda host, port: {"10.0.0.7"},
    )

    out = source_fetch("https://public-name.example/data")

    assert out["status"] == "blocked"
    assert out["error"].startswith("non_public_address")


def test_redirect_target_is_dns_revalidated_and_blocked_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolutions: list[str] = []

    def resolver(host: str, port: int) -> set[str]:
        del port
        resolutions.append(host)
        return {"93.184.216.34"} if host == "public.example" else {"10.0.0.8"}

    monkeypatch.setattr(source_fetch_mod, "_resolve_host_addresses", resolver)
    redirect = _FakeResponse(
        status_code=302,
        url="https://public.example/start",
        headers={"location": "http://internal.example/secret"},
    )
    calls = _install_client(monkeypatch, redirect)

    out = source_fetch("https://public.example/start")

    assert out["status"] == "blocked"
    assert resolutions == ["public.example", "internal.example"]
    assert calls == ["https://public.example/start"]


def test_excluded_redirect_target_is_blocked_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redirect = _FakeResponse(
        status_code=302,
        url="https://public.example/start",
        headers={"location": "https://blocked.example/secret"},
    )
    calls = _install_client(monkeypatch, redirect)

    out = source_fetch(
        "https://public.example/start",
        excluded_domains=["blocked.example"],
    )

    assert out["status"] == "blocked"
    assert out["error"] == "excluded_domain"
    assert calls == ["https://public.example/start"]


def test_empty_url_returns_well_formed_error() -> None:
    out = source_fetch("")

    assert out["status"] == "error"
    assert out["status_code"] == 0
    assert out["text"] == ""
    assert out["final_url"] == ""
    assert "fetched_at" in out
