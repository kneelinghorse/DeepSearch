"""Unit tests for ``deepsearch.agent.tools.url_liveness``.

Mocks httpx at the module boundary so no network calls are made. Covers:
alive (200), GET-confirmed dead (404), redirect (final_url reflects
destination), HEAD-405 fallback to ``Range: bytes=0-0`` GET, the S84 m01
browser-UA GET fallback on other 4xx (bot-wall rescue), timeout, DNS failure,
and the ``error`` transport-category field.
"""

from __future__ import annotations

import logging
import types

import pytest

import deepsearch.agent.tools.source_fetch as source_fetch_mod
import deepsearch.agent.tools.url_liveness as url_liveness_mod
from deepsearch.agent.tools.url_liveness import (
    BROWSER_USER_AGENT,
    url_liveness,
)


class _FakeResponse:
    def __init__(self, *, status_code: int, url: str = "") -> None:
        self.status_code = status_code
        self.url = url


class _FakeClient:
    """Records calls and returns scripted responses by method."""

    def __init__(
        self,
        *,
        head_response=None,
        head_raise=None,
        get_response=None,
        get_raise=None,
    ) -> None:
        self._head_response = head_response
        self._head_raise = head_raise
        self._get_response = get_response
        self._get_raise = get_raise
        self.head_calls: list[tuple] = []
        self.get_calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def head(self, url, headers=None):
        self.head_calls.append((url, headers))
        if self._head_raise is not None:
            raise self._head_raise
        return self._head_response

    def get(self, url, headers=None):
        self.get_calls.append((url, headers))
        if self._get_raise is not None:
            raise self._get_raise
        return self._get_response


def _make_fake_httpx(client: _FakeClient) -> types.SimpleNamespace:
    import httpx as real_httpx

    class _ClientFactory:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return client.__enter__()

        def __exit__(self, *args):
            return client.__exit__(*args)

    return types.SimpleNamespace(
        Client=_ClientFactory,
        TimeoutException=real_httpx.TimeoutException,
        ConnectError=real_httpx.ConnectError,
    )


def test_alive_200(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient(
        head_response=_FakeResponse(status_code=200, url="https://docs.aws.amazon.com/x")
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://docs.aws.amazon.com/x")

    assert out["alive"] is True
    assert out["status_code"] == 200
    assert out["final_url"] == "https://docs.aws.amazon.com/x"
    assert out["error"] is None
    assert len(client.head_calls) == 1
    assert client.get_calls == []


def test_dead_404_is_get_confirmed(monkeypatch: pytest.MonkeyPatch):
    """S84 m01: a HEAD 404 is no longer trusted alone — the browser-UA GET
    fallback runs and only its verdict is reported. Here the GET confirms."""

    client = _FakeClient(
        head_response=_FakeResponse(status_code=404, url="https://gone.example.com/x"),
        get_response=_FakeResponse(status_code=404, url="https://gone.example.com/x"),
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://gone.example.com/x")

    assert out["alive"] is False
    assert out["status_code"] == 404
    assert out["error"] is None
    assert len(client.head_calls) == 1
    assert len(client.get_calls) == 1
    _, headers = client.get_calls[0]
    assert headers == {"User-Agent": BROWSER_USER_AGENT}


def test_head_404_rescued_by_browser_ua_get(monkeypatch: pytest.MonkeyPatch):
    """Bot-wall serving 404 to the probe UA but the real page to a browser UA:
    the GET's 200 wins — the URL is alive."""

    client = _FakeClient(
        head_response=_FakeResponse(status_code=404, url="https://soft404.example.com/x"),
        get_response=_FakeResponse(status_code=200, url="https://soft404.example.com/x"),
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://soft404.example.com/x")

    assert out["alive"] is True
    assert out["status_code"] == 200
    assert out["error"] is None


def test_bot_wall_403_rescued_by_browser_ua_get(monkeypatch: pytest.MonkeyPatch):
    """The nngroup-class fix: 403 to the bot-UA HEAD, 200 to the browser-UA
    GET → alive. This is the CAT false-dead killer (decision #823)."""

    client = _FakeClient(
        head_response=_FakeResponse(status_code=403, url="https://www.nngroup.com/a"),
        get_response=_FakeResponse(status_code=200, url="https://www.nngroup.com/a"),
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://www.nngroup.com/a")

    assert out["alive"] is True
    assert out["status_code"] == 200
    assert out["error"] is None
    _, headers = client.get_calls[0]
    assert headers == {"User-Agent": BROWSER_USER_AGENT}


def test_head_403_then_refused_get_preserves_the_403(monkeypatch: pytest.MonkeyPatch):
    """S84 adversarial-review fix: a bot wall that answers the HEAD with 403
    and then refuses the fallback GET's connection must NOT masquerade as
    NXDOMAIN — the payload keeps the obtained 403 alongside error='connect',
    so the batch wrapper's connect-twice dead rule can never fire on it."""

    import httpx as real_httpx

    client = _FakeClient(
        head_response=_FakeResponse(status_code=403, url="https://wall.example.com/x"),
        get_raise=real_httpx.ConnectError("reset"),
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://wall.example.com/x")

    assert out["alive"] is False
    assert out["status_code"] == 403  # the HEAD's status is preserved
    assert out["error"] == "connect"


def test_head_405_then_range_get_transport_failure_preserves_405(
    monkeypatch: pytest.MonkeyPatch,
):
    """Transport failure on the 405 Range-GET keeps the 405 payload and skips
    the browser-UA re-GET (the chain is already known to be broken)."""

    import httpx as real_httpx

    client = _FakeClient(
        head_response=_FakeResponse(status_code=405, url="https://no-head.example.com/x"),
        get_raise=real_httpx.TimeoutException("slow"),
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://no-head.example.com/x")

    assert out["alive"] is False
    assert out["status_code"] == 405
    assert out["error"] == "timeout"
    assert len(client.get_calls) == 1  # no browser-UA attempt after the failure


def test_head_405_then_range_403_uses_third_browser_request(
    monkeypatch: pytest.MonkeyPatch,
):
    """The bounded provider inventory counts the full three-request ladder."""

    class _ScriptedGetClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__(
                head_response=_FakeResponse(
                    status_code=405,
                    url="https://no-head.example.com/x",
                )
            )
            self._responses = iter(
                [
                    _FakeResponse(
                        status_code=403,
                        url="https://no-head.example.com/x",
                    ),
                    _FakeResponse(
                        status_code=200,
                        url="https://no-head.example.com/x",
                    ),
                ]
            )

        def get(self, url, headers=None):
            self.get_calls.append((url, headers))
            return next(self._responses)

    client = _ScriptedGetClient()
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://no-head.example.com/x")

    assert out["alive"] is True
    assert out["status_code"] == 200
    assert len(client.head_calls) + len(client.get_calls) == 3
    assert client.get_calls[0][1] == {"Range": "bytes=0-0"}
    assert client.get_calls[1][1] == {"User-Agent": BROWSER_USER_AGENT}


def test_bot_wall_403_on_both_reports_unconfirmed_403(monkeypatch: pytest.MonkeyPatch):
    """403 to HEAD and to the browser-UA GET: reported as a plain 403 payload
    (alive=False, error=None). Classification to unknown-not-dead happens in
    the batch wrapper — the probe just reports honestly."""

    client = _FakeClient(
        head_response=_FakeResponse(status_code=403, url="https://www.zoro.com/x"),
        get_response=_FakeResponse(status_code=403, url="https://www.zoro.com/x"),
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://www.zoro.com/x")

    assert out["alive"] is False
    assert out["status_code"] == 403
    assert out["error"] is None
    assert len(client.get_calls) == 1


def test_redirect_final_url_reflects_destination(monkeypatch: pytest.MonkeyPatch):
    """With follow_redirects=True (set by the tool), httpx returns the final
    response after redirects and ``resp.url`` is the destination URL. We
    simulate that by returning a 200 response whose URL differs from the input."""

    client = _FakeClient(
        head_response=_FakeResponse(
            status_code=200,
            url="https://final.example.com/destination",
        )
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://redirect.example.com/start")

    assert out["alive"] is True
    assert out["status_code"] == 200
    assert out["final_url"] == "https://final.example.com/destination"


def test_head_405_falls_back_to_range_get(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient(
        head_response=_FakeResponse(status_code=405, url="https://no-head.example.com/x"),
        get_response=_FakeResponse(status_code=206, url="https://no-head.example.com/x"),
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://no-head.example.com/x")

    assert out["alive"] is True  # 206 is in [200, 400)
    assert out["status_code"] == 206
    assert out["error"] is None
    assert len(client.head_calls) == 1
    assert len(client.get_calls) == 1
    # The GET fallback uses Range: bytes=0-0
    _, headers = client.get_calls[0]
    assert headers == {"Range": "bytes=0-0"}


def test_timeout_reports_timeout_error_not_dead(monkeypatch: pytest.MonkeyPatch):
    """S84 m01: a timeout is NOT dead evidence — the payload carries
    error='timeout' with status 0 so the batch wrapper lands on unknown."""

    import httpx as real_httpx

    client = _FakeClient(head_raise=real_httpx.TimeoutException("slow"))
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://slow.example.com", timeout_s=1.0)

    assert out["alive"] is False
    assert out["status_code"] == 0
    assert out["final_url"] == "https://slow.example.com"
    assert out["error"] == "timeout"


def test_connect_error_reports_connect_category(monkeypatch: pytest.MonkeyPatch):
    """DNS NXDOMAIN / connection refused surface as error='connect' — the
    strong-evidence category the batch wrapper may treat as dead after its
    confirming retry."""

    import httpx as real_httpx

    client = _FakeClient(head_raise=real_httpx.ConnectError("dns"))
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://does-not-resolve.example.invalid")

    assert out["alive"] is False
    assert out["status_code"] == 0
    assert out["error"] == "connect"


def test_empty_url_returns_invalid_well_formed():
    out = url_liveness("")

    assert out == {
        "alive": False,
        "status_code": 0,
        "final_url": "",
        "error": "invalid",
    }


def test_never_raises_on_unexpected_exception(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient(head_raise=RuntimeError("unexpected"))
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    out = url_liveness("https://x.com")

    assert out["alive"] is False
    assert out["status_code"] == 0
    assert out["error"] == "other"


def test_fallback_exception_log_omits_url_and_exception_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_url = (
        "https://api-user:"
        "api-password@signed-host.example/path?token=query-secret"
    )
    client = _FakeClient(
        head_response=_FakeResponse(status_code=403, url=sensitive_url),
        get_raise=RuntimeError("exception-body-secret"),
    )
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    with caplog.at_level(logging.WARNING, logger=url_liveness_mod.__name__):
        out = url_liveness(sensitive_url)

    assert out["alive"] is False
    assert out["status_code"] == 403
    assert out["error"] == "other"
    assert "category=other exception=RuntimeError" in caplog.text
    for forbidden in (
        "signed-host.example",
        "api-user",
        "api-password",
        "query-secret",
        "exception-body-secret",
    ):
        assert forbidden not in caplog.text


def test_outer_exception_log_omits_url_and_exception_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_url = (
        "https://api-user:"
        "api-password@signed-host.example/path?token=query-secret"
    )
    client = _FakeClient(head_raise=RuntimeError("exception-body-secret"))
    monkeypatch.setattr(url_liveness_mod, "httpx", _make_fake_httpx(client))

    with caplog.at_level(logging.WARNING, logger=url_liveness_mod.__name__):
        out = url_liveness(sensitive_url)

    assert out["alive"] is False
    assert out["status_code"] == 0
    assert out["error"] == "other"
    assert "category=other exception=RuntimeError" in caplog.text
    for forbidden in (
        "signed-host.example",
        "api-user",
        "api-password",
        "query-secret",
        "exception-body-secret",
    ):
        assert forbidden not in caplog.text


class _RequestHookClientFactory:
    """Exercise the production request hook without opening a socket."""

    def __init__(
        self,
        *,
        redirect_target: str | None = None,
    ) -> None:
        self.redirect_target = redirect_target
        self.connected_urls: list[str] = []
        self._request_hooks = []

    def __call__(self, *args, **kwargs):
        del args
        self._request_hooks = list(kwargs["event_hooks"]["request"])
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return None

    def _before_request(self, url: str) -> None:
        import httpx as real_httpx

        request = real_httpx.Request("HEAD", url)
        for hook in self._request_hooks:
            hook(request)

    def head(self, url, headers=None):
        del headers
        self._before_request(url)
        self.connected_urls.append(url)
        if self.redirect_target is not None:
            self._before_request(self.redirect_target)
            self.connected_urls.append(self.redirect_target)
        return _FakeResponse(status_code=200, url=url)


def _install_hook_client(monkeypatch, factory: _RequestHookClientFactory) -> None:
    import httpx as real_httpx

    monkeypatch.setattr(
        url_liveness_mod,
        "httpx",
        types.SimpleNamespace(
            Client=factory,
            TimeoutException=real_httpx.TimeoutException,
            ConnectError=real_httpx.ConnectError,
        ),
    )


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/internal",
        "http://169.254.169.254/latest/meta-data",
    ],
)
def test_non_public_literal_target_is_rejected_before_connection(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    factory = _RequestHookClientFactory()
    _install_hook_client(monkeypatch, factory)

    out = url_liveness(target)

    assert out["error"] == "invalid"
    assert out["status_code"] == 0
    assert factory.connected_urls == []


def test_dns_resolution_to_private_address_is_rejected_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_fetch_mod,
        "_resolve_host_addresses",
        lambda _host, _port: {"10.42.0.7"},
    )
    factory = _RequestHookClientFactory()
    _install_hook_client(monkeypatch, factory)

    out = url_liveness("https://apparently-public.example/path")

    assert out["error"] == "invalid"
    assert factory.connected_urls == []


def test_redirect_to_cloud_metadata_is_rejected_before_second_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_fetch_mod,
        "_resolve_host_addresses",
        lambda host, _port: (
            {"93.184.216.34"}
            if host == "public.example"
            else {"169.254.169.254"}
        ),
    )
    metadata_url = "http://169.254.169.254/latest/meta-data"
    factory = _RequestHookClientFactory(redirect_target=metadata_url)
    _install_hook_client(monkeypatch, factory)

    out = url_liveness("https://public.example/start")

    assert out["error"] == "invalid"
    assert factory.connected_urls == ["https://public.example/start"]
