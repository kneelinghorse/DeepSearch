"""Linkup transport settings used by the execution-fence snapshot."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deepsearch.tools import linkup_search as linkup_module


@pytest.mark.asyncio
async def test_linkup_sdk_receives_the_declared_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The machine-readable timeout must describe the actual SDK invocation."""

    captured: dict[str, object] = {}

    class FakeLinkupClient:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

        def search(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(results=[])

    monkeypatch.setattr(linkup_module, "_LinkupClient", FakeLinkupClient)
    client = linkup_module.LinkupSearchClient(
        api_key="loopback-test-key",
        timeout=12.5,
        max_retries=0,
        circuit_breaker=None,
    )

    assert await client.search("bounded request") == []
    assert captured["timeout"] == 12.5
