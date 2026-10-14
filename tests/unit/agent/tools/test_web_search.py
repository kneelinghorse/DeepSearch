"""Unit tests for ``deepsearch.agent.tools.web_search``.

Mocks ``LinkupSearchClient`` at the module boundary so no Linkup API calls
are made. Mocks ``rerank`` so dense-model load is never triggered in CI —
the rerank module has its own dedicated unit tests at
``tests/unit/agent/tools/test_rerank.py``.

Covers: happy path (with rerank invoked + new return shape) +
rate-limit (429) + malformed payload (treated as ``unknown`` category by
the underlying client's exception translator) + empty Linkup return.
"""

from __future__ import annotations

from typing import Any

import pytest

import deepsearch.agent.tools.web_search as web_search_mod
from deepsearch.agent.tools.web_search import MAX_RESULTS_HARD_CAP, web_search
from deepsearch.tools.linkup_search import LinkupResult, LinkupSearchError


class _FakeClient:
    """Minimal stand-in for LinkupSearchClient used by the tool."""

    def __init__(self, *, results=None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error
        self.last_query: str | None = None
        self.last_max: int | None = None

    async def search(self, *, query: str, max_results: int):
        self.last_query = query
        self.last_max = max_results
        if self._error is not None:
            raise self._error
        return self._results


def _patch_client(monkeypatch: pytest.MonkeyPatch, fake: _FakeClient) -> None:
    monkeypatch.setattr(
        web_search_mod, "LinkupSearchClient", lambda *a, **kw: fake
    )


@pytest.fixture
def fake_rerank(monkeypatch: pytest.MonkeyPatch):
    """Replace rerank with a recording stub that appends score/tier deterministically."""

    calls: list[dict[str, Any]] = []

    def _stub(query: str, results, top_k: int):
        calls.append({"query": query, "results": list(results), "top_k": top_k})
        out = []
        for i, r in enumerate(results):
            item = dict(r)
            item["score"] = float(len(results) - i)
            item["authority_tier"] = "high" if "docs" in (item.get("url") or "") else "low"
            out.append(item)
        return out[: min(top_k, len(out))]

    monkeypatch.setattr(web_search_mod, "rerank", _stub)
    return calls


def test_happy_path_returns_reranked_shape(
    monkeypatch: pytest.MonkeyPatch, fake_rerank
):
    fake = _FakeClient(
        results=[
            LinkupResult(
                url="https://docs.aws.amazon.com/lambda/scaling",
                title="Lambda scaling",
                snippet="Concurrency limits...",
            ),
            LinkupResult(
                url="https://cloud.google.com/run/docs",
                title="Cloud Run docs",
                snippet="Container runtime...",
            ),
        ]
    )
    _patch_client(monkeypatch, fake)

    out = web_search("aws lambda scaling", max_results=5)

    assert len(out) == 2
    # Original keys preserved; rerank appends score and authority_tier
    expected_keys = {"title", "url", "snippet", "score", "authority_tier"}
    for item in out:
        assert set(item.keys()) == expected_keys
        assert isinstance(item["score"], float)
        assert item["authority_tier"] in {"high", "medium", "low"}
    urls = {r["url"] for r in out}
    assert "https://docs.aws.amazon.com/lambda/scaling" in urls
    assert "https://cloud.google.com/run/docs" in urls
    assert fake.last_query == "aws lambda scaling"
    assert fake.last_max == 5

    # rerank invoked exactly once with the right args
    assert len(fake_rerank) == 1
    call = fake_rerank[0]
    assert call["query"] == "aws lambda scaling"
    assert call["top_k"] == 5
    assert len(call["results"]) == 2
    # Pre-rerank shape is the legacy 3-key dict
    assert set(call["results"][0].keys()) == {"title", "url", "snippet"}


def test_empty_linkup_results_pass_through_rerank(
    monkeypatch: pytest.MonkeyPatch, fake_rerank
):
    """Empty Linkup payload → empty rerank → empty list to caller (not an error)."""

    _patch_client(monkeypatch, _FakeClient(results=[]))

    out = web_search("anything", max_results=10)

    assert out == []
    assert len(fake_rerank) == 1
    assert fake_rerank[0]["results"] == []


def test_rate_limit_returns_error_sentinel(monkeypatch: pytest.MonkeyPatch):
    err = LinkupSearchError(
        "Linkup rate limit exceeded", category="rate_limit", status_code=429
    )
    _patch_client(monkeypatch, _FakeClient(error=err))

    out = web_search("anything", max_results=10)

    assert len(out) == 1
    assert out[0]["error"] == "rate_limit"
    assert "rate limit" in out[0]["message"].lower()
    assert out[0]["title"] == "" and out[0]["url"] == "" and out[0]["snippet"] == ""


def test_malformed_payload_returns_unknown_sentinel(monkeypatch: pytest.MonkeyPatch):
    """Malformed-payload paths translate to a non-rate-limit, non-auth LinkupSearchError
    with category='unknown' inside LinkupSearchClient._translate_exception. The wrapper
    propagates that category through the sentinel."""

    err = LinkupSearchError(
        "Linkup search failed (KeyError): 'results'", category="unknown"
    )
    _patch_client(monkeypatch, _FakeClient(error=err))

    out = web_search("anything")

    assert len(out) == 1
    assert out[0]["error"] == "unknown"
    assert "KeyError" in out[0]["message"] or "results" in out[0]["message"]


def test_empty_query_returns_validation_sentinel():
    out = web_search("", max_results=5)
    assert len(out) == 1
    assert out[0]["error"] == "validation"
    assert "non-empty" in out[0]["message"]


def test_max_results_hard_capped(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeClient(results=[])
    _patch_client(monkeypatch, fake)

    web_search("q", max_results=999)

    assert fake.last_max == MAX_RESULTS_HARD_CAP


def test_max_results_min_one(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeClient(results=[])
    _patch_client(monkeypatch, fake)

    web_search("q", max_results=0)

    assert fake.last_max == 1


def test_unexpected_exception_returns_unknown_sentinel(monkeypatch: pytest.MonkeyPatch):
    """Non-LinkupSearchError exceptions still degrade to a sentinel, never raise."""

    _patch_client(monkeypatch, _FakeClient(error=RuntimeError("connection reset")))

    out = web_search("q")

    assert out[0]["error"] == "unknown"
    assert "RuntimeError" in out[0]["message"]
