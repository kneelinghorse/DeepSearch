"""Actor access to evidence tails must not cost another network fetch."""

from __future__ import annotations

import json
import threading

import pytest

from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.execution_control import ExecutionCancelledError
from deepsearch.agent.tools import source_fetch as fetch_module
from deepsearch.agent.tools.source_passages import select_source_passage

URL = "https://example.org/source"


def _reply(body: str, *, final_url: str = URL, date: str = "2026-09-18") -> dict:
    return {"text": body, "status": "ok", "status_code": 200,
            "fetched_at": date, "final_url": final_url, "text_truncated": False}




@pytest.mark.parametrize("kwargs,error", [
    ({"start": -1}, "invalid_start"), ({"length": 0}, "invalid_length"),
    ({"length": 12001}, "invalid_length"), ({"query": "x" * 201}, "invalid_query"),
    ({"query": "  "}, "invalid_query"),
    ({"query": "evidence", "refresh": True}, "invalid_refresh"),
    ({"start": 10}, "missing_stored_body"), ({"query": "evidence"}, "missing_stored_body"),
])
def test_invalid_or_missing_stored_reads_never_fetch(monkeypatch, kwargs: dict, error: str) -> None:
    def forbidden(*args, **kw):
        raise AssertionError("local request must not fetch")

    monkeypatch.setattr(fetch_module, "source_fetch", forbidden)
    bundle = runner.build_codeagent_tools()
    response = bundle.tools[1](URL, **kwargs)
    assert response["status"] == "error" and response["error"] == error
    assert not response["network_requested"]
    assert bundle.fetch_attempts == []
    assert bundle.tool_outcomes()["ledger_records"] == []


def test_end_no_match_and_out_of_range_are_distinct(monkeypatch) -> None:
    monkeypatch.setattr(fetch_module, "source_fetch", lambda *a: _reply("evidence"))
    tool = runner.build_codeagent_tools().tools[1]
    tool(URL)
    end = tool(URL, start=8)
    assert end["status"] == "ok" and end["text"] == "" and end["remaining_chars"] == 0
    missing = tool(URL, query="unknown")
    assert missing["status"] == "ok" and missing["matches"] == []
    assert missing["search_complete"] is True
    invalid = tool(URL, start=9)
    assert invalid["status"] == "error" and invalid["error"] == "range_out_of_bounds"


def test_find_is_literal_bounded_and_preserves_unicode_offsets() -> None:
    body = "Straße 🧭. " + "[Evidence]. " * 10
    result = select_source_passage(body, 0, 300, "[evidence]")
    assert len(result["matches"]) == 5 and result["matches_truncated"] is True
    assert result["search_complete"] is False
    for match in result["matches"]:
        assert body[match["start"]:match["end"]] == "[Evidence]"
    next_page = select_source_passage(body, result["matches"][-1]["end"], 300, "[evidence]")
    assert len(next_page["matches"]) == 5 and not next_page["matches_truncated"]


def test_refresh_replaces_shorter_body_and_proven_redirect_alias_only(monkeypatch) -> None:
    replies = iter([_reply("old evidence" * 50, final_url="https://www.example.org/source"),
                    _reply("new evidence", final_url="https://www.example.org/source", date="new")])
    calls: list[str] = []

    def fetch(url, timeout):
        calls.append(url)
        return next(replies)

    monkeypatch.setattr(fetch_module, "source_fetch", fetch)
    bundle = runner.build_codeagent_tools()
    tool = bundle.tools[1]
    first = tool(URL)
    repeated = tool(URL)
    redirected = tool("https://www.example.org/source")
    assert repeated["cache_hit"] and redirected["cache_hit"]
    assert len(calls) == 1
    new = tool(URL, refresh=True)
    assert new["source_id"] == first["source_id"]
    assert new["body_sha256"] != first["body_sha256"]
    assert tool("https://www.example.org/source")["text"] == "new evidence"
    assert bundle.sources_collected[0]["body"] == "new evidence"
    assert bundle.sources_fetched[0]["body"] == "new evidence"
    assert len(bundle.fetch_attempts) == 2


def test_no_inferred_www_alias_or_cross_attempt_cache(monkeypatch) -> None:
    monkeypatch.setattr(fetch_module, "source_fetch", lambda *a: _reply("evidence"))
    tool = runner.build_codeagent_tools().tools[1]
    tool(URL)
    result = tool("https://www.example.org/source", query="evidence")
    assert result["error"] == "missing_stored_body"
    fresh = runner.build_codeagent_tools().tools[1]
    assert fresh(URL, query="evidence")["error"] == "missing_stored_body"


def test_cancellation_and_incomplete_body_are_preserved(monkeypatch) -> None:
    out = _reply("evidence") | {"text_truncated": True}
    monkeypatch.setattr(fetch_module, "source_fetch", lambda *a: out)
    event = threading.Event()
    tool = runner.build_codeagent_tools(cancel_event=event).tools[1]
    assert tool(URL)["body_complete"] is False
    event.set()
    with pytest.raises(ExecutionCancelledError):
        tool(URL, query="evidence")


def test_encoded_payload_and_reported_ranges_remain_bounded(monkeypatch) -> None:
    body = "\x01\\\n" * 30000
    monkeypatch.setattr(fetch_module, "source_fetch", lambda *a: _reply(body))
    tool = runner.build_codeagent_tools().tools[1]
    tool(URL)
    result = tool(URL, start=20000)
    assert len(json.dumps(result, ensure_ascii=False, sort_keys=True)) <= 16000
    assert result["range_end"] == result["range_start"] + len(result["text"])
    assert result["text"] == body[result["range_start"]:result["range_end"]]
    assert result["remaining_chars"] == len(body) - result["range_end"]


def test_failed_refresh_retains_the_explicitly_old_snapshot(monkeypatch) -> None:
    replies = iter([_reply("evidence"), {"text": "", "status": "timeout", "status_code": 0,
                    "final_url": URL, "fetched_at": "later", "error": "timeout"}])
    monkeypatch.setattr(fetch_module, "source_fetch", lambda *a: next(replies))
    bundle = runner.build_codeagent_tools()
    tool = bundle.tools[1]
    original = tool(URL)
    assert tool(URL, refresh=True)["status"] == "timeout"
    cached = tool(URL, query="evidence")
    assert cached["body_sha256"] == original["body_sha256"]
    assert cached["fetched_at"] == "2026-09-18"
    assert len(bundle.fetch_attempts) == 2
    assert len(bundle.tool_outcomes()["ledger_records"]) == 2


def test_excluded_targets_remain_blocked_and_never_cache(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("excluded source must not fetch")

    monkeypatch.setattr(fetch_module, "source_fetch", forbidden)
    bundle = runner.build_codeagent_tools(excluded_domains=["example.org"])
    tool = bundle.tools[1]
    assert tool(URL)["status"] == "blocked"
    assert tool(URL, query="evidence")["status"] == "blocked"
    assert bundle.sources_collected == []
    assert bundle.tool_outcomes()["ledger_records"][0]["error_category"] == "excluded_domain"
