"""Real row/contract/report/protocol/writer seams with provider-free fixtures."""
from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.outcome import resolve_final_outcome_payload
from deepsearch.mission.contract import compile_canonical_contract_from_state
from deepsearch.report.protocol import MissionProtocolComplete
from deepsearch.report.scope_validation import count_report_words
from deepsearch.worker.converter import convert_row_to_state
from deepsearch.worker.lease import LeaseAttempt
from deepsearch.worker.poller import MissionRow
from deepsearch.worker.writer import ResultWriter, assert_result_identity, build_execution_metadata

FIXTURE = Path(__file__).parents[1] / "fixtures/authored_scope_v1/postgresql-authored.json"
URL1 = "https://www.postgresql.org/docs/current/functions-datetime.html"
URL2 = "https://www.postgresql.org/docs/current/explicit-locking.html"


def row_and_state() -> tuple[MissionRow, dict]:
    payload = json.loads(FIXTURE.read_text())
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    lease = LeaseAttempt(
        "offline-worker", "offline-token", now, now, now + timedelta(seconds=120), 1,
    )
    row = MissionRow(
        id="3f5c2641-7a47-45ed-a092-feebe9143ed1", mission_id=payload["mission_id"],
        title=payload["title"], objective=payload["objective"], context={},
        status="in_progress", queued_at=now, lease=lease,
        success_criteria=payload["success_criteria"], deliverables=payload["deliverables"],
        constraints=payload["constraints"], references=payload["references"],
        deliverable_format=payload["deliverable_format"],
    )
    state = convert_row_to_state(row)
    canonical_input = json.loads((FIXTURE.parent / "canonical-input.json").read_text())
    assert {key: state[key] for key in canonical_input} == canonical_input
    contract = compile_canonical_contract_from_state(state)
    assert contract.model_dump(mode="json") == json.loads(
        (FIXTURE.parent / "canonical-contract.json").read_text()
    )
    state["mission_contract"] = contract.model_dump(mode="json")
    state["mission_contract_id"] = contract.contract_id
    return row, state


def actor_result(words: int, sources: list[dict]) -> SynthesisResult:
    draft = ("# Note\n\n" + "evidence " * words
             + f"Timing matters. [cite:{URL1}] [1]\n\n"
             + f"## References\n[1] Date/time — {URL1}\n")
    return SynthesisResult(
        output_markdown=draft, step_count=5, self_terminated=True, hit_max_steps=False,
        runtime_seconds=1, token_usage={"input": 100, "output": 100, "total": 200},
        token_usage_complete=True, tool_call_summary={}, authority_tier_distribution={},
        pool_size=0, step_trace=[], sources_collected=sources,
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Acceptance must not call a network/provider")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))


@pytest.mark.parametrize("case,expected", [
    ("compliant", "complete"), ("too_long", "validation_failed"),
    ("22_sources", "validation_failed"), ("partial", "complete"),
])
def test_scope_survives_real_protocol_and_worker_persistence(
    monkeypatch: pytest.MonkeyPatch, case: str, expected: str,
) -> None:
    row, state = row_and_state()
    urls = [URL1] if case == "partial" else [URL1, URL2]
    if case == "22_sources":
        urls += [f"https://postgrespro.com/page/{i}" for i in range(20)]
    sources = [{"url": u, "title": "Evidence", "body": "Primary evidence " * 40,
                "body_source": "source_fetch", "content_quality": "readable_body"} for u in urls]
    synthesis = actor_result(600 if case == "too_long" else 310, sources)
    calls: list[str] = []

    def synthesize(**kwargs: Any) -> SynthesisResult:
        calls.append("actor")
        policy = kwargs["scope_admission"]
        assert policy.scope.allowed_urls == [URL1, URL2]
        assert URL1 in kwargs["system_instructions"]
        assert "2,500-6,000" not in kwargs["system_instructions"]
        # The actor must see the final wrapper cost before spending its final
        # answer; the authoritative persisted gate still runs afterwards.
        budget = kwargs["report_word_budget_check"](synthesis.output_markdown)
        assert budget["projected_words"] > budget["draft_words"]
        assert budget["max_words"] == policy.scope.max_words
        assert "report_word_budget" in kwargs["system_instructions"]
        if case == "partial":
            policy.unavailable[URL2] = "timeout"
        return synthesis

    def critique(text: str, evidence: list, context: dict, **kwargs: Any) -> CritiqueResult:
        calls.append("critic")
        assert len(evidence) <= 2
        assert all("postgrespro" not in s["url"] for s in evidence)
        assert kwargs["scope_admission"].scope.max_words == 500
        return CritiqueResult(
            output_markdown=text, claim_count=1, gap_notes_inserted=0, runtime_seconds=0,
            token_usage={"input": 10, "output": 10, "total": 20}, token_usage_complete=True,
        )

    monkeypatch.setattr(fnode, "run_codeagent_synthesis", synthesize)
    monkeypatch.setattr(fnode, "run_critique", critique)
    result = asyncio.run(fnode.codeagent_finalize_node(state))
    assert calls == ["actor", "critic"]  # Scope/partial mismatch never buys a second actor.
    assert result["mission_outcome"] == expected
    report = result["report_markdown"]
    assert report == (FIXTURE.parent / f"{case}-persisted.md").read_text()
    assert "evidence " * 100 in report
    telemetry = result["synthesis_telemetry"]
    audit = telemetry["authored_scope_validation"]
    assert audit == json.loads((FIXTURE.parent / f"{case}-expected.json").read_text())
    assert audit["persisted"]["words"] == count_report_words(report)
    assert audit["persisted"]["sha256"] == hashlib.sha256(report.encode()).hexdigest()
    assert audit["pre_render"]["words"] < audit["persisted"]["words"]
    assert telemetry["original_synthesis"]["text"] == synthesis.output_markdown
    if case == "22_sources":
        assert audit["consulted_count"] == 22
    if case == "partial":
        assert audit["verdict"] == "partial" and "Partial evidence:" in report
        assert result["delivery_quality"] == "degraded"
    if expected == "validation_failed":
        assert audit["violations"]
    protocol = MissionProtocolComplete.model_validate_json(json.dumps(result["protocol_payload"]))
    assert protocol.report_metadata.forensic["authored_scope_validation"] == audit
    original = protocol.report_metadata.forensic["original_synthesis"]
    assert original["text"] == synthesis.output_markdown
    final_state = {**state, **result}
    final_outcome = resolve_final_outcome_payload(final_state, completed_at=row.queued_at)
    final_state["telemetry"] = {**final_state["telemetry"], "final_outcome": final_outcome}
    metadata = build_execution_metadata(final_state, 2)
    assert metadata.final_outcome["authored_scope_validation"] == audit
    assert_result_identity(
        row_mission_id=row.mission_id, row_title=row.title, result_markdown=report,
        result_protocol=result["protocol_payload"], synthesis_telemetry=telemetry,
        markdown_from_artifact=True,
    )

    class CapturingDatabase:
        def __init__(self) -> None:
            self.writes: list[tuple] = []

        async def fetchval(self, sql: str, *args: Any) -> str:
            self.writes.append((sql, args))
            return row.id

    db = CapturingDatabase()
    writer = ResultWriter(db)
    if expected == "validation_failed":
        asyncio.run(writer.mark_failed(
            row.id, error_message=result["routing_reason"], outcome=expected,
            lease=row.lease, job_id="offline-s94", result_markdown=report,
            result_protocol=result["protocol_payload"], execution_metadata=metadata,
        ))
        sql, args = db.writes[0]
        assert sql == writer.MARK_VALIDATION_FAILED_SQL
        assert args[3] == report
        assert json.loads(args[4]) == result["protocol_payload"]
        assert json.loads(args[5])["final_outcome"]["authored_scope_validation"] == audit
    else:
        asyncio.run(writer.mark_completed(
            row.id, result_markdown=report, result_protocol=result["protocol_payload"],
            lease=row.lease, job_id="offline-s94", execution_metadata=metadata,
        ))
        assert report in db.writes[0][1]
    assert len(db.writes) == 1


@pytest.mark.parametrize("partial", [False, True])
def test_real_codeagent_and_critic_use_scripted_providers_and_mock_transport(
    monkeypatch: pytest.MonkeyPatch, partial: bool,
) -> None:
    """Exercise the actual actor/tool/critic stack without buying research."""
    import httpx
    from smolagents.models import ChatMessage, MessageRole, Model
    from smolagents.monitoring import TokenUsage

    from deepsearch.agent import codeagent_runner as runner
    from deepsearch.agent import critique as critique_module
    from deepsearch.agent.tools import scholar_search as scholar_module
    from deepsearch.agent.tools import source_fetch as fetch_module
    from deepsearch.agent.tools import web_search as search_module

    _, state = row_and_state()
    draft = actor_result(310, []).output_markdown
    codes = [
        "\n".join([
            "print(web_search('PostgreSQL timestamp comparison'))",
            "print(scholar_search('PostgreSQL timestamp comparison'))",
            "print(source_fetch('https://postgrespro.com/third-source'))",
            f"first = source_fetch({URL1!r})",
            f"second = source_fetch({URL2!r})",
            f"print(source_fetch({URL1!r}, start=20, length=50))",
        ]),
        f"draft = {draft!r}\ncompleteness_check(draft)\n"
        "scratchpad_write('report_draft', draft)\nfinal_answer(draft)",
    ]

    class ScriptedActor(Model):
        def __init__(self) -> None:
            super().__init__(model_id="offline-scripted-fixture")
            self.calls = 0

        def generate(self, messages: list, **kwargs: Any) -> ChatMessage:
            assert self.calls < len(codes), "Unexpected retry/provider call"
            content = "Thought: Execute the deterministic fixture.\n<code>\n"
            content += codes[self.calls] + "\n</code>"
            self.calls += 1
            return ChatMessage(
                role=MessageRole.ASSISTANT, content=content,
                token_usage=TokenUsage(input_tokens=100, output_tokens=20),
                raw={"choices": [{"finish_reason": "stop"}]},
            )

    actor = ScriptedActor()
    critic_calls: list[str] = []

    def critic(**kwargs: Any) -> ChatMessage:
        messages = kwargs["messages"]
        critic_calls.append(messages[1]["content"])
        assert "postgrespro.com" not in messages[1]["content"]
        assert "maximum 500 words" in messages[1]["content"]
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=critique_module.CRITIQUE_OUTPUT_DELIMITER + "\n" + json.dumps({
                "claim_count": 1, "annotations": [], "assessments": [{
                    "anchor": "Timing matters.", "verdict": "supported",
                    "evidence_ids": ["E1.1"],
                    "evidence_quotes": {"E1.1": "Timing matters."},
                }],
            }),
            token_usage=TokenUsage(input_tokens=10, output_tokens=10),
        )

    requests: list[str] = []
    def transport(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requests.append(url)
        assert url in [URL1, URL2]
        if url == URL2 and partial:
            return httpx.Response(503)
        return httpx.Response(200, text="Timing matters. " * 40,
                              headers={"content-type": "text/plain"})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        **kw, transport=httpx.MockTransport(transport)))
    monkeypatch.setattr(fetch_module, "_validate_public_target", lambda url: None)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Exact-page mission reached discovery provider")
    monkeypatch.setattr(search_module, "web_search", forbidden)
    monkeypatch.setattr(scholar_module, "scholar_search", forbidden)
    monkeypatch.setenv("LINKUP_API_KEY", "offline-fixture-key")
    monkeypatch.setattr(runner, "build_default_model", lambda: actor)
    monkeypatch.setattr(critique_module, "build_default_model", lambda: critic)
    result = asyncio.run(fnode.codeagent_finalize_node(state))
    assert actor.calls == 2 and len(critic_calls) == 1
    assert requests == [URL1, URL2]  # Stored passage read makes no third request.
    assert result["mission_outcome"] == "complete"
    telemetry = result["synthesis_telemetry"]
    assert telemetry["self_terminated"]
    assert telemetry["authored_scope_validation"]["verdict"] == (
        "partial" if partial else "compliant"
    )
    assert telemetry["authored_scope_validation"]["consulted_count"] == (1 if partial else 2)
    assert len(telemetry["authored_scope"]["rejected"]) == 3
    assert telemetry["original_synthesis"]["text"] == draft
    assert telemetry["critique_telemetry"]["assessment"]["accuracy_pass"] is None
