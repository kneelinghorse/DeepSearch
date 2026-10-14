"""S96: deterministic SDK and process plumbing, with no live provider calls."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError, OpenAI
from smolagents import OpenAIServerModel

from deepsearch.agent.codeagent_finalize_node import _build_synthesis_telemetry
from deepsearch.agent.codeagent_runner import run_codeagent_synthesis
from deepsearch.llm.runtime import request_failure
from deepsearch.llm.usage import observe_model_requests, summarize_usage
from deepsearch.worker.execution import MissionProcessExecutor
from deepsearch.worker.lease import LeaseAttempt
from deepsearch.worker.writer import ResultWriter, build_execution_metadata
from tests.unit.test_s91_m07_execution_fence import _policy

SENTINEL = "PRIVATE_request_prompt_header_body_reasoning"


def sdk_error(**body: Any) -> BadRequestError:
    request = httpx.Request("POST", "https://example.test/" + SENTINEL,
                            headers={"Authorization": SENTINEL}, content=SENTINEL)
    return BadRequestError(SENTINEL, response=httpx.Response(
        400, request=request, headers={"x-request-id": SENTINEL}), body=body)


@pytest.mark.parametrize(("code", "subtype"), [
    ("context_length_exceeded", "context_limit"),
    ("unsupported_parameter", "unsupported_parameter"),
    ("unsupported_value", "unsupported_value"),
    ("content_policy_violation", "content_rejection"),
    ("invalid_json", "invalid_json"),
    ("invalid_message", "invalid_message_structure"),
    ("invalid_request_error", "unknown"),
    (SENTINEL, "unknown"),
    ({"secret": SENTINEL}, "unknown"),
])
def test_only_fixed_rejection_labels_survive_sdk_projection(code: Any, subtype: str) -> None:
    error = sdk_error(code=code, param=SENTINEL, type=SENTINEL, message=SENTINEL)
    diagnostic = request_failure(error, model=SENTINEL, stage="synthesis")
    assert diagnostic["rejection"]["subtype"] == subtype
    assert diagnostic["rejection"]["parameter"] == "unknown"
    assert diagnostic["http_status"] == 400 and diagnostic["retryable"] is False
    assert SENTINEL not in json.dumps(diagnostic)
    assert len(json.dumps(diagnostic)) < 800


@pytest.mark.parametrize(("message", "expected"), [
    ("Invalid message structure: " + SENTINEL, "invalid_message_structure"),
    ("Content policy violation: " + SENTINEL, "content_rejection"),
    ("Unsupported parameter reasoning_effort: " + SENTINEL, "unsupported_parameter"),
    ("Invalid JSON " + SENTINEL, "invalid_json"),
    ("Maximum context length exceeded " + SENTINEL, "context_limit"),
    ("Content policy and invalid JSON " + SENTINEL, "unknown"),
    (SENTINEL, "unknown"),
])
def test_generic_rejection_has_bounded_explicit_evidence(message: str, expected: str) -> None:
    error = sdk_error(code="invalid_request_error", param="messages[12].content")
    error.message = message
    diagnostic = request_failure(error, model="deepseek-flash", stage="synthesis")
    assert diagnostic["rejection"]["subtype"] == expected
    assert diagnostic["rejection"]["parameter"] == "messages.content"
    assert SENTINEL not in json.dumps(diagnostic)


@pytest.mark.parametrize("secondary_failure", [True, False])
def test_real_sdk_failed_recovery_is_distinct_from_unknown_finish(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, secondary_failure: bool,
) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 2 or (len(calls) == 3 and secondary_failure):
            return httpx.Response(400, json={"error": {
                "code": "invalid_request_error", "type": "invalid_request_error",
                "message": "Invalid message structure: " + SENTINEL,
                "param": "messages[2].content",
            }}, headers={"x-request-id": SENTINEL})
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1,
            "model": "deepseek-flash", "choices": [{"index": 0, "message": {
                "role": "assistant", "content": '<code>print("retained action")</code>'
                if len(calls) == 1 else "unproven output",
            }, "finish_reason": "stop" if len(calls) == 1 else None}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5} if len(calls) == 1 else None,
        })

    client = OpenAI(api_key="offline", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(OpenAIServerModel, "create_client", lambda self: client)
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    model = OpenAIServerModel(model_id="deepseek-flash", api_key="offline")
    model.retry = False
    observe_model_requests(model, {"provider": "deepseek", "requested_model": "deepseek-flash"})
    result = run_codeagent_synthesis(system_instructions="Report.",
                                    user_directive=SENTINEL, model=model, max_steps=5)
    assert len(calls) == 3, "paid memory permits one finalization, never a fresh research run"
    assert "retained action" in calls[-1].content.decode()
    assert result.generation_recovery_result == (
        "provider_failed" if secondary_failure else "unknown_finish"
    )
    assert not result.output_markdown and result.error
    assert not result.token_usage_complete
    telemetry = _build_synthesis_telemetry(result)
    assert telemetry["generation_recovery_result"] == result.generation_recovery_result
    metadata = build_execution_metadata({"synthesis_telemetry": telemetry}, 1.0).to_dict()
    restored = json.loads(json.dumps(metadata))
    events = restored["synthesis_telemetry"]["provider_attempts"]
    assert [e["stage"] for e in events] == [
        "synthesis", "synthesis", "retained_memory_finalization"
    ]
    assert summarize_usage(events)["requests"] == 3
    assert summarize_usage(events)["observed"]["total"] == 15
    assert summarize_usage(events)["requests_with_missing_or_inconsistent_usage"] == 2
    assert events[1]["failure"]["rejection"]["subtype"] == "invalid_message_structure"
    assert events[1]["failure"]["request_shape"]["message_count"] > 0
    assert SENTINEL not in json.dumps(restored) and SENTINEL not in caplog.text


def _blocked_state_in_child() -> dict[str, Any]:
    """The production state route survives cleanup without a report artifact."""
    artifact_dir = Path(os.environ["DEEPSEARCH_REPORT_DIR"])
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "attempt-only.txt").write_text("discard me")
    return {"synthesis_telemetry": {"provider_attempts": [{"failure": request_failure(
        sdk_error(code="unsupported_parameter", param="tools"),
        model="deepseek-flash", stage="retained_memory_finalization",
    )}], "generation_recovery_result": "provider_failed"}, "mission_outcome": "blocked"}


def test_diagnostic_state_survives_real_spawn_and_sandbox_cleanup(tmp_path: Path) -> None:
    async def exercise() -> dict[str, Any]:
        executor = MissionProcessExecutor(_policy(), attempt_root=tmp_path)
        try:
            state = await executor.execute(_blocked_state_in_child, (), job_id="s96-diagnostics",
                                       hard_deadline_monotonic=time.monotonic() + 60)
            sandbox = executor.sandbox_path("s96-diagnostics")
            assert sandbox is not None and (sandbox / "reports/attempt-only.txt").exists()
            metadata = build_execution_metadata(state, 1.0)
            db = AsyncMock()
            db.fetchval.return_value = "mission-s96"
            writer = ResultWriter(db)
            now = datetime.now(timezone.utc)
            lease = LeaseAttempt("offline", "offline-token", now, now, now, 1)
            assert await writer.mark_failed(
                "mission-s96", "codeagent_synthesis_failed", lease=lease,
                job_id="s96-diagnostics", execution_metadata=metadata,
            )
            sql, *params = db.fetchval.call_args.args
            assert sql == writer.MARK_BLOCKED_SQL
            persisted = json.loads(params[5])
        finally:
            executor.cleanup("s96-diagnostics")
        assert not list(tmp_path.iterdir())
        return persisted

    metadata = asyncio.run(exercise())
    failure = metadata["synthesis_telemetry"]["provider_attempts"][0]["failure"]
    assert failure["rejection"]["subtype"] == "unsupported_parameter"
    assert SENTINEL not in json.dumps(metadata)


def test_shape_is_bounded_and_nontext_content_never_becomes_a_string() -> None:
    from deepsearch.llm.usage import usage_event

    record = usage_event(None, config={}, stage="synthesis", started_at="offline", elapsed=0,
                         error=sdk_error(), request={
                             "messages": [{"role": SENTINEL, "content": {"secret": SENTINEL}}]
                             * 5000,
                             "tools": [{"name": SENTINEL}] * 5000,
                         })
    shape = record["failure"]["request_shape"]
    assert shape["clipped"] and shape["message_count"] == 4096
    assert shape["nontext_or_unknown_messages"] == 4096
    assert shape["text_characters"] == 0
    assert SENTINEL not in json.dumps(record)


def test_missing_shape_is_unknown_not_zero() -> None:
    from deepsearch.llm.usage import usage_event

    record = usage_event(None, config={}, stage="compiler", started_at="offline", elapsed=0,
                         error=sdk_error())
    assert record["failure"]["request_shape"] == {"version": 1, "available": False}
