"""Offline orchestration lifecycle tests, not claims of model compliance."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
from smolagents.memory import ActionStep, Timing
from smolagents.models import ChatMessage, MessageRole, Model, TokenUsage

from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.completion_control import CompletionController
from deepsearch.agent.execution_control import ExecutionCancelledError
from deepsearch.agent.tools import source_fetch as fetch_module
from deepsearch.agent.tools import web_search as search_module

DRAFT = "## Findings\nA sourced finding [cite:https://example.org/evidence]."


def _step(number: int, *, output: str | None = "print('work')") -> ActionStep:
    return ActionStep(step_number=number, timing=Timing(start_time=0), model_output=output)


class _LifecycleModel(Model):
    """Scripted provider boundary driving the real installed CodeAgent lifecycle."""

    def __init__(self, responses: list[str | Exception]) -> None:
        super().__init__(model_id="offline-lifecycle")
        self.responses = iter(responses)
        self.requests: list[list[Any]] = []

    def generate(self, messages: list[Any], **kwargs: Any) -> ChatMessage:
        self.requests.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return ChatMessage(
            role=MessageRole.ASSISTANT, content=response,
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]),
        )


def _code(action: str) -> str:
    return "<code>\n" + action + "\n</code>"


def test_callback_transitions_before_step_30_and_checks_cited_draft() -> None:
    control = CompletionController(40, ["Markdown report"])
    control.after_step(_step(28), {})
    assert control.phase == "research" and control.remaining_steps == 12
    step = _step(29)
    control.after_step(step, {"report_draft": DRAFT})
    assert control.phase == "assembly" and control.remaining_steps == 11
    assert control.transitions[-1] == {"phase": "assembly", "next_step": 30}
    assert control.draft_present_at_assembly is True
    assert control.coverage["missing"] == []
    assert "remaining=11" in step.observations
    control.after_step(_step(38), {})
    assert control.phase == "finalization"
    assert control.retained_draft == DRAFT  # A later absent checkpoint cannot erase paid work.


def test_absent_draft_requires_named_bounded_gaps(monkeypatch) -> None:
    control = CompletionController(40)
    control.after_step(_step(29), {})
    assert control.draft_present_at_assembly is False
    calls: list[str] = []
    monkeypatch.setattr(search_module, "web_search", lambda q, n: calls.append(q) or [])
    bundle = runner.build_codeagent_tools(completion_controller=control)
    search = bundle.tools[0]
    assert search("more")[0]["error"] == "assembly_draft_required"
    bundle.tools[3]("report_draft", DRAFT)
    assert control.checks == 1 and control.coverage["assessment_status"] == "unassessed"
    assert search("more")[0]["error"] == "assembly_gap_required"
    for _ in range(3):
        assert search("specific question", gap="Evidence for a remaining caveat") == []
    assert search("more", gap="another caveat")[0]["error"] == "assembly_retrieval_budget_exhausted"
    assert len(calls) == 3
    assert control.blocked_retrievals == 3
    assert control.telemetry("pending")["coverage_check_status"] == "checked"


def test_finalization_still_allows_stored_passages_but_no_new_network(monkeypatch) -> None:
    control = CompletionController(40)
    calls: list[str] = []

    def fetch(url: str, timeout: float) -> dict:
        calls.append(url)
        return {"text": "evidence " * 5000, "status": "ok", "status_code": 200,
                "final_url": url, "fetched_at": "historical"}

    monkeypatch.setattr(fetch_module, "source_fetch", fetch)
    bundle = runner.build_codeagent_tools(completion_controller=control)
    tool = bundle.tools[1]
    tool("https://example.org/evidence")
    control.after_step(_step(38), {})
    assert tool("https://example.org/evidence", start=30000)["cache_hit"]
    assert tool("https://example.org/new", gap="caveat")["error"] == "finalization_phase"
    assert bundle.tools[2]("https://example.org/new", gap="caveat")["alive"] is None
    assert len(calls) == 1
    assert len(bundle.tool_outcomes()["ledger_records"]) == 1


def test_empty_generation_step_does_not_fabricate_paid_memory() -> None:
    control = CompletionController(40)
    step = _step(1, output=None)
    control.after_step(step, {})
    assert step.observations is None and control.completed_steps == 0
    assert control.telemetry("failed")["coverage_check_status"] == "not_checked"


@pytest.mark.parametrize("normal", [True, False])
def test_real_sdk_early_finish_and_exhaustion(monkeypatch, normal: bool) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    responses = [_code(f"final_answer({DRAFT!r})")] if normal else [
        _code("print('first action')"), _code("print('second action')"), DRAFT,
    ]
    model = _LifecycleModel(responses)
    result = runner.run_codeagent_synthesis(
        system_instructions="Complete research", user_directive="Task", model=model, max_steps=2,
        deliverable_spec=["Markdown report"],
    )
    assert result.error is None
    assert result.hit_max_steps is not normal
    assert result.self_terminated is normal
    assert len(model.requests) == (1 if normal else 3)
    control = result.completion_control
    expected_kind = "normal_final_answer" if normal else "step_limit_fallback"
    assert control["completion_kind"] == expected_kind
    assert control["observed_completed_steps"] == (1 if normal else 2)
    assert control["cited_draft_available"] and control["coverage_check_status"] == "checked"
    if not normal:
        assert result.step_count == 3  # Framework's synthetic fallback remains distinct.
        assert result.max_steps_fallback_complete


def test_real_sdk_generation_failure_finalizes_once_from_same_memory(monkeypatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    model = _LifecycleModel([
        _code(f"scratchpad_write('report_draft', {DRAFT!r})"),
        RuntimeError("provider unavailable"), DRAFT,
    ])
    result = runner.run_codeagent_synthesis(
        system_instructions="Complete research", user_directive="Task", model=model, max_steps=40,
    )
    assert result.generation_recovery_attempted
    assert result.generation_error_recovered
    assert len(model.requests) == 3  # No new agent research loop.
    assert result.completion_control["completion_kind"] == "generation_failure_recovery"
    assert result.completion_control["cited_draft_available"]
    assert result.completion_control["observed_completed_steps"] == 1
    assert "report_draft" in str(model.requests[-1])


def test_real_sdk_cancellation_does_not_enter_another_step(monkeypatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    event = threading.Event()

    class CancellingModel(_LifecycleModel):
        def generate(self, messages, **kwargs):
            message = super().generate(messages, **kwargs)
            event.set()
            return message

    model = CancellingModel([_code("print('retained work')")])
    with pytest.raises(ExecutionCancelledError):
        runner.run_codeagent_synthesis(
            system_instructions="Research", user_directive="Task", model=model,
            cancel_event=event,
        )
    assert len(model.requests) == 1


def test_real_sdk_step_30_receives_assembly_notice_and_tools_enforce_it(monkeypatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")

    def forbidden(*args, **kwargs):
        raise AssertionError("assembly request without draft must not search")

    monkeypatch.setattr(search_module, "web_search", forbidden)
    model = _LifecycleModel(
        [_code("print('retained action')")] * 29
        + [_code("print(web_search('unbounded next topic'))"),
           _code(f"scratchpad_write('report_draft', {DRAFT!r})\nfinal_answer({DRAFT!r})")]
    )
    result = runner.run_codeagent_synthesis(
        system_instructions="Research", user_directive="Task", model=model,
        deliverable_spec=["Markdown report"],
    )
    assert result.error is None and result.self_terminated
    assert result.step_count == 31 and not result.hit_max_steps
    assert "Phase=assembly; completed=29/40; remaining=11" in str(model.requests[29])
    control = result.completion_control
    assert control["blocked_retrievals"] == 1 and control["late_retrieval_calls"] == 0
    assert control["draft_present_at_assembly"] is False
    assert control["cited_draft_available"] and control["coverage_checks"] == 1


def test_last_cited_checkpoint_survives_overwrite_and_rejected_recovery(monkeypatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    model = _LifecycleModel([
        _code(f"scratchpad_write('report_draft', {DRAFT!r})"),
        _code("scratchpad_write('report_draft', 'unfinished replacement')"),
        RuntimeError("provider failed"), "Uncited incomplete answer",
    ])
    result = runner.run_codeagent_synthesis(
        system_instructions="Research", user_directive="Task", model=model,
    )
    assert len(model.requests) == 4
    assert result.error is not None and not result.generation_error_recovered
    assert result.recovered_from_scratchpad
    assert result.output_markdown == DRAFT
    assert result.completion_control["cited_draft_available"]
