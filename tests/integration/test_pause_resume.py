"""Integration tests for checkpoint controls and pause/resume helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pytest

from deepsearch.runner import execute_research_mission


class _StubWorkflow:
    """Workflow stub that mirrors LangGraph's async interface."""

    def __init__(self) -> None:
        self.invocations = 0
        self.last_config: Dict[str, Any] | None = None

    async def ainvoke(
        self,
        state: Dict[str, Any],
        *,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self.invocations += 1
        self.last_config = config
        return dict(state)


def _mission_path() -> Path:
    return Path("tests/fixtures/missions/test_mission.yaml").resolve()


@pytest.mark.asyncio
async def test_resume_and_checkpoint_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner should coordinate checkpoint utilities when pause/resume flags change."""

    workflow = _StubWorkflow()
    captured_checkpointer: List[Any] = []

    def fake_build_workflow(*, checkpointer: Any | None = None) -> _StubWorkflow:
        captured_checkpointer.append(checkpointer)
        return workflow

    class _StubSaver:
        pass

    stub_saver = _StubSaver()
    clear_calls: List[tuple[str, Any | None]] = []
    history_calls: List[tuple[str, Any | None]] = []

    async def fake_clear(
        thread_id: str,
        *,
        checkpointer: Any | None = None,
        **_: Any,
    ) -> None:
        clear_calls.append((thread_id, checkpointer))

    async def fake_history(
        thread_id: str,
        *,
        checkpointer: Any | None = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        history_calls.append((thread_id, checkpointer))
        return [
            {
                "checkpoint_id": "cp-1",
                "parent_checkpoint_id": None,
                "metadata": {"step": "research"},
            }
        ]

    @asynccontextmanager
    async def fake_create_checkpointer(*_: Any, **__: Any):
        yield stub_saver

    monkeypatch.setattr("deepsearch.agent.graph.build_workflow", fake_build_workflow)
    monkeypatch.setattr("deepsearch.runner.create_checkpointer", fake_create_checkpointer)
    monkeypatch.setattr("deepsearch.runner.clear_thread_checkpoints", fake_clear)
    monkeypatch.setattr("deepsearch.runner.get_checkpoint_history", fake_history)

    result = await execute_research_mission(
        _mission_path(),
        resume=False,
        clear_checkpoints=True,
        show_checkpoints=True,
    )

    assert workflow.invocations == 1
    assert captured_checkpointer[0] is stub_saver
    assert workflow.last_config is not None
    assert workflow.last_config["configurable"]["thread_id"] == result["thread_id"]

    assert len(clear_calls) == 2
    assert clear_calls[0][0] == result["thread_id"]
    assert clear_calls[1][0] == result["thread_id"]
    assert all(call[1] is stub_saver for call in clear_calls)

    assert history_calls == [(result["thread_id"], stub_saver)]
    assert result["checkpoint_history"][0]["checkpoint_id"] == "cp-1"
    assert result["persistence_enabled"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("from_request", [False, True])
async def test_checkpoint_failure_after_execution_never_replays_paid_graph(
    monkeypatch: pytest.MonkeyPatch, from_request: bool,
) -> None:
    """Checkpoint errors cannot prove that the failed invocation spent nothing."""
    from deepsearch.api.models.requests import ExecuteMissionRequest
    from deepsearch.runner import run_research_from_request

    calls: list[str] = []
    closed: list[bool] = []

    class FailedWorkflow:
        async def ainvoke(self, state, **kwargs):
            calls.append(state["mission_id"])
            raise AttributeError("Connection has no attribute is_alive")

    @asynccontextmanager
    async def checkpoint(*args, **kwargs):
        try:
            yield object()
        finally:
            closed.append(True)

    monkeypatch.setattr("deepsearch.agent.graph.build_workflow", lambda **kw: FailedWorkflow())
    monkeypatch.setattr("deepsearch.runner.create_checkpointer", checkpoint)
    with pytest.raises(AttributeError, match="is_alive"):
        if from_request:
            await run_research_from_request(ExecuteMissionRequest(
                mission_id="LOCAL-PAID-GUARD", title="Paid guard", objectives=["Research"],
            ))
        else:
            await execute_research_mission(_mission_path(), resume=True)
    assert len(calls) == 1
    assert closed == [True]
