"""Integration tests for the DeepSearch mission runner and CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from deepsearch import cli
from deepsearch.runner import execute_research_mission


class _DummyWorkflow:
    """Small async stand-in for the LangGraph application."""

    def __init__(self) -> None:
        self.invocations = 0

    async def ainvoke(
        self,
        state: Dict[str, Any],
        *,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self.invocations += 1
        updated = dict(state)
        updated["research_loop_count"] = state.get("research_loop_count", 0) + 1
        updated["findings"] = ["LangGraph enables stateful research loops."]
        updated["sources_found"] = [
            {"title": "LangGraph Docs", "url": "https://example.com/langgraph"}
        ]
        return updated


@pytest.fixture(autouse=True)
def isolated_report_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ensure integration tests write reports into a temp directory."""

    target = tmp_path / "reports"
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(target))
    return target


@pytest.fixture
def dummy_workflow(monkeypatch: pytest.MonkeyPatch) -> _DummyWorkflow:
    """Patch the graph builder to return a deterministic workflow."""

    workflow = _DummyWorkflow()
    monkeypatch.setattr("deepsearch.agent.graph.build_workflow", lambda *_, **__: workflow)
    return workflow


def _mission_path() -> Path:
    return Path("tests/fixtures/missions/test_mission.yaml").resolve()


@pytest.mark.asyncio
async def test_execute_research_mission_generates_report(
    dummy_workflow: _DummyWorkflow,
) -> None:
    """Runner should produce a report and include metadata."""

    mission_file = _mission_path()
    result = await execute_research_mission(mission_file)

    assert result["status"] == "complete"
    assert result["loops_executed"] == 1
    assert result["sources_count"] == 1
    assert dummy_workflow.invocations == 1

    report_path = Path(result["report_path"])
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Test Research Mission" in content
    assert "## Executive Summary" in content


def test_cli_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI dry-run mode should validate missions without execution."""

    mission_file = _mission_path()
    exit_code = cli.main([str(mission_file), "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Dry Run Complete" in output
    assert "Mission ID" in output
