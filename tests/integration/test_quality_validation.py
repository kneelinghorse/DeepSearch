"""Integration tests for the hybrid quality validation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from deepsearch.agent.state import AgentState, empty_state
from deepsearch.runner import execute_research_mission
from deepsearch.validation import (
    HybridEvaluator,
    PerformanceMetrics,
    QualityScores,
    QualityThresholds,
    ValidationResult,
    calculate_citation_rate,
)
from deepsearch.validation.citation_rate import CitationRateResult


class _DummyWorkflow:
    """Deterministic LangGraph stand-in used by validation tests."""

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
        updated["findings"] = ["Test finding"]
        updated["sources_found"] = [
            {
                "title": "Source A",
                "snippet": "Sample snippet [1].",
                "url": "https://example.com/a",
            }
        ]
        updated["reflection_scores"] = {"overall_coverage_score": 0.92}
        updated["report_path"] = state.get("report_path")
        return updated


@pytest.fixture(autouse=True)
def isolated_report_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ensure integration tests write artifacts into a temporary directory."""

    reports = tmp_path / "reports"
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(reports))
    return reports


@pytest.fixture
def dummy_workflow(monkeypatch: pytest.MonkeyPatch) -> _DummyWorkflow:
    """Patch the graph builder so missions do not hit real LLM/tooling."""

    workflow = _DummyWorkflow()
    monkeypatch.setattr("deepsearch.agent.graph.build_workflow", lambda *_, **__: workflow)
    return workflow


@pytest.mark.asyncio
async def test_hybrid_evaluator_persists_results(tmp_path: Path) -> None:
    """Hybrid evaluator should persist metrics and flag pass/fail states."""

    report_path = tmp_path / "report.md"
    report_path.write_text("Sentence one [1]. Sentence two [2].", encoding="utf-8")
    state = empty_state(mission_id="TEST.1", mission_objectives=["Objective A"])
    state["mission_context"]["title"] = "Validation Mission"
    state["sources_found"] = [{"title": "Doc", "snippet": "Context paragraph", "url": "https://example.com"}]

    evaluator = HybridEvaluator(
        thresholds=QualityThresholds(faithfulness=0.7, relevance=0.7, citation_rate=0.7),
        output_dir=tmp_path / "validation",
        faithfulness_fn=lambda *_: 0.91,
        relevance_fn=lambda *_: 0.82,
    )
    metrics = PerformanceMetrics(
        total_duration_seconds=45.0,
        loops_executed=2,
        api_calls={"llm_research": 2},
        token_usage={"prompt_tokens": 1200, "completion_tokens": 800},
        estimated_cost_usd=0.001,
    )
    result = await evaluator.evaluate(
        state=state,
        report_path=str(report_path),
        report_text=report_path.read_text(encoding="utf-8"),
        performance_metrics=metrics,
    )

    assert result.overall_pass is True
    assert result.output_path is not None
    payload = json.loads(Path(result.output_path).read_text(encoding="utf-8"))
    assert payload["quality_scores"]["faithfulness"] == pytest.approx(0.91, abs=1e-4)
    assert payload["quality_scores"]["relevance"] == pytest.approx(0.82, abs=1e-4)
    assert payload["quality_scores"]["citation_rate"] >= 0.7
    assert payload["performance_metrics"]["loops_executed"] == 2


def test_calculate_citation_rate_detects_gaps() -> None:
    """Citation helper should flag uncited sentences."""

    text = "Claim one [1]. Claim two. Claim three (Doe, 2024)."
    result = calculate_citation_rate(text, threshold=0.8)

    assert result.total_sentences == 3
    assert result.cited_sentences == 2
    assert result.citation_rate == pytest.approx(2 / 3, rel=1e-3)
    assert result.meets_threshold is False
    assert "Claim two" in result.sentences_without_citations[0]


@pytest.mark.asyncio
async def test_runner_emits_validation_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dummy_workflow: _DummyWorkflow,
) -> None:
    """Runner should attach validation metadata when --validate is enabled."""

    mission_file = Path("tests/fixtures/missions/test_mission.yaml").resolve()

    async def _fake_evaluate(
        self: HybridEvaluator,
        *,
        state: AgentState,
        report_path: str,
        performance_metrics: PerformanceMetrics,
        report_text: str | None = None,
    ) -> ValidationResult:
        scores = QualityScores(faithfulness=0.93, relevance=0.85, citation_rate=0.88)
        thresholds = QualityThresholds()
        citation = CitationRateResult(
            citation_rate=0.88,
            total_sentences=10,
            cited_sentences=9,
            meets_threshold=True,
            sentences_without_citations=[],
        )
        result = ValidationResult(
            mission_id=state.get("mission_id", "TEST"),
            mission_title="Fixture Mission",
            report_path=report_path,
            timestamp="2025-01-01T00:00:00Z",
            quality_scores=scores,
            thresholds=thresholds,
            passes_standards={
                "faithfulness": True,
                "relevance": True,
                "citation_rate": True,
            },
            citation_details=citation,
            performance_metrics=performance_metrics,
            reflection_scores={"overall_coverage_score": 0.9},
            mission_objectives=state.get("mission_objectives") or [],
            sources_evaluated=1,
            overall_pass=True,
        )
        result.output_path = str(tmp_path / "fake_validation.json")
        return result

    monkeypatch.setattr(
        "deepsearch.validation.hybrid_evaluator.HybridEvaluator.evaluate",
        _fake_evaluate,
    )

    result = await execute_research_mission(mission_file, validate=True)

    assert result["validation"] is not None
    assert result["validation"]["quality_scores"]["faithfulness"] == pytest.approx(0.93, abs=1e-5)
    assert result["performance_metrics"]["loops_executed"] == 1
    assert result["validation"]["output_path"].endswith("fake_validation.json")
