"""Hybrid quality evaluation built from Sprint 02 research."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Sequence

from deepsearch.agent.state import AgentState
from deepsearch.data_paths import data_path
from deepsearch.mission.contract import resolve_research_objectives
from deepsearch.validation.citation_rate import CitationRateResult, calculate_citation_rate

logger = logging.getLogger(__name__)

# Timeout for external LLM-backed DeepEval evaluations (seconds).
# DeepEval calls OpenAI under the hood; long reports can exceed default timeouts.
DEEPEVAL_TIMEOUT_SECONDS = float(os.getenv("DEEPEVAL_TIMEOUT_SECONDS", "180"))


FaithfulnessFn = Callable[[str, Sequence[str]], Awaitable[float] | float]
RelevanceFn = Callable[[str, str], Awaitable[float] | float]
CitationFn = Callable[[str], Awaitable[CitationRateResult] | CitationRateResult]


@dataclass(slots=True)
class QualityThresholds:
    """Threshold configuration pulled from MASTER_CONTEXT.research_quality."""

    faithfulness: float = 0.9
    relevance: float = 0.8
    citation_rate: float = 0.8

    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "citation_rate": self.citation_rate,
        }


@dataclass(slots=True)
class QualityScores:
    """Scores emitted by the hybrid framework."""

    faithfulness: float
    relevance: float
    citation_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "relevance": round(self.relevance, 4),
            "citation_rate": round(self.citation_rate, 4),
        }


@dataclass(slots=True)
class PerformanceMetrics:
    """Operational metrics captured during mission execution."""

    total_duration_seconds: float
    loops_executed: int
    api_calls: Dict[str, int]
    token_usage: Dict[str, int]
    estimated_cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "loops_executed": int(self.loops_executed),
            "api_calls": dict(self.api_calls),
            "token_usage": dict(self.token_usage),
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }


@dataclass(slots=True)
class ValidationResult:
    """Structured payload stored on disk for future regressions."""

    mission_id: str
    mission_title: str
    report_path: str
    timestamp: str
    quality_scores: QualityScores
    thresholds: QualityThresholds
    passes_standards: Dict[str, bool]
    citation_details: CitationRateResult
    performance_metrics: PerformanceMetrics
    reflection_scores: Mapping[str, Any]
    mission_objectives: Sequence[str]
    sources_evaluated: int
    overall_pass: bool
    output_path: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "mission_title": self.mission_title,
            "report_path": self.report_path,
            "timestamp": self.timestamp,
            "quality_scores": self.quality_scores.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "passes_standards": dict(self.passes_standards),
            "performance_metrics": self.performance_metrics.to_dict(),
            "reflection_scores": dict(self.reflection_scores or {}),
            "mission_objectives": list(self.mission_objectives),
            "sources_evaluated": self.sources_evaluated,
            "overall_pass": self.overall_pass,
            "citation_details": asdict(self.citation_details),
            "output_path": self.output_path,
        }


class HybridEvaluator:
    """Coordinate hybrid faithfulness/relevance/citation scoring."""

    def __init__(
        self,
        *,
        thresholds: QualityThresholds | None = None,
        output_dir: Path | None = None,
        faithfulness_fn: FaithfulnessFn | None = None,
        relevance_fn: RelevanceFn | None = None,
        citation_fn: CitationFn | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._thresholds = thresholds or QualityThresholds()
        self._output_dir = Path(output_dir) if output_dir is not None else data_path("validation")
        self._faithfulness_fn = faithfulness_fn or _default_faithfulness
        self._relevance_fn = relevance_fn or _default_relevance
        self._citation_fn = citation_fn or _default_citation
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

    async def evaluate(
        self,
        *,
        state: AgentState,
        report_path: str,
        performance_metrics: PerformanceMetrics,
        report_text: str | None = None,
    ) -> ValidationResult:
        """Run the full hybrid evaluation and persist the output."""

        mission_id = state.get("mission_id") or "UNKNOWN"
        mission_context = state.get("mission_context") or {}
        mission_title = str(
            mission_context.get("title")
            or mission_context.get("summary")
            or mission_id
        )
        objectives = resolve_research_objectives(
            state.get("mission_contract"),
            fallback=state.get("mission_objectives") or [],
        )
        text = report_text or _read_report(report_path)
        source_contexts = _build_source_contexts([])  # S83 m06: sources_found stripped at S78 (no writer)
        objective_prompt = (
            "\n".join(objectives)
            or mission_context.get("background")
            or mission_title
        )

        logger.info("Evaluating mission %s – Hybrid validation starting", mission_id)
        faithfulness = await self._invoke_handler(self._faithfulness_fn, text, source_contexts)
        relevance = await self._invoke_handler(self._relevance_fn, objective_prompt, text)
        citation_details = await self._invoke_handler(self._citation_fn, text)

        scores = QualityScores(
            faithfulness=float(_clamp_score(faithfulness)),
            relevance=float(_clamp_score(relevance)),
            citation_rate=float(_clamp_score(citation_details.citation_rate)),
        )
        passes = {
            "faithfulness": scores.faithfulness >= self._thresholds.faithfulness,
            "relevance": scores.relevance >= self._thresholds.relevance,
            "citation_rate": scores.citation_rate >= self._thresholds.citation_rate,
        }
        overall_pass = all(passes.values())
        timestamp = self._clock().replace(microsecond=0).isoformat().replace("+00:00", "Z")
        reflection_scores = dict(state.get("reflection_scores") or {})

        result = ValidationResult(
            mission_id=mission_id,
            mission_title=mission_title,
            report_path=report_path,
            timestamp=timestamp,
            quality_scores=scores,
            thresholds=self._thresholds,
            passes_standards=passes,
            citation_details=citation_details,
            performance_metrics=performance_metrics,
            reflection_scores=reflection_scores,
            mission_objectives=objectives,
            sources_evaluated=len(source_contexts),
            overall_pass=overall_pass,
        )
        output_path = self._persist_result(result)
        result.output_path = str(output_path) if output_path else None
        logger.info("Hybrid validation complete for %s (overall_pass=%s)", mission_id, overall_pass)
        return result

    async def _invoke_handler(self, handler: Callable[..., Any], *args: Any) -> Any:
        value = handler(*args)
        if asyncio.iscoroutine(value):
            return await value
        return value

    def _persist_result(self, result: ValidationResult) -> Path | None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem guard
            logger.error("Unable to create validation directory %s: %s", self._output_dir, exc)
            return None
        safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", result.mission_id).strip("-") or "mission"
        safe_ts = result.timestamp.replace(":", "-")
        output_path = self._output_dir / f"{safe_id}_{safe_ts}_metrics.json"
        payload = result.to_dict()
        payload["validation_version"] = "hybrid-1.0"
        try:
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:  # pragma: no cover - filesystem guard
            logger.error("Failed to persist validation output: %s", exc)
            return None
        return output_path


async def _default_faithfulness(report_text: str, contexts: Sequence[str]) -> float:
    """Execute faithfulness check using DeepEval (more reliable than UQLM).

    NOTE: Switched from UQLM to DeepEval due to UQLM v0.4.1 API breaking changes.
    DeepEval's FaithfulnessMetric is battle-tested and works consistently.
    """

    contexts_list = list(contexts) or []

    def _score() -> float:
        try:
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "deepeval is not installed; run `pip install deepeval` to enable validation"
            ) from exc

        # DeepEval requires context list (retrieval_context)
        test_case = LLMTestCase(
            input="Research evaluation",  # Not used for faithfulness
            actual_output=report_text,
            retrieval_context=contexts_list
        )

        # Create metric and measure
        metric = FaithfulnessMetric(threshold=0.9)
        metric.measure(test_case)

        return float(metric.score)

    try:
        return float(await asyncio.wait_for(
            asyncio.to_thread(_score),
            timeout=DEEPEVAL_TIMEOUT_SECONDS,
        ))
    except asyncio.TimeoutError:
        logger.warning(
            "DeepEval faithfulness timed out after %.0fs — returning 0.0",
            DEEPEVAL_TIMEOUT_SECONDS,
        )
        return 0.0


async def _default_relevance(mission_objectives: str, report_text: str) -> float:
    """Execute DeepEval answer relevancy metric with Ollama/Gemini backends."""

    def _score() -> float:
        try:
            from deepeval.metrics import AnswerRelevancyMetric
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("deepeval is not installed; run `pip install deepeval`") from exc

        test_case = LLMTestCase(input=mission_objectives, actual_output=report_text)
        metric = AnswerRelevancyMetric(threshold=0.8)
        metric.measure(test_case)
        return float(metric.score)

    try:
        return float(await asyncio.wait_for(
            asyncio.to_thread(_score),
            timeout=DEEPEVAL_TIMEOUT_SECONDS,
        ))
    except asyncio.TimeoutError:
        logger.warning(
            "DeepEval relevance timed out after %.0fs — returning 0.0",
            DEEPEVAL_TIMEOUT_SECONDS,
        )
        return 0.0


async def _default_citation(report_text: str) -> CitationRateResult:
    return calculate_citation_rate(report_text)


def _read_report(path: str) -> str:
    report_path = Path(path)
    return report_path.read_text(encoding="utf-8")


def _build_source_contexts(sources: Iterable[Mapping[str, Any]]) -> Sequence[str]:
    contexts: list[str] = []
    for source in sources:
        snippet = str(source.get("snippet") or source.get("content") or "").strip()
        if snippet:
            contexts.append(snippet)
            continue
        url = source.get("url")
        if isinstance(url, str) and url:
            contexts.append(url)
    return contexts or [""]


def _clamp_score(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


__all__ = [
    "HybridEvaluator",
    "PerformanceMetrics",
    "QualityScores",
    "QualityThresholds",
    "ValidationResult",
]
