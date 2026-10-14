"""Quantitative evaluation helpers for the reflection node."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from .coverage import SourceCoverageGateResult


def _force_synthesis_enabled() -> bool:
    # Default OFF: forced synthesis is a smoke-only override and should never
    # be the ambient routing mode for normal executions.
    value = os.getenv("DEEPSEARCH_FORCE_SYNTHESIS", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}

DISCRETE_SCORE_SCALE: tuple[float, float, float, float, float] = (
    1.0,
    0.8,
    0.7,
    0.5,
    0.0,
)


def _snap_to_scale(value: float) -> float:
    """Return the nearest discrete value from the supported scoring scale."""

    target = float(value)
    return min(
        DISCRETE_SCORE_SCALE,
        key=lambda candidate: (abs(candidate - target), candidate),
    )


class ObjectiveScore(BaseModel):
    """Score payload for an individual mission objective."""

    objective_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)

    @field_validator("score", mode="before")
    @classmethod
    def _enforce_scale(cls, value: float) -> float:
        return _snap_to_scale(float(value))


class ResearchQualityEvaluation(BaseModel):
    """Schema emitted by the reflection prompt (DRM.0.5 compliant)."""

    overall_coverage_score: float = Field(..., ge=0.0, le=1.0)
    is_mission_complete: bool
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    aggregate_rationale: str = Field(..., min_length=1)
    per_objective_scores: List[ObjectiveScore]

    @field_validator("overall_coverage_score", mode="before")
    @classmethod
    def _enforce_overall_scale(cls, value: float) -> float:
        return _snap_to_scale(float(value))

    @field_validator("per_objective_scores")
    @classmethod
    def _require_objectives(cls, value: List[ObjectiveScore]) -> List[ObjectiveScore]:
        if not value:
            raise ValueError("per_objective_scores must contain at least one entry")
        return value

    def recompute_overall(self) -> float:
        """Return the average of the objective scores following DRM.0.5."""

        if not self.per_objective_scores:
            return self.overall_coverage_score
        average = mean(score.score for score in self.per_objective_scores)
        snapped = _snap_to_scale(average)
        self.overall_coverage_score = snapped
        return snapped


@dataclass(slots=True)
class ConsensusResult:
    """Calculated mode score and supporting metadata for reflection decisions."""

    score: float
    confidence: float
    representative: ResearchQualityEvaluation


def calculate_consensus_score(
    evaluations: Sequence[ResearchQualityEvaluation],
    *,
    total_runs: int | None = None,
) -> ConsensusResult:
    """Return the mode score, agreement rate, and representative evaluation.

    B13.5: Conservative tie-breaking and representative selection.
    - When two scores tie for mode, the LOWER score wins (conservative)
    - Representative is the eval with the LOWEST min per-objective score
      among mode-matching candidates, for better gap identification

    Args:
        evaluations: Successful evaluation payloads.
        total_runs: Optional number of attempts used when computing the
            agreement rate. When omitted, the denominator defaults to the
            number of successful evaluations.
    """

    if not evaluations:
        raise ValueError("At least one evaluation result is required")
    scores = [round(eval_.recompute_overall(), 2) for eval_ in evaluations]
    counts = Counter(scores)
    top = counts.most_common()
    max_count = top[0][1]
    # B13.5: When multiple scores tie for mode, prefer the lower score
    tied_modes = [score for score, count in top if count == max_count]
    mode_score = min(tied_modes)
    mode_count = max_count
    denominator = total_runs if total_runs and total_runs > 0 else len(evaluations)
    confidence = mode_count / denominator if denominator else 0.0
    candidates = [
        evaluation
        for evaluation in evaluations
        if round(evaluation.overall_coverage_score, 2) == mode_score
    ]
    # B13.5: Pick the most conservative representative (lowest min per-objective)
    # This surfaces gaps more accurately for downstream query derivation
    representative = sorted(
        candidates,
        key=lambda eval_: (
            min((score.score for score in eval_.per_objective_scores), default=0.0),
            eval_.overall_coverage_score,
        ),
    )[0]
    return ConsensusResult(score=mode_score, confidence=confidence, representative=representative)


def apply_gated_finalization(evaluation: ResearchQualityEvaluation) -> bool:
    """Return True when both DRM.0.5 completion gates are satisfied."""

    if not evaluation.per_objective_scores:
        return False
    scores = [score.score for score in evaluation.per_objective_scores]
    overall_high = evaluation.overall_coverage_score > 0.95
    no_critical_gaps = min(scores) > 0.85
    return overall_high and no_critical_gaps


@dataclass(slots=True)
class ConvergenceParams:
    """Convergence parameters extracted from depth_config.

    B14.2: Single source of truth for the three convergence knobs used by
    graph.py (should_continue), nodes.py (reflection_node), and
    nodes.py (_determine_early_termination).
    """

    threshold: float
    window: int
    min_score: float


def extract_convergence_params(depth_config: Dict[str, Any]) -> ConvergenceParams:
    """Extract convergence parameters from a depth_config dict.

    B14.2: Replaces duplicated inline extraction at three call-sites.

    Args:
        depth_config: The depth tier configuration dictionary (from AgentState).

    Returns:
        ConvergenceParams with threshold, window, and min_score.
    """
    return ConvergenceParams(
        threshold=depth_config.get("convergence_threshold", 0.05),
        window=depth_config.get("convergence_window", 3),
        min_score=depth_config.get("convergence_min_score", 0.7),
    )


def check_convergence(
    score_history: Sequence[float],
    *,
    delta: float = 0.05,
    window: int = 3,
    min_score: float = 0.7,
) -> bool:
    """Return True when scores have converged at a satisfactory level.

    B7.5.1 FIX: Convergence only triggers when:
    1. Score improvement falls below delta over the entire window (stagnation)
    2. AND the latest score exceeds min_score (quality threshold)

    B10.5: Depth-tier aware convergence detection:
    - BASELINE: 5% threshold over 3 loops
    - DEEP: 4% threshold over 3 loops
    - ALPHA: 3% threshold over 4 loops

    Args:
        score_history: Sequence of overall coverage scores from reflection.
        delta: Maximum score delta for convergence (from depth_config.convergence_threshold).
        window: Number of consecutive loops to check (from depth_config.convergence_window).
        min_score: Minimum acceptable score to allow convergence.

    Returns:
        True if scores have stabilized at an acceptable level over the window.
    """
    if len(score_history) < window:
        return False

    # Get the last `window` scores
    recent_scores = score_history[-window:]
    latest = recent_scores[-1]

    # Check if latest score meets quality threshold
    if latest < min_score:
        return False

    # Check if all scores in window are within delta of the latest
    # This ensures sustained stability, not just pairwise comparison
    for score in recent_scores[:-1]:
        if abs(latest - score) > delta:
            return False

    return True


def flatten_objective_scores(scores: Iterable[ObjectiveScore]) -> List[dict]:
    """Return primitive dictionaries for AgentState storage."""

    serialised: List[dict] = []
    for item in scores:
        serialised.append(
            {
                "objective_id": item.objective_id,
                "score": item.score,
                "reasoning": item.reasoning,
            }
        )
    return serialised


@dataclass(slots=True)
class QualityGateResult:
    """Result of quality gate checks before finalization.

    B11.3: Deterministic quality gates that prevent finalization
    when output quality is below tier-specific thresholds.
    """

    passed: bool
    total_unique_concepts: int
    total_sources: int
    unique_domains: int
    unique_concepts_required: int
    sources_required: int
    domains_required: int
    failed_gates: List[str]

    @property
    def total_findings(self) -> int:
        """Compatibility alias retained during the unique-concept migration."""

        return self.total_unique_concepts

    @property
    def findings_required(self) -> int:
        """Compatibility alias retained during the unique-concept migration."""

        return self.unique_concepts_required

    @property
    def details(self) -> str:
        """Human-readable summary of gate check results."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Quality Gates: {status}",
            (
                "  Unique Concepts: "
                f"{self.total_unique_concepts}/{self.unique_concepts_required}"
            ),
            f"  Sources: {self.total_sources}/{self.sources_required}",
            f"  Unique Domains: {self.unique_domains}/{self.domains_required}",
        ]
        if self.failed_gates:
            lines.append(f"  Failed: {', '.join(self.failed_gates)}")
        return "\n".join(lines)


@dataclass(slots=True)
class RoutingDecision:
    """Actual post-reflection routing outcome used by the graph."""

    next_node: Literal["research", "finalize"]
    next_action: Literal["continue", "synthesize"]
    reason: str
    quality_gates_passed: bool
    quality_gate_failures: List[str]
    coverage_gates_passed: bool
    coverage_gate_failures: List[str]
    converged: bool
    loop_count: int
    max_loops: int
    min_loops: int
    circuit_breaker_active: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_node": self.next_node,
            "next_action": self.next_action,
            "reason": self.reason,
            "quality_gates_passed": self.quality_gates_passed,
            "quality_gate_failures": list(self.quality_gate_failures),
            "coverage_gates_passed": self.coverage_gates_passed,
            "coverage_gate_failures": list(self.coverage_gate_failures),
            "converged": self.converged,
            "loop_count": self.loop_count,
            "max_loops": self.max_loops,
            "min_loops": self.min_loops,
            "circuit_breaker_active": self.circuit_breaker_active,
        }


def check_quality_gates(
    concept_clusters: Sequence[Any] | int,
    sources: Sequence[dict],
    *,
    min_unique_concepts: int | None = None,
    min_total_findings: int | None = None,
    min_total_sources: int = 3,
    min_unique_domains: int = 2,
) -> QualityGateResult:
    """Check if research output meets tier-specific quality thresholds.

    B11.3: Deterministic quality gates prevent finalization when:
    - Unique concept count < min_unique_concepts
    - Total sources < min_total_sources
    - Unique domains < min_unique_domains

    Args:
        concept_clusters: Cluster payloads (preferred), raw findings, or a precomputed count.
        sources: List of source dictionaries with 'url' key.
        min_unique_concepts: Minimum unique concepts required (from depth_config).
        min_total_findings: Compatibility alias for older call sites.
        min_total_sources: Minimum sources required (from depth_config).
        min_unique_domains: Minimum unique domains required (from depth_config).

    Returns:
        QualityGateResult with pass/fail status and details.
    """
    from urllib.parse import urlparse

    required_unique_concepts = (
        int(min_unique_concepts)
        if min_unique_concepts is not None
        else int(min_total_findings if min_total_findings is not None else 5)
    )
    total_unique_concepts = _concept_count(concept_clusters)
    total_sources = len(sources)

    # Extract unique domains from source URLs
    domains: set[str] = set()
    for source in sources:
        url = source.get("url", "")
        if url:
            try:
                parsed = urlparse(url)
                if parsed.netloc:
                    # Remove www. prefix for better domain grouping
                    domain = parsed.netloc.lower()
                    if domain.startswith("www."):
                        domain = domain[4:]
                    domains.add(domain)
            except Exception:
                pass
    unique_domains = len(domains)

    # Check each gate
    failed_gates: List[str] = []
    if total_unique_concepts < required_unique_concepts:
        failed_gates.append(
            f"unique_concepts ({total_unique_concepts}<{required_unique_concepts})"
        )
    if total_sources < min_total_sources:
        failed_gates.append(f"sources ({total_sources}<{min_total_sources})")
    if unique_domains < min_unique_domains:
        failed_gates.append(f"domains ({unique_domains}<{min_unique_domains})")

    return QualityGateResult(
        passed=len(failed_gates) == 0,
        total_unique_concepts=total_unique_concepts,
        total_sources=total_sources,
        unique_domains=unique_domains,
        unique_concepts_required=required_unique_concepts,
        sources_required=min_total_sources,
        domains_required=min_unique_domains,
        failed_gates=failed_gates,
    )


def _concept_count(concept_clusters: Sequence[Any] | int) -> int:
    if isinstance(concept_clusters, int):
        return max(concept_clusters, 0)
    if not isinstance(concept_clusters, Sequence) or isinstance(concept_clusters, (str, bytes)):
        return 0
    return len(concept_clusters)


def determine_routing_decision(
    *,
    loop_count: int,
    max_loops: int,
    min_loops: int,
    circuit_breaker_active: bool,
    gate_result: QualityGateResult,
    coverage_gates: SourceCoverageGateResult,
    converged: bool,
) -> RoutingDecision:
    """Return the graph's actual routing outcome and its reason."""

    if _force_synthesis_enabled() and loop_count >= max(min_loops, 1):
        return RoutingDecision(
            next_node="finalize",
            next_action="synthesize",
            reason="forced_synthesis",
            quality_gates_passed=True,
            quality_gate_failures=[],
            coverage_gates_passed=True,
            coverage_gate_failures=[],
            converged=converged,
            loop_count=loop_count,
            max_loops=max_loops,
            min_loops=min_loops,
            circuit_breaker_active=False,
        )

    if circuit_breaker_active:
        return RoutingDecision(
            next_node="finalize",
            next_action="synthesize",
            reason="circuit_breaker_active",
            quality_gates_passed=gate_result.passed,
            quality_gate_failures=list(gate_result.failed_gates),
            coverage_gates_passed=coverage_gates.passed,
            coverage_gate_failures=list(coverage_gates.failed_gates),
            converged=converged,
            loop_count=loop_count,
            max_loops=max_loops,
            min_loops=min_loops,
            circuit_breaker_active=True,
        )

    if loop_count >= max_loops:
        return RoutingDecision(
            next_node="finalize",
            next_action="synthesize",
            reason=(
                "max_loops_reached_coverage_met"
                if coverage_gates.passed and gate_result.passed
                else "max_loops_reached_coverage_unmet"
            ),
            quality_gates_passed=gate_result.passed,
            quality_gate_failures=list(gate_result.failed_gates),
            coverage_gates_passed=coverage_gates.passed,
            coverage_gate_failures=list(coverage_gates.failed_gates),
            converged=converged,
            loop_count=loop_count,
            max_loops=max_loops,
            min_loops=min_loops,
            circuit_breaker_active=False,
        )

    if not gate_result.passed:
        return RoutingDecision(
            next_node="research",
            next_action="continue",
            reason="quality_gates_failed",
            quality_gates_passed=False,
            quality_gate_failures=list(gate_result.failed_gates),
            coverage_gates_passed=coverage_gates.passed,
            coverage_gate_failures=list(coverage_gates.failed_gates),
            converged=converged,
            loop_count=loop_count,
            max_loops=max_loops,
            min_loops=min_loops,
            circuit_breaker_active=False,
        )

    if not coverage_gates.passed:
        return RoutingDecision(
            next_node="research",
            next_action="continue",
            reason="coverage_gates_failed",
            quality_gates_passed=True,
            quality_gate_failures=[],
            coverage_gates_passed=False,
            coverage_gate_failures=list(coverage_gates.failed_gates),
            converged=converged,
            loop_count=loop_count,
            max_loops=max_loops,
            min_loops=min_loops,
            circuit_breaker_active=False,
        )

    if converged and loop_count >= min_loops:
        return RoutingDecision(
            next_node="finalize",
            next_action="synthesize",
            reason="converged_with_coverage",
            quality_gates_passed=True,
            quality_gate_failures=[],
            coverage_gates_passed=True,
            coverage_gate_failures=[],
            converged=True,
            loop_count=loop_count,
            max_loops=max_loops,
            min_loops=min_loops,
            circuit_breaker_active=False,
        )

    if converged:
        return RoutingDecision(
            next_node="research",
            next_action="continue",
            reason="minimum_loops_not_reached",
            quality_gates_passed=True,
            quality_gate_failures=[],
            coverage_gates_passed=True,
            coverage_gate_failures=[],
            converged=True,
            loop_count=loop_count,
            max_loops=max_loops,
            min_loops=min_loops,
            circuit_breaker_active=False,
        )

    return RoutingDecision(
        next_node="research",
        next_action="continue",
        reason="not_yet_converged",
        quality_gates_passed=True,
        quality_gate_failures=[],
        coverage_gates_passed=True,
        coverage_gate_failures=[],
        converged=False,
        loop_count=loop_count,
        max_loops=max_loops,
        min_loops=min_loops,
        circuit_breaker_active=False,
    )


__all__ = [
    "ConvergenceParams",
    "ObjectiveScore",
    "ResearchQualityEvaluation",
    "ConsensusResult",
    "RoutingDecision",
    "QualityGateResult",
    "calculate_consensus_score",
    "apply_gated_finalization",
    "check_convergence",
    "check_quality_gates",
    "determine_routing_decision",
    "extract_convergence_params",
    "flatten_objective_scores",
    "ValidationError",
]
