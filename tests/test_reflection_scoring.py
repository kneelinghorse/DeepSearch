"""Unit tests for the quantitative reflection scoring helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepsearch.agent.evaluation import (
    ObjectiveScore,
    ResearchQualityEvaluation,
    apply_gated_finalization,
    calculate_consensus_score,
    check_convergence,
    flatten_objective_scores,
)


def _build_evaluation(scores: list[float], *, overall: float = 0.0) -> ResearchQualityEvaluation:
    payload = {
        "overall_coverage_score": overall,
        "is_mission_complete": False,
        "confidence_score": 0.0,
        "aggregate_rationale": "Test rationale",
        "per_objective_scores": [
            {"objective_id": f"OBJ{index + 1:02d}", "score": score, "reasoning": "detail"}
            for index, score in enumerate(scores)
        ],
    }
    evaluation = ResearchQualityEvaluation.model_validate(payload)
    evaluation.recompute_overall()
    return evaluation


def test_discrete_score_scale_enforced_conservatively() -> None:
    """Scores should snap to the discrete scale while breaking ties downward."""

    evaluation = _build_evaluation([0.91, 0.74], overall=0.93)

    assert evaluation.per_objective_scores[0].score == 1.0
    assert evaluation.per_objective_scores[1].score == 0.7
    assert evaluation.overall_coverage_score == 0.8


def test_gated_finalization_rule() -> None:
    """Gated completion should only pass when both thresholds hold."""

    passing = _build_evaluation([1.0, 0.96, 0.96], overall=0.96)
    failing = _build_evaluation([1.0, 0.9, 0.8], overall=0.95)

    assert apply_gated_finalization(passing) is True
    assert apply_gated_finalization(failing) is False


def test_consensus_calculation_uses_total_runs() -> None:
    """Consensus confidence should divide by requested run count."""

    evaluations = [
        _build_evaluation([0.8], overall=0.8),
        _build_evaluation([0.8], overall=0.8),
        _build_evaluation([1.0], overall=1.0),
        _build_evaluation([0.5], overall=0.5),
        _build_evaluation([0.8], overall=0.8),
    ]

    consensus = calculate_consensus_score(evaluations, total_runs=5)

    assert consensus.score == 0.8
    assert consensus.confidence == pytest.approx(0.6)
    assert consensus.representative.overall_coverage_score == 0.8


def test_check_convergence_detects_plateau() -> None:
    """Convergence helper should fire once improvement drops below delta.

    B10.5: Updated to use window parameter. Default window=3, delta=0.05.
    """
    # 3 scores within delta=0.05 of each other and latest >= 0.7
    assert check_convergence([0.85, 0.858, 0.86]) is True
    # Scores too far apart (0.95 - 0.85 = 0.10 > 0.05)
    assert check_convergence([0.85, 0.90, 0.95]) is False


def test_flatten_objective_scores_serialises_models() -> None:
    """Flatten helper should expose JSON-compatible payloads."""

    scores = [
        ObjectiveScore(objective_id="OBJ01", score=1.0, reasoning="done"),
        ObjectiveScore(objective_id="OBJ02", score=0.8, reasoning="partial"),
    ]

    flattened = flatten_objective_scores(scores)

    assert flattened == [
        {"objective_id": "OBJ01", "score": 1.0, "reasoning": "done"},
        {"objective_id": "OBJ02", "score": 0.8, "reasoning": "partial"},
    ]


def test_evaluation_validation_requires_objectives() -> None:
    """Schema should reject payloads missing objective scores."""

    payload = {
        "overall_coverage_score": 0.8,
        "is_mission_complete": False,
        "confidence_score": 0.0,
        "aggregate_rationale": "Missing objectives",
        "per_objective_scores": [],
    }

    with pytest.raises(ValidationError):
        ResearchQualityEvaluation.model_validate(payload)
