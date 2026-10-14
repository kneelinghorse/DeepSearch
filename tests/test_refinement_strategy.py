"""Unit tests for the refinement helpers powering smart research loops."""

from __future__ import annotations

from deepsearch.agent import refinement


def test_analyze_objective_gaps_identifies_low_scores() -> None:
    """Gap analysis should surface objectives falling below the threshold."""

    scores = {
        "per_objective_scores": [
            {"objective_id": "OBJ01", "score": 1.0, "reasoning": "Complete"},
            {"objective_id": "OBJ02", "score": 0.4, "reasoning": "Needs depth"},
            {"objective_id": "OBJ03", "score": 0.75, "reasoning": "Almost there"},
        ]
    }

    gaps = refinement.analyze_objective_gaps(scores)

    assert [gap["objective_id"] for gap in gaps] == ["OBJ02", "OBJ03"]
    assert gaps[0]["priority"] == "high"


def test_generate_targeted_queries_deduplicates_previous() -> None:
    """Targeted query generation should avoid repeating earlier searches."""

    gaps = [
        {"objective_id": "OBJ02", "current_score": 0.5, "reasoning": "impact on reliability"},
        {"objective_id": "OBJ03", "current_score": 0.4, "reasoning": "scalability evidence"},
    ]
    mission_context = {"summary": "LangGraph research agent"}
    previous = ["langgraph research agent OBJ02 impact on reliability"]

    queries = refinement.generate_targeted_queries(
        gaps,
        mission_context,
        previous_queries=previous,
        objective_lookup=["Objective 1", "Objective 2", "Objective 3"],
    )

    assert queries
    assert all(query not in previous for query in queries)


def test_detect_stuck_objectives_flags_plateaus() -> None:
    """Stuck detection should mark objectives with negligible improvement."""

    history = [
        {"loop": 1, "scores": [{"objective_id": "OBJ01", "score": 0.3}]},
        {"loop": 2, "scores": [{"objective_id": "OBJ01", "score": 0.32}]},
        {"loop": 3, "scores": [{"objective_id": "OBJ01", "score": 0.34}]},
    ]

    stuck = refinement.detect_stuck_objectives(history)

    assert stuck == ["OBJ01"]


def test_prioritize_objectives_orders_scores() -> None:
    """Prioritization should sort by ascending score and label statuses."""

    scores = [
        {"objective_id": "OBJ01", "score": 0.95},
        {"objective_id": "OBJ02", "score": 0.4},
        {"objective_id": "OBJ03", "score": 0.85},
    ]

    prioritized = refinement.prioritize_objectives(scores)

    assert [item[0] for item in prioritized] == ["OBJ02", "OBJ03", "OBJ01"]
    assert prioritized[-1][2] == "complete"
