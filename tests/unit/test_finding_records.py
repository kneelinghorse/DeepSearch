"""Unit tests for structured finding record attribution."""

from __future__ import annotations

from deepsearch.agent.finding import build_finding_records


def test_build_finding_records_allows_strong_multi_objective_overlap() -> None:
    summary = (
        "Checkpoint durability and crash recovery trade-offs interact with "
        "concurrency behavior for long-running agents."
    )
    contract = {
        "objectives": [
            {
                "id": "objective-0",
                "text": "Compare checkpoint durability and crash recovery trade-offs",
            },
            {
                "id": "objective-1",
                "text": "Explain concurrency behavior for long-running agents",
            },
            {
                "id": "objective-2",
                "text": "Describe observability and debugging workflows",
            },
        ],
        "evidence_slots": [],
    }

    records = build_finding_records(
        [summary],
        sources=[],
        loop_index=0,
        mission_contract=contract,
        mission_objectives=[],
    )

    assert len(records) == 1
    assert records[0].objective_ids == ["objective-0", "objective-1"]


def test_build_finding_records_keeps_weak_match_single_objective() -> None:
    summary = "Behavior changed under load."
    contract = {
        "objectives": [
            {"id": "objective-0", "text": "Explain service behavior"},
            {"id": "objective-1", "text": "Summarize runtime behavior"},
        ],
        "evidence_slots": [],
    }

    records = build_finding_records(
        [summary],
        sources=[],
        loop_index=0,
        mission_contract=contract,
        mission_objectives=[],
    )

    assert len(records) == 1
    assert records[0].objective_ids == ["objective-0"]
