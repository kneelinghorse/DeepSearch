"""Unit tests for coverage-gate aggregation."""

from __future__ import annotations

from typing import Any

from deepsearch.agent.coverage import aggregate_source_coverage


def _slot_coverage(
    *,
    objective_statuses: dict[str, str] | None = None,
    entity_statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    objective_statuses = objective_statuses or {
        "slot-objective-0": "filled",
        "slot-objective-1": "filled",
    }
    entity_statuses = entity_statuses or {
        "slot-entity-saplma": "filled",
        "slot-entity-bm25-ranking": "filled",
    }
    slot_rows: list[dict[str, Any]] = []
    for slot_id, status in {**objective_statuses, **entity_statuses}.items():
        if slot_id.startswith("slot-objective-"):
            target = {"objective_id": slot_id.removeprefix("slot-")}
        else:
            target = {
                "entity": "SAPLMA" if slot_id.endswith("saplma") else "BM25 ranking"
            }
        reasons = [] if status == "filled" else (["1 source"] if status == "thin" else [])
        source_count = 2 if status == "filled" else (1 if status == "thin" else 0)
        slot_rows.append(
            {
                "slot_id": slot_id,
                "status": status,
                "target": target,
                "source_count": source_count,
                "domains": ["example.com"] if source_count else [],
                "reasons": reasons,
            }
        )

    return {
        "total_slots": len(slot_rows),
        "filled_slots": sum(1 for row in slot_rows if row["status"] == "filled"),
        "thin_slots": [row for row in slot_rows if row["status"] == "thin"],
        "unaddressed_slots": [row for row in slot_rows if row["status"] == "unaddressed"],
        "slots": slot_rows,
    }


def _contract(
    *,
    named_entities: list[str] | None = None,
    objectives: list[dict[str, Any]] | None = None,
    evidence_slots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "named_entities": list(named_entities or ["SAPLMA", "BM25 ranking"]),
        "objectives": list(
            objectives
            or [
                {"id": "objective-0", "text": "Explain the primary mechanism", "required": True},
                {"id": "objective-1", "text": "Compare retrieval behavior", "required": True},
            ]
        ),
        "evidence_slots": list(
            evidence_slots
            or [
                {"id": "slot-objective-0", "target": {"objective_id": "objective-0"}},
                {"id": "slot-objective-1", "target": {"objective_id": "objective-1"}},
                {"id": "slot-entity-saplma", "target": {"entity": "SAPLMA"}},
                {"id": "slot-entity-bm25-ranking", "target": {"entity": "BM25 ranking"}},
            ]
        ),
    }


def _state(
    *,
    cluster_summaries: list[dict[str, Any]] | None = None,
    working_set_titles: list[str] | None = None,
    per_objective_scores: list[dict[str, Any]] | dict[str, float] | None = None,
    slot_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entries = [
        {"source": {"title": title, "url": f"https://example.com/{idx}"}}
        for idx, title in enumerate(working_set_titles or ["BM25 ranking guide"], start=1)
    ]
    return {
        "cluster_summaries": list(
            cluster_summaries
            or [
                {
                    "representative_summary": "SAPLMA improves retrieval planning for audits.",
                    "citations": ["https://example.com/saplma"],
                }
            ]
        ),
        "working_set": {"entries": entries},
        "reflection_scores": {
            "per_objective_scores": per_objective_scores
            or [
                {"objective_id": "OBJ01", "score": 0.92},
                {"objective_id": "OBJ02", "score": 0.84},
            ]
        },
        "slot_coverage": slot_coverage
        or _slot_coverage(),
    }


def test_aggregate_source_coverage_all_gates_pass() -> None:
    result = aggregate_source_coverage(_state(), _contract())

    assert result.passed is True
    assert result.failed_gates == []
    assert result.entity.covered == ["SAPLMA", "BM25 ranking"]
    assert result.objective.per_objective_scores == {
        "objective-0": 0.92,
        "objective-1": 0.84,
    }
    assert result.objective.uncovered_objective_ids == []
    assert result.slot.unaddressed_slot_ids == []
    assert result.to_dict()["passed"] is True


def test_aggregate_source_coverage_missing_entity_also_fails_slot_gate() -> None:
    state = _state(
        slot_coverage=_slot_coverage(
            entity_statuses={
                "slot-entity-saplma": "filled",
                "slot-entity-bm25-ranking": "unaddressed",
            }
        )
    )

    result = aggregate_source_coverage(state, _contract())

    assert result.passed is False
    assert result.failed_gates == ["entity", "slot"]
    assert result.entity.covered == ["SAPLMA"]
    assert result.entity.uncovered == ["BM25 ranking"]
    assert result.entity.threshold == 1.0
    assert result.objective.passed is True
    assert result.slot.passed is False


def test_aggregate_source_coverage_objective_gate_fails_independently() -> None:
    state = _state(
        per_objective_scores=[
            {"objective_id": "OBJ01", "score": 0.92},
            {"objective_id": "OBJ02", "score": 0.45},
        ]
    )

    result = aggregate_source_coverage(state, _contract())

    assert result.passed is False
    assert result.failed_gates == ["objective"]
    assert result.objective.uncovered_objective_ids == ["objective-1"]
    assert result.entity.passed is True
    assert result.slot.passed is True


def test_aggregate_source_coverage_slot_gate_fails_independently() -> None:
    state = _state(
        slot_coverage=_slot_coverage(
            objective_statuses={
                "slot-objective-0": "filled",
                "slot-objective-1": "unaddressed",
            }
        )
    )

    result = aggregate_source_coverage(state, _contract())

    assert result.passed is False
    assert result.failed_gates == ["slot"]
    assert result.slot.unaddressed_slot_ids == ["slot-objective-1"]
    assert result.entity.passed is True
    assert result.objective.passed is True


def test_aggregate_source_coverage_degrades_gracefully_for_incomplete_contract() -> None:
    contract = {
        "named_entities": ["SAPLMA"],
        "objectives": [{"id": "objective-0", "required": True}],
    }

    result = aggregate_source_coverage(_state(), contract)

    assert result.passed is False
    assert result.failed_gates == ["contract_incomplete"]
    assert result.entity.passed is False
    assert result.objective.passed is False
    assert result.slot.passed is False


def test_entity_coverage_requires_slot_backed_evidence_even_if_mentions_exist() -> None:
    state = _state(
        cluster_summaries=[
            {
                "representative_summary": "SAPLMA and BM25 ranking are both mentioned here.",
                "citations": ["https://example.com/bm25-ranking-guide"],
            }
        ],
        working_set_titles=["SAPLMA BM25 ranking guide"],
        per_objective_scores=[],
        slot_coverage=_slot_coverage(
            objective_statuses={},
            entity_statuses={
                "slot-entity-saplma": "unaddressed",
                "slot-entity-bm25-ranking": "unaddressed",
            },
        ),
    )

    result = aggregate_source_coverage(state, _contract(objectives=[]))

    assert result.entity.passed is False
    assert result.entity.covered == []
    assert result.entity.uncovered == ["SAPLMA", "BM25 ranking"]


def test_entity_coverage_requires_all_required_backends() -> None:
    contract = _contract(
        named_entities=[
            "MemorySaver",
            "SqliteSaver",
            "AsyncSqliteSaver",
            "PostgresSaver",
        ],
        objectives=[],
        evidence_slots=[
            {"id": "slot-entity-memorysaver", "target": {"entity": "MemorySaver"}},
            {"id": "slot-entity-sqlitesaver", "target": {"entity": "SqliteSaver"}},
            {"id": "slot-entity-asyncsqlitesaver", "target": {"entity": "AsyncSqliteSaver"}},
            {"id": "slot-entity-postgressaver", "target": {"entity": "PostgresSaver"}},
        ],
    )
    state = _state(
        per_objective_scores=[],
        slot_coverage={
            "total_slots": 4,
            "filled_slots": 3,
            "thin_slots": [],
            "unaddressed_slots": [
                {
                    "slot_id": "slot-entity-postgressaver",
                    "target": {"entity": "PostgresSaver"},
                    "status": "unaddressed",
                    "source_count": 0,
                    "domains": [],
                    "reasons": [],
                }
            ],
            "slots": [
                {
                    "slot_id": "slot-entity-memorysaver",
                    "target": {"entity": "MemorySaver"},
                    "status": "filled",
                    "source_count": 2,
                    "domains": ["example.com"],
                    "reasons": [],
                },
                {
                    "slot_id": "slot-entity-sqlitesaver",
                    "target": {"entity": "SqliteSaver"},
                    "status": "filled",
                    "source_count": 2,
                    "domains": ["example.com"],
                    "reasons": [],
                },
                {
                    "slot_id": "slot-entity-asyncsqlitesaver",
                    "target": {"entity": "AsyncSqliteSaver"},
                    "status": "filled",
                    "source_count": 2,
                    "domains": ["example.com"],
                    "reasons": [],
                },
                {
                    "slot_id": "slot-entity-postgressaver",
                    "target": {"entity": "PostgresSaver"},
                    "status": "unaddressed",
                    "source_count": 0,
                    "domains": [],
                    "reasons": [],
                },
            ],
        },
    )

    result = aggregate_source_coverage(state, contract)

    assert result.entity.passed is False
    assert result.entity.covered == ["MemorySaver", "SqliteSaver", "AsyncSqliteSaver"]
    assert result.entity.uncovered == ["PostgresSaver"]


def test_objective_coverage_allows_optional_objectives_to_remain_unscored() -> None:
    state = _state(
        per_objective_scores=[
            {"objective_id": "OBJ01", "score": 0.85},
            {"objective_id": "OBJ02", "score": 0.90},
        ],
        slot_coverage=_slot_coverage(objective_statuses={}, entity_statuses={}),
    )
    contract = _contract(
        named_entities=[],
        objectives=[
            {"id": "objective-0", "text": "Primary", "required": True},
            {"id": "objective-1", "text": "Secondary", "required": True},
            {"id": "objective-2", "text": "Optional", "required": False},
        ],
        evidence_slots=[],
    )

    result = aggregate_source_coverage(state, contract)

    assert result.objective.passed is True
    assert result.objective.per_objective_scores == {
        "objective-0": 0.85,
        "objective-1": 0.90,
    }
    assert result.objective.uncovered_objective_ids == []


def test_objective_coverage_maps_obj_aliases_to_required_objectives_after_optional_umbrella() -> None:
    state = _state(
        per_objective_scores=[
            {"objective_id": "OBJ01", "score": 0.85},
            {"objective_id": "OBJ02", "score": 0.90},
        ],
        slot_coverage=_slot_coverage(objective_statuses={}, entity_statuses={}),
    )
    contract = _contract(
        named_entities=[],
        objectives=[
            {"id": "objective-0", "text": "Umbrella", "required": False},
            {"id": "objective-1", "text": "First required", "required": True},
            {"id": "objective-2", "text": "Second required", "required": True},
        ],
        evidence_slots=[],
    )

    result = aggregate_source_coverage(state, contract)

    assert result.objective.passed is True
    assert result.objective.per_objective_scores == {
        "objective-1": 0.85,
        "objective-2": 0.90,
    }
    assert "objective-0" not in result.objective.per_objective_scores
    assert result.objective.uncovered_objective_ids == []


def test_objective_coverage_fails_when_required_score_is_missing() -> None:
    state = _state(
        per_objective_scores=[{"objective_id": "OBJ01", "score": 0.90}],
        slot_coverage=_slot_coverage(objective_statuses={}, entity_statuses={}),
    )

    result = aggregate_source_coverage(
        state,
        _contract(named_entities=[], evidence_slots=[]),
    )

    assert result.objective.passed is False
    assert result.objective.uncovered_objective_ids == ["objective-1"]


def test_objective_coverage_accepts_adequate_score_at_gate_threshold() -> None:
    state = _state(
        per_objective_scores=[
            {"objective_id": "OBJ01", "score": 0.70},
            {"objective_id": "OBJ02", "score": 0.80},
        ],
        slot_coverage=_slot_coverage(objective_statuses={}, entity_statuses={}),
    )
    contract = _contract(named_entities=[], evidence_slots=[])

    result = aggregate_source_coverage(state, contract)

    assert result.objective.passed is True
    assert result.objective.per_objective_scores == {
        "objective-0": 0.7,
        "objective-1": 0.8,
    }
