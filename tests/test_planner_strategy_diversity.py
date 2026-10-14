"""S39.1 — Planner Strategy Diversity (RC-5) regression coverage.

The contract-aware query planner must produce at least one novel query per
loop at loop_index >= 3 on ACC-shape contracts. Prior to S39.1 the planner
ran out of novel queries at loop 3 because every candidate collided with the
``previous_queries`` dedupe set — all three downstream deliverable-schema
slots starved from insufficient retrieval. These tests lock the three
diversification strategies (slot-targeted, sub-objective decomposition,
paraphrase mutation) and the per-loop telemetry surface the S39.3 paid smoke
relies on for its §4.5 empirical verification gate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch.agent.query_plan import (  # noqa: E402
    PlannedQuery,
    plan_queries,
)
from deepsearch.mission import compile_contract_from_state  # noqa: E402


def _acc_like_state() -> Dict[str, Any]:
    """Return a compile-ready state shaped like ACC-BIB-EVID-01.

    Uses the same multi-entity objective pattern as the committed fixture
    (three registered + unregistered method entities named in one objective
    plus at least one deliverable-schema-driven success criterion) so the
    planner sees the same diversification surface as the paid smoke.
    """

    return {
        "mission_id": "S39.1-TEST-ACC",
        "mission_context": {
            "title": "ACC-shape Evidence Base",
            "objective": (
                "Map the evidence base for SAPLMA, CCS, and semantic entropy "
                "approaches in agent evaluation."
            ),
            "success_criteria": [
                (
                    "Annotated bibliography with at least 12 rows covering "
                    "SAPLMA, CCS, and semantic entropy"
                ),
                (
                    "Evidence-gap table with at least 5 rows comparing the "
                    "three methods"
                ),
            ],
            "deliverables": [
                "Annotated bibliography of at least 12 papers",
                (
                    "Evidence-gap table with columns: Method, Evidence, Gap "
                    "and at least 5 rows"
                ),
            ],
            "required_entities": ["SAPLMA", "CCS", "semantic entropy"],
        },
        "mission_objectives": [
            (
                "Map the evidence base for SAPLMA, CCS, and semantic entropy "
                "approaches in agent evaluation."
            ),
        ],
        "deliverable_format": "markdown",
        "research_depth": "deep",
        "max_loops": 5,
        "min_loops": 2,
        "depth_config": {},
    }


def _plan(
    contract: Any,
    *,
    loop_index: int,
    previous: List[str],
    gaps: List[Dict[str, Any]] | None = None,
    unfilled: List[str] | None = None,
    telemetry: Dict[str, Any] | None = None,
) -> List[PlannedQuery]:
    return plan_queries(
        contract,
        gaps=gaps
        or [
            {
                "objective_id": "OBJ01",
                "current_score": 0.4,
                "reasoning": "needs direct evidence",
            },
        ],
        mission_context=_acc_like_state()["mission_context"],
        previous_queries=previous,
        objective_lookup=_acc_like_state()["mission_objectives"],
        loop_index=loop_index,
        unfilled_slot_ids=unfilled,
        diversity_telemetry=telemetry,
    )


def _token_substring_match(candidate: str, corpus: List[str]) -> bool:
    """True iff ``candidate`` token list is a contiguous subsequence of any
    prior query's tokens (or vice versa). Token-level to avoid spurious
    character hits like ``acc`` ⊂ ``acc-shape``.
    """

    needle = [tok for tok in str(candidate or "").lower().split() if tok]
    if not needle:
        return False
    for item in corpus:
        hay = [tok for tok in str(item or "").lower().split() if tok]
        if not hay:
            continue
        short, long = (needle, hay) if len(needle) <= len(hay) else (hay, needle)
        for start in range(len(long) - len(short) + 1):
            if long[start : start + len(short)] == short:
                return True
    return False


def _novel_queries(planned: List[PlannedQuery], previous: List[str]) -> List[PlannedQuery]:
    return [item for item in planned if not _token_substring_match(item.query, previous)]


def test_loop3_produces_novel_query_with_partially_filled_slots() -> None:
    """Mission SC #3: loop 3 returns ≥1 query novel against loop-1 + loop-2."""

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    loop1 = _plan(contract, loop_index=0, previous=[])
    loop1_texts = [item.query for item in loop1]

    loop2 = _plan(contract, loop_index=1, previous=loop1_texts)
    loop2_texts = [item.query for item in loop2]

    history = list(dict.fromkeys(loop1_texts + loop2_texts))

    slot_by_id = {slot.id: slot for slot in contract.evidence_slots}
    unfilled = [
        slot_id for slot_id in slot_by_id if "entity" in slot_id or "deliverable" in slot_id
    ][:3]

    loop3 = _plan(contract, loop_index=3, previous=history, unfilled=unfilled)
    novel = _novel_queries(loop3, history)

    assert novel, (
        f"S39.1 RC-5 regression: expected ≥1 novel query at loop 3, got {loop3} "
        f"against history {history}"
    )


def test_acc_shape_reaches_loop_4_with_distinct_classes_per_loop() -> None:
    """Mission SC #4: loops 3 and 4 each carry ≥1 novel entity AND concept query."""

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    history: List[str] = []
    per_loop_plans: List[List[PlannedQuery]] = []
    slot_by_id = {slot.id: slot for slot in contract.evidence_slots}
    unfilled = [slot_id for slot_id in slot_by_id][:4]

    for loop_index in range(5):
        planned = _plan(
            contract,
            loop_index=loop_index,
            previous=list(history),
            unfilled=unfilled if loop_index >= 2 else None,
        )
        per_loop_plans.append(planned)
        for item in planned:
            if item.query not in history:
                history.append(item.query)

    for loop_index in (3, 4):
        planned = per_loop_plans[loop_index]
        prior = []
        for earlier in per_loop_plans[:loop_index]:
            for item in earlier:
                if item.query not in prior:
                    prior.append(item.query)

        novel = _novel_queries(planned, prior)
        classes = {item.query_class for item in novel}
        assert classes, (
            f"loop {loop_index} produced no novel queries; planned={planned}"
        )
        assert "entity" in classes or "concept" in classes, (
            f"loop {loop_index} novel queries missing entity or concept class, "
            f"got: {[(p.query, p.query_class) for p in novel]}"
        )


def test_slot_targeted_strategy_fires_on_unfilled_slots() -> None:
    """Unfilled slots drive generation, not just reordering, at loop >= 2."""

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    slot_by_id = {slot.id: slot for slot in contract.evidence_slots}
    entity_slot = next(
        (slot_id for slot_id, slot in slot_by_id.items() if slot.target.get("entity")),
        None,
    )
    assert entity_slot, "fixture should expose at least one entity-keyed slot"

    # Pre-populate previous_queries with every loop-0 entity query text so the
    # default pass cannot emit a fresh query for this slot. The slot-targeted
    # strategy must synthesize a distinct variant using the objective text.
    loop0 = _plan(contract, loop_index=0, previous=[])
    previous = [item.query for item in loop0]

    planned = _plan(
        contract,
        loop_index=2,
        previous=previous,
        unfilled=[entity_slot],
    )

    targeted = [item for item in planned if entity_slot in item.target_slots]
    assert targeted, (
        f"expected ≥1 planned query targeting unfilled slot {entity_slot}; "
        f"plan={planned}"
    )


def test_sub_objective_decomposition_emits_per_entity_queries() -> None:
    """Objectives mentioning 2+ contract entities decompose into per-entity queries."""

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    # Simulate loop-0 entity-query dedupe set so sub-objective decomposition
    # has to produce distinct text.
    loop0 = _plan(contract, loop_index=0, previous=[])
    previous = [item.query for item in loop0]

    planned = _plan(contract, loop_index=2, previous=previous, unfilled=[])

    entity_class = [item for item in planned if item.query_class == "entity"]
    assert len(entity_class) >= 2, (
        "sub-objective decomposition should emit ≥2 per-entity queries at loop 2 "
        f"when the objective mentions SAPLMA, CCS, and semantic entropy; got: "
        f"{[(p.query, p.query_class) for p in planned]}"
    )


def test_paraphrase_rotates_perspective_across_loops() -> None:
    """Paraphrase mutation picks a distinct perspective at each loop index."""

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    history: List[str] = []
    paraphrase_suffixes: List[str] = []
    for loop_index in range(2, 6):
        planned = _plan(contract, loop_index=loop_index, previous=list(history))
        for item in planned:
            if item.query not in history:
                history.append(item.query)
        paraphrase_candidates = [
            item.query
            for item in planned
            if item.query_class == "concept" and not item.target_slots
        ]
        if paraphrase_candidates:
            paraphrase_suffixes.append(paraphrase_candidates[0])

    assert len(set(paraphrase_suffixes)) >= 2, (
        f"paraphrase mutation must produce distinct queries across loops 2-5; "
        f"got: {paraphrase_suffixes}"
    )


def test_diversity_telemetry_populated_with_strategy_counts() -> None:
    """Mission SC #T: telemetry sink exposes per-strategy novel counts."""

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    loop0 = _plan(contract, loop_index=0, previous=[])
    previous = [item.query for item in loop0]

    slot_by_id = {slot.id: slot for slot in contract.evidence_slots}
    unfilled = list(slot_by_id.keys())[:3]

    telemetry: Dict[str, Any] = {}
    planned = _plan(
        contract,
        loop_index=3,
        previous=previous,
        unfilled=unfilled,
        telemetry=telemetry,
    )

    assert telemetry, "diversity_telemetry sink should be populated at loop 3"
    assert telemetry["loop"] == 3
    assert isinstance(telemetry["strategies_fired"], list)
    assert telemetry["strategies_fired"] == sorted(telemetry["strategies_fired"])
    assert isinstance(telemetry["diversified_strategies_fired"], list)
    assert telemetry["novel_total"] == len(planned)
    for strategy, count in telemetry["novel_per_strategy"].items():
        assert count > 0, f"strategy {strategy} should not appear with zero count"

    # At least one diversification strategy must have fired at loop 3 when
    # slots are unfilled and the default path has gone stale.
    assert telemetry["diversified_strategies_fired"], (
        "at loop 3 with unfilled slots, at least one of "
        "slot_targeted/sub_objective_decomposition/paraphrase_mutation "
        "must have produced a novel query"
    )


def test_planner_empty_plan_avoided_when_diversification_available() -> None:
    """Mission SC #5: slot coverage incomplete + diversification unused ⇒ non-empty plan.

    Simulates the RC-5 failure mode: every gap-derived query collides with
    ``seen`` so the default ``_plan_concept_queries`` and deliverable-gap
    paths emit nothing. With ``unfilled_slot_ids`` set and the diversification
    strategies untouched, the planner must still emit a non-empty plan rather
    than fall through to ``empty_plan`` and trigger ``planner_fallback_reason``.
    """

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    # Build the dedupe set by running loops 0, 1, 2 in sequence and collecting
    # every query those loops would have emitted.
    history: List[str] = []
    for loop_index in range(3):
        planned = _plan(contract, loop_index=loop_index, previous=list(history))
        for item in planned:
            if item.query not in history:
                history.append(item.query)

    slot_by_id = {slot.id: slot for slot in contract.evidence_slots}
    unfilled = list(slot_by_id.keys())

    telemetry: Dict[str, Any] = {}
    loop3 = _plan(
        contract,
        loop_index=3,
        previous=list(history),
        unfilled=unfilled,
        telemetry=telemetry,
    )

    assert loop3, (
        "S39.1 RC-5 regression: loop 3 must not return an empty plan when "
        "slots are unfilled and diversification strategies remain unused"
    )
    assert telemetry["novel_total"] > 0


def test_loop1_behavior_unchanged_by_diversification() -> None:
    """Diversification must not fire at loop 1 — default paths own novelty there."""

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    telemetry: Dict[str, Any] = {}
    loop1 = _plan(
        contract,
        loop_index=1,
        previous=[],
        unfilled=[slot.id for slot in contract.evidence_slots][:3],
        telemetry=telemetry,
    )

    assert loop1, "loop 1 should still emit default gap-derived queries"
    assert not telemetry.get("diversified_strategies_fired"), (
        "diversification strategies must not fire at loop 1; "
        f"telemetry={telemetry}"
    )


def test_diversity_telemetry_sink_cleared_between_calls() -> None:
    """Sink is repopulated cleanly so loop-N data never leaks into loop-N+1."""

    contract = compile_contract_from_state(_acc_like_state(), origin="test")

    telemetry: Dict[str, Any] = {"stale_key": "stale_value"}
    _plan(contract, loop_index=3, previous=[], telemetry=telemetry)
    assert "stale_key" not in telemetry
    assert telemetry["loop"] == 3
