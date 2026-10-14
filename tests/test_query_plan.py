"""Unit tests for contract-aware retrieval query planning."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch.agent import refinement  # noqa: E402
from deepsearch.agent.query_plan import _query_for_slot_target, plan_queries  # noqa: E402
from deepsearch.agent.refinement import generate_targeted_queries  # noqa: E402
from deepsearch.mission import compile_contract_from_state  # noqa: E402


def _state() -> dict[str, object]:
    mission_context = {
        "title": "Planner Test",
        "summary": "agent reliability evaluation",
        "objective": "Assess SAPLMA and CCS evidence for agent reliability evaluation",
        "success_criteria": [
            "Compare SAPLMA against CCS",
            "Document evidence gaps for agent reliability calibration",
        ],
        "deliverables": [
            "Comparison table with at least 4 rows and columns: Technique, Evidence, Gap",
            "Annotated bibliography of at least 6 papers",
        ],
    }
    return {
        "mission_id": "PLAN.1",
        "mission_context": mission_context,
        "mission_objectives": list(mission_context["success_criteria"]),
        "deliverable_format": "markdown",
        "research_depth": "deep",
        "max_loops": 4,
        "min_loops": 2,
        "depth_config": {},
    }


def test_plan_queries_emits_entity_queries_for_cold_start() -> None:
    """Loop 1 should fan out explicit entity queries for compiled named entities."""

    state = _state()
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=0,
    )

    entity_queries = [item for item in planned if item.query_class == "entity"]

    assert entity_queries
    # S38.1: registered acronyms expand to canonical text in the outbound
    # query while the slot binding stays keyed on the raw entity string.
    expected_query_texts = {
        contract.named_entities_canonical.get(entity, entity)
        for entity in contract.named_entities
    }
    assert {item.query for item in entity_queries} == expected_query_texts
    assert all(item.target_slots for item in entity_queries)


def test_plan_queries_keeps_refinement_and_planner_query_text_in_sync() -> None:
    """Concept queries should mirror the refinement layer's compact query text."""

    state = _state()
    contract = compile_contract_from_state(state, origin="test")
    gaps = [
        {"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "needs sharper comparison"},
    ]

    legacy_queries = generate_targeted_queries(
        gaps,
        state["mission_context"],
        objective_lookup=state["mission_objectives"],
    )
    planned = plan_queries(
        contract,
        gaps=gaps,
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )

    concept_queries = [item for item in planned if item.query_class == "concept"]

    assert concept_queries
    assert concept_queries[0].query == legacy_queries[0]
    assert concept_queries[0].target_slots == ["slot-objective-1"]
    assert "compare saplma against ccs" not in concept_queries[0].query.lower()


def test_plan_queries_maps_multiple_gap_objectives_to_expected_slot_ids() -> None:
    """Concept queries should retain slot targets for every low-scoring objective."""

    state = {
        "mission_id": "FORCE-SMOKE-04",
        "mission_context": {
            "title": "LangGraph Checkpointing Patterns for Long-Running Agents",
            "objective": (
                "Produce a short, readable research brief that explains the primary "
                "LangGraph checkpointing backends for long-running agent workflows."
            ),
            "success_criteria": [
                "Briefly describe each of the main LangGraph checkpointer backends",
                "Explain when to use each backend in practice",
                "Note one concrete operational gotcha for each backend",
            ],
            "deliverables": [
                "Short markdown research brief suitable for teammate handoff",
            ],
        },
        "mission_objectives": [
            "Briefly describe each of the main LangGraph checkpointer backends",
            "Explain when to use each backend in practice",
            "Note one concrete operational gotcha for each backend",
        ],
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 2,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[
            {"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "needs backend overview"},
            {"objective_id": "OBJ03", "current_score": 0.3, "reasoning": "needs gotcha evidence"},
        ],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )

    concept_queries = [item for item in planned if item.query_class == "concept"]

    assert len(concept_queries) == 2
    assert [item.target_slots for item in concept_queries] == [
        ["slot-objective-1"],
        ["slot-objective-3"],
    ]


def test_plan_queries_maps_hyphenated_objective_aliases_to_slot_ids() -> None:
    """Non-cold-start objective gaps should keep slot targets even with OBJ-1 aliases."""

    state = {
        "mission_id": "FORCE-SMOKE-13",
        "mission_context": {
            "title": "LangChain Retriever Selection for RAG Pipelines",
            "objective": (
                "Produce a short, readable research brief comparing BM25Retriever, "
                "MultiQueryRetriever, ParentDocumentRetriever, and "
                "ContextualCompressionRetriever for LangChain RAG pipelines."
            ),
            "success_criteria": [
                (
                    "Briefly describe BM25Retriever, MultiQueryRetriever, "
                    "ParentDocumentRetriever, and ContextualCompressionRetriever"
                ),
                "Explain when to use each retriever in practice",
                "Note one concrete operational gotcha for each retriever",
            ],
            "deliverables": [
                "Compact comparison table with columns: Retriever, Use When, Avoid When",
            ],
        },
        "mission_objectives": [
            (
                "Briefly describe BM25Retriever, MultiQueryRetriever, "
                "ParentDocumentRetriever, and ContextualCompressionRetriever"
            ),
            "Explain when to use each retriever in practice",
            "Note one concrete operational gotcha for each retriever",
        ],
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[
            {"objective_id": "OBJ-1", "current_score": 0.4, "reasoning": "needs direct evidence"},
            {"objective_id": "OBJ-3", "current_score": 0.3, "reasoning": "needs gotcha evidence"},
        ],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )

    concept_queries = [item for item in planned if item.query_class == "concept"]

    assert concept_queries
    assert any(
        item.target_slots == ["slot-objective-1"]
        and "BM25Retriever" in item.query
        and "overview" in item.query.lower()
        and "briefly describe" not in item.query.lower()
        for item in concept_queries
    )
    assert any(item.target_slots == ["slot-objective-3"] for item in concept_queries)


def test_plan_queries_maps_numeric_objective_aliases_to_slot_ids() -> None:
    """Bare numeric objective ids from reflection output should still bind slots."""

    state = {
        "mission_id": "FORCE-SMOKE-13",
        "mission_context": {
            "title": "LangChain Retriever Selection for RAG Pipelines",
            "objective": (
                "Produce a short, readable research brief comparing BM25Retriever, "
                "MultiQueryRetriever, ParentDocumentRetriever, and "
                "ContextualCompressionRetriever for LangChain RAG pipelines."
            ),
            "success_criteria": [
                (
                    "Briefly describe BM25Retriever, MultiQueryRetriever, "
                    "ParentDocumentRetriever, and ContextualCompressionRetriever"
                ),
                "Explain when to use each retriever in practice",
                "Note one concrete operational gotcha for each retriever",
            ],
        },
        "mission_objectives": [
            (
                "Briefly describe BM25Retriever, MultiQueryRetriever, "
                "ParentDocumentRetriever, and ContextualCompressionRetriever"
            ),
            "Explain when to use each retriever in practice",
            "Note one concrete operational gotcha for each retriever",
        ],
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[
            {"objective_id": "1", "current_score": 0.4, "reasoning": "needs direct evidence"},
            {"objective_id": "3", "current_score": 0.3, "reasoning": "needs gotcha evidence"},
        ],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )

    concept_queries = [item for item in planned if item.query_class == "concept"]

    assert concept_queries
    assert any(
        item.target_slots == ["slot-objective-1"]
        and "BM25Retriever" in item.query
        and "overview" in item.query.lower()
        and "briefly describe" not in item.query.lower()
        for item in concept_queries
    )
    assert any(item.target_slots == ["slot-objective-3"] for item in concept_queries)


def test_plan_queries_maps_exact_objective_text_aliases_to_slot_ids() -> None:
    """Exact objective text emitted as the gap id should still bind contract slots."""

    state = {
        "mission_id": "FORCE-SMOKE-13",
        "mission_context": {
            "title": "LangChain Retriever Selection for RAG Pipelines",
            "objective": (
                "Produce a short, readable research brief comparing BM25Retriever, "
                "MultiQueryRetriever, ParentDocumentRetriever, and "
                "ContextualCompressionRetriever for LangChain RAG pipelines."
            ),
            "success_criteria": [
                (
                    "Briefly describe BM25Retriever, MultiQueryRetriever, "
                    "ParentDocumentRetriever, and ContextualCompressionRetriever"
                ),
                "Explain when to use each retriever in practice",
                "Note one concrete operational gotcha for each retriever",
            ],
        },
        "mission_objectives": [
            (
                "Briefly describe BM25Retriever, MultiQueryRetriever, "
                "ParentDocumentRetriever, and ContextualCompressionRetriever"
            ),
            "Explain when to use each retriever in practice",
            "Note one concrete operational gotcha for each retriever",
        ],
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[
            {
                "objective_id": state["mission_objectives"][0],
                "current_score": 0.4,
                "reasoning": "needs direct evidence",
            }
        ],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )

    concept_queries = [item for item in planned if item.query_class == "concept"]

    assert concept_queries
    assert any(
        item.target_slots == ["slot-objective-1"]
        and "BM25Retriever" in item.query
        and "overview" in item.query.lower()
        and "briefly describe" not in item.query.lower()
        for item in concept_queries
    )


def test_plan_queries_expands_each_backend_gap_into_entity_specific_queries() -> None:
    """Backend-wide low-score objectives should fan out into backend-specific queries."""

    mission_context = {
        "title": "LangGraph Checkpoint Backend Comparison",
        "objective": (
            "Produce practical guidance covering MemorySaver, SqliteSaver, "
            "AsyncSqliteSaver, and PostgresSaver in LangGraph."
        ),
        "success_criteria": [
            "Explain when to use each backend in practice",
            "Note one concrete operational gotcha for each backend",
        ],
    }
    state = {
        "mission_id": "FORCE-SMOKE-13",
        "mission_context": mission_context,
        "mission_objectives": list(mission_context["success_criteria"]),
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[
            {
                "objective_id": "OBJ02",
                "current_score": 0.4,
                "reasoning": "needs gotcha evidence",
            }
        ],
        mission_context=mission_context,
        objective_lookup=state["mission_objectives"],
        loop_index=1,
        max_concept_queries=4,
    )

    concept_queries = [item for item in planned if item.query_class == "concept"]

    assert len(concept_queries) == 4
    assert all(item.target_slots == ["slot-objective-2"] for item in concept_queries)
    assert {item.query for item in concept_queries} == {
        "LangGraph MemorySaver operational gotchas",
        "LangGraph SqliteSaver operational gotchas",
        "LangGraph AsyncSqliteSaver operational gotchas",
        "LangGraph PostgresSaver operational gotchas",
    }
    assert all(
        "note one concrete operational gotcha" not in item.query.lower()
        for item in concept_queries
    )


def test_plan_queries_compacts_success_criteria_into_keyword_queries() -> None:
    """B35.9: concept queries should drop sentence-style success-criteria phrasing."""

    mission_context = {
        "title": "LangGraph Checkpoint Backend Comparison",
        "objective": (
            "Produce practical guidance covering MemorySaver, SqliteSaver, "
            "AsyncSqliteSaver, and PostgresSaver in LangGraph."
        ),
        "success_criteria": [
            "Explain when to use each backend in practice",
            "Note one concrete operational gotcha for each backend",
        ],
    }
    state = {
        "mission_id": "B35.9-COMPACT-QUERY-SHAPE",
        "mission_context": mission_context,
        "mission_objectives": list(mission_context["success_criteria"]),
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[
            {"objective_id": "OBJ01", "current_score": 0.3, "reasoning": "needs use-when evidence"},
            {"objective_id": "OBJ02", "current_score": 0.3, "reasoning": "needs gotcha evidence"},
        ],
        mission_context=mission_context,
        objective_lookup=state["mission_objectives"],
        loop_index=1,
        max_concept_queries=8,
    )

    concept_queries = [item.query for item in planned if item.query_class == "concept"]

    assert concept_queries
    assert any("use cases" in query.lower() for query in concept_queries)
    assert any("operational gotchas" in query.lower() for query in concept_queries)
    assert not any(
        "explain when to use each backend in practice" in query.lower()
        for query in concept_queries
    )
    assert not any(
        "note one concrete operational gotcha for each backend" in query.lower()
        for query in concept_queries
    )


def test_plan_queries_force_smoke_13_slot_binding_shape() -> None:
    """Lock the FORCE-SMOKE-13-RERUN-04 slot-binding shape in one regression.

    This mirrors the production contract that became the Sprint 33 fallback
    baseline: three required success criteria against an optional umbrella
    objective-0 (``required=False``) plus a distributive "each retriever"
    phrase that must fan into entity-specific concept queries.

    The test exercises three invariants at once:

    * The compiled contract leaves ``objective-0`` as the non-required umbrella
      while ``objective-1`` / ``objective-2`` / ``objective-3`` are required.
    * Reflection-emitted ``OBJ01`` / ``OBJ02`` / ``OBJ03`` gap ids bind by
      required-objective order only, so concept queries resolve against
      ``slot-objective-1`` / ``slot-objective-2`` / ``slot-objective-3`` and
      never against a would-be ``slot-objective-0``.
    * A distributive ``each retriever`` objective fans into one concept query
      per concrete retriever entity, all still bound to the originating slot.
    """

    success_criteria = [
        (
            "Briefly describe BM25Retriever, MultiQueryRetriever, "
            "ParentDocumentRetriever, and ContextualCompressionRetriever"
        ),
        "Explain when to use each retriever in practice",
        "Note one concrete operational gotcha for each retriever",
    ]
    mission_context = {
        "title": "LangChain Retriever Selection for RAG Pipelines",
        "objective": (
            "Produce a short, readable research brief comparing BM25Retriever, "
            "MultiQueryRetriever, ParentDocumentRetriever, and "
            "ContextualCompressionRetriever for LangChain RAG pipelines."
        ),
        "success_criteria": success_criteria,
        "deliverables": [
            "Compact comparison table with columns: Retriever, Use When, Avoid When",
        ],
    }
    state = {
        "mission_id": "FORCE-SMOKE-13-RERUN-04",
        "mission_context": mission_context,
        "mission_objectives": list(success_criteria),
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    objective_lookup = {obj.id: obj for obj in contract.objectives}
    assert objective_lookup["objective-0"].required is False
    assert all(
        objective_lookup[f"objective-{idx}"].required for idx in (1, 2, 3)
    )

    planned = plan_queries(
        contract,
        gaps=[
            {"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "needs direct evidence"},
            {"objective_id": "OBJ02", "current_score": 0.3, "reasoning": "needs use-when evidence"},
            {"objective_id": "OBJ03", "current_score": 0.3, "reasoning": "needs gotcha evidence"},
        ],
        mission_context=mission_context,
        objective_lookup=state["mission_objectives"],
        loop_index=1,
        max_concept_queries=16,
    )

    concept_queries = [item for item in planned if item.query_class == "concept"]
    assert concept_queries
    for item in concept_queries:
        assert item.target_slots, f"concept query missing slot target: {item.query}"
        assert "slot-objective-0" not in item.target_slots, (
            "OBJnn gap must never bind to the optional umbrella objective-0"
        )

    retriever_entities = {
        "BM25Retriever",
        "MultiQueryRetriever",
        "ParentDocumentRetriever",
        "ContextualCompressionRetriever",
    }
    obj1_queries = [
        item for item in concept_queries if item.target_slots == ["slot-objective-1"]
    ]
    assert len(obj1_queries) == 1
    assert "overview" in obj1_queries[0].query.lower()
    assert "briefly describe" not in obj1_queries[0].query.lower()
    assert all(entity in obj1_queries[0].query for entity in retriever_entities)

    def _single_entity_fanout(target_slot: str) -> set[str]:
        return {
            item.query
            for item in concept_queries
            if item.target_slots == [target_slot]
            and sum(entity in item.query for entity in retriever_entities) == 1
        }

    for target_slot, phrase in (
        ("slot-objective-2", "use"),
        ("slot-objective-3", "gotcha"),
    ):
        fanout = _single_entity_fanout(target_slot)
        assert len(fanout) == len(retriever_entities), (
            f"expected one {phrase} query per retriever entity, got {fanout!r}"
        )
        covered = {
            entity for entity in retriever_entities if any(entity in query for query in fanout)
        }
        assert covered == retriever_entities, (
            f"distributive fan-out missed retrievers: {retriever_entities - covered}"
        )


def test_plan_queries_prioritizes_unfilled_slot_targets_in_later_loops() -> None:
    """B35.2: post-loop-1 unfilled-slot state must steer loop-2 query order.

    Mirrors the FORCE-SMOKE-13 contract: after loop-1, slot-objective-1 is
    still unaddressed (no sources matched the BM25Retriever description gap),
    while slot-objective-2 and slot-objective-3 already have evidence. The
    loop-2 plan must surface objective-1 queries before objective-2/3 queries
    so the next loop's token budget targets the actual gap.

    Stable invariants preserved: target_slots assignment is unchanged (the
    B34.1 slot-binding regression continues to pass — see
    test_plan_queries_force_smoke_13_slot_binding_shape).
    """

    success_criteria = [
        (
            "Briefly describe BM25Retriever, MultiQueryRetriever, "
            "ParentDocumentRetriever, and ContextualCompressionRetriever"
        ),
        "Explain when to use each retriever in practice",
        "Note one concrete operational gotcha for each retriever",
    ]
    mission_context = {
        "title": "LangChain Retriever Selection for RAG Pipelines",
        "objective": (
            "Produce a short, readable research brief comparing BM25Retriever, "
            "MultiQueryRetriever, ParentDocumentRetriever, and "
            "ContextualCompressionRetriever for LangChain RAG pipelines."
        ),
        "success_criteria": success_criteria,
        "deliverables": [
            "Compact comparison table with columns: Retriever, Use When, Avoid When",
        ],
    }
    state = {
        "mission_id": "B35.2-LOOP2-UNFILLED-OBJ1",
        "mission_context": mission_context,
        "mission_objectives": list(success_criteria),
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    # Loop-2 planning, with the same gap shape that loop-1 surfaced.
    gaps = [
        {"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "needs direct evidence"},
        {"objective_id": "OBJ02", "current_score": 0.3, "reasoning": "needs use-when evidence"},
        {"objective_id": "OBJ03", "current_score": 0.3, "reasoning": "needs gotcha evidence"},
    ]

    # No coverage signal → behavior is identical to today.
    baseline_plan = plan_queries(
        contract,
        gaps=gaps,
        mission_context=mission_context,
        objective_lookup=state["mission_objectives"],
        loop_index=1,
        max_concept_queries=16,
    )

    # With slot-objective-1 marked unfilled, queries targeting it must come
    # before queries that target only objective-2 or objective-3.
    coverage_plan = plan_queries(
        contract,
        gaps=gaps,
        mission_context=mission_context,
        objective_lookup=state["mission_objectives"],
        loop_index=1,
        max_concept_queries=16,
        unfilled_slot_ids=["slot-objective-1"],
    )

    # Same set of queries — ordering is the only thing that changes.
    assert {item.query for item in baseline_plan} == {item.query for item in coverage_plan}

    obj1_indices = [
        i for i, item in enumerate(coverage_plan) if "slot-objective-1" in item.target_slots
    ]
    other_indices = [
        i
        for i, item in enumerate(coverage_plan)
        if "slot-objective-1" not in item.target_slots
        and any(
            slot in item.target_slots
            for slot in ("slot-objective-2", "slot-objective-3")
        )
    ]
    assert obj1_indices, "expected at least one query targeting slot-objective-1"
    assert other_indices, "expected at least one query targeting other objective slots"
    assert max(obj1_indices) < min(other_indices), (
        "B35.2: unfilled-slot-targeting queries must come before queries "
        "targeting already-covered slots"
    )

    # Slot assignment for the BM25Retriever obj1 query is unchanged.
    obj1_queries = [
        item for item in coverage_plan
        if "slot-objective-1" in item.target_slots
        and "BM25Retriever" in item.query
        and "overview" in item.query.lower()
        and "briefly describe" not in item.query.lower()
    ]
    assert obj1_queries
    for item in obj1_queries:
        assert item.target_slots == ["slot-objective-1"]


def test_plan_queries_unfilled_slots_param_is_noop_without_matches() -> None:
    """When no planned query targets an unfilled slot, ordering is preserved."""

    state = _state()
    contract = compile_contract_from_state(state, origin="test")

    baseline = plan_queries(
        contract,
        gaps=[
            {"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "x"},
            {"objective_id": "OBJ02", "current_score": 0.3, "reasoning": "y"},
        ],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )
    with_unfilled = plan_queries(
        contract,
        gaps=[
            {"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "x"},
            {"objective_id": "OBJ02", "current_score": 0.3, "reasoning": "y"},
        ],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
        unfilled_slot_ids=["slot-does-not-exist"],
    )
    assert [item.query for item in baseline] == [item.query for item in with_unfilled]


def test_plan_queries_empty_unfilled_slot_ids_is_noop() -> None:
    """Empty / None unfilled_slot_ids must not change query order."""

    state = _state()
    contract = compile_contract_from_state(state, origin="test")

    baseline = plan_queries(
        contract,
        gaps=[],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )
    for unfilled in (None, [], [""]):
        candidate = plan_queries(
            contract,
            gaps=[],
            mission_context=state["mission_context"],
            objective_lookup=state["mission_objectives"],
            loop_index=1,
            unfilled_slot_ids=unfilled,
        )
        assert [item.query for item in candidate] == [item.query for item in baseline]


def test_plan_queries_emits_only_unsatisfied_deliverable_gap_queries() -> None:
    """Deliverable-gap queries should skip schemas already marked satisfied."""

    state = _state()
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract.model_dump(mode="json"),
        gaps=[],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
        reflection_scores={
            "deliverable_schema_scores": [
                {"schema_id": "deliverable-schema-1", "score": 1.0},
            ]
        },
    )

    deliverable_queries = [item for item in planned if item.query_class == "deliverable_gap"]

    assert len(deliverable_queries) == 1
    assert deliverable_queries[0].target_slots == ["slot-deliverable-schema-2"]
    assert deliverable_queries[0].query == contract.objective
    assert "Annotated bibliography" not in deliverable_queries[0].query
    assert "Technique, Evidence, Gap" not in deliverable_queries[0].query


def test_plan_queries_prioritizes_concrete_backend_entities_before_umbrella_entities() -> None:
    mission_context = {
        "title": "LangGraph checkpoint backend comparison",
        "summary": "checkpoint backend tradeoff analysis",
        "objective": (
            "Compare MemorySaver, SqliteSaver, AsyncSqliteSaver, and "
            "PostgresSaver in LangGraph"
        ),
        "success_criteria": [
            "Keep source selection on-contract and focused on the required backends",
        ],
        "deliverables": [
            "Comparison table with columns: Backend, Use When, Avoid When, Operational Gotcha",
        ],
    }
    state = {
        "mission_id": "FORCE-SMOKE-11",
        "mission_context": mission_context,
        "mission_objectives": list(mission_context["success_criteria"]),
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[],
        mission_context=mission_context,
        objective_lookup=state["mission_objectives"],
        loop_index=0,
    )

    entity_queries = [item.query for item in planned if item.query_class == "entity"]

    assert entity_queries[:4] == [
        "MemorySaver",
        "SqliteSaver",
        "AsyncSqliteSaver",
        "PostgresSaver",
    ]
    assert entity_queries[-1] == "LangGraph"


def test_plan_queries_drops_unmapped_internal_objective_ids_and_excluded_terms() -> None:
    mission_context = {
        "title": "LangGraph Checkpoint Backend Comparison",
        "summary": "checkpoint backend comparison",
        "objective": (
            "Compare MemorySaver, SqliteSaver, AsyncSqliteSaver, and "
            "PostgresSaver in LangGraph"
        ),
        "success_criteria": [
            "Explain when to use each backend in practice",
            "Capture one concrete operational gotcha for each backend",
            (
                "Keep source selection and findings on-contract: do not include "
                "RedisSaver or other off-contract sibling backends unless "
                "explicitly required"
            ),
            "Include a compact comparison table with columns: Backend, Use When, Avoid When",
            "Executive summary should synthesize the backend tradeoffs",
        ],
    }
    state = {
        "mission_id": "FORCE-SMOKE-11",
        "mission_context": mission_context,
        "mission_objectives": list(mission_context["success_criteria"]),
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    planned = plan_queries(
        contract,
        gaps=[{"objective_id": "OBJ03", "current_score": 0.4, "reasoning": "needs evidence"}],
        mission_context=mission_context,
        objective_lookup=state["mission_objectives"],
        loop_index=1,
        reflection_scores={"deliverable_schema_scores": []},
    )

    queries = [item.query for item in planned]

    assert queries
    assert not any("OBJ03" in query for query in queries)
    assert not any("RedisSaver" in query for query in queries)
    assert not any("Backend, Use When, Avoid When" in query for query in queries)


def test_query_for_slot_target_recovers_objective_content_tokens() -> None:
    """Sprint 47 regression: prose-led objective slots must carry criterion content tokens.

    Anchored on the D-R2-COMPARE-02 5-criterion fixture (TraceLab UUID
    57c5e256-8c58-4e5c-a9fb-fad8b7391e66) which surfaced the
    `_objective_query_terms` truncation defect — slot-objective-3 fired
    `Multi-Agent Coordination, Treaty Mechanics & Anti-Gaming in AI Systems
    identify 2 works` with zero lexicographic content. Asserts the post-fix
    queries through the real planner path (`_query_for_slot_target`), not
    the helper unit alone.
    """

    mission_context = {
        "title": "D-R2-COMPARE-02",
        "focus": "Multi-Agent Coordination, Treaty Mechanics & Anti-Gaming in AI Systems",
        "objective": (
            "Survey the multi-agent coordination, anti-gaming, and lexicographic "
            "multi-objective optimization literature relevant to Paper D"
        ),
        "success_criteria": [
            "Identify 3+ foundational works on multi-agent coordination, mechanism design, or cooperative game theory applicable to AI systems",
            "Identify 2+ works on anti-gaming, specification gaming, or reward hacking in AI with proposed mitigations",
            "Identify 2+ works on multi-objective optimization for AI with lexicographic priorities or constraint hierarchies",
            "Identify any existing treaty or agreement frameworks in distributed AI or multi-agent systems",
            "Produce a positioning summary for §7 (multi-agent treaties) and §6 (anti-gaming adjudication) of Paper D",
        ],
        "deliverables": [],
    }
    state = {
        "mission_id": "D-R2-COMPARE-TEST",
        "mission_context": mission_context,
        "mission_objectives": list(mission_context["success_criteria"]),
        "deliverable_format": "markdown",
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 1,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    canonical_map = getattr(contract, "named_entities_canonical", None) or {}
    objective_text_by_id = {
        objective.id: str(objective.text).strip() for objective in contract.objectives
    }
    schema_title_by_id = {
        schema.id: str(getattr(schema, "title", "") or "").strip()
        for schema in contract.deliverable_schemas
    }
    focus_hint = refinement._extract_focus_hint(dict(mission_context))

    def _query_for_objective(objective_id: str) -> str:
        return _query_for_slot_target(
            {"objective_id": objective_id},
            contract=contract,
            canonical_map=canonical_map,
            objective_text_by_id=objective_text_by_id,
            schema_title_by_id=schema_title_by_id,
            focus_hint=focus_hint,
            objective_lookup=state["mission_objectives"],
        )

    # The contract compiler stores the main objective at id "objective-0"
    # and the five success criteria at "objective-1" .. "objective-5"
    # (contract.py:592-608). So criterion-3 ("…lexicographic priorities…")
    # is "objective-3".
    obj1_query = _query_for_objective("objective-1")
    obj2_query = _query_for_objective("objective-2")
    obj3_query = _query_for_objective("objective-3")

    # Slot-objective-3 must carry the lexicographic-MORL content tokens that
    # the pre-S47 truncation dropped.
    assert "multi objective" in obj3_query
    assert "lexicographic" in obj3_query
    assert "constraint hierarchies" in obj3_query

    # Slot-objective-2 and slot-objective-3 must NOT collapse to the same
    # query — the S46-era duplicate-query regression that motivated the fix.
    assert obj2_query != obj3_query

    # Bonus insurance: slot-objective-1 should carry its own criterion content.
    assert "multi agent coordination" in obj1_query
