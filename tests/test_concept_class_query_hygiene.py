"""S38.2 — Concept-class query hygiene: success-criteria fusion + Map-the truncation.

Locks two narrow string-handling regressions in the concept-class query path:

* DEFECT A (B35.9 parity): success-criteria text must not be serialized into
  the outbound PlannedQuery.query field for concept-class queries. The S37.6
  paid smoke captured the failing query
  ``"Acceptance Bibliography Evidence Map evidence gaps SAPLMA CCS Map the"``
  which fused title + success-criteria + objective fragments.
* DEFECT B (Map-the truncation): preserved-term extraction via
  PROPER_NOUN_PATTERN admitted ``"Map the"`` (Capitalized + connective) as a
  preserved phrase, leaving a grammatical fragment at the tail of every
  concept-class query for the ACC-BIB-EVID-01 fixture.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch.agent.query_plan import (  # noqa: E402
    _CONCEPT_QUERY_TRAILING_CONNECTIVES,
    plan_queries,
)
from deepsearch.mission import (  # noqa: E402
    compile_contract_from_state,
    load_mission_from_yaml,
    prepare_execution_contract,
)

FIXTURE = Path("tests/fixtures/missions/acceptance_acc_bib_evid_01.yaml")


def _compile_fixture():
    state = load_mission_from_yaml(FIXTURE)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = prepare_execution_contract(state, origin="yaml", output_dir=Path(tmp))
    return state, prepared.contract


def _concept_queries_for_acc_fixture():
    state, contract = _compile_fixture()
    gaps = [{"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "gap"}]
    planned = plan_queries(
        contract,
        gaps=gaps,
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )
    return [item for item in planned if item.query_class == "concept"], contract


def test_concept_queries_do_not_contain_success_criteria_substrings() -> None:
    queries, _ = _concept_queries_for_acc_fixture()

    forbidden = (
        "Surface the strongest bibliography",
        "Call out evidence gaps",
    )
    for planned in queries:
        for fragment in forbidden:
            assert fragment.lower() not in planned.query.lower(), (
                f"concept query must not contain success-criteria substring "
                f"{fragment!r}; got: {planned.query!r}"
            )


def test_concept_queries_do_not_end_on_connective_fragment() -> None:
    queries, _ = _concept_queries_for_acc_fixture()

    assert queries, "expected at least one concept query for the ACC fixture"
    for planned in queries:
        tokens = planned.query.split()
        assert tokens, f"empty concept query: {planned!r}"
        assert tokens[-1].lower() not in _CONCEPT_QUERY_TRAILING_CONNECTIVES, (
            f"concept query ends with connective {tokens[-1]!r} "
            f"(grammatical fragment): {planned.query!r}"
        )
        # The captured S37.6 failure ended with "Map the" — assert that exact tail
        # is dead even after sanitization.
        assert not planned.query.lower().endswith(" map the"), (
            f"concept query still trails with 'Map the': {planned.query!r}"
        )


def test_concept_queries_retain_meaningful_concept_tokens() -> None:
    queries, _ = _concept_queries_for_acc_fixture()

    assert queries, "expected at least one concept query"
    for planned in queries:
        tokens = planned.query.split()
        # Should not collapse to empty / focus_hint-only after sanitization.
        assert len(tokens) >= 3, (
            f"concept query degenerated after sanitization: {planned.query!r}"
        )


def test_b35_9_parity_entity_queries_have_no_sc_leakage() -> None:
    state, contract = _compile_fixture()
    planned = plan_queries(
        contract,
        gaps=[],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=0,
    )
    entity_queries = [item for item in planned if item.query_class == "entity"]
    assert entity_queries, "expected entity queries on cold start"
    forbidden = (
        "Surface the strongest bibliography",
        "Call out evidence gaps",
    )
    for planned_query in entity_queries:
        for fragment in forbidden:
            assert fragment.lower() not in planned_query.query.lower(), (
                f"entity query must not contain SC substring "
                f"{fragment!r}; got: {planned_query.query!r}"
            )


def test_deliverable_gap_queries_remain_objective_text() -> None:
    state, contract = _compile_fixture()
    planned = plan_queries(
        contract,
        gaps=[],
        mission_context=state["mission_context"],
        objective_lookup=state["mission_objectives"],
        loop_index=1,
    )
    gap_queries = [item for item in planned if item.query_class == "deliverable_gap"]
    assert gap_queries, "expected deliverable_gap queries"
    for planned_query in gap_queries:
        # deliverable_gap queries are built from contract.objective directly.
        assert "Map the evidence base" in planned_query.query, (
            f"deliverable_gap must surface the objective text, got: "
            f"{planned_query.query!r}"
        )


def test_short_objective_concept_query_is_not_truncated() -> None:
    contract = compile_contract_from_state(
        {
            "mission_id": "S38.2-SHORT",
            "mission_context": {
                "title": "Short Objective Probe",
                "objective": "Document the foo bar baseline.",
                "deliverables": ["Annotated bibliography of at least 5 papers"],
            },
            "mission_objectives": ["Document the foo bar baseline."],
            "deliverable_format": "markdown",
            "research_depth": "deep",
            "max_loops": 4,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )
    gaps = [{"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "gap"}]
    planned = plan_queries(
        contract,
        gaps=gaps,
        mission_context={
            "title": "Short Objective Probe",
            "objective": "Document the foo bar baseline.",
        },
        objective_lookup=["Document the foo bar baseline."],
        loop_index=1,
    )
    concept_queries = [item for item in planned if item.query_class == "concept"]
    assert concept_queries
    for planned_query in concept_queries:
        # Trailing-connective strip must not produce an empty query for a
        # short, well-formed objective.
        assert planned_query.query.strip(), (
            f"sanitization wiped a short-objective concept query: {planned_query!r}"
        )


def test_long_objective_concept_query_truncation_at_word_boundary() -> None:
    long_objective = (
        "Compare modern retrieval-augmented generation evaluation rigs across "
        "BASELINE, DEEP, and ALPHA tiers in production agent evaluation pipelines "
        "for academic literature, industry whitepapers, and reference implementations."
    )
    contract = compile_contract_from_state(
        {
            "mission_id": "S38.2-LONG",
            "mission_context": {
                "title": "Long Objective Probe",
                "objective": long_objective,
                "deliverables": ["Annotated bibliography of at least 12 papers"],
            },
            "mission_objectives": [long_objective],
            "deliverable_format": "markdown",
            "research_depth": "deep",
            "max_loops": 4,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )
    gaps = [{"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "gap"}]
    planned = plan_queries(
        contract,
        gaps=gaps,
        mission_context={
            "title": "Long Objective Probe",
            "objective": long_objective,
        },
        objective_lookup=[long_objective],
        loop_index=1,
    )
    concept_queries = [item for item in planned if item.query_class == "concept"]
    assert concept_queries
    for planned_query in concept_queries:
        tokens = planned_query.query.split()
        assert tokens
        assert tokens[-1].lower() not in _CONCEPT_QUERY_TRAILING_CONNECTIVES, (
            f"long objective concept query ends with connective fragment: "
            f"{planned_query.query!r}"
        )
