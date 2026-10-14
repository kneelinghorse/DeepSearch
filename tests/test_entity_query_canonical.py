"""S38.1 — Acronym registry canonical map wired into entity query construction.

Covers MissionContract.named_entities_canonical population at compile time and
``_plan_entity_queries`` reading the canonical expansion when building the
outbound PlannedQuery.query field. Slot IDs remain keyed off the raw entity
string so downstream slot-binding code is unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch.agent.query_plan import _plan_entity_queries  # noqa: E402
from deepsearch.mission import (  # noqa: E402
    compile_contract_from_state,
    load_mission_from_yaml,
    prepare_execution_contract,
)

FIXTURE = Path("tests/fixtures/missions/acceptance_acc_bib_evid_01.yaml")


def _compile(objective: str, *, title: str = "Canonical Map Test") -> object:
    return compile_contract_from_state(
        {
            "mission_id": "S38.1-TEST",
            "mission_context": {
                "title": title,
                "objective": objective,
                "deliverables": ["Annotated bibliography of at least 5 papers"],
            },
            "mission_objectives": [objective],
            "deliverable_format": "markdown",
            "research_depth": "deep",
            "max_loops": 4,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )


def test_acc_bib_evid_01_fixture_compiles_canonical_map() -> None:
    state = load_mission_from_yaml(FIXTURE)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        prepared = prepare_execution_contract(
            state,
            origin="yaml",
            output_dir=Path(tmp) / "acc_bib_evid_01",
        )
    canonical = prepared.contract.named_entities_canonical
    assert canonical.get("CCS") == "Contrast-Consistent Search"
    assert "SAPLMA" in canonical, (
        "SAPLMA registry entry exists; canonical map must record coverage"
    )
    assert canonical["SAPLMA"] == "SAPLMA"


def test_plan_entity_queries_uses_canonical_expansion() -> None:
    contract = _compile("Map the evidence base for CCS approaches in agent evaluation.")

    queries = _plan_entity_queries(contract, seen=set())
    query_texts = [planned.query for planned in queries]

    assert any("Contrast-Consistent Search" in text for text in query_texts), (
        f"expected canonical expansion in entity queries, got: {query_texts}"
    )
    assert "CCS" not in query_texts, (
        f"bare acronym CCS must not be the outbound query when canonical "
        f"expansion is registered, got: {query_texts}"
    )


def test_canonical_query_keeps_raw_entity_slot_id() -> None:
    contract = _compile("Map the evidence base for CCS in agent evaluation.")

    queries = _plan_entity_queries(contract, seen=set())
    ccs_query = next(
        (planned for planned in queries if "Contrast-Consistent Search" in planned.query),
        None,
    )
    assert ccs_query is not None, f"missing canonical CCS query in {queries}"
    assert "slot-entity-ccs" in ccs_query.target_slots, (
        f"slot binding must remain on raw-entity slot id, got: {ccs_query.target_slots}"
    )


def test_non_registry_entity_falls_back_to_raw_string() -> None:
    contract = _compile(
        "Compare XYZQ and ABCDE methods for retrieval scoring in agent evaluation."
    )

    queries = _plan_entity_queries(contract, seen=set())
    query_texts = [planned.query for planned in queries]

    # Non-registry entities should pass through as raw strings.
    assert any("XYZQ" in text for text in query_texts), (
        f"expected XYZQ raw fallback in queries, got: {query_texts}"
    )


def test_multi_word_method_entity_query_unchanged() -> None:
    # semantic entropy is multi-word and not in the registry — must stay raw.
    contract = compile_contract_from_state(
        {
            "mission_id": "S38.1-TEST-MW",
            "mission_context": {
                "title": "Multi-Word Method",
                "objective": (
                    "Compare semantic entropy approaches for hallucination "
                    "detection in agent evaluation."
                ),
                "deliverables": ["Annotated bibliography of at least 5 papers"],
                "required_entities": ["semantic entropy"],
            },
            "mission_objectives": [
                "Compare semantic entropy approaches for hallucination "
                "detection in agent evaluation.",
            ],
            "deliverable_format": "markdown",
            "research_depth": "deep",
            "max_loops": 4,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert "semantic entropy" not in contract.named_entities_canonical, (
        "multi-word non-registry entity must not appear in canonical map"
    )

    queries = _plan_entity_queries(contract, seen=set())
    query_texts = [planned.query for planned in queries]
    assert any("semantic entropy" in text for text in query_texts), (
        f"raw multi-word entity must survive into queries, got: {query_texts}"
    )


def test_named_entities_list_is_not_mutated_by_canonical_population() -> None:
    contract = _compile("Map the evidence base for CCS approaches in agent evaluation.")

    # Raw entity string still present — the canonical map is a sibling, not a
    # rewrite of named_entities.
    assert "CCS" in contract.named_entities
    assert "Contrast-Consistent Search" not in contract.named_entities
