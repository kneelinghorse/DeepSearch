"""Unit tests for S54.3 query-planner ambiguous-acronym disambiguation."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from deepsearch.agent.query_plan import (
    _disambiguate_entity_surface,
    _is_ambiguous_acronym_shape,
    _plan_entity_queries,
)
from deepsearch.mission.contract import (
    ExecutionBudget,
    MissionContract,
)


# --------------------------------------------------------------------------- #
# Shape detector
# --------------------------------------------------------------------------- #


def test_ambiguous_acronym_shape_detector() -> None:
    # Positives: short all-caps acronyms that benefit from disambiguation.
    assert _is_ambiguous_acronym_shape("OODS") is True
    assert _is_ambiguous_acronym_shape("CCS") is True
    assert _is_ambiguous_acronym_shape("ARIA") is True
    assert _is_ambiguous_acronym_shape("HTML") is True
    # Boundary: 5 chars OK, 6+ rejected.
    assert _is_ambiguous_acronym_shape("ABCDE") is True
    assert _is_ambiguous_acronym_shape("SAPLMA") is False  # 6 chars

    # Negatives: hyphenated or mixed-case names are not ambiguous in this sense.
    assert _is_ambiguous_acronym_shape("axe-core") is False
    assert _is_ambiguous_acronym_shape("t-SNE") is False
    assert _is_ambiguous_acronym_shape("eslint-plugin-jsx-a11y") is False
    assert _is_ambiguous_acronym_shape("Pa11y") is False
    assert _is_ambiguous_acronym_shape("LangGraph") is False
    # Numbers without alpha are rejected.
    assert _is_ambiguous_acronym_shape("123") is False
    assert _is_ambiguous_acronym_shape("") is False


# --------------------------------------------------------------------------- #
# _disambiguate_entity_surface
# --------------------------------------------------------------------------- #


def test_disambiguation_appends_context_for_ambiguous_acronym() -> None:
    out = _disambiguate_entity_surface(
        "OODS",
        base_surface="OODS",
        disambiguation_map={"OODS": "Object-Oriented Design System"},
    )
    assert out == "OODS Object-Oriented Design System"


def test_disambiguation_skips_when_context_already_present() -> None:
    """Idempotency: if base already contains the context, don't double-append."""

    out = _disambiguate_entity_surface(
        "OODS",
        base_surface="OODS Object-Oriented Design System",
        disambiguation_map={"OODS": "Object-Oriented Design System"},
    )
    assert out is None


def test_disambiguation_skips_unambiguous_entity_with_context_set() -> None:
    """Belt-and-suspenders: even if LLM populates context for a hyphenated tool,
    the shape check vetoes the mutation."""

    out = _disambiguate_entity_surface(
        "axe-core",
        base_surface="axe-core",
        disambiguation_map={"axe-core": "an accessibility testing tool"},
    )
    assert out is None


def test_disambiguation_skips_when_context_empty() -> None:
    out = _disambiguate_entity_surface(
        "OODS",
        base_surface="OODS",
        disambiguation_map={"OODS": ""},
    )
    assert out is None


def test_disambiguation_skips_when_entity_missing_from_map() -> None:
    out = _disambiguate_entity_surface(
        "OODS",
        base_surface="OODS",
        disambiguation_map={},
    )
    assert out is None


# --------------------------------------------------------------------------- #
# _plan_entity_queries integration
# --------------------------------------------------------------------------- #


def _build_contract(
    *,
    entities: list[str],
    canonical: dict[str, str] | None = None,
    disambiguation: dict[str, str] | None = None,
    query_surfaces: dict[str, list[str]] | None = None,
) -> MissionContract:
    return MissionContract(
        contract_id="test",
        compiled_at="2026-04-26T00:00:00Z",
        origin="test",
        mission_id="S54-TEST",
        title="S54 disambiguation test",
        objective="catalogue tools",
        named_entities=list(entities),
        named_entities_canonical=dict(canonical or {entity: entity for entity in entities}),
        named_entities_disambiguation=dict(disambiguation or {}),
        named_entities_query_surfaces=dict(query_surfaces or {}),
        execution_budget=ExecutionBudget(
            research_depth="baseline",
            deliverable_format="markdown",
            max_loops=3,
            min_loops=1,
        ),
    )


def test_plan_entity_queries_disambiguates_oods() -> None:
    """SEM-R09 OODS scenario: the entity-slot query gains the context tokens."""

    contract = _build_contract(
        entities=["OODS"],
        disambiguation={"OODS": "Object-Oriented Design System"},
    )
    planned = _plan_entity_queries(contract, set())
    queries = [item.query for item in planned]
    assert any("OODS" in q and "Object-Oriented Design System" in q for q in queries), queries


def test_plan_entity_queries_does_not_disambiguate_axe_core() -> None:
    """Hyphenated unambiguous tool stays unchanged."""

    contract = _build_contract(
        entities=["axe-core"],
        # Even if a future LLM fills context here, shape detector vetoes.
        disambiguation={"axe-core": "Accessibility Engine"},
    )
    planned = _plan_entity_queries(contract, set())
    queries = [item.query for item in planned]
    assert queries == ["axe-core"], queries


def test_plan_entity_queries_byte_equal_when_disambiguation_empty() -> None:
    """Regex-fallback path / flag OFF: empty disambiguation = no query mutation."""

    contract_with_empty = _build_contract(
        entities=["OODS"],
        disambiguation={},
    )
    contract_baseline = _build_contract(
        entities=["OODS"],
    )
    planned_a = [item.query for item in _plan_entity_queries(contract_with_empty, set())]
    planned_b = [item.query for item in _plan_entity_queries(contract_baseline, set())]
    assert planned_a == planned_b
    # And specifically: no augmentation happened.
    assert all("Object-Oriented" not in q for q in planned_a)


def test_plan_entity_queries_disambiguation_emits_telemetry_log(caplog) -> None:
    """When disambiguation fires, a structured log entry is recorded."""

    import logging

    contract = _build_contract(
        entities=["OODS"],
        disambiguation={"OODS": "Object-Oriented Design System"},
    )
    with caplog.at_level(logging.INFO, logger="deepsearch.agent.query_plan"):
        _plan_entity_queries(contract, set())
    matching = [
        record
        for record in caplog.records
        if "entity_query_disambiguation" in record.getMessage()
    ]
    assert matching, "expected an entity_query_disambiguation log entry"
    payload = matching[0].getMessage()
    assert "OODS" in payload
    assert "Object-Oriented Design System" in payload


def test_plan_entity_queries_dedupes_disambiguated_against_unaugmented() -> None:
    """If the same query was already seen in raw form, the disambiguated form
    must NOT be incorrectly suppressed — they are different strings."""

    contract = _build_contract(
        entities=["OODS"],
        disambiguation={"OODS": "Object-Oriented Design System"},
    )
    seen_raw = {"oods"}  # normalized lowercase
    planned = _plan_entity_queries(contract, seen_raw)
    queries = [item.query for item in planned]
    # Disambiguated form is a different normalized string and should plan.
    assert any("Object-Oriented Design System" in q for q in queries), queries


# --------------------------------------------------------------------------- #
# S56.3 — LLM-supplied query surfaces reach the planner's emitted queries
# --------------------------------------------------------------------------- #


def test_plan_entity_queries_emits_llm_supplied_query_surfaces() -> None:
    """S56.3: planner emits one query per LLM-supplied surface, canonical first."""

    contract = _build_contract(
        entities=["automated ARIA validation"],
        query_surfaces={
            "automated ARIA validation": ["ARIA validation", "ARIA", "axe-core"],
        },
    )
    planned = _plan_entity_queries(contract, set())
    queries = [item.query for item in planned]
    # Canonical comes first, then LLM surfaces in declared order.
    assert queries[:4] == [
        "automated ARIA validation",
        "ARIA validation",
        "ARIA",
        "axe-core",
    ]


def test_plan_entity_queries_dedupes_llm_surface_against_canonical() -> None:
    """S56.3: a surface that duplicates the canonical name is not re-emitted."""

    contract = _build_contract(
        entities=["axe-core"],
        query_surfaces={"axe-core": ["axe-core", "axe"]},
    )
    planned = _plan_entity_queries(contract, set())
    queries = [item.query for item in planned]
    assert queries.count("axe-core") == 1
    assert "axe" in queries


def test_plan_entity_queries_byte_equal_when_query_surfaces_empty() -> None:
    """S56.3 OFF/empty path: no surfaces means no extra emitted queries."""

    contract_with_empty = _build_contract(
        entities=["axe-core"],
        query_surfaces={},
    )
    contract_baseline = _build_contract(entities=["axe-core"])
    planned_a = [item.query for item in _plan_entity_queries(contract_with_empty, set())]
    planned_b = [item.query for item in _plan_entity_queries(contract_baseline, set())]
    assert planned_a == planned_b == ["axe-core"]
