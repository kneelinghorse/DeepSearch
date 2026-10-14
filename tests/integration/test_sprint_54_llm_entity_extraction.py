"""Sprint 54 combined integration suite — L1 (LLM extractor) × L2′ (query disambiguation).

These tests bridge:

* S54.2's contract compiler wiring (entity_extractor_llm + named_entities_disambiguation
  + entity_extraction_decisions audit rows).
* S54.3's query-planner consumption of the disambiguation map.

The LLM is mocked deterministically via ``_stub_llm_extractor``. The fixtures
mirror real mission shapes from SEM-R09-S53-RUN-01 and DSV-S52-RUN-01 so the
extractor sees objectives verbatim and the assertions reflect the actual failure
vectors those paid smokes surfaced (participial-fragment leakage, OODS homonym
admission, missing enumerated tools).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pydantic")

from deepsearch.mission.contract import compile_contract_from_state
from deepsearch.mission.entity_extractor_llm import (
    EntityExtractionOutcome,
    ExtractedEntity,
)
from deepsearch.agent.query_plan import _plan_entity_queries


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _stub_llm_extractor(monkeypatch, outcome: EntityExtractionOutcome | None) -> None:
    """Patch the contract module so the LLM-first path returns ``outcome``.

    Also stubs S60.1 declared-entity disambiguation to return ``None`` by
    default — keeps tests deterministic when ``required_entities`` is empty
    (disambiguation pass returns ``skipped_no_declared`` naturally and the
    stub is a no-op). Tests that exercise the declared path can override the
    disambiguation stub via :func:`_stub_declared_disambiguation`.
    """

    from deepsearch.mission import contract as contract_module
    from deepsearch.mission import entity_extractor_llm
    import deepsearch.config as config_module

    class _StubSettings:
        @classmethod
        def load(cls):
            return cls()

        def build_structured_extractor(self):
            return object()

    monkeypatch.setattr(contract_module, "DeepSearchSettings", _StubSettings, raising=False)
    monkeypatch.setattr(config_module, "DeepSearchSettings", _StubSettings, raising=False)
    monkeypatch.setattr(
        entity_extractor_llm,
        "run_llm_entity_extraction_sync",
        lambda *args, **kwargs: outcome,
    )
    monkeypatch.setattr(
        entity_extractor_llm,
        "run_declared_entity_disambiguation_sync",
        lambda *args, **kwargs: None,
    )




# Verbatim SEM-R09 objectives from cmos/research/sprint_53_paid_smoke_SEM-R09-S53-RUN-01-results.md




def _dsv_s52_state() -> dict[str, Any]:
    return {
        "mission_id": "DSV-S52-S54-TEST",
        "mission_context": {
            "title": "DSV-S52 EDA Visualization Toolkit Survey",
            "objective": (
                "Survey EDA visualization techniques including histograms, "
                "boxplots, scatter plots, scatterplot matrices, and t-SNE."
            ),
            "success_criteria": [
                "Compare techniques across distribution shape and outlier detection."
            ],
            "deliverables": [
                "Per-technique classification table with shape and use-case columns."
            ],
            "required_entities": ["histograms", "boxplots", "scatter plots", "scatterplot matrices"],
        },
    }


# --------------------------------------------------------------------------- #
# 1. SEM-R09 entity extraction
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# 2. DSV-S52 entity extraction
# --------------------------------------------------------------------------- #


def test_dsv_s52_preserves_declared_entities_and_avoids_substring(monkeypatch) -> None:
    """DSV-S52 declared techniques survive; SNE substring of t-SNE does NOT appear."""

    outcome = EntityExtractionOutcome(
        entity_names=[
            "histograms",
            "boxplots",
            "scatter plots",
            "scatterplot matrices",
            "t-SNE",
        ],
        disambiguation={},
        extracted=[],
        rejected=[],
        path="llm_primary",
    )
    _stub_llm_extractor(monkeypatch, outcome)

    contract = compile_contract_from_state(_dsv_s52_state(), origin="test")
    entity_lower = {entity.lower() for entity in contract.named_entities}

    for entity in ("histograms", "boxplots", "scatter plots", "scatterplot matrices"):
        assert entity.lower() in entity_lower, (
            f"missing declared entity {entity}: have {sorted(entity_lower)}"
        )
    # Substring SNE must NOT appear when t-SNE is in the entity list — _classify_named_entity_candidates
    # plus the safety-net fragment filter should reject it. Even the LLM
    # extractor doesn't emit it, but verify defensively.
    assert "sne" not in entity_lower, (
        "substring SNE leaked into entity slots when t-SNE is present"
    )


# --------------------------------------------------------------------------- #
# 3. Participial-fragment corpus
# --------------------------------------------------------------------------- #


def test_participial_fragment_corpus_all_rejected_by_llm(monkeypatch) -> None:
    """5+ participial phrases all surface as rejected_llm rows, none become entities."""

    fragments = [
        "including review workflows",
        "spanning initial data assessment",
        "based on token analysis",
        "covering authority tier coverage",
        "missingness inspection",
    ]
    outcome = EntityExtractionOutcome(
        entity_names=["axe-core"],
        disambiguation={},
        extracted=[],
        rejected=[
            ExtractedEntity(
                name=fragment,
                rejected=True,
                rejection_reason="participial_phrase_fragment",
            )
            for fragment in fragments
        ],
        path="llm_primary",
    )
    _stub_llm_extractor(monkeypatch, outcome)

    state = {
        "mission_id": "PARTICIPIAL-CORPUS-TEST",
        "mission_context": {
            "title": "Participial fragment corpus",
            "objective": (
                "Survey practices including review workflows, spanning initial "
                "data assessment, based on token analysis, covering authority "
                "tier coverage, and missingness inspection."
            ),
            "success_criteria": [],
            "deliverables": [],
            "required_entities": [],
        },
    }
    contract = compile_contract_from_state(state, origin="test")
    entity_lower = {entity.lower() for entity in contract.named_entities}
    for fragment in fragments:
        assert fragment.lower() not in entity_lower
    rejection_rows = [
        row
        for row in contract.entity_extraction_decisions
        if row.get("decision") == "rejected_llm"
    ]
    assert len(rejection_rows) == len(fragments)
    rejected_names = {row.get("entity") for row in rejection_rows}
    assert set(fragments) == rejected_names


# --------------------------------------------------------------------------- #
# 4. Enumerated-tool extraction at scale
# --------------------------------------------------------------------------- #


def test_enumerated_tool_extraction_8_items_all_present(monkeypatch) -> None:
    """An 8-tool enumeration produces all 8 entity slots, not just the first 2 or 3."""

    tools = [
        "axe-core",
        "eslint-plugin-jsx-a11y",
        "Pa11y",
        "HTMLHint",
        "markuplint",
        "accesslint",
        "Lighthouse",
        "Playwright",
    ]
    outcome = EntityExtractionOutcome(
        entity_names=tools,
        disambiguation={},
        extracted=[],
        rejected=[],
        path="llm_primary",
    )
    _stub_llm_extractor(monkeypatch, outcome)

    state = {
        "mission_id": "ENUMERATED-8-TEST",
        "mission_context": {
            "title": "Enumerated tool list",
            "objective": (
                "Catalogue tools such as axe-core, eslint-plugin-jsx-a11y, "
                "Pa11y, HTMLHint, markuplint, accesslint, Lighthouse, and "
                "Playwright."
            ),
            "success_criteria": [],
            "deliverables": [],
            "required_entities": [],
        },
    }
    contract = compile_contract_from_state(state, origin="test")
    entity_lower = {entity.lower() for entity in contract.named_entities}
    for tool in tools:
        assert tool.lower() in entity_lower, f"missing tool {tool}"


# --------------------------------------------------------------------------- #
# 5. OODS disambiguation integration (S54.2 → S54.3 bridge)
# --------------------------------------------------------------------------- #


def test_oods_disambiguation_propagates_into_query_planner(monkeypatch) -> None:
    """End-to-end: LLM extractor populates context → planner emits disambiguated query."""

    outcome = EntityExtractionOutcome(
        entity_names=["OODS"],
        disambiguation={"OODS": "Object-Oriented Design System"},
        extracted=[],
        rejected=[],
        path="llm_primary",
    )
    _stub_llm_extractor(monkeypatch, outcome)

    state = {
        "mission_id": "OODS-DISAMBIG-TEST",
        "mission_context": {
            "title": "OODS disambiguation integration",
            "objective": (
                "Identify gaps in current tooling where OODS tooling could "
                "add value."
            ),
            "success_criteria": [],
            "deliverables": [],
            "required_entities": [],
        },
    }
    contract = compile_contract_from_state(state, origin="test")
    queries = [item.query for item in _plan_entity_queries(contract, set())]
    assert any("OODS" in q and "Object-Oriented Design System" in q for q in queries), queries


def test_axe_core_disambiguation_does_not_fire(monkeypatch) -> None:
    """Hyphenated tool stays unchanged in the planner."""

    outcome = EntityExtractionOutcome(
        entity_names=["axe-core"],
        disambiguation={},
        extracted=[],
        rejected=[],
        path="llm_primary",
    )
    _stub_llm_extractor(monkeypatch, outcome)

    state = {
        "mission_id": "AXECORE-NOOP-TEST",
        "mission_context": {
            "title": "axe-core unambiguous",
            "objective": "Catalogue axe-core capabilities.",
            "success_criteria": [],
            "deliverables": [],
            "required_entities": [],
        },
    }
    contract = compile_contract_from_state(state, origin="test")
    queries = [item.query for item in _plan_entity_queries(contract, set())]
    # The axe-core query must be exactly the canonical form, no augmentation.
    assert queries == ["axe-core"], queries


# --------------------------------------------------------------------------- #
# 6. Regex-fallback path discipline
# --------------------------------------------------------------------------- #


def test_regex_fallback_path_disambiguation_is_noop(monkeypatch) -> None:
    """When the LLM call returns None: regex extractor runs, no query mutation."""

    _stub_llm_extractor(monkeypatch, None)

    # S59.2 — required_entities intentionally empty so the regex_fallback path
    # is reachable. When the author has declared, the contract compiler now
    # short-circuits the extractor entirely; that path is exercised by
    # ``test_required_entities_short_circuits_extractor`` below.
    state = {
        "mission_id": "REGEX-FALLBACK-TEST",
        "mission_context": {
            "title": "Regex fallback discipline",
            "objective": "Identify gaps where OODS tooling could add value.",
            "success_criteria": [],
            "deliverables": [],
            "required_entities": [],
        },
    }
    contract = compile_contract_from_state(state, origin="test")
    # Regex extractor produced entities (OODS appears in the objective).
    assert any(entity.lower() == "oods" for entity in contract.named_entities)
    # Disambiguation map empty (no LLM signal).
    assert contract.named_entities_disambiguation == {}
    # Telemetry tagged regex_fallback.
    paths = [
        row.get("reason")
        for row in contract.entity_extraction_decisions
        if row.get("decision") == "extraction_path"
    ]
    assert paths == ["regex_fallback"]
    # End-to-end: planner does NOT augment any entity query.
    queries = [item.query for item in _plan_entity_queries(contract, set())]
    assert all("Object-Oriented Design System" not in q for q in queries), queries


# --------------------------------------------------------------------------- #
# 7. Sprint 59.2 — declared-authoritative short-circuit
# --------------------------------------------------------------------------- #
