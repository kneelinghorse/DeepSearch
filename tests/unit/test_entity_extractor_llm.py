"""Unit tests for the S54.2 LLM-first entity extractor + contract wiring."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pydantic")

from deepsearch.mission.entity_extractor_llm import (
    EntityExtractionOutcome,
    EntityExtractionResult,
    ExtractedEntity,
    LLMEntityExtractor,
    run_llm_entity_extraction_sync,
)


# --------------------------------------------------------------------------- #
# LLMEntityExtractor + run_llm_entity_extraction_sync
# --------------------------------------------------------------------------- #


class _FakeStructuredExtractor:
    def __init__(self, result: EntityExtractionResult | None) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def extract(self, *, system_instruction: str, prompt: str, schema: type) -> Any:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "prompt": prompt,
                "schema": schema,
            }
        )
        return self.result


def test_run_llm_entity_extraction_sync_happy_path() -> None:
    """SEM-R09 enumerated tools: every named item appears as a separate slot."""

    result = EntityExtractionResult(
        entities=[
            ExtractedEntity(name="axe-core"),
            ExtractedEntity(name="eslint-plugin-jsx-a11y"),
            ExtractedEntity(name="Pa11y"),
            ExtractedEntity(
                name="OODS",
                disambiguating_context="Object-Oriented Design System",
                is_acronym=True,
            ),
            ExtractedEntity(
                name="including review workflows",
                rejected=True,
                rejection_reason="participial_phrase_fragment",
            ),
        ]
    )
    fake = _FakeStructuredExtractor(result)
    extractor = LLMEntityExtractor(structured_extractor=fake)

    outcome = run_llm_entity_extraction_sync(
        extractor,
        objectives=["Catalogue semantic enforcement tools such as axe-core, ..."],
        deliverables=[],
        required_entities=[],
    )

    assert outcome is not None
    assert outcome.entity_names == ["axe-core", "eslint-plugin-jsx-a11y", "Pa11y", "OODS"]
    assert outcome.disambiguation == {"OODS": "Object-Oriented Design System"}
    assert len(outcome.rejected) == 1
    assert outcome.rejected[0].name == "including review workflows"
    assert outcome.path == "llm_primary"
    # System instruction must reach the structured extractor.
    assert "REJECT participial phrases" in fake.calls[0]["system_instruction"]


def test_run_llm_entity_extraction_sync_returns_none_on_failure() -> None:
    fake = _FakeStructuredExtractor(None)
    extractor = LLMEntityExtractor(structured_extractor=fake)
    outcome = run_llm_entity_extraction_sync(
        extractor,
        objectives=["x"],
        deliverables=[],
        required_entities=[],
    )
    assert outcome is None


def test_extracted_entity_query_surfaces_default_empty_list() -> None:
    """S56.3: query_surfaces defaults to an empty list on the schema model."""

    entity = ExtractedEntity(name="axe-core")
    assert entity.query_surfaces == []


def test_system_instruction_includes_query_surfaces_rule() -> None:
    """S56.3: rule 7 instructs the LLM to populate retrieval-friendly variants."""

    from deepsearch.mission.entity_extractor_llm import SYSTEM_INSTRUCTION

    assert "7." in SYSTEM_INSTRUCTION
    assert "query_surfaces" in SYSTEM_INSTRUCTION
    assert "retrieval-friendly variants" in SYSTEM_INSTRUCTION
    # Rules of thumb (a)-(d) all referenced.
    for clue in (
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "acronym alone",
        "head noun phrase",
        "well-known concrete tool name",
    ):
        assert clue in SYSTEM_INSTRUCTION, f"missing rule-of-thumb {clue!r}"


def test_run_llm_entity_extraction_sync_populates_query_surfaces() -> None:
    """S56.3: outcome.query_surfaces carries the LLM-emitted variants."""

    result = EntityExtractionResult(
        entities=[
            ExtractedEntity(
                name="automated ARIA validation",
                query_surfaces=["ARIA validation", "ARIA", "axe-core"],
            ),
            ExtractedEntity(
                name="axe-core",
                query_surfaces=[],
            ),
            ExtractedEntity(
                name="OODS",
                disambiguating_context="Object-Oriented Design System",
                is_acronym=True,
                # Self-reference and a duplicate to exercise dedup.
                query_surfaces=["OODS", "Object-Oriented Design System", "OODS"],
            ),
        ]
    )
    extractor = LLMEntityExtractor(structured_extractor=_FakeStructuredExtractor(result))
    outcome = run_llm_entity_extraction_sync(
        extractor,
        objectives=["x"],
        deliverables=[],
        required_entities=[],
    )

    assert outcome is not None
    assert outcome.query_surfaces["automated ARIA validation"] == [
        "ARIA validation",
        "ARIA",
        "axe-core",
    ]
    # Empty list entries do not pollute the map.
    assert "axe-core" not in outcome.query_surfaces
    # Self-reference (canonical name) is filtered; remaining surfaces deduped.
    assert outcome.query_surfaces["OODS"] == ["Object-Oriented Design System"]


def test_run_llm_entity_extraction_sync_dedupes() -> None:
    result = EntityExtractionResult(
        entities=[
            ExtractedEntity(name="axe-core"),
            ExtractedEntity(name="AXE-core"),  # case-different duplicate
            ExtractedEntity(name=""),  # empty name skipped
        ]
    )
    extractor = LLMEntityExtractor(structured_extractor=_FakeStructuredExtractor(result))
    outcome = run_llm_entity_extraction_sync(
        extractor,
        objectives=["x"],
        deliverables=[],
        required_entities=[],
    )
    assert outcome is not None
    assert outcome.entity_names == ["axe-core"]


# --------------------------------------------------------------------------- #
# Contract compiler integration
# --------------------------------------------------------------------------- #


def _base_state(*, objective: str, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "mission_id": "S54-TEST",
        "mission_context": {
            "title": "S54 LLM Entity Extraction Test",
            "objective": objective,
            "deliverables": [],
            "success_criteria": [],
            "required_entities": required or [],
        },
    }


def _stub_settings_with_outcome(
    monkeypatch, outcome: EntityExtractionOutcome | None
) -> None:
    """Patch the contract module so the LLM path returns ``outcome`` deterministically."""

    from deepsearch.mission import contract as contract_module
    from deepsearch.mission import entity_extractor_llm

    class _StubSettings:
        @classmethod
        def load(cls):
            return cls()

        def build_structured_extractor(self):
            return object()  # actual structured extractor never used

    monkeypatch.setattr(contract_module, "DeepSearchSettings", _StubSettings, raising=False)

    # Patch the inner-import path used by _extract_entities_llm_first.
    import deepsearch.config as config_module

    monkeypatch.setattr(config_module, "DeepSearchSettings", _StubSettings, raising=False)

    monkeypatch.setattr(
        entity_extractor_llm,
        "run_llm_entity_extraction_sync",
        lambda *args, **kwargs: outcome,
    )


def test_compile_contract_uses_llm_extractor_output(monkeypatch) -> None:
    """A successful LLM call populates named_entities and the disambiguation map."""

    from deepsearch.mission.contract import compile_contract_from_state

    outcome = EntityExtractionOutcome(
        entity_names=["axe-core", "eslint-plugin-jsx-a11y", "Pa11y", "OODS"],
        disambiguation={"OODS": "Object-Oriented Design System"},
        extracted=[],
        rejected=[
            ExtractedEntity(
                name="including review workflows",
                rejected=True,
                rejection_reason="participial_phrase_fragment",
            )
        ],
        path="llm_primary",
    )
    _stub_settings_with_outcome(monkeypatch, outcome)

    state = _base_state(
        objective=(
            "Catalogue semantic enforcement tools such as axe-core, "
            "eslint-plugin-jsx-a11y, Pa11y, and identify gaps where OODS "
            "tooling could add value."
        ),
        required=[],
    )

    contract = compile_contract_from_state(state, origin="test")

    # SEM-R09 anti-regression: enumerated tools all appear as entity slots.
    for entity in ("axe-core", "eslint-plugin-jsx-a11y", "Pa11y"):
        assert any(name.lower() == entity.lower() for name in contract.named_entities), (
            f"missing entity {entity}"
        )

    # SEM-R09 anti-regression: OODS has disambiguating context propagated.
    assert contract.named_entities_disambiguation.get("OODS") == "Object-Oriented Design System"

    # Participial fragment must NOT appear as an entity slot.
    assert not any(
        "including review workflows" in entity.lower() for entity in contract.named_entities
    )

    # Audit row for the LLM rejection must be visible.
    rejection_rows = [
        row
        for row in contract.entity_extraction_decisions
        if row.get("decision") == "rejected_llm"
    ]
    assert len(rejection_rows) == 1
    assert rejection_rows[0]["entity"] == "including review workflows"

    # Telemetry path tagged llm_primary.
    paths = [
        row.get("reason")
        for row in contract.entity_extraction_decisions
        if row.get("decision") == "extraction_path"
    ]
    assert paths == ["llm_primary"]


def test_compile_contract_falls_back_to_regex_when_llm_fails(monkeypatch) -> None:
    """When the LLM extractor returns None, the regex extractor still runs."""

    from deepsearch.mission.contract import compile_contract_from_state

    _stub_settings_with_outcome(monkeypatch, None)

    state = _base_state(
        objective="Survey axe-core, Pa11y, and ARIA validators",
        required=[],
    )
    contract = compile_contract_from_state(state, origin="test")

    # Disambiguation map empty (no LLM context).
    assert contract.named_entities_disambiguation == {}
    paths = [
        row.get("reason")
        for row in contract.entity_extraction_decisions
        if row.get("decision") == "extraction_path"
    ]
    assert paths == ["regex_fallback"]
    # Regex path still produced entity slots.
    assert contract.named_entities, "regex fallback should produce some entities"


def test_compile_contract_drops_disambiguation_for_excluded_entities(
    monkeypatch,
) -> None:
    """When an LLM-extracted entity is later excluded, its disambiguation must drop too."""

    from deepsearch.mission.contract import compile_contract_from_state

    outcome = EntityExtractionOutcome(
        entity_names=["axe-core", "OODS"],
        disambiguation={"OODS": "Object-Oriented Design System"},
        extracted=[],
        rejected=[],
        path="llm_primary",
    )
    _stub_settings_with_outcome(monkeypatch, outcome)

    state = _base_state(
        objective="Catalogue axe-core and OODS tooling",
        required=[],
    )
    state["mission_context"]["excluded_entities"] = ["OODS"]

    contract = compile_contract_from_state(state, origin="test")
    assert "OODS" not in [e for e in contract.named_entities]
    assert "OODS" not in contract.named_entities_disambiguation


def test_compile_contract_plumbs_query_surfaces_from_llm_outcome(monkeypatch) -> None:
    """S56.3: named_entities_query_surfaces is populated from outcome.query_surfaces."""

    from deepsearch.mission.contract import compile_contract_from_state

    outcome = EntityExtractionOutcome(
        entity_names=["axe-core", "automated ARIA validation"],
        disambiguation={},
        extracted=[],
        rejected=[],
        path="llm_primary",
        query_surfaces={
            "automated ARIA validation": ["ARIA validation", "ARIA", "axe-core"],
        },
    )
    _stub_settings_with_outcome(monkeypatch, outcome)

    state = _base_state(
        objective="Catalogue automated ARIA validation tooling such as axe-core",
        required=[],
    )
    contract = compile_contract_from_state(state, origin="test")

    assert contract.named_entities_query_surfaces == {
        "automated ARIA validation": ["ARIA validation", "ARIA", "axe-core"],
    }


def test_compile_contract_drops_query_surfaces_for_excluded_entities(monkeypatch) -> None:
    """S56.3: surfaces map cleanup mirrors the disambiguation cleanup."""

    from deepsearch.mission.contract import compile_contract_from_state

    outcome = EntityExtractionOutcome(
        entity_names=["axe-core", "OODS"],
        disambiguation={"OODS": "Object-Oriented Design System"},
        extracted=[],
        rejected=[],
        path="llm_primary",
        query_surfaces={"OODS": ["Object-Oriented Design System"], "axe-core": ["axe"]},
    )
    _stub_settings_with_outcome(monkeypatch, outcome)

    state = _base_state(
        objective="Catalogue axe-core and OODS tooling",
        required=[],
    )
    state["mission_context"]["excluded_entities"] = ["OODS"]

    contract = compile_contract_from_state(state, origin="test")
    assert "OODS" not in contract.named_entities_query_surfaces
    assert contract.named_entities_query_surfaces.get("axe-core") == ["axe"]
