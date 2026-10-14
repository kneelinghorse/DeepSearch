"""S39.2 — ExecuteMissionRequest ↔ MissionContractPreviewRequest parity tests.

The /missions/execute and /missions/preview endpoints must accept the same
extended mission-context surface (required_entities, excluded_entities,
expected_output_schema, coverage_thresholds, validation_thresholds,
deliverable_format, min_loops, references). Until S39.2 only the preview
model carried these fields, so any direct execute submission lost them
before compile_contract_from_state ran — which is why slot-entity-semantic-
entropy kept failing to compile in paid smokes even though the fixture
authored required_entities correctly.

These tests lock:

* the shared ``ExtendedMissionFields`` mixin surface (Execute + Preview)
* camelCase alias coverage on every extended field
* round-trip parity: Execute and Preview produce identical compiled
  named_entities / evidence_slots for the same extended payload
* the runner's ``build_mission_context_from_execute_request`` helper
  threads every extended field into mission_context where the compiler
  reads it
* the research_depth normalization validator is shared (Preview gains
  the same reject-invalid behavior Execute has always had)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch.mission import compile_contract_from_state  # noqa: E402

from deepsearch.api.models.requests import (  # noqa: E402
    EXTENDED_CONTEXT_FIELDS,
    ExecuteMissionRequest,
    ExtendedMissionFields,
    MissionContext,
    MissionContractPreviewRequest,
    build_mission_context_from_execute_request,
)


EXTENDED_PAYLOAD: Dict[str, Any] = {
    "required_entities": ["SAPLMA", "CCS", "semantic entropy"],
    "excluded_entities": ["obsolete-method"],
    "expected_output_schema": [
        {
            "kind": "annotated_bibliography",
            "title": "Annotated bibliography",
            "format": "annotated_bibliography",
            "minimum_citation_count": 12,
        }
    ],
    "coverage_thresholds": {"entity": 0.75, "objective": 0.70},
    "validation_thresholds": {"annotation_depth": 0.80},
    "deliverable_format": "markdown",
    "min_loops": 3,
    "references": [{"id": "ref-1", "title": "Reference paper on SAPLMA"}],
    "authored_scope": {"max_words": 500},
}


def _make_execute_request(**overrides: Any) -> ExecuteMissionRequest:
    payload: Dict[str, Any] = {
        "mission_id": "S39.2-PARITY-TEST",
        "title": "ACC-BIB-EVID-01 parity verification mission",
        "objectives": [
            "Map the evidence base for SAPLMA, CCS, and semantic entropy "
            "approaches in agent evaluation."
        ],
        "success_criteria": [
            "Annotated bibliography of at least 12 papers covering SAPLMA, "
            "CCS, and semantic entropy",
        ],
        "research_depth": "deep",
        "max_loops": 5,
        **EXTENDED_PAYLOAD,
    }
    payload.update(overrides)
    return ExecuteMissionRequest(**payload)


def test_execute_request_accepts_all_extended_fields_snake_case() -> None:
    """Mission SC #1: ExecuteMissionRequest carries every extended field."""

    request = _make_execute_request()

    assert request.required_entities == ["SAPLMA", "CCS", "semantic entropy"]
    assert request.excluded_entities == ["obsolete-method"]
    assert request.expected_output_schema
    assert request.coverage_thresholds == {"entity": 0.75, "objective": 0.70}
    assert request.validation_thresholds == {"annotation_depth": 0.80}
    assert request.deliverable_format == "markdown"
    assert request.min_loops == 3
    assert request.references and request.references[0]["id"] == "ref-1"


def test_execute_request_accepts_camel_case_aliases() -> None:
    """Mission SC #2: camelCase aliases mirror the preview model exactly."""

    camel_payload = {
        "mission_id": "S39.2-PARITY-CAMEL",
        "title": "ACC-BIB-EVID-01 camelCase parity verification mission",
        "objectives": ["Map the evidence base for SAPLMA and CCS."],
        "success_criteria": ["Annotated bibliography of at least 5 papers"],
        "requiredEntities": ["SAPLMA", "CCS"],
        "excludedEntities": ["legacy-X"],
        "expectedOutputSchema": [{"kind": "comparison_table", "title": "Table", "format": "table"}],
        "coverageThresholds": {"entity": 0.8},
        "validationThresholds": {"annotation_depth": 0.7},
        "deliverableFormat": "markdown",
        "minLoops": 2,
        "researchDepth": "deep",
        "maxLoops": 4,
    }

    request = ExecuteMissionRequest(**camel_payload)

    assert request.required_entities == ["SAPLMA", "CCS"]
    assert request.excluded_entities == ["legacy-X"]
    assert request.expected_output_schema
    assert request.coverage_thresholds == {"entity": 0.8}
    assert request.validation_thresholds == {"annotation_depth": 0.7}
    assert request.deliverable_format == "markdown"
    assert request.min_loops == 2
    assert request.research_depth == "deep"
    assert request.max_loops == 4


def test_execute_and_preview_share_extended_fields_mixin() -> None:
    """Shared source-of-truth: both models inherit the same field set."""

    assert issubclass(ExecuteMissionRequest, ExtendedMissionFields)
    assert issubclass(MissionContractPreviewRequest, ExtendedMissionFields)

    for field_name in (
        "required_entities",
        "excluded_entities",
        "expected_output_schema",
        "coverage_thresholds",
        "validation_thresholds",
        "deliverable_format",
        "min_loops",
        "references",
    ):
        assert field_name in ExecuteMissionRequest.model_fields, (
            f"Execute request is missing shared field: {field_name}"
        )
        assert field_name in MissionContractPreviewRequest.model_fields, (
            f"Preview request is missing shared field: {field_name}"
        )


def test_build_mission_context_propagates_extended_fields_into_payload() -> None:
    """Mission SC #4: worker unpack path carries every extended field into context."""

    request = _make_execute_request(
        context=MissionContext(background="prior work", focus="SAPLMA mechanism"),
    )

    context = build_mission_context_from_execute_request(request)

    assert context["title"] == request.title
    assert context["objective"] == "; ".join(request.objectives)
    assert context["success_criteria"] == request.success_criteria
    assert context["background"] == "prior work"
    assert context["focus"] == "SAPLMA mechanism"

    for field_name in EXTENDED_CONTEXT_FIELDS:
        assert field_name in context, (
            f"build_mission_context dropped extended field: {field_name}"
        )
    assert context["required_entities"] == ["SAPLMA", "CCS", "semantic entropy"]
    # min_loops travels on state['min_loops'], not inside mission_context.
    assert "min_loops" not in context


def test_execute_compile_registers_required_entity_slots() -> None:
    """Mission SC #5: Execute path produces slot-entity-{saplma,ccs,semantic-entropy}."""

    request = _make_execute_request()
    context = build_mission_context_from_execute_request(request)

    state = {
        "mission_id": request.mission_id,
        "mission_context": context,
        "mission_objectives": request.success_criteria,
        "deliverable_format": request.deliverable_format or "markdown",
        "research_depth": request.research_depth or "deep",
        "max_loops": request.max_loops,
        "min_loops": request.min_loops or 0,
        "depth_config": {},
    }
    contract = compile_contract_from_state(state, origin="test")

    slot_ids = {slot.id for slot in contract.evidence_slots}
    for expected in (
        "slot-entity-saplma",
        "slot-entity-ccs",
        "slot-entity-semantic-entropy",
    ):
        assert expected in slot_ids, (
            f"Execute path must compile {expected} from required_entities; "
            f"got slots: {sorted(slot_ids)}"
        )


def test_execute_and_preview_produce_identical_entity_slots() -> None:
    """Mission SC #6: same extended payload ⇒ identical compiled contract entity/slots.

    Proves parity, not just accept-without-reject. The payload shapes differ
    slightly (Execute uses ``objectives``/``success_criteria``, Preview uses
    ``objective``/``success_criteria``) but every field that shapes the
    compiled contract must produce the same named_entities and entity slots.
    """

    execute_request = _make_execute_request()
    execute_context = build_mission_context_from_execute_request(execute_request)

    preview_request = MissionContractPreviewRequest(
        mission_id="S39.2-PARITY-TEST",
        title=execute_request.title,
        objective=execute_request.objectives[0],
        success_criteria=execute_request.success_criteria,
        research_depth="deep",
        max_loops=5,
        **EXTENDED_PAYLOAD,
    )
    preview_context = preview_request.model_dump(by_alias=False, exclude_none=True)

    def _compile(context: Dict[str, Any], mission_objectives: list[str]) -> Any:
        state = {
            "mission_id": "S39.2-PARITY-TEST",
            "mission_context": context,
            "mission_objectives": mission_objectives,
            "deliverable_format": context.get("deliverable_format") or "markdown",
            "research_depth": context.get("research_depth") or "deep",
            "max_loops": context.get("max_loops") or 5,
            "min_loops": context.get("min_loops") or 0,
            "depth_config": {},
        }
        return compile_contract_from_state(state, origin="test")

    execute_contract = _compile(execute_context, execute_request.success_criteria)
    preview_contract = _compile(preview_context, preview_request.success_criteria)

    entity_slots_execute = {
        slot.id
        for slot in execute_contract.evidence_slots
        if slot.target.get("entity")
    }
    entity_slots_preview = {
        slot.id
        for slot in preview_contract.evidence_slots
        if slot.target.get("entity")
    }
    assert entity_slots_execute == entity_slots_preview, (
        f"entity-slot set differs between execute and preview paths: "
        f"execute={sorted(entity_slots_execute)}, "
        f"preview={sorted(entity_slots_preview)}"
    )

    assert set(execute_contract.named_entities) == set(preview_contract.named_entities)


def test_execute_and_preview_share_research_depth_validator() -> None:
    """Mission SC #7: shared validator rejects bad depth strings on both models."""

    with pytest.raises(ValueError):
        ExecuteMissionRequest(
            mission_id="S39.2-INVALID-DEPTH",
            title="Invalid depth smoke test scenario",
            objectives=["Check that depth validator rejects junk strings"],
            success_criteria=["sc"],
            research_depth="ultra",
        )

    with pytest.raises(ValueError):
        MissionContractPreviewRequest(
            mission_id="S39.2-INVALID-DEPTH",
            objective="Check that depth validator rejects junk strings",
            research_depth="ultra",
        )


def test_execute_default_values_preserve_backward_compat() -> None:
    """Execute requests missing every extended field still validate (back-compat)."""

    request = ExecuteMissionRequest(
        mission_id="S39.2-BACKCOMPAT",
        title="Backward compatibility check",
        objectives=["Assert that legacy execute payloads still accept"],
        success_criteria=["Produce a short writeup"],
    )

    assert request.required_entities == []
    assert request.excluded_entities == []
    assert request.expected_output_schema is None
    assert request.coverage_thresholds is None
    assert request.validation_thresholds is None
    assert request.deliverable_format is None
    assert request.min_loops is None
    assert request.references is None


def test_build_mission_context_drops_empty_extended_fields() -> None:
    """Empty/default values must not leak into mission_context.

    The compiler's ``_normalize_string_list`` and ``_extract_threshold_overrides``
    tolerate missing keys but treat an empty list as an explicit clearing
    signal. Keeping absent-when-unused preserves legacy behavior for payloads
    that set only a subset of extended fields.
    """

    request = ExecuteMissionRequest(
        mission_id="S39.2-EMPTY",
        title="Empty extended fields stay absent",
        objectives=["Verify legacy payload shape still compiles"],
        success_criteria=["sc"],
    )

    context = build_mission_context_from_execute_request(request)

    assert "required_entities" not in context
    assert "excluded_entities" not in context
    assert "coverage_thresholds" not in context
    assert "validation_thresholds" not in context
    assert "expected_output_schema" not in context
    assert "deliverable_format" not in context
    assert "references" not in context


def test_execute_request_extended_fields_survive_json_roundtrip() -> None:
    """Job queue persists via model_validate_json — extended fields must survive."""

    request = _make_execute_request()
    roundtripped = ExecuteMissionRequest.model_validate_json(request.model_dump_json())

    assert roundtripped.required_entities == request.required_entities
    assert roundtripped.excluded_entities == request.excluded_entities
    assert roundtripped.coverage_thresholds == request.coverage_thresholds
    assert roundtripped.validation_thresholds == request.validation_thresholds
    assert roundtripped.expected_output_schema == request.expected_output_schema
    assert roundtripped.deliverable_format == request.deliverable_format
    assert roundtripped.min_loops == request.min_loops
    assert roundtripped.references == request.references
