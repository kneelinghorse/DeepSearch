"""Deterministic regressions for Sprint 37 entity-compiler hardening."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

import deepsearch.mission.contract as mission_contract_module  # noqa: E402
from deepsearch.mission import (  # noqa: E402
    compile_contract_from_state,
    load_mission_from_yaml,
    prepare_execution_contract,
)

FIXTURE = Path("tests/fixtures/missions/acceptance_acc_bib_evid_01.yaml")


def _compile_contract(objective: str, *, title: str = "Entity Compiler Hardening") -> object:
    return compile_contract_from_state(
        {
            "mission_id": "S37.1-TEST",
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


def test_compiler_enriches_ccs_slot_and_drops_authoring_phrase_slots() -> None:
    contract = _compile_contract(
        "Compare SAPLMA/self-evaluation, CCS, and semantic entropy as evidence bases."
    )

    slots = {slot.id: slot for slot in contract.evidence_slots}
    assert "slot-entity-ccs" in slots
    ccs_target = slots["slot-entity-ccs"].target
    assert ccs_target["entity_name"] == "Contrast-Consistent Search"
    assert {"carbon capture", "ccs.neu.edu"}.issubset(set(ccs_target["rejection_patterns"]))

    assert "slot-entity-saplma-self-evaluation" not in slots
    assert "slot-entity-semantic-entropy-as-evidence-bases" not in slots
    assert "slot-entity-self-evaluation" in slots
    assert "slot-entity-semantic-entropy" in slots


def test_normalize_enumerated_entity_rejects_authoring_phrase_and_objective_keeps_it() -> None:
    assert mission_contract_module._normalize_enumerated_entity(
        "semantic entropy as evidence bases"
    ) is None

    contract = _compile_contract("Compare semantic entropy as evidence bases.")
    assert any(
        "semantic entropy as evidence bases" in objective.text.lower()
        for objective in contract.objectives
    )


def test_normalize_enumerated_entity_splits_composite_phrase() -> None:
    assert mission_contract_module._normalize_enumerated_entity(
        "SAPLMA/self-evaluation"
    ) == ["SAPLMA", "self-evaluation"]


def test_acceptance_fixture_serializes_ccs_rejection_patterns(tmp_path: Path) -> None:
    state = load_mission_from_yaml(FIXTURE)
    prepared = prepare_execution_contract(state, origin="yaml", output_dir=tmp_path)
    payload = json.loads(prepared.path.read_text(encoding="utf-8"))

    ccs_slots = [
        slot for slot in payload["evidence_slots"] if slot["id"] == "slot-entity-ccs"
    ]
    assert ccs_slots
    target = ccs_slots[0]["target"]
    assert target["entity_name"] == "Contrast-Consistent Search"
    assert {"carbon capture", "ccs.neu.edu"}.issubset(set(target["rejection_patterns"]))


def test_unknown_acronym_logs_missing_registry_once(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=mission_contract_module.__name__):
        _compile_contract(
            "Compare XYZQ and XYZQ adapters for tests.",
            title="XYZQ Review",
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if "Missing known-expansion registry entries" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "XYZQ" in messages[0]


# ---------------------------------------------------------------------------
# S38.3 — Wikipedia disambiguation detector + CCS rejection_patterns extension
# ---------------------------------------------------------------------------


def test_ccs_registry_includes_s38_3_rejection_patterns(tmp_path: Path) -> None:
    state = load_mission_from_yaml(FIXTURE)
    prepared = prepare_execution_contract(state, origin="yaml", output_dir=tmp_path)
    payload = json.loads(prepared.path.read_text(encoding="utf-8"))

    ccs_slots = [
        slot for slot in payload["evidence_slots"] if slot["id"] == "slot-entity-ccs"
    ]
    assert ccs_slots
    rejection_patterns = set(ccs_slots[0]["target"]["rejection_patterns"])

    # S37.1 patterns preserved.
    assert {"carbon capture", "ccs.neu.edu"}.issubset(rejection_patterns)

    # S38.3 additions.
    new_patterns = {
        "disambiguation",
        "may refer to",
        "circuit city",
        "cardcaptor",
        "cross-currency",
        "cross currency swap",
        "skateboard",
        "british music",
    }
    assert new_patterns.issubset(rejection_patterns)


def test_saplma_registry_rejection_patterns_unchanged() -> None:
    entry = mission_contract_module._known_acronym_entry("SAPLMA")
    assert entry is not None
    # SAPLMA registry has no rejection_patterns (S38.3 only extends CCS).
    rejection_patterns = entry.get("rejection_patterns") or ()
    assert tuple(rejection_patterns) == ()


