"""S40.2 — Oxford-comma entity extractor verb coverage.

The S39.3 paid smoke failed RC-A because the committed ACC-BIB-EVID-01
fixture's objective uses the verb "Map" which was absent from the
enumerated-entity intro pattern. This suite locks in:

* Coverage of the new verb set (map, chart, catalog, document, review,
  investigate, study, explore) across both direct and scoped enumeration.
* The scoped ``Map X for A, B, and C`` shape — the failing production
  case — yields exactly the three authored required-entities.
* Stopword-prefix rejection suppresses "the X for Y" spurious first items.
* B35.8 negation semantics preserved: ``except``-bearing lists do not emit
  partial entity sets.
* Existing extractor tests (S37.1 compiler hardening, B35.8 bundler
  landscape) remain green.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

import deepsearch.mission.contract as mission_contract_module  # noqa: E402
from deepsearch.mission import (  # noqa: E402
    compile_contract_from_state,
    load_mission_from_yaml,
    prepare_execution_contract,
)


_FIXTURE = Path("tests/fixtures/missions/acceptance_acc_bib_evid_01.yaml")


def _compile(objective: str, *, required_entities: list[str] | None = None):
    """Compile a minimal contract from an objective string."""

    mission_context: dict[str, object] = {
        "title": "S40.2 extractor test",
        "objective": objective,
    }
    if required_entities is not None:
        mission_context["required_entities"] = required_entities
    return compile_contract_from_state(
        {
            "mission_id": "S40.2-TEST",
            "mission_context": mission_context,
            "mission_objectives": [objective],
            "deliverable_format": "markdown",
            "research_depth": "deep",
            "max_loops": 4,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )


def _extract(text: str) -> set[str]:
    return set(mission_contract_module._extract_enumerated_entities(text))


# ---------------------------------------------------------------------------
# The production failing case from S39.3
# ---------------------------------------------------------------------------


def test_map_scoped_enumeration_extracts_all_three_required_entities() -> None:
    """The committed ACC-BIB-EVID-01 objective must yield {SAPLMA, CCS, semantic entropy}.

    Without this, TraceLab's forwarder dropping ``required_entities`` leaves
    DeepSearch with no way to compile ``slot-entity-semantic-entropy`` —
    exactly the S39.3 §3.1 RC-A failure.
    """

    extracted = _extract(
        "Map the evidence base for SAPLMA, CCS, and semantic entropy "
        "approaches used in agent evaluation."
    )
    assert {"SAPLMA", "CCS", "semantic entropy"}.issubset(extracted)


def test_acc_bib_evid_01_fixture_compiles_three_required_entity_slots(
    tmp_path: Path,
) -> None:
    """Compiling the committed fixture must surface slot-entity-{saplma,ccs,semantic-entropy}."""

    state = load_mission_from_yaml(_FIXTURE)
    prepared = prepare_execution_contract(state, origin="yaml", output_dir=tmp_path)
    payload = prepared.path.read_text(encoding="utf-8")
    import json

    contract = json.loads(payload)
    slot_ids = {slot["id"] for slot in contract["evidence_slots"]}
    for expected in ("slot-entity-saplma", "slot-entity-ccs", "slot-entity-semantic-entropy"):
        assert expected in slot_ids, f"missing {expected}"


def test_scoped_extraction_drops_stopword_prefix_spurious_entity() -> None:
    """The scoped pattern must suppress 'the evidence base' — it is the
    prepositional prefix, not an entity. Direct pattern also rejects it
    via the new stopword-prefix guard.
    """

    extracted = _extract(
        "Map the evidence base for SAPLMA, CCS, and semantic entropy."
    )
    assert "the evidence base" not in extracted
    assert "the" not in extracted
    assert all(
        not item.lower().startswith("the ")
        and not item.lower().startswith("a ")
        for item in extracted
    )


# ---------------------------------------------------------------------------
# Verb coverage — each new verb in the S40.2 brief
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verb",
    ["map", "chart", "catalog", "document", "review", "investigate", "study", "explore"],
)
def test_new_verbs_fire_direct_enumeration(verb: str) -> None:
    text = f"{verb.capitalize()} SAPLMA, CCS, and semantic entropy."
    assert {"SAPLMA", "CCS", "semantic entropy"}.issubset(_extract(text))


@pytest.mark.parametrize(
    "verb",
    ["map", "chart", "catalog", "document", "review", "investigate", "study", "explore"],
)
def test_new_verbs_fire_scoped_enumeration(verb: str) -> None:
    text = f"{verb.capitalize()} the evidence base for SAPLMA, CCS, and semantic entropy."
    assert {"SAPLMA", "CCS", "semantic entropy"}.issubset(_extract(text))


# ---------------------------------------------------------------------------
# Oxford-comma variants
# ---------------------------------------------------------------------------


def test_two_item_list_and_variant() -> None:
    assert {"SAPLMA", "CCS"}.issubset(_extract("Study SAPLMA and CCS."))


def test_three_item_with_oxford_comma() -> None:
    assert {"SAPLMA", "CCS", "semantic entropy"}.issubset(
        _extract("Study SAPLMA, CCS, and semantic entropy.")
    )


def test_three_item_without_oxford_comma() -> None:
    assert {"SAPLMA", "CCS", "semantic entropy"}.issubset(
        _extract("Study SAPLMA, CCS and semantic entropy.")
    )


def test_four_item_with_oxford_comma() -> None:
    assert {"arxiv", "openalex", "semantic scholar", "openreview"}.issubset(
        _extract("Catalog arxiv, openalex, semantic scholar, and openreview.")
    )


# ---------------------------------------------------------------------------
# Lowercase multi-word entities
# ---------------------------------------------------------------------------


def test_lowercase_two_word_entity_survives() -> None:
    """'semantic entropy' is lowercase + two words — the bare acronym /
    Title Case patterns cannot surface it, so Oxford-comma extraction is
    the only compile path for it."""

    assert "semantic entropy" in _extract(
        "Review SAPLMA, CCS, and semantic entropy."
    )


def test_lowercase_entity_scoped_with_trailing_approaches_qualifier() -> None:
    """The literal S39.3 shape: ``semantic entropy`` followed by the
    trailing qualifier ``approaches used in agent evaluation`` must be
    recovered via the new mid-rejection-token truncation."""

    assert "semantic entropy" in _extract(
        "Map the evidence base for SAPLMA, CCS, and semantic entropy "
        "approaches used in agent evaluation."
    )


# ---------------------------------------------------------------------------
# Negation preservation (B35.8)
# ---------------------------------------------------------------------------


def test_map_with_except_does_not_emit_entities() -> None:
    """``Map everything except SAPLMA`` should not enumerate SAPLMA as an entity."""

    extracted = _extract("Map everything except SAPLMA, CCS, and semantic entropy.")
    assert "SAPLMA" not in extracted
    assert "CCS" not in extracted
    assert "semantic entropy" not in extracted


def test_survey_with_not_does_not_emit_entities() -> None:
    """Inline ``but not`` inside the enumeration bails extraction."""

    extracted = _extract("Survey X, Y, and Z, but not W.")
    # None of X/Y/Z/W should be emitted — safer than partial extraction.
    assert extracted == set()


def test_compile_contract_negation_preserves_b35_8_semantics() -> None:
    """The B35.8 bundler test scenario — constraint-style success criteria
    must not emit their negated-subject tokens as named entities, even
    with the extended verb set."""

    contract = compile_contract_from_state(
        {
            "mission_id": "B35.8-MIXED-S40",
            "mission_context": {
                "title": "JavaScript bundler landscape",
                "objective": (
                    "Compare Vite, esbuild, Webpack, Rollup, and Turbopack "
                    "architectures."
                ),
                "success_criteria": [
                    "Compare Vite, esbuild, Webpack, Rollup, and Turbopack architectures",
                    "No Medium, dev.to, LinkedIn, Towards Data Science used as primary citations",
                ],
            },
            "mission_objectives": [
                "Compare Vite, esbuild, Webpack, Rollup, and Turbopack architectures",
                "No Medium, dev.to, LinkedIn, Towards Data Science used as primary citations",
            ],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert {"Vite", "esbuild", "Webpack", "Rollup", "Turbopack"}.issubset(
        set(contract.named_entities)
    )
    assert "Medium" not in contract.named_entities
    assert "LinkedIn" not in contract.named_entities


# ---------------------------------------------------------------------------
# Regression — existing S37.1 compiler-hardening tests must stay green
# ---------------------------------------------------------------------------


def test_existing_compare_authoring_phrase_slot_drop_regression() -> None:
    contract = _compile(
        "Compare SAPLMA/self-evaluation, CCS, and semantic entropy as evidence bases."
    )
    slot_ids = {slot.id for slot in contract.evidence_slots}
    assert "slot-entity-ccs" in slot_ids
    assert "slot-entity-saplma-self-evaluation" not in slot_ids
    assert "slot-entity-semantic-entropy-as-evidence-bases" not in slot_ids
    assert "slot-entity-self-evaluation" in slot_ids
    assert "slot-entity-semantic-entropy" in slot_ids


def test_normalize_still_rejects_authoring_phrase_on_its_own() -> None:
    """The S37.1 invariant — the bare phrase ``semantic entropy as evidence bases``
    must still be rejected by the normalizer (the outer candidate generator
    is the only path that can surface it as a valid entity)."""

    assert (
        mission_contract_module._normalize_enumerated_entity(
            "semantic entropy as evidence bases"
        )
        is None
    )


def test_stopword_prefix_rejects_the_evidence_base() -> None:
    """Direct test of the new stopword-prefix reject path."""

    assert (
        mission_contract_module._normalize_enumerated_entity("the evidence base")
        is None
    )


def test_column_table_guard_still_bails_out() -> None:
    """Legacy guard: enumeration containing 'column' or 'table' is treated as
    a schema description and skipped."""

    assert (
        mission_contract_module._extract_enumerated_entities(
            "Compare SAPLMA, CCS, semantic entropy; table with columns Theme, Evidence, Gap."
        )
        == []
    )
