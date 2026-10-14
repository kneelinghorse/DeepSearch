"""Source-coverage gate result types and pure aggregation helpers.

S79 m02.c (PD97 Option B) — honest renaming. The aggregate measures
*source-side* coverage: per-named-entity slot coverage, per-objective
reflection scores, evidence-slot fill status. It does NOT measure
deliverable-section presence in the final draft. Mission 2
(STRUCTURED-OUTPUT-VIEWER-01) returned ``coverage_gate_failures: []`` despite
the agent self-reporting Coverage 0% precisely because this gate was never
about deliverable shape.

S79 carries the gate forward by renaming its symbols (``SourceCoverageGateResult``,
``aggregate_source_coverage``, etc.) so the module name reflects what the
function checks. The S64+ codeagent path does not invoke
``aggregate_source_coverage`` (returns ``missing_source_coverage_gate_result()``
which surfaces ``coverage_gate_failures == []`` per S75 carry / PD102 #13).
The ``coverage_gate_failures`` field on ``ExecutionMetadata`` keeps its
public name so downstream consumers (Postgres, dashboard reporting) are
unaffected.

The deliverable-section coverage signal is the new ``deliverable_coverage``
field on ``synthesis_telemetry`` (computed by ``completeness_check`` at
codeagent_finalize_node post-synthesis). Telemetry-only — no gating
decision per Option B; aggressive Option A (gate on ≥30% deliverable
shortfall) deferred to S80 candidate after telemetry shows pattern across
3+ real missions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .concept_clustering import build_cluster_summaries

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_REQUIRED_CONTRACT_FIELDS = ("named_entities", "objectives", "evidence_slots")
_MISSING_COVERAGE_DETAIL = "Coverage gates were not computed before routing."


def _unique_strings(values: Iterable[Any]) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        items.append(text)
        seen.add(text)
    return items


def _sequence_payload(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _mapping_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


@dataclass(slots=True)
class EntityCoverageGate:
    """Coverage result for named-entity requirements."""

    passed: bool = False
    covered: List[str] = field(default_factory=list)
    uncovered: List[str] = field(default_factory=list)
    threshold: float = 1.0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "covered": list(self.covered),
            "uncovered": list(self.uncovered),
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ObjectiveCoverageGate:
    """Coverage result for contract objectives."""

    passed: bool = False
    per_objective_scores: Dict[str, float] = field(default_factory=dict)
    uncovered_objective_ids: List[str] = field(default_factory=list)
    threshold: float = 0.70
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "per_objective_scores": dict(self.per_objective_scores),
            "uncovered_objective_ids": list(self.uncovered_objective_ids),
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(slots=True)
class SlotCoverageGate:
    """Coverage result for evidence slots."""

    passed: bool = False
    total_slots: int = 0
    filled_slots: int = 0
    thin_slot_ids: List[str] = field(default_factory=list)
    unaddressed_slot_ids: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "total_slots": self.total_slots,
            "filled_slots": self.filled_slots,
            "thin_slot_ids": list(self.thin_slot_ids),
            "unaddressed_slot_ids": list(self.unaddressed_slot_ids),
            "detail": self.detail,
        }


@dataclass(slots=True)
class SourceCoverageGateResult:
    """Aggregate coverage result consumed by later routing stages."""

    passed: bool = False
    failed_gates: List[str] = field(default_factory=list)
    entity: EntityCoverageGate = field(default_factory=EntityCoverageGate)
    objective: ObjectiveCoverageGate = field(default_factory=ObjectiveCoverageGate)
    slot: SlotCoverageGate = field(default_factory=SlotCoverageGate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failed_gates": list(self.failed_gates),
            "entity": self.entity.to_dict(),
            "objective": self.objective.to_dict(),
            "slot": self.slot.to_dict(),
        }


def missing_source_coverage_gate_result() -> SourceCoverageGateResult:
    """Return an empty coverage result when routing lacks coverage data.

    Post-S64 entry-point flip the codeagent path does not run coverage gates.
    The deployed ``execution_metadata.final_outcome.coverage_gate_failures``
    must read ``[]`` (the pre-S75 sentinel that emitted on every codeagent run
    was killed in m01).
    """

    return SourceCoverageGateResult(
        passed=False,
        failed_gates=[],
        entity=EntityCoverageGate(detail=""),
        objective=ObjectiveCoverageGate(detail=""),
        slot=SlotCoverageGate(detail=""),
    )


def coerce_source_coverage_gate_result(value: Any) -> SourceCoverageGateResult:
    """Coerce serialized coverage state into a SourceCoverageGateResult."""

    if isinstance(value, SourceCoverageGateResult):
        return value

    payload = _mapping_payload(value)
    if not payload:
        return missing_source_coverage_gate_result()

    entity_payload = _mapping_payload(payload.get("entity"))
    objective_payload = _mapping_payload(payload.get("objective"))
    slot_payload = _mapping_payload(payload.get("slot"))

    return SourceCoverageGateResult(
        passed=bool(payload.get("passed")),
        failed_gates=_unique_strings(payload.get("failed_gates", [])),
        entity=EntityCoverageGate(
            passed=bool(entity_payload.get("passed")),
            covered=_unique_strings(entity_payload.get("covered", [])),
            uncovered=_unique_strings(entity_payload.get("uncovered", [])),
            threshold=float(entity_payload.get("threshold", 1.0) or 1.0),
            detail=str(entity_payload.get("detail") or ""),
        ),
        objective=ObjectiveCoverageGate(
            passed=bool(objective_payload.get("passed")),
            per_objective_scores={
                str(key): float(value or 0.0)
                for key, value in _mapping_payload(
                    objective_payload.get("per_objective_scores")
                ).items()
            },
            uncovered_objective_ids=_unique_strings(
                objective_payload.get("uncovered_objective_ids", [])
            ),
            threshold=float(objective_payload.get("threshold", 0.70) or 0.70),
            detail=str(objective_payload.get("detail") or ""),
        ),
        slot=SlotCoverageGate(
            passed=bool(slot_payload.get("passed")),
            total_slots=int(slot_payload.get("total_slots", 0) or 0),
            filled_slots=int(slot_payload.get("filled_slots", 0) or 0),
            thin_slot_ids=_unique_strings(slot_payload.get("thin_slot_ids", [])),
            unaddressed_slot_ids=_unique_strings(
                slot_payload.get("unaddressed_slot_ids", [])
            ),
            detail=str(slot_payload.get("detail") or ""),
        ),
    )


def aggregate_source_coverage(
    state: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
) -> SourceCoverageGateResult:
    """Aggregate slot, objective, and entity coverage into one result."""

    if not _contract_is_complete(contract):
        detail = "Mission contract missing one or more required coverage fields."
        return SourceCoverageGateResult(
            passed=False,
            failed_gates=["contract_incomplete"],
            entity=EntityCoverageGate(detail=detail),
            objective=ObjectiveCoverageGate(detail=detail),
            slot=SlotCoverageGate(detail=detail),
        )

    normalized_state = _mapping_payload(state)
    normalized_contract = _mapping_payload(contract)

    entity_gate = _aggregate_entity_gate(normalized_state, normalized_contract)
    objective_gate = _aggregate_objective_gate(normalized_state, normalized_contract)
    slot_gate = _aggregate_slot_gate(normalized_state, normalized_contract)

    failed_gates = [
        gate_name
        for gate_name, passed in (
            ("entity", entity_gate.passed),
            ("objective", objective_gate.passed),
            ("slot", slot_gate.passed),
        )
        if not passed
    ]
    return SourceCoverageGateResult(
        passed=not failed_gates,
        failed_gates=failed_gates,
        entity=entity_gate,
        objective=objective_gate,
        slot=slot_gate,
    )


def _contract_is_complete(contract: Mapping[str, Any] | None) -> bool:
    if not isinstance(contract, Mapping):
        return False
    return all(
        isinstance(contract.get(field_name), Sequence)
        and not isinstance(contract.get(field_name), (str, bytes))
        for field_name in _REQUIRED_CONTRACT_FIELDS
    )


def _aggregate_entity_gate(
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> EntityCoverageGate:
    named_entities = _unique_strings(contract.get("named_entities", []))
    threshold = 1.0
    if not named_entities:
        return EntityCoverageGate(
            passed=True,
            threshold=threshold,
            detail="No named entities were required by the mission contract.",
        )

    slot_statuses = _entity_slot_statuses(state, contract)
    covered = [
        entity
        for entity in named_entities
        if slot_statuses.get(entity) and all(status == "filled" for status in slot_statuses[entity])
    ]
    uncovered = [entity for entity in named_entities if entity not in covered]
    passed = not uncovered
    return EntityCoverageGate(
        passed=passed,
        covered=covered,
        uncovered=uncovered,
        threshold=threshold,
        detail=(
            f"Covered {len(covered)}/{len(named_entities)} named entities "
            f"with slot-backed evidence from the working set."
        ),
    )


def _entity_slot_statuses(
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, List[str]]:
    entity_slots: Dict[str, List[str]] = {}
    for slot in _sequence_payload(contract.get("evidence_slots")):
        payload = _mapping_payload(slot)
        target = _mapping_payload(payload.get("target"))
        entity = str(target.get("entity") or "").strip()
        slot_id = str(payload.get("id") or "").strip()
        if entity and slot_id:
            entity_slots.setdefault(entity, []).append(slot_id)

    raw_slot_coverage = _mapping_payload(state.get("slot_coverage"))
    slot_rows = [
        _mapping_payload(item)
        for item in _sequence_payload(raw_slot_coverage.get("slots"))
        if isinstance(item, Mapping)
    ]
    status_by_slot = {
        str(item.get("slot_id") or "").strip(): str(item.get("status") or "").strip()
        for item in slot_rows
        if str(item.get("slot_id") or "").strip()
    }
    return {
        entity: [
            status_by_slot.get(slot_id, "unaddressed")
            for slot_id in slot_ids
        ]
        for entity, slot_ids in entity_slots.items()
    }


def _aggregate_objective_gate(
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> ObjectiveCoverageGate:
    threshold = _resolve_threshold(contract, "objective", 0.70)
    reflection_scores = _mapping_payload(state.get("reflection_scores"))
    objectives = [
        _mapping_payload(objective)
        for objective in _sequence_payload(contract.get("objectives"))
        if isinstance(objective, Mapping)
    ]
    per_objective_scores = _normalize_per_objective_scores(
        reflection_scores.get("per_objective_scores"),
        objectives,
    )

    required_objective_ids: List[str] = []
    for payload in objectives:
        objective_id = str(payload.get("id") or "").strip()
        if not objective_id:
            continue
        if bool(payload.get("required", True)):
            required_objective_ids.append(objective_id)

    uncovered = [
        objective_id
        for objective_id in required_objective_ids
        if per_objective_scores.get(objective_id, 0.0) < threshold
    ]
    passed = not uncovered
    return ObjectiveCoverageGate(
        passed=passed,
        per_objective_scores=per_objective_scores,
        uncovered_objective_ids=uncovered,
        threshold=threshold,
        detail=(
            f"{len(required_objective_ids) - len(uncovered)}/"
            f"{len(required_objective_ids)} required objectives meet the score threshold."
        ),
    )


def _aggregate_slot_gate(
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> SlotCoverageGate:
    raw_slot_coverage = _mapping_payload(state.get("slot_coverage"))
    contract_slot_ids = [
        str(slot.get("id") or "").strip()
        for slot in _sequence_payload(contract.get("evidence_slots"))
        if isinstance(slot, Mapping) and str(slot.get("id") or "").strip()
    ]
    thin_slot_ids = _extract_slot_ids(raw_slot_coverage.get("thin_slots"))
    unaddressed_slot_ids = _extract_slot_ids(raw_slot_coverage.get("unaddressed_slots"))

    if not raw_slot_coverage and contract_slot_ids:
        unaddressed_slot_ids = contract_slot_ids

    total_slots = int(raw_slot_coverage.get("total_slots") or len(contract_slot_ids))
    filled_slots = int(raw_slot_coverage.get("filled_slots") or 0)
    passed = total_slots == 0 or not unaddressed_slot_ids
    return SlotCoverageGate(
        passed=passed,
        total_slots=total_slots,
        filled_slots=filled_slots,
        thin_slot_ids=thin_slot_ids,
        unaddressed_slot_ids=unaddressed_slot_ids,
        detail=f"Filled {filled_slots}/{total_slots} evidence slots.",
    )


def _extract_slot_ids(values: Any) -> List[str]:
    slot_ids: List[str] = []
    for item in _sequence_payload(values):
        payload = _mapping_payload(item)
        slot_id = str(payload.get("slot_id") or payload.get("id") or "").strip()
        if slot_id:
            slot_ids.append(slot_id)
    return _unique_strings(slot_ids)


def _entity_is_covered(entity: str, search_texts: Sequence[str]) -> bool:
    return any(_entity_matches(entity, text) for text in search_texts)


def _entity_matches(entity: str, text: str) -> bool:
    normalized_entity = _normalized_match_text(entity)
    normalized_text = _normalized_match_text(text)
    if not normalized_entity or not normalized_text:
        return False
    pattern = r"\b" + re.escape(normalized_entity).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None


def _normalized_match_text(value: Any) -> str:
    lowered = str(value or "").lower()
    collapsed = _SLUG_PATTERN.sub(" ", lowered)
    return " ".join(collapsed.split())


def _entity_search_texts(state: Mapping[str, Any]) -> List[str]:
    search_texts: List[str] = []
    for summary in _cluster_summaries_for_matching(state):
        payload = _mapping_payload(summary)
        search_texts.extend(
            _unique_strings(
                [
                    payload.get("representative_summary"),
                    *(_sequence_payload(payload.get("citations")) or ()),
                ]
            )
        )
    for cluster in _sequence_payload(state.get("concept_clusters")):
        payload = _mapping_payload(cluster)
        search_texts.extend(_unique_strings(payload.get("merged_citations")))
    # S83 m06 (PD83-b): the working_set/sources_found title source was deleted —
    # both fields were stripped from AgentState at S78, so it only ever returned [].
    return _unique_strings(search_texts)


def _cluster_summaries_for_matching(state: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw_cluster_summaries = [
        _mapping_payload(summary)
        for summary in _sequence_payload(state.get("cluster_summaries"))
        if isinstance(summary, Mapping)
    ]
    if raw_cluster_summaries:
        return raw_cluster_summaries
    return build_cluster_summaries(
        state.get("finding_records") or [],
        state.get("concept_clusters") or [],
    )


def _normalize_per_objective_scores(
    raw_scores: Any,
    objectives: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    raw_score_map: Dict[str, float] = {}
    if isinstance(raw_scores, Mapping):
        raw_score_map = {
            str(objective_id).strip(): _coerce_float(score)
            for objective_id, score in raw_scores.items()
            if str(objective_id).strip()
        }
    else:
        for entry in _sequence_payload(raw_scores):
            payload = _mapping_payload(entry)
            objective_id = str(payload.get("objective_id") or payload.get("id") or "").strip()
            if not objective_id:
                continue
            raw_score_map[objective_id] = _coerce_float(payload.get("score"))

    normalized: Dict[str, float] = {}
    required_objective_index = 0
    for objective in objectives:
        objective_id = str(objective.get("id") or "").strip()
        if not objective_id:
            continue
        alias_index: int | None = None
        # Reflection emits OBJnn IDs from the required-objective list only;
        # optional umbrella objectives must not consume those aliases.
        if bool(objective.get("required", True)):
            required_objective_index += 1
            alias_index = required_objective_index
        for alias in _objective_id_aliases(
            objective_id,
            required_objective_index=alias_index,
        ):
            if alias in raw_score_map:
                normalized[objective_id] = raw_score_map[alias]
                break
    return normalized


def _objective_id_aliases(
    objective_id: str,
    *,
    required_objective_index: int | None,
) -> List[str]:
    aliases = [objective_id]
    if required_objective_index is not None:
        aliases.append(f"OBJ{required_objective_index:02d}")
        if required_objective_index == 1:
            aliases.append("primary")
        else:
            aliases.append(f"sc_{required_objective_index - 2}")
    return _unique_strings(aliases)


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_threshold(
    contract: Mapping[str, Any],
    gate_name: str,
    default: float,
) -> float:
    for key in ("coverage_thresholds", "validation_thresholds"):
        payload = _mapping_payload(contract.get(key))
        value = payload.get(gate_name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


__all__ = [
    "SourceCoverageGateResult",
    "EntityCoverageGate",
    "ObjectiveCoverageGate",
    "SlotCoverageGate",
    "aggregate_source_coverage",
]
