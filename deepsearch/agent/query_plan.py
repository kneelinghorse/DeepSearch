"""Contract-aware query planning primitives for retrieval orchestration."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence

from deepsearch.mission.contract import MissionContract, resolve_research_objectives

from . import refinement

logger = logging.getLogger(__name__)

QueryClass = Literal["entity", "concept", "deliverable_gap"]

_DISTRIBUTIVE_OBJECTIVE_PATTERN = re.compile(r"\b(each|every)\b", re.IGNORECASE)
_DISTRIBUTIVE_TARGET_PATTERN = re.compile(
    r"\b(each|every)\s+"
    r"(?:of\s+the\s+main\s+)?"
    r"(?:required\s+)?"
    r"(?:langgraph\s+)?"
    r"(?:checkpointer\s+)?"
    r"(?:backend|backends|classifier|classifiers|model|models|entity|entities|saver|savers)\b",
    re.IGNORECASE,
)
_DISTRIBUTIVE_FOR_EACH_PATTERN = re.compile(
    r"\bfor\s+each\s+"
    r"(?:required\s+)?"
    r"(?:langgraph\s+)?"
    r"(?:checkpointer\s+)?"
    r"(?:backend|backends|classifier|classifiers|model|models|entity|entities|saver|savers)\b",
    re.IGNORECASE,
)
_CONCEPT_QUERY_TRAILING_CONNECTIVES = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "than",
        "then",
        "to",
        "via",
        "with",
    }
)
_CONCEPT_QUERY_SC_FUSION_MIN_WINDOW = 4

# S39.1 — per-loop perspective rotation for paraphrase mutation. Cycled by
# loop_index so loop 3 tries a different angle than loop 2, loop 2 from loop 1.
_PARAPHRASE_PERSPECTIVES: tuple[str, ...] = (
    "methodology",
    "recent advances",
    "empirical evaluation",
    "limitations",
    "benchmarks",
    "survey",
)

_STRATEGY_SLOT_TARGETED = "slot_targeted"
_STRATEGY_SUB_OBJECTIVE = "sub_objective_decomposition"
_STRATEGY_PARAPHRASE = "paraphrase_mutation"
_DIVERSIFIED_STRATEGIES: tuple[str, ...] = (
    _STRATEGY_SLOT_TARGETED,
    _STRATEGY_SUB_OBJECTIVE,
    _STRATEGY_PARAPHRASE,
)

@dataclass(slots=True)
class PlannedQuery:
    """A retrieval query annotated with intent and contract slot targets."""

    query: str
    query_class: QueryClass
    target_slots: List[str]


class ContractAwareQueryPlannerError(Exception):
    """Raised when acceptance-contract planning cannot produce a valid plan."""

    def __init__(self, message: str, telemetry: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.telemetry = dict(telemetry)


def plan_queries(
    contract: MissionContract | Mapping[str, Any],
    *,
    gaps: Sequence[Mapping[str, Any]],
    mission_context: Mapping[str, Any] | None,
    previous_queries: Sequence[str] | None = None,
    objective_lookup: Sequence[str] | None = None,
    loop_index: int = 0,
    reflection_scores: Mapping[str, Any] | None = None,
    max_concept_queries: int = refinement.MAX_TARGETED_QUERIES,
    unfilled_slot_ids: Sequence[str] | None = None,
    diversity_telemetry: Dict[str, Any] | None = None,
) -> List[PlannedQuery]:
    """Build contract-aware planned queries for the current research loop.

    ``unfilled_slot_ids`` (B35.2): when provided, queries whose ``target_slots``
    intersect the unfilled-slot set are reordered to the front of the returned
    list. The reorder is a stable sort over the full plan, so existing class
    blocks (entity → concept → deliverable_gap) and within-block ordering are
    preserved when no slot is unfilled. Slot assignment (which slot each query
    binds to) is unchanged — only the order is.

    ``diversity_telemetry`` (S39.1): optional out-param. When provided, the
    planner populates it with per-loop strategy counts so downstream telemetry
    can observe how many novel queries each diversification strategy produced.
    The sink is cleared in-place and re-populated on every call.
    """

    compiled_contract = _coerce_contract(contract)
    previous = previous_queries or []
    scores = reflection_scores or {}
    resolved_objectives = resolve_research_objectives(
        compiled_contract,
        fallback=objective_lookup,
    )

    seen = {refinement._normalise_query(item) for item in refinement._clean_queries(previous)}
    planned: List[PlannedQuery] = []
    novel_per_strategy: Dict[str, int] = {}

    if loop_index == 0:
        entity_planned = _plan_entity_queries(compiled_contract, seen)
        planned.extend(entity_planned)
        if entity_planned:
            novel_per_strategy["entity_cold_start"] = len(entity_planned)

    concept_planned = _plan_concept_queries(
        compiled_contract,
        gaps=gaps,
        mission_context=mission_context,
        objective_lookup=resolved_objectives,
        seen=seen,
        max_queries=max_concept_queries,
    )
    planned.extend(concept_planned)
    if concept_planned:
        novel_per_strategy["concept_gap"] = len(concept_planned)

    deliverable_planned = _plan_deliverable_gap_queries(
        compiled_contract,
        mission_context=mission_context,
        reflection_scores=scores,
        seen=seen,
    )
    planned.extend(deliverable_planned)
    if deliverable_planned:
        novel_per_strategy["deliverable_gap"] = len(deliverable_planned)

    # S39.1 — diversification stage: fire on loop_index >= 2 so we produce
    # novel queries at loop 3-4 before the planner fail-loud guard trips.
    # Loops 0-1 are left to the original entity / gap-derived paths, which
    # still have natural novelty at that point. Each strategy respects the
    # shared ``seen`` set so anything collision-bound is dropped inside the
    # strategy rather than inflating counts here.
    if loop_index >= 2:
        diversified = _plan_diversified_queries(
            compiled_contract,
            mission_context=mission_context,
            objective_lookup=resolved_objectives,
            unfilled_slot_ids=unfilled_slot_ids,
            loop_index=loop_index,
            seen=seen,
            stats=novel_per_strategy,
        )
        planned.extend(diversified)

    if diversity_telemetry is not None:
        diversity_telemetry.clear()
        diversity_telemetry.update(
            _build_diversity_telemetry(
                loop_index=loop_index,
                novel_per_strategy=novel_per_strategy,
                planned=planned,
            )
        )

    return _reorder_by_unfilled_slots(planned, unfilled_slot_ids)


def _build_diversity_telemetry(
    *,
    loop_index: int,
    novel_per_strategy: Mapping[str, int],
    planned: Sequence[PlannedQuery],
) -> Dict[str, Any]:
    non_zero = {name: count for name, count in novel_per_strategy.items() if count > 0}
    diversified_fired = sorted(
        name for name in non_zero if name in _DIVERSIFIED_STRATEGIES
    )
    return {
        "loop": int(loop_index),
        "strategies_fired": sorted(non_zero.keys()),
        "diversified_strategies_fired": diversified_fired,
        "novel_per_strategy": {name: int(count) for name, count in non_zero.items()},
        "novel_total": int(len(planned)),
    }


def _reorder_by_unfilled_slots(
    planned: List[PlannedQuery],
    unfilled_slot_ids: Sequence[str] | None,
) -> List[PlannedQuery]:
    """Stable-sort queries so unfilled-slot targets come first.

    No-op when ``unfilled_slot_ids`` is empty / None. Within each priority
    bucket the original order is preserved, so when a planner adds entity
    queries followed by concept queries followed by deliverable-gap queries,
    that block ordering survives intact for any query that is NOT touching
    an unfilled slot.
    """

    if not unfilled_slot_ids:
        return planned

    unfilled = {str(slot_id).strip() for slot_id in unfilled_slot_ids if str(slot_id).strip()}
    if not unfilled:
        return planned

    def _priority(item: PlannedQuery) -> int:
        return 0 if any(slot in unfilled for slot in item.target_slots) else 1

    return [
        item
        for _, item in sorted(
            enumerate(planned),
            key=lambda pair: (_priority(pair[1]), pair[0]),
        )
    ]


def resolve_contract_objective_id(
    contract: MissionContract | Mapping[str, Any],
    objective_id: str,
    *,
    objective_lookup: Sequence[str] | None = None,
) -> str:
    """Resolve a raw objective identifier to the compiled contract objective id."""

    compiled_contract = _coerce_contract(contract)
    resolved_id = str(objective_id or "").strip()
    if not resolved_id:
        return ""
    objective_map = _contract_objective_id_map(compiled_contract, objective_lookup)
    return objective_map.get(resolved_id, objective_map.get(resolved_id.lower(), resolved_id))


def coerce_planned_query(query: str | PlannedQuery) -> PlannedQuery:
    """Normalize a legacy query string or ``PlannedQuery`` to ``PlannedQuery``."""

    if isinstance(query, PlannedQuery):
        return PlannedQuery(
            query=query.query,
            query_class=query.query_class,
            target_slots=list(query.target_slots),
        )
    return PlannedQuery(query=str(query).strip(), query_class="concept", target_slots=[])


def attach_query_provenance(
    sources: Sequence[Mapping[str, Any]],
    planned_query: PlannedQuery,
) -> List[Dict[str, Any]]:
    """Attach query provenance and contract slot targets to retrieved sources."""

    target_slots = _unique_strings(planned_query.target_slots)
    attached: List[Dict[str, Any]] = []
    for source in sources:
        payload = dict(source)
        if planned_query.query and not str(payload.get("query") or "").strip():
            payload["query"] = planned_query.query
        payload["target_slots"] = _merge_unique_strings(payload.get("target_slots"), target_slots)
        attached.append(payload)
    return attached


def _coerce_contract(contract: MissionContract | Mapping[str, Any]) -> MissionContract:
    if isinstance(contract, MissionContract):
        return contract
    return MissionContract.model_validate(dict(contract))


def _plan_entity_queries(
    contract: MissionContract,
    seen: set[str],
) -> List[PlannedQuery]:
    planned: List[PlannedQuery] = []
    slot_map = _entity_slot_map(contract)
    canonical_map = getattr(contract, "named_entities_canonical", None) or {}
    disambiguation_map = getattr(contract, "named_entities_disambiguation", None) or {}
    for entity in _prioritize_entity_queries(contract.named_entities):
        surfaces = list(
            _entity_query_surfaces(
                entity,
                contract=contract,
                canonical_map=canonical_map,
            )
        )
        # S54.3 — augment the canonical/primary entity surface with the LLM-supplied
        # disambiguating_context for ambiguous acronyms. We only mutate the FIRST
        # surface (the canonical/raw form) so concept-like expansions retain their
        # bounded re-phrasing.
        if surfaces:
            disambiguated = _disambiguate_entity_surface(
                entity,
                base_surface=surfaces[0],
                disambiguation_map=disambiguation_map,
            )
            if disambiguated is not None:
                surfaces[0] = disambiguated

        for query_text in surfaces:
            query = str(query_text).strip()[: refinement.MAX_QUERY_LENGTH]
            normalized = refinement._normalise_query(query)
            if not normalized or normalized in seen:
                continue
            planned.append(
                PlannedQuery(
                    query=query,
                    query_class="entity",
                    target_slots=slot_map.get(entity, []),
                )
            )
            seen.add(normalized)
    return planned


def _disambiguate_entity_surface(
    entity: str,
    *,
    base_surface: str,
    disambiguation_map: Mapping[str, str],
) -> str | None:
    """Return the disambiguated entity query, or ``None`` to leave it unchanged.

    S54.3 — fires only when:
    1. The contract carries a non-empty ``disambiguating_context`` for this entity
       (the LLM extractor in S54.2 is the source of truth — it sets this only for
       ambiguous acronyms whose retrieval would otherwise pull homonym pollution).
    2. AND the entity matches the ambiguous-acronym shape: short (≤5 chars),
       all-uppercase, with at least one alphabetic character. Belt-and-suspenders
       check so an LLM that mistakenly populates ``disambiguating_context`` for
       a hyphenated tool like ``axe-core`` cannot still mutate that query.

    When fires, append the disambiguating_context tokens to the canonical query
    surface and emit a structured log line for the query-telemetry surface.
    """

    entity_str = str(entity or "").strip()
    if not entity_str:
        return None
    context = str(disambiguation_map.get(entity_str) or "").strip()
    if not context:
        return None
    if not _is_ambiguous_acronym_shape(entity_str):
        return None

    base = str(base_surface or "").strip()
    if not base:
        return None
    if context.lower() in base.lower():
        return None  # already augmented (idempotency)

    disambiguated = f"{base} {context}".strip()
    logger.info(
        "entity_query_disambiguation: %s",
        {
            "entity": entity_str,
            "original_query": base,
            "disambiguated_query": disambiguated,
        },
    )
    return disambiguated


def _is_ambiguous_acronym_shape(entity: str) -> bool:
    """Return True for the shape that benefits from disambiguating context.

    Targets short all-caps tokens that frequently collide with unrelated
    homonyms in retrieval (CCS, OODS, SAPLMA, etc.). Hyphenated or mixed-case
    names (axe-core, t-SNE) are NOT ambiguous in this sense and stay unchanged.
    """

    text = (entity or "").strip()
    if not text or len(text) > 5:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    return text.upper() == text and not any(ch.islower() for ch in text)


def _plan_concept_queries(
    contract: MissionContract,
    *,
    gaps: Sequence[Mapping[str, Any]],
    mission_context: Mapping[str, Any] | None,
    objective_lookup: Sequence[str] | None,
    seen: set[str],
    max_queries: int,
) -> List[PlannedQuery]:
    planned: List[PlannedQuery] = []
    focus_hint = refinement._extract_focus_hint(dict(mission_context or {}))
    preserved_terms = _filter_preserved_terms_for_concept_class(
        refinement._extract_preserved_terms(
            dict(mission_context or {}),
            objective_lookup,
        )
    )
    slot_map = _objective_slot_map(contract)
    objective_map = _contract_objective_id_map(contract, objective_lookup)

    for gap in gaps:
        candidate = refinement._build_query_from_gap(dict(gap), focus_hint, objective_lookup)
        if not candidate:
            continue

        gap_objective_id = str(gap.get("objective_id") or "").strip()
        contract_objective_id = objective_map.get(gap_objective_id, gap_objective_id)
        objective_text = _contract_objective_text(contract, contract_objective_id)
        target_slots = slot_map.get(contract_objective_id, [])

        expanded_queries = _expand_distributive_objective_queries(
            contract,
            objective_text=objective_text,
            focus_hint=focus_hint,
        )
        for expanded in expanded_queries:
            sanitized = _sanitize_concept_query(expanded, contract)
            normalized = refinement._normalise_query(sanitized)
            if not normalized or normalized in seen:
                continue
            planned.append(
                PlannedQuery(
                    query=sanitized,
                    query_class="concept",
                    target_slots=target_slots,
                )
            )
            seen.add(normalized)
            if len(planned) >= max_queries:
                return planned

        candidate = refinement._append_preserved_terms(candidate, preserved_terms)
        candidate = _sanitize_concept_query(candidate, contract)
        normalized = refinement._normalise_query(candidate)
        if not normalized or normalized in seen:
            continue
        planned.append(
            PlannedQuery(
                query=candidate,
                query_class="concept",
                target_slots=target_slots,
            )
        )
        seen.add(normalized)
        if len(planned) >= max_queries:
            break

    return planned


def _sanitize_concept_query(query: str, contract: MissionContract) -> str:
    """S38.2 — strip success-criteria fragments and trailing connective tokens.

    Two narrow string-handling fixes co-located in one helper because both leak
    the same way (success-criteria text → preserved-term extractor → outbound
    concept-class query body):

    * DEFECT A: drop verbatim 4+ word substrings of any success criterion. This
      is the B35.9 parity rule applied to concept queries; entity and
      deliverable_gap classes already enforce it via their own construction
      paths.
    * DEFECT B: drop trailing tokens that are bare connective function words
      (the, of, and, for, is, …). These leak in via PROPER_NOUN_PATTERN
      matching a [Capitalized] [connective] pair (e.g. "Map the") that gets
      appended as a preserved term, leaving a grammatical fragment at the end.
    """

    cleaned = _strip_success_criteria_fragments(
        str(query or ""),
        getattr(contract, "success_criteria", ()) or (),
    )
    cleaned = _strip_trailing_connectives(cleaned)
    return cleaned.strip()


def _strip_trailing_connectives(query: str) -> str:
    tokens = str(query or "").split()
    while tokens and tokens[-1].lower() in _CONCEPT_QUERY_TRAILING_CONNECTIVES:
        tokens.pop()
    return " ".join(tokens)


def _filter_preserved_terms_for_concept_class(terms: Sequence[str]) -> List[str]:
    """S38.2 — drop preserved-term phrases that end in a connective function word.

    PROPER_NOUN_PATTERN admits 2-token fragments like "Map the" (Capitalized +
    "the"/"of"/"and"/...). Once appended into a concept query, these leave a
    grammatical fragment dangling. Filter the whole phrase here before append so
    we never need to reconstruct word boundaries downstream.
    """

    filtered: List[str] = []
    for term in terms or []:
        text = str(term or "").strip()
        if not text:
            continue
        tokens = text.split()
        if tokens[-1].lower() in _CONCEPT_QUERY_TRAILING_CONNECTIVES:
            continue
        filtered.append(text)
    return filtered


def _strip_success_criteria_fragments(
    query: str,
    success_criteria: Sequence[str],
) -> str:
    if not query or not success_criteria:
        return query
    cleaned = query
    for criterion in success_criteria:
        normalized_criterion = " ".join(str(criterion or "").lower().split())
        if not normalized_criterion:
            continue
        tokens = normalized_criterion.split()
        if len(tokens) < _CONCEPT_QUERY_SC_FUSION_MIN_WINDOW:
            continue
        # Walk longest-first so a maximal phrase strip happens before its substrings.
        for window in range(len(tokens), _CONCEPT_QUERY_SC_FUSION_MIN_WINDOW - 1, -1):
            for start in range(0, len(tokens) - window + 1):
                phrase = " ".join(tokens[start : start + window])
                if phrase in cleaned.lower():
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    cleaned = pattern.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _plan_deliverable_gap_queries(
    contract: MissionContract,
    *,
    mission_context: Mapping[str, Any] | None,
    reflection_scores: Mapping[str, Any],
    seen: set[str],
) -> List[PlannedQuery]:
    planned: List[PlannedQuery] = []
    satisfied_schema_ids = _resolve_satisfied_schema_ids(contract, reflection_scores)
    slot_map = _deliverable_slot_map(contract)
    topic_hint = (
        contract.objective
        or refinement._extract_focus_hint(dict(mission_context or {}))
        or contract.title
    )

    for schema in contract.deliverable_schemas:
        if schema.id in satisfied_schema_ids:
            continue
        query = str(topic_hint or schema.title or "").strip()[: refinement.MAX_QUERY_LENGTH]
        normalized = refinement._normalise_query(query)
        if not normalized or normalized in seen:
            continue
        planned.append(
            PlannedQuery(
                query=query,
                query_class="deliverable_gap",
                target_slots=slot_map.get(schema.id, []),
            )
        )
        seen.add(normalized)

    return planned


def _plan_diversified_queries(
    contract: MissionContract,
    *,
    mission_context: Mapping[str, Any] | None,
    objective_lookup: Sequence[str] | None,
    unfilled_slot_ids: Sequence[str] | None,
    loop_index: int,
    seen: set[str],
    stats: Dict[str, int],
) -> List[PlannedQuery]:
    """S39.1 — produce novel queries targeting gaps the default pass missed.

    Three strategies, each deduped against the shared ``seen`` set:

    * ``slot_targeted``: for each unfilled slot, synthesize a concrete query
      from the slot target (entity, objective_id, or schema_id) combined with
      the mission focus hint. This is the primary lever that prevents
      loops 3-5 from starving when gap-derived concept queries collide.
    * ``sub_objective_decomposition``: when an objective mentions 2+ contract
      entities (including Oxford-comma lists), emit per-entity sub-queries.
    * ``paraphrase_mutation``: append a perspective keyword (methodology,
      recent advances, benchmarks, …) rotated by ``loop_index`` so a query
      that already ran returns as a distinct novel variant.

    ``stats`` is mutated in place with the non-zero novel count per strategy.
    """

    planned: List[PlannedQuery] = []

    slot_queries = _plan_slot_targeted_queries(
        contract,
        unfilled_slot_ids=unfilled_slot_ids,
        mission_context=mission_context,
        objective_lookup=objective_lookup,
        seen=seen,
    )
    if slot_queries:
        stats[_STRATEGY_SLOT_TARGETED] = len(slot_queries)
        planned.extend(slot_queries)

    sub_objective_queries = _plan_sub_objective_queries(
        contract,
        mission_context=mission_context,
        objective_lookup=objective_lookup,
        seen=seen,
    )
    if sub_objective_queries:
        stats[_STRATEGY_SUB_OBJECTIVE] = len(sub_objective_queries)
        planned.extend(sub_objective_queries)

    paraphrase_queries = _plan_paraphrase_queries(
        contract,
        mission_context=mission_context,
        loop_index=loop_index,
        seen=seen,
    )
    if paraphrase_queries:
        stats[_STRATEGY_PARAPHRASE] = len(paraphrase_queries)
        planned.extend(paraphrase_queries)

    return planned


def _plan_slot_targeted_queries(
    contract: MissionContract,
    *,
    unfilled_slot_ids: Sequence[str] | None,
    mission_context: Mapping[str, Any] | None,
    objective_lookup: Sequence[str] | None,
    seen: set[str],
) -> List[PlannedQuery]:
    if not unfilled_slot_ids:
        return []

    focus_hint = refinement._extract_focus_hint(dict(mission_context or {}))
    if focus_hint and refinement._is_internal_label(focus_hint):
        focus_hint = ""
    canonical_map = getattr(contract, "named_entities_canonical", None) or {}
    objective_text_by_id = {
        objective.id: str(objective.text).strip() for objective in contract.objectives
    }
    schema_title_by_id = {
        schema.id: str(getattr(schema, "title", "") or "").strip()
        for schema in contract.deliverable_schemas
    }
    slot_by_id = {slot.id: slot for slot in contract.evidence_slots}

    planned: List[PlannedQuery] = []
    for slot_id in unfilled_slot_ids:
        slot = slot_by_id.get(str(slot_id).strip())
        if slot is None:
            continue
        target = slot.target or {}

        query_text = _query_for_slot_target(
            target,
            contract=contract,
            canonical_map=canonical_map,
            objective_text_by_id=objective_text_by_id,
            schema_title_by_id=schema_title_by_id,
            focus_hint=focus_hint,
            objective_lookup=objective_lookup,
        )
        if not query_text:
            continue

        sanitized = _sanitize_concept_query(query_text, contract)
        normalized = refinement._normalise_query(sanitized)
        if not normalized or normalized in seen:
            continue
        planned.append(
            PlannedQuery(
                query=sanitized[: refinement.MAX_QUERY_LENGTH],
                query_class=_slot_query_class(target),
                target_slots=[slot.id],
            )
        )
        seen.add(normalized)

    return planned


def _query_for_slot_target(
    target: Mapping[str, Any],
    *,
    contract: MissionContract,
    canonical_map: Mapping[str, str],
    objective_text_by_id: Mapping[str, str],
    schema_title_by_id: Mapping[str, str],
    focus_hint: str,
    objective_lookup: Sequence[str] | None,
) -> str:
    entity = str(target.get("entity") or "").strip()
    if entity:
        expansion = _primary_entity_query_surface(
            entity,
            contract=contract,
            canonical_map=canonical_map,
        )
        objective_id = str(target.get("objective_id") or "").strip()
        objective_text = objective_text_by_id.get(objective_id, "")
        objective_terms = refinement._objective_query_terms(objective_text)
        return refinement._compose_objective_query(
            focus_hint=focus_hint,
            subject_terms=[expansion],
            objective_terms=objective_terms,
        )

    schema_id = str(target.get("schema_id") or "").strip()
    if schema_id:
        schema_title = schema_title_by_id.get(schema_id, "")
        topic = focus_hint or str(contract.objective or "").strip() or str(contract.title or "")
        parts = [part for part in (topic, schema_title, "evidence") if part]
        return " ".join(parts).strip()

    objective_id = str(target.get("objective_id") or "").strip()
    if objective_id:
        objective_text = objective_text_by_id.get(objective_id, "")
        if not objective_text:
            return ""
        return refinement._compose_objective_query(
            focus_hint=focus_hint,
            subject_terms=refinement._extract_objective_subject_terms(objective_text),
            objective_terms=refinement._objective_query_terms(objective_text),
        )

    return ""


def _primary_entity_query_surface(
    entity: str,
    *,
    contract: MissionContract,
    canonical_map: Mapping[str, str],
) -> str:
    surfaces = _entity_query_surfaces(
        entity,
        contract=contract,
        canonical_map=canonical_map,
    )
    if len(surfaces) > 1:
        return surfaces[1]
    return surfaces[0] if surfaces else ""


def _entity_query_surfaces(
    entity: str,
    *,
    contract: MissionContract,
    canonical_map: Mapping[str, str],
) -> List[str]:
    base = str(canonical_map.get(entity) or entity).strip()
    if not base:
        return []

    surfaces: List[str] = [base]
    # S56.3: prefer the LLM-supplied query_surfaces. The extractor emits 2-4
    # retrieval-friendly variants per entity (head noun, acronym alone, well-
    # known concrete tool name) — appending them lets compound-phrase
    # entities like "automated ARIA validation" reach evidence the canonical
    # surface alone would miss, without growing regex-era decomposition rules
    # in the planner.
    llm_surfaces = getattr(contract, "named_entities_query_surfaces", None) or {}
    for surface in llm_surfaces.get(entity, []) or []:
        cleaned = str(surface or "").strip()
        if cleaned:
            surfaces.append(cleaned)
    return _unique_strings(surfaces)


def _slot_query_class(target: Mapping[str, Any]) -> QueryClass:
    if str(target.get("entity") or "").strip():
        return "entity"
    if str(target.get("schema_id") or "").strip():
        return "deliverable_gap"
    return "concept"


def _plan_sub_objective_queries(
    contract: MissionContract,
    *,
    mission_context: Mapping[str, Any] | None,
    objective_lookup: Sequence[str] | None,
    seen: set[str],
) -> List[PlannedQuery]:
    entities = [str(entity).strip() for entity in contract.named_entities if str(entity).strip()]
    if len(entities) < 2:
        return []

    canonical_map = getattr(contract, "named_entities_canonical", None) or {}
    focus_hint = refinement._extract_focus_hint(dict(mission_context or {}))
    if focus_hint and refinement._is_internal_label(focus_hint):
        focus_hint = ""

    entity_slot_map = _entity_slot_map(contract)

    planned: List[PlannedQuery] = []
    for objective in contract.objectives:
        objective_text = str(objective.text or "").strip()
        if not objective_text:
            continue
        mentioned = _entities_mentioned_in_text(objective_text, entities)
        if len(mentioned) < 2:
            continue

        objective_terms = refinement._objective_query_terms(objective_text)
        for entity in mentioned:
            expansion = str(canonical_map.get(entity) or entity).strip()
            query_text = refinement._compose_objective_query(
                focus_hint=focus_hint,
                subject_terms=[expansion],
                objective_terms=objective_terms,
            )
            if not query_text:
                continue
            sanitized = _sanitize_concept_query(query_text, contract)
            normalized = refinement._normalise_query(sanitized)
            if not normalized or normalized in seen:
                continue
            planned.append(
                PlannedQuery(
                    query=sanitized[: refinement.MAX_QUERY_LENGTH],
                    query_class="entity",
                    target_slots=entity_slot_map.get(entity, []),
                )
            )
            seen.add(normalized)
    return planned


def _entities_mentioned_in_text(text: str, entities: Sequence[str]) -> List[str]:
    lowered = text.lower()
    mentioned: List[str] = []
    seen_entities: set[str] = set()
    for entity in entities:
        if not entity or entity in seen_entities:
            continue
        if entity.lower() in lowered:
            mentioned.append(entity)
            seen_entities.add(entity)
    return mentioned


def _plan_paraphrase_queries(
    contract: MissionContract,
    *,
    mission_context: Mapping[str, Any] | None,
    loop_index: int,
    seen: set[str],
) -> List[PlannedQuery]:
    focus_hint = refinement._extract_focus_hint(dict(mission_context or {}))
    if focus_hint and refinement._is_internal_label(focus_hint):
        focus_hint = ""
    base_topic = (
        focus_hint
        or str(contract.objective or "").strip()
        or str(contract.title or "").strip()
    )
    if not base_topic:
        return []

    planned: List[PlannedQuery] = []
    perspective_count = len(_PARAPHRASE_PERSPECTIVES)
    # Rotate by loop so loop 2, 3, 4 each pick a distinct perspective first.
    offset = max(0, int(loop_index)) % perspective_count
    attempt_order = [
        _PARAPHRASE_PERSPECTIVES[(offset + i) % perspective_count]
        for i in range(perspective_count)
    ]

    for perspective in attempt_order:
        candidate = f"{base_topic} {perspective}".strip()
        sanitized = _sanitize_concept_query(candidate, contract)
        normalized = refinement._normalise_query(sanitized)
        if not normalized or normalized in seen:
            continue
        planned.append(
            PlannedQuery(
                query=sanitized[: refinement.MAX_QUERY_LENGTH],
                query_class="concept",
                target_slots=[],
            )
        )
        seen.add(normalized)
        # One paraphrase per loop is enough — additional perspectives rotate
        # into place on subsequent loops via ``offset``.
        break

    return planned


def _resolve_satisfied_schema_ids(
    contract: MissionContract,
    reflection_scores: Mapping[str, Any],
) -> set[str]:
    if reflection_scores.get("is_mission_complete"):
        return {schema.id for schema in contract.deliverable_schemas}

    satisfied: set[str] = set()
    raw_entries = (
        reflection_scores.get("deliverable_schema_scores")
        or reflection_scores.get("deliverable_schema_statuses")
        or []
    )
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            continue
        schema_id = str(
            entry.get("schema_id")
            or entry.get("id")
            or entry.get("deliverable_schema_id")
            or ""
        ).strip()
        if not schema_id:
            continue
        status = str(entry.get("status") or "").strip().lower()
        passed = entry.get("passed")
        score = entry.get("score")
        if status in {"satisfied", "complete", "passed"}:
            satisfied.add(schema_id)
            continue
        if passed is True:
            satisfied.add(schema_id)
            continue
        try:
            if score is not None and float(score) >= 0.8:
                satisfied.add(schema_id)
        except (TypeError, ValueError):
            continue
    return satisfied


def _contract_objective_id_map(
    contract: MissionContract,
    objective_lookup: Sequence[str] | None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    objectives = list(contract.objectives)
    required_objectives = [objective for objective in objectives if objective.required]

    lookup_items = [str(item).strip() for item in objective_lookup or [] if str(item).strip()]
    text_to_contract_id = {objective.text: objective.id for objective in objectives}
    if lookup_items:
        for index, objective_text in enumerate(lookup_items, start=1):
            contract_objective_id = text_to_contract_id.get(objective_text)
            if contract_objective_id:
                _register_objective_aliases(
                    mapping,
                    contract_objective_id,
                    objective_text=objective_text,
                    required_objective_index=index,
                )

    for index, objective in enumerate(required_objectives, start=1):
        _register_objective_aliases(
            mapping,
            objective.id,
            objective_text=objective.text,
            required_objective_index=index,
        )

    for objective in objectives:
        _register_objective_aliases(
            mapping,
            objective.id,
            objective_text=objective.text,
            required_objective_index=None,
        )
    return mapping


def _register_objective_aliases(
    mapping: Dict[str, str],
    objective_id: str,
    *,
    objective_text: str = "",
    required_objective_index: int | None,
) -> None:
    _register_objective_alias(mapping, objective_id, objective_id)
    _register_objective_alias(mapping, objective_text, objective_id)

    contract_index = _objective_id_index(objective_id)
    if contract_index is not None:
        _register_objective_alias(mapping, f"objective-{contract_index}", objective_id)
        _register_objective_alias(mapping, f"objective{contract_index}", objective_id)
        _register_objective_alias(mapping, str(contract_index), objective_id)
        if contract_index > 0:
            _register_objective_alias(mapping, f"{contract_index:02d}", objective_id)

    if required_objective_index is None:
        return

    _register_objective_alias(mapping, f"OBJ{required_objective_index:02d}", objective_id)
    _register_objective_alias(mapping, f"OBJ{required_objective_index}", objective_id)
    _register_objective_alias(mapping, f"OBJ-{required_objective_index}", objective_id)
    _register_objective_alias(mapping, str(required_objective_index), objective_id)
    _register_objective_alias(mapping, f"{required_objective_index:02d}", objective_id)
    if required_objective_index == 1:
        _register_objective_alias(mapping, "primary", objective_id)
    else:
        _register_objective_alias(
            mapping,
            f"sc_{required_objective_index - 2}",
            objective_id,
        )


def _register_objective_alias(
    mapping: Dict[str, str],
    alias: str,
    objective_id: str,
) -> None:
    text = str(alias).strip()
    if not text:
        return
    for variant in (text, text.lower(), text.upper()):
        mapping.setdefault(variant, objective_id)


def _objective_id_index(objective_id: str) -> int | None:
    digits = "".join(ch for ch in str(objective_id or "") if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _objective_slot_map(contract: MissionContract) -> Dict[str, List[str]]:
    slot_map: Dict[str, List[str]] = {}
    for slot in contract.evidence_slots:
        objective_id = str(slot.target.get("objective_id") or "").strip()
        if not objective_id:
            continue
        slot_map.setdefault(objective_id, []).append(slot.id)
    return slot_map


def _entity_slot_map(contract: MissionContract) -> Dict[str, List[str]]:
    slot_map: Dict[str, List[str]] = {}
    for slot in contract.evidence_slots:
        entity = str(slot.target.get("entity") or "").strip()
        if not entity:
            continue
        slot_map.setdefault(entity, []).append(slot.id)
    return slot_map


def _deliverable_slot_map(contract: MissionContract) -> Dict[str, List[str]]:
    slot_map: Dict[str, List[str]] = {}
    for slot in contract.evidence_slots:
        schema_id = str(slot.target.get("schema_id") or "").strip()
        if not schema_id:
            continue
        slot_map.setdefault(schema_id, []).append(slot.id)
    return slot_map


def _contract_objective_text(contract: MissionContract, objective_id: str) -> str:
    for objective in contract.objectives:
        if objective.id == objective_id:
            return str(objective.text).strip()
    return ""


def _expand_distributive_objective_queries(
    contract: MissionContract,
    *,
    objective_text: str,
    focus_hint: str,
) -> List[str]:
    cleaned_objective = str(objective_text or "").strip()
    if not cleaned_objective or not _DISTRIBUTIVE_OBJECTIVE_PATTERN.search(cleaned_objective):
        return []

    focus_entities = _focus_entities_for_distributive_queries(contract.named_entities)
    if len(focus_entities) < 2:
        return []

    queries: List[str] = []
    seen: set[str] = set()
    for entity in focus_entities:
        query = _build_entity_specific_query(
            entity=entity,
            objective_text=cleaned_objective,
            focus_hint=focus_hint,
        )
        normalized = refinement._normalise_query(query)
        if not normalized or normalized in seen:
            continue
        queries.append(query)
        seen.add(normalized)
    return queries


def _focus_entities_for_distributive_queries(entities: Sequence[str]) -> List[str]:
    prioritized = _prioritize_entity_queries(entities)
    if not prioritized:
        return []

    family_counts: Dict[str, int] = {}
    for entity in prioritized:
        family = _entity_family(entity)
        if not family:
            continue
        family_counts[family] = family_counts.get(family, 0) + 1

    focused = [
        entity
        for entity in prioritized
        if family_counts.get(_entity_family(entity), 0) > 1
    ]
    return focused or prioritized


def _build_entity_specific_query(
    *,
    entity: str,
    objective_text: str,
    focus_hint: str,
) -> str:
    query_focus_hint = focus_hint
    if focus_hint and "langgraph" in focus_hint.lower():
        query_focus_hint = "LangGraph"
    query = refinement._compose_objective_query(
        focus_hint=query_focus_hint,
        subject_terms=[entity],
        objective_terms=refinement._objective_query_terms(objective_text),
    )
    if query:
        return query[: refinement.MAX_QUERY_LENGTH]
    fallback = f"{query_focus_hint} {entity}".strip()
    return fallback[: refinement.MAX_QUERY_LENGTH]


def _camel_case_parts(value: str) -> List[str]:
    return re.findall(r"[A-Z]+(?=[A-Z][a-z0-9]|\b)|[A-Z]?[a-z0-9]+", value)


def _entity_family(entity: str) -> str:
    parts = _camel_case_parts(str(entity or "").strip())
    if len(parts) < 2:
        return ""
    return parts[-1].lower()


def _prioritize_entity_queries(entities: Sequence[str]) -> List[str]:
    family_counts: Dict[str, int] = {}
    for entity in entities:
        family = _entity_family(entity)
        if not family:
            continue
        family_counts[family] = family_counts.get(family, 0) + 1

    if not any(count > 1 for count in family_counts.values()):
        return [str(entity).strip() for entity in entities if str(entity).strip()]

    ranked = list(enumerate(entities))
    ranked.sort(
        key=lambda item: (
            0 if family_counts.get(_entity_family(item[1]), 0) > 1 else 1,
            item[0],
        )
    )
    return [str(entity).strip() for _, entity in ranked if str(entity).strip()]


def _merge_unique_strings(
    existing: Any,
    new_values: Sequence[str],
) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for value in _iter_string_values(existing):
        if value in seen:
            continue
        merged.append(value)
        seen.add(value)
    for value in _unique_strings(new_values):
        if value in seen:
            continue
        merged.append(value)
        seen.add(value)
    return merged


def _unique_strings(values: Sequence[str] | None) -> List[str]:
    unique: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        unique.append(text)
        seen.add(text)
    return unique


def _iter_string_values(values: Any) -> Iterable[str]:
    if isinstance(values, str):
        text = values.strip()
        if text:
            yield text
        return
    if not isinstance(values, Sequence):
        return
    for value in values:
        text = str(value).strip()
        if text:
            yield text


__all__ = [
    "ContractAwareQueryPlannerError",
    "PlannedQuery",
    "QueryClass",
    "attach_query_provenance",
    "coerce_planned_query",
    "plan_queries",
    "resolve_contract_objective_id",
]
