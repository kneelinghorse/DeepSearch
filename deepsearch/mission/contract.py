"""Mission contract compilation and persistence helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, cast

from pydantic import BaseModel, Field, model_validator

from deepsearch.agent.deliverable_schemas import (
    OutputSchema,
    extract_deliverable_schemas,
    is_deliverable_attribute_list,
    is_deliverable_attribute_phrase,
)
from deepsearch.data_paths import data_path
from deepsearch.domain_policy import normalize_domain
from deepsearch.mission.compiler_provenance import (
    CANONICAL_CONTRACT_COMPILED_AT,
    CANONICAL_CONTRACT_ORIGIN,
    CONFIGURED_ENRICHMENT_MODE,
    CONTRACT_COMPILER_REVISION,
    CONTRACT_SCHEMA_VERSION,
    NO_ENRICHMENT_MODE,
    CompilerEnrichmentMode,
)
from deepsearch.mission.scope import AuthoredScope, compile_authored_scope
from deepsearch.mission.title_utils import normalize_mission_title

CONTRACT_DIR_ENV = "DEEPSEARCH_CONTRACT_DIR"
logger = logging.getLogger(__name__)

_AUTHOR_YEAR_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+et al\.)?\s+\d{4}\b")
_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")
_CAMEL_CASE_PATTERN = re.compile(r"\b(?:[A-Z]{2,}[a-z0-9]*|[A-Z][a-z0-9]+)(?:[A-Z][a-z0-9]+)+\b")
_ENUMERATED_ENTITY_INTRO_PATTERN = re.compile(
    r"\b(?:compare|assess|evaluate|examine|analyze|contrast|benchmark|survey"
    r"|map|chart|catalog|document|review|investigate|study|explore)\b"
    r"(?P<rest>[^.!?;\n]+)",
    re.IGNORECASE,
)

# S40.2 — scoped Oxford-comma pattern for Map-style objectives where the
# verb is separated from the entity list by a scoping prepositional phrase
# (e.g. "Map the evidence base **for** SAPLMA, CCS, and semantic entropy").
# Captures only the text AFTER the scoping preposition so the prefix
# ("the evidence base for") does not pollute the enumeration's first item.
# The ``rest`` group intentionally mirrors the direct-enumeration pattern
# so both can share the same split / candidate / normalize pipeline.
_ENUMERATED_ENTITY_SCOPED_INTRO_PATTERN = re.compile(
    r"\b(?:map|chart|catalog|document|review|investigate|study|explore"
    r"|survey|assess|evaluate|examine|analyze|contrast|benchmark|compare)\b"
    r"[^.!?;\n]*?"
    r"\b(?:for|of|about|on|across|regarding|concerning)\b"
    r"(?P<rest>[^.!?;\n]+)",
    re.IGNORECASE,
)

# S40.2 — negation tokens that, when present inside the enumeration's
# ``rest`` capture, mean the list is not a list-of-entities (e.g. "A, B, and
# C, except D"). Abandoning extraction is safer than emitting a partial
# set — the authored ``required_entities`` field remains the canonical
# source for constraint-bearing missions.
_ENUMERATED_ENTITY_INLINE_NEGATION = re.compile(
    r"\b(?:except|excluding|not|without|minus|but\s+not|do\s+not\s+include"
    r"|must\s+not\s+include|should\s+not\s+include)\b",
    re.IGNORECASE,
)

# S40.2 — stopword prefixes that indicate an enumerated part is a common
# noun phrase (e.g. "the evidence base"), not a named entity. Rejected at
# normalize time so the scoped and direct patterns can both run without
# the direct pattern surfacing the "for"-prefixed first item as a
# spurious entity.
_ENUMERATED_ENTITY_STOPWORD_PREFIX_TOKENS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "some",
        "any",
        "all",
    }
)
_ENUMERATED_ENTITY_LEADING_CONJUNCTION_PATTERN = re.compile(
    r"^(?:and|or|nor|vs\.?)\s+",
    re.IGNORECASE,
)
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_GENERIC_ACRONYM_STOPWORDS = frozenset(
    {
        "API",
        "APIS",
        "SDK",
        "CLI",
        "URL",
        "HTTP",
        "HTTPS",
    }
)
_KNOWN_ACRONYM_REGISTRY: Dict[str, tuple[Dict[str, Sequence[str] | str], ...]] = {
    "CCS": (
        {
            "canonical": "Contrast-Consistent Search",
            "aliases": (
                "CCS",
                "Contrast-Consistent Search",
                "contrast-consistent search",
                "contrast consistent search",
                "contrast consistent",
            ),
            "rejection_patterns": (
                "carbon capture",
                "carbon sequestration",
                "ccs.neu.edu",
                "college of computer",
                "disambiguation",
                "may refer to",
                "circuit city",
                "cardcaptor",
                "cross-currency",
                "cross currency swap",
                "skateboard",
                "british music",
            ),
        },
    ),
    "SAPLMA": (
        {
            "canonical": "SAPLMA",
            "aliases": ("SAPLMA",),
            "rejection_patterns": (),
        },
    ),
}
_SCOPE_GUARDRAIL_PATTERNS = (
    re.compile(
        r"\b(?:do|must|should)\s+not\s+(?:include|mention|recommend|promote|cover|discuss)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bexclude(?:\b|\s)", re.IGNORECASE),
    re.compile(r"\b(?:off-contract|on-contract|out of scope|outside scope)\b", re.IGNORECASE),
    re.compile(r"\bkeep source selection\b", re.IGNORECASE),
    re.compile(r"\bstay focused\b", re.IGNORECASE),
    re.compile(r"\bfocused on the required\b", re.IGNORECASE),
    re.compile(r"\bunless explicitly required\b", re.IGNORECASE),
)
_STRUCTURAL_SUCCESS_CRITERION_PATTERNS = (
    re.compile(r"\bexecutive summary\b", re.IGNORECASE),
    re.compile(r"\b(?:comparison\s+)?table\b", re.IGNORECASE),
    re.compile(r"\bmatrix\b", re.IGNORECASE),
    re.compile(r"\bmarkdown\b", re.IGNORECASE),
    re.compile(r"\bcolumns?\b", re.IGNORECASE),
    re.compile(r"\bsection\b", re.IGNORECASE),
    re.compile(r"\bheading\b", re.IGNORECASE),
)
_CONSTRAINT_ACCEPTANCE_GATE_PATTERN = re.compile(
    r"^\s*(?:no|not|must not|should not|avoid|exclude|without|do not)\b",
    re.IGNORECASE,
)
_NEGATION_ENTITY_PATTERNS = (
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bwithout\b", re.IGNORECASE),
    re.compile(r"\bexclude\b", re.IGNORECASE),
    re.compile(r"\bexcluding\b", re.IGNORECASE),
    re.compile(r"\bdo not include\b", re.IGNORECASE),
    re.compile(r"\bmust not include\b", re.IGNORECASE),
    re.compile(r"\bshould not include\b", re.IGNORECASE),
    re.compile(r"\bavoid\b", re.IGNORECASE),
    re.compile(r"\bminus\b", re.IGNORECASE),
    re.compile(r"\bexcept\b", re.IGNORECASE),
)
_ENUMERATED_ENTITY_TRUNCATION_WORDS = frozenset(
    {
        "in",
        "for",
        "with",
        "on",
        "at",
        "across",
        "among",
    }
)
_ENUMERATED_ENTITY_CONNECTIVE_WORDS = frozenset(
    {
        "as",
        "with",
        "for",
        "in",
        "of",
        "via",
        "using",
        "based",
        "driven",
    }
)
_ENUMERATED_ENTITY_TRAILING_DESCRIPTORS = frozenset(
    {
        "architecture",
        "architectures",
        "backend",
        "backends",
        "bundler",
        "bundlers",
        "framework",
        "frameworks",
        "patterns",
        "pattern",
        "retriever",
        "retrievers",
    }
)
_ENUMERATED_ENTITY_REJECTION_TRAILING_TOKENS = frozenset(
    {
        "evidence",
        "bases",
        "approach",
        "approaches",
        "method",
        "methods",
        "technique",
        "techniques",
        "family",
        "families",
        "signal",
        "signals",
        "generation",
        "pipeline",
        "pipelines",
    }
)
_DEPTH_BUDGET_KEYS = (
    "recursion_limit",
    "search_results_per_query",
    "max_results_per_query",
    "max_tokens_per_loop",
    "max_total_sources",
    "min_total_sources",
    "min_unique_concepts",
    "min_total_findings",
    "min_unique_domains_gate",
)
_ACCEPTANCE_RELEVANCE_THRESHOLDS: Dict[str, float] = {
    "deep": 0.35,
    "alpha": 0.45,
}
_TIER_DEFAULT_RELEVANCE_THRESHOLDS: Dict[str, float] = {
    "baseline": 0.30,
    "deep": 0.40,
    "alpha": 0.45,
}


class ObjectiveContract(BaseModel):
    """Normalized objective entry in the compiled mission contract."""

    id: str
    text: str
    source: str
    required: bool = True


class DeliverableSchemaContract(BaseModel):
    """Structural deliverable requirement in the compiled contract."""

    id: str
    kind: str
    title: str
    format: str
    source_text: str
    min_items: Optional[int] = None
    minimum_citation_count: Optional[int] = None
    columns: List[str] = Field(default_factory=list)
    allowed_values: List[str] = Field(default_factory=list)
    required_sections: List[str] = Field(default_factory=list)
    entity_extraction_required: bool = False
    entity_types: List[str] = Field(default_factory=list)
    row_identifier_column: Optional[str] = None
    recommendation_required: bool = False
    evidence_gaps_section_required: bool = False
    corpus_reference_required: bool = False
    claim_to_evidence_mapping_required: bool = False
    limitations_section_required: bool = False
    notes: List[str] = Field(default_factory=list)


class EvidenceSlot(BaseModel):
    """Discrete evidence need that the pipeline should satisfy."""

    id: str
    kind: str
    description: str
    source: str
    target: Dict[str, Any] = Field(default_factory=dict)
    required: bool = True


class AcceptanceCheck(BaseModel):
    """Deterministic or reviewer-assisted check derived from the contract."""

    id: str
    kind: str
    description: str
    target: Dict[str, Any] = Field(default_factory=dict)
    required: bool = True


class ExecutionBudget(BaseModel):
    """Execution limits compiled into the contract."""

    research_depth: str
    deliverable_format: str
    max_loops: int
    min_loops: int
    search_results_per_query: Optional[int] = None
    sources_per_loop: Optional[int] = None
    max_tokens_per_loop: Optional[int] = None
    depth_config: Dict[str, Any] = Field(default_factory=dict)
    budget_fields: Dict[str, Any] = Field(default_factory=dict)


class MissionContract(BaseModel):
    """Executable contract compiled from a mission definition."""

    contract_version: str = CONTRACT_SCHEMA_VERSION
    contract_schema_version: str = CONTRACT_SCHEMA_VERSION
    compiler_revision: int = 0
    contract_id: str
    compiled_at: str
    origin: str
    mission_id: str
    project_id: Optional[str] = None
    title: str
    objective: str
    background: str = ""
    focus: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    excluded_domains: List[str] = Field(default_factory=list)
    reference_titles: List[str] = Field(default_factory=list)
    authored_scope: AuthoredScope = Field(default_factory=AuthoredScope)
    authority_seed_domains: List[str] = Field(default_factory=list)
    required_entities: List[str] = Field(default_factory=list)
    excluded_entities: List[str] = Field(default_factory=list)
    named_entities: List[str] = Field(default_factory=list)
    named_entities_canonical: Dict[str, str] = Field(default_factory=dict)
    # S54.2 — populated when the LLM extractor returns at least one entity
    # with a non-empty ``disambiguating_context`` field. S54.3's query
    # planner reads this map to defuse acronym homonym pollution upstream
    # of retrieval.
    named_entities_disambiguation: Dict[str, str] = Field(default_factory=dict)
    # S56.3 — populated when the LLM extractor returns retrieval-friendly
    # query surfaces alongside the canonical entity name. The query planner's
    # ``_entity_query_surfaces`` consumes this map to emit additional
    # retrieval queries per entity (e.g., 'automated ARIA validation' →
    # ['ARIA validation', 'ARIA', 'axe-core']) without growing regex-era
    # decomposition rules.
    named_entities_query_surfaces: Dict[str, List[str]] = Field(default_factory=dict)
    entity_extraction_decisions: List[Dict[str, str]] = Field(default_factory=list)
    coverage_thresholds: Dict[str, float] = Field(default_factory=dict)
    validation_thresholds: Dict[str, float] = Field(default_factory=dict)
    research_depth: str = "baseline"
    objectives: List[ObjectiveContract] = Field(default_factory=list)
    deliverable_schemas: List[DeliverableSchemaContract] = Field(default_factory=list)
    evidence_slots: List[EvidenceSlot] = Field(default_factory=list)
    acceptance_checks: List[AcceptanceCheck] = Field(default_factory=list)
    execution_budget: ExecutionBudget
    # S49.4 — opt-in flag enabling validate_objective_coverage_section
    # in run_validators. Set True by compile_contract_from_state when
    # the compiled contract carries at least one required objective.
    # Frozen JSON contracts loaded from older fixtures don't carry this
    # field and skip the new validator (preserving prior validation
    # surfaces byte-identical).
    objective_coverage_section_required: bool = False

    @model_validator(mode="before")
    @classmethod
    def _preserve_legacy_provenance(cls, value: Any) -> Any:
        """Keep old artifacts honest instead of attributing them to the new compiler."""

        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        payload.setdefault(
            "contract_schema_version",
            payload.get("contract_version", CONTRACT_SCHEMA_VERSION),
        )
        return payload


@dataclass(slots=True)
class PreparedExecutionContract:
    """Persisted contract bundle attached to an execution state."""

    contract: MissionContract
    path: Path


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    """Runtime retrieval guidance derived from a compiled execution budget."""

    per_loop: Optional[int] = None
    per_lane: Optional[int] = None
    per_query: Optional[int] = None
    max_tokens_per_loop: Optional[int] = None


def compile_contract_from_state(
    state: Mapping[str, Any],
    *,
    origin: str,
    enrichment_mode: CompilerEnrichmentMode = CONFIGURED_ENRICHMENT_MODE,
    compiled_at: str | None = None,
) -> MissionContract:
    """Compile a mission contract from an ``AgentState``-like payload.

    The default keeps the configured model-backed entity enrichment path.
    ``enrichment_mode="none"`` is the deterministic structural path used by
    cross-service compiler parity tests and never constructs or invokes a model.
    """

    if enrichment_mode not in {CONFIGURED_ENRICHMENT_MODE, NO_ENRICHMENT_MODE}:
        raise ValueError(f"Unsupported compiler enrichment mode: {enrichment_mode}")

    mission_context = _coerce_mapping(state.get("mission_context"))
    authored_scope = compile_authored_scope(mission_context)
    mission_id = _coerce_text(state.get("mission_id"), fallback="unknown-mission")
    title = normalize_mission_title(
        _coerce_text(mission_context.get("title"), fallback=mission_id),
        mission_id,
    )
    objective = _coerce_text(
        mission_context.get("objective"),
        fallback=_first_text(
            _normalize_string_list(state.get("mission_objectives")),
            default=title,
        ),
    )
    background = _coerce_optional_text(mission_context.get("background")) or ""
    focus = _coerce_optional_text(mission_context.get("focus")) or ""
    success_criteria = _normalize_string_list(mission_context.get("success_criteria"))
    if not success_criteria:
        success_criteria = _normalize_string_list(state.get("mission_objectives"))
        success_criteria = [item for item in success_criteria if item != objective]
    deliverables = _normalize_string_list(mission_context.get("deliverables"))
    constraints = _normalize_string_list(mission_context.get("constraints"))
    excluded_domains = _normalize_domain_list(mission_context.get("excluded_domains"))
    required_entities = _normalize_string_list(mission_context.get("required_entities"))
    excluded_entities = _normalize_string_list(mission_context.get("excluded_entities"))
    (
        research_success_criteria,
        auxiliary_success_criteria,
        acceptance_gate_criteria,
    ) = _partition_success_criteria(success_criteria)
    constraints = _merge_unique_strings(
        constraints,
        auxiliary_success_criteria,
        acceptance_gate_criteria,
    )
    references = mission_context.get("references")
    reference_titles = _merge_unique_strings(
        _normalize_string_list(mission_context.get("reference_titles")),
        _extract_reference_titles(references),
    )
    authority_seed_domains = _merge_unique_strings(
        _normalize_domain_list(
            mission_context.get("authority_seed_domains")
            or mission_context.get("primary_domains")
            or mission_context.get("authority_domains")
        ),
        _extract_reference_domains(references),
    )
    coverage_thresholds = _extract_threshold_overrides(
        mission_context.get("coverage_thresholds") or state.get("coverage_thresholds")
    )
    validation_thresholds = _extract_threshold_overrides(
        mission_context.get("validation_thresholds") or state.get("validation_thresholds")
    )

    objectives = _build_objective_contracts(objective, research_success_criteria)
    authored_output_schema = mission_context.get("expected_output_schema")
    if authored_output_schema is not None:
        # An explicit empty mapping/list clears both legacy context and
        # prose-derived schema inference. Only SQL NULL/absence permits the
        # older inference path.
        output_schemas = _extract_authored_output_schemas(authored_output_schema)
    else:
        output_schemas = extract_deliverable_schemas(
            deliverables,
            [objective, *success_criteria],
        )
    deliverable_schemas = _build_deliverable_schema_contracts(output_schemas)
    missing_registry_acronyms: set[str] = set()
    extraction_inputs = [
        title,
        objective,
        background,
        focus,
        *research_success_criteria,
        *deliverables,
        *reference_titles,
    ]
    compiler_model_diagnostics: List[Dict[str, Any]] = []
    if enrichment_mode == NO_ENRICHMENT_MODE:
        if required_entities:
            extracted_named_entities = []
            entity_extraction_path = "skipped_declared_authoritative"
        else:
            extracted_named_entities = _extract_named_entities(
                extraction_inputs,
                missing_registry_acronyms=missing_registry_acronyms,
            )
            entity_extraction_path = "regex_no_enrichment"
        named_entities_disambiguation: Dict[str, str] = {}
        named_entities_query_surfaces: Dict[str, List[str]] = {}
        llm_extraction_records: List[Dict[str, str]] = []
    else:
        (
            extracted_named_entities,
            named_entities_disambiguation,
            named_entities_query_surfaces,
            entity_extraction_path,
            llm_extraction_records,
        ) = _extract_entities_llm_first(
            extraction_inputs=extraction_inputs,
            objective=objective,
            deliverables=deliverables,
            required_entities=required_entities,
            missing_registry_acronyms=missing_registry_acronyms,
            diagnostics=compiler_model_diagnostics,
        )

    # S60.1 — when the author declared ``required_entities``, the prose
    # extractor short-circuits (S59.2) and leaves disambiguation /
    # query_surfaces empty. Single-word brand entities with strong homonyms
    # (Railway → railroad industry; Fly.io → ambiguous) get retrieval
    # pollution because the entity-lane query is the bare entity name with
    # no qualifier. Run a focused LLM disambiguation pass that populates the
    # S54.3 + S56.3 surfaces from the declared list. Trust-the-LLM-first
    # principle: no author-override hints field, the existing ``background``
    # field is the author-side surface (planning-pass amendment §1).
    if required_entities and enrichment_mode == CONFIGURED_ENRICHMENT_MODE:
        (
            named_entities_disambiguation,
            named_entities_query_surfaces,
            entity_extraction_path,
        ) = _disambiguate_declared_entities(
            declared_entities=required_entities,
            objective=objective,
            background="\n\n".join(value for value in (background, focus) if value),
            deliverables=deliverables,
            diagnostics=compiler_model_diagnostics,
        )
    named_entities, entity_extraction_decisions = _classify_named_entity_candidates(
        required_entities,
        extracted_named_entities,
    )
    _log_missing_acronym_registry_entries(missing_registry_acronyms)
    named_entities, entity_exclusion_decisions = _exclude_named_entities_with_decisions(
        named_entities,
        excluded_entities,
    )
    entity_extraction_decisions.extend(entity_exclusion_decisions)
    if llm_extraction_records:
        entity_extraction_decisions.extend(llm_extraction_records)
    if entity_extraction_path:
        entity_extraction_decisions.append(
            {"entity": "", "decision": "extraction_path", "reason": entity_extraction_path}
        )
    entity_extraction_decisions.extend(
        {
            "entity": "", "decision": (
                "provider_usage" if failure.get("kind") == "provider_usage" else "provider_failure"
            ),
            "reason": json.dumps(failure, sort_keys=True),
        }
        for failure in compiler_model_diagnostics
    )
    named_entities_canonical = _build_named_entities_canonical(named_entities)
    # Drop disambiguation entries for entities that did not survive
    # classification (declared/excluded/dropped) — keep the contract's
    # disambiguation map aligned with the final entity slot list.
    retained_keys = {entity.lower() for entity in named_entities}
    if named_entities_disambiguation:
        named_entities_disambiguation = {
            key: value
            for key, value in named_entities_disambiguation.items()
            if key.lower() in retained_keys
        }
    # S56.3: mirror the cleanup for the LLM-emitted query_surfaces map so the
    # planner only sees surfaces for entities still in the slot list.
    if named_entities_query_surfaces:
        named_entities_query_surfaces = {
            key: list(value)
            for key, value in named_entities_query_surfaces.items()
            if key.lower() in retained_keys
        }
    evidence_slots = _build_evidence_slots(named_entities, objectives, deliverable_schemas)
    acceptance_checks = _build_acceptance_checks(
        named_entities,
        objectives,
        deliverable_schemas,
        acceptance_gate_criteria=acceptance_gate_criteria,
    )
    if (authored_scope.restriction != "unrestricted" or authored_scope.max_sources is not None
            or authored_scope.min_words is not None or authored_scope.max_words is not None):
        acceptance_checks.append(AcceptanceCheck(
            id="check-authored-scope", kind="authored_scope",
            description="Persisted words and all consulted evidence must honor authored scope.",
            target=authored_scope.model_dump(mode="json"), required=True,
        ))
    execution_budget = _build_execution_budget(state)

    # S49.4 — enable the per-objective rendered-coverage validator
    # whenever the compiled contract carries at least one required
    # objective. Older frozen contracts (loaded as JSON) won't have this
    # flag set and remain unaffected.
    has_required_objective = any(item.required for item in objectives)

    contract_payload = {
        "contract_version": CONTRACT_SCHEMA_VERSION,
        "origin": origin,
        "mission_id": mission_id,
        "project_id": _coerce_optional_text(state.get("project_id")),
        "title": title,
        "objective": objective,
        "background": background,
        "focus": focus,
        "success_criteria": success_criteria,
        "deliverables": deliverables,
        "constraints": constraints,
        "excluded_domains": excluded_domains,
        "reference_titles": reference_titles,
        "authored_scope": authored_scope.model_dump(mode="json"),
        "authority_seed_domains": authority_seed_domains,
        "required_entities": required_entities,
        "excluded_entities": excluded_entities,
        "named_entities": named_entities,
        "named_entities_canonical": named_entities_canonical,
        "named_entities_disambiguation": named_entities_disambiguation,
        "named_entities_query_surfaces": named_entities_query_surfaces,
        "entity_extraction_decisions": entity_extraction_decisions,
        "coverage_thresholds": coverage_thresholds,
        "validation_thresholds": validation_thresholds,
        "research_depth": execution_budget.research_depth,
        "objectives": [item.model_dump(mode="json") for item in objectives],
        "deliverable_schemas": [item.model_dump(mode="json") for item in deliverable_schemas],
        "evidence_slots": [item.model_dump(mode="json") for item in evidence_slots],
        "acceptance_checks": [item.model_dump(mode="json") for item in acceptance_checks],
        "execution_budget": execution_budget.model_dump(mode="json"),
        "objective_coverage_section_required": has_required_objective,
    }
    contract_id = _contract_id_for_payload(contract_payload)

    return MissionContract.model_validate(
        {
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "compiler_revision": CONTRACT_COMPILER_REVISION,
            "contract_id": contract_id,
            "compiled_at": _resolve_compiled_at(compiled_at),
            **contract_payload,
        }
    )


def compile_canonical_contract_from_state(state: Mapping[str, Any]) -> MissionContract:
    """Compile stable canonical output without provider-backed enrichment."""

    return compile_contract_from_state(
        state,
        origin=CANONICAL_CONTRACT_ORIGIN,
        enrichment_mode=NO_ENRICHMENT_MODE,
        compiled_at=CANONICAL_CONTRACT_COMPILED_AT,
    )


def resolve_research_objectives(
    contract: MissionContract | Mapping[str, Any] | None,
    *,
    fallback: Sequence[str] | None = None,
    include_optional: bool = False,
) -> List[str]:
    """Return research objectives derived from the compiled contract when present."""

    resolved: List[str] = []
    seen: set[str] = set()

    for payload in _objective_payloads(contract):
        if isinstance(payload, ObjectiveContract):
            text = payload.text
            required = payload.required
        elif isinstance(payload, Mapping):
            text = str(payload.get("text") or "").strip()
            required = bool(payload.get("required", True))
        else:
            continue
        if not text or (not include_optional and not required):
            continue
        key = text.lower()
        if key in seen:
            continue
        resolved.append(text)
        seen.add(key)

    if resolved:
        return resolved
    return _unique_strings(_normalize_string_list(fallback))


def persist_contract(
    contract: MissionContract,
    *,
    output_dir: Path | str | None = None,
) -> Path:
    """Persist a compiled mission contract to disk and return the artifact path."""

    base_dir = _resolve_contract_dir(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_mission_id = _safe_filename(contract.mission_id)
    path = base_dir / f"mission-{safe_mission_id}-{contract.contract_id}.json"
    path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
    return path


def prepare_execution_contract(
    state: MutableMapping[str, Any],
    *,
    origin: str,
    output_dir: Path | str | None = None,
) -> PreparedExecutionContract:
    """Compile, persist, and attach a mission contract to a mutable execution state."""

    from deepsearch.agent.execution_control import raise_if_execution_cancelled

    raise_if_execution_cancelled()
    contract = compile_contract_from_state(state, origin=origin)
    raise_if_execution_cancelled()
    path = persist_contract(contract, output_dir=output_dir)
    raise_if_execution_cancelled()

    state["mission_contract"] = contract.model_dump(mode="json")
    state["mission_contract_id"] = contract.contract_id
    state["mission_contract_path"] = str(path)
    state["depth_config"] = dict(contract.execution_budget.depth_config)

    telemetry = state.get("telemetry")
    if not isinstance(telemetry, MutableMapping):
        telemetry = {}
    telemetry["contract_id"] = contract.contract_id
    telemetry["contract_path"] = str(path)
    telemetry["contract_version"] = contract.contract_version
    telemetry["contract_schema_version"] = contract.contract_schema_version
    telemetry["compiler_revision"] = contract.compiler_revision
    telemetry["contract_origin"] = origin
    state["telemetry"] = cast(Dict[str, Any], dict(telemetry))

    return PreparedExecutionContract(contract=contract, path=path)


def build_retrieval_budget(
    contract: MissionContract | Mapping[str, Any],
    *,
    query_count: int,
    lane_count: int,
) -> RetrievalBudget:
    """Build retrieval guidance from ``contract.execution_budget``."""

    del query_count, lane_count
    compiled_contract = (
        contract
        if isinstance(contract, MissionContract)
        else MissionContract.model_validate(contract)
    )
    execution_budget = compiled_contract.execution_budget

    max_tokens_per_loop = _coerce_optional_int(
        execution_budget.max_tokens_per_loop
        or execution_budget.budget_fields.get("max_tokens_per_loop")
        or execution_budget.depth_config.get("max_tokens_per_loop")
    )

    return RetrievalBudget(
        max_tokens_per_loop=max_tokens_per_loop,
    )


def _build_objective_contracts(
    objective: str,
    success_criteria: Sequence[str],
) -> List[ObjectiveContract]:
    # When explicit success criteria are present, treat the top-level objective
    # as an umbrella statement rather than a separate hard gate.
    objectives = [
        ObjectiveContract(
            id="objective-0",
            text=objective,
            source="objective",
            required=not bool(success_criteria),
        )
    ]
    for index, criterion in enumerate(success_criteria, start=1):
        objectives.append(
            ObjectiveContract(
                id=f"objective-{index}",
                text=criterion,
                source="success_criteria",
                required=True,
            )
        )
    return objectives


def _partition_success_criteria(
    success_criteria: Sequence[str],
) -> tuple[List[str], List[str], List[str]]:
    research_criteria: List[str] = []
    auxiliary_criteria: List[str] = []
    acceptance_gate_criteria: List[str] = []
    for criterion in success_criteria:
        text = str(criterion or "").strip()
        if not text:
            continue
        if _is_constraint_acceptance_gate(text):
            acceptance_gate_criteria.append(text)
            continue
        if _is_scope_guardrail_criterion(text) or _is_structural_success_criterion(text):
            auxiliary_criteria.append(text)
            continue
        research_criteria.append(text)
    return research_criteria, auxiliary_criteria, acceptance_gate_criteria


def _is_constraint_acceptance_gate(text: str) -> bool:
    return bool(_CONSTRAINT_ACCEPTANCE_GATE_PATTERN.search(text))


def _is_scope_guardrail_criterion(text: str) -> bool:
    lowered = text.lower()
    if "redis" in lowered and "memorysaver" in lowered and "do not" not in lowered:
        return False
    return any(pattern.search(text) for pattern in _SCOPE_GUARDRAIL_PATTERNS)


def _is_structural_success_criterion(text: str) -> bool:
    if extract_deliverable_schemas([], [text]):
        return True
    return any(pattern.search(text) for pattern in _STRUCTURAL_SUCCESS_CRITERION_PATTERNS)


def _build_deliverable_schema_contracts(
    schemas: Sequence[OutputSchema],
) -> List[DeliverableSchemaContract]:
    compiled: List[DeliverableSchemaContract] = []
    for index, schema in enumerate(schemas, start=1):
        compiled.append(
            DeliverableSchemaContract(
                id=f"deliverable-schema-{index}",
                kind=schema.kind,
                title=schema.title,
                format=schema.format,
                source_text=schema.source_text,
                min_items=schema.min_items,
                minimum_citation_count=schema.minimum_citation_count,
                columns=list(schema.columns),
                allowed_values=list(schema.allowed_values),
                required_sections=list(schema.required_sections),
                entity_extraction_required=schema.entity_extraction_required,
                entity_types=list(schema.entity_types),
                row_identifier_column=schema.row_identifier_column,
                recommendation_required=schema.recommendation_required,
                evidence_gaps_section_required=schema.evidence_gaps_section_required,
                corpus_reference_required=schema.corpus_reference_required,
                claim_to_evidence_mapping_required=schema.claim_to_evidence_mapping_required,
                limitations_section_required=schema.limitations_section_required,
                notes=list(schema.notes),
            )
        )
    return compiled


def _extract_authored_output_schemas(raw_value: Any) -> List[OutputSchema]:
    if raw_value is None:
        return []
    if isinstance(raw_value, Mapping):
        if not raw_value:
            return []
        items: Sequence[Any] = [raw_value]
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        items = list(raw_value)
    else:
        raise ValueError(
            "mission_context.expected_output_schema must be a mapping or sequence of mappings"
        )
    if not items:
        return []

    schemas: List[OutputSchema] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(
                "mission_context.expected_output_schema items must be mappings " f"(item {index})"
            )
        schemas.append(_build_authored_output_schema(item, index=index))
    return schemas


def _build_authored_output_schema(payload: Mapping[str, Any], *, index: int) -> OutputSchema:
    kind = _required_authored_schema_text(payload.get("kind"), field="kind", index=index)
    title = _required_authored_schema_text(payload.get("title"), field="title", index=index)
    format_name = _required_authored_schema_text(
        payload.get("format"),
        field="format",
        index=index,
    )
    source_text = _coerce_optional_text(payload.get("source_text")) or (
        f"authored expected_output_schema[{index}]: {title}"
    )

    return OutputSchema(
        kind=kind,
        title=title,
        format=format_name,
        source_text=source_text,
        min_items=_optional_authored_schema_int(
            payload.get("min_items"),
            field="min_items",
            index=index,
        ),
        minimum_citation_count=_optional_authored_schema_int(
            payload.get("minimum_citation_count"),
            field="minimum_citation_count",
            index=index,
        ),
        columns=_authored_schema_string_tuple(payload.get("columns"), field="columns", index=index),
        allowed_values=_authored_schema_string_tuple(
            payload.get("allowed_values"),
            field="allowed_values",
            index=index,
        ),
        required_sections=_authored_schema_string_tuple(
            payload.get("required_sections"),
            field="required_sections",
            index=index,
        ),
        entity_extraction_required=_authored_schema_bool(
            payload.get("entity_extraction_required"),
            field="entity_extraction_required",
            index=index,
        ),
        entity_types=_authored_schema_string_tuple(
            payload.get("entity_types"),
            field="entity_types",
            index=index,
        ),
        row_identifier_column=_optional_authored_schema_text(
            payload.get("row_identifier_column"),
            field="row_identifier_column",
            index=index,
        ),
        recommendation_required=_authored_schema_bool(
            payload.get("recommendation_required"),
            field="recommendation_required",
            index=index,
        ),
        evidence_gaps_section_required=_authored_schema_bool(
            payload.get("evidence_gaps_section_required"),
            field="evidence_gaps_section_required",
            index=index,
        ),
        corpus_reference_required=_authored_schema_bool(
            payload.get("corpus_reference_required"),
            field="corpus_reference_required",
            index=index,
        ),
        claim_to_evidence_mapping_required=_authored_schema_bool(
            payload.get("claim_to_evidence_mapping_required"),
            field="claim_to_evidence_mapping_required",
            index=index,
        ),
        limitations_section_required=_authored_schema_bool(
            payload.get("limitations_section_required"),
            field="limitations_section_required",
            index=index,
        ),
        notes=_authored_schema_string_tuple(payload.get("notes"), field="notes", index=index),
    )


def _classify_named_entity_candidates(
    declared_entities: Sequence[str],
    extracted_entities: Sequence[str],
) -> tuple[List[str], List[Dict[str, str]]]:
    """Classify declared and extracted entity candidates for contract audit.

    Declared ``required_entities`` are authoritative and bypass the
    deliverable-attribute and fragment filters. Extracted candidates remain
    heuristic and can be rejected before they become evidence slots.
    """

    decisions: List[Dict[str, str]] = []
    named_entities: List[str] = []
    seen: set[str] = set()
    extracted_keys = {
        str(entity or "").strip().lower()
        for entity in extracted_entities
        if str(entity or "").strip()
    }
    declared_keys: set[str] = set()
    decision_keys: set[tuple[str, str]] = set()

    for entity in declared_entities:
        cleaned = str(entity or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        declared_keys.add(key)
        if key in seen:
            decisions.append(
                {
                    "entity": cleaned,
                    "decision": "dropped",
                    "reason": "duplicate_declared",
                }
            )
            continue
        named_entities.append(cleaned)
        seen.add(key)
        if key in extracted_keys:
            continue
        decisions.append({"entity": cleaned, "decision": "declared"})
        decision_keys.add((key, "declared"))

    extracted_after_attribute_filter: List[str] = []
    extracted_seen: set[str] = set()
    for entity in extracted_entities:
        cleaned = str(entity or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen or key in extracted_seen:
            if key in declared_keys and (key, "extracted") not in decision_keys:
                decisions.append({"entity": cleaned, "decision": "extracted"})
                decision_keys.add((key, "extracted"))
                continue
            decisions.append(
                {
                    "entity": cleaned,
                    "decision": "dropped",
                    "reason": "duplicate",
                }
            )
            continue
        if is_deliverable_attribute_phrase(cleaned):
            decisions.append(
                {
                    "entity": cleaned,
                    "decision": "rejected_attribute_phrase",
                }
            )
            continue
        extracted_after_attribute_filter.append(cleaned)
        extracted_seen.add(key)

    comparison_entities = [*named_entities, *extracted_after_attribute_filter]
    for entity in extracted_after_attribute_filter:
        fragment_of = _fragment_container_entity(entity, comparison_entities)
        if fragment_of:
            decisions.append(
                {
                    "entity": entity,
                    "decision": "rejected_fragment",
                    "reason": f"fragment_of:{fragment_of}",
                }
            )
            continue
        named_entities.append(entity)
        seen.add(entity.lower())
        decisions.append({"entity": entity, "decision": "extracted"})
        decision_keys.add((entity.lower(), "extracted"))

    return named_entities, decisions


def _fragment_container_entity(entity: str, candidates: Sequence[str]) -> str:
    entity_str = str(entity or "").strip()
    if not entity_str:
        return ""
    for candidate in candidates:
        candidate_str = str(candidate or "").strip()
        if not candidate_str or candidate_str.lower() == entity_str.lower():
            continue
        if _is_whole_entity_fragment(entity_str, candidate_str):
            return candidate_str
    return ""


def _is_whole_entity_fragment(entity: str, container: str) -> bool:
    entity_str = str(entity or "").strip()
    container_str = str(container or "").strip()
    if not entity_str or not container_str:
        return False
    if entity_str.lower() == container_str.lower():
        return False
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(entity_str) + r"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(container_str):
        if re.fullmatch(r"[A-Z][A-Z0-9]{1,}", entity_str):
            before = container_str[match.start() - 1] if match.start() > 0 else ""
            after = container_str[match.end()] if match.end() < len(container_str) else ""
            if before == "-" or after == "-":
                return True
            continue
        return True
    return False


def _exclude_named_entities_with_decisions(
    named_entities: Sequence[str],
    excluded_entities: Sequence[str],
) -> tuple[List[str], List[Dict[str, str]]]:
    excluded = {
        str(entity or "").strip().lower()
        for entity in excluded_entities
        if str(entity or "").strip()
    }
    if not excluded:
        return list(named_entities), []

    retained: List[str] = []
    decisions: List[Dict[str, str]] = []
    for entity in named_entities:
        if str(entity or "").strip().lower() in excluded:
            decisions.append(
                {
                    "entity": str(entity or "").strip(),
                    "decision": "dropped",
                    "reason": "excluded_entity",
                }
            )
            continue
        retained.append(entity)
    return retained, decisions


def _objective_ids_referencing_entity(
    entity: str,
    objectives: Sequence[ObjectiveContract],
) -> List[str]:
    """Return required-objective ids whose text names ``entity``.

    Why: S52.3 / D-S51-C — the packet router needs per-objective entity
    bindings to reserve K source slots per objective (where K = entities
    bound to that objective). Whole-token, case-insensitive matching avoids
    substring false positives — ``LangChain`` must not bind a ``LangGraph``
    objective even though both share the ``Lang`` prefix.
    """

    entity_str = str(entity or "").strip()
    if not entity_str:
        return []
    pattern = re.compile(r"\b" + re.escape(entity_str) + r"\b", flags=re.IGNORECASE)
    matched: List[str] = []
    seen: set[str] = set()
    for objective in objectives:
        if not objective.required:
            continue
        objective_id = str(getattr(objective, "id", "") or "").strip()
        objective_text = str(getattr(objective, "text", "") or "")
        if not objective_id or not objective_text or objective_id in seen:
            continue
        if pattern.search(objective_text):
            matched.append(objective_id)
            seen.add(objective_id)
    return matched


def _build_evidence_slots(
    named_entities: Sequence[str],
    objectives: Sequence[ObjectiveContract],
    deliverable_schemas: Sequence[DeliverableSchemaContract],
) -> List[EvidenceSlot]:
    slots: List[EvidenceSlot] = []
    for objective in objectives:
        if not objective.required:
            continue
        slots.append(
            EvidenceSlot(
                id=f"slot-{objective.id}",
                kind="objective_coverage",
                description=f"Gather evidence that directly addresses: {objective.text}",
                source=objective.source,
                target={"objective_id": objective.id},
            )
        )
    for entity in named_entities:
        target = _build_entity_target(entity)
        entity_name = str(target.get("entity_name") or entity).strip()
        objective_ids = _objective_ids_referencing_entity(entity, objectives)
        if objective_ids:
            target["objective_ids"] = objective_ids
        slots.append(
            EvidenceSlot(
                id=f"slot-entity-{_slug(entity)}",
                kind="named_entity",
                description=f"Cover named entity '{entity_name}' with source-grounded evidence.",
                source="named_entities",
                target=target,
            )
        )
    for schema in deliverable_schemas:
        target: Dict[str, Any] = {"schema_id": schema.id, "format": schema.format}
        _extend_schema_target(target, schema)
        slots.append(
            EvidenceSlot(
                id=f"slot-{schema.id}",
                kind="deliverable_schema",
                description=f"Satisfy structural deliverable '{schema.title}'.",
                source="deliverables",
                target=target,
            )
        )
    return slots


def _build_acceptance_checks(
    named_entities: Sequence[str],
    objectives: Sequence[ObjectiveContract],
    deliverable_schemas: Sequence[DeliverableSchemaContract],
    *,
    acceptance_gate_criteria: Sequence[str] = (),
) -> List[AcceptanceCheck]:
    checks: List[AcceptanceCheck] = []
    for objective in objectives:
        if not objective.required:
            continue
        checks.append(
            AcceptanceCheck(
                id=f"check-{objective.id}",
                kind="objective_coverage",
                description=f"Objective must be substantively addressed: {objective.text}",
                target={"objective_id": objective.id},
            )
        )
    for entity in named_entities:
        checks.append(
            AcceptanceCheck(
                id=f"check-entity-{_slug(entity)}",
                kind="named_entity_coverage",
                description=(
                    f"Named entity '{entity}' must appear in grounded evidence or synthesis."
                ),
                target={"entity": entity},
            )
        )
    for index, criterion in enumerate(acceptance_gate_criteria, start=1):
        checks.append(
            AcceptanceCheck(
                id=f"check-acceptance-gate-{index}",
                kind="acceptance_gate",
                description=f"Rendered output must satisfy: {criterion}",
                target={"criterion": criterion, "evaluation_stage": "post_synthesis"},
            )
        )
    for schema in deliverable_schemas:
        target: Dict[str, Any] = {"schema_id": schema.id, "format": schema.format}
        _extend_schema_target(target, schema)
        checks.append(
            AcceptanceCheck(
                id=f"check-{schema.id}",
                kind="structural_output",
                description=f"Output must satisfy the '{schema.title}' structural contract.",
                target=target,
            )
        )
    return checks


def _extend_schema_target(
    target: Dict[str, Any],
    schema: DeliverableSchemaContract,
) -> None:
    if schema.min_items is not None:
        target["min_items"] = schema.min_items
    if schema.minimum_citation_count is not None:
        target["minimum_citation_count"] = schema.minimum_citation_count
    if schema.columns:
        target["columns"] = list(schema.columns)
    if schema.allowed_values:
        target["allowed_values"] = list(schema.allowed_values)
    if schema.required_sections:
        target["required_sections"] = list(schema.required_sections)
    if schema.entity_extraction_required:
        target["entity_extraction_required"] = True
    if schema.entity_types:
        target["entity_types"] = list(schema.entity_types)
    if schema.row_identifier_column:
        target["row_identifier_column"] = schema.row_identifier_column
    if schema.recommendation_required:
        target["recommendation_required"] = True
    if schema.evidence_gaps_section_required:
        target["evidence_gaps_section_required"] = True
    if schema.corpus_reference_required:
        target["corpus_reference_required"] = True
    if schema.claim_to_evidence_mapping_required:
        target["claim_to_evidence_mapping_required"] = True
    if schema.limitations_section_required:
        target["limitations_section_required"] = True


def _build_execution_budget(state: Mapping[str, Any]) -> ExecutionBudget:
    depth_config = dict(_coerce_mapping(state.get("depth_config")))
    research_depth = _coerce_text(state.get("research_depth"), fallback="baseline")
    mission_id = _coerce_text(state.get("mission_id"), fallback="unknown-mission")
    _apply_acceptance_relevance_threshold_defaults(
        depth_config,
        mission_id=mission_id,
        research_depth=research_depth,
    )
    budget_fields = {
        key: depth_config[key]
        for key in _DEPTH_BUDGET_KEYS
        if key in depth_config and depth_config[key] is not None
    }
    if "max_results_per_query" in budget_fields and "search_results_per_query" not in budget_fields:
        budget_fields["search_results_per_query"] = budget_fields["max_results_per_query"]

    search_results_per_query = _coerce_int(
        depth_config.get("max_results_per_query"),
        default=5,
    )
    max_tokens_per_loop = _coerce_optional_int(depth_config.get("max_tokens_per_loop"))
    if max_tokens_per_loop is None:
        max_tokens_per_loop = _default_max_tokens_per_loop(research_depth)

    return ExecutionBudget(
        research_depth=research_depth,
        deliverable_format=_coerce_text(state.get("deliverable_format"), fallback="markdown"),
        max_loops=_coerce_int(state.get("max_loops"), default=3),
        min_loops=_coerce_int(state.get("min_loops"), default=0),
        search_results_per_query=search_results_per_query,
        sources_per_loop=None,
        max_tokens_per_loop=max_tokens_per_loop,
        depth_config=depth_config,
        budget_fields=budget_fields,
    )


def _apply_acceptance_relevance_threshold_defaults(
    depth_config: Dict[str, Any],
    *,
    mission_id: str,
    research_depth: str,
) -> None:
    normalized_depth = str(research_depth or "").strip().lower()
    acceptance_default = _ACCEPTANCE_RELEVANCE_THRESHOLDS.get(normalized_depth)
    current_threshold = _coerce_optional_float(depth_config.get("relevance_threshold"))
    tier_default = _TIER_DEFAULT_RELEVANCE_THRESHOLDS.get(normalized_depth)

    if acceptance_default is None or not _identifier_looks_acceptance_scoped(mission_id):
        return

    if current_threshold is None or current_threshold == tier_default:
        depth_config["relevance_threshold"] = acceptance_default
        logger.info(
            "Applied acceptance relevance_threshold %.2f for mission %s (%s)",
            acceptance_default,
            mission_id,
            normalized_depth,
        )
        return

    logger.info(
        "Preserving authored relevance_threshold %.2f for acceptance mission %s (%s)",
        current_threshold,
        mission_id,
        normalized_depth,
    )


def _identifier_looks_acceptance_scoped(value: Any) -> bool:
    return str(value or "").strip().upper().startswith("ACC-")


def is_acceptance_contract_payload(payload: Mapping[str, Any] | None) -> bool:
    """Return True when a contract payload belongs to an acceptance mission."""

    if not isinstance(payload, Mapping):
        return False
    return _identifier_looks_acceptance_scoped(payload.get("contract_id")) or (
        _identifier_looks_acceptance_scoped(payload.get("mission_id"))
    )


def _record_compiler_failure(
    diagnostics: List[Dict[str, Any]] | None, exc: BaseException, settings: Any = None,
) -> None:
    from deepsearch.llm.runtime import request_failure

    if diagnostics is not None:
        provider = getattr(settings, "llm_backend", "")
        model = (getattr(settings, f"{provider}_model", "")
                 if provider in {"deepseek", "openai"} else "")
        diagnostics.append(request_failure(exc, model=model, stage="compiler"))


def _extract_entities_llm_first(
    *,
    extraction_inputs: Sequence[str],
    objective: str,
    deliverables: Sequence[str],
    required_entities: Sequence[str],
    missing_registry_acronyms: set[str],
    diagnostics: List[Dict[str, Any]] | None = None,
) -> tuple[List[str], Dict[str, str], Dict[str, List[str]], str, List[Dict[str, str]]]:
    """Run LLM-first entity extraction with regex fallback (S54.2 + S56.3).

    Returns a 5-tuple
    ``(entity_names, disambiguation, query_surfaces, path, llm_decisions)``:

    - ``entity_names`` — entity strings to feed
      ``_classify_named_entity_candidates`` (replaces the regex extractor's
      output on the LLM happy path; identical to it on the fallback path).
    - ``disambiguation`` — mapping entity → disambiguating_context populated
      only when the LLM extractor sets non-empty context for an entity.
      Empty dict on the regex fallback path.
    - ``query_surfaces`` — S56.3. Mapping entity → list of retrieval-friendly
      variants the LLM emitted alongside the canonical name. Empty dict on
      the regex fallback path.
    - ``path`` — telemetry tag: ``"llm_primary"`` (LLM call succeeded),
      ``"regex_fallback"`` (LLM construction or call failed), or
      ``"skipped_declared_authoritative"`` (S59.2 — author declared
      ``required_entities`` so the prose-extraction path is bypassed).
    - ``llm_decisions`` — extraction-decision rows for entities the LLM
      explicitly rejected (e.g., participial fragments). Each row carries
      ``decision="rejected_llm"`` and a ``reason`` from the model's
      ``rejection_reason`` field. Empty list on the regex fallback path.
    """

    # S59.2 — when the author explicitly declares ``required_entities``, the
    # declared list is the canonical anchor. Running the LLM/regex extractor
    # on the prose pollutes the matrix candidate_set with prose-derived
    # entities (S58.3 OODS-FIGMA-HOST-01: 10 prose entities competed with
    # 5 declared platforms; matrix shipped 10 em-dash rows). Trust-the-LLM-
    # first principle: when the author declared, stop second-guessing.
    if any(str(e or "").strip() for e in required_entities):
        return [], {}, {}, "skipped_declared_authoritative", []

    from deepsearch.mission.entity_extractor_llm import (
        LLMEntityExtractor,
        run_llm_entity_extraction_sync,
    )

    try:
        from deepsearch.config import DeepSearchSettings
    except ImportError:  # pragma: no cover - defensive
        regex_entities = _extract_named_entities(
            extraction_inputs,
            missing_registry_acronyms=missing_registry_acronyms,
        )
        return regex_entities, {}, {}, "regex_fallback", []

    settings = None
    try:
        settings = DeepSearchSettings.load()
        structured_extractor = settings.build_structured_extractor()
    except Exception as exc:
        _record_compiler_failure(diagnostics, exc, settings)
        logger.warning("Compiler model construction failed; using deterministic fallback")
        regex_entities = _extract_named_entities(
            extraction_inputs,
            missing_registry_acronyms=missing_registry_acronyms,
        )
        return regex_entities, {}, {}, "regex_fallback", []

    extractor = LLMEntityExtractor(structured_extractor=structured_extractor)
    objective_text = str(objective or "").strip()
    objective_inputs = [objective_text] if objective_text else []
    outcome = run_llm_entity_extraction_sync(
        extractor,
        objectives=objective_inputs or list(extraction_inputs),
        deliverables=list(deliverables),
        required_entities=list(required_entities),
    )

    if diagnostics is not None:
        diagnostics.extend(getattr(structured_extractor, "diagnostics", []))
    if outcome is None:
        regex_entities = _extract_named_entities(
            extraction_inputs,
            missing_registry_acronyms=missing_registry_acronyms,
        )
        return regex_entities, {}, {}, "regex_fallback", []

    # Surface explicit rejections in entity_extraction_decisions so the
    # audit row makes the extractor's classification visible. Filter empty
    # rejection_reason to a stable placeholder.
    rejection_rows: List[Dict[str, str]] = []
    for entry in outcome.rejected:
        rejection_rows.append(
            {
                "entity": str(entry.name or "").strip(),
                "decision": "rejected_llm",
                "reason": (entry.rejection_reason or "").strip() or "llm_rejected",
            }
        )

    return (
        outcome.entity_names,
        dict(outcome.disambiguation),
        {entity: list(surfaces) for entity, surfaces in outcome.query_surfaces.items()},
        outcome.path,
        rejection_rows,
    )


def _disambiguate_declared_entities(
    *,
    declared_entities: Sequence[str],
    objective: str,
    background: str,
    deliverables: Sequence[str],
    diagnostics: List[Dict[str, Any]] | None = None,
) -> tuple[Dict[str, str], Dict[str, List[str]], str]:
    """S60.1 — focused LLM disambiguation pass for declared entities.

    Returns ``(disambiguation, query_surfaces, path)``:
    - ``disambiguation`` — entity → disambiguating-context phrase (≤120 chars).
    - ``query_surfaces`` — entity → list of retrieval-friendly query variants.
    - ``path`` — telemetry tag: ``"llm_declared"`` (LLM call succeeded),
      ``"regex_fallback"`` (LLM build/call failed), or
      ``"skipped_no_declared"`` (declared list empty).

    The declared list is authoritative — the LLM does NOT add or remove
    entities. Output is filtered to declared entities (case-insensitive).
    """

    if not any(str(e or "").strip() for e in declared_entities):
        return {}, {}, "skipped_no_declared"

    from deepsearch.mission.entity_extractor_llm import (
        DeclaredEntityDisambiguator,
        run_declared_entity_disambiguation_sync,
    )

    try:
        from deepsearch.config import DeepSearchSettings
    except ImportError:  # pragma: no cover - defensive
        return {}, {}, "regex_fallback"

    settings = None
    try:
        settings = DeepSearchSettings.load()
        structured_extractor = settings.build_structured_extractor()
    except Exception as exc:
        _record_compiler_failure(diagnostics, exc, settings)
        logger.warning("Compiler model construction failed; skipping disambiguation")
        return {}, {}, "regex_fallback"

    disambiguator = DeclaredEntityDisambiguator(structured_extractor=structured_extractor)
    outcome = run_declared_entity_disambiguation_sync(
        disambiguator,
        declared_entities=declared_entities,
        objective=objective,
        background=background,
        deliverables=deliverables,
    )

    if diagnostics is not None:
        diagnostics.extend(getattr(structured_extractor, "diagnostics", []))
    if outcome is None:
        return {}, {}, "regex_fallback"

    return (
        dict(outcome.disambiguation),
        {entity: list(surfaces) for entity, surfaces in outcome.query_surfaces.items()},
        outcome.path,
    )


def _extract_named_entities(
    texts: Sequence[str],
    *,
    missing_registry_acronyms: Optional[set[str]] = None,
) -> List[str]:
    entities: List[str] = []
    seen: set[str] = set()
    for text in texts:
        cleaned = str(text or "").strip()
        if not cleaned:
            continue
        for match in _AUTHOR_YEAR_PATTERN.finditer(cleaned):
            if _is_negated_entity_match(cleaned, match.start()):
                continue
            normalized = match.group(0).strip()
            key = normalized.lower()
            if key not in seen:
                entities.append(normalized)
                seen.add(key)
        for match in _ACRONYM_PATTERN.finditer(cleaned):
            if _is_negated_entity_match(cleaned, match.start()):
                continue
            normalized = match.group(0).strip()
            if normalized in {"MUST", "SHOULD"} or normalized in _GENERIC_ACRONYM_STOPWORDS:
                continue
            if missing_registry_acronyms is not None and normalized not in _KNOWN_ACRONYM_REGISTRY:
                missing_registry_acronyms.add(normalized)
            key = normalized.lower()
            if key not in seen:
                entities.append(normalized)
                seen.add(key)
        for match in _CAMEL_CASE_PATTERN.finditer(cleaned):
            if _is_negated_entity_match(cleaned, match.start()):
                continue
            normalized = match.group(0).strip()
            key = normalized.lower()
            if key not in seen:
                entities.append(normalized)
                seen.add(key)
        for entity in _extract_enumerated_entities(cleaned):
            key = entity.lower()
            if key not in seen:
                entities.append(entity)
                seen.add(key)
    return entities


def _is_negated_entity_match(text: str, start_index: int) -> bool:
    window = text[max(0, start_index - 96) : start_index]
    if not window:
        return False
    fragment = re.split(r"[.!?;\n]", window)[-1]
    normalized_fragment = " ".join(fragment.strip().split())
    if not normalized_fragment:
        return False
    return any(pattern.search(normalized_fragment) for pattern in _NEGATION_ENTITY_PATTERNS)


def _extract_enumerated_entities(text: str) -> List[str]:
    lowered = text.lower()
    if "column" in lowered or "table" in lowered:
        return []

    entities: List[str] = []
    for pattern in (
        _ENUMERATED_ENTITY_SCOPED_INTRO_PATTERN,
        _ENUMERATED_ENTITY_INTRO_PATTERN,
    ):
        for match in pattern.finditer(text):
            rest = match.group("rest")
            # S40.2 — bail out on any negation token inside the enumeration;
            # the authored ``required_entities`` field is the right surface
            # for constrained missions rather than partial extraction.
            if _ENUMERATED_ENTITY_INLINE_NEGATION.search(rest):
                continue
            attribute_list = is_deliverable_attribute_list(rest)
            parts = [
                item.strip(" .`'\"()")
                for item in re.split(r",|\band\b", rest)
                if item.strip(" .`'\"()")
            ]
            if len(parts) < 2:
                continue
            for part in parts:
                for candidate in _enumerated_entity_candidates(part):
                    if attribute_list or is_deliverable_attribute_phrase(candidate):
                        continue
                    normalized = _normalize_enumerated_entity(candidate)
                    if normalized:
                        entities.extend(normalized)
    return _unique_strings(entities)


def _enumerated_entity_candidates(part: str) -> List[str]:
    cleaned = str(part or "").strip(" .`'\"()")
    if not cleaned:
        return []
    candidates = [cleaned]
    tokens = [token for token in cleaned.split() if token]
    for index, token in enumerate(tokens):
        if index > 0 and _normalized_entity_token(token) in _ENUMERATED_ENTITY_CONNECTIVE_WORDS:
            prefix = " ".join(tokens[:index]).strip()
            if prefix:
                candidates.append(prefix)
            break

    # S40.2 — when a REJECTION-trailing token appears mid-enumeration (not
    # just at the tail), surface the prefix as a candidate. Example:
    # "semantic entropy approaches used in agent evaluation" has "approaches"
    # at index 2; the scoped intro pattern gives us the enumeration cleanly
    # but the trailing qualifier "approaches used in agent evaluation" still
    # rides along on the last list item. Truncating before the first
    # rejection token recovers "semantic entropy" as a valid candidate.
    for index, token in enumerate(tokens):
        if (
            index > 0
            and _normalized_entity_token(token) in _ENUMERATED_ENTITY_REJECTION_TRAILING_TOKENS
        ):
            prefix = " ".join(tokens[:index]).strip()
            if prefix:
                candidates.append(prefix)
            break

    trimmed_tokens = list(tokens)
    while (
        len(trimmed_tokens) > 1
        and _normalized_entity_token(trimmed_tokens[-1])
        in _ENUMERATED_ENTITY_REJECTION_TRAILING_TOKENS
    ):
        trimmed_tokens = trimmed_tokens[:-1]
        prefix = " ".join(trimmed_tokens).strip()
        if prefix:
            candidates.append(prefix)
    return _unique_strings(candidates)


def _normalize_enumerated_entity(part: str) -> Optional[List[str]]:
    cleaned = str(part or "").strip(" .`'\"()")
    cleaned = _strip_leading_entity_conjunction(cleaned)
    if not cleaned:
        return None
    if is_deliverable_attribute_phrase(cleaned):
        return None

    if "/" in cleaned:
        entities: List[str] = []
        for component in cleaned.split("/"):
            normalized = _normalize_enumerated_entity(component)
            if normalized:
                entities.extend(normalized)
        unique = _unique_strings(entities)
        return unique or None

    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return None

    for index, token in enumerate(tokens):
        if index > 0 and _normalized_entity_token(token) in _ENUMERATED_ENTITY_TRUNCATION_WORDS:
            tokens = tokens[:index]
            break

    while (
        len(tokens) > 1
        and _normalized_entity_token(tokens[-1]) in _ENUMERATED_ENTITY_TRAILING_DESCRIPTORS
    ):
        tokens = tokens[:-1]

    if not tokens:
        return None
    # S40.2 — reject enumerated parts that begin with a determiner or other
    # stopword prefix. These are common-noun phrases (e.g. "the evidence
    # base", "these frameworks") that the scoped intro pattern's "for"
    # prefix tends to surface; leaving them in ``named_entities`` would
    # compile spurious ``slot-entity-*`` slots that can never be filled.
    if _normalized_entity_token(tokens[0]) in _ENUMERATED_ENTITY_STOPWORD_PREFIX_TOKENS:
        return None
    if len(tokens) > 4:
        return None
    if any(
        index > 0 and _normalized_entity_token(token) in _ENUMERATED_ENTITY_CONNECTIVE_WORDS
        for index, token in enumerate(tokens)
    ):
        return None
    # S40.2 — reject entities that carry an authoring-descriptor token
    # anywhere in the token sequence (previously only the trailing token
    # was checked). A token from REJECTION_TRAILING like "approaches" in
    # mid-position signals the enumerated item is still dragging trailing
    # qualifier text (e.g. "semantic entropy approaches used"), and the
    # mid-position truncation candidate will surface the clean entity —
    # so rejecting the un-truncated variant is safe.
    if any(
        _normalized_entity_token(token) in _ENUMERATED_ENTITY_REJECTION_TRAILING_TOKENS
        for token in tokens
    ):
        return None

    normalized = " ".join(token.strip(" .`'\"()") for token in tokens if token.strip(" .`'\"()"))
    if not normalized:
        return None
    return [normalized]


def _strip_leading_entity_conjunction(text: str) -> str:
    cleaned = str(text or "").strip()
    return _ENUMERATED_ENTITY_LEADING_CONJUNCTION_PATTERN.sub("", cleaned, count=1).strip()


def _normalized_entity_token(token: str) -> str:
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", str(token or "").lower())


def _build_entity_target(entity: str) -> Dict[str, Any]:
    target: Dict[str, Any] = {
        "entity": entity,
        "entity_name": entity,
    }
    registry_entry = _known_acronym_entry(entity)
    if not registry_entry:
        return target

    canonical = str(registry_entry.get("canonical") or entity).strip() or entity
    aliases = _unique_strings(
        [
            canonical,
            entity,
            *(
                registry_entry.get("aliases")
                if isinstance(registry_entry.get("aliases"), Sequence)
                and not isinstance(registry_entry.get("aliases"), (str, bytes))
                else [registry_entry.get("aliases")]
            ),
        ]
    )
    rejection_patterns = _unique_strings(
        registry_entry.get("rejection_patterns")
        if isinstance(registry_entry.get("rejection_patterns"), Sequence)
        and not isinstance(registry_entry.get("rejection_patterns"), (str, bytes))
        else [registry_entry.get("rejection_patterns")]
    )
    target["entity_name"] = canonical
    if aliases:
        target["aliases"] = aliases
    if rejection_patterns:
        target["rejection_patterns"] = rejection_patterns
    return target


def _known_acronym_entry(entity: str) -> Optional[Dict[str, Sequence[str] | str]]:
    entries = _KNOWN_ACRONYM_REGISTRY.get(str(entity or "").strip().upper()) or ()
    if not entries:
        return None
    return dict(entries[0])


def _build_named_entities_canonical(named_entities: Sequence[str]) -> Dict[str, str]:
    canonical_map: Dict[str, str] = {}
    for entity in named_entities:
        raw = str(entity or "").strip()
        if not raw:
            continue
        registry_entry = _known_acronym_entry(raw)
        if not registry_entry:
            continue
        canonical = str(registry_entry.get("canonical") or "").strip()
        if not canonical:
            continue
        canonical_map[raw] = canonical
    return canonical_map


def _log_missing_acronym_registry_entries(acronyms: Sequence[str]) -> None:
    missing = sorted(
        {str(acronym or "").strip().upper() for acronym in acronyms if str(acronym or "").strip()}
    )
    if not missing:
        return
    logger.info(
        "Missing known-expansion registry entries for acronym(s): %s",
        ", ".join(missing),
    )


def _required_authored_schema_text(
    raw_value: Any,
    *,
    field: str,
    index: int,
) -> str:
    text = _optional_authored_schema_text(raw_value, field=field, index=index)
    if text is None:
        raise ValueError(f"mission_context.expected_output_schema[{index}].{field} is required")
    return text


def _optional_authored_schema_text(
    raw_value: Any,
    *,
    field: str,
    index: int,
) -> Optional[str]:
    text = _coerce_optional_text(raw_value)
    if text is None:
        return None
    return text


def _optional_authored_schema_int(
    raw_value: Any,
    *,
    field: str,
    index: int,
) -> Optional[int]:
    value = _coerce_optional_int(raw_value)
    if raw_value not in (None, "") and value is None:
        raise ValueError(
            f"mission_context.expected_output_schema[{index}].{field} must be an integer"
        )
    return value


def _authored_schema_string_tuple(
    raw_value: Any,
    *,
    field: str,
    index: int,
) -> tuple[str, ...]:
    if raw_value in (None, ""):
        return ()
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ValueError(
            f"mission_context.expected_output_schema[{index}].{field} must be a sequence"
        )
    return tuple(str(item).strip() for item in raw_value if str(item).strip())


def _authored_schema_bool(
    raw_value: Any,
    *,
    field: str,
    index: int,
) -> bool:
    if raw_value in (None, ""):
        return False
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        lowered = raw_value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"mission_context.expected_output_schema[{index}].{field} must be a boolean")


def _objective_payloads(
    contract: MissionContract | Mapping[str, Any] | None,
) -> List[ObjectiveContract | Mapping[str, Any]]:
    if isinstance(contract, MissionContract):
        return list(contract.objectives)
    if isinstance(contract, Mapping):
        raw_objectives = contract.get("objectives")
        if isinstance(raw_objectives, Sequence) and not isinstance(raw_objectives, (str, bytes)):
            return list(raw_objectives)
    return []


def _extract_threshold_overrides(raw_value: Any) -> Dict[str, float]:
    if not isinstance(raw_value, Mapping):
        return {}

    overrides: Dict[str, float] = {}
    for key, value in raw_value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        try:
            overrides[key_text] = float(value)
        except (TypeError, ValueError):
            continue
    return overrides


def _reference_items(raw_references: Any) -> List[tuple[str, Any]]:
    if isinstance(raw_references, Mapping):
        return [(str(key), value) for key, value in raw_references.items()]
    if isinstance(raw_references, Sequence) and not isinstance(
        raw_references, (str, bytes)
    ):
        return [("", value) for value in raw_references]
    return []


def _extract_reference_titles(raw_references: Any) -> List[str]:
    titles: List[str] = []
    for label, item in _reference_items(raw_references):
        if isinstance(item, Mapping):
            title = _coerce_optional_text(item.get("title") or item.get("name"))
            if title:
                titles.append(title)
        elif label and isinstance(item, str) and item.startswith(("http://", "https://")):
            titles.append(label)
    return _unique_strings(titles)


def _extract_reference_domains(raw_references: Any) -> List[str]:
    domains: List[str] = []
    for _, item in _reference_items(raw_references):
        candidates: List[Any]
        if isinstance(item, Mapping):
            candidates = [
                item.get("url"),
                item.get("href"),
                item.get("source_url"),
            ]
        else:
            candidates = [item]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text.startswith(("http://", "https://")):
                continue
            domain = normalize_domain(text)
            if domain:
                domains.append(domain)
    return _unique_strings(domains)


def _normalize_string_list(raw_value: Any) -> List[str]:
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value.strip()]
    return []


def _normalize_domain_list(raw_value: Any) -> List[str]:
    return _unique_strings(_normalize_domain(item) for item in _normalize_string_list(raw_value))


def _normalize_domain(raw_value: Any) -> str:
    return normalize_domain(raw_value)


def _merge_unique_strings(*values: Sequence[str]) -> List[str]:
    return _unique_strings(item for group in values for item in group)


def _unique_strings(values: Sequence[str] | Any) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        items.append(text)
        seen.add(key)
    return items


def _coerce_mapping(raw_value: Any) -> Dict[str, Any]:
    if isinstance(raw_value, Mapping):
        return dict(raw_value)
    return {}


def _coerce_text(raw_value: Any, *, fallback: str) -> str:
    text = _coerce_optional_text(raw_value)
    return text or fallback


def _coerce_optional_text(raw_value: Any) -> Optional[str]:
    text = str(raw_value or "").strip()
    return text or None


def _coerce_optional_int(raw_value: Any) -> Optional[int]:
    if raw_value in (None, ""):
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(raw_value: Any) -> Optional[float]:
    if raw_value in (None, ""):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _coerce_int(raw_value: Any, *, default: int) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _first_text(items: Sequence[str], *, default: str) -> str:
    for item in items:
        text = str(item or "").strip()
        if text:
            return text
    return default


def _resolve_contract_dir(output_dir: Path | str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    override = os.getenv(CONTRACT_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return data_path("contracts")


def _contract_id_for_payload(payload: Mapping[str, Any]) -> str:
    # Usage timestamps/IDs describe execution, not the semantic contract. The
    # existing audit channel carries them without changing legacy/canonical IDs.
    semantic = dict(payload)
    decisions = semantic.get("entity_extraction_decisions")
    if isinstance(decisions, list):
        semantic["entity_extraction_decisions"] = [
            row for row in decisions
            if not isinstance(row, Mapping) or row.get("decision") != "provider_usage"
        ]
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _resolve_compiled_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("compiled_at must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("compiled_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _slug(value: str) -> str:
    normalized = _SLUG_PATTERN.sub("-", value.lower()).strip("-")
    return normalized or "item"


def _default_max_tokens_per_loop(research_depth: str) -> int:
    normalized = str(research_depth).strip().lower()
    if normalized == "alpha":
        return 48000
    if normalized == "deep":
        return 32000
    return 16000


__all__ = [
    "AcceptanceCheck",
    "CONTRACT_COMPILER_REVISION",
    "CONTRACT_SCHEMA_VERSION",
    "DeliverableSchemaContract",
    "EvidenceSlot",
    "ExecutionBudget",
    "MissionContract",
    "ObjectiveContract",
    "PreparedExecutionContract",
    "RetrievalBudget",
    "build_retrieval_budget",
    "compile_canonical_contract_from_state",
    "compile_contract_from_state",
    "is_acceptance_contract_payload",
    "persist_contract",
    "prepare_execution_contract",
    "resolve_research_objectives",
]
