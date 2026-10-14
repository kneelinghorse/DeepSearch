"""MissionProtocolComplete payload generator for TraceLab ingestion.

This module orchestrates all component generators (Evidence, KeyQuestions,
Synthesis, ResearchStatement) into a complete MissionProtocolComplete payload
ready for TraceLab upload.

Quality Gate Requirements (all must be "pass"):
- research_statement: ResearchStatement has topic, objective, scope
- evidence_links: At least 1 Evidence item
- synthesis_quality: Synthesis has key_insights, recommendations, next_steps
- traceability: Findings link to sources
- contradictions_resolved: No unresolved contradictions
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, Field

from deepsearch.agent.slot_coverage import extract_failing_slot_details
from deepsearch.agent.state import AgentState
from deepsearch.report.citation import CitationAnnotation
from deepsearch.report.evidence import Evidence, generate_evidence_list
from deepsearch.report.questions import KeyQuestion, generate_key_questions
from deepsearch.report.synthesis import (
    ResearchStatement,
    Synthesis,
    generate_research_statement,
    generate_synthesis,
)

# The 5 required TraceLab quality gates for mission completion
REQUIRED_COMPLETION_GATES: List[str] = [
    "research_statement",
    "evidence_links",
    "synthesis_quality",
    "traceability",
    "contradictions_resolved",
]

_STRUCTURAL_VALIDATION_FAILURE_REASON = "structural_validation_failed"


def canonicalize_protocol_payload(value: Any) -> Any:
    """Return a JSON-shaped protocol value with PostgreSQL-unsafe NULs removed.

    Protocol identity is established before the worker write. Canonicalizing
    here ensures the file, inline process-fence payload, identity hash, and
    stored JSONB all describe the same object instead of relying on a silent
    writer-side mutation after hashing.
    """

    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {
            key: canonicalize_protocol_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_protocol_payload(item) for item in value]
    return value


class QualityCheckpoint(BaseModel):
    """TraceLab-compatible QualityCheckpoint for gate validation.

    Each checkpoint represents one of the 5 required quality gates
    that must pass for a mission to be uploaded to TraceLab.
    """

    gate: str = Field(description="Quality gate name")
    status: Literal["pass", "fail", "skip"] = Field(
        default="pass",
        description="Gate status: pass, fail, or skip",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notes about this checkpoint",
    )
    validated_by: Optional[str] = Field(
        default=None,
        description="Agent or system that validated this gate",
    )
    validated_at: Optional[datetime] = Field(
        default=None,
        description="When validation occurred",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = self.model_dump(exclude_none=True)
        if self.validated_at:
            data["validated_at"] = self.validated_at.isoformat()
        return data


REPORT_METADATA_SCHEMA_VERSION: str = "s72-1.0"


class ReportMetadata(BaseModel):
    """S72 PD22 — run-integrity surface populated post-rebuild.

    Single truth surface for "did the agent really do work" gut-checking.
    Replaces the legacy ``execution_metadata`` fields (``sources_count``,
    ``loops_executed``, ``coverage``, ``quality_gates_passed``) which read
    0 / ``coverage_not_computed`` / ``False`` post-S64 because the stages
    that populated them are bypassed.

    The header (4 fields) is at-a-glance gut-check only — anything that
    needs explanation belongs in :pyattr:`forensic`. PD22 explicitly held
    ``build_hash`` and ``schema_version`` out of the header for that
    reason; both live under :pyattr:`forensic`.

    Populated by ``deepsearch.agent.codeagent_finalize_node`` from the
    locked S64 retrieval / S68 liveness / S67 citation telemetry signals.
    Any other caller that does not populate it leaves the field ``None``;
    legacy paths therefore stay round-trip-stable.
    """

    sources_collected: int = Field(
        default=0,
        description="Distinct URLs the runner accumulator touched (PD22 header).",
    )
    references: int = Field(
        default=0,
        description="Distinct URLs cited in the final report markdown (PD22 header).",
    )
    agent_steps_used: int = Field(
        default=0,
        description="CodeAgent step count for the synthesis run (PD22 header).",
    )
    agent_steps_max: int = Field(
        default=0,
        description="CodeAgent ``max_steps`` ceiling for the synthesis run (PD22 header).",
    )
    runtime_seconds: float = Field(
        default=0.0,
        description="Synthesis wall-clock runtime in seconds (PD22 header).",
    )
    forensic: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "PD22 forensic block — anything that needs explanation. Carries "
            "schema_version, build_hash (when reachable), liveness telemetry, "
            "citation telemetry, distinct-domain count, agent-loop terminators, "
            "token usage, tool-call summary, and the reconciled quality record."
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict; mirrors the other protocol-component to_dict() forms."""
        return self.model_dump()


class DepthConfigMeta(BaseModel):
    """Depth tier configuration metadata for downstream consumers.

    Captures the research depth settings used during mission execution,
    allowing consumers to understand the thoroughness level applied.
    """

    tier: str = Field(description="Research depth tier (baseline/deep/alpha)")
    max_loops: int = Field(description="Maximum research loops configured")
    min_loops: int = Field(description="Minimum loops before convergence")
    max_sources: int = Field(
        description="Legacy advisory source target retained for protocol compatibility"
    )
    convergence_threshold: float = Field(description="Score delta for convergence")
    source_quality_floor: float = Field(description="Minimum source authority score")
    require_source_diversity: bool = Field(
        default=False,
        description="Whether domain diversity was enforced",
    )
    enable_contradiction_detection: bool = Field(
        default=False,
        description="Whether contradiction detection was enabled",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()


class MissionProtocolComplete(BaseModel):
    """TraceLab-compatible MissionProtocolComplete payload.

    This model represents a complete research mission ready for TraceLab
    ingestion. All required fields must be populated and all 5 quality
    checkpoints must have status="pass" for successful upload.
    """

    mission_id: str = Field(description="Unique mission identifier")
    version: str = Field(default="1.0.0", description="Protocol version")
    title: str = Field(description="Mission title")
    summary: Optional[str] = Field(default=None, description="Mission summary")
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    status: Literal["review", "complete"] = Field(
        default="complete",
        description="Mission status",
    )
    owner: Optional[str] = Field(default=None, description="Mission owner")
    research_statement: ResearchStatement = Field(
        description="Research statement with topic, objective, scope"
    )
    key_questions: List[KeyQuestion] = Field(
        default_factory=list,
        description="Key research questions (>=1 answered required)",
    )
    synthesis: Synthesis = Field(description="Research synthesis")
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Evidence items (>=1 required)",
    )
    citations: List[CitationAnnotation] = Field(
        default_factory=list,
        description=(
            "Per-claim citation annotations with codepoint-indexed spans into "
            "the exact persisted result_markdown artifact."
        ),
    )
    sources_collected: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Runner-side retrieval-payload accumulator (S71, widened S72 PD24). "
            "One entry per distinct URL touched by web_search / source_fetch / "
            "url_liveness; shape "
            "{url, title, snippet, score, authority_tier, body, body_source, alive?} "
            "as produced by deepsearch.agent.tools.* — no separate schema. "
            "``body`` carries the full source_fetch text when available; "
            "``body_source`` discriminates ``\"snippet\"`` (300-char "
            "web_search preview) vs ``\"full\"`` (post-source_fetch substrate)."
        ),
    )
    report_metadata: Optional[ReportMetadata] = Field(
        default=None,
        description=(
            "S72 PD22 run-integrity block — at-a-glance header (4 fields) + "
            "forensic dict. Populated by codeagent_finalize_node from the "
            "post-S64 telemetry surface; legacy callers leave it None."
        ),
    )
    quality_checkpoints: List[QualityCheckpoint] = Field(
        default_factory=list,
        description="All 5 quality gates must be present and pass",
    )
    tags: List[str] = Field(default_factory=list, description="Mission tags")
    depth_config: Optional[DepthConfigMeta] = Field(
        default=None,
        description="Research depth configuration used during execution",
    )
    graph_provenance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Graph search provenance data (related research context)",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Creation timestamp",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = self.model_dump(exclude_none=True)
        # Convert nested models
        data["research_statement"] = self.research_statement.to_dict()
        data["synthesis"] = self.synthesis.to_dict()
        data["evidence"] = [e.to_dict() for e in self.evidence]
        data["citations"] = [c.to_dict() for c in self.citations]
        data["key_questions"] = [q.to_dict() for q in self.key_questions]
        data["quality_checkpoints"] = [c.to_dict() for c in self.quality_checkpoints]
        if self.depth_config:
            data["depth_config"] = self.depth_config.to_dict()
        if self.report_metadata is not None:
            data["report_metadata"] = self.report_metadata.to_dict()
        # Format timestamps
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        return data

    def all_gates_pass(self) -> bool:
        """Check that every required gate is present and NONE failed.

        S84 m04 — 'skip' (gate not evaluated) does not count as failure:
        a checkpoint whose validator does not exist reports skip honestly
        rather than a placeholder pass, and an unevaluated gate must not
        read as a failed one.
        """
        gate_status = {cp.gate: cp.status for cp in self.quality_checkpoints}
        for gate in REQUIRED_COMPLETION_GATES:
            if gate not in gate_status or gate_status[gate] == "fail":
                return False
        return True


def generate_mission_protocol(state: AgentState) -> MissionProtocolComplete:
    """Generate complete TraceLab mission protocol from AgentState.

    Orchestrates all component generators (B4.1-B4.4) and assembles
    into a MissionProtocolComplete payload ready for TraceLab ingestion.

    Args:
        state: AgentState from completed research mission.

    Returns:
        MissionProtocolComplete ready for TraceLab ingestion.

    Raises:
        ValidationError: If payload fails Pydantic schema validation.

    Example:
        >>> state = {
        ...     "mission_id": "DRM.0.1",
        ...     "mission_context": {"title": "API Research"},
        ...     "mission_objectives": ["Document endpoints"],
        ...     "findings": ["Endpoint /api/v1 discovered"],
        ...     # S83 m06: evidence_links + traceability gates now read the
        ...     # citation evidence the codeagent path produces.
        ...     "synthesis_telemetry": {
        ...         "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4}
        ...     },
        ... }
        >>> protocol = generate_mission_protocol(state)
    """
    # Generate all components from state
    evidence = generate_evidence_list(state)
    key_questions = generate_key_questions(state)
    synthesis = generate_synthesis(state)
    research_statement = generate_research_statement(state)

    # Ensure at least 1 evidence item (TraceLab requirement)
    if not evidence:
        evidence = [
            Evidence(
                evidence_id="EV-001",
                source=state.get("mission_id", "research"),
                summary="Research conducted via autonomous web search",
                source_type="web_search",
            )
        ]

    # Build quality checkpoints from actual state and synthesis signals.
    quality_checkpoints = _build_quality_checkpoints(
        state,
        research_statement=research_statement,
        synthesis=synthesis,
        evidence=evidence,
    )

    # Derive title and summary
    title = _derive_title(state)
    summary = _derive_summary(state)

    # Build depth config metadata for downstream consumers
    depth_config_meta = _build_depth_config_meta(state)
    graph_provenance = _build_graph_provenance(state)

    # Build the complete protocol
    now = datetime.now(timezone.utc)
    protocol = MissionProtocolComplete(
        mission_id=state.get("mission_id", "UNKNOWN"),
        version="1.0.0",
        title=title,
        summary=summary,
        project_id=state.get("project_id"),
        # S84 m04 truth repair: the pre-m04 hardcode said "complete" even
        # when the outcome gate resolved VALIDATION_FAILED — and the worker's
        # protocol-regeneration fallback ships this builder's output for
        # BLOCKED missions too (adversarial-review fix: blocked also maps to
        # 'review'). Absent/complete outcomes keep 'complete' (legacy states).
        status=(
            "review"
            if str(state.get("mission_outcome") or "").lower()
            in ("validation_failed", "blocked")
            else "complete"
        ),
        owner="deepsearch-agent",
        research_statement=research_statement,
        key_questions=key_questions,
        synthesis=synthesis,
        evidence=evidence,
        quality_checkpoints=quality_checkpoints,
        tags=_extract_tags(state),
        depth_config=depth_config_meta,
        graph_provenance=graph_provenance,
        created_at=now,
        updated_at=now,
    )

    return protocol


def _build_depth_config_meta(state: AgentState) -> Optional[DepthConfigMeta]:
    """Build depth config metadata from state for protocol export.

    Args:
        state: AgentState dictionary with depth_config and research_depth.

    Returns:
        DepthConfigMeta if depth info is available, None otherwise.
    """
    depth_config = state.get("depth_config")
    research_depth = state.get("research_depth")

    if not depth_config or not research_depth:
        return None

    # Handle both dict and object forms
    if hasattr(depth_config, "to_dict"):
        config_dict = depth_config.to_dict()
    elif isinstance(depth_config, dict):
        config_dict = depth_config
    else:
        return None

    return DepthConfigMeta(
        tier=str(research_depth).lower(),
        max_loops=int(config_dict.get("max_loops", 3)),
        min_loops=int(config_dict.get("min_loops", 2)),
        max_sources=int(
            config_dict.get(
                "max_sources",
                _default_protocol_max_sources(str(research_depth)),
            )
        ),
        convergence_threshold=float(config_dict.get("convergence_threshold", 0.05)),
        source_quality_floor=float(config_dict.get("source_quality_floor", 0.5)),
        require_source_diversity=bool(config_dict.get("require_source_diversity", False)),
        enable_contradiction_detection=bool(
            config_dict.get("enable_contradiction_detection", False)
        ),
    )


def _default_protocol_max_sources(research_depth: str) -> int:
    normalized = str(research_depth).strip().lower()
    if normalized == "alpha":
        return 45
    if normalized == "deep":
        return 30
    return 15


def _build_graph_provenance(state: AgentState) -> Optional[Dict[str, Any]]:
    """Build graph provenance payload from AgentState for protocol export."""
    pedr_result = state.get("pedr_result")
    graph_context = (
        pedr_result.get("graph_context")
        if isinstance(pedr_result, Mapping)
        else None
    )
    runtime_diagnostics = _build_runtime_diagnostics(state)
    if not isinstance(graph_context, Mapping) or not graph_context:
        if not runtime_diagnostics:
            return None
        return {"metadata": {"runtime_diagnostics": runtime_diagnostics}}

    related_research: List[Dict[str, Any]] = []
    results = graph_context.get("results") or graph_context.get("matches") or []
    if isinstance(results, list):
        for entry in results:
            if not isinstance(entry, Mapping):
                continue
            mission_id = entry.get("mission_id") or entry.get("missionId")
            title = entry.get("title")
            score = (
                entry.get("rrf_score")
                or entry.get("rrfScore")
                or entry.get("similarity_score")
                or entry.get("similarity")
                or entry.get("score")
            )
            if mission_id or title or score is not None:
                related_research.append(
                    {
                        "mission_id": mission_id,
                        "title": title,
                        "similarity_score": score,
                    }
                )

    related_entities: List[Dict[str, Any]] = []
    entities = graph_context.get("related_entities") or []
    if isinstance(entities, list):
        for entry in entities:
            if not isinstance(entry, Mapping):
                continue
            related_entities.append(
                {
                    "label": entry.get("label") or entry.get("name"),
                    "entity_type": entry.get("entity_type") or entry.get("type"),
                    "score": entry.get("score"),
                    "mission_id": entry.get("mission_id") or entry.get("missionId"),
                    "mission_uuid": entry.get("mission_uuid") or entry.get("missionUuid"),
                }
            )

    metadata = graph_context.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    metadata_dict = dict(metadata)
    if runtime_diagnostics:
        metadata_dict["runtime_diagnostics"] = runtime_diagnostics

    provenance: Dict[str, Any] = {
        "query": graph_context.get("query"),
        "metadata": metadata_dict,
    }
    if related_research:
        provenance["related_research"] = related_research
    if related_entities:
        provenance["related_entities"] = related_entities

    if not related_research and not related_entities and not metadata_dict:
        return None
    return provenance


def _build_runtime_diagnostics(state: AgentState) -> Optional[Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}

    synthesis_telemetry = _extract_synthesis_telemetry(state)
    finalize: Dict[str, Any] = {}
    report_render_path = str(synthesis_telemetry.get("report_render_path") or "").strip()
    report_render_reason = str(synthesis_telemetry.get("report_render_reason") or "").strip()
    if report_render_path:
        finalize["report_render_path"] = report_render_path
    if report_render_reason:
        finalize["report_render_reason"] = report_render_reason
    if "finalize_contract_present" in synthesis_telemetry:
        finalize["contract_present"] = bool(synthesis_telemetry.get("finalize_contract_present"))
    if "finalize_has_deliverable_schemas" in synthesis_telemetry:
        finalize["has_deliverable_schemas"] = bool(
            synthesis_telemetry.get("finalize_has_deliverable_schemas")
        )
    schema_ids = _normalize_string_list(
        synthesis_telemetry.get("finalize_deliverable_schema_ids")
    )
    if schema_ids:
        finalize["deliverable_schema_ids"] = schema_ids
    if finalize:
        diagnostics["finalize"] = finalize

    slot_coverage = state.get("slot_coverage")
    failed_slots = extract_failing_slot_details(
        slot_coverage if isinstance(slot_coverage, Mapping) else None
    )
    if failed_slots:
        diagnostics["coverage"] = {
            "failed_slot_ids": [str(slot["slot_id"]) for slot in failed_slots],
            "failed_slot_kinds": _normalize_string_list(slot.get("kind") for slot in failed_slots),
            "failed_slots": failed_slots,
        }

    return diagnostics or None


def _build_quality_checkpoints(
    state: AgentState | None = None,
    *,
    research_statement: ResearchStatement | None = None,
    synthesis: Synthesis | None = None,
    evidence: List[Evidence] | None = None,
) -> List[QualityCheckpoint]:
    """Generate the required quality checkpoints from actual mission state."""
    normalized_state: AgentState = state or {}
    now = datetime.now(timezone.utc)
    gate_evaluations = {
        "research_statement": _evaluate_research_statement_gate(
            normalized_state,
            research_statement=research_statement,
        ),
        "evidence_links": _evaluate_evidence_links_gate(
            normalized_state,
            evidence=evidence,
        ),
        "synthesis_quality": _evaluate_synthesis_quality_gate(
            normalized_state,
            synthesis=synthesis,
        ),
        "traceability": _evaluate_traceability_gate(normalized_state),
        "contradictions_resolved": _evaluate_contradictions_gate(),
    }

    checkpoints: List[QualityCheckpoint] = []
    for gate_name in REQUIRED_COMPLETION_GATES:
        status, notes = gate_evaluations[gate_name]
        checkpoints.append(
            QualityCheckpoint(
                gate=gate_name,
                status=status,
                notes=notes,
                validated_by="deepsearch-agent",
                validated_at=now,
            )
        )

    return checkpoints


def _evaluate_research_statement_gate(
    state: AgentState,
    *,
    research_statement: ResearchStatement | None,
) -> tuple[Literal["pass", "fail", "skip"], str]:
    objectives = state.get("mission_objectives")
    objective_count = len(objectives) if isinstance(objectives, list) else 0
    if objective_count > 0:
        return "pass", f"Mission declared {objective_count} objective(s)."

    contract = state.get("mission_contract")
    if isinstance(contract, Mapping):
        contract_objectives = contract.get("objectives")
        if isinstance(contract_objectives, list) and contract_objectives:
            return "pass", (
                f"Compiled contract declared {len(contract_objectives)} objective(s)."
            )

    context = state.get("mission_context")
    if (
        isinstance(context, Mapping)
        and research_statement is not None
        and research_statement.passes_quality_gate()
        and any(str(context.get(key) or "").strip() for key in ("title", "objective", "summary"))
    ):
        return "pass", "Research statement derived from explicit mission context."

    if research_statement is not None and research_statement.passes_quality_gate():
        return (
            "fail",
            "Research statement text exists, but no mission objectives were provided.",
        )
    return "fail", "Missing mission objectives for research-statement validation."


def _evaluate_evidence_links_gate(
    state: AgentState,
    *,
    evidence: List[Evidence] | None,
) -> tuple[Literal["pass", "fail", "skip"], str]:
    # S83 m06 (PD83-b) — REPOINTED. The old branches read sources_found /
    # working_set, both stripped from AgentState at S78 (no production writer),
    # so this gate degenerated to a CONSTANT misleading 'fail' shipped to
    # TraceLab on every codeagent dispatch. The codeagent path genuinely
    # produces citation evidence, so the gate now reads the citations the
    # synthesis extracted (canonical access mirrors protocol.py:838-842 /
    # outcome.py:173-177) — a truthful pass/fail rather than discarded info.
    synthesis_telemetry = _extract_synthesis_telemetry(state)
    citation_telemetry = synthesis_telemetry.get("citation_telemetry")
    citations_extracted = 0
    if isinstance(citation_telemetry, Mapping):
        try:
            citations_extracted = int(citation_telemetry.get("citations_extracted") or 0)
        except (TypeError, ValueError):
            citations_extracted = 0

    if citations_extracted > 0:
        return "pass", (
            f"Synthesis extracted {citations_extracted} citation(s) backing the "
            "evidence links."
        )

    evidence_count = len(evidence or [])
    if evidence_count:
        return (
            "fail",
            f"Protocol synthesized {evidence_count} fallback evidence item(s), but "
            "synthesis extracted no citations.",
        )
    return "fail", "Synthesis extracted no citations for evidence-link validation."


def _evaluate_synthesis_quality_gate(
    state: AgentState,
    *,
    synthesis: Synthesis | None,
) -> tuple[Literal["pass", "fail", "skip"], str]:
    synthesis_telemetry = _extract_synthesis_telemetry(state)
    structural_failures = _normalize_string_list(
        state.get("structural_validation_failures")
    ) or _normalize_string_list(
        synthesis_telemetry.get("structural_validation_failed_schemas")
    ) or _normalize_string_list(
        _extract_final_outcome(state).get("structural_validation_failures")
    )
    structural_validation_passed = synthesis_telemetry.get("structural_validation_passed")
    routing_reason = str(
        state.get("routing_reason")
        or synthesis_telemetry.get("routing_reason")
        or _extract_final_outcome(state).get("routing_reason")
        or ""
    ).strip()
    has_structural_contract = _state_has_deliverable_schemas(state)

    if structural_validation_passed is True:
        return "pass", "Deterministic structural validation passed after synthesis."
    if structural_validation_passed is False or structural_failures:
        failed = ", ".join(structural_failures) if structural_failures else "unknown schema"
        return "fail", f"Deterministic structural validation failed for: {failed}."
    if routing_reason == _STRUCTURAL_VALIDATION_FAILURE_REASON:
        return "fail", "Synthesis exhausted retries and routed as structural_validation_failed."
    if has_structural_contract:
        return (
            "skip",
            "Structural validation was retired in S76; deliverable schemas remain "
            "boundary data, but this gate is not applicable and was not evaluated.",
        )
    if synthesis is not None and synthesis.passes_quality_gate():
        return "pass", "Synthesis object satisfied baseline TraceLab content checks."
    return "fail", "Synthesis output did not satisfy baseline content checks."


def _evaluate_traceability_gate(
    state: AgentState,
) -> tuple[Literal["pass", "fail", "skip"], str]:
    # S83 m06 (PD83-b) — REPOINTED. The old gate read citation_density (CONFIRMED
    # zero production writers) and fell back to the S78-stripped sources_found, so
    # it also shipped a CONSTANT misleading 'fail' to TraceLab. Repointed onto the
    # same citation_telemetry block the codeagent path produces: a report that
    # cites distinct URLs is traceable.
    synthesis_telemetry = _extract_synthesis_telemetry(state)
    citation_telemetry = synthesis_telemetry.get("citation_telemetry")
    citations_extracted = 0
    distinct_urls = 0
    if isinstance(citation_telemetry, Mapping):
        try:
            citations_extracted = int(citation_telemetry.get("citations_extracted") or 0)
        except (TypeError, ValueError):
            citations_extracted = 0
        try:
            distinct_urls = int(citation_telemetry.get("distinct_urls_cited") or 0)
        except (TypeError, ValueError):
            distinct_urls = 0

    if citations_extracted > 0:
        return (
            "pass",
            f"Synthesis cited {distinct_urls} distinct URL(s) across "
            f"{citations_extracted} citation(s).",
        )
    return "fail", "No citations extracted for traceability validation."


def _evaluate_contradictions_gate() -> tuple[Literal["pass", "fail", "skip"], str]:
    # S84 m04 truth repair: no contradiction validator exists, so the gate
    # reports 'skip' (not evaluated) instead of the Sprint 31 placeholder
    # unconditional 'pass'. all_gates_pass() treats skip as non-failure.
    return (
        "skip",
        "contradiction validator not implemented; gate not evaluated.",
    )


def _extract_final_outcome(state: AgentState) -> Mapping[str, Any]:
    telemetry = state.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return {}
    final_outcome = telemetry.get("final_outcome")
    return final_outcome if isinstance(final_outcome, Mapping) else {}


def _extract_synthesis_telemetry(state: AgentState) -> Mapping[str, Any]:
    synthesis_telemetry = state.get("synthesis_telemetry")
    if isinstance(synthesis_telemetry, Mapping):
        return synthesis_telemetry
    telemetry = state.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return {}
    nested = telemetry.get("synthesis_telemetry")
    return nested if isinstance(nested, Mapping) else {}


def _state_has_deliverable_schemas(state: AgentState) -> bool:
    contract = state.get("mission_contract")
    if not isinstance(contract, Mapping):
        return False
    deliverable_schemas = contract.get("deliverable_schemas")
    return isinstance(deliverable_schemas, list) and bool(deliverable_schemas)


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, (str, bytes)) or value is None:
        return []
    if not isinstance(value, list):
        try:
            value = list(value)
        except TypeError:
            return []
    normalized: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _derive_title(state: AgentState) -> str:
    """Read the canonical mission title from state.

    S80 m02 (PD107): the independent title/objective/mission_id fallback
    resolution that lived here pre-S80 has been deleted. The canonical title
    is pinned once inside ``empty_state()`` (see deepsearch/agent/state.py)
    from the loader/runner/worker call sites — protocol JSON ``.title`` and
    the wrapper H1 emitted by codeagent_finalize_node read the SAME key so
    the two strings cannot diverge for the same execution.

    Args:
        state: AgentState dictionary.

    Returns:
        Non-empty title string. Falls back to ``mission_id`` (or
        ``"Research"`` when neither key is present) for hand-constructed
        states that bypass ``empty_state()``.
    """
    return state.get("mission_title") or state.get("mission_id") or "Research"


def _derive_summary(state: AgentState) -> str:
    """Derive mission summary from state.

    Sources summary from (in order of preference):
    1. Explicit summary in mission_context (≥20 chars)
    2. S81 m05 (PD126): synthesis_telemetry fields populated by the
       CodeAgent finalize node — sources_collected count, citations
       extracted count, and the deliverable_coverage satisfied/total ratio
       when the post-m01 coverage gate has populated it
    3. Generic non-empty string when synthesis_telemetry is empty (so
       protocol.summary never reads as a literal falsehood like the pre-
       S81 "0 unique concepts from 0 sources" anti-cue)

    Pre-S81: the synthesized branch read state['sources_found'] /
    summarize_state_findings(state) / reflection_scores — all dead state
    paths post-S77. Every CodeAgent mission's protocol.summary read the
    literal "Autonomous research mission with 0 unique concepts (0 raw
    findings) from 0 sources. Coverage: 0%" regardless of actual outcome.
    Cosmetic-only defect (summary string rarely surfaced in user-facing
    reports) but visible falsehood undermines confidence in protocol
    metadata.

    Args:
        state: AgentState dictionary.

    Returns:
        Non-empty summary string (guaranteed).
    """
    context = state.get("mission_context") or {}

    # Try explicit summary
    summary = context.get("summary", "")
    if summary and len(summary.strip()) >= 20:
        return summary[:500].strip()

    # S81 m05 (PD126): reroute through synthesis_telemetry. The post-S77
    # state paths (sources_found / summarize_state_findings /
    # reflection_scores) all return empty values regardless of mission
    # outcome. The CodeAgent finalize node populates synthesis_telemetry
    # with live counts; read those instead.
    synthesis_telemetry = (
        state.get("synthesis_telemetry")
        or (state.get("telemetry") or {}).get("synthesis_telemetry")
        or {}
    )
    if not isinstance(synthesis_telemetry, Mapping):
        synthesis_telemetry = {}

    try:
        sources = int(synthesis_telemetry.get("sources_collected") or 0)
    except (TypeError, ValueError):
        sources = 0
    citation_telemetry = synthesis_telemetry.get("citation_telemetry") or {}
    if not isinstance(citation_telemetry, Mapping):
        citation_telemetry = {}
    try:
        citations = int(citation_telemetry.get("citations_extracted") or 0)
    except (TypeError, ValueError):
        citations = 0

    coverage = synthesis_telemetry.get("deliverable_coverage") or {}
    if not isinstance(coverage, Mapping):
        coverage = {}
    satisfied_list = coverage.get("satisfied") or []
    missing_list = coverage.get("missing") or []
    satisfied = len(satisfied_list) if isinstance(satisfied_list, (list, tuple)) else 0
    missing = len(missing_list) if isinstance(missing_list, (list, tuple)) else 0
    total_deliverables = satisfied + missing

    # S84 m04 review fix: "N citations FROM M sources" asserted a provenance
    # relationship that is false — sources_collected counts every URL the
    # accumulator touched (SERP results never opened, probe-only URLs,
    # failed fetches). The semicolon phrasing reports both counts honestly
    # without claiming the citations came from all of them.
    if total_deliverables > 0:
        coverage_pct = satisfied / total_deliverables
        return (
            f"Autonomous research mission with {citations} citations; "
            f"{sources} sources collected. Deliverables: {satisfied}/{total_deliverables} "
            f"satisfied ({coverage_pct:.0%})."
        )
    if sources > 0 or citations > 0:
        return (
            f"Autonomous research mission with {citations} citations; "
            f"{sources} sources collected."
        )
    # Defensive: synthesis_telemetry empty or absent (e.g. early-block path
    # before the finalize node ran). Return a non-empty generic string
    # rather than the pre-S81 zero-cue falsehood.
    return "Autonomous research mission (telemetry not yet populated)."


def _extract_tags(state: AgentState) -> List[str]:
    """Extract tags from mission metadata.

    Generates tags from:
    - Base tags: deepsearch, autonomous-research
    - Mission ID prefix (e.g., "drm" from "DRM.0.1")

    Args:
        state: AgentState dictionary.

    Returns:
        List of tag strings.
    """
    tags = ["deepsearch", "autonomous-research"]

    mission_id = state.get("mission_id", "")
    if mission_id:
        # Extract prefix (e.g., "DRM" from "DRM.0.1")
        prefix = mission_id.split(".")[0] if "." in mission_id else mission_id
        tags.append(prefix.lower())

    return tags


__all__ = [
    "MissionProtocolComplete",
    "QualityCheckpoint",
    "DepthConfigMeta",
    "ReportMetadata",
    "REPORT_METADATA_SCHEMA_VERSION",
    "REQUIRED_COMPLETION_GATES",
    "canonicalize_protocol_payload",
    "generate_mission_protocol",
]
