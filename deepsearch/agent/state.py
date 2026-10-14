"""Typed agent state definition used throughout the DeepSearch LangGraph graph.

This module is a leaf dependency — it imports only from depth.py (also a leaf)
and standard library / LangChain core. No other agent/ modules may be imported.

Field ownership is documented inline so every field justifies its existence.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Optional

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
else:  # pragma: no cover - executed only when running without langchain_core
    try:
        from langchain_core.messages import BaseMessage
    except ImportError:

        class BaseMessage:
            """Fallback message type used until langchain_core is installed."""

            def __init__(self, content: str) -> None:
                self.content = content


class AgentState(TypedDict, total=False):
    """State container shared between LangGraph nodes.

    Every field has a documented owner — the module responsible for reading
    and writing it. Fields that exist only as bug patches have been removed;
    flow-control concerns belong in graph.py, not state.
    """

    # --- LangGraph message history (owner: graph.py) ---
    messages: Annotated[List[BaseMessage], operator.add]

    # --- Loop control (owner: graph.py / should_continue) ---
    research_loop_count: int
    max_loops: int
    min_loops: int

    # --- Mission identity (owner: loader.py / runner.py) ---
    mission_id: str
    # S80 m02 (PD107): canonical resolved mission title pinned ONCE inside
    # empty_state() from the passed mission_context. Both downstream readers
    # — codeagent_finalize_node (wrapper H1 + report filename) and
    # protocol._derive_title (returned protocol JSON .title) — read this key.
    # Replaces the divergent path-1/path-2 fallback resolution that produced
    # different strings for the same mission when context["title"] was empty.
    mission_title: str
    # S80 m04 (PD109): TraceLab system-assigned mission UUID, propagated by
    # ``deepsearch/worker/converter.convert_row_to_state`` from
    # ``MissionRow.id``. Persisted into save_report / save_protocol filenames
    # via an 8-hex-char suffix so re-runs of the same mission_id slug stop
    # silently overwriting the prior run's `.md` and `_protocol.json` on
    # disk. None for locally-loaded YAML missions and the local API path
    # — the persistence layer falls back to a random suffix in that case.
    mission_uuid: Optional[str]
    project_id: Optional[str]
    mission_objectives: List[str]
    mission_context: Dict[str, Any]
    deliverable_format: str
    mission_contract: Dict[str, Any]
    mission_contract_id: Optional[str]
    mission_contract_path: Optional[str]

    # --- Research accumulation (owner: research_node) ---
    # working_set + sources_found stripped at S78 m02 (PD71 conservative).
    # Both fields belonged to the pre-rebuild research-loop accumulator and
    # the post-S77 codeagent_finalize_node path no longer writes them. S83 m06
    # (PD83-b, aggressive flavor) stripped the untested dead reads, deleted
    # coverage._working_set_titles_for_matching, and repointed the evidence_links
    # / traceability protocol gates onto citation_telemetry. Four TESTED source/
    # report utilities still read the (dead-in-production) field rather than be
    # gutted (decision B): runner.py result sources_count,
    # agent/evidence.py:extract_sources_from_state, report/generator.py report
    # sources, report/evidence.py:generate_evidence_list. See
    # test_s78_state_field_strip_invariants.py + test_s83_m06_invariants.py for
    # the consumer-scan AST guards (never-reappears, with those carve-outs).
    finding_records: List[Dict[str, Any]]
    concept_clusters: List[Dict[str, Any]]
    findings: List[str]
    previous_queries: List[str]
    objective_embedding: List[float]

    # --- Reflection evaluation (owner: reflection_node) ---
    reflection_result: Dict[str, Any]
    reflection_scores: Dict[str, Any]
    slot_coverage: Dict[str, Any]
    deliverable_slot_fills: Dict[str, str]
    planner_fallback_reason: Dict[str, Any]
    coverage_gates: Dict[str, Any]
    refinement_log: List[str]
    mission_outcome: str
    routing_reason: str

    # --- Convergence tracking (owner: evaluation.py) ---
    convergence_history: List[float]
    objective_score_history: List[Dict[str, Any]]

    # --- Error & resilience (owner: search_orchestrator / graph.py) ---
    error_history: List[Dict[str, Any]]
    circuit_breaker_active: bool
    degraded_mode: bool
    degradation_policy: str
    termination_reason: str

    # --- Telemetry (owner: nodes / runner.py) ---
    telemetry: Dict[str, Any]

    # --- PEDR integration (owner: research_node) ---
    pedr_checked: bool
    pedr_reused: bool
    pedr_reviewed: bool
    pedr_result: Optional[Dict[str, Any]]

    # --- Report output (owner: finalize_node) ---
    report_path: Optional[str]
    protocol_path: Optional[str]
    # Exact finalize-emitted artifacts carried across the worker process fence.
    # Paths remain the preferred transport; these JSON-safe in-memory values are
    # the lossless fallback when attempt-local persistence is unavailable.
    report_markdown: Optional[str]
    protocol_payload: Optional[Dict[str, Any]]
    bibliography_admission_audit: List[Dict[str, Any]]
    entity_slot_admission_audit: List[Dict[str, Any]]
    # S83 m02 — finalize node-return channels. codeagent_finalize_node writes
    # result["synthesis_used"]=True after a successful synthesis (:420) and
    # accumulates result.setdefault("generation_errors", []).append(...) on a
    # report/protocol persistence failure (:467/:486). SAME S42.2 pitfall as the
    # block below: undeclared keys are silently dropped on the node-return merge,
    # so they MUST be declared channels. The generalized declare-channel guard in
    # test_s83_m02_invariants.py enforces this over the ENTIRE node-return surface
    # (every result[...]/result.setdefault key), not just the three gate keys —
    # closing the bug class rather than pinning the current keys by name.
    synthesis_used: bool
    generation_errors: List[str]
    # S42.2 — LangGraph 1.x StateGraph registers channels from the TypedDict
    # and silently drops undeclared keys on node-return merge. finalize_node
    # writes synthesis_telemetry["structural_validation"] and the sibling
    # structural_validation_failures list; without these declarations the
    # writes vanish and resolve_final_outcome_payload sees empty state.
    synthesis_telemetry: Dict[str, Any]
    structural_validation_failures: List[str]
    # S82 m01/m02 (PD136) — SAME S42.2 pitfall as above: the worker-path outcome
    # gate (evaluate_outcome_gate) writes these onto the finalize state-update,
    # and resolve_final_outcome_payload reads them back to populate
    # FinalOutcomePayload. They MUST be declared channels or LangGraph silently
    # drops them on node-return merge. The post-deploy cross-store re-dispatch
    # (62eab1b0) caught the omission: the gate computed delivery_quality
    # 'degraded' but final_outcome shipped empty failure lists + no
    # delivery_quality because these were undeclared (only the value nested
    # inside the declared synthesis_telemetry survived).
    delivery_quality: str
    quality_gate_failures: List[str]
    coverage_gate_failures: List[str]

    # --- Research depth tier (owner: loader.py; boundary string only post-S76) ---
    research_depth: str
    depth_config: Dict[str, Any]

    # --- Contradiction detection (owner: contradiction.py, Alpha only) ---
    contradiction_report: Optional[Dict[str, Any]]


def empty_state(
    *,
    mission_id: str,
    mission_objectives: Optional[List[str]] = None,
    project_id: Optional[str] = None,
    mission_context: Optional[Dict[str, Any]] = None,
    deliverable_format: str = "markdown",
    research_depth: str = "baseline",
    max_loops: Optional[int] = None,
    min_loops: Optional[int] = None,
    mission_uuid: Optional[str] = None,
) -> AgentState:
    """Return an initialised AgentState for a new mission.

    Args:
        mission_id: Mission identifier pulled from Mission Protocol YAML.
        mission_objectives: Optional list of objectives to seed the state.
        project_id: Optional TraceLab project identifier.
        mission_context: Optional dictionary containing mission background.
        deliverable_format: Output format preference (markdown/json/etc.).
        research_depth: Research depth tier label as a plain string
            ("baseline"/"deep"/"alpha"). Boundary data carrier post-S76;
            no in-process dispatch.
        max_loops: Maximum loops; defaults to 3 when None.
        min_loops: Minimum loops; defaults to 2 when None.

    Returns:
        AgentState populated with sensible defaults ready for LangGraph.
    """
    depth_str = research_depth.strip().lower() if isinstance(research_depth, str) else "baseline"
    if depth_str not in {"baseline", "deep", "alpha"}:
        depth_str = "baseline"

    effective_max_loops = max_loops if max_loops is not None else 3
    effective_min_loops = min_loops if min_loops is not None else 2

    objectives = list(mission_objectives or [])
    context = dict(mission_context or {})
    # S80 m02 (PD107): canonical mission_title pin. Each empty_state caller —
    # mission/loader._build_agent_state, runner.run_research_for_request,
    # worker/converter.convert_mission_row_to_state — has already normalized
    # context["title"] (loader + worker via mission.title_utils.normalize_mission_title).
    # The state-field pin makes the canonical string available to both readers
    # (codeagent_finalize_node wrapper H1 + protocol._derive_title) without
    # either site re-resolving from context. Falls back to mission_id when
    # title is empty (per feedback_no_distributed_advisory_fixes.md — one
    # binding authority, not N readers).
    canonical_mission_title = str(context.get("title") or "").strip() or mission_id
    return AgentState(
        messages=[],
        research_loop_count=0,
        max_loops=effective_max_loops,
        min_loops=effective_min_loops,
        mission_id=mission_id,
        mission_title=canonical_mission_title,
        mission_uuid=mission_uuid,
        project_id=project_id,
        mission_objectives=objectives,
        mission_context=context,
        deliverable_format=deliverable_format,
        mission_contract={},
        mission_contract_id=None,
        mission_contract_path=None,
        finding_records=[],
        concept_clusters=[],
        findings=[],
        previous_queries=[],
        reflection_result={
            "decision": "continue",
            "reasoning": "Not yet evaluated",
            "criteria_coverage": {},
            "evaluated_at_loop": 0,
        },
        reflection_scores={
            "overall_coverage_score": 0.0,
            "confidence_score": 0.0,
            "per_objective_scores": [],
            "is_mission_complete": False,
            "converged": False,
        },
        slot_coverage={},
        deliverable_slot_fills={},
        planner_fallback_reason={},
        coverage_gates={
            "passed": False,
            "failed_gates": [],
            "entity": {
                "passed": False,
                "covered": [],
                "uncovered": [],
                "threshold": 0.75,
                "detail": "",
            },
            "objective": {
                "passed": False,
                "per_objective_scores": {},
                "uncovered_objective_ids": [],
                "threshold": 0.70,
                "detail": "",
            },
            "slot": {
                "passed": False,
                "total_slots": 0,
                "filled_slots": 0,
                "thin_slot_ids": [],
                "unaddressed_slot_ids": [],
                "detail": "Coverage gates were not computed before routing.",
            },
        },
        convergence_history=[],
        objective_score_history=[],
        refinement_log=[],
        mission_outcome="",
        routing_reason="",
        error_history=[],
        circuit_breaker_active=False,
        degraded_mode=False,
        degradation_policy="normal",
        telemetry={
            "api_calls": {},
            "token_usage": {},
            "estimated_cost_usd": 0.0,
        },
        pedr_checked=False,
        pedr_reused=False,
        pedr_reviewed=False,
        pedr_result=None,
        report_path=None,
        protocol_path=None,
        report_markdown=None,
        protocol_payload=None,
        bibliography_admission_audit=[],
        entity_slot_admission_audit=[],
        synthesis_used=False,
        generation_errors=[],
        synthesis_telemetry={},
        structural_validation_failures=[],
        delivery_quality="",
        quality_gate_failures=[],
        coverage_gate_failures=[],
        research_depth=depth_str,
        depth_config={},
        contradiction_report=None,
    )
