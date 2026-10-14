"""Mission outcome helpers for finalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from typing_extensions import NotRequired, TypedDict

from .coverage import coerce_source_coverage_gate_result
from .critique_policy import CRITIQUE_RATIO_MIN_APPLIED
from .quality_series import build_quality_record


class MissionOutcome(str, Enum):
    """Persisted terminal outcome for a mission run."""

    COMPLETE = "complete"
    VALIDATION_FAILED = "validation_failed"
    BLOCKED = "blocked"


class FinalOutcomePayload(TypedDict):
    """Structured final outcome stored in telemetry and execution metadata."""

    outcome: str
    routing_reason: str
    coverage_gate_failures: list[str]
    quality_gate_failures: list[str]
    structural_validation_failures: list[str]
    # S41.2 — full ValidationReport so post-run audit can see every
    # validator by id, not only failures. Surfaces passing + failing
    # validators with their detail and remediation_hint, enabling Phase A
    # scoring to distinguish "validator fired and passed" from "validator
    # didn't run in this build". Additive; structural_validation_failures
    # list shape stays unchanged for backward compatibility.
    validation_report: NotRequired[dict[str, Any] | None]
    # S41.4 — per-entity-slot admission audit surfaced from state so §4.5
    # S40.1 empirical gates become scoreable via MCP get_mission without
    # shell access to the Railway container. Each entry describes one
    # source-slot evaluation: slot_id, entity (canonical), source_url,
    # source_title, coherence_score, relevance_source
    # ('embedding' | 'tfidf_fallback'), floor_used, bound (bool), loop.
    # Additive; state shape is unchanged.
    entity_slot_admission_audit: NotRequired[list[dict[str, Any]]]
    # S82 m01 (PD136) — tiered delivery-quality signal set by the worker-path
    # outcome gate: 'clean' | 'usable_with_warning' | 'degraded' | 'failed'.
    # Additive (NotRequired) so legacy/local paths that never set it keep shape.
    delivery_quality: NotRequired[str]
    authored_scope_validation: NotRequired[dict[str, Any]]
    # S84 m04 — loop_count DELETED: the legacy research_loop_count is
    # initialized 0 and never incremented on the single-node codeagent graph,
    # so the payload claimed zero work on missions whose step_count was
    # 20-40. The honest effort signal is synthesis_telemetry.step_count.
    completed_at: str


def map_reason_to_outcome(routing_reason: str, coverage_gates: Any) -> MissionOutcome:
    """Map a routing reason plus coverage state to a terminal mission outcome."""

    normalized_reason = str(routing_reason or "").strip()

    if normalized_reason == "circuit_breaker_active":
        return MissionOutcome.BLOCKED
    # Coverage and structural shortfalls are annotations on a completed
    # payload, not reasons to suppress user-facing output.
    #
    # NOTE (S82 m01): this legacy mapping is intentionally left COMPLETE-biased.
    # It is the FALLBACK used by resolve_mission_outcome when state carries no
    # explicit mission_outcome (local-runner / legacy paths). The production
    # worker path now sets state["mission_outcome"] from evaluate_outcome_gate
    # below, so it never reaches this function for the gated decision.
    return MissionOutcome.COMPLETE


# ---------------------------------------------------------------------------
# S82 m01 (PD136) — tiered outcome gate.
#
# Until S82, codeagent_finalize_node hardcoded mission_outcome=COMPLETE and the
# only real gate hid behind runner.py validate=True (never passed on the worker
# dispatch path). The pipeline COMPUTED shortfall_score, dead/orphan counts,
# citations_extracted and hit_max_steps but ENFORCED none of them. This gate
# makes those computed signals load-bearing on the production worker path.
#
# Tiers (PD136, user-approved 2026-05-28):
#   CATASTROPHIC -> VALIDATION_FAILED (report still persisted + reviewable):
#     - citations_extracted == 0 on a non-trivial report (>= MIN_REPORT_WORDS), OR
#     - output truncation / critique validation failure / major critique damage.
#   DEGRADED -> COMPLETE + delivery_quality='degraded' + populated failure lists:
#     - ANY positive deliverable shortfall (incl. a broad multi-deliverable 1.0), OR
#     - dead-link rate >= SOFT_DEAD_RATE, OR
#     - orphan count >= SOFT_ORPHAN.
#   USABLE-WITH-WARNING -> COMPLETE when the artifact remains usable but
#     validation is incomplete or carries bounded non-material defects,
#     including a structurally valid max-step fallback.
#   CLEAN -> COMPLETE + delivery_quality='clean' only when all critical
#     validation facts are present and healthy.
#   BLOCKED (empty output / runner error / circuit breaker) is handled upstream
#   in codeagent_finalize_node and never reaches this gate.
#
# A deliverable-coverage shortfall DEGRADES rather than hard-fails (DSV-045,
# 2026-06-15; extends the single-deliverable carve-out user-approved 2026-05-28
# to ALL shortfalls). completeness_check is a recall-biased structural heuristic
# whose nested/misnested edge cases cannot be resolved from markdown structure
# alone, so a coverage shortfall must not be the SOLE trigger for catastrophic
# failure: a live 3.5k-word, well-cited report was failed 0/4 by a completeness
# false-MISSING. A genuinely empty run still hard-fails via the zero-citations
# trigger. The downstream consumer is agents that DOWNLOAD AND USE even
# degraded/failed reports, so the gate's value is the FLAG, not suppression.
#
# Legacy thresholds remain pinned by preserved telemetry fixtures. The S87
# critique ratio bands are calibrated from the cached CAT distributions in
# cmos/reports/sprint_87_m04_quality_calibration.md.
# ---------------------------------------------------------------------------

# NON-LOAD-BEARING since DSV-045 (2026-06-15). These two pinned the OLD
# catastrophic hard-shortfall trigger (shortfall_score >= HARD_SHORTFALL over
# >= MIN_DELIVERABLES_FOR_HARD_SHORTFALL deliverables -> VALIDATION_FAILED). That
# trigger was REMOVED — a deliverable-coverage shortfall now DEGRADES, never
# hard-fails (see evaluate_outcome_gate; decision #820: do NOT re-introduce it).
# Retained, not deleted, because (a) test_s82_m01_invariants pins their values as
# the historical record, (b) quality_series documents them in the admitted PD136
# threshold set, and (c) the PD136 calibration roadmap item still references
# HARD_SHORTFALL as a provisional threshold to revisit. The dead
# GATE_REASON_HARD_SHORTFALL string (never emitted, never exported) was deleted.
HARD_SHORTFALL = 0.60
MIN_DELIVERABLES_FOR_HARD_SHORTFALL = 2
SOFT_DEAD_RATE = 0.10
SOFT_ORPHAN = 3
MIN_REPORT_WORDS = 200
CRITIQUE_DEGRADED_RATIO = 0.20
CRITIQUE_FAILED_RATIO = 0.50

DELIVERY_QUALITY_CLEAN = "clean"
DELIVERY_QUALITY_USABLE_WITH_WARNING = "usable_with_warning"
DELIVERY_QUALITY_DEGRADED = "degraded"
DELIVERY_QUALITY_FAILED = "failed"

GATE_REASON_ZERO_CITATIONS = "zero_citations_on_nontrivial_report"
GATE_REASON_MAX_STEPS_NO_FINAL = "hit_max_steps_without_final_answer"
# S88 — the final-answer generation was cut at max_tokens (finish_reason ==
# "length"): the shipped report is truncated mid-content. Catastrophic like
# max-steps-without-final — a budget-truncated report is an unambiguous
# incompleteness that must never ship as COMPLETE/usable. It also drives the
# bounded recovery pass (a truncated report carries a coverage shortfall), so a
# clean recovery still lands COMPLETE; only an unrecovered truncation ships
# VALIDATION_FAILED (reviewable artifact).
GATE_REASON_OUTPUT_TRUNCATED = "final_answer_truncated_at_max_tokens"
GATE_REASON_GENERATION_ERROR_RECOVERED = (
    "generation_error_recovered_from_paid_memory"
)
GATE_REASON_GENERATION_FAILURE_RECOVERED_BY_BOUNDED_PASS = (
    "generation_failure_recovered_by_bounded_pass"
)
GATE_REASON_GENERATION_FAILURE_UNRECOVERED = "generation_failure_unrecovered"
GATE_REASON_GENERATION_RECOVERY_UNPROVEN = "generation_recovery_unproven"
GATE_REASON_CRITIQUE_ERROR = "critique_validation_error"


@dataclass(frozen=True)
class OutcomeGateResult:
    """Deterministic tiered gate decision over the synthesis telemetry."""

    outcome: MissionOutcome
    routing_reason: str
    delivery_quality: str
    quality_gate_failures: list[str]
    coverage_gate_failures: list[str]


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def evaluate_outcome_gate(
    synthesis_telemetry: Mapping[str, Any],
    *,
    report_word_count: int,
) -> OutcomeGateResult:
    """Resolve the terminal outcome + delivery quality from computed signals.

    Pure function over the finalize-node ``synthesis_telemetry`` envelope (plus
    the shipped report's word count). No I/O, no state mutation — the single
    source of truth for the S82 m01 (PD136) tiered failure policy. See the
    module comment above for the tier definitions and threshold rationale.

    Args:
        synthesis_telemetry: the finalize-node telemetry envelope. Reads
            ``citation_telemetry.citations_extracted``,
            ``deliverable_coverage.{shortfall_score,satisfied,missing}``,
            ``liveness_telemetry.{checked,dead,orphan}``, and the top-level
            ``hit_max_steps`` / ``self_terminated`` flags plus the structured
            quality-record facts for critique, liveness, retrieval, and
            recovery. A complete max-step fallback is a warning; independent
            evidence defects still fail. Missing critical validation is
            warning/failed, never a clean zero. The gate never raises.
        report_word_count: word count of the shipped report markdown, used by
            the zero-citations catastrophic trigger to distinguish a non-trivial
            report from a legitimately terse "no data found" note.
    """

    citation_telemetry = synthesis_telemetry.get("citation_telemetry")
    citation_telemetry = (
        citation_telemetry if isinstance(citation_telemetry, Mapping) else {}
    )
    citations_extracted = _coerce_int(citation_telemetry.get("citations_extracted"))

    coverage = synthesis_telemetry.get("deliverable_coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    shortfall_score = _coerce_float(coverage.get("shortfall_score"))
    satisfied = coverage.get("satisfied")
    missing = coverage.get("missing")
    deliverable_count = (len(satisfied) if isinstance(satisfied, list) else 0) + (
        len(missing) if isinstance(missing, list) else 0
    )

    liveness = synthesis_telemetry.get("liveness_telemetry")
    liveness = liveness if isinstance(liveness, Mapping) else {}
    checked = _coerce_int(liveness.get("checked"))
    dead = _coerce_int(liveness.get("dead"))
    orphan = _coerce_int(liveness.get("orphan"))
    dead_rate = (dead / checked) if checked > 0 else 0.0

    hit_max_steps = bool(synthesis_telemetry.get("hit_max_steps"))
    self_terminated = bool(synthesis_telemetry.get("self_terminated"))
    recovered_from_scratchpad = bool(
        synthesis_telemetry.get("recovered_from_scratchpad")
    )
    max_steps_fallback_complete = bool(
        synthesis_telemetry.get("max_steps_fallback_complete")
    )
    output_truncated = bool(synthesis_telemetry.get("output_truncated"))
    generation_error_recovered = bool(
        synthesis_telemetry.get("generation_error_recovered")
    )
    generation_failure_present = bool(
        synthesis_telemetry.get("generation_failure_error")
    )
    synthesis_error_present = bool(synthesis_telemetry.get("synthesis_error"))
    generation_recovery_attempted = bool(
        synthesis_telemetry.get("generation_recovery_attempted")
    )
    generation_recovery_result = str(
        synthesis_telemetry.get("generation_recovery_result") or ""
    )
    generation_failure_unrecovered = bool(
        generation_failure_present and synthesis_error_present
    )
    generation_recovery_finish_reason = str(
        synthesis_telemetry.get("generation_recovery_finish_reason") or ""
    )
    same_memory_recovery_proven = bool(
        generation_error_recovered
        and generation_failure_present
        and not synthesis_error_present
        and generation_recovery_attempted
        and generation_recovery_result == "success"
        and generation_recovery_finish_reason == "stop"
    )
    raw_recovery = synthesis_telemetry.get("recovery")
    raw_recovery = raw_recovery if isinstance(raw_recovery, Mapping) else {}
    bounded_pass_recovery_proven = bool(
        generation_failure_present
        and not generation_error_recovered
        and not synthesis_error_present
        and raw_recovery.get("status") == "completed"
        and raw_recovery.get("selected_artifact") == "recovery"
    )
    generation_failure_resolved = bool(
        same_memory_recovery_proven or bounded_pass_recovery_proven
    )
    generation_recovery_unproven = bool(
        (generation_error_recovered and not same_memory_recovery_proven)
        or (
            generation_failure_present
            and not synthesis_error_present
            and not generation_failure_resolved
        )
    )
    quality_record = build_quality_record(synthesis_telemetry, report_word_count)
    critique = quality_record["critique"]
    critique_status = str(critique.get("status") or "unknown")
    critique_applied = critique.get("applied")
    critique_applied_count = (
        int(critique_applied) if isinstance(critique_applied, int) else None
    )
    critique_ratio_raw = critique.get("applied_ratio")
    critique_ratio = (
        float(critique_ratio_raw)
        if isinstance(critique_ratio_raw, (int, float))
        else None
    )
    quality_liveness = quality_record["liveness"]
    liveness_unknown = _coerce_int(quality_liveness.get("unknown"))
    retrieval_failures = quality_record["retrieval_failures"]
    retrieval_failure_total = retrieval_failures.get("total")
    recovery = quality_record["recovery"]
    recovery_outcome = str(recovery.get("outcome") or "unknown")
    context_truncation = quality_record["context_truncation"]

    quality_failures: list[str] = []
    coverage_failures: list[str] = []

    # --- CATASTROPHIC -> VALIDATION_FAILED ---
    catastrophic_reasons: list[str] = []
    if citations_extracted == 0 and (
        report_word_count >= MIN_REPORT_WORDS or generation_failure_resolved
    ):
        catastrophic_reasons.append(GATE_REASON_ZERO_CITATIONS)
        quality_failures.append(f"zero_citations_on_{report_word_count}_word_report")
    if generation_recovery_unproven:
        catastrophic_reasons.append(GATE_REASON_GENERATION_RECOVERY_UNPROVEN)
        quality_failures.append(GATE_REASON_GENERATION_RECOVERY_UNPROVEN)
    if generation_failure_unrecovered:
        catastrophic_reasons.append(GATE_REASON_GENERATION_FAILURE_UNRECOVERED)
        quality_failures.append(GATE_REASON_GENERATION_FAILURE_UNRECOVERED)
    if hit_max_steps and not self_terminated and (
        citations_extracted == 0
        or recovered_from_scratchpad
        or not max_steps_fallback_complete
    ):
        catastrophic_reasons.append(GATE_REASON_MAX_STEPS_NO_FINAL)
        quality_failures.append(GATE_REASON_MAX_STEPS_NO_FINAL)
    if output_truncated:
        catastrophic_reasons.append(GATE_REASON_OUTPUT_TRUNCATED)
        quality_failures.append(GATE_REASON_OUTPUT_TRUNCATED)
    if critique_status == "error":
        catastrophic_reasons.append(GATE_REASON_CRITIQUE_ERROR)
        quality_failures.append("critique_validation_error")
    # DSV-045 (2026-06-15): a deliverable-coverage shortfall no longer triggers
    # catastrophic VALIDATION_FAILED. completeness_check is a recall-biased
    # structural heuristic (its nested/misnested edge cases can over- or
    # under-count), so a coverage shortfall DEGRADES to a reviewable COMPLETE
    # rather than killing the report. Catastrophic stays for the UNAMBIGUOUS
    # signals only — zero citations on a non-trivial report, output truncation,
    # or critique execution failure. A live 3.5k-word, well-cited report was failed
    # 0/4 by a completeness false-MISSING under the old multi-deliverable
    # hard-shortfall trigger; this removes that class of catastrophe at the gate,
    # independent of the matcher's accuracy.

    scope_validation = synthesis_telemetry.get("authored_scope_validation") or {}
    if isinstance(scope_validation, Mapping) and scope_validation.get("violations"):
        catastrophic_reasons.append("authored_scope_violated")
        quality_failures.extend(str(item) for item in scope_validation["violations"])

    # S97 m04 — an authored-scope violation judges the persisted artifact's
    # conformance, not its evidence. When it is the only terminal reason, keep
    # evaluating so independent findings stay visible beside the failed verdict.
    scope_failure_only = catastrophic_reasons == ["authored_scope_violated"]
    if catastrophic_reasons and not scope_failure_only:
        return OutcomeGateResult(
            outcome=MissionOutcome.VALIDATION_FAILED,
            routing_reason=catastrophic_reasons[0],
            delivery_quality=DELIVERY_QUALITY_FAILED,
            quality_gate_failures=quality_failures,
            coverage_gate_failures=coverage_failures,
        )

    scope_warnings = list(scope_validation.get("warnings") or [])
    quality_failures.extend(scope_warnings)

    # --- DEGRADED -> COMPLETE + delivery_quality='degraded' ---
    # ALL positive shortfalls land here now (incl. broad multi-deliverable
    # non-delivery, which previously hard-failed) — flagged, not suppressed.
    if shortfall_score > 0.0:
        coverage_failures.append(
            f"deliverable_shortfall_{shortfall_score:.3f}_across_{deliverable_count}_deliverables"
        )
    if dead_rate >= SOFT_DEAD_RATE:
        quality_failures.append(f"dead_link_rate_{dead_rate:.3f}")
    if orphan >= SOFT_ORPHAN:
        quality_failures.append(f"orphan_citations_{orphan}")
    if (
        critique_ratio is not None
        and CRITIQUE_DEGRADED_RATIO <= critique_ratio < CRITIQUE_FAILED_RATIO
    ):
        quality_failures.append(
            f"material_critique_damage_ratio_{critique_ratio:.3f}_count_"
            f"{critique_applied_count or 0}"
        )
    # ``claim_count`` is declared by the critique model, not deterministically
    # derived from the report. Production replay showed identical 154-citation
    # artifacts receive denominators of 9 and 45 across two passes. Preserve
    # the high ratio as material review telemetry, but never grant that unstable
    # denominator terminal authority over an otherwise reviewable artifact.
    if (
        critique_applied_count is not None
        and critique_applied_count >= CRITIQUE_RATIO_MIN_APPLIED
        and critique_ratio is not None
        and critique_ratio >= CRITIQUE_FAILED_RATIO
    ):
        quality_failures.append(
            f"critique_applied_ratio_{critique_ratio:.3f}_count_"
            f"{critique_applied_count}"
        )
    if recovery_outcome in {"failed", "budget_exhausted"}:
        quality_failures.append(f"recovery_{recovery_outcome}")

    max_steps_fallback_used = hit_max_steps and not self_terminated
    # --- USABLE WITH WARNING -> COMPLETE + explicit review facts ---
    warnings: list[str] = []
    if max_steps_fallback_used:
        warnings.append(GATE_REASON_MAX_STEPS_NO_FINAL)
    if same_memory_recovery_proven:
        warnings.append(GATE_REASON_GENERATION_ERROR_RECOVERED)
    elif bounded_pass_recovery_proven:
        warnings.append(GATE_REASON_GENERATION_FAILURE_RECOVERED_BY_BOUNDED_PASS)
    if critique_status in {"unknown", "legacy_unstructured"}:
        warnings.append(f"critique_{critique_status}")
    claim_assessment = quality_record["claim_assessment"]
    if claim_assessment["version"] == 1:
        if any(claim_assessment[key] is None for key in (
            "assessed", "supported", "contradicted", "unassessed"
        )):
            warnings.append("critique_assessment_incomplete")
        contradicted = _coerce_int(claim_assessment["contradicted"])
        unassessed = _coerce_int(claim_assessment["unassessed"])
        if contradicted:
            quality_failures.append(f"critique_contradicted_{contradicted}")
        if unassessed:
            warnings.append(f"critique_unassessed_claims_{unassessed}")
        if not _coerce_int(claim_assessment["assessed"]):
            warnings.append("critique_assessment_unassessed")
    elif critique_status == "ok":
        warnings.append("critique_assessment_unknown")
    if critique_applied_count is not None and critique_applied_count > 0:
        warnings.append(f"critique_applied_{critique_applied_count}")
    if not bool(quality_liveness.get("available")):
        warnings.append("liveness_unknown_missing_telemetry")
    elif liveness_unknown > 0:
        warnings.append(f"liveness_unknown_{liveness_unknown}")
    dependencies = quality_record["evidence_dependencies"]
    if dependencies.get("version") == 1 and dependencies.get("available") is True:
        for key in ("missing_cited_sources", "cited_without_readable_body",
                    "truncated_cited_bodies", "unresolved_cited_fetch_failures"):
            count = _coerce_int(dependencies.get(key))
            if count:
                warnings.append(f"{key}_{count}")
        if dependencies.get("fetch_failure_attribution_complete") is not True:
            warnings.append("retrieval_dependency_attribution_unknown")
    elif not bool(retrieval_failures.get("available")):
        warnings.append("retrieval_failures_unknown")
    elif isinstance(retrieval_failure_total, int) and retrieval_failure_total > 0:
        # Legacy aggregate counts cannot prove whether failures were recovered.
        warnings.append("retrieval_failure_impact_unknown")
    if quality_record["citation_surface"]["actionable"]:
        warnings.append("citation_reference_mapping_inconsistent")
    for key in ("hallucinated_quarantined", "citations_unmappable", "markers_quarantined"):
        count = _coerce_int(synthesis_telemetry.get(key))
        if count:
            warnings.append(f"{key}_{count}")
    for key in ("source_projection", "citation_url_projection", "critique_url_projection"):
        projection = synthesis_telemetry.get(key)
        count = (
            _coerce_int(projection.get("dropped_count"))
            if isinstance(projection, Mapping) else 0
        )
        if count:
            warnings.append(f"{key}_dropped_{count}")
    if recovery_outcome == "completed":
        warnings.append("recovery_completed")
    elif recovery_outcome not in {"not_triggered", "failed", "budget_exhausted"}:
        warnings.append(f"recovery_{recovery_outcome}")
    # S90 (#957) — warn on EVIDENCE LOSS, not on any truncation. The grounding
    # packer drops uncited overflow by design, so keying this on ``occurred``
    # fired on essentially every source-rich mission and downgraded
    # delivery_quality from 'clean' for a non-event. ``evidence_loss_occurred``
    # is True only when the critique lost material it needed: a dropped CITED
    # source, or a truncated draft / liveness table. Falls back to ``occurred``
    # for v2 quality records that predate the split.
    truncation_signal = context_truncation.get("evidence_loss_occurred")
    if truncation_signal is None:
        truncation_signal = context_truncation.get("occurred")
    if truncation_signal is True:
        warnings.append("validation_context_truncated")

    if scope_failure_only:
        return OutcomeGateResult(
            outcome=MissionOutcome.VALIDATION_FAILED,
            routing_reason="authored_scope_violated",
            delivery_quality=DELIVERY_QUALITY_FAILED,
            quality_gate_failures=[*quality_failures, *warnings],
            coverage_gate_failures=coverage_failures,
        )

    if quality_failures or coverage_failures:
        return OutcomeGateResult(
            outcome=MissionOutcome.COMPLETE,
            routing_reason="authored_scope_partial" if scope_warnings else "",
            delivery_quality=DELIVERY_QUALITY_DEGRADED,
            quality_gate_failures=[*quality_failures, *warnings],
            coverage_gate_failures=coverage_failures,
        )

    if warnings:
        return OutcomeGateResult(
            outcome=MissionOutcome.COMPLETE,
            routing_reason="",
            delivery_quality=DELIVERY_QUALITY_USABLE_WITH_WARNING,
            quality_gate_failures=warnings,
            coverage_gate_failures=[],
        )

    # --- CLEAN -> COMPLETE + delivery_quality='clean' ---
    return OutcomeGateResult(
        outcome=MissionOutcome.COMPLETE,
        routing_reason="",
        delivery_quality=DELIVERY_QUALITY_CLEAN,
        quality_gate_failures=[],
        coverage_gate_failures=[],
    )


def build_final_outcome_payload(
    *,
    outcome: MissionOutcome | str,
    routing_reason: str,
    coverage_gate_failures: list[str],
    quality_gate_failures: list[str],
    structural_validation_failures: list[str] | None = None,
    validation_report: Mapping[str, Any] | None = None,
    entity_slot_admission_audit: list[dict[str, Any]] | None = None,
    delivery_quality: str | None = None,
    completed_at: datetime,
) -> FinalOutcomePayload:
    """Build the canonical final_outcome payload for telemetry and JSONB."""

    if isinstance(outcome, MissionOutcome):
        outcome_value = outcome.value
    else:
        outcome_value = str(outcome or "").strip().lower()

    payload: FinalOutcomePayload = {
        "outcome": outcome_value,
        "routing_reason": str(routing_reason or "").strip(),
        "coverage_gate_failures": list(coverage_gate_failures),
        "quality_gate_failures": list(quality_gate_failures),
        "structural_validation_failures": _normalize_string_list(
            structural_validation_failures
        ),
        "completed_at": completed_at.isoformat(),
    }
    # S82 m01 (PD136) — additive delivery-quality signal. Omitted when not set
    # (legacy / local-runner paths) so existing payload shapes are unchanged.
    normalized_delivery_quality = str(delivery_quality or "").strip()
    if normalized_delivery_quality:
        payload["delivery_quality"] = normalized_delivery_quality
    if isinstance(validation_report, Mapping) and validation_report:
        payload["validation_report"] = dict(validation_report)
    if entity_slot_admission_audit:
        payload["entity_slot_admission_audit"] = [
            dict(entry) for entry in entity_slot_admission_audit if isinstance(entry, Mapping)
        ]
    return payload


def route_summary_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the last recorded routing trace entry, if present."""

    telemetry = state.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return {}
    routing_trace = telemetry.get("routing_trace")
    if not isinstance(routing_trace, list) or not routing_trace:
        return {}
    summary = routing_trace[-1]
    return dict(summary) if isinstance(summary, Mapping) else {}


def resolve_mission_outcome(state: Mapping[str, Any]) -> MissionOutcome:
    """Resolve a mission outcome from state, falling back to routing data."""

    raw_outcome = str(state.get("mission_outcome") or "").strip().lower()
    if raw_outcome:
        try:
            return MissionOutcome(raw_outcome)
        except ValueError:
            pass

    summary = route_summary_from_state(state)
    routing_reason = str(
        state.get("routing_reason")
        or summary.get("reason")
        or ""
    ).strip()
    return map_reason_to_outcome(routing_reason, state.get("coverage_gates"))


def resolve_final_outcome_payload(
    state: Mapping[str, Any],
    *,
    completed_at: datetime,
) -> FinalOutcomePayload:
    """Derive the canonical final_outcome payload from state and telemetry."""

    summary = route_summary_from_state(state)
    telemetry = state.get("telemetry")
    normalized_telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    routing_reason = str(
        state.get("routing_reason")
        or summary.get("reason")
        or ""
    ).strip()
    # S82 m01 (PD136) — the worker-path outcome gate (evaluate_outcome_gate)
    # writes the populated failure lists + delivery_quality directly onto the
    # finalize state update. Prefer those when present; otherwise fall back to
    # the routing_trace summary / coverage coercion (legacy / local-runner
    # paths that never run the gate).
    state_coverage = state.get("coverage_gate_failures")
    state_quality = state.get("quality_gate_failures")

    coverage_failures = (
        state_coverage
        if isinstance(state_coverage, list)
        else summary.get("coverage_gate_failures")
    )
    quality_failures = (
        state_quality
        if isinstance(state_quality, list)
        else summary.get("quality_gate_failures")
    )

    if not isinstance(coverage_failures, list):
        coverage_failures = coerce_source_coverage_gate_result(
            state.get("coverage_gates")
        ).failed_gates
    if not isinstance(quality_failures, list):
        quality_failures = []
    structural_failures = _normalize_string_list(
        state.get("structural_validation_failures")
    )
    synthesis_telemetry = state.get("synthesis_telemetry")
    if not structural_failures:
        if isinstance(synthesis_telemetry, Mapping):
            structural_failures = _normalize_string_list(
                synthesis_telemetry.get("structural_validation_failed_schemas")
            )
    if not structural_failures:
        structural_failures = _normalize_string_list(
            normalized_telemetry.get("structural_validation_failures")
        )
    if (
        not quality_failures
        and (
            routing_reason == "structural_validation_failed"
            or structural_failures
        )
    ):
        quality_failures = ["synthesis_quality"]

    # S41.2 — surface the full ValidationReport so downstream auditors can
    # distinguish "validator fired and passed" from "validator did not run
    # in this build" without needing shell access to Railway. finalize_node
    # writes `validation_report.to_dict()` into synthesis_telemetry under
    # `structural_validation` (all results, not only failures).
    validation_report: dict[str, Any] | None = None
    if isinstance(synthesis_telemetry, Mapping):
        raw_report = synthesis_telemetry.get("structural_validation")
        if isinstance(raw_report, Mapping) and raw_report:
            validation_report = dict(raw_report)

    # S41.4 — surface entity_slot_admission_audit so the S40.1 empirical
    # gate is scoreable via MCP get_mission. The audit accumulates on state
    # across research loops (research_node.py merges each loop's entries),
    # and lives as List[Dict[str, Any]]. Pass through unchanged as a list
    # of per-slot-per-source records.
    raw_entity_audit = state.get("entity_slot_admission_audit")
    entity_slot_audit: list[dict[str, Any]] | None = None
    if isinstance(raw_entity_audit, list) and raw_entity_audit:
        entity_slot_audit = [
            dict(entry) for entry in raw_entity_audit if isinstance(entry, Mapping)
        ]

    # S82 m01 (PD136) — thread the gate's delivery_quality through to the
    # persisted payload (None on legacy / local-runner paths → omitted).
    raw_delivery_quality = state.get("delivery_quality")
    delivery_quality = (
        str(raw_delivery_quality).strip() if raw_delivery_quality else None
    )

    payload = build_final_outcome_payload(
        outcome=resolve_mission_outcome(state),
        routing_reason=routing_reason,
        coverage_gate_failures=[str(item) for item in coverage_failures],
        quality_gate_failures=[str(item) for item in quality_failures],
        structural_validation_failures=structural_failures,
        validation_report=validation_report,
        entity_slot_admission_audit=entity_slot_audit,
        delivery_quality=delivery_quality,
        completed_at=completed_at,
    )
    if isinstance(synthesis_telemetry, Mapping):
        scope_validation = synthesis_telemetry.get("authored_scope_validation")
        if isinstance(scope_validation, Mapping):
            payload["authored_scope_validation"] = dict(scope_validation)
    return payload


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


__all__ = [
    "DELIVERY_QUALITY_CLEAN",
    "DELIVERY_QUALITY_USABLE_WITH_WARNING",
    "DELIVERY_QUALITY_DEGRADED",
    "DELIVERY_QUALITY_FAILED",
    "CRITIQUE_DEGRADED_RATIO",
    "CRITIQUE_FAILED_RATIO",
    "FinalOutcomePayload",
    "HARD_SHORTFALL",
    "MIN_DELIVERABLES_FOR_HARD_SHORTFALL",
    "MIN_REPORT_WORDS",
    "MissionOutcome",
    "OutcomeGateResult",
    "SOFT_DEAD_RATE",
    "SOFT_ORPHAN",
    "build_final_outcome_payload",
    "evaluate_outcome_gate",
    "map_reason_to_outcome",
    "resolve_final_outcome_payload",
    "resolve_mission_outcome",
    "route_summary_from_state",
]
