"""S64 CodeAgent finalize node — entry-point synthesis node.

Per ``cmos/reports/sprint_64_approach_and_plan.md`` "Implementation seam":

- LangGraph stages 1–5 are bypassed at the call-graph level (the graph entry
  point is now ``codeagent_finalize_node`` itself; ``research_node``,
  ``cluster_findings``, ``reflection_node`` remain registered but unreachable
  on the synthesis path — kept for S72 deletion).
- The CodeAgent retrieves cold via the live tool surface
  (``web_search``/``source_fetch``/``url_liveness``/``scratchpad_*``) — no
  upstream admitted pool is harvested.
- Stage 10 (validators inside the legacy ``finalize_node``) is bypassed
  permissively — em-dash gap notes are intentional, not failures.

The node mirrors the post-finalize state contract that
``deepsearch.worker.main._handle_mission`` reads downstream (``report_path``,
``protocol_path``, ``mission_outcome``, ``routing_reason``,
``synthesis_telemetry``, ``telemetry``) so the worker code is unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from typing import AbstractSet, Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple
from urllib.parse import urlparse

from deepsearch.error_safety import (
    normalize_safe_error_text,
    safe_error_from_parts,
    safe_exception_text,
)
from deepsearch.llm.usage import compiler_usage_events, summarize_usage
from deepsearch.mission.scope import AuthoredScope, ScopeAdmission, resolve_authored_scope
from deepsearch.report.citation import (
    CitationAnnotation,
    CitationSpanRebaseError,
    rebase_citation_annotations,
)
from deepsearch.report.generator import (
    generate_markdown_report,
    prepare_synthesized_content,
    save_protocol,
    save_report,
)
from deepsearch.report.protocol import (
    REPORT_METADATA_SCHEMA_VERSION,
    MissionProtocolComplete,
    ReportMetadata,
    canonicalize_protocol_payload,
    generate_mission_protocol,
)
from deepsearch.report.scope_validation import _WHITESPACE as _REPORT_WHITESPACE
from deepsearch.report.scope_validation import (
    count_report_words,
    measured_text,
    validate_persisted_scope,
)
from deepsearch.report.text import canonicalize_report_text

from .citation_extraction import (
    _CITE_MARKER_RE,
    _SENTENCE_TERMINATORS,
    _UNICODE_SUPER_MARKER_RE,
    _UNICODE_SUPER_REFERENCE_ENTRY_RE,
    _build_references_map,
    _find_code_spans,
    _is_inside_span,
    _source_identity,
    _super_to_int,
    extract_citations,
    reads_superscripts,
)
from .citation_liveness import (
    CLASSIFICATION_DEAD,
    CLASSIFICATION_DEAD_UNCOLLECTED,
    LivenessTelemetry,
    _normalise_url_for_cache,
    build_source_url_set,
    check_citations_liveness,
    classify_citation,
    quarantine_dead_citations,
    source_named_dead_urls,
    strip_dead_cite_markers,
)
from .citation_surface import reader_markers, superscript_runs_ignored
from .codeagent_runner import (
    DEFAULT_MAX_STEPS,
    LEDGER_RECORD_MAX_ENTRIES,
    SMOLAGENTS_VERBOSITY_OFF,
    CodeAgentRunnerError,
    SynthesisResult,
    _canonical_ledger_url,
    _canonicalize_sources_collected,
    _validate_strict_ledger_record_envelope,
    run_codeagent_synthesis,
)
from .contract_to_prompt import (
    build_codeagent_prompts,
    resolve_deliverables,
    resolve_retrieval_policy,
)
from .critique import CritiqueAnnotation, CritiqueResult, run_critique
from .execution_control import (
    ExecutionCancelledError,
    current_execution_cancel_event,
    raise_if_execution_cancelled,
)
from .outcome import (
    DELIVERY_QUALITY_CLEAN,
    DELIVERY_QUALITY_FAILED,
    DELIVERY_QUALITY_USABLE_WITH_WARNING,
    MissionOutcome,
    evaluate_outcome_gate,
)
from .quality_series import build_evidence_dependencies, build_quality_record
from .recovery import (
    build_recovery_prompt,
    decide_recovery,
    evaluate_recovery_regression,
    remaining_recovery_steps,
)
from .state import AgentState
from .telemetry import clone_telemetry
from .tools.completeness_check import completeness_check

logger = logging.getLogger(__name__)

StateUpdate = Dict[str, Any]


_BLOCKED_REASON_RUNNER_ERROR = "codeagent_runner_error"
_BLOCKED_REASON_AGENT_ERROR = "codeagent_synthesis_failed"
_BLOCKED_REASON_EMPTY_OUTPUT = "codeagent_empty_output"
_BLOCKED_REASON_REPORT_CONSTRUCTION = "report_construction_failed"
_BLOCKED_REASON_PROTOCOL_CONSTRUCTION = "protocol_construction_failed"
_VALIDATION_REASON_PARTIAL_OUTPUT = "codeagent_partial_output_preserved"


# S81 m04 (PD125) — Context-Pressure Observability.
#
# Historical S81 baseline retained for artifact-reading callers. Runtime pressure
# uses SynthesisResult.effective_config and is unassessed when that is absent.
_DEEPSEEK_V4_PRO_MAX_CONTEXT = 1_000_000  # 1M token context window

# Threshold above which the latest measured input is considered "under
# pressure" AND used in combination with self_terminated=True and a measured
# final step to flag `self_term_under_pressure`. The 0.7 (70% of context
# window) value is observational, not load-bearing. Future analysis can
# re-bucket the threshold without code changes — the raw `pressure_pct` plus
# its measurement provenance are the durable signals.
_CONTEXT_PRESSURE_THRESHOLD = 0.7


def _latest_step_input_measurement(
    step_trace: list[dict[str, Any]],
) -> tuple[int, int | None, bool]:
    """Return the latest input measurement and its step provenance.

    The step trace is a list of dicts as serialised by ``_step_to_dict``
    in codeagent_runner.py. Each step optionally carries
    ``{"token_usage": {"input_tokens": int, ...}}``. A failed generation
    ActionStep has no usage; scanning backward keeps the last successful
    context visible without claiming it measured the failed request.

    Returns:
        ``(input_tokens, measurement_step, final_step_usage_available)``.
        ``measurement_step`` is ``None`` when no usable measurement exists.
    """

    final_trace_index = len(step_trace) - 1
    for trace_index in range(final_trace_index, -1, -1):
        step = step_trace[trace_index]
        if not isinstance(step, dict):
            continue
        token_usage = step.get("token_usage")
        if not isinstance(token_usage, dict):
            continue
        raw = token_usage.get("input_tokens")
        if not isinstance(raw, int) or isinstance(raw, bool):
            continue
        raw_measurement_step = step.get("step_number")
        if not isinstance(raw_measurement_step, int) or isinstance(
            raw_measurement_step, bool
        ):
            raw_measurement_step = step.get("step_index")
        measurement_step = (
            raw_measurement_step
            if isinstance(raw_measurement_step, int)
            and not isinstance(raw_measurement_step, bool)
            else trace_index
        )
        return raw, measurement_step, trace_index == final_trace_index
    return 0, None, False


def _last_step_input_tokens(step_trace: list[dict[str, Any]]) -> int:
    """Return the latest recorded input-token count, retaining compatibility."""

    input_tokens, _, _ = _latest_step_input_measurement(step_trace)
    return input_tokens


def _build_context_pressure(result: SynthesisResult) -> Dict[str, Any]:
    """Compute the S81 m04 context_pressure telemetry envelope.

    Pure-function over the SynthesisResult; no I/O. Pressure uses the latest
    recorded per-step input measurement. On a generation failure, the final
    ActionStep has no usage, so this is explicitly a prior-step measurement;
    it must not be interpreted as what the failed request saw. Cumulative input
    across all steps (already on ``token_usage.input``) is included only for
    convenience.
    """
    (
        last_step_input,
        measurement_step,
        final_step_usage_available,
    ) = _latest_step_input_measurement(result.step_trace)
    model_max = result.effective_config.get("context_window")
    if not isinstance(model_max, int) or model_max <= 0:
        model_max = None
    pressure_pct = round(last_step_input / model_max, 4) if model_max else None
    cumulative_input = int(result.token_usage.get("input") or 0)
    failed_step_input_unknown = bool(
        getattr(result, "generation_failure_error", None)
        and not final_step_usage_available
    )
    self_term_under_pressure = bool(
        result.self_terminated
        and final_step_usage_available
        and pressure_pct is not None
        and pressure_pct > _CONTEXT_PRESSURE_THRESHOLD
    )
    return {
        "model_max_context": model_max,
        "last_step_input_tokens": last_step_input,
        "measurement_step": measurement_step,
        "final_step_usage_available": final_step_usage_available,
        "failed_step_input_unknown": failed_step_input_unknown,
        "pressure_pct": pressure_pct,
        "self_term_under_pressure": self_term_under_pressure,
        "cumulative_input_tokens": cumulative_input,
        "step_count": int(result.step_count),
    }


def _build_generation_failure_telemetry(
    result: SynthesisResult,
) -> Dict[str, Any]:
    """Project one generation failure to a bounded, secret-safe envelope."""

    generation_failure_error = normalize_safe_error_text(
        getattr(result, "generation_failure_error", None),
        default_category="execution",
    )
    generation_failure_reason = str(
        getattr(result, "generation_failure_reason", "") or ""
    )
    if generation_failure_reason not in {
        "authentication",
        "balance",
        "connection",
        "context_length",
        "invalid_request",
        "other",
        "overloaded",
        "rate_limit",
        "server",
        "timeout",
    }:
        generation_failure_reason = ""
    generation_failure_reason_source = str(
        getattr(result, "generation_failure_reason_source", "unknown") or "unknown"
    )
    if generation_failure_reason_source not in {
        "class",
        "message_marker",
        "status",
        "unknown",
    }:
        generation_failure_reason_source = "unknown"
    raw_root_class = getattr(result, "generation_failure_root_class", "")
    generation_failure_root_class = (
        safe_error_from_parts("execution", raw_root_class).exception_class
        if generation_failure_error
        else ""
    )
    raw_http_status = getattr(result, "generation_failure_http_status", None)
    generation_failure_http_status = (
        raw_http_status
        if isinstance(raw_http_status, int)
        and not isinstance(raw_http_status, bool)
        and 100 <= raw_http_status <= 599
        else None
    )
    generation_failure_retry_suppressed_reason = str(
        getattr(
            result,
            "generation_failure_retry_suppressed_reason",
            "none",
        )
        or "none"
    )
    if generation_failure_retry_suppressed_reason not in {
        "attempts_exhausted",
        "deterministic",
        "none",
        "paid_progress",
    }:
        generation_failure_retry_suppressed_reason = "none"
    generation_recovery_result = str(
        getattr(result, "generation_recovery_result", "not_attempted")
        or "not_attempted"
    )
    if generation_recovery_result not in {
        "empty",
        "failed",
        "provider_failed",
        "invalid_wrapper",
        "not_attempted",
        "success",
        "truncated",
        "uncited",
        "unavailable",
        "unknown_finish",
    }:
        generation_recovery_result = "failed"
    generation_recovery_finish_reason = str(
        getattr(result, "generation_recovery_finish_reason", "") or ""
    )
    if generation_recovery_finish_reason not in {"", "length", "stop"}:
        generation_recovery_finish_reason = ""

    return {
        "generation_failure_error": generation_failure_error,
        "generation_failure_reason": generation_failure_reason,
        "generation_failure_reason_source": generation_failure_reason_source,
        "generation_failure_root_class": generation_failure_root_class,
        "generation_failure_http_status": generation_failure_http_status,
        "generation_failure_transient": bool(
            getattr(result, "generation_failure_transient", False)
        ),
        "generation_failure_step": max(
            0, int(getattr(result, "generation_failure_step", 0) or 0)
        ),
        "generation_failure_pass": max(
            0, int(getattr(result, "generation_failure_pass", 0) or 0)
        ),
        "generation_failure_pass_count": max(
            0, int(getattr(result, "generation_failure_pass_count", 0) or 0)
        ),
        "generation_failure_consumed_tokens": max(
            0,
            int(getattr(result, "generation_failure_consumed_tokens", 0) or 0),
        ),
        "generation_failure_retry_eligible": bool(
            getattr(result, "generation_failure_retry_eligible", False)
        ),
        "generation_failure_retry_suppressed_reason": (
            generation_failure_retry_suppressed_reason
        ),
        "generation_recovery_attempted": bool(
            getattr(result, "generation_recovery_attempted", False)
        ),
        "generation_recovery_result": generation_recovery_result,
        "generation_recovery_finish_reason": generation_recovery_finish_reason,
        "generation_error_recovered": bool(
            getattr(result, "generation_error_recovered", False)
        ),
    }


def _build_synthesis_telemetry(result: SynthesisResult) -> Dict[str, Any]:
    return {
        "engine": "codeagent",
        "authored_scope": dict(result.scope_telemetry),
        # S84 m04 — the model id actually bound for the run (SynthesisResult
        # threads it out of the runner); the locked default is only a
        # fallback for legacy fixtures that predate the field.
        "model_id": getattr(result, "model_id", "") or "deepseek-v4-pro",
        "framework": _framework_label(),
        "step_count": result.step_count,
        "self_terminated": result.self_terminated,
        "completion_control": dict(getattr(result, "completion_control", {}) or {}),
        "hit_max_steps": result.hit_max_steps,
        # S88 — final-answer generation cut at max_tokens (finish_reason=length);
        # the outcome gate routes it to VALIDATION_FAILED so a truncated report is
        # flagged/recovered, not silently shipped. getattr-defensive for legacy
        # SynthesisResult stubs that predate the field.
        "output_truncated": getattr(result, "output_truncated", False),
        # Gate-facing provenance describes only the selected artifact. Detailed
        # first/second-pass history lives under ``recovery``.
        "recovered_from_scratchpad": getattr(
            result, "recovered_from_scratchpad", False
        ),
        "output_wrapper_removed": bool(
            getattr(result, "output_wrapper_removed", False)
        ),
        "max_steps_fallback_complete": bool(
            getattr(result, "max_steps_fallback_complete", False)
        ),
        # Preserve safe evidence of an exception recovered from retained paid
        # memory. Raw provider messages never cross this boundary.
        **_build_generation_failure_telemetry(result),
        "runtime_seconds": result.runtime_seconds,
        "provider_attempts": list(result.provider_attempts),
        "token_usage": dict(result.token_usage),
        "token_usage_complete": bool(
            getattr(result, "token_usage_complete", False)
        ),
        "tool_call_summary": dict(result.tool_call_summary),
        # S84 m04 — per-call OUTCOME telemetry (status/error-category/result
        # counts/sentinel/circuit-open): a Linkup outage and a thin topic no
        # longer persist identically. Sibling of the pinned tool_call_summary.
        "tool_outcomes": dict(getattr(result, "tool_outcomes", {}) or {}),
        # S84 m04 — truth repair: this key had ZERO writers, so every
        # production summary read "N citations from 0 sources"
        # (_derive_summary reads it per its S81-m05 AST pins).
        "sources_collected": len(result.sources_collected or []),
        "retrieval_state": dict(getattr(result, "retrieval_state", {}) or {}),
        "recovery_passes": int(getattr(result, "recovery_passes", 0) or 0),
        "synthesis_error": normalize_safe_error_text(
            result.error,
            default_category="execution",
        ),
        # S90 (#959) — True when synthesis exhausted its retries on a transient,
        # zero-token generation failure. Rides synthesis_telemetry (an already
        # declared AgentState channel) rather than a new one: an undeclared
        # finalize field is silently dropped by LangGraph before the worker sees
        # it, which is exactly how 00b69f4 lost delivery_quality. The worker
        # reads this to requeue instead of landing the mission in 'blocked'.
        # getattr-defensive for legacy SynthesisResult stubs.
        "synthesis_error_transient": bool(
            getattr(result, "error_transient", False)
        ),
        # S81 m04 (PD125) — context-pressure observability. Pure telemetry;
        # does NOT gate dispatch, does NOT affect agent behavior. Consumed
        # by post-mission forensic analysis to distinguish "agent finished
        # honestly" from "agent self-terminated under context pressure".
        "context_pressure": _build_context_pressure(result),
        "effective_model": dict(result.effective_config),
    }


def _framework_label() -> str:
    """S84 m04 — derive the framework label from the installed package.

    The pre-m04 hardcoded "smolagents 1.24.0 CodeAgent" literal silently
    lied on any dependency bump; read the real version, fall back to a
    version-less label when metadata is unavailable.
    """
    try:
        from importlib.metadata import version

        return f"smolagents {version('smolagents')} CodeAgent"
    except Exception:  # noqa: BLE001 — metadata lookup must never break telemetry
        return "smolagents CodeAgent"


def _critique_telemetry_dict(result: CritiqueResult) -> Dict[str, Any]:
    return {
        "claim_count": result.claim_count,
        "gap_notes_inserted": result.gap_notes_inserted,
        "runtime_seconds": result.runtime_seconds,
        "provider_attempts": list(result.provider_attempts),
        "token_usage": dict(result.token_usage),
        "token_usage_complete": bool(result.token_usage_complete),
        "error": normalize_safe_error_text(
            result.error,
            default_category="execution",
        ),
        "grounding": dict(getattr(result, "grounding_telemetry", {}) or {}),
        "assessment": dict(result.assessment),
        "evidence_manifest": list(result.evidence_manifest),
        "annotation_summary": result.annotation_summary(),
        "annotations": [
            annotation.to_dict() for annotation in result.annotations
        ],
    }


@dataclass(slots=True)
class _ValidationPass:
    """One extraction → liveness → critique pass with no persistence."""

    stripped_markdown: str
    final_markdown: str
    citations: List[CitationAnnotation]
    critique_result: CritiqueResult
    critique_outer_error: str | None
    telemetry: Dict[str, Any]
    # S103 m02: whether the draft reads superscripts as reader markers, decided
    # once on the text extraction read and carried to every later audit.
    superscripts: bool


def _canonicalize_synthesis_sources(
    result: SynthesisResult,
) -> tuple[SynthesisResult, dict[str, Any]]:
    """Project the safe source subset without discarding paid synthesis.

    The strict Ledger helper remains the authority for one source record. A
    malformed record is quarantined; canonical-equivalent records that carry
    conflicting projection metadata collapse to one deterministic, richer
    record. Neither condition can invalidate the report body produced before
    this projection boundary.
    """

    reasons: Counter[str] = Counter()
    producer_projection = getattr(result, "source_collection_projection", {})
    producer_input_count = len(result.sources_collected or [])
    producer_dropped_count = 0
    producer_merged_count = 0
    if isinstance(producer_projection, Mapping):
        producer_input_count = int(
            producer_projection.get("input_records") or producer_input_count
        )
        producer_dropped_count = int(
            producer_projection.get("dropped_records") or 0
        )
        producer_merged_count = int(
            producer_projection.get("merged_records") or 0
        )
        producer_reasons = producer_projection.get("reason_counts")
        if isinstance(producer_reasons, Mapping):
            for reason, count in producer_reasons.items():
                reasons[str(reason)] += int(count or 0)
    producer_error = getattr(result, "source_collection_error", None)
    if producer_error and not reasons:
        reasons["producer_projection_rejected"] += 1
        producer_dropped_count = max(1, producer_dropped_count)

    canonical_sources: list[dict[str, Any]] = []
    index_by_url: dict[str, int] = {}
    for raw_source in result.sources_collected or []:
        try:
            canonical_source = _canonicalize_sources_collected([raw_source])[0]
        except (IndexError, TypeError, ValueError):
            reasons["invalid_source_record"] += 1
            continue

        url = str(canonical_source["url"])
        existing_index = index_by_url.get(url)
        if existing_index is None:
            index_by_url[url] = len(canonical_sources)
            canonical_sources.append(canonical_source)
            continue

        existing = canonical_sources[existing_index]
        try:
            canonical_sources[existing_index] = _canonicalize_sources_collected(
                [existing, canonical_source]
            )[0]
        except ValueError:
            reasons["canonical_metadata_conflict"] += 1
            existing_rank = (
                existing.get("content_quality") == "readable_body",
                max(
                    len(str(existing.get("body") or "")),
                    len(str(existing.get("snippet") or "")),
                ),
            )
            incoming_rank = (
                canonical_source.get("content_quality") == "readable_body",
                max(
                    len(str(canonical_source.get("body") or "")),
                    len(str(canonical_source.get("snippet") or "")),
                ),
            )
            if incoming_rank > existing_rank:
                canonical_sources[existing_index] = canonical_source
    tier_counts: Counter[str] = Counter(
        str(source.get("authority_tier") or "low")
        for source in canonical_sources
    )
    retrieval_state = dict(getattr(result, "retrieval_state", {}) or {})
    retrieval_state["grounding_sources"] = len(canonical_sources)
    projected = replace(
        result,
        error=normalize_safe_error_text(
            result.error,
            default_category="execution",
        ),
        sources_collected=canonical_sources,
        authority_tier_distribution={
            "high": int(tier_counts.get("high", 0)),
            "medium": int(tier_counts.get("medium", 0)),
            "low": int(tier_counts.get("low", 0)),
        },
        retrieval_state=retrieval_state,
    )
    local_drop_count = sum(reasons.values()) - producer_dropped_count
    return projected, {
        "input_count": producer_input_count,
        "kept_count": len(canonical_sources),
        "dropped_count": producer_dropped_count + max(0, local_drop_count),
        "merged_count": producer_merged_count,
        "reasons": dict(sorted(reasons.items())),
    }


def _canonicalize_citation_urls(
    annotations: Sequence[CitationAnnotation],
) -> tuple[list[CitationAnnotation], dict[str, Any]]:
    """Canonicalize citations and quarantine only unsafe URL annotations."""

    canonical: list[CitationAnnotation] = []
    invalid = 0
    for annotation in annotations:
        try:
            url = _canonical_ledger_url(annotation.url)
        except (TypeError, ValueError):
            invalid += 1
            continue
        canonical.append(annotation.model_copy(update={"url": url}))
    return canonical, {
        "input_count": len(annotations),
        "kept_count": len(canonical),
        "dropped_count": invalid,
        "reasons": {"invalid_url": invalid} if invalid else {},
    }


def _canonicalize_critique_urls(
    result: CritiqueResult,
) -> tuple[CritiqueResult, dict[str, Any]]:
    """Canonicalize critique URLs while quarantining only unsafe values."""

    annotations: list[CritiqueAnnotation] = []
    invalid = 0
    input_count = 0
    for annotation in result.annotations:
        urls: list[str] = []
        for raw_url in annotation.citation_urls:
            input_count += 1
            try:
                url = _canonical_ledger_url(raw_url)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if url not in urls:
                urls.append(url)
        annotations.append(replace(annotation, citation_urls=tuple(urls)))
    return (
        replace(
            result,
            error=normalize_safe_error_text(
                result.error,
                default_category="execution",
            ),
            annotations=annotations,
        ),
        {
            "input_count": input_count,
            "kept_count": input_count - invalid,
            "dropped_count": invalid,
            "reasons": {"invalid_url": invalid} if invalid else {},
        },
    )


async def _run_validation_pass(
    *,
    mission_id: str,
    draft_markdown: str,
    sources_collected: list[dict[str, Any]],
    mission_context: Mapping[str, Any],
    ledger_outcomes: dict[str, Any] | None = None,
    scope_admission: ScopeAdmission | None = None,
) -> _ValidationPass:
    """Run the canonical pre-persistence validation sequence once."""

    raise_if_execution_cancelled()
    # Actor text is canonicalized exactly once before citation extraction
    # assigns absolute codepoint coordinates. Every later persistence seam is
    # strict and may not move those coordinates silently.
    draft_markdown = canonicalize_report_text(draft_markdown)
    # S101 m02: a kept inline citation shows the reader number its References
    # entry gives it. Every coordinate, hash and audit below reads this draft.
    draft_markdown, reader_number_rendering, rendered_markers = _render_reader_numbers(
        draft_markdown
    )
    pass_telemetry: Dict[str, Any] = {"reader_number_rendering": reader_number_rendering}
    if scope_admission is not None:
        sources_collected = scope_admission.filter_sources(sources_collected, "pre_critique")
    superscripts = reads_superscripts(draft_markdown)
    stripped_markdown, citations = _extract_and_resolve_citations(
        draft_markdown,
        sources_collected=sources_collected,
        synthesis_telemetry=pass_telemetry,
        rendered_markers=rendered_markers,
    )
    citations, liveness = await _run_liveness_check(
        mission_id=mission_id,
        annotations=citations,
        sources_collected=sources_collected,
        scope_admission=scope_admission,
    )
    if ledger_outcomes is not None:
        _accumulate_ledger_outcomes(
            ledger_outcomes,
            {
                "ledger_records": list(liveness.ledger_records),
                "ledger_records_truncated": int(
                    liveness.ledger_records_truncated
                ),
            },
        )
    raise_if_execution_cancelled()
    pass_telemetry["liveness_telemetry"] = liveness.to_dict()
    # S87-m07: offload the whole-document critique off the event-loop thread.
    # run_critique is a sync, minutes-long LLM call; running it inline froze the
    # loop and starved the worker lease heartbeat (CAT-R001-S87 lease-loss
    # incident). This mirrors the synthesis offload above.
    critique_result, outer_error = await asyncio.to_thread(
        _invoke_critique,
        mission_id=mission_id,
        draft_markdown=stripped_markdown,
        sources_collected=sources_collected,
        mission_context=mission_context,
        citations=citations,
        scope_admission=scope_admission,
    )
    critique_result, critique_url_projection = _canonicalize_critique_urls(
        critique_result
    )
    pass_telemetry["critique_url_projection"] = critique_url_projection
    raise_if_execution_cancelled()
    critique_skipped = bool(critique_result.error or outer_error)
    final_markdown = canonicalize_report_text(
        stripped_markdown
        if critique_skipped
        else critique_result.output_markdown or stripped_markdown
    )
    pass_telemetry["critique_telemetry"] = _critique_telemetry_dict(
        critique_result
    )
    pass_telemetry["critique_skipped"] = critique_skipped
    if critique_skipped:
        pass_telemetry["critique_skip_reason"] = outer_error or critique_result.error
    return _ValidationPass(
        stripped_markdown=stripped_markdown,
        final_markdown=final_markdown,
        citations=citations,
        critique_result=critique_result,
        critique_outer_error=outer_error,
        telemetry=pass_telemetry,
        superscripts=superscripts,
    )


def _merge_sources_for_recovery(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge recovery sources by URL without degrading richer content."""

    merged: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for source in [*first, *second]:
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        incoming = dict(source)
        if url not in index:
            index[url] = len(merged)
            merged.append(incoming)
            continue
        existing = merged[index[url]]
        existing_quality = 2 if existing.get("content_quality") == "readable_body" else 1
        incoming_quality = 2 if incoming.get("content_quality") == "readable_body" else 1
        existing_chars = max(
            len(str(existing.get("body") or "")),
            len(str(existing.get("snippet") or "")),
        )
        incoming_chars = max(
            len(str(incoming.get("body") or "")),
            len(str(incoming.get("snippet") or "")),
        )
        content_upgrade = (incoming_quality, incoming_chars) >= (
            existing_quality,
            existing_chars,
        )
        for key, value in incoming.items():
            if value is None or value == "":
                continue
            if key in {"body", "snippet", "body_source", "content_quality"}:
                if not content_upgrade:
                    continue
            if key == "authority_tier" and existing.get("score") is not None:
                if incoming.get("score") is None:
                    continue
            existing[key] = value
    return merged


def _merge_tool_summaries(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    by_tool: Counter[str] = Counter()
    for summary in (first, second):
        raw = summary.get("by_tool")
        if isinstance(raw, Mapping):
            by_tool.update(
                {
                    str(key): int(value)
                    for key, value in raw.items()
                    if isinstance(value, int)
                }
            )
    scratchpad_key_set: set[str] = set()
    for summary in (first, second):
        raw_keys = summary.get("scratchpad_keys")
        if isinstance(raw_keys, Sequence) and not isinstance(raw_keys, (str, bytes)):
            scratchpad_key_set.update(str(key) for key in raw_keys)
    scratchpad_keys = sorted(scratchpad_key_set)
    return {
        "total_calls": sum(by_tool.values()),
        "by_tool": dict(by_tool),
        "scratchpad_keys": scratchpad_keys,
    }


def _merge_tool_outcomes(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    max_records: int = 250,
) -> dict[str, Any]:
    by_tool: dict[str, dict[str, Any]] = {}
    for outcomes in (first, second):
        raw_by_tool = outcomes.get("by_tool")
        if not isinstance(raw_by_tool, Mapping):
            continue
        for tool_name, raw_entry in raw_by_tool.items():
            if not isinstance(raw_entry, Mapping):
                continue
            target = by_tool.setdefault(
                str(tool_name),
                {
                    "calls": 0,
                    "ok": 0,
                    "error": 0,
                    "results_total": 0,
                    "circuit_open": 0,
                    "by_category": {},
                },
            )
            for key in ("calls", "ok", "error", "results_total", "circuit_open"):
                target[key] += int(raw_entry.get(key) or 0)
            raw_categories = raw_entry.get("by_category")
            if isinstance(raw_categories, Mapping):
                for category, count in raw_categories.items():
                    name = str(category)
                    target["by_category"][name] = (
                        target["by_category"].get(name, 0) + int(count or 0)
                    )
    all_records: list[dict[str, Any]] = []
    for outcomes in (first, second):
        raw_records = outcomes.get("records")
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records,
            (str, bytes),
        ):
            continue
        all_records.extend(
            dict(record) for record in raw_records if isinstance(record, Mapping)
        )
    prior_truncated = sum(
        int(outcomes.get("records_truncated") or 0)
        for outcomes in (first, second)
    )
    ledger = _merge_ledger_outcomes(first, second)
    return {
        "by_tool": by_tool,
        "records": all_records[:max_records],
        "records_truncated": prior_truncated + max(0, len(all_records) - max_records),
        **ledger,
    }


def _merge_ledger_outcomes(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    max_records: int = LEDGER_RECORD_MAX_ENTRIES,
) -> dict[str, Any]:
    """Concatenate every relevant attempt without generic sampling or dedupe.

    First-pass order precedes recovery-pass order.  Exact duplicates remain in
    the audit because they are distinct attempts; TraceLab owns deterministic
    projected-item sorting and exact deduplication.  Malformed rows and
    over-budget rows increment the loud truncation counter instead of being
    silently discarded.
    """

    records: list[dict[str, Any]] = []
    truncated = 0
    for outcomes in (first, second):
        if "ledger_records_truncated" not in outcomes:
            truncated += 1
        raw_truncated = outcomes.get("ledger_records_truncated")
        if isinstance(raw_truncated, int) and not isinstance(raw_truncated, bool):
            truncated += max(0, raw_truncated)
        elif "ledger_records_truncated" in outcomes:
            truncated += 1
        if "ledger_records" not in outcomes:
            truncated += 1
            continue
        raw_records = outcomes.get("ledger_records")
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records,
            (str, bytes),
        ):
            truncated += 1
            continue
        for record in raw_records:
            if not isinstance(record, Mapping):
                truncated += 1
                continue
            if len(records) >= max_records:
                truncated += 1
                continue
            try:
                records.append(_validate_strict_ledger_record_envelope(record))
            except (TypeError, ValueError):
                truncated += 1
    return {
        "ledger_records": records,
        "ledger_records_truncated": truncated,
    }


def _accumulate_ledger_outcomes(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
) -> None:
    """Append one producer/pass envelope into a shared lossless accumulator."""

    merged = _merge_ledger_outcomes(target, incoming)
    target.clear()
    target.update(merged)


def _install_ledger_outcomes(
    synthesis_telemetry: dict[str, Any],
    ledger_outcomes: Mapping[str, Any],
) -> None:
    """Install the cumulative audit without disturbing generic tool telemetry."""

    raw_tool_outcomes = synthesis_telemetry.get("tool_outcomes")
    tool_outcomes = (
        dict(raw_tool_outcomes)
        if isinstance(raw_tool_outcomes, Mapping)
        else {}
    )
    tool_outcomes.update(ledger_outcomes)
    synthesis_telemetry["tool_outcomes"] = tool_outcomes


def _ledger_projection_upper_bound(
    *,
    source_count: int,
    citation_count: int,
    synthesis_telemetry: Mapping[str, Any],
) -> int:
    """Return a conservative bound for TraceLab's total projected entries.

    TraceLab can collapse cited sources, enrich rejected sources with tool
    failures, and exact-dedupe projected items.  Counting every input as a
    distinct item therefore safely overestimates without reimplementing its
    server-owned semantic projection.
    """

    tool_outcomes = synthesis_telemetry.get("tool_outcomes")
    ledger_count = 0
    ledger_truncated = 0
    if isinstance(tool_outcomes, Mapping):
        raw_records = tool_outcomes.get("ledger_records")
        if isinstance(raw_records, Sequence) and not isinstance(
            raw_records,
            (str, bytes),
        ):
            ledger_count = len(raw_records)
        raw_truncated = tool_outcomes.get("ledger_records_truncated")
        if isinstance(raw_truncated, int) and not isinstance(raw_truncated, bool):
            ledger_truncated = max(0, raw_truncated)

    critique_count = 0
    critique_telemetry = synthesis_telemetry.get("critique_telemetry")
    if isinstance(critique_telemetry, Mapping):
        raw_annotations = critique_telemetry.get("annotations")
        if isinstance(raw_annotations, Sequence) and not isinstance(
            raw_annotations,
            (str, bytes),
        ):
            for annotation in raw_annotations:
                if not isinstance(annotation, Mapping):
                    continue
                if annotation.get("applied") is not True or annotation.get(
                    "verdict"
                ) not in {"unsupported", "hallucinated"}:
                    continue
                raw_urls = annotation.get("citation_urls")
                if isinstance(raw_urls, Sequence) and not isinstance(
                    raw_urls,
                    (str, bytes),
                ):
                    critique_count += len(raw_urls)

    return (
        max(0, source_count)
        + max(0, citation_count)
        + critique_count
        + ledger_count
        + ledger_truncated
    )


def _enforce_ledger_projection_budget(
    *,
    source_count: int,
    citation_count: int,
    synthesis_telemetry: dict[str, Any],
) -> int:
    """Make any unprovable >1,000-entry handoff fail loudly downstream."""

    # The deployed 043 contract has no legacy fallback: missing active-result
    # keys are unknown completeness, not an implicit empty audit.  Normalize
    # only to a loud nonzero-truncation envelope so downstream persistence can
    # retain a typed diagnostic without ever presenting it as capture-ready.
    raw_outcomes = synthesis_telemetry.get("tool_outcomes")
    outcomes = dict(raw_outcomes) if isinstance(raw_outcomes, Mapping) else {}
    envelope_errors = int(not isinstance(raw_outcomes, Mapping))
    raw_records = outcomes.get("ledger_records")
    if not isinstance(raw_records, list):
        outcomes["ledger_records"] = []
        envelope_errors += 1
    raw_truncated = outcomes.get("ledger_records_truncated")
    if (
        not isinstance(raw_truncated, int)
        or isinstance(raw_truncated, bool)
        or raw_truncated < 0
    ):
        raw_truncated = 0
        envelope_errors += 1
    if envelope_errors:
        outcomes["ledger_records_truncated"] = raw_truncated + envelope_errors
        synthesis_telemetry["tool_outcomes"] = outcomes

    upper_bound = _ledger_projection_upper_bound(
        source_count=source_count,
        citation_count=citation_count,
        synthesis_telemetry=synthesis_telemetry,
    )
    if upper_bound <= LEDGER_RECORD_MAX_ENTRIES:
        return upper_bound

    raw_outcomes = synthesis_telemetry.get("tool_outcomes")
    outcomes = dict(raw_outcomes) if isinstance(raw_outcomes, Mapping) else {}
    raw_truncated = outcomes.get("ledger_records_truncated", 0)
    prior_truncated = (
        raw_truncated
        if isinstance(raw_truncated, int)
        and not isinstance(raw_truncated, bool)
        and raw_truncated >= 0
        else 1
    )
    outcomes["ledger_records_truncated"] = prior_truncated + (
        upper_bound - LEDGER_RECORD_MAX_ENTRIES
    )
    synthesis_telemetry["tool_outcomes"] = outcomes
    return upper_bound


def _combine_synthesis_results(
    first: SynthesisResult,
    recovery: SynthesisResult,
) -> SynthesisResult:
    """Combine two bounded passes into the selected recovery result.

    Usage and retrieval facts are cumulative across both attempts. Termination
    flags are not: the outcome gate grades the selected artifact, so those
    flags must describe the recovery pass that produced it. Per-attempt flags
    remain available in the recovery audit telemetry assembled by the caller.
    """

    sources = _merge_sources_for_recovery(
        first.sources_collected,
        recovery.sources_collected,
    )
    discovered = _merge_sources_for_recovery(
        getattr(first, "sources_discovered", []),
        getattr(recovery, "sources_discovered", []),
    )
    fetched = _merge_sources_for_recovery(
        getattr(first, "sources_fetched", []),
        getattr(recovery, "sources_fetched", []),
    )
    tier_counts: Counter[str] = Counter(
        str(source.get("authority_tier") or "low") for source in sources
    )
    token_usage = {
        key: int(first.token_usage.get(key) or 0)
        + int(recovery.token_usage.get(key) or 0)
        for key in ("input", "output", "total")
    }
    first_attempts = list(getattr(first, "fetch_attempts", []))
    recovery_attempts = list(getattr(recovery, "fetch_attempts", []))
    first_probes = list(getattr(first, "probes", []))
    recovery_probes = list(getattr(recovery, "probes", []))
    failure_source = (
        recovery
        if getattr(recovery, "generation_failure_error", None)
        else first
    )
    failure_pass = int(
        bool(getattr(recovery, "generation_failure_error", None))
    )
    failure_pass_count = max(
        0, int(getattr(first, "generation_failure_pass_count", 0) or 0)
    ) + max(
        0, int(getattr(recovery, "generation_failure_pass_count", 0) or 0)
    )
    return replace(
        recovery,
        step_count=first.step_count + recovery.step_count,
        self_terminated=bool(recovery.self_terminated),
        hit_max_steps=bool(recovery.hit_max_steps),
        output_truncated=bool(getattr(recovery, "output_truncated", False)),
        runtime_seconds=round(first.runtime_seconds + recovery.runtime_seconds, 2),
        token_usage=token_usage,
        provider_attempts=[*first.provider_attempts, *recovery.provider_attempts],
        token_usage_complete=bool(
            first.token_usage_complete and recovery.token_usage_complete
        ),
        tool_call_summary=_merge_tool_summaries(
            first.tool_call_summary,
            recovery.tool_call_summary,
        ),
        authority_tier_distribution={
            "high": int(tier_counts.get("high", 0)),
            "medium": int(tier_counts.get("medium", 0)),
            "low": int(tier_counts.get("low", 0)),
        },
        step_trace=[
            *[{**step, "recovery_pass": 0} for step in first.step_trace],
            *[{**step, "recovery_pass": 1} for step in recovery.step_trace],
        ],
        sources_collected=sources,
        sources_discovered=discovered,
        sources_fetched=fetched,
        fetch_attempts=[*first_attempts, *recovery_attempts],
        probes=[*first_probes, *recovery_probes],
        retrieval_state={
            "discovered": len(discovered),
            "fetched_readable": len(fetched),
            "fetch_attempts": len(first_attempts) + len(recovery_attempts),
            "probes": len(first_probes) + len(recovery_probes),
            "grounding_sources": len(sources),
        },
        # Gate-facing artifact flags describe the selected recovery only. The
        # first-pass salvage remains in recovery.first_pass audit telemetry.
        recovered_from_scratchpad=bool(recovery.recovered_from_scratchpad),
        tool_outcomes=_merge_tool_outcomes(
            getattr(first, "tool_outcomes", {}) or {},
            getattr(recovery, "tool_outcomes", {}) or {},
        ),
        recovery_passes=1,
        generation_failure_error=getattr(
            failure_source, "generation_failure_error", None
        ),
        generation_failure_reason=str(
            getattr(failure_source, "generation_failure_reason", "") or ""
        ),
        generation_failure_reason_source=str(
            getattr(failure_source, "generation_failure_reason_source", "unknown")
            or "unknown"
        ),
        generation_failure_root_class=str(
            getattr(failure_source, "generation_failure_root_class", "Exception")
            or "Exception"
        ),
        generation_failure_http_status=getattr(
            failure_source, "generation_failure_http_status", None
        ),
        generation_failure_transient=bool(
            getattr(failure_source, "generation_failure_transient", False)
        ),
        generation_failure_step=int(
            getattr(failure_source, "generation_failure_step", 0) or 0
        ),
        generation_failure_pass=failure_pass,
        generation_failure_pass_count=failure_pass_count,
        generation_failure_consumed_tokens=int(
            getattr(failure_source, "generation_failure_consumed_tokens", 0) or 0
        ),
        generation_failure_retry_eligible=bool(
            getattr(failure_source, "generation_failure_retry_eligible", False)
        ),
        generation_failure_retry_suppressed_reason=str(
            getattr(
                failure_source,
                "generation_failure_retry_suppressed_reason",
                "none",
            )
            or "none"
        ),
        generation_recovery_attempted=bool(
            getattr(failure_source, "generation_recovery_attempted", False)
        ),
        generation_recovery_result=str(
            getattr(failure_source, "generation_recovery_result", "not_attempted")
            or "not_attempted"
        ),
        generation_recovery_finish_reason=str(
            getattr(failure_source, "generation_recovery_finish_reason", "") or ""
        ),
        generation_error_recovered=bool(
            getattr(first, "generation_error_recovered", False)
            or getattr(recovery, "generation_error_recovered", False)
        ),
    )


def _sum_token_usage(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Sum provider usage envelopes without inventing unavailable usage."""

    return {
        key: sum(int(record.get(key) or 0) for record in records)
        for key in ("input", "output", "total")
    }


def _build_attempt_accounting(
    synthesis_results: Sequence[SynthesisResult],
    critique_results: Sequence[CritiqueResult],
    *,
    token_usage_complete: bool,
    compiler_events: Sequence[dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Account for every paid pass, independent of artifact selection."""

    synthesis_usage = _sum_token_usage(
        [dict(result.token_usage or {}) for result in synthesis_results]
    )
    critique_usage = _sum_token_usage(
        [dict(result.token_usage or {}) for result in critique_results]
    )
    generation_usage_complete = all(
        bool(getattr(result, "token_usage_complete", False))
        and not getattr(result, "generation_failure_error", None)
        and not result.error
        for result in synthesis_results
    )
    critique_usage_complete = all(
        bool(result.token_usage_complete) for result in critique_results
    )
    events = list(compiler_events)
    for pass_index, synthesis in enumerate(synthesis_results, 1):
        events.extend({**e, "pipeline_pass": pass_index} for e in synthesis.provider_attempts)
    for pass_index, critique in enumerate(critique_results, 1):
        events.extend({**e, "pipeline_pass": pass_index} for e in critique.provider_attempts)
    model_accounting = summarize_usage(events)
    model_accounting["observation_coverage"] = (
        "complete" if all(r.provider_attempts for r in synthesis_results)
        and all(r.provider_attempts or r.token_usage.get("total") == 0 for r in critique_results)
        else "unknown_legacy_or_injected_model"
    )
    if model_accounting["observation_coverage"] != "complete":
        model_accounting["usage_complete"] = False
    model_accounting["compiler_observation"] = (
        "observed" if compiler_events else "absent_not_proof_of_no_call"
    )
    model_accounting["legacy_synthesis_critique_total"] = (
        synthesis_usage["total"] + critique_usage["total"]
    )
    model_accounting["compiler_observed_total"] = (
        summarize_usage(list(compiler_events))["observed"]["total"]
    )
    model_accounting["reconciliation_delta"] = (
        model_accounting["observed"]["total"] - model_accounting["compiler_observed_total"]
        - model_accounting["legacy_synthesis_critique_total"]
    )
    return {
        "schema_version": 1,
        "provider_attempts": events,
        "model_accounting": model_accounting,
        "retrieval_accounting": {
            "usd": None, "status": "provider_fees_not_observed",
            "by_pass": [r.tool_outcomes.get("by_tool", {}) for r in synthesis_results],
        },
        "human_repair_minutes": None,
        "synthesis_passes": len(synthesis_results),
        "critique_passes": len(critique_results),
        "steps_used": sum(int(result.step_count or 0) for result in synthesis_results),
        "synthesis_runtime_seconds": round(
            sum(float(result.runtime_seconds or 0.0) for result in synthesis_results),
            3,
        ),
        "critique_runtime_seconds": round(
            sum(float(result.runtime_seconds or 0.0) for result in critique_results),
            3,
        ),
        "synthesis_token_usage": synthesis_usage,
        "critique_token_usage": critique_usage,
        "synthesis_token_usage_complete": generation_usage_complete,
        "critique_token_usage_complete": critique_usage_complete,
        "token_usage": {
            key: synthesis_usage[key] + critique_usage[key]
            for key in ("input", "output", "total")
        },
        # Provider failures do not always return usage. The observed total is
        # retained as a lower bound; completeness is explicit, never guessed.
        "token_usage_complete": bool(
            token_usage_complete
            and generation_usage_complete
            and critique_usage_complete
        ),
    }


def _count_live_citations(citations: Sequence[CitationAnnotation]) -> int:
    """Count citations classified live (probe-confirmed or body-trusted)."""

    return sum(1 for annotation in citations if annotation.live is True)


async def codeagent_finalize_node(state: AgentState) -> StateUpdate:
    """Synthesize the final report via the S64 CodeAgent and persist artifacts.

    Returns the same downstream-readable state-update shape that the legacy
    ``finalize_node`` returned: ``report_path``, ``protocol_path``,
    ``mission_outcome``, ``routing_reason``, ``synthesis_telemetry``,
    ``telemetry``. On hard error: ``mission_outcome=BLOCKED`` with a structured
    ``routing_reason``; the worker handles BLOCKED via ``mark_failed``.
    """

    cancel_event = current_execution_cancel_event()
    raise_if_execution_cancelled(cancel_event)
    mission_id = str(state.get("mission_id") or "MISSION")
    context = (
        state.get("mission_context")
        if isinstance(state.get("mission_context"), Mapping)
        else {}
    )
    # S80 m02 (PD107): wrapper H1 + local report filename read the canonical
    # state["mission_title"] pinned once in empty_state(). The legacy
    # title-or-objective-or-mission_id resolution diverged from the path used
    # by protocol._derive_title and produced mismatched strings for the same
    # execution; reading the single state key collapses both paths.
    title = str(state.get("mission_title") or mission_id)
    # S80 m04 (PD109): TraceLab system-assigned UUID. Threaded into save_report
    # / save_protocol so re-runs of the same mission_id slug land in distinct
    # files on disk. None on the local YAML/API path; persistence layer falls
    # back to a random 4-byte hex suffix.
    mission_uuid = state.get("mission_uuid") or None
    telemetry = clone_telemetry(state)

    result: StateUpdate = {
        "telemetry": telemetry,
    }

    try:
        scope_admission = ScopeAdmission(resolve_authored_scope(state))
        system_instructions, user_directive = build_codeagent_prompts(state)
    except (ValueError, TypeError) as exc:
        safe_error = safe_exception_text(exc, category="validation")
        logger.error(
            "CodeAgent finalize: prompt construction failed for %s: %s",
            mission_id,
            safe_error,
        )
        return _block_result(
            result,
            telemetry=telemetry,
            reason=_BLOCKED_REASON_RUNNER_ERROR,
            synthesis_telemetry={
                "engine": "codeagent",
                "synthesis_skipped": True,
                "synthesis_skip_reason": _BLOCKED_REASON_RUNNER_ERROR,
                "synthesis_error": safe_error,
            },
        )

    logger.info(
        "CodeAgent finalize: starting synthesis for mission %s (live retrieval)",
        mission_id,
    )

    deliverable_spec = resolve_deliverables(state)
    excluded_domains, high_authority_domains = resolve_retrieval_policy(state)
    synthesis_kwargs: dict[str, Any] = {
        "system_instructions": system_instructions,
        "user_directive": user_directive,
        "verbosity": SMOLAGENTS_VERBOSITY_OFF,
        "deliverable_spec": deliverable_spec,
        "excluded_domains": excluded_domains,
        "high_authority_domains": high_authority_domains,
        "scope_admission": scope_admission,
    }
    scope = scope_admission.scope
    if scope.min_words is not None or scope.max_words is not None:
        synthesis_kwargs["report_word_budget_check"] = partial(
            _preview_report_word_budget, report_state=state.copy(), scope=scope,
        )
        system_instructions += (
            "\n\n# Persisted report word budget\n"
            "Authored word bounds override the generic per-section length targets. "
            "Use compact mission sections. Before final_answer, call completeness_check "
            "on the exact draft and inspect report_word_budget: it estimates the full "
            "rendered report, including title, metadata, objectives and context footer, "
            "after machine citation markers and the duplicate title are removed. "
            "Shorten or expand the draft as indicated, then check the revised draft. "
            "Leave room for later critique edits; final persisted validation still applies. "
            "Do this within the current step budget; do not start new research.\n"
        )
        synthesis_kwargs["system_instructions"] = system_instructions
    if cancel_event is not None:
        synthesis_kwargs["cancel_event"] = cancel_event

    try:
        synthesis_result = await asyncio.to_thread(
            run_codeagent_synthesis,
            **synthesis_kwargs,
        )
    except CodeAgentRunnerError as exc:
        safe_error = safe_exception_text(exc, category="execution")
        logger.error(
            "CodeAgent finalize: runner error for %s: %s",
            mission_id,
            safe_error,
        )
        return _block_result(
            result,
            telemetry=telemetry,
            reason=_BLOCKED_REASON_RUNNER_ERROR,
            synthesis_telemetry={
                "engine": "codeagent",
                "synthesis_skipped": True,
                "synthesis_skip_reason": _BLOCKED_REASON_RUNNER_ERROR,
                "synthesis_error": safe_error,
            },
        )
    raise_if_execution_cancelled(cancel_event)

    synthesis_result, source_projection = _canonicalize_synthesis_sources(
        synthesis_result
    )
    synthesis_result = replace(
        synthesis_result,
        output_markdown=canonicalize_report_text(
            synthesis_result.output_markdown
        ),
    )

    synthesis_telemetry = _build_synthesis_telemetry(synthesis_result)
    synthesis_telemetry["scope_compilation"] = {
        "contract_id": state.get("mission_contract_id"),
        "contract_schema_version": (state.get("mission_contract") or {}).get(
            "contract_schema_version"
        ),
        "compiler_revision": (state.get("mission_contract") or {}).get("compiler_revision"),
        "contract_sha256": _protocol_identity_sha256(state.get("mission_contract") or {}),
        "authoring_context_sha256": _protocol_identity_sha256(context or {}),
        "system_prompt_sha256": hashlib.sha256(system_instructions.encode()).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_directive.encode()).hexdigest(),
    }
    scope = scope_admission.scope
    bounded_scope = (scope.restriction != "unrestricted" or scope.max_sources is not None
                     or scope.min_words is not None or scope.max_words is not None)
    original_synthesis = synthesis_result.output_markdown
    if bounded_scope:
        synthesis_telemetry["original_synthesis"] = {
            **measured_text(original_synthesis), "text": original_synthesis,
            "surface": "canonical_actor_output_before_critique",
        }
    synthesis_telemetry["source_projection"] = source_projection
    if source_projection["dropped_count"]:
        logger.warning(
            "CodeAgent finalize: quarantined %d unsafe source projection(s) "
            "for %s (reasons=%s)",
            source_projection["dropped_count"],
            mission_id,
            source_projection["reasons"],
        )

    # S82 m04 (PD135) — deliverable_coverage is now computed on the FINAL
    # post-critique markdown below (after the critique pass), not on the raw
    # pre-critique synthesis output. Recomputing on what actually ships keeps
    # the signal the S82 m01 gate reads honest: critique can drop content (gap
    # notes replacing dropped claims), so pre-critique coverage would overcount.

    if not synthesis_result.output_markdown.strip():
        reason = (
            _BLOCKED_REASON_AGENT_ERROR
            if synthesis_result.error
            else _BLOCKED_REASON_EMPTY_OUTPUT
        )
        logger.error(
            "CodeAgent finalize: synthesis produced no output for %s (reason=%s, error=%s)",
            mission_id,
            reason,
            synthesis_result.error,
        )
        return _block_result(
            result,
            telemetry=telemetry,
            reason=reason,
            synthesis_telemetry=synthesis_telemetry,
        )
    if synthesis_result.error:
        synthesis_telemetry["partial_output_preserved"] = True
        logger.warning(
            "CodeAgent finalize: synthesis returned a reviewable partial output "
            "with an execution error for %s; preserving it for quality grading",
            mission_id,
        )

    # S87 m03 — validate once, then optionally perform ONE bounded recovery
    # synthesis inside finalize before the persistence boundary. There is no
    # LangGraph self-loop and no path back to this decision after recovery.
    cumulative_ledger_outcomes = _merge_ledger_outcomes(
        synthesis_telemetry.get("tool_outcomes", {}),
        {"ledger_records": [], "ledger_records_truncated": 0},
    )
    first_pass = await _run_validation_pass(
        mission_id=mission_id,
        draft_markdown=synthesis_result.output_markdown,
        sources_collected=synthesis_result.sources_collected,
        mission_context=context,
        scope_admission=scope_admission,
        ledger_outcomes=cumulative_ledger_outcomes,
    )
    accounted_synthesis_results = [synthesis_result]
    accounted_critique_results = [first_pass.critique_result]
    attempt_token_usage_complete = True
    first_coverage = completeness_check(first_pass.final_markdown, deliverable_spec)
    decision = decide_recovery(first_pass.critique_result, first_coverage)
    if bounded_scope and decision.triggered:
        decision = replace(decision, triggered=False, blocked_reason="authored_scope_no_restart")
    step_budget = remaining_recovery_steps(
        total_steps=DEFAULT_MAX_STEPS,
        used_steps=synthesis_result.step_count,
    )
    recovery_telemetry: Dict[str, Any] = {
        "passes": 0,
        "status": "not_triggered",
        "decision": decision.to_dict(),
        "remaining_step_budget": step_budget,
        "selected_artifact": "first_pass",
        "first_pass": {
            **first_pass.telemetry,
            "completion_control": dict(synthesis_result.completion_control),
            "deliverable_coverage": first_coverage,
            "synthesis_accounting": {
                "runtime_seconds": float(synthesis_result.runtime_seconds or 0.0),
                "token_usage": dict(synthesis_result.token_usage or {}),
            },
            "generation_failure": _build_generation_failure_telemetry(
                synthesis_result
            ),
            "termination": {
                "step_count": int(synthesis_result.step_count),
                "self_terminated": bool(synthesis_result.self_terminated),
                "hit_max_steps": bool(synthesis_result.hit_max_steps),
                "output_truncated": bool(
                    getattr(synthesis_result, "output_truncated", False)
                ),
            },
        },
    }
    selected_pass = first_pass

    if decision.triggered and step_budget <= 0:
        recovery_telemetry["status"] = "budget_exhausted"
        recovery_telemetry["error"] = safe_error_from_parts(
            "validation",
            "RecoveryBudgetError",
        ).to_text()
    elif decision.triggered:
        recovery_telemetry["passes"] = 1
        synthesis_result = replace(synthesis_result, recovery_passes=1)
        live_urls = [
            annotation.url
            for annotation in first_pass.citations
            if annotation.live is True
        ]
        recovery_prompt = build_recovery_prompt(
            original_markdown=synthesis_result.output_markdown,
            annotations=first_pass.critique_result.annotations,
            coverage=first_coverage,
            sources=synthesis_result.sources_collected,
            live_citation_urls=live_urls,
            remaining_steps=step_budget,
        )
        recovery_telemetry["status"] = "attempted"
        recovery_telemetry["prompt"] = recovery_prompt.telemetry
        recovery_kwargs: dict[str, Any] = {
            "system_instructions": (
                "You are the single bounded recovery actor for a research report. "
                "Follow the recovery directive exactly, use live retrieval only "
                "when existing evidence is insufficient, and return one complete "
                "cited Markdown report.\n" + scope_admission.scope.prompt()
            ),
            "user_directive": recovery_prompt.directive,
            "max_steps": step_budget,
            "verbosity": SMOLAGENTS_VERBOSITY_OFF,
            "deliverable_spec": deliverable_spec,
            "excluded_domains": excluded_domains,
            "high_authority_domains": high_authority_domains,
            "scope_admission": scope_admission,
        }
        if cancel_event is not None:
            recovery_kwargs["cancel_event"] = cancel_event
        recovery_result: SynthesisResult | None = None
        try:
            recovery_result = await asyncio.to_thread(
                run_codeagent_synthesis,
                **recovery_kwargs,
            )
            accounted_synthesis_results.append(recovery_result)
            raise_if_execution_cancelled(cancel_event)
            # Ledger evidence describes retrieval attempts, not only the
            # artifact ultimately selected by the regression guard.  Preserve
            # a failed or rejected recovery pass here so its late fetch and
            # liveness outcomes survive into the terminal metadata as well.
            _accumulate_ledger_outcomes(
                cumulative_ledger_outcomes,
                getattr(recovery_result, "tool_outcomes", {}) or {},
            )
            recovery_result, recovery_source_projection = (
                _canonicalize_synthesis_sources(recovery_result)
            )
            recovery_telemetry["source_projection"] = recovery_source_projection
            recovery_result = replace(
                recovery_result,
                output_markdown=canonicalize_report_text(
                    recovery_result.output_markdown
                ),
            )
            recovery_telemetry["steps_used"] = recovery_result.step_count
            recovery_telemetry["steps_remaining_after"] = max(
                0, step_budget - recovery_result.step_count
            )
            recovery_telemetry["second_pass"] = {
                "completion_control": dict(recovery_result.completion_control),
                "synthesis_accounting": {
                    "runtime_seconds": float(recovery_result.runtime_seconds or 0.0),
                    "token_usage": dict(recovery_result.token_usage or {}),
                },
                "generation_failure": _build_generation_failure_telemetry(
                    recovery_result
                ),
                "termination": {
                    "step_count": int(recovery_result.step_count),
                    "self_terminated": bool(recovery_result.self_terminated),
                    "hit_max_steps": bool(recovery_result.hit_max_steps),
                    "output_truncated": bool(
                        getattr(recovery_result, "output_truncated", False)
                    ),
                }
            }
            if recovery_result.error or not recovery_result.output_markdown.strip():
                recovery_telemetry["status"] = "failed"
                recovery_telemetry["error"] = (
                    normalize_safe_error_text(
                        recovery_result.error,
                        default_category="execution",
                        default_exception_class="RecoveryOutputError",
                    )
                    if recovery_result.error
                    else safe_error_from_parts(
                        "execution",
                        "RecoveryOutputError",
                    ).to_text()
                )
            else:
                combined_result = _combine_synthesis_results(
                    synthesis_result,
                    recovery_result,
                )
                recovered_pass = await _run_validation_pass(
                    mission_id=mission_id,
                    draft_markdown=combined_result.output_markdown,
                    sources_collected=combined_result.sources_collected,
                    mission_context=context,
                    scope_admission=scope_admission,
                    ledger_outcomes=cumulative_ledger_outcomes,
                )
                accounted_critique_results.append(recovered_pass.critique_result)
                if recovered_pass.critique_result.error:
                    recovery_telemetry["status"] = "failed"
                    recovery_telemetry["error"] = normalize_safe_error_text(
                        recovered_pass.critique_result.error,
                        default_category="execution",
                        default_exception_class="CritiqueError",
                    )
                else:
                    # S87 hardening (decision #823 follow-up): the recovery pass
                    # is an un-diffed whole-document rewrite. Only select it when
                    # it is non-regressive vs the first pass on BOTH deliverable
                    # coverage and live citations; otherwise keep the reviewable
                    # first pass rather than persist a gutted report the outcome
                    # gate would still grade COMPLETE/degraded.
                    recovered_coverage = completeness_check(
                        recovered_pass.final_markdown,
                        deliverable_spec,
                    )
                    guard = evaluate_recovery_regression(
                        first_coverage=first_coverage,
                        recovered_coverage=recovered_coverage,
                        first_live_citations=_count_live_citations(
                            first_pass.citations
                        ),
                        recovered_live_citations=_count_live_citations(
                            recovered_pass.citations
                        ),
                        # S88 lever 4 — symmetric applied extraction from each
                        # pass's own annotation ledger (T2 is graded on the
                        # selected pass; recovery must not add damage).
                        first_applied=first_pass.critique_result.annotation_summary()[
                            "applied"
                        ],
                        recovered_applied=(
                            recovered_pass.critique_result.annotation_summary()[
                                "applied"
                            ]
                        ),
                    )
                    recovery_telemetry["second_pass"].update(
                        {
                            **recovered_pass.telemetry,
                            "deliverable_coverage": recovered_coverage,
                        }
                    )
                    recovery_telemetry["regression_guard"] = guard.to_dict()
                    if guard.non_regressive:
                        synthesis_result = combined_result
                        selected_pass = recovered_pass
                        recovery_telemetry["status"] = "completed"
                        recovery_telemetry["selected_artifact"] = "recovery"
                    else:
                        recovery_telemetry["status"] = "rejected_regression"
                        recovery_telemetry["selected_artifact"] = "first_pass"
                        logger.warning(
                            "CodeAgent finalize: recovery rejected as regressive "
                            "for %s (dimensions=%s, shortfall %.3f->%.3f, "
                            "live_citations %d->%d)",
                            mission_id,
                            ",".join(guard.regressed_dimensions),
                            guard.first_shortfall,
                            guard.recovered_shortfall,
                            guard.first_live_citations,
                            guard.recovered_live_citations,
                        )
        except ExecutionCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — first pass remains reviewable
            if recovery_result is None:
                attempt_token_usage_complete = False
            safe_error = safe_exception_text(exc, category="execution")
            logger.error(
                "CodeAgent finalize: bounded recovery failed for %s: %s",
                mission_id,
                safe_error,
            )
            recovery_telemetry["status"] = "failed"
            recovery_telemetry["error"] = safe_error

    recovery_telemetry["attempt_accounting"] = _build_attempt_accounting(
        accounted_synthesis_results,
        accounted_critique_results,
        token_usage_complete=attempt_token_usage_complete,
        compiler_events=compiler_usage_events(state.get("mission_contract") or {}),
    )

    # Rebuild runtime telemetry if a recovery result was selected, then expose
    # the selected validation pass under the canonical top-level keys.
    if recovery_telemetry["selected_artifact"] == "recovery":
        synthesis_telemetry = _build_synthesis_telemetry(synthesis_result)
        selected_source_projection = recovery_telemetry.get("source_projection")
        if isinstance(selected_source_projection, Mapping):
            combined_reasons: Counter[str] = Counter()
            for projection in (source_projection, selected_source_projection):
                raw_reasons = projection.get("reasons")
                if isinstance(raw_reasons, Mapping):
                    for reason, count in raw_reasons.items():
                        combined_reasons[str(reason)] += int(count or 0)
            synthesis_telemetry["source_projection"] = {
                "input_count": int(source_projection.get("input_count") or 0)
                + int(selected_source_projection.get("input_count") or 0),
                "kept_count": len(synthesis_result.sources_collected or []),
                "dropped_count": int(
                    source_projection.get("dropped_count") or 0
                )
                + int(selected_source_projection.get("dropped_count") or 0),
                "merged_count": int(source_projection.get("merged_count") or 0)
                + int(selected_source_projection.get("merged_count") or 0),
                "reasons": dict(sorted(combined_reasons.items())),
            }
    else:
        synthesis_telemetry["recovery_passes"] = synthesis_result.recovery_passes
    _install_ledger_outcomes(
        synthesis_telemetry,
        cumulative_ledger_outcomes,
    )
    synthesis_telemetry.update(selected_pass.telemetry)
    synthesis_telemetry["recovery"] = recovery_telemetry
    stripped_markdown = selected_pass.stripped_markdown
    final_markdown = selected_pass.final_markdown
    citation_annotations = selected_pass.citations
    # The audits below read texts with no [cite:URL] marker left, so they never
    # decide the reading again (S103 m02).
    superscripts = selected_pass.superscripts
    critique_result = selected_pass.critique_result
    if not selected_pass.telemetry.get("critique_skipped"):
        logger.info(
            "CodeAgent finalize: selected critique pass for %s "
            "(claims=%d, gap_notes=%d, recovery_passes=%d)",
            mission_id,
            critique_result.claim_count,
            critique_result.gap_notes_inserted,
            recovery_telemetry["passes"],
        )

    # ---------------------------------------------------------------------
    # S84 m02 (decision #823) — quarantine DOWNGRADES, never deletes. The
    # S82-m02/S83-m03 quarantine deleted DEAD + HALLUCINATED citations, which
    # made it the amplifier that turned false-dead verdicts into gutted
    # reports (45-91 markers stripped per CAT run). The drop set is now
    # HALLUCINATED only (fabricated URL, not in sources_collected — hard
    # quarantine unchanged). DEAD-with-collected-source citations SURVIVE
    # into protocol.citations carrying live=False — post-m02 that value
    # unambiguously means unverified-dead — and are flagged in the References
    # list by the consistency pass below. ORPHAN (live, not-in-sources) and
    # unknown (live=None) are preserved unchanged (decision #792). Runs on
    # BOTH the critiqued and critique_skipped paths. The defensive marker
    # strip is normally a no-op (markers are stripped pre-critique) — leaked
    # hallucinated markers gap-note; every other leaked marker strips
    # silently (no gap note for collected-source claims — m02 criterion 1).
    # The ``quarantined_dead`` variable name is kept for the wiring guards.
    # ---------------------------------------------------------------------
    citation_annotations, quarantined_dead = quarantine_dead_citations(
        citation_annotations,
        synthesis_result.sources_collected,
    )
    final_markdown, _dead_markers_stripped = strip_dead_cite_markers(
        final_markdown,
        quarantined_dead,
        synthesis_result.sources_collected,
    )
    # S84 m02 — markers_quarantined KEEPS its name and int shape (PD137
    # quality-record consumer) but now counts the DOWNGRADED dead citations
    # (acted-upon count, derived from the KEPT set via the shared
    # classifier); hallucinated_quarantined counts the hard drops. BOTH stay
    # top-level synthesis_telemetry siblings — never inside liveness_telemetry
    # (PD7 10-field shape pinned), and nested inside the already-declared
    # synthesis_telemetry channel (no new node-return key).
    _quarantine_source_set = build_source_url_set(synthesis_result.sources_collected)
    # S101 m03: a dead link a collected body names is dead, not fabricated; it
    # is downgraded and flagged unverified like any dead citation.
    _source_named = source_named_dead_urls(
        citation_annotations, synthesis_result.sources_collected
    )
    _classes = [
        classify_citation(
            live=ann.live,
            url=ann.url,
            source_url_set=_quarantine_source_set,
            source_named_urls=_source_named,
        )
        for ann in citation_annotations
    ]
    downgraded_dead = [
        ann
        for ann, cls in zip(citation_annotations, _classes)
        if cls in (CLASSIFICATION_DEAD, CLASSIFICATION_DEAD_UNCOLLECTED)
    ]
    dead_quarantined = len(downgraded_dead)
    hallucinated_quarantined = len(quarantined_dead)
    synthesis_telemetry["markers_quarantined"] = dead_quarantined
    synthesis_telemetry["hallucinated_quarantined"] = hallucinated_quarantined
    synthesis_telemetry["source_named_dead"] = _classes.count(CLASSIFICATION_DEAD_UNCOLLECTED)
    if quarantined_dead or downgraded_dead:
        logger.info(
            "CodeAgent finalize: citation post-processing for %s "
            "(%d dead downgraded to unverified, %d hallucinated quarantined)",
            mission_id,
            dead_quarantined,
            hallucinated_quarantined,
        )

    # S84 m02 — post-processing consistency pass. CAT-R004 shipped a body,
    # References list, and protocol citation set that were mutually out of
    # sync (9 quarantined refs still cited in the body). Annotate the
    # References entries for downgraded-dead (unverified) and quarantined-
    # hallucinated (withdrawn) URLs, then record the three-surface agreement
    # in telemetry (nested in the declared synthesis_telemetry channel).
    final_markdown, citation_consistency = _annotate_references_and_check_consistency(
        final_markdown,
        kept_annotations=citation_annotations,
        downgraded_dead=downgraded_dead,
        quarantined=quarantined_dead,
        citation_reference_markdown=stripped_markdown,
        require_number_mapping=True,
        superscripts=superscripts,
    )
    unavailable = scope_admission.snapshot()["unavailable"]
    if bounded_scope and unavailable:
        # A compact reader-visible warning accompanies the full machine-readable
        # URL/reason receipts. It is part of the text counted below, never an
        # uncounted footer appended after validation.
        final_markdown += (
            "\n\n> Partial evidence: " + str(len(unavailable))
            + " requested source(s) were unavailable or not retrieved. "
            "The source boundary was retained; gaps remain unresolved.\n"
        )
    # This is the canonical report body scored by the outcome gate. Citation
    # coordinates are still relative to stripped_markdown and are rebased only
    # after the wrapper has been rendered, so normalization remains lossless.
    final_markdown = canonicalize_report_text(final_markdown)
    synthesis_telemetry["citation_consistency"] = citation_consistency
    if not citation_consistency.get("consistent", True):
        logger.warning(
            "CodeAgent finalize: citation surfaces inconsistent for %s: %s",
            mission_id,
            citation_consistency,
        )

    projection_upper_bound = _enforce_ledger_projection_budget(
        source_count=len(synthesis_result.sources_collected or []),
        citation_count=len(citation_annotations),
        synthesis_telemetry=synthesis_telemetry,
    )
    if projection_upper_bound > LEDGER_RECORD_MAX_ENTRIES:
        logger.error(
            "CodeAgent finalize: Evidence Ledger projection upper bound exceeds "
            "the %d-entry contract for %s (upper_bound=%d); marking the "
            "ledger audit incomplete so persistence/delivery fails loudly",
            LEDGER_RECORD_MAX_ENTRIES,
            mission_id,
            projection_upper_bound,
        )

    # S82 m04 (PD135) — compute deliverable_coverage on the FINAL post-critique
    # markdown (requirement d). The hardened completeness_check is heading-
    # scoped + substance-aware + entity-count-aware, so the shortfall_score the
    # gate reads below reflects what actually shipped, not a prose echo.
    synthesis_telemetry["deliverable_coverage"] = completeness_check(
        final_markdown,
        deliverable_spec,
    )

    # ---------------------------------------------------------------------
    # S82 m01 (PD136) — tiered outcome gate. Replaces the unconditional
    # ``mission_outcome = COMPLETE`` hardcode. Reads the already-computed
    # quality signals on synthesis_telemetry (citation_telemetry,
    # deliverable_coverage, liveness_telemetry, hit_max_steps/self_terminated)
    # plus the shipped report's word count, and resolves the terminal outcome,
    # delivery_quality, and populated gate-failure lists. The lists +
    # delivery_quality are written onto the state update so
    # ``resolve_final_outcome_payload`` threads them into FinalOutcomePayload /
    # execution_metadata (visible via ``tracelab_mission`` get). A
    # VALIDATION_FAILED verdict still persists the report downstream (worker
    # ``mark_failed`` → writer keeps result_markdown + protocol) — the gate
    # adds a reviewable FLAG, it does not suppress the artifact. The BLOCKED
    # path (empty output / runner error) is handled above and never reaches
    # here.
    # ---------------------------------------------------------------------
    synthesis_telemetry["authored_scope"] = scope_admission.snapshot()
    synthesis_telemetry["evidence_dependencies"] = build_evidence_dependencies(
        [annotation.url for annotation in citation_annotations],
        synthesis_result.sources_collected or [],
        synthesis_telemetry.get("tool_outcomes") or {},
    )
    report_word_count = count_report_words(final_markdown)
    # S83 m05 — persist the shipped word count onto the declared synthesis_telemetry
    # channel so the worker's final_state carries it into the quality-series
    # record (it is computed-but-never-persisted today, so a builder fed the
    # worker final_state would silently see word_count=0 — the S42.2-class trap).
    synthesis_telemetry["report_word_count"] = report_word_count
    # S84 m03 — result-identity envelope (decision #823 / CAT-R003): record the
    # exact text the gate scores (final_markdown as of this line) and, below,
    # the exact string save_report writes, so the worker can assert
    # persisted == scored before writeback. Nested in the declared
    # synthesis_telemetry channel; report_sha256/report_len are added in the
    # report-persistence block (same dict object — the mutation is visible on
    # the state update).
    result_identity: Dict[str, Any] = {
        "algo": "sha256-utf8-nl",
        "mission_id": mission_id,
        "title": title,
        "scored_draft_sha256": _identity_sha256(final_markdown),
        "scored_draft_len": len(final_markdown),
    }
    synthesis_telemetry["result_identity"] = result_identity
    gate = evaluate_outcome_gate(
        synthesis_telemetry,
        report_word_count=report_word_count,
    )
    routing_reason = gate.routing_reason
    mission_outcome = gate.outcome
    delivery_quality = gate.delivery_quality
    quality_gate_failures = list(gate.quality_gate_failures)
    if synthesis_result.error:
        mission_outcome = MissionOutcome.VALIDATION_FAILED
        delivery_quality = DELIVERY_QUALITY_FAILED
        if not routing_reason:
            routing_reason = _VALIDATION_REASON_PARTIAL_OUTPUT
        if _VALIDATION_REASON_PARTIAL_OUTPUT not in quality_gate_failures:
            quality_gate_failures.append(_VALIDATION_REASON_PARTIAL_OUTPUT)
    projection_drops = 0
    for projection_key in (
        "source_projection",
        "citation_url_projection",
        "critique_url_projection",
    ):
        projection_record = synthesis_telemetry.get(projection_key)
        if isinstance(projection_record, Mapping):
            projection_drops += int(
                projection_record.get("dropped_count") or 0
            )
    if projection_drops and delivery_quality == DELIVERY_QUALITY_CLEAN:
        delivery_quality = DELIVERY_QUALITY_USABLE_WITH_WARNING
    synthesis_telemetry["delivery_quality"] = delivery_quality
    # S87 m04 — materialize the canonical quality fact record before protocol
    # construction. The worker reuses this exact dict, and ReportMetadata puts
    # it in the protocol forensic block, so protocol / execution metadata /
    # benchmark extraction share one definition rather than recomputing zeros.
    synthesis_telemetry["quality_record"] = build_quality_record(
        synthesis_telemetry,
        report_word_count,
    )
    result["routing_reason"] = routing_reason
    result["mission_outcome"] = mission_outcome.value
    result["delivery_quality"] = delivery_quality
    result["quality_gate_failures"] = quality_gate_failures
    result["coverage_gate_failures"] = gate.coverage_gate_failures
    result["synthesis_telemetry"] = synthesis_telemetry
    result["synthesis_used"] = True

    report_state = dict(state)
    report_state["telemetry"] = telemetry
    report_state["routing_reason"] = routing_reason
    report_state["mission_outcome"] = mission_outcome.value
    original_annotations = list(citation_annotations)
    report_metadata = _build_report_metadata(
        synthesis_result=synthesis_result,
        synthesis_telemetry=synthesis_telemetry,
        citation_annotations=original_annotations,
        telemetry=telemetry,
    )
    # Keep the S80 wrapper-H1 invariant at the finalize call site: the report
    # renderer owns the canonical H1, so the model-authored leading H1 is
    # removed before any metadata preamble is concatenated.
    final_markdown_without_h1 = _strip_leading_h1(final_markdown)
    report_content = ""
    anchor_failure: tuple[list[dict[str, Any]], str] | None = None

    # The preamble includes the post-rebase reference count. Render and rebase
    # until that one visible field stabilizes (normally one pass; two when a
    # coordinate is quarantined). The loop is deterministic and provider-free.
    for _render_attempt in range(3):
        report_state["synthesis_telemetry"] = dict(synthesis_telemetry)
        raise_if_execution_cancelled(cancel_event)
        try:
            report_content, persisted_synthesized_segment = _render_report_document(
                report_state=report_state,
                report_metadata=report_metadata,
                final_markdown=final_markdown_without_h1,
            )
        except Exception as exc:  # pragma: no cover — deterministic template fault
            safe_error = safe_exception_text(exc, category="persistence")
            logger.error(
                "CodeAgent finalize: report construction failed for %s: %s",
                mission_id,
                safe_error,
            )
            result.setdefault("generation_errors", []).append(safe_error)
            # The renderer is deterministic plumbing, not the paid research.
            # Preserve the exact canonical draft that the quality gate scored
            # so a template fault cannot erase substantive synthesis. It is a
            # forensic BLOCKED artifact, not a fabricated successful report.
            forensic_report = canonicalize_report_text(final_markdown)
            result_identity["report_sha256"] = _identity_sha256(forensic_report)
            result_identity["report_len"] = len(forensic_report)
            result_identity["span_reference_len"] = len(forensic_report)
            synthesis_telemetry["artifact_preservation"] = {
                "report": "scored_draft_inline",
                "protocol": "unavailable",
                "reason": _BLOCKED_REASON_REPORT_CONSTRUCTION,
            }
            result["report_markdown"] = forensic_report
            result["synthesis_telemetry"] = synthesis_telemetry
            return _block_result(
                result,
                telemetry=telemetry,
                reason=_BLOCKED_REASON_REPORT_CONSTRUCTION,
                synthesis_telemetry=synthesis_telemetry,
            )

        if anchor_failure is not None:
            citation_annotations = []
            dropped_annotations, anchor_error = anchor_failure
        else:
            try:
                rebase_outcome = rebase_citation_annotations(
                    reference_markdown=stripped_markdown,
                    persisted_segment_markdown=persisted_synthesized_segment,
                    persisted_markdown=report_content,
                    annotations=original_annotations,
                )
                citation_annotations = rebase_outcome.annotations
                dropped_annotations = rebase_outcome.dropped
                anchor_error = None
            except CitationSpanRebaseError as exc:
                safe_error = safe_exception_text(exc, category="validation")
                logger.error(
                    "CodeAgent finalize: citation span anchor failed for %s: %s "
                    "(reason=%s annotation_index=%s matches=%s); persisting the "
                    "report without citation coordinates",
                    mission_id,
                    safe_error,
                    exc.reason,
                    exc.annotation_index,
                    exc.match_count,
                )
                result.setdefault("generation_errors", []).append(safe_error)
                citation_annotations = []
                dropped_annotations = [
                    {
                        "annotation_index": index,
                        "reason": exc.reason,
                        "url": annotation.url,
                    }
                    for index, annotation in enumerate(original_annotations)
                ]
                anchor_error = exc.reason
                anchor_failure = (dropped_annotations, anchor_error)

        _record_citation_rebase_telemetry(
            synthesis_telemetry,
            mission_id=mission_id,
            dropped=dropped_annotations,
            anchor_error=anchor_error,
        )
        # Grade the artifact that will actually persist. Citation extraction
        # happens before critique/render rebasing, so its original counts are
        # only pre-rebase telemetry; letting them drive the terminal verdict
        # can label a long report COMPLETE after every coordinate was dropped.
        citation_telemetry = dict(
            synthesis_telemetry.get("citation_telemetry") or {}
        )
        citation_telemetry.setdefault(
            "citations_extracted_pre_rebase",
            int(citation_telemetry.get("citations_extracted") or 0),
        )
        citation_telemetry.setdefault(
            "distinct_urls_cited_pre_rebase",
            int(citation_telemetry.get("distinct_urls_cited") or 0),
        )
        source_urls = {
            str(source.get("url") or "")
            for source in synthesis_result.sources_collected or []
            if isinstance(source, Mapping)
        }
        citation_telemetry["citations_extracted"] = len(citation_annotations)
        citation_telemetry["distinct_urls_cited"] = len(
            {annotation.url for annotation in citation_annotations}
        )
        citation_telemetry["persisted_provenance"] = dict(Counter(
            annotation._extraction_mode or "unknown" for annotation in citation_annotations
        ))
        citation_telemetry["orphan_marker_count"] = sum(
            annotation.url not in source_urls for annotation in citation_annotations
        )
        synthesis_telemetry["citation_telemetry"] = citation_telemetry

        _final_source_named = source_named_dead_urls(
            citation_annotations, synthesis_result.sources_collected
        )
        final_downgraded_dead = [
            annotation
            for annotation in citation_annotations
            if classify_citation(
                live=annotation.live,
                url=annotation.url,
                source_url_set=_quarantine_source_set,
                source_named_urls=_final_source_named,
            )
            in (CLASSIFICATION_DEAD, CLASSIFICATION_DEAD_UNCOLLECTED)
        ]
        # The earlier pass can annotate the References surface, but only the
        # post-rebase annotations are eligible for protocol persistence. Audit
        # the exact rendered artifact without mutating it after coordinates
        # have been bound; any mismatch degrades visibly and never destroys the
        # paid report.
        _, final_citation_consistency = (
            _annotate_references_and_check_consistency(
                report_content,
                kept_annotations=citation_annotations,
                downgraded_dead=final_downgraded_dead,
                quarantined=quarantined_dead,
                citation_reference_markdown=report_content,
                require_number_mapping=True,
                annotate_flags=False,
                # Quarantined annotations are intentionally not rebased for
                # persistence. Their exact reader-marker binding was audited
                # against stripped_markdown before the report wrapper was
                # rendered, so carry that result into this final byte-exact
                # surface audit instead of applying stale coordinates.
                audit_quarantined_mapping=False,
                superscripts=superscripts,
            )
        )
        final_citation_consistency["quarantined_mapping_checked"] = bool(
            citation_consistency.get("quarantined_mapping_checked", False)
        )
        final_citation_consistency["unmapped_quarantined_citations"] = int(
            citation_consistency.get("unmapped_quarantined_citations", 0) or 0
        )
        if quarantined_dead and (
            not final_citation_consistency["quarantined_mapping_checked"]
            or final_citation_consistency["unmapped_quarantined_citations"]
        ):
            final_citation_consistency["consistent"] = False
        synthesis_telemetry["citation_consistency"] = final_citation_consistency
        # S101 m02: where each citation whose reader number was rendered ended
        # up. Copied, so the recovery pass record keeps its per-pass counts.
        # Diagnostic only: a failure is recorded and never costs the report.
        reader_number_rendering = synthesis_telemetry.get("reader_number_rendering")
        if isinstance(reader_number_rendering, Mapping):
            fates: Dict[str, int] | None
            try:
                fates = _reader_number_fates(
                    kept=original_annotations,
                    persisted=citation_annotations,
                    dropped=dropped_annotations,
                    quarantined=quarantined_dead,
                    applied_edit_spans=critique_result.applied_edit_spans,
                    critique_input=stripped_markdown,
                    report_content=report_content,
                    expected_urls={
                        _reference_url_identity(annotation.url)
                        for annotation in [*citation_annotations, *quarantined_dead]
                    },
                    superscripts=superscripts,
                )
            except Exception as exc:  # noqa: BLE001 — telemetry must never cost the report
                logger.error(
                    "CodeAgent finalize: reader-number fates failed for %s: %s",
                    mission_id,
                    safe_exception_text(exc, category="internal"),
                )
                fates = None
            synthesis_telemetry["reader_number_rendering"] = {
                **reader_number_rendering,
                "fates": fates,
                **({"fates_error": True} if fates is None else {}),
            }
        synthesis_telemetry["evidence_dependencies"] = build_evidence_dependencies(
            [annotation.url for annotation in citation_annotations],
            synthesis_result.sources_collected or [],
            synthesis_telemetry.get("tool_outcomes") or {},
        )

        scope_validation = validate_persisted_scope(
            admission=scope_admission,
            pre_render=final_markdown,
            persisted=report_content,
            observed_sources=[source for attempt in accounted_synthesis_results
                              for source in attempt.sources_collected],
            cited_urls=[annotation.url for annotation in original_annotations]
            + [url for attempt in accounted_critique_results
               for annotation in attempt.annotations for url in annotation.citation_urls],
        )
        synthesis_telemetry["authored_scope_validation"] = scope_validation
        # Legacy quality series continue to use the body count. The versioned
        # authored bound exclusively uses the persisted count above.
        persisted_gate = evaluate_outcome_gate(
            synthesis_telemetry,
            report_word_count=report_word_count,
        )
        routing_reason = persisted_gate.routing_reason
        mission_outcome = persisted_gate.outcome
        delivery_quality = persisted_gate.delivery_quality
        quality_gate_failures = list(persisted_gate.quality_gate_failures)
        if synthesis_result.error:
            mission_outcome = MissionOutcome.VALIDATION_FAILED
            delivery_quality = DELIVERY_QUALITY_FAILED
            if not routing_reason:
                routing_reason = _VALIDATION_REASON_PARTIAL_OUTPUT
            if _VALIDATION_REASON_PARTIAL_OUTPUT not in quality_gate_failures:
                quality_gate_failures.append(_VALIDATION_REASON_PARTIAL_OUTPUT)
        if (
            (projection_drops or dropped_annotations)
            and delivery_quality == DELIVERY_QUALITY_CLEAN
        ):
            delivery_quality = DELIVERY_QUALITY_USABLE_WITH_WARNING
        synthesis_telemetry["delivery_quality"] = delivery_quality
        synthesis_telemetry["quality_record"] = build_quality_record(
            synthesis_telemetry,
            report_word_count,
        )
        result["routing_reason"] = routing_reason
        result["mission_outcome"] = mission_outcome.value
        result["delivery_quality"] = delivery_quality
        result["quality_gate_failures"] = quality_gate_failures
        result["coverage_gate_failures"] = list(
            persisted_gate.coverage_gate_failures
        )
        reconciled_metadata = _build_report_metadata(
            synthesis_result=synthesis_result,
            synthesis_telemetry=synthesis_telemetry,
            citation_annotations=citation_annotations,
            telemetry=telemetry,
        )
        if reconciled_metadata.references == report_metadata.references:
            report_metadata = reconciled_metadata
            break
        report_metadata = reconciled_metadata
    else:  # pragma: no cover — deterministic reference count cannot oscillate
        raise RuntimeError("citation reference count did not stabilize")

    result["synthesis_telemetry"] = synthesis_telemetry
    result_identity["report_sha256"] = _identity_sha256(report_content)
    result_identity["report_len"] = len(report_content)
    result_identity["span_reference_len"] = len(report_content)
    # Exact in-memory payload is the primary correctness channel. Local files
    # are attempt-scoped staging only; either representation is identity-checked
    # by the worker before terminal persistence.
    result["report_markdown"] = report_content

    raise_if_execution_cancelled(cancel_event)
    try:
        result["report_path"] = save_report(
            report_content,
            mission_id=mission_id,
            title=title,
            mission_uuid=mission_uuid,
        )
    except Exception as exc:  # pragma: no cover — disk failure surface
        safe_error = safe_exception_text(exc, category="persistence")
        logger.error(
            "CodeAgent finalize: report staging failed for %s: %s; using the "
            "identity-checked in-memory artifact",
            mission_id,
            safe_error,
        )
        result.setdefault("generation_errors", []).append(safe_error)

    report_state.update(result)
    report_state["synthesis_telemetry"] = dict(synthesis_telemetry)
    raise_if_execution_cancelled(cancel_event)
    try:
        protocol = generate_mission_protocol(dict(report_state))
        protocol.citations = citation_annotations
        protocol.sources_collected = list(synthesis_result.sources_collected or [])
        protocol.report_metadata = report_metadata
        protocol_payload = canonicalize_protocol_payload(
            protocol.model_dump(mode="json")
        )
        protocol = MissionProtocolComplete.model_validate(protocol_payload)
    except Exception as exc:  # pragma: no cover — deterministic schema fault
        safe_error = safe_exception_text(exc, category="persistence")
        logger.error(
            "CodeAgent finalize: protocol construction failed for %s: %s",
            mission_id,
            safe_error,
        )
        result.setdefault("generation_errors", []).append(safe_error)
        return _block_result(
            result,
            telemetry=telemetry,
            reason=_BLOCKED_REASON_PROTOCOL_CONSTRUCTION,
            synthesis_telemetry=synthesis_telemetry,
        )

    result["protocol_payload"] = protocol_payload
    result_identity["protocol_sha256"] = _protocol_identity_sha256(
        protocol_payload
    )
    raise_if_execution_cancelled(cancel_event)
    try:
        result["protocol_path"] = save_protocol(
            protocol,
            mission_id=mission_id,
            title=title,
            mission_uuid=mission_uuid,
        )
    except Exception as exc:  # pragma: no cover — disk failure surface
        safe_error = safe_exception_text(exc, category="persistence")
        logger.error(
            "CodeAgent finalize: protocol staging failed for %s: %s; using the "
            "identity-checked in-memory artifact",
            mission_id,
            safe_error,
        )
        result.setdefault("generation_errors", []).append(safe_error)

    raise_if_execution_cancelled(cancel_event)
    logger.info(
        "CodeAgent finalize complete for %s: steps=%d runtime=%.1fs tokens=%s",
        mission_id,
        synthesis_result.step_count,
        synthesis_result.runtime_seconds,
        synthesis_result.token_usage,
    )
    return result


def _record_citation_rebase_telemetry(
    synthesis_telemetry: Dict[str, Any],
    *,
    mission_id: str,
    dropped: Sequence[Mapping[str, Any]],
    anchor_error: str | None,
) -> None:
    """Record citation coordinates lost at persistence, without report text.

    S92 m06. The rebase used to raise on the first unmappable slice, and the
    worker turned that into a hard block with no artifacts at all -- three
    production missions lost a completed report to one citation. Loss is now
    reported here instead: counts and structural reason codes only, so the
    signal survives into the protocol without carrying cited claim text.
    """

    record: Dict[str, Any] = {
        "dropped_count": len(dropped),
        "reasons": dict(sorted(Counter(str(item["reason"]) for item in dropped).items())),
    }
    if anchor_error is not None:
        record["anchor_error"] = anchor_error
    synthesis_telemetry["citation_rebase"] = record
    synthesis_telemetry["citations_unmappable"] = len(dropped)
    if not dropped:
        return
    logger.warning(
        "CodeAgent finalize: %d citation coordinate(s) unmappable for %s "
        "(reasons=%s anchor_error=%s); report ships with the mappable set",
        len(dropped),
        mission_id,
        record["reasons"],
        anchor_error,
    )


def _block_result(
    base: StateUpdate,
    *,
    telemetry: Dict[str, Any],
    reason: str,
    synthesis_telemetry: Dict[str, Any],
) -> StateUpdate:
    base["routing_reason"] = reason
    base["mission_outcome"] = MissionOutcome.BLOCKED.value
    # Never send a failed CodeAgent state through the retired legacy generator:
    # it manufactures a contradictory "no findings / no sources" report even
    # when paid retrieval telemetry records substantial work.
    base["synthesis_used"] = True
    base["synthesis_telemetry"] = synthesis_telemetry
    base["telemetry"] = telemetry
    return base


def _extract_and_resolve_citations(
    final_markdown: str,
    *,
    sources_collected: list[dict[str, Any]] | None,
    synthesis_telemetry: Dict[str, Any],
    rendered_markers: Mapping[tuple[int, str], int] | None = None,
) -> tuple[str, List[CitationAnnotation]]:
    """Strip ``[cite:URL]`` markers, resolve titles, and write telemetry.

    Single callsite for the extractor in the success path. Always populates
    ``synthesis_telemetry["citation_telemetry"]`` so downstream consumers
    don't have to branch on missing-vs-empty. ``rendered_markers`` flags each
    primary citation whose own marker received a rendered reader number.
    """

    citation_t0 = time.monotonic()
    reconciliation: dict[str, Any] = {}
    stripped_markdown, annotations = extract_citations(final_markdown, telemetry=reconciliation)
    for annotation in annotations:
        annotation._cited_url = annotation.url
        if annotation._extraction_mode == "primary":
            annotation._reader_number_rendered = (rendered_markers or {}).get(
                (annotation.end_index, annotation.url)
            )
    annotations, url_projection = _canonicalize_citation_urls(annotations)
    citation_runtime = round(time.monotonic() - citation_t0, 3)

    source_url_to_title: dict[str, str] = {}
    for src in sources_collected or []:
        if not isinstance(src, Mapping):
            continue
        url = str(src.get("url") or "").strip()
        if not url:
            continue
        title = str(src.get("title") or "").strip()
        if title and url not in source_url_to_title:
            source_url_to_title[url] = title

    orphan_count = 0
    for ann in annotations:
        title = source_url_to_title.get(ann.url)
        if title:
            ann.title = title
        if ann.url not in source_url_to_title:
            orphan_count += 1

    distinct_urls = len({ann.url for ann in annotations})

    synthesis_telemetry["citation_telemetry"] = {
        "citations_extracted": len(annotations),
        "orphan_marker_count": orphan_count,
        "distinct_urls_cited": distinct_urls,
        # S83 m05 (PD116c) — forensic-only count of numbered refs the body
        # carries; nested inside the declared synthesis_telemetry channel.
        "numbered_refs_parsed": reconciliation["numbered_annotations"],
        "reconciliation": reconciliation,
        "runtime_seconds": citation_runtime,
    }
    synthesis_telemetry["citation_url_projection"] = url_projection

    return stripped_markdown, annotations


def _invoke_critique(
    *,
    mission_id: str,
    draft_markdown: str,
    sources_collected: list[dict[str, Any]] | None,
    mission_context: Mapping,
    citations: List[CitationAnnotation],
    scope_admission: ScopeAdmission | None = None,
) -> tuple[CritiqueResult, str | None]:
    """Run the critique pass with defense-in-depth fallthrough.

    The critique module's ``run_critique`` already absorbs LLM errors and
    returns a ``CritiqueResult`` with ``output_markdown == draft_markdown`` on
    failure (the caller logs at WARNING). The outer try/except here is belt
    and braces — if the module ever leaks an exception despite that contract,
    the finalize node still completes the mission with the un-critiqued
    draft (the stripped markdown — no ``[cite:`` markers leak even if a
    misbehaving mock returns the raw synthesis text).

    ``citations`` is the S68 live-checked annotation list; the critique
    builder injects a "# Citation Liveness Context" block from it so the
    LLM can apply rule 8 (drop orphan/dead markers + gap-note their claims).
    """

    try:
        critique_result = run_critique(
            draft_markdown,
            list(sources_collected or []),
            mission_context or {},
            citations=citations,
            scope_admission=scope_admission,
        )
    except Exception as exc:  # noqa: BLE001 — defense in depth
        safe_error = safe_exception_text(exc, category="execution")
        logger.error(
            "CodeAgent finalize: critique raised unexpectedly for %s: %s — "
            "falling through with un-critiqued draft",
            mission_id,
            safe_error,
        )
        synthetic = CritiqueResult(
            output_markdown=draft_markdown,
            claim_count=0,
            gap_notes_inserted=0,
            runtime_seconds=0.0,
            token_usage={"input": 0, "output": 0, "total": 0},
            error=safe_error,
        )
        return synthetic, safe_error

    critique_result = replace(
        critique_result,
        error=normalize_safe_error_text(
            critique_result.error,
            default_category="execution",
        ),
    )
    if critique_result.error:
        logger.warning(
            "CodeAgent finalize: critique skipped for %s: %s",
            mission_id,
            critique_result.error,
        )
    return critique_result, None


async def _run_liveness_check(
    *,
    mission_id: str,
    annotations: List[CitationAnnotation],
    sources_collected: list[dict[str, Any]] | None,
    scope_admission: ScopeAdmission | None = None,
) -> tuple[List[CitationAnnotation], LivenessTelemetry]:
    """Defense-in-depth wrapper around :func:`check_citations_liveness`.

    The gate's own contract is no-raise (outer try/except inside the module
    falls through with all ``unknown`` on internal exception). This wrapper
    is belt-and-braces against a hypothetical contract violation — on any
    leaked exception we leave annotations unchanged (``live`` already
    defaults to ``None``), preserve the 10-field PD7 public shape, and mark
    the internal Ledger envelope incomplete because already-started probes
    cannot be reconstructed at this boundary.
    """

    if not annotations:
        return annotations, LivenessTelemetry()

    try:
        return await check_citations_liveness(
            annotations, sources_collected or [], scope_admission=scope_admission,
        )
    except Exception as exc:  # noqa: BLE001 — defense in depth
        safe_error = safe_exception_text(exc, category="transport")
        logger.error(
            "CodeAgent finalize: citation liveness gate raised unexpectedly "
            "for %s (%s) — falling through with all unknown",
            mission_id,
            safe_error,
        )
        for ann in annotations:
            ann.live = None
        return annotations, LivenessTelemetry(
            checked=len(annotations),
            unknown=len(annotations),
            ledger_records_truncated=1,
        )


def _distinct_domain_count(sources_collected: list[dict[str, Any]] | None) -> int:
    """Count distinct hostnames across the runner's sources_collected.

    Carried in :pyattr:`ReportMetadata.forensic` to give the user a coarse
    "did the agent search across multiple hosts" signal without polluting
    the at-a-glance header (PD22 — anything needing context goes forensic).
    """

    domains: set[str] = set()
    for src in sources_collected or []:
        if not isinstance(src, Mapping):
            continue
        url = str(src.get("url") or "").strip()
        if not url:
            continue
        try:
            host = (urlparse(url).hostname or "").lower()
        except Exception:
            continue
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.add(host)
    return len(domains)


def _build_report_metadata(
    *,
    synthesis_result: SynthesisResult,
    synthesis_telemetry: Mapping[str, Any],
    citation_annotations: List[CitationAnnotation],
    telemetry: Mapping[str, Any],
) -> ReportMetadata:
    """Assemble the S72 PD22 run-integrity block from finalize-node signals.

    The 4 at-a-glance header fields:

    * ``sources_collected`` — distinct discovery/readable-evidence grounding URLs.
    * ``references`` — distinct URLs cited in the final markdown (post-
      extraction; the citation_telemetry surface is authoritative).
    * ``agent_steps_used`` / ``agent_steps_max`` — CodeAgent step trace
      from the SynthesisResult; ``max`` falls back to the locked
      ``DEFAULT_MAX_STEPS`` constant when the runner did not record it.
    * ``runtime_seconds`` — cumulative synthesis wall-clock across every paid
      pass, including a rejected recovery; direct helper callers fall back to
      the selected SynthesisResult.

    Forensic block carries everything that needs explanation: schema
    version, build_hash (when reachable on telemetry.runtime_identity),
    liveness telemetry, citation telemetry, distinct-domain count, agent
    self_terminated/hit_max_steps flags, token usage, tool-call summary.
    """

    citation_telemetry = synthesis_telemetry.get("citation_telemetry")
    # Modern finalize calls rebuild this block after persisted-coordinate
    # rebasing and carry ``citation_rebase`` telemetry. Count only annotations
    # that actually ship, including the truthful zero when every coordinate was
    # dropped. Legacy/direct callers without that signal retain the historical
    # liveness/citation telemetry fallback.
    if isinstance(synthesis_telemetry.get("citation_rebase"), Mapping):
        references = len(
            {ann.url for ann in citation_annotations if ann.live is True}
        )
    elif citation_annotations:
        # Direct/legacy helper callers may supply resolved annotations without
        # running liveness. Count every resolved URL when all verdicts are
        # unknown; once any verdict exists, retain the historical live-only
        # header contract.
        if any(annotation.live is not None for annotation in citation_annotations):
            references = len(
                {
                    annotation.url
                    for annotation in citation_annotations
                    if annotation.live is True
                }
            )
        else:
            references = len(
                {annotation.url for annotation in citation_annotations}
            )
    else:
        liveness_telemetry_raw = synthesis_telemetry.get("liveness_telemetry")
        if isinstance(liveness_telemetry_raw, Mapping) and "live" in liveness_telemetry_raw:
            references = int(liveness_telemetry_raw.get("live") or 0)
        elif isinstance(citation_telemetry, Mapping):
            references = int(citation_telemetry.get("distinct_urls_cited") or 0)
        else:
            references = 0

    runtime_identity_raw = (
        telemetry.get("runtime_identity") if isinstance(telemetry, Mapping) else None
    )
    runtime_identity: Mapping[str, Any] = (
        runtime_identity_raw if isinstance(runtime_identity_raw, Mapping) else {}
    )

    recovery_raw = synthesis_telemetry.get("recovery")
    recovery = recovery_raw if isinstance(recovery_raw, Mapping) else {}
    accounting_raw = recovery.get("attempt_accounting")
    attempt_accounting = (
        accounting_raw if isinstance(accounting_raw, Mapping) else {}
    )
    cumulative_usage_raw = attempt_accounting.get("token_usage")
    cumulative_usage = (
        cumulative_usage_raw
        if isinstance(cumulative_usage_raw, Mapping)
        else synthesis_result.token_usage
    )
    cumulative_steps = int(
        attempt_accounting.get("steps_used") or synthesis_result.step_count or 0
    )
    runtime_seconds = float(
        attempt_accounting.get("synthesis_runtime_seconds")
        or synthesis_result.runtime_seconds
        or 0.0
    )

    forensic: Dict[str, Any] = {
        "schema_version": REPORT_METADATA_SCHEMA_VERSION,
        "build_hash": runtime_identity.get("build_hash"),
        "self_terminated": bool(synthesis_result.self_terminated),
        "hit_max_steps": bool(synthesis_result.hit_max_steps),
        "distinct_domains": _distinct_domain_count(synthesis_result.sources_collected),
        "token_usage": dict(cumulative_usage or {}),
        "tool_call_summary": dict(synthesis_result.tool_call_summary or {}),
    }
    if attempt_accounting:
        forensic["attempt_accounting"] = dict(attempt_accounting)
    liveness_telemetry = synthesis_telemetry.get("liveness_telemetry")
    if isinstance(liveness_telemetry, Mapping):
        forensic["liveness_telemetry"] = dict(liveness_telemetry)
    if isinstance(citation_telemetry, Mapping):
        forensic["citation_telemetry"] = dict(citation_telemetry)
    critique_telemetry = synthesis_telemetry.get("critique_telemetry")
    if isinstance(critique_telemetry, Mapping):
        forensic["critique_telemetry"] = dict(critique_telemetry)
    quality_record = synthesis_telemetry.get("quality_record")
    if isinstance(quality_record, Mapping):
        forensic["quality_record"] = dict(quality_record)
    for projection_key in (
        "source_projection",
        "citation_url_projection",
        "critique_url_projection",
        "citation_rebase",
        "authored_scope",
        "scope_compilation",
        "authored_scope_validation",
        "original_synthesis",
    ):
        projection = synthesis_telemetry.get(projection_key)
        if isinstance(projection, Mapping):
            forensic[projection_key] = dict(projection)

    return ReportMetadata(
        sources_collected=len(synthesis_result.sources_collected or []),
        references=references,
        agent_steps_used=cumulative_steps,
        agent_steps_max=int(DEFAULT_MAX_STEPS),
        runtime_seconds=runtime_seconds,
        forensic=forensic,
    )


def _render_report_document(
    *,
    report_state: AgentState,
    report_metadata: ReportMetadata,
    final_markdown: str,
) -> tuple[str, str]:
    """Render canonical report bytes and the exact embedded synthesis segment."""

    final_markdown_with_preamble = (
        _render_report_metadata_preamble(report_metadata)
        + final_markdown
    )
    persisted_segment = prepare_synthesized_content(
        final_markdown_with_preamble
    )
    report_content = canonicalize_report_text(
        generate_markdown_report(
            report_state,
            synthesized_content=final_markdown_with_preamble,
        )
    )
    return report_content, persisted_segment


def _format_runtime_seconds(seconds: float) -> str:
    """Render a wall-clock runtime as ``Xm Ys`` (or ``X.Ys`` under one minute)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def _preview_report_word_budget(
    draft: str, *, report_state: AgentState, scope: AuthoredScope,
) -> dict[str, Any]:
    """Estimate persisted length with the real renderer before finalization.

    Numeric metadata counts occupy one token regardless of value. Reserve the
    two-token runtime format; sub-minute runs use one fewer. Later critique and
    validation edits remain unknown, so this is feedback, never an acceptance.
    """
    stripped, _ = extract_citations(canonicalize_report_text(draft))
    report, _ = _render_report_document(
        report_state=report_state,
        report_metadata=ReportMetadata(
            agent_steps_max=DEFAULT_MAX_STEPS, runtime_seconds=60,
        ),
        final_markdown=_strip_leading_h1(stripped),
    )
    words = count_report_words(report)
    return {
        "version": 1,
        "available": True,
        "basis": "pre_critique_render_estimate",
        "projected_words": words,
        "draft_words": count_report_words(stripped),
        "min_words": scope.min_words,
        "max_words": scope.max_words,
        "words_over_maximum": max(0, words - scope.max_words)
        if scope.max_words is not None else 0,
        "words_below_minimum": max(0, scope.min_words - words)
        if scope.min_words is not None else 0,
    }


# S84 m02 — References-list flags for the two acted-upon citation classes.
# Deliberately NOT the gap-note literal: gap notes are reserved for claims
# with no collected source at all.
_UNVERIFIED_REF_SUFFIX = " — [unverified: link could not be re-verified]"
_WITHDRAWN_REF_SUFFIX = " — [citation withdrawn: source could not be verified]"

_REFERENCES_HEADING_RE = re.compile(
    r"^(#{1,3})\s+(?:\d+\.\s+)?References\b.*$", re.MULTILINE
)
_HEADING_LEVEL_RE = re.compile(r"^(#{1,6})\s")
_NUMBERED_REFERENCE_PREFIX_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?:\*\*|__)?(?:\[(\d+)\]|(\d+)[.)])"
    r"(?:\*\*|__)?(?:\s|$)"
)
_NUMBERED_BODY_MARKER_RE = re.compile(r"\[(\d+)\]")
_TRAILING_NUMBERED_MARKERS_RE = re.compile(r"((?:\[\d+\]\s*)+)$")
_LEADING_NUMBERED_MARKERS_RE = re.compile(r"^\s*((?:\[\d+\]\s*)+)")
# Whitespace/angle-bracket delimited — parentheses and brackets are captured
# and trimmed by _clean_ref_url only when UNBALANCED, so Wikipedia-style
# ``.../Foo_(bar)`` paths survive intact while ``(see https://x.com/p)`` and
# ``[t](https://x.com/p)`` shed the closing delimiter (S84 m02 review fix).
_URL_IN_LINE_RE = re.compile(r"https?://[^\s<>]+")


def _reference_url_identity(url: str) -> str:
    """Return the TraceLab-canonical identity used across citation surfaces."""

    try:
        canonical = _canonical_ledger_url(url)
    except (TypeError, ValueError):
        # The consistency pass is diagnostic and must not kill an otherwise
        # reviewable report. Retain an invalid reference as a distinct raw
        # identity so it cannot disappear into a false ``consistent=True``.
        canonical = url
    return _normalise_url_for_cache(canonical)


def _identity_sha256(text: str) -> str:
    """Newline-normalized sha256 for the S84 m03 result-identity envelope.

    MUST mirror ``deepsearch.worker.writer.identity_sha256`` (kept local —
    the import direction is worker → agent, never the reverse). The worker
    reads artifacts through universal-newline translation, so both sides
    hash a ``\\r\\n``/``\\r`` → ``\\n`` normalized form.
    """
    return hashlib.sha256(
        text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    ).hexdigest()


def _protocol_identity_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the canonical JSON shape used by the worker identity fence."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_ref_url(
    raw: str,
    *,
    expected_urls: AbstractSet[str] = frozenset(),
) -> str:
    """Resolve a captured reference URL without corrupting legal suffixes.

    The raw token is the first candidate because punctuation and balanced
    parentheses are legal URL path/query characters. Progressively trimmed
    candidates then remove Markdown wrappers, prose punctuation, and only
    unmatched closing delimiters. When a candidate matches a known protocol
    or quarantine identity, that exact spelling wins; otherwise the most
    fully unwrapped candidate preserves the historical reference-only shape.
    """
    candidates = [raw]
    candidate = raw
    while candidate:
        if candidate[-1] in "*`'\"":
            candidate = candidate[:-1]
        elif candidate[-1] in ".,;:!?":
            candidate = candidate[:-1]
        elif candidate[-1] in ")]":
            opener = "(" if candidate[-1] == ")" else "["
            if candidate.count(candidate[-1]) <= candidate.count(opener):
                break
            candidate = candidate[:-1]
        else:
            break
        candidates.append(candidate)

    for candidate in candidates:
        if _reference_url_identity(candidate) in expected_urls:
            return candidate
    return candidates[-1]


class _ReferencesEntries(NamedTuple):
    """The References audit's parse of one report's References section.

    ``lines`` is the text after the heading split on newlines; the first
    ``section_lines`` belong to the section. ``line_urls`` maps each section
    line carrying a URL to that line's URL identities, and ``line_numbers``
    maps each numbered one to its reference number.
    """

    heading_start: int
    section_start: int
    section_end: int
    lines: list[str]
    section_lines: int
    line_urls: dict[int, set[str]]
    refs_urls: set[str]
    numbered_urls: dict[int, set[str]]
    number_collisions: set[int]
    unsupported_reference_rows: int
    line_numbers: dict[int, int]


def _parse_references_entries(
    markdown: str,
    *,
    expected_urls: AbstractSet[str],
) -> _ReferencesEntries | None:
    """Parse the References section as the audit reads it; ``None`` without one.

    S101 m02 lifts this out of the audit unchanged, so reader-number rendering
    decides on exactly the entries the audit maps against.
    """

    heading_match = _REFERENCES_HEADING_RE.search(markdown or "")
    if not heading_match:
        return None
    section_start = heading_match.end()
    lines = (markdown[section_start:]).split("\n")
    reference_heading_level = len(heading_match.group(1))

    def _section_end() -> int:
        for line_index, candidate_line in enumerate(lines):
            nested_heading = _HEADING_LEVEL_RE.match(candidate_line)
            if (
                nested_heading
                and len(nested_heading.group(1)) <= reference_heading_level
            ):
                return line_index
        return len(lines)

    # A nested heading such as ``### Annotated Source List`` belongs to an
    # H2 References section. The old any-heading stop made R2 report 0/30
    # even though all numbered URL rows followed that nested heading.
    section_lines = _section_end()
    line_urls: dict[int, set[str]] = {}
    line_numbers: dict[int, int] = {}
    refs_urls: set[str] = set()
    numbered_urls: dict[int, set[str]] = {}
    numbered_sources: dict[int, set[str]] = {}
    number_collisions: set[int] = set()
    unsupported_reference_rows = 0
    section_end = min(len(markdown), section_start + sum(
        len(line) + 1 for line in lines[:section_lines]
    ))
    for i, line in enumerate(lines[:section_lines]):
        found = _URL_IN_LINE_RE.findall(line)
        if not found:
            continue
        line_urls[i] = {
            _reference_url_identity(_clean_ref_url(u, expected_urls=expected_urls))
            for u in found
        }
        refs_urls |= line_urls[i]
        numbered_match = _NUMBERED_REFERENCE_PREFIX_RE.match(line)
        reference_number: int | None = None
        if numbered_match:
            reference_number = int(numbered_match.group(1) or numbered_match.group(2))
        else:
            superscript_match = _UNICODE_SUPER_REFERENCE_ENTRY_RE.match(line)
            if superscript_match:
                reference_number = _super_to_int(superscript_match.group(1))
        if reference_number is None and re.match(r"^\s*(?:\*\*)?\(?\d+", line):
            unsupported_reference_rows += 1
        if reference_number is not None:
            line_numbers[i] = reference_number
            # S100 m02: links naming one source (an arXiv abstract and its
            # HTML page) are one entry, as in the extractor. numbered_urls
            # keeps every link, so the body mapping accepts either form.
            line_sources = {_source_identity(url) for url in line_urls[i]}
            prior = numbered_sources.setdefault(reference_number, line_sources)
            if len(line_sources) != 1 or prior != line_sources:
                number_collisions.add(reference_number)
            numbered_urls.setdefault(reference_number, set()).update(line_urls[i])
    return _ReferencesEntries(
        heading_start=heading_match.start(),
        section_start=section_start,
        section_end=section_end,
        lines=lines,
        section_lines=section_lines,
        line_urls=line_urls,
        refs_urls=refs_urls,
        numbered_urls=numbered_urls,
        number_collisions=number_collisions,
        unsupported_reference_rows=unsupported_reference_rows,
        line_numbers=line_numbers,
    )


def _boundary_numbers(
    reference_text: str,
    annotation: CitationAnnotation,
    *,
    superscripts: bool,
    numbered_fallback: bool = False,
) -> set[int]:
    """The reader numbers the References audit binds to one citation's boundary.

    S101 m02 lifts this out of the audit unchanged, so rendering and mapping
    read a citation's reader number the same way. S103 m02 reads superscripts
    only when ``superscripts`` says the report cites that way, and matches each
    run in the full text, so the character before it decides an exponent.
    """

    start = max(
        0, min(len(reference_text), int(annotation.start_index))
    )
    end = max(
        start, min(len(reference_text), int(annotation.end_index))
    )
    span_numbers: set[int] = set()
    span_text = reference_text[start:end]
    # Primary [cite:URL] extraction trims the space before the
    # machine marker, leaving the human-readable number at the
    # end of the persisted citation span. Inspect that owned
    # marker before the suffix: in ``[1][cite:A][2][cite:B]``,
    # A's persisted span ends at [1] while its suffix starts at
    # B's [2]. Suffix-first inspection falsely binds A to [2].
    trailing_match = _TRAILING_NUMBERED_MARKERS_RE.search(
        span_text
    )
    if trailing_match:
        span_numbers.update(
            int(match)
            for match in _NUMBERED_BODY_MARKER_RE.findall(
                trailing_match.group(1)
            )
        )
    # A primary claim may cite multiple sources as ``claim¹ ²``;
    # every trailing whitespace-separated superscript marker is
    # adjacent to the shared machine-citation coordinate.
    superscript_matches = list(
        _UNICODE_SUPER_MARKER_RE.finditer(reference_text, start, end)
    ) if superscripts else []
    trailing_start = end
    for match in reversed(superscript_matches):
        if reference_text[match.end() : trailing_start].strip():
            break
        number = _super_to_int(match.group(1))
        if number is not None:
            span_numbers.add(number)
        trailing_start = match.start()
    # Numbered-fallback annotations end immediately before their
    # own reader marker. When several markers are adjacent, only
    # the first leading marker belongs to this coordinate; later
    # ones belong to later annotations.
    suffix_numbers: set[int] = set()
    suffix_text = reference_text[end : end + 128]
    suffix_match = _LEADING_NUMBERED_MARKERS_RE.match(suffix_text)
    if suffix_match:
        first_number = _NUMBERED_BODY_MARKER_RE.search(
            suffix_match.group(1)
        )
        if first_number:
            suffix_numbers.add(int(first_number.group(1)))
    leading_offset = len(suffix_text) - len(suffix_text.lstrip())
    superscript_match = _UNICODE_SUPER_MARKER_RE.match(
        reference_text, end + leading_offset, end + len(suffix_text)
    ) if superscripts else None
    if superscript_match:
        number = _super_to_int(superscript_match.group(1))
        if number is not None:
            suffix_numbers.add(number)

    # Primary and numbered-fallback extraction can emit the same
    # coordinate shape with opposite ownership. Primary A in
    # ``[1][cite:A][2][cite:B]`` owns trailing [1]; fallback B in
    # ``[1][2]`` owns suffix [2]. Extraction mode is therefore
    # required to choose precedence without regressing either.
    owns_suffix = annotation._extraction_mode == "numbered" or (
        annotation._extraction_mode is None and numbered_fallback
    )
    if not owns_suffix and span_numbers:
        return span_numbers
    if suffix_numbers:
        return suffix_numbers
    return span_numbers


# S101 m02 — reader numbers for kept inline citations.
READER_NUMBER_RENDERING_VERSION = 1
_READER_NUMBER_SKIPS = (
    "skipped_not_in_references",
    "skipped_multiple_numbers",
    "skipped_conflicting_number",
    "skipped_link_syntax",
    "skipped_parser_disagreement",
    "skipped_sentence_boundary",
    "skipped_line_start",
)
_READER_NUMBER_FATES = (
    "mapped", "unmapped", "quarantined", "lost_with_critique_edit", "lost_at_rebase",
)


# A line start holding only Markdown block syntax: heading, quote, bullet and
# ordered-list markers, and task boxes.
_BLOCK_PREFIX_RE = re.compile(r"(?:\s*(?:#{1,6}|>+|[-*+]|\d{1,9}[.)]|\[[ xX]\]))*\s*")
# A token a GFM autolink would extend over a number glued to its end.
_BARE_LINK_RE = re.compile(r"[(*_~]*(?:https?://|www\.)", re.IGNORECASE)


def _is_report_whitespace(char: str) -> bool:
    """Whether ``count_report_words`` splits on ``char``."""

    return bool(_REPORT_WHITESPACE.fullmatch(char))


def _is_line_break(char: str) -> bool:
    """Report whitespace that ends a line, as ``str.splitlines`` reads it."""

    return _is_report_whitespace(char) and len(f"a{char}b".splitlines()) == 2


def _is_spacing(char: str) -> bool:
    return _is_report_whitespace(char) and not _is_line_break(char)


def _render_reader_numbers(
    draft: str,
) -> tuple[str, Dict[str, Any], Dict[tuple[int, str], int]]:
    """Render the reader number a kept inline citation lacks (S101 m02).

    A ``[cite:URL]`` marker whose citation has no reader number at its boundary
    gets ``[N]`` immediately before it when the References audit lists the URL
    under exactly one number N, N is no collision, the extractor resolves N to
    the URL it deduplicates against this marker, and the marker's group is not
    followed by ``(``. A number the agent wrote is never changed and no
    References entry is added. The stripped draft keeps its word count: the
    spacing inside a rendered group is removed, and a group followed by
    whitespace is glued to the last visible character, unless that character
    ends a sentence or the group starts a line, where the group is skipped.

    Never a gate: on any error the draft comes back unchanged with ``error``
    set. Returns the draft, the pass telemetry, and each rendered marker's
    ``(end_index, URL)`` in the rendered draft's extraction with its number.
    """

    telemetry: Dict[str, Any] = {
        "version": READER_NUMBER_RENDERING_VERSION,
        "rendered": 0,
        **dict.fromkeys(_READER_NUMBER_SKIPS, 0),
        "words_before": 0,
        "words_after": 0,
        "draft_chars_before": len(draft),
        "draft_chars_after": len(draft),
        "error": False,
    }
    try:
        rendered, keys = _place_reader_numbers(draft, telemetry)
    except Exception as exc:  # noqa: BLE001 — rendering must never cost the report
        logger.error(
            "reader-number rendering failed: %s — keeping the draft unchanged",
            safe_exception_text(exc, category="internal"),
        )
        telemetry.update(dict.fromkeys(("rendered", *_READER_NUMBER_SKIPS), 0))
        telemetry.update(
            words_after=telemetry["words_before"], draft_chars_after=len(draft), error=True,
        )
        return draft, telemetry, {}
    telemetry["rendered"] = len(keys)
    telemetry["draft_chars_after"] = len(rendered)
    return rendered, telemetry, keys


def _place_reader_numbers(
    draft: str, telemetry: Dict[str, Any],
) -> tuple[str, Dict[tuple[int, str], int]]:
    code_spans = _find_code_spans(draft)
    markers = [m for m in _CITE_MARKER_RE.finditer(draft)
               if not _is_inside_span(m.start(), code_spans)]
    if not markers:
        # Nothing to render, and extraction would strip nothing.
        telemetry["words_before"] = telemetry["words_after"] = count_report_words(draft)
        return draft, {}
    stripped, annotations = extract_citations(draft)
    # The marked draft decides as the validation pass's extraction does: it has
    # a [cite:URL] marker, so only its labels can switch superscripts on.
    superscripts = reads_superscripts(draft)
    telemetry["words_before"] = telemetry["words_after"] = count_report_words(stripped)
    entries = _parse_references_entries(
        stripped, expected_urls={_reference_url_identity(a.url) for a in annotations},
    )
    # Without a References section no URL is listed under any number.
    numbered = entries.numbered_urls if entries else {}
    collisions = entries.number_collisions if entries else set()
    section = (entries.heading_start, entries.section_end) if entries else (len(stripped),) * 2
    listed: dict[str, set[int]] = {}
    for number, urls in numbered.items():
        for url in urls:
            listed.setdefault(url, set()).add(number)
    merged: set[int] = set()
    resolved, _ = _build_references_map(stripped, merged=merged)
    primary = {(a.end_index, a.url): a for a in annotations if a._extraction_mode == "primary"}

    numbers: list[int | None] = []
    positions: list[int] = []
    decided: set[int] = set()
    removed = 0
    for marker in markers:
        position = marker.start() - removed
        removed += marker.end() - marker.start()
        numbers.append(None)
        positions.append(position)
        url = marker.group(1)
        if section[0] <= position < section[1] or position == section[1] == len(stripped):
            continue  # the References section is never rendered
        try:
            _canonical_ledger_url(url)
        except (TypeError, ValueError):
            continue  # an unsafe URL never reaches the protocol
        end = position
        while end > 0 and stripped[end - 1].isspace():
            end -= 1
        citation = primary.get((end, url))
        if citation is None or id(citation) in decided:
            # A repeat: the extractor keeps one citation per claim and URL,
            # decided at its first marker.
            continue
        decided.add(id(citation))
        identity = _reference_url_identity(url)
        boundary = _boundary_numbers(stripped, citation, superscripts=superscripts)
        candidates = listed.get(identity, set())
        if boundary:
            if not any(identity in numbered.get(n, set()) for n in boundary):
                telemetry["skipped_conflicting_number"] += 1
            continue
        if not candidates:
            telemetry["skipped_not_in_references"] += 1
            continue
        if len(candidates) > 1:
            telemetry["skipped_multiple_numbers"] += 1
            continue
        (number,) = candidates
        target = resolved.get(number)
        if number in collisions or target is None or not (
            target == url
            or (number in merged and _source_identity(target) == _source_identity(url))
        ):
            telemetry["skipped_parser_disagreement"] += 1
        else:
            numbers[-1] = number

    # A group is a run of markers separated by nothing or only by spacing.
    groups: list[tuple[int, int]] = []
    first = 0
    for index in range(1, len(markers) + 1):
        if index == len(markers) or not all(
            _is_spacing(char) for char in draft[markers[index - 1].end():markers[index].start()]
        ):
            groups.append((first, index - 1))
            first = index
    rendered = draft
    for first, last in reversed(groups):
        members = range(first, last + 1)
        if all(numbers[k] is None for k in members):
            continue
        group_end = markers[last].end()
        after = draft[group_end:group_end + 1]
        start = markers[first].start()
        visible = start
        while visible > 0 and _is_report_whitespace(draft[visible - 1]):
            visible -= 1
        line_start = positions[first]
        while line_start > 0 and not _is_line_break(stripped[line_start - 1]):
            line_start -= 1
        line_prefix = stripped[line_start:positions[first]]
        skip = None
        if after == "(":
            skip = "skipped_link_syntax"
        elif _BLOCK_PREFIX_RE.fullmatch(line_prefix):
            # Nothing but block syntax precedes the group on its line. A number
            # there could hide a sentence end from the span finder, or turn a
            # list item, heading or quote into prose.
            skip = "skipped_line_start"
        elif not after or _is_report_whitespace(after):
            # Glue to the last visible character so no word is added, unless
            # that would hide a sentence end from the span finder.
            if draft[visible - 1] in _SENTENCE_TERMINATORS:
                skip = "skipped_sentence_boundary"
            else:
                start = visible
        lead = ""
        if skip is None and start > 0 and not _is_report_whitespace(draft[start - 1]):
            if after and not _is_report_whitespace(after):
                # Followed by text with no whitespace before it, the group's
                # inner spacing alone kept the claim apart from that text: it
                # moves in front of the numbers instead of being removed.
                lead = "".join(draft[markers[k].end():markers[k + 1].start()]
                               for k in range(first, last))
            if not lead:
                # The first number joins the visible token before the group.
                token = line_prefix.split()[-1]
                if _BARE_LINK_RE.match(token):
                    skip = "skipped_link_syntax"  # an autolink would swallow it
                elif (len(token) - len(token.rstrip("\\"))) % 2:
                    skip = "skipped_parser_disagreement"  # an escaped [N] is no number
        if skip:
            for k in members:
                if numbers[k] is not None:
                    telemetry[skip] += 1
                    numbers[k] = None
            continue
        group = lead + "".join(
            (f"[{numbers[k]}]" if numbers[k] is not None else "") + markers[k].group(0)
            for k in members
        )
        rendered = rendered[:start] + group + rendered[group_end:]

    keys: Dict[tuple[int, str], int] = {}
    parts: list[str] = []
    cursor = removed = 0
    rendered_spans = _find_code_spans(rendered)
    rendered_markers = [m for m in _CITE_MARKER_RE.finditer(rendered)
                        if not _is_inside_span(m.start(), rendered_spans)]
    if len(rendered_markers) != len(markers):
        raise ValueError("reader-number rendering changed the inline markers")
    for assigned, marker in zip(numbers, rendered_markers):
        parts.append(rendered[cursor:marker.start()])
        cursor = marker.end()
        if assigned is not None:
            keys[(marker.start() - removed, marker.group(1))] = assigned
        removed += marker.end() - marker.start()
    parts.append(rendered[cursor:])
    words_after = count_report_words("".join(parts))
    if words_after != telemetry["words_before"]:
        raise ValueError(
            f"reader-number rendering changed the word count "
            f"({telemetry['words_before']} -> {words_after})"
        )
    return rendered, keys


def _reader_number_fates(
    *,
    kept: Sequence[CitationAnnotation],
    persisted: Sequence[CitationAnnotation],
    dropped: Sequence[Mapping[str, Any]],
    quarantined: Sequence[CitationAnnotation],
    applied_edit_spans: Sequence[tuple[int, int]],
    critique_input: str,
    report_content: str,
    expected_urls: AbstractSet[str],
    superscripts: bool | None = None,
) -> Dict[str, int]:
    """Where each citation whose reader number was rendered ended up (S101 m02).

    ``kept`` indexes ``dropped`` and holds coordinates in ``critique_input``,
    as the applied edit spans do; ``persisted`` holds the rebased survivors.
    The rebase binds cited text in document order, so a loss belongs to the
    critique whenever an applied edit removed any occurrence of that text.
    Finalize passes the selected pass's superscript reading; without one it is
    decided on ``report_content`` (S103 m02).
    """

    if superscripts is None:
        superscripts = reads_superscripts(report_content)
    fates: Counter[str] = Counter()
    fates["quarantined"] = sum(a._reader_number_rendered is not None for a in quarantined)

    def _edited(start: int, end: int) -> bool:
        return any(start < edit_end and edit_start < end
                   for edit_start, edit_end in applied_edit_spans)

    for record in dropped:
        annotation = kept[int(record["annotation_index"])]
        if annotation._reader_number_rendered is None:
            continue
        spans = [(annotation.start_index, annotation.end_index)]
        cited = critique_input[annotation.start_index:annotation.end_index].strip()
        found = critique_input.find(cited) if cited else -1
        while found != -1:
            spans.append((found, found + len(cited)))
            found = critique_input.find(cited, found + 1)
        edited = any(_edited(start, end) for start, end in spans)
        fates["lost_with_critique_edit" if edited else "lost_at_rebase"] += 1
    rendered_survivors = [a for a in persisted if a._reader_number_rendered is not None]
    entries = _parse_references_entries(
        report_content, expected_urls=expected_urls,
    ) if rendered_survivors else None
    numbered = entries.numbered_urls if entries else {}
    for annotation in rendered_survivors:
        number = annotation._reader_number_rendered
        mapped = number in _boundary_numbers(
            report_content, annotation, superscripts=superscripts) and (
            _reference_url_identity(annotation.url) in numbered.get(number, set())
        )
        fates["mapped" if mapped else "unmapped"] += 1
    return {fate: fates[fate] for fate in _READER_NUMBER_FATES}


def _annotate_references_and_check_consistency(
    markdown: str,
    *,
    kept_annotations: Sequence[CitationAnnotation],
    downgraded_dead: Sequence[CitationAnnotation],
    quarantined: Sequence[CitationAnnotation],
    citation_reference_markdown: str | None = None,
    require_number_mapping: bool = False,
    annotate_flags: bool = True,
    audit_quarantined_mapping: bool = True,
    numbered_fallback: bool = False,
    superscripts: bool | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """S84 m02 post-processing consistency pass.

    Annotates the agent-authored ``## References`` entries in place —
    downgraded-dead URLs get the unverified flag, quarantined-hallucinated
    URLs get the withdrawn flag (the numbered entry stays so body ``[n]``
    anchors keep resolving) — then records the agreement between the three
    citation surfaces (References list, protocol.citations, quarantined set)
    as a telemetry dict. CAT-R004 shipped these surfaces mutually out of
    sync; post-m02 the only structural divergence source is the hallucinated
    hard quarantine, which the withdrawn flag makes visible instead of
    silent. S92 m07 understands nested subsections and can verify the exact
    body ``[N]`` -> References URL mapping against persisted annotations.
    Missing mappings are reported, never guessed. Never raises. S103 m02:
    finalize passes the reading of superscripts it decided on the extracted
    draft; without one it is decided on ``markdown``.
    """

    kept_urls = {_reference_url_identity(a.url) for a in kept_annotations}
    downgraded_urls = {_reference_url_identity(a.url) for a in downgraded_dead}
    quarantined_urls = {_reference_url_identity(a.url) for a in quarantined}
    expected_urls = kept_urls | downgraded_urls | quarantined_urls

    consistency: Dict[str, Any] = {
        "references_section_found": False,
        "references_listed": 0,
        "protocol_citations": len(kept_urls),
        "unverified_flagged": 0,
        "withdrawn_flagged": 0,
        "missing_unverified_flags": 0,
        "missing_withdrawn_flags": 0,
        "protocol_only": 0,
        "quarantined_missing": 0,
        "references_only": 0,
        "numbered_references": 0,
        "body_mapping_checked": False,
        "quarantined_mapping_checked": False,
        "unmapped_protocol_citations": 0,
        "unmapped_quarantined_citations": 0,
        "reference_number_collisions": 0,
        "reader_markers_checked": require_number_mapping,
        "reader_markers": 0,
        "unresolved_reader_markers": 0,
        "dangling_reader_numbers": [],
        "ambiguous_year_labels": [],
        "unused_reference_numbers": [],
        "unsupported_reference_rows": 0,
        "annotation_coordinate_drift": 0,
        "annotations_without_reader_marker": 0,
        "references_entry_citations": 0,
        "references_entry_unmapped": 0,
        "superscripts_read": bool(superscripts),
        "superscript_runs_ignored": 0,
        "consistent": True,
    }

    try:
        if superscripts is None:
            superscripts = reads_superscripts(markdown)
            consistency["superscripts_read"] = superscripts
        entries = _parse_references_entries(markdown, expected_urls=expected_urls)
        if entries is None:
            numbers = reader_markers(
                markdown, superscripts=superscripts) if require_number_mapping else []
            consistency["superscript_runs_ignored"] = superscript_runs_ignored(
                markdown, superscripts=superscripts) if require_number_mapping else 0
            ambiguous = {n for n in numbers if 1900 <= n <= 2100}
            unresolved = [n for n in numbers if n not in ambiguous]
            consistency.update({
                "reader_markers": len(numbers),
                "unresolved_reader_markers": len(unresolved),
                "dangling_reader_numbers": sorted(set(unresolved)),
                "ambiguous_year_labels": sorted(ambiguous),
            })
            consistency.update(
                {
                    "protocol_only": len(kept_urls),
                    "quarantined_missing": len(quarantined_urls),
                    "body_mapping_checked": require_number_mapping,
                    "quarantined_mapping_checked": (
                        require_number_mapping and audit_quarantined_mapping
                    ),
                    "consistent": not kept_urls and not quarantined_urls and not unresolved,
                }
            )
            return markdown, consistency
        consistency["references_section_found"] = True

        lines = list(entries.lines)
        refs_urls = entries.refs_urls
        numbered_urls = entries.numbered_urls
        number_collisions = entries.number_collisions
        unsupported_reference_rows = entries.unsupported_reference_rows
        unverified_flagged = 0
        withdrawn_flagged = 0
        missing_unverified_flags = 0
        missing_withdrawn_flags = 0
        references_range = (entries.heading_start, entries.section_end)
        numbers = reader_markers(
            markdown, references_range, superscripts=superscripts,
        ) if require_number_mapping else []
        if require_number_mapping:
            consistency["superscript_runs_ignored"] = superscript_runs_ignored(
                markdown, references_range, superscripts=superscripts)
        for i, line_urls in entries.line_urls.items():
            # Independent flags: a nonstandard line carrying both a downgraded
            # and a quarantined URL gets BOTH suffixes (first-match-wins lost
            # the withdrawn flag — S84 m02 review fix).
            if line_urls & downgraded_urls:
                if _UNVERIFIED_REF_SUFFIX not in lines[i]:
                    if annotate_flags:
                        lines[i] = lines[i] + _UNVERIFIED_REF_SUFFIX
                    else:
                        missing_unverified_flags += 1
                if _UNVERIFIED_REF_SUFFIX in lines[i]:
                    unverified_flagged += 1
            if line_urls & quarantined_urls:
                if _WITHDRAWN_REF_SUFFIX not in lines[i]:
                    if annotate_flags:
                        lines[i] = lines[i] + _WITHDRAWN_REF_SUFFIX
                    else:
                        missing_withdrawn_flags += 1
                if _WITHDRAWN_REF_SUFFIX in lines[i]:
                    withdrawn_flagged += 1

        annotated = markdown[:entries.section_start] + "\n".join(lines)

        protocol_only = kept_urls - refs_urls
        quarantined_missing = quarantined_urls - refs_urls
        references_only = refs_urls - kept_urls - quarantined_urls
        unmapped_protocol_citations = 0
        unmapped_quarantined_citations = 0
        if require_number_mapping:
            consistency["body_mapping_checked"] = True
            reference_text = (
                citation_reference_markdown
                if citation_reference_markdown is not None
                else markdown
            )
            # S102 m04: an inline citation on a numbered References entry is
            # checked against that entry, whose label is its reader number. The
            # lines come from the audited text, which the coordinates refer to.
            reference_entries = (
                entries if reference_text is markdown
                else _parse_references_entries(reference_text, expected_urls=expected_urls)
            )
            entry_lines: list[tuple[int, int, set[str]]] = []
            if reference_entries is not None:
                line_start = reference_entries.section_start
                for index, line in enumerate(
                    reference_entries.lines[:reference_entries.section_lines]
                ):
                    if index in reference_entries.line_numbers:
                        entry_lines.append((
                            line_start, line_start + len(line),
                            reference_entries.line_urls[index],
                        ))
                    line_start += len(line) + 1

            def _count_unmapped(
                annotations: Sequence[CitationAnnotation],
            ) -> int:
                unmapped = 0
                seen_annotations: set[tuple[int, int, str]] = set()
                for annotation in annotations:
                    identity = _reference_url_identity(annotation.url)
                    annotation_key = (
                        int(annotation.start_index),
                        int(annotation.end_index),
                        identity,
                    )
                    if annotation_key in seen_annotations:
                        continue
                    seen_annotations.add(annotation_key)
                    if not (
                        0 <= annotation.start_index < annotation.end_index <= len(reference_text)
                    ):
                        consistency["annotation_coordinate_drift"] += 1
                        unmapped += 1
                        continue
                    # A numbered-fallback citation owns the marker after its span,
                    # so the boundary rule keeps binding it.
                    owns_suffix = annotation._extraction_mode == "numbered" or (
                        annotation._extraction_mode is None and numbered_fallback
                    )
                    entry_urls = None if owns_suffix else next(
                        (urls for start, end, urls in entry_lines
                         if start <= annotation.start_index < end),
                        None,
                    )
                    if entry_urls is not None:
                        consistency["references_entry_citations"] += 1
                        if identity not in entry_urls:
                            consistency["references_entry_unmapped"] += 1
                            unmapped += 1
                        continue
                    span_numbers = _boundary_numbers(
                        reference_text, annotation, superscripts=superscripts,
                        numbered_fallback=numbered_fallback,
                    )
                    if not span_numbers:
                        consistency["annotations_without_reader_marker"] += 1
                    if not span_numbers and not numbered_urls:
                        # Machine-only citation reports can carry an unnumbered
                        # Markdown-link References list. URL-set agreement is
                        # still audited; exact number binding is simply not an
                        # applicable surface for that supported format.
                        continue
                    if not any(
                        identity in numbered_urls.get(number, set())
                        for number in span_numbers
                    ):
                        unmapped += 1
                return unmapped

            unmapped_protocol_citations = _count_unmapped(kept_annotations)
            if audit_quarantined_mapping:
                consistency["quarantined_mapping_checked"] = True
                unmapped_quarantined_citations = _count_unmapped(quarantined)
        # A bracketed year in a source title is not enough to assert a citation
        # defect. Keep it visible as ambiguous unless an exact reference resolves it.
        ambiguous_years = {n for n in numbers if 1900 <= n <= 2100 and n not in numbered_urls}
        dangling = set(numbers) - set(numbered_urls) - ambiguous_years
        unused = sorted(set(numbered_urls) - set(numbers)) if require_number_mapping else []
        consistency.update(
            {
                "reader_markers": len(numbers),
                "ambiguous_year_labels": sorted(ambiguous_years),
                "unresolved_reader_markers": sum(n in dangling for n in numbers),
                "dangling_reader_numbers": sorted(dangling),
                "unused_reference_numbers": unused,
                "unsupported_reference_rows": unsupported_reference_rows,
                "references_listed": len(refs_urls),
                "unverified_flagged": unverified_flagged,
                "withdrawn_flagged": withdrawn_flagged,
                "missing_unverified_flags": missing_unverified_flags,
                "missing_withdrawn_flags": missing_withdrawn_flags,
                "protocol_only": len(protocol_only),
                "quarantined_missing": len(quarantined_missing),
                "references_only": len(references_only),
                "numbered_references": len(numbered_urls),
                "unmapped_protocol_citations": unmapped_protocol_citations,
                "unmapped_quarantined_citations": (
                    unmapped_quarantined_citations
                ),
                "reference_number_collisions": len(number_collisions),
                "consistent": (
                    not dangling
                    and not unused
                    and not unsupported_reference_rows
                    and not protocol_only
                    and not quarantined_missing
                    and not references_only
                    and not number_collisions
                    and not unmapped_protocol_citations
                    and not unmapped_quarantined_citations
                    and not missing_unverified_flags
                    and not missing_withdrawn_flags
                ),
            }
        )
        return annotated, consistency
    except Exception as exc:  # noqa: BLE001 — cleanup pass must never kill the mission
        safe_error = safe_exception_text(exc, category="internal")
        logger.error(
            "citation consistency pass failed: %s — shipping unannotated body",
            safe_error,
        )
        consistency["consistent"] = False
        consistency["error"] = safe_error
        return markdown, consistency


def _strip_leading_h1(markdown: str) -> str:
    """Return ``markdown`` with the leading top-level H1 line removed.

    S80 m03 (PD108) — every report carries TWO `# ` H1 lines pre-S80: one
    deterministic wrapper (`# Research Report: <id> - <title>`) prepended by
    ``build_report_header`` inside ``generate_markdown_report``, and one
    non-deterministic agent-chosen restating at the top of ``final_markdown``.
    The S79 review session locked the resolution: WRAPPER H1 WINS — the
    wrapper carries the canonical mission_id + normalized title (m02) and
    is the line you'd grep on; the agent-emitted H1 is a non-deterministic
    duplicate that adds no identity signal.

    Strip semantics — conservative, structural:

    * If the first non-blank line begins with ``# `` (a single hash + space),
      that line is removed AND any blank lines immediately following it are
      also removed.
    * ``## Heading``, ``### Heading``, etc. are unchanged (not H1s).
    * H1s mid-body are unchanged (only the LEADING one is stripped — content
      H1s would be unusual but the rule keeps the helper structural).
    * Multiple consecutive leading H1s — only the first is stripped (rare;
      pinned by parametrized test so the behavior cannot drift silently).
    * No leading H1 — returns the input unchanged.

    Per ``feedback_no_speculative_feature_flags.md`` — no flag for which H1
    wins.
    """
    lines = markdown.split("\n")

    # Skip preceding blank lines.
    cursor = 0
    while cursor < len(lines) and lines[cursor].strip() == "":
        cursor += 1

    if cursor >= len(lines):
        return markdown  # entirely blank input — nothing to strip

    # Test the leading non-blank line. ``lstrip`` allows for any indentation
    # before the hash (markdown allows up to 3 spaces of indent before a
    # heading marker; we accept the convention without enforcing the limit).
    first = lines[cursor].lstrip()
    if not first.startswith("# "):
        return markdown  # not an H1 — unchanged

    # Drop the leading H1 line.
    cursor += 1
    # Drop any blank lines immediately following the dropped H1 so the
    # subsequent content (typically H2/prose) attaches cleanly to the
    # wrapper-emitted preamble.
    while cursor < len(lines) and lines[cursor].strip() == "":
        cursor += 1

    return "\n".join(lines[cursor:])


def _render_report_metadata_preamble(meta: ReportMetadata) -> str:
    """Render the PD22 at-a-glance preamble — exactly 4 lines, no decoration.

    Format: bold-key + plain-value, one fact per line. No heading, no
    separator, no bullet markers. Inserted at the top of the synthesized
    content for gut-check; the forensic dict on ``protocol.report_metadata
    .forensic`` carries everything else for machine-readable consumers.
    """

    lines = [
        f"**Sources collected:** {meta.sources_collected}",
        f"**Verified references:** {meta.references}",
        f"**Agent steps:** {meta.agent_steps_used} / {meta.agent_steps_max}",
        f"**Runtime:** {_format_runtime_seconds(meta.runtime_seconds)}",
        "",
        "",
    ]
    return "\n".join(lines)


__all__ = ["codeagent_finalize_node"]
