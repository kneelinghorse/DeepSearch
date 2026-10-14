"""Telemetry helpers for node-level and mission-level telemetry tracking."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict

from .slot_coverage import extract_failing_slot_details

_USAGE_KEYS = {
    "prompt_tokens": "prompt_tokens",
    "input_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "output_tokens": "completion_tokens",
    "total_tokens": "total_tokens",
    "thoughts_tokens": "thinking_tokens",
    "thinking_tokens": "thinking_tokens",
    "candidates_tokens": "candidate_tokens",
    "candidate_tokens": "candidate_tokens",
}


def clone_telemetry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy telemetry dict from state for safe mutation."""
    telemetry: Dict[str, Any] = {}
    existing = state.get("telemetry")
    if isinstance(existing, dict):
        telemetry.update(deepcopy(existing))
    api_calls = telemetry.get("api_calls") or {}
    token_usage = telemetry.get("token_usage") or {}
    depth_metrics = telemetry.get("depth_metrics") or {}
    llm_calls = telemetry.get("llm_calls") or []
    coverage_gate_trace = telemetry.get("coverage_gate_trace") or []
    telemetry["api_calls"] = dict(api_calls)
    telemetry["token_usage"] = dict(token_usage)
    telemetry["depth_metrics"] = dict(depth_metrics) if isinstance(depth_metrics, Mapping) else {}
    telemetry["llm_calls"] = [
        dict(item) if isinstance(item, Mapping) else item
        for item in llm_calls
    ]
    telemetry["coverage_gate_trace"] = [
        dict(item) if isinstance(item, Mapping) else item
        for item in coverage_gate_trace
    ]
    telemetry["estimated_cost_usd"] = float(telemetry.get("estimated_cost_usd") or 0.0)
    return telemetry


def normalize_final_telemetry(
    state: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Normalize the persisted mission telemetry payload from final state.

    Reflection stores some history on top-level state while other signals live
    inside the nested telemetry dict. This helper projects both into the final
    JSONL payload so runner and worker persistence stay in sync.
    """

    raw = telemetry if isinstance(telemetry, Mapping) else state.get("telemetry")
    normalized: Dict[str, Any] = (
        deepcopy(dict(raw))
        if isinstance(raw, Mapping)
        else {}
    )
    normalized["api_calls"] = _normalize_int_map(normalized.get("api_calls"))
    normalized["token_usage"] = _normalize_int_map(normalized.get("token_usage"))
    normalized["routing_trace"] = _normalize_mapping_list(normalized.get("routing_trace"))
    normalized["coverage_gate_trace"] = _normalize_mapping_list(
        normalized.get("coverage_gate_trace")
    )
    normalized["query_telemetry"] = _normalize_mapping_list(
        normalized.get("query_telemetry")
    )
    normalized["token_budget_events"] = _normalize_mapping_list(
        normalized.get("token_budget_events")
    )

    objective_history = state.get("objective_score_history")
    if not isinstance(objective_history, list):
        objective_history = normalized.get("objective_score_history")
    normalized["objective_score_history"] = _normalize_mapping_list(objective_history)

    convergence_history = state.get("convergence_history")
    if not isinstance(convergence_history, list):
        convergence_history = normalized.get("convergence_history")
    normalized["convergence_history"] = _normalize_float_list(convergence_history)
    structural_validation_failures = state.get("structural_validation_failures")
    if not isinstance(structural_validation_failures, list):
        synthesis_telemetry = state.get("synthesis_telemetry")
        if isinstance(synthesis_telemetry, Mapping):
            structural_validation_failures = synthesis_telemetry.get(
                "structural_validation_failed_schemas"
            )
        else:
            structural_validation_failures = normalized.get(
                "structural_validation_failures"
            )
    normalized["structural_validation_failures"] = _normalize_string_list(
        structural_validation_failures
    )
    slot_diagnostics = _build_slot_diagnostics(state)
    if slot_diagnostics:
        normalized["slot_diagnostics"] = slot_diagnostics
    finalize_diagnostics = _build_finalize_diagnostics(state)
    if finalize_diagnostics:
        normalized["finalize_diagnostics"] = finalize_diagnostics

    normalized["estimated_cost_usd"] = float(normalized.get("estimated_cost_usd") or 0.0)
    return normalized


def increment_api_call(telemetry: Dict[str, Any], key: str, count: int = 1) -> None:
    """Increment an API call counter in the telemetry dict."""
    api_calls = telemetry.get("api_calls") or {}
    current = int(api_calls.get(key, 0))
    api_calls[key] = current + max(1, count)
    telemetry["api_calls"] = api_calls


def update_finding_counts(
    telemetry: Dict[str, Any],
    *,
    raw_findings: int,
    unique_concepts: int,
    cluster_count: int,
    loop_index: int | None = None,
) -> None:
    """Persist raw-vs-unique finding telemetry on the mission payload."""

    raw_count = max(int(raw_findings), 0)
    unique_count = max(min(int(unique_concepts), raw_count), 0)
    cluster_total = max(int(cluster_count), 0)
    dedup_ratio = round((raw_count - unique_count) / raw_count, 4) if raw_count else 0.0

    finding_counts = telemetry.get("finding_counts")
    if not isinstance(finding_counts, Mapping):
        finding_counts = {}
    payload = dict(finding_counts)
    payload.update(
        {
            "raw_findings": raw_count,
            "unique_concepts": unique_count,
            "cluster_count": cluster_total,
            "deduped_findings": unique_count,
            "dedup_ratio": dedup_ratio,
        }
    )

    if loop_index is not None:
        loop_payload = {
            "loop_index": max(int(loop_index), 0),
            "raw_findings": raw_count,
            "unique_concepts": unique_count,
            "cluster_count": cluster_total,
            "deduped_findings": unique_count,
            "dedup_ratio": dedup_ratio,
        }
        existing_loops = payload.get("loops")
        normalized_loops = (
            [dict(item) for item in existing_loops if isinstance(item, Mapping)]
            if isinstance(existing_loops, list)
            else []
        )
        for idx, entry in enumerate(normalized_loops):
            if int(entry.get("loop_index", -1) or -1) == loop_payload["loop_index"]:
                normalized_loops[idx] = loop_payload
                break
        else:
            normalized_loops.append(loop_payload)
        normalized_loops.sort(key=lambda item: int(item.get("loop_index", 0) or 0))
        payload["loops"] = normalized_loops

    telemetry["finding_counts"] = payload


def record_llm_usage(
    telemetry: Dict[str, Any],
    message: Any,
    *,
    call_type: str | None = None,
    llm_config: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Record token usage from an LLM response message."""
    usage_summary = _extract_usage_metadata(message)
    if usage_summary:
        token_usage = telemetry.get("token_usage") or {}
        for key, value in usage_summary.items():
            token_usage[key] = int(token_usage.get(key, 0)) + int(value)
        telemetry["token_usage"] = token_usage

    if not usage_summary and call_type is None and llm_config is None and not metadata:
        return

    llm_calls = telemetry.get("llm_calls") or []
    record: Dict[str, Any] = {}
    if call_type:
        record["call_type"] = call_type
    if llm_config is not None:
        for attr in ("backend", "model", "max_output_tokens", "thinking_budget"):
            value = getattr(llm_config, attr, None)
            if value is not None:
                record[attr] = value
    record.update(usage_summary)
    if metadata:
        record.update(dict(metadata))
    llm_calls.append(record)
    telemetry["llm_calls"] = llm_calls


def _normalize_int_map(value: Any) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    if not isinstance(value, Mapping):
        return normalized
    for key, item in value.items():
        try:
            normalized[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_mapping_list(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _build_slot_diagnostics(state: Mapping[str, Any]) -> Dict[str, Any]:
    slot_coverage = state.get("slot_coverage")
    failed_slots = extract_failing_slot_details(slot_coverage if isinstance(slot_coverage, Mapping) else None)
    if not failed_slots:
        return {}
    return {
        "failed_slot_ids": [str(slot["slot_id"]) for slot in failed_slots],
        "failed_slot_kinds": _unique_string_list(slot.get("kind") for slot in failed_slots),
        "failed_slots": failed_slots,
    }


def _build_finalize_diagnostics(state: Mapping[str, Any]) -> Dict[str, Any]:
    synthesis_telemetry = state.get("synthesis_telemetry")
    if not isinstance(synthesis_telemetry, Mapping):
        return {}

    diagnostics: Dict[str, Any] = {}
    report_render_path = str(synthesis_telemetry.get("report_render_path") or "").strip()
    report_render_reason = str(synthesis_telemetry.get("report_render_reason") or "").strip()
    if report_render_path:
        diagnostics["report_render_path"] = report_render_path
    if report_render_reason:
        diagnostics["report_render_reason"] = report_render_reason

    if "finalize_contract_present" in synthesis_telemetry:
        diagnostics["contract_present"] = bool(synthesis_telemetry.get("finalize_contract_present"))
    if "finalize_has_deliverable_schemas" in synthesis_telemetry:
        diagnostics["has_deliverable_schemas"] = bool(
            synthesis_telemetry.get("finalize_has_deliverable_schemas")
        )

    schema_ids = _normalize_string_list(
        synthesis_telemetry.get("finalize_deliverable_schema_ids")
    )
    if schema_ids:
        diagnostics["deliverable_schema_ids"] = schema_ids

    try:
        schema_count = int(synthesis_telemetry.get("finalize_deliverable_schema_count", 0) or 0)
    except (TypeError, ValueError):
        schema_count = 0
    if schema_count or "finalize_deliverable_schema_count" in synthesis_telemetry:
        diagnostics["deliverable_schema_count"] = schema_count

    return diagnostics


def _unique_string_list(values: Any) -> list[str]:
    if not isinstance(values, list) and not isinstance(values, tuple):
        values = list(values) if values is not None and not isinstance(values, (str, bytes)) else []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _normalize_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    normalized: list[float] = []
    for item in value:
        try:
            normalized.append(float(item))
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _extract_usage_metadata(message: Any) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    for attr in ("usage_metadata", "response_metadata"):
        data = getattr(message, attr, None)
        if isinstance(data, Mapping):
            usage.update(_normalize_usage_dict(data))
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, Mapping):
        for key in ("usage_metadata", "token_usage", "usage"):
            extras = additional.get(key)
            if isinstance(extras, Mapping):
                usage.update(_normalize_usage_dict(extras))
    return usage


def _normalize_usage_dict(source: Mapping[str, Any]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for raw_key, canonical in _USAGE_KEYS.items():
        value = source.get(raw_key)
        if value is None:
            continue
        try:
            normalized[canonical] = normalized.get(canonical, 0) + int(value)
        except (TypeError, ValueError):
            continue
    total = source.get("total_tokens")
    if total is not None:
        try:
            normalized["total_tokens"] = int(total)
        except (TypeError, ValueError):
            pass
    return normalized
