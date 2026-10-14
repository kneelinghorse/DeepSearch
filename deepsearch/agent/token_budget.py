"""Prompt token budgeting helpers for per-loop execution ceilings."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple


def estimate_text_tokens(text: str) -> int:
    """Return a cheap token estimate from plain text."""

    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 3) // 4)


def accumulated_prompt_tokens(
    telemetry: Mapping[str, Any],
    *,
    loop_index: int,
) -> int:
    """Return prompt-token usage already attributed to the current loop."""

    total = 0
    for item in telemetry.get("llm_calls") or []:
        if not isinstance(item, Mapping):
            continue
        if int(item.get("loop_index") or -1) != loop_index:
            continue
        total += _coerce_int(item.get("prompt_tokens") or item.get("estimated_prompt_tokens"))
    return total


def apply_source_token_budget(
    *,
    sources: Sequence[Dict[str, Any]],
    prompt_factory: Callable[[Sequence[Dict[str, Any]]], Tuple[str, str]],
    telemetry: Dict[str, Any],
    loop_index: int,
    llm_config: Any,
    max_tokens_per_loop: int | None,
    node_name: str,
) -> Tuple[List[Dict[str, Any]], str, str, int]:
    """Trim source inputs until the next LLM call fits within the loop ceiling."""

    current_sources = [dict(source) for source in sources]
    system_prompt, human_prompt = prompt_factory(current_sources)
    prompt_tokens = estimate_text_tokens(system_prompt) + estimate_text_tokens(human_prompt)
    projected_total = (
        accumulated_prompt_tokens(telemetry, loop_index=loop_index)
        + prompt_tokens
        + _estimated_next_call_tokens(llm_config)
    )
    if max_tokens_per_loop is None or projected_total <= max_tokens_per_loop:
        return current_sources, system_prompt, human_prompt, prompt_tokens

    before_count = len(current_sources)
    while current_sources and projected_total > max_tokens_per_loop:
        current_sources = current_sources[:-1]
        system_prompt, human_prompt = prompt_factory(current_sources)
        prompt_tokens = estimate_text_tokens(system_prompt) + estimate_text_tokens(human_prompt)
        projected_total = (
            accumulated_prompt_tokens(telemetry, loop_index=loop_index)
            + prompt_tokens
            + _estimated_next_call_tokens(llm_config)
        )

    token_budget_events = telemetry.get("token_budget_events") or []
    token_budget_events.append(
        {
            "event": "token_budget_triggered",
            "node": node_name,
            "loop_index": loop_index,
            "max_tokens_per_loop": max_tokens_per_loop,
            "sources_before": before_count,
            "sources_after": len(current_sources),
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_total_tokens": projected_total,
        }
    )
    telemetry["token_budget_events"] = token_budget_events
    return current_sources, system_prompt, human_prompt, prompt_tokens


def _estimated_next_call_tokens(llm_config: Any) -> int:
    max_output_tokens = _coerce_int(getattr(llm_config, "max_output_tokens", 0))
    thinking_budget = _coerce_int(getattr(llm_config, "thinking_budget", 0))
    return max_output_tokens + thinking_budget


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "accumulated_prompt_tokens",
    "apply_source_token_budget",
    "estimate_text_tokens",
]
