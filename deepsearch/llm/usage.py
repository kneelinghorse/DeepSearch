"""Content-free provider usage observations; cache/reasoning are token subsets."""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalize_usage(raw: Any) -> dict[str, Any]:
    """Normalize observed counts, retaining missing fields and contradictory totals."""
    incoming = _count(_get(raw, "prompt_tokens", _get(raw, "input_tokens")))
    outgoing = _count(_get(raw, "completion_tokens", _get(raw, "output_tokens")))
    total = _count(_get(raw, "total_tokens"))
    prompt_details = _get(raw, "prompt_tokens_details", _get(raw, "input_token_details", {}))
    output_details = _get(raw, "completion_tokens_details", _get(raw, "output_token_details", {}))
    cached = _count(_get(raw, "prompt_cache_hit_tokens", _get(
        prompt_details, "cached_tokens", _get(prompt_details, "cache_read")
    )))
    cache_write = _count(_get(prompt_details, "cache_creation_tokens", _get(
        prompt_details, "cache_creation"
    )))
    miss = _count(_get(raw, "prompt_cache_miss_tokens"))
    reasoning = _count(_get(output_details, "reasoning_tokens", _get(output_details, "reasoning")))
    components = incoming is not None and outgoing is not None
    computed = incoming + outgoing if incoming is not None and outgoing is not None else None
    consistent = (total is None or computed is None or total == computed)
    if incoming is not None:
        consistent = consistent and (cached is None or cached <= incoming)
        consistent = consistent and (cache_write is None or cache_write <= incoming)
        if cached is not None and cache_write is not None:
            consistent = consistent and cached + cache_write <= incoming
        if cached is not None and miss is not None:
            consistent = consistent and cached + miss == incoming
    if outgoing is not None and reasoning is not None:
        consistent = consistent and reasoning <= outgoing
    return {
        "input": incoming, "output": outgoing, "total": total if total is not None else computed,
        "cache_read_input": cached, "cache_write_input": cache_write,
        "cache_miss_input": miss, "reasoning_output": reasoning,
        "complete": bool(components and consistent), "consistent": bool(consistent),
    }


def _usage_event(
    response: Any, *, config: Mapping[str, Any], stage: str, started_at: str,
    elapsed: float, error: BaseException | None = None, attempt: int = 1,
    step: int | None = None,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one request without content, credentials, headers or provider errors."""
    metadata = _get(response, "response_metadata", {}) or {}
    raw_usage = _get(response, "usage") or _get(metadata, "token_usage")
    if raw_usage is None:
        raw_usage = _get(response, "usage_metadata")
    returned = _get(response, "model", _get(metadata, "model_name"))
    returned = returned if (
        isinstance(returned, str) and re.fullmatch(r"[\w.:-]{1,120}", returned)
    ) else None
    choices = _get(response, "choices", []) or []
    finish = _get(choices[0], "finish_reason") if choices else _get(metadata, "finish_reason")
    allowed_finish = {"stop", "length", "tool_calls", "content_filter", "aborted",
                      "insufficient_system_resource", "function_call"}
    finish = finish if isinstance(finish, str) and finish in allowed_finish else None
    from deepsearch.llm.runtime import request_failure

    failure = request_failure(error, model=str(config.get("requested_model") or ""),
                              stage=stage) if error else None
    if failure is not None:
        failure["request_shape"] = _request_shape(request)
    return {
        "kind": "provider_usage", "version": 1, "event_id": uuid.uuid4().hex,
        "stage": stage, "attempt": attempt, "step": step,
        "requested_model": config.get("requested_model"), "returned_model": returned,
        "provider": config.get("provider"), "endpoint_family": "chat_completions",
        "parameters": {key: config.get(key) for key in (
            "max_output_tokens", "reasoning_effort", "temperature", "request_timeout_seconds"
        )},
        "started_at": started_at, "duration_seconds": round(elapsed, 6),
        "finish_reason": finish, "status": "failed" if error else "returned",
        "usage": normalize_usage(raw_usage),
        "failure": failure,
    }


def _request_shape(request: Mapping[str, Any] | None) -> dict[str, Any]:
    """Count the SDK input without retaining content, names, hashes or headers."""
    messages = request.get("messages") if request is not None else None
    if not isinstance(messages, list):
        return {"version": 1, "available": False}
    roles = dict.fromkeys(("system", "developer", "user", "assistant", "tool", "other"), 0)
    characters = 0
    unknown_content = 0
    # Bounded traversal and counts are explicit lower bounds when clipped.
    for message in messages[:4096]:
        if not isinstance(message, Mapping):
            roles["other"] += 1
            unknown_content += 1
            continue
        role = message.get("role")
        roles[role if isinstance(role, str) and role in roles else "other"] += 1
        content = message.get("content")
        if isinstance(content, str):
            characters += len(content)
        elif content is not None:
            unknown_content += 1
    tools = request.get("tools") if request is not None else None
    return {
        "version": 1, "available": True, "message_count": min(len(messages), 4096),
        "roles": roles, "text_characters": min(characters, 10_000_000),
        "nontext_or_unknown_messages": unknown_content,
        "tool_count": min(len(tools), 4096) if isinstance(tools, list) else None,
        "clipped": len(messages) > 4096 or characters > 10_000_000
        or (isinstance(tools, list) and len(tools) > 4096),
    }


def usage_event(response: Any, **kwargs: Any) -> dict[str, Any]:
    """Observe usage without allowing malformed provider metadata to erase work."""
    try:
        return _usage_event(response, **kwargs)
    except Exception:
        event = _usage_event(None, **kwargs)
        event["observation_error"] = True
        return event


def observe_model_requests(model: Any, config: Mapping[str, Any]) -> None:
    """Observe the installed non-streaming SDK boundary, including parse failures.

    SDK and smolagents retries are disabled by the existing model factory. Thus
    every invocation observed here is one HTTP request, including retained-memory
    finalization; response parsing cannot erase the original usage envelope.
    """
    model.provider_attempts = []
    model.usage_stage = "synthesis"
    model.usage_step = None
    model.usage_attempt = 1
    create = model.client.chat.completions.create

    def observed_create(*args: Any, **kwargs: Any) -> Any:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        response = None
        error = None
        try:
            response = create(*args, **kwargs)
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            model.provider_attempts.append(usage_event(
                response, config=config, stage=model.usage_stage, started_at=started,
                elapsed=time.monotonic() - t0, error=error,
                attempt=model.usage_attempt, step=model.usage_step,
                request=kwargs,
            ))

    model.client.chat.completions.create = observed_create


def compiler_usage_events(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read additive compiler observations from the existing contract audit field."""
    import json

    events = []
    for row in contract.get("entity_extraction_decisions", []) or []:
        if row.get("decision") != "provider_usage":
            continue
        try:
            record = json.loads(row["reason"])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(record, dict) and record.get("kind") == "provider_usage":
            events.append(record)
    return events


def summarize_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum independent requests exactly once; omitted usage stays an explicit gap."""
    unique = {e["event_id"]: e for e in events}
    observations = [e["usage"] for e in unique.values()]
    return {
        "requests": len(unique), "duplicate_records": len(events) - len(unique),
        "observed": {key: sum(u.get(key) or 0 for u in observations)
                     for key in ("input", "output", "total")},
        "usage_complete": bool(observations) and all(
            u.get("complete") is True for u in observations
        ),
        "requests_with_missing_or_inconsistent_usage": sum(
            u.get("complete") is not True for u in observations
        ),
    }
