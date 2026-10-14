"""Dated cost estimates over persisted usage, isolated from production runtime."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

PRICE_SNAPSHOT = Path(__file__).parent / "fixtures/s93/prices.json"


def estimate_request(
    event: Mapping[str, Any], *, prices: Mapping[str, Any] | None = None,
    chinese_public_holiday: bool | None = None,
) -> dict[str, Any]:
    """Estimate a request's dollar range without adding token subsets twice.

    Missing usage/rates cannot produce a finite upper bound. Missing cache writes,
    cache hits or holiday knowledge widen the estimate; billing is never inferred
    from a model alias alone. Long-context multipliers apply to the full request.
    """
    prices = prices if prices is not None else json.loads(PRICE_SNAPSHOT.read_text())
    rates = prices.get(str(event.get("requested_model")))
    usage = event.get("usage") or {}
    result = {"as_of": prices.get("as_of"), "usd_min": 0.0, "usd_max": None,
              "exact_estimate": False, "basis": "published_requested_model_rates"}
    if not isinstance(rates, Mapping):
        return {**result, "reason": "unknown_rate"}
    if usage.get("complete") is not True:
        return {**result, "reason": "missing_or_inconsistent_usage"}
    incoming, outgoing = usage["input"], usage["output"]
    cached, writes = usage.get("cache_read_input"), usage.get("cache_write_input")
    cached_range = [0, incoming] if cached is None else [cached]
    costs: list[float] = []
    multipliers = [1.0]
    if "off_peak_multiplier" in rates:
        multipliers = [rates["off_peak_multiplier"], 1.0]
        try:
            start = datetime.fromisoformat(str(event["started_at"]).replace("Z", "+00:00"))
            from datetime import timezone

            if start.tzinfo is None:
                raise ValueError("Unknown billing timezone")
            start = start.astimezone(timezone.utc)
            end = start + timedelta(seconds=float(event.get("duration_seconds") or 0))
            # A call spanning a rate boundary retains both possible rates.
            def peak(moment: datetime) -> bool:
                return moment.weekday() < 5 and any(
                    a <= moment.hour < b for a, b in rates["peak_utc_weekdays"]
                )
            if not peak(start) and not peak(end) and end - start < timedelta(hours=1):
                multipliers = [rates["off_peak_multiplier"]]
            elif chinese_public_holiday is True:
                multipliers = [rates["off_peak_multiplier"]]
            elif (peak(start) and peak(end) and chinese_public_holiday is False
                  and end - start < timedelta(hours=1)):
                multipliers = [1.0]
        except (KeyError, TypeError, ValueError):
            pass
    long = incoming > rates.get("long_context_threshold", float("inf"))
    input_multiplier = rates.get("long_context_input_multiplier", 1) if long else 1
    output_multiplier = rates.get("long_context_output_multiplier", 1) if long else 1
    for hit in cached_range:
        noncached = incoming - hit
        write_range = [0, noncached] if writes is None else [writes]
        for written in write_range:
            if written > noncached:
                continue
            input_cost = (hit * rates["cached_input"] + (noncached - written) * rates["input"]
                          + written * rates["input"] * rates["cache_write_multiplier"])
            base = input_cost * input_multiplier + outgoing * rates["output"] * output_multiplier
            costs.extend(base * factor / 1_000_000 for factor in multipliers)
    if not costs:
        return {**result, "reason": "inconsistent_cache_usage"}
    low, high = round(min(costs), 9), round(max(costs), 9)
    return {**result, "usd_min": low, "usd_max": high,
            "exact_estimate": low == high, "reason": "observed_tokens_rate_estimate"}


def estimate_all_attempts(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Include failed/rejected requests once, with explicit unpriced components."""
    unique = {e["event_id"]: e for e in events}
    estimates = [estimate_request(event) for event in unique.values()]
    return {
        "requests": len(unique), "model_usd_min": round(sum(e["usd_min"] for e in estimates), 9),
        "model_usd_max": (round(sum(e["usd_max"] for e in estimates), 9)
                          if estimates and all(e["usd_max"] is not None for e in estimates)
                          else None),
        "unpriced_requests": sum(e["usd_max"] is None for e in estimates),
        "retrieval_usd": None, "human_repair_minutes": None, "actual_invoice_usd": None,
    }


def collect_attempt_events(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect current and retained prior lease attempts, deduplicated by event ID."""
    events: dict[str, dict[str, Any]] = {}
    for prior in metadata.get("prior_execution_attempts", []) or []:
        for event in collect_attempt_events(prior):
            events[event["event_id"]] = event
    synthesis = metadata.get("synthesis_telemetry") or {}
    accounting = (synthesis.get("recovery") or {}).get("attempt_accounting") or {}
    critique = synthesis.get("critique_telemetry") or {}
    for records in (synthesis.get("provider_attempts"), synthesis.get("compiler_provider_attempts"),
                    critique.get("provider_attempts"), accounting.get("provider_attempts")):
        for event in records or []:
            if isinstance(event, dict) and isinstance(event.get("event_id"), str):
                events[event["event_id"]] = event
    return list(events.values())
