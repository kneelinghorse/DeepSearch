"""Helpers for summarizing evidence-slot coverage from a working set."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


def extract_failing_slot_details(
    slot_coverage: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Return normalized thin/unaddressed slot diagnostics."""

    if not isinstance(slot_coverage, Mapping):
        return []

    raw_slots = slot_coverage.get("slots")
    if isinstance(raw_slots, list) and raw_slots:
        candidates = raw_slots
    else:
        candidates = [
            *(slot_coverage.get("thin_slots") or []),
            *(slot_coverage.get("unaddressed_slots") or []),
        ]

    diagnostics: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for slot in candidates:
        if not isinstance(slot, Mapping):
            continue
        status = str(slot.get("status") or "").strip().lower()
        if status not in {"thin", "unaddressed"}:
            continue
        slot_id = str(slot.get("slot_id") or slot.get("id") or "").strip()
        if not slot_id or slot_id in seen:
            continue
        seen.add(slot_id)
        diagnostics.append(
            {
                "slot_id": slot_id,
                "kind": str(slot.get("kind") or "").strip(),
                "status": status,
                "description": str(slot.get("description") or "").strip(),
                "source_count": int(slot.get("source_count", 0) or 0),
                "max_authority_score": round(_coerce_float(slot.get("max_authority_score")), 4),
                "target": dict(slot.get("target") or {}),
                "reasons": [
                    str(reason).strip()
                    for reason in slot.get("reasons") or []
                    if str(reason).strip()
                ],
                "domains": [
                    str(domain).strip()
                    for domain in slot.get("domains") or []
                    if str(domain).strip()
                ],
            }
        )
    diagnostics.sort(
        key=lambda item: (
            0 if item["status"] == "unaddressed" else 1,
            item["slot_id"],
        )
    )
    return diagnostics


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["extract_failing_slot_details"]
