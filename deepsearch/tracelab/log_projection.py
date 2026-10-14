"""Identity-free structured observations shared by the child pipe and HTTP sink."""
from __future__ import annotations

import math
from typing import Any, Mapping

from deepsearch.error_safety import SAFE_ERROR_CATEGORIES

_SAFE_PHASES = frozenset(
    {"loading", "compiled", "running", "critique", "upload", "complete", "error"}
)
_SAFE_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_SAFE_LOG_CATEGORIES = SAFE_ERROR_CATEGORIES | frozenset(
    {"debug", "info", "warning", "error", "critical"}
)
_SAFE_EVENTS = frozenset(
    {
        "mission_progress",
        "mission_warning",
        "mission_error",
        "phase_started",
        "phase_completed",
        "artifact_persisted",
        "validation_result",
    }
)
_MAX_SAFE_NUMERIC_VALUE = 1_000_000_000.0



def project_log_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only fixed enums and finite, bounded measurements across a boundary."""

    projected: dict[str, Any] = {}
    for name, allowed in (("phase", _SAFE_PHASES), ("event", _SAFE_EVENTS),
                          ("category", _SAFE_LOG_CATEGORIES)):
        value = fields.get(f"tracelab_{name}")
        if isinstance(value, str) and value in allowed:
            projected[f"tracelab_{name}"] = value
    for name in ("count", "duration_seconds"):
        value = fields.get(f"tracelab_{name}")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and 0 <= value <= _MAX_SAFE_NUMERIC_VALUE):
            projected[f"tracelab_{name}"] = value
    return projected
