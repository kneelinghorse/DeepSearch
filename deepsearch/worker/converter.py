"""Row-to-AgentState converter for TraceLab mission database rows.

This module converts MissionRow objects (from PostgreSQL polling) into
AgentState objects compatible with the DeepSearch LangGraph workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from deepsearch.agent.state import AgentState, empty_state
from deepsearch.mission.title_utils import normalize_mission_title

from .poller import MissionRow

logger = logging.getLogger(__name__)

# Baseline defaults (used when depth not specified)
DEFAULT_MAX_LOOPS = 3
DEFAULT_MIN_LOOPS = 2
DEFAULT_TOKEN_LIMIT = 8000
DEFAULT_DELIVERABLE_FORMAT = "markdown"


class ConversionError(Exception):
    """Raised when mission row cannot be converted to AgentState."""
    pass


def convert_row_to_state(row: MissionRow) -> AgentState:
    """Convert a TraceLab MissionRow to a DeepSearch AgentState.

    Maps all relevant fields from the database row to the AgentState format
    expected by the LangGraph research workflow.

    Args:
        row: MissionRow instance from the mission poller

    Returns:
        AgentState initialized with mission data, ready for research execution

    Raises:
        ConversionError: If required fields are missing or invalid
    """
    if not row.id:
        raise ConversionError("Mission row missing required 'id' field")

    if not row.title:
        raise ConversionError("Mission row missing required 'title' field")

    # Extract objectives - required for research
    objectives = _extract_objectives(row)
    if not objectives:
        raise ConversionError(
            f"Mission {row.id} has no objectives defined - cannot execute research"
        )

    # Build mission context from row data
    mission_context = _build_mission_context(row)

    # Extract configuration values with defaults
    max_loops = _extract_max_loops(row)
    min_loops = _extract_min_loops(row)
    deliverable_format = _extract_deliverable_format(row)
    research_depth = _extract_depth(row)

    logger.info(
        "Converting mission %s (%s): %d objectives, max_loops=%d, depth=%s",
        row.mission_id,
        row.title,
        len(objectives),
        max_loops,
        research_depth,
    )

    return empty_state(
        mission_id=row.mission_id,  # Use human-readable mission_id
        # S80 m04 (PD109): propagate the TraceLab system-assigned UUID into
        # AgentState so the per-run report and protocol filenames pick up an
        # 8-hex-char disambiguator. Pre-S80 same-slug re-runs silently
        # overwrote the prior run's `.md` and `_protocol.json` on disk;
        # `row.id` is the only field that distinguishes two re-runs of the
        # same mission_id slug at the boundary contract.
        mission_uuid=row.id,
        mission_objectives=objectives,
        project_id=None,  # TraceLab manages project association internally
        mission_context=mission_context,
        deliverable_format=deliverable_format,
        research_depth=research_depth,
        max_loops=max_loops,
        min_loops=min_loops,
    )


def _extract_objectives(row: MissionRow) -> List[str]:
    """Extract research objectives from the mission row.

    When explicit success criteria are present, use them as the runtime
    objectives and treat ``row.objective`` as umbrella context only. This keeps
    reflection/query planning aligned with the compiled contract, where the
    top-level objective becomes optional when success criteria exist.

    Args:
        row: MissionRow with objective data

    Returns:
        List of objective strings, filtered for non-empty values
    """
    objectives: List[str] = []

    if row.success_criteria:
        for criterion in row.success_criteria:
            if isinstance(criterion, str) and criterion.strip():
                objectives.append(criterion.strip())
    elif row.objective and row.objective.strip():
        objectives.append(row.objective.strip())

    # Fallback: check context for additional success_criteria
    if not objectives and row.context:
        success_criteria = row.context.get("success_criteria") or row.context.get("successCriteria")
        if isinstance(success_criteria, list):
            objectives = [
                str(item).strip() for item in success_criteria
                if item and str(item).strip()
            ]

    return objectives


def _build_mission_context(row: MissionRow) -> Dict[str, Any]:
    """Build mission context dictionary from row data.

    Merges row.context JSONB with title, objective, and other metadata.
    Preserves all TraceLab context fields for use in research/reporting.

    Args:
        row: MissionRow with context and metadata

    Returns:
        Dictionary suitable for AgentState.mission_context
    """
    context: Dict[str, Any] = {}

    # Merge in context from JSONB column
    if row.context and isinstance(row.context, dict):
        context.update(row.context)

    normalized_title = normalize_mission_title(
        str(context.get("title") or row.title),
        row.mission_id,
    )

    # Always set title and objective at top level
    context["title"] = normalized_title
    context["objective"] = row.objective or normalized_title
    context["mission_id"] = row.mission_id

    # Preserve TraceLab metadata
    if row.id:
        context["tracelab_uuid"] = row.id
    if row.queued_at:
        context["queued_at"] = row.queued_at.isoformat()

    # Include research_phases if present
    if row.research_phases:
        context.setdefault("research_phases", row.research_phases)

    # Include success_criteria if present
    if row.success_criteria:
        context.setdefault("success_criteria", row.success_criteria)
    if row.deliverables:
        context.setdefault("deliverables", row.deliverables)

    # Dedicated authoring columns are authoritative when non-null. Assigning
    # rather than using truthiness/setdefault is intentional: an authored
    # empty list/object clears a legacy value nested in ``context``; only a
    # NULL column permits that legacy fallback.
    for field_name in (
        "background",
        "focus",
        "references",
        "required_entities",
        "excluded_entities",
        "expected_output_schema",
        "coverage_thresholds",
        "validation_thresholds",
        "deliverable_format",
        "max_loops",
        "min_loops",
        "constraints",
    ):
        value = getattr(row, field_name)
        if value is not None:
            context[field_name] = value

    return context


def _extract_max_loops(row: MissionRow) -> int:
    """Extract max_loops configuration value.

    Checks the dedicated authoring column, research_phases, then context,
    before falling back to the default.

    Args:
        row: MissionRow with potential config

    Returns:
        Maximum research loops to execute
    """
    if row.max_loops is not None:
        return row.max_loops if row.max_loops > 0 else DEFAULT_MAX_LOOPS

    if row.research_phases:
        for key in ("max_loops", "maxLoops"):
            value = row.research_phases.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str) and value.isdigit():
                parsed = int(value)
                if parsed > 0:
                    return parsed

    # Check context next
    if row.context:
        for key in ("max_loops", "maxLoops"):
            value = row.context.get(key)
            if isinstance(value, int) and value > 0:
                return value

    return DEFAULT_MAX_LOOPS


def _extract_min_loops(row: MissionRow) -> int:
    """Extract min_loops configuration value.

    Args:
        row: MissionRow with potential config

    Returns:
        Minimum research loops before convergence can trigger. Falls back to
        depth tier defaults when not explicitly configured.
    """
    if row.min_loops is not None:
        return row.min_loops if row.min_loops > 0 else DEFAULT_MIN_LOOPS

    if row.research_phases:
        for key in ("min_loops", "minLoops"):
            value = row.research_phases.get(key)
            if isinstance(value, int) and value > 0:
                return value

    if row.context:
        for key in ("min_loops", "minLoops"):
            value = row.context.get(key)
            if isinstance(value, int) and value > 0:
                return value

    return DEFAULT_MIN_LOOPS


def _extract_deliverable_format(row: MissionRow) -> str:
    """Extract deliverable format preference.

    Args:
        row: MissionRow with potential config

    Returns:
        Deliverable format string (e.g., "markdown", "json")
    """
    if row.deliverable_format is not None:
        return row.deliverable_format.strip().lower() or DEFAULT_DELIVERABLE_FORMAT

    if row.research_phases:
        for key in ("deliverable_format", "deliverableFormat", "format"):
            value = row.research_phases.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    if row.context:
        for key in ("deliverable_format", "deliverableFormat"):
            value = row.context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    return DEFAULT_DELIVERABLE_FORMAT


def _extract_depth(row: MissionRow) -> str:
    """Extract research depth label from the mission row.

    Priority order:
    1. row.research_depth (direct column from TraceLab)
    2. research_phases JSONB
    3. context JSONB
    4. "baseline" default

    Boundary string carrier post-S76 — no in-process dispatch.
    """
    if row.research_depth and row.research_depth.strip():
        return _normalize_depth_label(row.research_depth)

    if row.research_phases:
        for key in ("depth", "research_depth", "researchDepth"):
            value = row.research_phases.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_depth_label(value)

    if row.context:
        for key in ("depth", "research_depth", "researchDepth"):
            value = row.context.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_depth_label(value)

    return "baseline"


def _normalize_depth_label(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"baseline", "deep", "alpha"}:
        return "baseline"
    return normalized


__all__ = [
    "convert_row_to_state",
    "ConversionError",
    "DEFAULT_MAX_LOOPS",
    "DEFAULT_MIN_LOOPS",
    "DEFAULT_TOKEN_LIMIT",
    "DEFAULT_DELIVERABLE_FORMAT",
    "_extract_depth",
]
