"""Mission Protocol YAML loader utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from pydantic import ValidationError

from deepsearch.agent.state import AgentState, empty_state
from deepsearch.mission.schema import MissionYAML
from deepsearch.mission.title_utils import normalize_mission_title

logger = logging.getLogger(__name__)

DEFAULT_DELIVERABLE_FORMAT = "markdown"
DEFAULT_MAX_LOOPS = 3


class MissionLoadError(RuntimeError):
    """Raised when mission metadata cannot be loaded or validated."""


def load_mission_from_yaml(path: str | Path, *, encoding: str = "utf-8") -> AgentState:
    """Load a Mission Protocol YAML file and return an initialised ``AgentState``."""

    mission_path = Path(path)
    if not mission_path.exists():
        raise MissionLoadError(f"Mission file not found: {mission_path}")

    try:
        payload = yaml.safe_load(mission_path.read_text(encoding=encoding))
    except yaml.YAMLError as exc:  # pragma: no cover - depends on YAML parser internals
        raise MissionLoadError(f"Failed to parse YAML: {mission_path}") from exc

    if not isinstance(payload, Mapping):
        raise MissionLoadError("Mission YAML must define a mapping at the top level")

    mission = validate_mission_yaml(payload)
    return _build_agent_state(mission)


def validate_mission_yaml(data: Mapping[str, Any]) -> MissionYAML:
    """Validate raw mission data and return a ``MissionYAML`` model."""

    if not isinstance(data, Mapping):
        raise MissionLoadError("Mission payload must be a mapping")

    try:
        return MissionYAML.model_validate(data)
    except ValidationError as exc:
        formatted = _format_validation_errors(exc)
        raise MissionLoadError(formatted) from exc


def _build_agent_state(mission: MissionYAML) -> AgentState:
    """Transform a ``MissionYAML`` model into an ``AgentState``."""

    normalized_title = normalize_mission_title(mission.title, mission.mission_id)
    objectives = [item for item in mission.success_criteria if item]
    if not objectives:
        objectives = [mission.objective]

    context_payload = dict(mission.context)
    context_payload.setdefault("objective", mission.objective)
    context_payload["title"] = normalize_mission_title(
        str(context_payload.get("title") or normalized_title),
        mission.mission_id,
    )
    if mission.success_criteria:
        context_payload.setdefault("success_criteria", mission.success_criteria)
    if mission.deliverables:
        context_payload.setdefault("deliverables", mission.deliverables)
    if mission.metadata:
        context_payload.setdefault("metadata", mission.metadata)
    if mission.research_phases:
        context_payload.setdefault(
            "research_phases",
            {name: phase.model_dump() for name, phase in mission.research_phases.items()},
        )

    deliverable_format = _resolve_deliverable_format(mission)
    max_loops = _resolve_max_loops(mission)
    research_depth = _resolve_research_depth(mission)

    return empty_state(
        mission_id=mission.mission_id,
        mission_objectives=objectives,
        project_id=_extract_project_id(mission.mission_id),
        mission_context=context_payload,
        deliverable_format=deliverable_format,
        research_depth=research_depth,
        max_loops=max_loops,
    )


def _extract_project_id(mission_id: str) -> Optional[str]:
    """Best-effort project identifier extraction from the mission ID."""

    if "." in mission_id:
        prefix = mission_id.split(".", 1)[0].strip()
        return prefix or None
    if "-" in mission_id:
        prefix = mission_id.split("-", 1)[0].strip()
        return prefix or None
    return None


def _resolve_deliverable_format(mission: MissionYAML) -> str:
    """Determine the correct deliverable format with a sensible fallback."""

    explicit = mission.deliverable_format
    if explicit:
        return explicit.strip().lower()

    metadata = mission.metadata or {}
    for key in (
        "deliverable_format",
        "deliverableFormat",
        "output_format",
        "outputFormat",
        "output",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return DEFAULT_DELIVERABLE_FORMAT


def _resolve_max_loops(mission: MissionYAML) -> Optional[int]:
    """Determine an appropriate max loop count from mission metadata.

    Returns None if no explicit max_loops is specified, allowing the depth
    tier configuration to provide the default.
    """

    if mission.max_loops:
        return mission.max_loops

    metadata = mission.metadata or {}
    for key in ("max_loops", "maxLoops", "researchLoops"):
        value = metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            parsed = int(value)
            if parsed > 0:
                return parsed

    return None


def _resolve_research_depth(mission: MissionYAML) -> str:
    """Determine research depth label from mission schema or metadata.

    Returns a normalised string ("baseline"/"deep"/"alpha") and falls back to
    "baseline" when no value is specified or the value is unknown. Boundary
    string carrier post-S76 — no in-process dispatch.
    """
    depth_str = mission.research_depth

    if not depth_str:
        metadata = mission.metadata or {}
        for key in ("research_depth", "researchDepth", "depth"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                depth_str = value.strip()
                break

    if not depth_str:
        return "baseline"

    normalized = depth_str.strip().lower()
    if normalized not in {"baseline", "deep", "alpha"}:
        logger.warning(
            "Unknown research depth '%s' - defaulting to baseline. "
            "Valid values: baseline, deep, alpha",
            depth_str,
        )
        return "baseline"
    return normalized


def _format_validation_errors(error: ValidationError) -> str:
    """Return a compact error string for pydantic validation failures."""

    formatted = []
    for entry in error.errors():
        location = ".".join(str(item) for item in entry.get("loc", ()))
        formatted.append(f"{location or 'root'}: {entry.get('msg')}")
    return "Invalid mission data: " + "; ".join(formatted)


__all__ = [
    "MissionLoadError",
    "load_mission_from_yaml",
    "validate_mission_yaml",
    "_build_agent_state",
    "_resolve_research_depth",
]
