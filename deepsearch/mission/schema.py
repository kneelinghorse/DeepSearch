"""Pydantic schema definitions for Mission Protocol YAML documents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ResearchPhase(BaseModel):
    """Minimal representation of a research phase entry."""

    model_config = ConfigDict(extra="allow")

    title: str
    tasks: List[str] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("title cannot be empty")
        return text


class MissionYAML(BaseModel):
    """Validated view of a Mission Protocol YAML payload."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    mission_id: str = Field(
        ...,
        alias="missionId",
        validation_alias=AliasChoices("missionId", "mission_id"),
    )
    title: str
    objective: str
    success_criteria: List[str] = Field(
        ...,
        alias="successCriteria",
        validation_alias=AliasChoices("successCriteria", "success_criteria"),
    )
    deliverables: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("deliverables", "deliverableItems"),
    )
    context: Dict[str, Any] = Field(default_factory=dict)
    research_phases: Dict[str, ResearchPhase] = Field(
        default_factory=dict,
        alias="researchPhases",
        validation_alias=AliasChoices("researchPhases", "research_phases"),
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    deliverable_format: Optional[str] = Field(
        default=None,
        alias="deliverableFormat",
        validation_alias=AliasChoices("deliverableFormat", "deliverable_format"),
    )
    max_loops: Optional[int] = Field(
        default=None,
        alias="maxLoops",
        validation_alias=AliasChoices("maxLoops", "max_loops"),
    )
    research_depth: Optional[str] = Field(
        default=None,
        alias="researchDepth",
        validation_alias=AliasChoices("researchDepth", "research_depth", "depth"),
    )

    @field_validator("mission_id", "title", "objective", mode="before")
    @classmethod
    def _strip_required_fields(cls, value: Any, info: ValidationInfo) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{info.field_name} cannot be empty")
        return text

    @field_validator("context", mode="before")
    @classmethod
    def _ensure_dict(cls, value: Any) -> Dict[str, Any]:
        if value in (None, "", []):
            return {}
        if isinstance(value, dict):
            return value
        raise TypeError("context must be a mapping")

    @field_validator("metadata", mode="before")
    @classmethod
    def _ensure_metadata_dict(cls, value: Any) -> Dict[str, Any]:
        if value in (None, "", []):
            return {}
        if isinstance(value, dict):
            return value
        raise TypeError("metadata must be a mapping")

    @field_validator("success_criteria")
    @classmethod
    def _normalise_criteria(cls, value: List[Any]) -> List[str]:
        normalised = [str(item).strip() for item in value if str(item or "").strip()]
        if not normalised:
            raise ValueError("successCriteria must contain at least one entry")
        return normalised

    @field_validator("deliverables")
    @classmethod
    def _normalise_deliverables(cls, value: List[Any]) -> List[str]:
        return [str(item).strip() for item in value if str(item or "").strip()]

    @field_validator("max_loops")
    @classmethod
    def _validate_max_loops(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("maxLoops must be greater than zero")
        return value


__all__ = ["MissionYAML", "ResearchPhase"]
