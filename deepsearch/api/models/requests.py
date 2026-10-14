"""Request models for DeepSearch API."""

from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl, field_validator

from deepsearch.mission.scope import AuthoredScope


def _normalize_research_depth(value: object) -> Optional[str]:
    """S39.2 — shared research_depth normalization used by Execute + Preview.

    Called from a ``field_validator(mode="before")`` on both
    :class:`ExecuteMissionRequest` and :class:`MissionContractPreviewRequest`.
    Keeping one implementation guarantees the two endpoints cannot drift on
    accepted depth strings (``baseline``, ``deep``, ``alpha``).
    """

    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in {"baseline", "deep", "alpha"}:
        raise ValueError("research_depth must be one of: baseline, deep, alpha")
    return text


class ExtendedMissionFields(BaseModel):
    """Shared extended mission-context fields used by Execute and Preview.

    The mission-contract compiler (``compile_contract_from_state``) reads all
    of these keys directly from ``mission_context``. Centralising them on a
    single mixin means the /missions/execute and /missions/preview request
    surfaces cannot silently diverge — adding a new field here surfaces on
    both endpoints at once, and AliasChoices camelCase variants stay in sync.

    Everything is optional so both endpoints remain backward compatible with
    payloads that omit these fields.
    """

    required_entities: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("required_entities", "requiredEntities"),
    )
    excluded_entities: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("excluded_entities", "excludedEntities"),
    )
    expected_output_schema: Any = Field(
        default=None,
        validation_alias=AliasChoices("expected_output_schema", "expectedOutputSchema"),
    )
    coverage_thresholds: Optional[dict[str, float]] = Field(
        default=None,
        validation_alias=AliasChoices("coverage_thresholds", "coverageThresholds"),
    )
    validation_thresholds: Optional[dict[str, float]] = Field(
        default=None,
        validation_alias=AliasChoices("validation_thresholds", "validationThresholds"),
    )
    deliverable_format: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("deliverable_format", "deliverableFormat"),
    )
    min_loops: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("min_loops", "minLoops"),
    )
    references: Optional[list[dict[str, Any]]] = None
    authored_scope: Optional[AuthoredScope] = Field(
        default=None, validation_alias=AliasChoices("authored_scope", "authoredScope"),
    )


class MissionContext(BaseModel):
    """Optional mission context."""
    background: Optional[str] = None
    constraints: Optional[list[str]] = None
    focus: Optional[str] = None


class ExecuteMissionRequest(ExtendedMissionFields):
    """Request to execute a research mission."""

    mission_id: str = Field(
        ...,
        min_length=1,
        description="Unique mission identifier",
        validation_alias=AliasChoices("mission_id", "missionId"),
    )
    title: str = Field(..., min_length=1, description="Mission title")
    objectives: list[str] = Field(..., min_length=1, description="Research objectives")
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Success criteria for the mission",
        validation_alias=AliasChoices("success_criteria", "successCriteria"),
    )
    context: Optional[MissionContext] = None
    callback_url: Optional[HttpUrl] = Field(
        None,
        description="Webhook URL for result delivery",
        validation_alias=AliasChoices("callback_url", "callbackUrl"),
    )
    research_depth: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("research_depth", "researchDepth", "depth"),
        description="Research depth tier (baseline/deep/alpha)",
    )
    max_loops: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum research loops",
        validation_alias=AliasChoices("max_loops", "maxLoops"),
    )
    token_limit: int = Field(
        default=8000,
        ge=1000,
        le=32000,
        description="Token limit per loop",
        validation_alias=AliasChoices("token_limit", "tokenLimit"),
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "mission_id": "DRM.0.5",
                "title": "Best practices for API authentication",
                "objectives": [
                    "Compare OAuth 2.0 vs JWT for API auth",
                    "Identify security best practices",
                ],
                "success_criteria": [
                    "Comparison table with pros/cons",
                    "3+ code examples",
                ],
                "callback_url": "https://tracelab.namozine.com/webhooks/deepsearch",
                "research_depth": "baseline",
                "max_loops": 3,
                "token_limit": 8000,
            }
        },
    )

    @field_validator("research_depth", mode="before")
    @classmethod
    def _validate_research_depth(cls, value: object) -> Optional[str]:
        return _normalize_research_depth(value)


class MissionContractPreviewRequest(ExtendedMissionFields):
    """Request to preview a compiled mission contract from authored context.

    This mirrors the mission_context surface consumed by compile_contract_from_state.
    Extra fields are preserved so TraceLab-authored payloads can round-trip through
    the preview endpoint without DeepSearch maintaining a second hand-written schema.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    mission_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("mission_id", "missionId"),
    )
    title: Optional[str] = None
    objective: Optional[str] = None
    success_criteria: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("success_criteria", "successCriteria"),
    )
    deliverables: list[str] = Field(default_factory=list)
    background: Optional[str] = None
    constraints: Optional[list[str]] = None
    focus: Optional[str] = None
    research_depth: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("research_depth", "researchDepth", "depth"),
    )
    max_loops: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_loops", "maxLoops"),
    )
    compiler_mode: Literal["default", "no_enrichment"] = Field(
        default="default",
        validation_alias=AliasChoices("compiler_mode", "compilerMode"),
        description=(
            "Compiler execution mode. no_enrichment produces canonical, "
            "provider-independent parity output."
        ),
    )

    @field_validator("research_depth", mode="before")
    @classmethod
    def _validate_research_depth(cls, value: object) -> Optional[str]:
        return _normalize_research_depth(value)


EXTENDED_CONTEXT_FIELDS: tuple[str, ...] = (
    "required_entities",
    "excluded_entities",
    "expected_output_schema",
    "coverage_thresholds",
    "validation_thresholds",
    "deliverable_format",
    "references",
    "authored_scope",
)
"""Names of the extended fields that belong inside ``mission_context`` when
the compiler runs. ``min_loops`` is intentionally NOT in this list because
the state builder threads it through ``state['min_loops']`` instead."""


def build_mission_context_from_execute_request(
    request: ExecuteMissionRequest,
) -> dict[str, Any]:
    """Return the ``mission_context`` dict that ``compile_contract_from_state`` expects.

    Mirrors the preview-endpoint path: the extended fields travel inside
    ``mission_context`` so the mission-contract compiler reads them by the
    same keys (``required_entities``, ``excluded_entities``, ``references``,
    ``coverage_thresholds``, ``validation_thresholds``, ``expected_output_schema``,
    ``deliverable_format``). ``min_loops`` is surfaced on the request directly
    and threaded into state by the caller, not into the context.
    """

    payload: dict[str, Any] = {
        "title": request.title,
        "objective": "; ".join(request.objectives),
    }
    if request.success_criteria:
        payload["success_criteria"] = list(request.success_criteria)
    if request.context:
        if request.context.background:
            payload["background"] = request.context.background
        if request.context.constraints:
            payload["constraints"] = list(request.context.constraints)
        if request.context.focus:
            payload["focus"] = request.context.focus

    for field_name in EXTENDED_CONTEXT_FIELDS:
        value = getattr(request, field_name, None)
        if value is None:
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        payload[field_name] = (
            value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        )

    return payload
