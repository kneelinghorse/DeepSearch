"""Pydantic schemas for TraceLab PEDR graph search API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class PEDRQueryIntent(str, Enum):
    """Intent classification for PEDR search queries.

    B13.3: Added 'unknown' value per TraceLab S29 API update.
    """

    EXPLORATORY = "exploratory"
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    DEEP_DIVE = "deep_dive"
    UNKNOWN = "unknown"


class MissionActionableError(BaseModel):
    """Structured error response from TraceLab S29 with actionable suggestions.

    B13.3: TraceLab S29 returns this structure for mission-level errors,
    providing a human-readable suggestion and the mission's current status
    to help clients recover gracefully.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    message: str = Field(..., description="Error message")
    suggestion: Optional[str] = Field(
        default=None,
        description="Actionable suggestion for resolving the error",
    )
    current_status: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("current_status", "currentStatus"),
        description="Current mission status in TraceLab",
    )
    error_code: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("error_code", "errorCode", "code"),
        description="Machine-readable error code",
    )


class GraphSearchRequest(BaseModel):
    """Request payload for PEDR graph search (L6 graph layer)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    query: str = Field(..., description="Search query text")
    project_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("project_id", "projectId"),
    )
    top_k: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("top_k", "topK"),
    )
    source_origin: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("source_origin", "sourceOrigin"),
        description="B13.3: Filter by source origin (e.g., 'synthesized', 'uploaded')",
    )
    include_related: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_related", "includeRelated"),
    )
    include_embeddings: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_embeddings", "includeEmbeddings"),
    )
    enable_graph: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable_graph", "enableGraph"),
    )
    graph_depth: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("graph_depth", "graphDepth"),
    )
    graph_decay: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("graph_decay", "graphDecay"),
    )
    graph_edge_types: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("graph_edge_types", "graphEdgeTypes"),
    )
    graph_weight: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("graph_weight", "graphWeight"),
    )

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("query cannot be empty")
        return text

    @field_validator("graph_edge_types", mode="before")
    @classmethod
    def _normalize_graph_edge_types(cls, value: Any) -> List[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item or "").strip()]
        raise TypeError("graph_edge_types must be a list of strings")


class GraphRelatedEntity(BaseModel):
    """Related entity surfaced by graph expansion."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    entity_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("entity_id", "entityId", "id"),
    )
    label: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("label", "name", "entity_label", "entityLabel"),
    )
    entity_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("entity_type", "entityType", "type", "kind"),
    )
    score: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("score", "similarity", "weight"),
    )
    mission_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("mission_id", "missionId"),
    )
    mission_uuid: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("mission_uuid", "missionUuid"),
    )


class ScoreFusion(BaseModel):
    """Score fusion metadata from TraceLab S29 PEDR responses.

    TraceLab S29 decomposes the single similarity score into four components,
    allowing consumers to make smarter relevance decisions.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    base_score: float = Field(
        default=0.0,
        validation_alias=AliasChoices("base_score", "baseScore"),
        description="Raw embedding similarity before any boosts",
    )
    type_boost: float = Field(
        default=0.0,
        validation_alias=AliasChoices("type_boost", "typeBoost"),
        description="Boost applied based on mission type match",
    )
    intent_boost: float = Field(
        default=0.0,
        validation_alias=AliasChoices("intent_boost", "intentBoost"),
        description="Boost applied based on query intent alignment",
    )
    quality_multiplier: float = Field(
        default=1.0,
        validation_alias=AliasChoices("quality_multiplier", "qualityMultiplier"),
        description="Multiplier based on source quality gates",
    )

    @property
    def composite_score(self) -> float:
        """Compute the composite score from fusion components."""
        return (self.base_score + self.type_boost + self.intent_boost) * self.quality_multiplier


class GraphSearchResult(BaseModel):
    """Graph-augmented search result returned by PEDR."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    mission_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("mission_id", "missionId"),
    )
    mission_uuid: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("mission_uuid", "missionUuid"),
    )
    title: Optional[str] = None
    objective: Optional[str] = None
    summary: Optional[str] = None
    rrf_score: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("rrf_score", "rrfScore"),
    )
    layer_ranks: Dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("layer_ranks", "layerRanks"),
    )
    related_entities: List[GraphRelatedEntity] = Field(
        default_factory=list,
        validation_alias=AliasChoices("related_entities", "relatedEntities"),
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("embedding", "embedding_vector", "embeddingVector"),
    )
    embedding_score: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "embedding_score", "embeddingScore", "embedding_similarity", "embeddingSimilarity"
        ),
    )
    embedding_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("embedding_model", "embeddingModel"),
    )
    score_fusion: Optional[ScoreFusion] = Field(
        default=None,
        validation_alias=AliasChoices("score_fusion", "scoreFusion"),
        description="S29 score fusion metadata decomposing similarity into components",
    )

    @field_validator("layer_ranks", mode="before")
    @classmethod
    def _normalize_layer_ranks(cls, value: Any) -> Dict[str, float]:
        if value in (None, "", []):
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            ranks: Dict[str, float] = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                layer = item.get("layer") or item.get("name")
                rank = item.get("rank") or item.get("score")
                if layer is None or rank is None:
                    continue
                try:
                    ranks[str(layer)] = float(rank)
                except (TypeError, ValueError):
                    continue
            return ranks
        raise TypeError("layer_ranks must be a mapping")

    @field_validator("related_entities", mode="before")
    @classmethod
    def _normalize_related_entities(cls, value: Any) -> List[Any]:
        if value in (None, "", []):
            return []
        return value


class GraphSearchMetadata(BaseModel):
    """Metadata describing the graph search execution."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    graph_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("graph_enabled", "graphEnabled"),
    )
    graph_candidates_expanded: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("graph_candidates_expanded", "graphCandidatesExpanded"),
    )
    timings: Dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("timings", "timings_ms", "timingsMs"),
    )

    @field_validator("timings", mode="before")
    @classmethod
    def _normalize_timings(cls, value: Any) -> Dict[str, float]:
        if value in (None, "", []):
            return {}
        if isinstance(value, dict):
            return value
        raise TypeError("timings must be a mapping")


class GraphSearchResponse(BaseModel):
    """Top-level response for PEDR graph search."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    query: Optional[str] = None
    results: List[GraphSearchResult] = Field(
        default_factory=list,
        validation_alias=AliasChoices("results", "matches", "items"),
    )
    metadata: Optional[GraphSearchMetadata] = None
    total: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("total", "result_count", "resultCount", "count"),
    )


__all__ = [
    "GraphRelatedEntity",
    "GraphSearchMetadata",
    "GraphSearchRequest",
    "GraphSearchResponse",
    "GraphSearchResult",
    "MissionActionableError",
    "PEDRQueryIntent",
    "ScoreFusion",
]
