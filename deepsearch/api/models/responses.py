"""Response models for DeepSearch API."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ExecuteMissionResponse(BaseModel):
    """Response after mission submission."""

    job_id: str
    mission_id: str
    status: str = "queued"
    queued_at: datetime


class ErrorDetail(BaseModel):
    """Field-level error detail."""

    field: str
    message: str
    type: str


class ErrorResponse(BaseModel):
    """Error response with details."""

    error: str
    details: Optional[list[ErrorDetail]] = None


class JobProgressInfo(BaseModel):
    """Progress information for running jobs."""

    current_loop: int
    max_loops: int
    findings_count: int
    percentage: int


class JobStatusResponse(BaseModel):
    """Job status response."""

    job_id: str
    mission_id: str
    status: str  # queued, running, complete, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[JobProgressInfo] = None
    error: Optional[str] = None
    result_url: Optional[str] = None


class JobResultResponse(BaseModel):
    """Job result response with report and protocol."""

    job_id: str
    mission_id: str
    status: str
    completed_at: datetime
    report_markdown: str
    protocol_json: dict[str, Any]
    execution_metadata: dict[str, Any]


class MissionContractPreviewResponse(BaseModel):
    """Response for contract-preview requests."""

    contract_schema_version: str
    compiler_revision: int
    compiler_manifest_sha256: str
    compiler_manifest: dict[str, Any]
    contract: dict[str, Any]
    named_entities: list[str]
    objectives: list[dict[str, Any]]
    evidence_slots: list[dict[str, Any]]
    acceptance_checks: list[dict[str, Any]]
    deliverable_schemas: list[dict[str, Any]]
    coverage_thresholds: dict[str, float]
    validation_thresholds: dict[str, float]
