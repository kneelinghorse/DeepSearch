"""Mission execution and preview endpoints."""

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from deepsearch.api.auth.hmac import require_tracelab_signature
from deepsearch.api.models.requests import ExecuteMissionRequest, MissionContractPreviewRequest
from deepsearch.api.models.responses import (
    ExecuteMissionResponse,
    MissionContractPreviewResponse,
)
from deepsearch.api.queue import JobQueue
from deepsearch.api.validators import validate_mission
from deepsearch.mission import (
    CONFIGURED_ENRICHMENT_MODE,
    NO_ENRICHMENT_MODE,
    build_contract_compiler_manifest,
    compile_canonical_contract_from_state,
    compile_contract_from_state,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/missions")
_PREVIEW_ERROR_PATTERN = re.compile(
    r"^(?P<field>[A-Za-z0-9_\.\[\]]+)\s+(?P<message>is required|must be .+)$"
)


def get_queue(request: Request) -> JobQueue:
    """Get the job queue from app state."""
    return request.app.state.queue


@router.post("/execute", response_model=ExecuteMissionResponse)
async def execute_mission(
    request: Request, mission_request: ExecuteMissionRequest
) -> ExecuteMissionResponse:
    """
    Submit a research mission for execution.

    Returns a job_id that can be used to poll status and retrieve results.
    If callback_url is provided, results will also be POSTed there.
    """
    # Semantic validation
    validation = validate_mission(mission_request)
    if not validation.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Mission validation failed",
                "validation_errors": [e.model_dump() for e in validation.errors],
            },
        )

    queue = get_queue(request)

    # Create job in queue
    job = queue.create(mission_request)
    logger.info(
        f"Created job {job.id} for mission {mission_request.mission_id}"
    )

    return ExecuteMissionResponse(
        job_id=job.id,
        mission_id=job.mission_id,
        status=job.status.value,
        queued_at=job.created_at,
    )


@router.post(
    "/preview",
    response_model=MissionContractPreviewResponse,
    dependencies=[Depends(require_tracelab_signature)],
)
async def preview_mission_contract(
    preview_request: MissionContractPreviewRequest,
) -> MissionContractPreviewResponse:
    """Preview the compiled mission contract for an authored mission_context payload.

    Method: POST
    Path: /api/v1/missions/preview
    Auth: requires `X-TraceLab-Signature: sha256=<hex>` using the shared
    `TRACELAB_DEEPSEARCH_SERVICE_SECRET` HMAC secret. Local dev skips
    validation when the secret is unset.
    """

    try:
        state = _build_preview_state(preview_request)
        if preview_request.compiler_mode == "no_enrichment":
            contract = compile_canonical_contract_from_state(state)
            enrichment_mode = NO_ENRICHMENT_MODE
        else:
            contract = compile_contract_from_state(state, origin="api_preview")
            enrichment_mode = CONFIGURED_ENRICHMENT_MODE
        manifest = build_contract_compiler_manifest(
            enrichment_mode=enrichment_mode,
        )
    except ValueError as exc:
        detail = _preview_error_detail(exc)
        raise HTTPException(status_code=422, detail=detail) from exc

    return MissionContractPreviewResponse(
        contract_schema_version=contract.contract_schema_version,
        compiler_revision=contract.compiler_revision,
        compiler_manifest_sha256=str(manifest["compiler_manifest_sha256"]),
        compiler_manifest=manifest,
        contract=contract.model_dump(mode="json"),
        named_entities=list(contract.named_entities),
        objectives=[item.model_dump(mode="json") for item in contract.objectives],
        evidence_slots=[item.model_dump(mode="json") for item in contract.evidence_slots],
        acceptance_checks=[item.model_dump(mode="json") for item in contract.acceptance_checks],
        deliverable_schemas=[
            item.model_dump(mode="json") for item in contract.deliverable_schemas
        ],
        coverage_thresholds=dict(contract.coverage_thresholds),
        validation_thresholds=dict(contract.validation_thresholds),
    )


def _build_preview_state(preview_request: MissionContractPreviewRequest) -> dict[str, Any]:
    mission_context = preview_request.model_dump(
        by_alias=False,
        exclude_none=True,
        exclude={"compiler_mode"},
    )
    mission_id = str(mission_context.get("mission_id") or "preview-mission").strip()
    objective = str(mission_context.get("objective") or "").strip()
    success_criteria = [
        str(item).strip()
        for item in mission_context.get("success_criteria") or []
        if str(item).strip()
    ]
    mission_objectives = success_criteria or ([objective] if objective else [])

    return {
        "mission_id": mission_id or "preview-mission",
        "mission_context": mission_context,
        "mission_objectives": mission_objectives,
        "deliverable_format": str(mission_context.get("deliverable_format") or "markdown"),
        "research_depth": str(mission_context.get("research_depth") or "baseline"),
        "max_loops": mission_context.get("max_loops") or 3,
        "min_loops": mission_context.get("min_loops") or 0,
        "depth_config": {},
    }


def _preview_error_detail(exc: ValueError) -> dict[str, str]:
    message = str(exc).strip() or "Mission contract preview failed"
    match = _PREVIEW_ERROR_PATTERN.match(message)
    if match:
        return {
            "error": "Mission contract preview failed",
            "field": match.group("field"),
            "message": match.group("message"),
        }
    return {
        "error": "Mission contract preview failed",
        "field": "mission_context",
        "message": message,
    }
