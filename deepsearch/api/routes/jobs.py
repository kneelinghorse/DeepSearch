"""Job status and result endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from deepsearch.api.models.responses import (
    JobProgressInfo,
    JobResultResponse,
    JobStatusResponse,
)
from deepsearch.api.queue import JobStatus

router = APIRouter(prefix="/api/v1/jobs")


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    """
    Get current status of a job.

    Returns progress information for running jobs.
    Returns result_url for completed jobs.
    """
    queue = request.app.state.queue
    job = queue.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Build progress info for running jobs
    progress = None
    if job.status == JobStatus.RUNNING:
        percentage = (
            int((job.current_loop / job.max_loops) * 100) if job.max_loops > 0 else 0
        )
        progress = JobProgressInfo(
            current_loop=job.current_loop,
            max_loops=job.max_loops,
            findings_count=job.findings_count,
            percentage=percentage,
        )

    # Build result URL for completed jobs
    result_url = None
    if job.status == JobStatus.COMPLETE:
        result_url = f"/api/v1/jobs/{job_id}/result"

    return JobStatusResponse(
        job_id=job.id,
        mission_id=job.mission_id,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=progress,
        error=job.error if job.status == JobStatus.FAILED else None,
        result_url=result_url,
    )


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str, request: Request) -> JobResultResponse:
    """
    Get result of a completed job.

    Returns 404 if job not found.
    Returns 400 if job not yet complete.
    """
    queue = request.app.state.queue
    job = queue.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} not complete (status: {job.status.value})",
        )

    # Load report and protocol from result_path
    if not job.result_path:
        raise HTTPException(
            status_code=500,
            detail=f"Job {job_id} complete but no result path",
        )

    report_path = Path(job.result_path)
    protocol_path = report_path.with_name(report_path.stem + "_protocol.json")

    # Read report markdown
    if not report_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Report file not found: {report_path}",
        )
    report_markdown = report_path.read_text()

    # Read protocol JSON
    protocol_json = {}
    if protocol_path.exists():
        protocol_json = json.loads(protocol_path.read_text())

    # Build execution metadata
    execution_metadata = {
        "duration_seconds": None,
        "loops_completed": job.current_loop,
        "findings_count": job.findings_count,
        "sources_count": len(protocol_json.get("evidence", [])),
    }

    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()
        execution_metadata["duration_seconds"] = duration

    return JobResultResponse(
        job_id=job.id,
        mission_id=job.mission_id,
        status=job.status.value,
        completed_at=job.completed_at,
        report_markdown=report_markdown,
        protocol_json=protocol_json,
        execution_metadata=execution_metadata,
    )
