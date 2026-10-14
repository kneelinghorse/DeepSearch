"""Health check endpoints."""

from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    """Readiness check response."""
    status: str
    timestamp: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Simple liveness check."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check() -> ReadyResponse:
    """Readiness check with dependency status."""
    checks = {}

    # Check database connectivity
    try:
        # Will add actual check when queue is implemented
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"

    overall = "ready" if all(v == "ok" for v in checks.values()) else "not_ready"

    return ReadyResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks=checks,
    )
