"""SQLite-backed job queue for async mission execution."""

import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from deepsearch.api.models.requests import ExecuteMissionRequest


class JobStatus(str, Enum):
    """Job status values."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Job(BaseModel):
    """Job record model."""

    id: str
    mission_id: str
    status: JobStatus
    request: ExecuteMissionRequest
    callback_url: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result_path: Optional[str] = None

    # Progress tracking for status endpoint
    current_loop: int = 0
    max_loops: int = 3
    findings_count: int = 0


class JobQueue:
    """SQLite-backed job queue."""

    def __init__(self, db_path: str = "deepsearch_jobs.db"):
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables if not exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                mission_id TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                callback_url TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error TEXT,
                result_path TEXT,
                current_loop INTEGER DEFAULT 0,
                max_loops INTEGER DEFAULT 3,
                findings_count INTEGER DEFAULT 0
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs (status)
        """
        )
        conn.commit()
        conn.close()

    def create(self, request: ExecuteMissionRequest) -> Job:
        """Create and queue a new job."""
        effective_request = request
        effective_max_loops = request.max_loops

        job = Job(
            id=str(uuid.uuid4()),
            mission_id=request.mission_id,
            status=JobStatus.QUEUED,
            request=effective_request,
            callback_url=str(request.callback_url) if request.callback_url else None,
            created_at=datetime.now(timezone.utc),
            max_loops=effective_max_loops,
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO jobs
               (id, mission_id, status, request_json, callback_url, created_at, max_loops)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                job.id,
                job.mission_id,
                job.status.value,
                job.request.model_dump_json(),
                job.callback_url,
                job.created_at.isoformat(),
                job.max_loops,
            ),
        )
        conn.commit()
        conn.close()

        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_job(row)

    def get_next_queued(self) -> Optional[Job]:
        """Get next queued job (FIFO)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT * FROM jobs
               WHERE status = ?
               ORDER BY created_at ASC
               LIMIT 1""",
            (JobStatus.QUEUED.value,),
        ).fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_job(row)

    def has_running_job(self) -> bool:
        """Check if there's a job currently running."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = ?",
            (JobStatus.RUNNING.value,),
        ).fetchone()
        conn.close()
        return row[0] > 0

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        result_path: Optional[str] = None,
    ) -> None:
        """Update job status."""
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)

        if status == JobStatus.RUNNING:
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
                (status.value, now, job_id),
            )
        elif status in (JobStatus.COMPLETE, JobStatus.FAILED):
            conn.execute(
                """UPDATE jobs
                   SET status = ?, completed_at = ?, error = ?, result_path = ?
                   WHERE id = ?""",
                (status.value, now, error, result_path, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET status = ? WHERE id = ?",
                (status.value, job_id),
            )

        conn.commit()
        conn.close()

    def update_progress(
        self,
        job_id: str,
        current_loop: int,
        findings_count: int,
    ) -> None:
        """Update job progress for status polling."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """UPDATE jobs
               SET current_loop = ?, findings_count = ?
               WHERE id = ?""",
            (current_loop, findings_count, job_id),
        )
        conn.commit()
        conn.close()

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs with optional status filter."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        if status:
            rows = conn.execute(
                """SELECT * FROM jobs
                   WHERE status = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (status.value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM jobs
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        conn.close()
        return [self._row_to_job(row) for row in rows]

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert DB row to Job model."""
        return Job(
            id=row["id"],
            mission_id=row["mission_id"],
            status=JobStatus(row["status"]),
            request=ExecuteMissionRequest.model_validate_json(row["request_json"]),
            callback_url=row["callback_url"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(
                datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            error=row["error"],
            result_path=row["result_path"],
            current_loop=row["current_loop"] or 0,
            max_loops=row["max_loops"] or 3,
            findings_count=row["findings_count"] or 0,
        )
