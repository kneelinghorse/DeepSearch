"""Unit tests for JobQueue CRUD operations."""

import os
import tempfile
from datetime import datetime, timezone

import pytest
pytest.importorskip("fastapi")

from deepsearch.api.models.requests import ExecuteMissionRequest, MissionContext
from deepsearch.api.queue import Job, JobQueue, JobStatus


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def queue(temp_db_path):
    """Create a JobQueue with a temporary database."""
    return JobQueue(db_path=temp_db_path)


@pytest.fixture
def sample_request():
    """Create a sample ExecuteMissionRequest."""
    return ExecuteMissionRequest(
        mission_id="TEST.1",
        title="Test Mission",
        objectives=["Objective 1", "Objective 2"],
        success_criteria=["Criterion 1"],
        context=MissionContext(
            background="Test background",
            constraints=["Constraint 1"],
            focus="Test focus",
        ),
        max_loops=3,
        token_limit=8000,
    )


class TestJobQueueCreate:
    """Tests for job creation."""

    def test_create_job(self, queue, sample_request):
        """Create a job and verify its properties."""
        job = queue.create(sample_request)

        assert job.id is not None
        assert len(job.id) == 36  # UUID format
        assert job.mission_id == "TEST.1"
        assert job.status == JobStatus.QUEUED
        assert job.request.title == "Test Mission"
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.error is None
        assert job.max_loops == 3

    def test_create_job_persists_to_db(self, queue, sample_request):
        """Verify job persists to SQLite."""
        job = queue.create(sample_request)

        # Create new queue instance with same db
        queue2 = JobQueue(db_path=str(queue.db_path))
        retrieved = queue2.get(job.id)

        assert retrieved is not None
        assert retrieved.id == job.id
        assert retrieved.mission_id == job.mission_id
        assert retrieved.status == JobStatus.QUEUED

    def test_create_multiple_jobs(self, queue, sample_request):
        """Create multiple jobs and verify they're independent."""
        job1 = queue.create(sample_request)
        job2 = queue.create(sample_request)

        assert job1.id != job2.id
        assert queue.get(job1.id) is not None
        assert queue.get(job2.id) is not None

    def test_create_job_with_callback_url(self, queue):
        """Verify callback URL is stored correctly."""
        request = ExecuteMissionRequest(
            mission_id="TEST.2",
            title="Test with callback",
            objectives=["Test"],
            callback_url="https://example.com/webhook",
        )
        job = queue.create(request)

        assert job.callback_url == "https://example.com/webhook"
        retrieved = queue.get(job.id)
        assert retrieved.callback_url == "https://example.com/webhook"

class TestJobQueueGet:
    """Tests for job retrieval."""

    def test_get_existing_job(self, queue, sample_request):
        """Retrieve an existing job by ID."""
        created = queue.create(sample_request)
        retrieved = queue.get(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.mission_id == created.mission_id

    def test_get_nonexistent_job(self, queue):
        """Return None for non-existent job ID."""
        result = queue.get("nonexistent-uuid")
        assert result is None

    def test_get_next_queued(self, queue, sample_request):
        """Get next queued job in FIFO order."""
        job1 = queue.create(sample_request)
        job2 = queue.create(sample_request)

        next_job = queue.get_next_queued()

        assert next_job is not None
        assert next_job.id == job1.id  # First created should be first

    def test_get_next_queued_empty(self, queue):
        """Return None when no queued jobs."""
        result = queue.get_next_queued()
        assert result is None

    def test_get_next_queued_skips_running(self, queue, sample_request):
        """Skip jobs that are running."""
        job1 = queue.create(sample_request)
        job2 = queue.create(sample_request)

        # Mark first as running
        queue.update_status(job1.id, JobStatus.RUNNING)

        next_job = queue.get_next_queued()
        assert next_job is not None
        assert next_job.id == job2.id


class TestJobQueueUpdateStatus:
    """Tests for status updates."""

    def test_update_to_running(self, queue, sample_request):
        """Update job status to running."""
        job = queue.create(sample_request)
        queue.update_status(job.id, JobStatus.RUNNING)

        updated = queue.get(job.id)
        assert updated.status == JobStatus.RUNNING
        assert updated.started_at is not None

    def test_update_to_complete(self, queue, sample_request):
        """Update job status to complete."""
        job = queue.create(sample_request)
        queue.update_status(job.id, JobStatus.RUNNING)
        queue.update_status(
            job.id,
            JobStatus.COMPLETE,
            result_path="/path/to/report.md",
        )

        updated = queue.get(job.id)
        assert updated.status == JobStatus.COMPLETE
        assert updated.completed_at is not None
        assert updated.result_path == "/path/to/report.md"

    def test_update_to_failed(self, queue, sample_request):
        """Update job status to failed with error."""
        job = queue.create(sample_request)
        queue.update_status(job.id, JobStatus.RUNNING)
        queue.update_status(
            job.id,
            JobStatus.FAILED,
            error="RuntimeError: Something went wrong",
        )

        updated = queue.get(job.id)
        assert updated.status == JobStatus.FAILED
        assert updated.completed_at is not None
        assert "RuntimeError" in updated.error


class TestJobQueueUpdateProgress:
    """Tests for progress updates."""

    def test_update_progress(self, queue, sample_request):
        """Update job progress."""
        job = queue.create(sample_request)
        queue.update_status(job.id, JobStatus.RUNNING)
        queue.update_progress(job.id, current_loop=2, findings_count=5)

        updated = queue.get(job.id)
        assert updated.current_loop == 2
        assert updated.findings_count == 5


class TestJobQueueHasRunning:
    """Tests for checking running jobs."""

    def test_has_running_job_false(self, queue):
        """No running jobs initially."""
        assert queue.has_running_job() is False

    def test_has_running_job_true(self, queue, sample_request):
        """Detect when a job is running."""
        job = queue.create(sample_request)
        queue.update_status(job.id, JobStatus.RUNNING)

        assert queue.has_running_job() is True

    def test_has_running_job_after_complete(self, queue, sample_request):
        """No running job after completion."""
        job = queue.create(sample_request)
        queue.update_status(job.id, JobStatus.RUNNING)
        queue.update_status(job.id, JobStatus.COMPLETE)

        assert queue.has_running_job() is False


class TestJobQueueListJobs:
    """Tests for listing jobs."""

    def test_list_all_jobs(self, queue, sample_request):
        """List all jobs."""
        queue.create(sample_request)
        queue.create(sample_request)
        queue.create(sample_request)

        jobs = queue.list_jobs()
        assert len(jobs) == 3

    def test_list_jobs_by_status(self, queue, sample_request):
        """List jobs filtered by status."""
        job1 = queue.create(sample_request)
        job2 = queue.create(sample_request)
        queue.update_status(job1.id, JobStatus.RUNNING)

        queued = queue.list_jobs(status=JobStatus.QUEUED)
        running = queue.list_jobs(status=JobStatus.RUNNING)

        assert len(queued) == 1
        assert queued[0].id == job2.id
        assert len(running) == 1
        assert running[0].id == job1.id

    def test_list_jobs_limit(self, queue, sample_request):
        """Limit number of returned jobs."""
        for _ in range(5):
            queue.create(sample_request)

        jobs = queue.list_jobs(limit=2)
        assert len(jobs) == 2


class TestJobQueuePersistence:
    """Tests for database persistence."""

    def test_jobs_survive_restart(self, temp_db_path, sample_request):
        """Jobs persist across queue instances."""
        # Create queue and add job
        queue1 = JobQueue(db_path=temp_db_path)
        job = queue1.create(sample_request)
        queue1.update_status(job.id, JobStatus.RUNNING)
        job_id = job.id

        # Simulate restart with new queue instance
        queue2 = JobQueue(db_path=temp_db_path)
        retrieved = queue2.get(job_id)

        assert retrieved is not None
        assert retrieved.id == job_id
        assert retrieved.status == JobStatus.RUNNING
        assert retrieved.request.title == "Test Mission"

    def test_schema_is_idempotent(self, temp_db_path):
        """Schema creation is idempotent."""
        queue1 = JobQueue(db_path=temp_db_path)
        queue2 = JobQueue(db_path=temp_db_path)  # Should not fail
        queue3 = JobQueue(db_path=temp_db_path)  # Should not fail

        # All should work
        assert queue1 is not None
        assert queue2 is not None
        assert queue3 is not None
