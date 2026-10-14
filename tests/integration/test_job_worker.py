"""Integration tests for JobWorker execution."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
pytest.importorskip("fastapi")

from deepsearch.api.models.requests import ExecuteMissionRequest
from deepsearch.api.queue import JobQueue, JobStatus
from deepsearch.api.worker import JobWorker


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
def worker(queue):
    """Create a JobWorker with a short poll interval."""
    return JobWorker(queue, poll_interval=0.1)


@pytest.fixture
def sample_request():
    """Create a sample ExecuteMissionRequest."""
    return ExecuteMissionRequest(
        mission_id="INT.TEST.1",
        title="Integration Test Mission",
        objectives=["Test objective"],
        max_loops=1,
    )


class TestWorkerJobProcessing:
    """Tests for job processing by the worker."""

    @pytest.mark.asyncio
    async def test_worker_processes_queued_job(self, queue, worker, sample_request):
        """Worker processes a queued job."""
        job = queue.create(sample_request)

        # Mock the runner to return quickly
        mock_result = {
            "status": "complete",
            "report_path": "/tmp/test_report.md",
            "loops_executed": 1,
        }

        with patch(
            "deepsearch.runner.run_research_from_request",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = mock_result

            # Start worker and let it process
            worker_task = asyncio.create_task(worker.start())

            # Wait for job to be processed
            await asyncio.sleep(0.5)

            # Stop worker
            worker._shutdown = True
            await worker_task

        # Verify job was processed
        updated = queue.get(job.id)
        assert updated.status == JobStatus.COMPLETE
        assert updated.result_path == "/tmp/test_report.md"
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_captures_failures(self, queue, worker, sample_request):
        """Worker captures job failures with error details."""
        job = queue.create(sample_request)

        with patch(
            "deepsearch.runner.run_research_from_request",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.side_effect = RuntimeError("Test error message")

            # Start worker and let it process
            worker_task = asyncio.create_task(worker.start())
            await asyncio.sleep(0.5)
            worker._shutdown = True
            await worker_task

        # Verify job failed with error
        updated = queue.get(job.id)
        assert updated.status == JobStatus.FAILED
        assert "RuntimeError" in updated.error
        assert "Test error message" in updated.error

    @pytest.mark.asyncio
    async def test_worker_updates_progress(self, queue, worker, sample_request):
        """Worker updates job progress during execution."""
        job = queue.create(sample_request)

        async def mock_run_with_progress(request, progress_callback=None, **kwargs):
            if progress_callback:
                progress_callback(1, 3)  # Loop 1, 3 findings
                progress_callback(2, 7)  # Loop 2, 7 findings
            return {"status": "complete", "report_path": "/tmp/report.md"}

        with patch(
            "deepsearch.runner.run_research_from_request",
            side_effect=mock_run_with_progress,
        ):
            worker_task = asyncio.create_task(worker.start())
            await asyncio.sleep(0.5)
            worker._shutdown = True
            await worker_task

        # Check final progress
        updated = queue.get(job.id)
        assert updated.current_loop == 2
        assert updated.findings_count == 7

    @pytest.mark.asyncio
    async def test_worker_processes_jobs_sequentially(
        self, queue, worker, sample_request
    ):
        """Worker processes only one job at a time."""
        job1 = queue.create(sample_request)
        job2 = queue.create(sample_request)

        processing_order = []

        async def mock_run_tracking(request, **kwargs):
            processing_order.append(request.mission_id)
            await asyncio.sleep(0.1)  # Simulate work
            return {"status": "complete", "report_path": "/tmp/report.md"}

        with patch(
            "deepsearch.runner.run_research_from_request",
            side_effect=mock_run_tracking,
        ):
            worker_task = asyncio.create_task(worker.start())
            await asyncio.sleep(0.8)  # Give time for both jobs
            worker._shutdown = True
            await worker_task

        # Both jobs should have been processed
        assert len(processing_order) == 2

        # Verify both completed
        assert queue.get(job1.id).status == JobStatus.COMPLETE
        assert queue.get(job2.id).status == JobStatus.COMPLETE


class TestWorkerShutdown:
    """Tests for worker shutdown behavior."""

    @pytest.mark.asyncio
    async def test_worker_graceful_shutdown(self, queue, worker):
        """Worker shuts down gracefully on signal."""
        worker_task = asyncio.create_task(worker.start())

        await asyncio.sleep(0.2)
        worker._shutdown = True

        # Should complete within timeout
        await asyncio.wait_for(worker_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_worker_handles_shutdown_during_processing(
        self, queue, worker, sample_request
    ):
        """Worker completes current job before shutdown."""
        job = queue.create(sample_request)
        processing_complete = asyncio.Event()

        async def slow_mock_run(**kwargs):
            await asyncio.sleep(0.3)
            processing_complete.set()
            return {"status": "complete", "report_path": "/tmp/report.md"}

        with patch(
            "deepsearch.runner.run_research_from_request",
            side_effect=slow_mock_run,
        ):
            worker_task = asyncio.create_task(worker.start())

            # Wait for processing to start
            await asyncio.sleep(0.2)
            worker._shutdown = True

            # Wait for completion
            await asyncio.wait_for(processing_complete.wait(), timeout=2.0)
            await asyncio.wait_for(worker_task, timeout=2.0)

        # Job should still be marked complete
        updated = queue.get(job.id)
        assert updated.status == JobStatus.COMPLETE


class TestWorkerWithEmptyQueue:
    """Tests for worker behavior with empty queue."""

    @pytest.mark.asyncio
    async def test_worker_polls_empty_queue(self, queue, worker):
        """Worker continues polling when queue is empty."""
        poll_count = 0
        original_get_next = queue.get_next_queued

        def counting_get_next():
            nonlocal poll_count
            poll_count += 1
            return original_get_next()

        queue.get_next_queued = counting_get_next

        worker_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.35)  # ~3 poll intervals
        worker._shutdown = True
        await worker_task

        assert poll_count >= 3


class TestWorkerSingleJobConstraint:
    """Tests for single-job-at-a-time constraint."""

    @pytest.mark.asyncio
    async def test_worker_respects_running_job(self, queue, worker, sample_request):
        """Worker doesn't start new job while one is running."""
        # Create job and mark as running (simulating external start)
        job1 = queue.create(sample_request)
        queue.update_status(job1.id, JobStatus.RUNNING)

        # Create another queued job
        job2 = queue.create(sample_request)

        with patch(
            "deepsearch.runner.run_research_from_request",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {
                "status": "complete",
                "report_path": "/tmp/report.md",
            }

            worker_task = asyncio.create_task(worker.start())
            await asyncio.sleep(0.3)
            worker._shutdown = True
            await worker_task

        # Mock should not have been called (job1 was already running)
        mock_run.assert_not_called()

        # job2 should still be queued
        assert queue.get(job2.id).status == JobStatus.QUEUED
