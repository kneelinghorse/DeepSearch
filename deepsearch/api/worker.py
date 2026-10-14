"""Async job worker for research execution."""

import asyncio
import logging
import signal
import traceback
from typing import Any, Callable, Dict, Optional

from deepsearch.api.queue import Job, JobQueue, JobStatus

logger = logging.getLogger(__name__)


class JobWorker:
    """Background worker that processes queued jobs."""

    def __init__(
        self,
        queue: JobQueue,
        poll_interval: float = 5.0,
    ):
        self.queue = queue
        self.poll_interval = poll_interval
        self._shutdown = False
        self._current_job: Optional[Job] = None

    async def start(self) -> None:
        """Start the worker loop."""
        logger.info("Job worker starting")

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                # Windows doesn't support signal handlers in async loops
                pass

        while not self._shutdown:
            try:
                # Don't process new jobs if one is already running
                if self.queue.has_running_job():
                    await asyncio.sleep(self.poll_interval)
                    continue

                job = self.queue.get_next_queued()

                if job:
                    await self._process_job(job)
                else:
                    await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(self.poll_interval)

        logger.info("Job worker stopped")

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._shutdown = True

    async def _process_job(self, job: Job) -> None:
        """Process a single job."""
        self._current_job = job
        logger.info(f"Processing job {job.id} for mission {job.mission_id}")

        # Mark as running
        self.queue.update_status(job.id, JobStatus.RUNNING)

        try:
            # Import here to avoid circular imports
            from deepsearch.runner import run_research_from_request

            # Create a progress callback that updates the queue
            def progress_callback(loop_num: int, findings_count: int) -> None:
                self.queue.update_progress(job.id, loop_num, findings_count)

            # Run research with the request
            result = await run_research_from_request(
                request=job.request,
                progress_callback=progress_callback,
            )

            # Mark complete with result path
            self.queue.update_status(
                job.id,
                JobStatus.COMPLETE,
                result_path=result.get("report_path"),
            )

            logger.info(f"Job {job.id} completed successfully")

            # Trigger webhook (B8.5 will implement)
            if job.callback_url:
                await self._send_webhook(job, result)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Job {job.id} failed: {error_msg}")

            self.queue.update_status(
                job.id,
                JobStatus.FAILED,
                error=error_msg,
            )

        finally:
            self._current_job = None

    async def _send_webhook(self, job: Job, result: Dict[str, Any]) -> None:
        """Send webhook callback to TraceLab.

        Builds payload from job and result, then delivers via WebhookClient
        with automatic retry on 5xx errors.
        """
        from deepsearch.api.webhook import WebhookClient, build_webhook_payload

        # Build execution metadata
        execution_metadata: Dict[str, Any] = {
            "duration_seconds": None,
            "loops_completed": job.current_loop,
            "findings_count": job.findings_count,
            "sources_count": len(result.get("sources", [])),
        }

        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()
            execution_metadata["duration_seconds"] = duration

        # Build payload from result files
        try:
            payload = build_webhook_payload(
                job_id=job.id,
                mission_id=job.mission_id,
                report_path=result["report_path"],
                protocol_path=result["protocol_path"],
                execution_metadata=execution_metadata,
            )
        except (KeyError, FileNotFoundError) as e:
            logger.error(f"Failed to build webhook payload for job {job.id}: {e}")
            return

        # Deliver webhook
        client = WebhookClient()
        webhook_result = await client.deliver(job.callback_url, payload)

        if webhook_result.success:
            logger.info(f"Webhook delivered for job {job.id}")
        else:
            logger.error(
                f"Webhook failed for job {job.id}: {webhook_result.error}"
            )
