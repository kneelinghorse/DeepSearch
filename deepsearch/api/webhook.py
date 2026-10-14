"""Webhook callback for delivering job results to TraceLab."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from deepsearch.data_paths import data_path

logger = logging.getLogger(__name__)


class WebhookPayload(BaseModel):
    """Webhook payload for TraceLab.

    Matches TraceLab's expected contract for DeepSearch results.
    """

    job_id: str
    mission_id: str
    status: str
    completed_at: datetime
    execution_metadata: dict[str, Any]
    report_markdown: str
    protocol: dict[str, Any]


class WebhookResult(BaseModel):
    """Result of webhook delivery attempt."""

    success: bool
    attempts: int
    final_status_code: Optional[int] = None
    error: Optional[str] = None


class WebhookClient:
    """Client for delivering webhooks with retry logic.

    Implements retry on 5xx errors with exponential backoff.
    Gives up immediately on 4xx errors (client error, no point retrying).
    Failed payloads are saved for manual recovery.
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout: float = 30.0,
        backoff_base: float = 2.0,
        failed_payloads_dir: str | Path | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.failed_payloads_dir = (
            Path(failed_payloads_dir) if failed_payloads_dir is not None
            else data_path("webhooks", "failed")
        )
        self.failed_payloads_dir.mkdir(parents=True, exist_ok=True)

    async def deliver(
        self,
        callback_url: str,
        payload: WebhookPayload,
    ) -> WebhookResult:
        """Deliver webhook payload with retry logic.

        Retries on 5xx errors with exponential backoff.
        Gives up immediately on 4xx errors.

        Args:
            callback_url: URL to POST the payload to.
            payload: WebhookPayload to deliver.

        Returns:
            WebhookResult with success status and metadata.
        """
        attempts = 0
        last_error = None
        last_status = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while attempts < self.max_retries:
                attempts += 1

                try:
                    logger.info(
                        f"Webhook attempt {attempts}/{self.max_retries} to {callback_url}"
                    )

                    response = await client.post(
                        callback_url,
                        json=payload.model_dump(mode="json"),
                        headers={"Content-Type": "application/json"},
                    )

                    last_status = response.status_code

                    if response.is_success:
                        logger.info(
                            f"Webhook delivered successfully: {response.status_code}"
                        )
                        return WebhookResult(
                            success=True,
                            attempts=attempts,
                            final_status_code=last_status,
                        )

                    elif 400 <= response.status_code < 500:
                        # Client error - don't retry
                        last_error = f"Client error: {response.status_code} - {response.text}"
                        logger.warning(f"Webhook client error (not retrying): {last_error}")
                        break

                    else:
                        # Server error - retry
                        last_error = f"Server error: {response.status_code}"
                        logger.warning(f"Webhook server error (will retry): {last_error}")

                except httpx.RequestError as e:
                    last_error = f"Request error: {type(e).__name__}: {str(e)}"
                    logger.warning(f"Webhook request error (will retry): {last_error}")

                # Exponential backoff before retry
                if attempts < self.max_retries:
                    wait_time = self.backoff_base ** attempts
                    logger.info(f"Waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)

        # All retries exhausted - save for manual recovery
        logger.error(f"Webhook delivery failed after {attempts} attempts: {last_error}")
        await self._save_failed_payload(payload, callback_url, last_error)

        return WebhookResult(
            success=False,
            attempts=attempts,
            final_status_code=last_status,
            error=last_error,
        )

    async def _save_failed_payload(
        self,
        payload: WebhookPayload,
        callback_url: str,
        error: Optional[str],
    ) -> None:
        """Save failed payload for manual retry.

        Args:
            payload: The failed WebhookPayload.
            callback_url: The URL that was targeted.
            error: Error message describing the failure.
        """
        filename = f"{payload.job_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.failed_payloads_dir / filename

        recovery_data = {
            "callback_url": callback_url,
            "error": error,
            "payload": payload.model_dump(mode="json"),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

        filepath.write_text(json.dumps(recovery_data, indent=2))
        logger.info(f"Failed webhook payload saved to {filepath}")


def build_webhook_payload(
    job_id: str,
    mission_id: str,
    report_path: str,
    protocol_path: str,
    execution_metadata: dict[str, Any],
) -> WebhookPayload:
    """Build webhook payload from job result files.

    Args:
        job_id: Unique job identifier.
        mission_id: Mission identifier.
        report_path: Path to the markdown report file.
        protocol_path: Path to the protocol JSON file.
        execution_metadata: Execution metrics dictionary.

    Returns:
        WebhookPayload ready for delivery.
    """
    report_markdown = Path(report_path).read_text()
    protocol = json.loads(Path(protocol_path).read_text())

    return WebhookPayload(
        job_id=job_id,
        mission_id=mission_id,
        status="complete",
        completed_at=datetime.now(timezone.utc),
        execution_metadata=execution_metadata,
        report_markdown=report_markdown,
        protocol=protocol,
    )


__all__ = [
    "WebhookClient",
    "WebhookPayload",
    "WebhookResult",
    "build_webhook_payload",
]
