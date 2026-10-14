"""Signed, minimal post-commit completion receipts for TraceLab."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

import httpx

from deepsearch.config import TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV

from .auth import is_secure_tracelab_url

_transport_logger = logging.getLogger("tracelab.receipt")

DEFAULT_RECEIPT_TIMEOUT_SECONDS = 10.0
DEFAULT_RECEIPT_MAX_ATTEMPTS = 4
DEFAULT_RECEIPT_BACKOFF_BASE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class CompletionReceipt:
    """The complete TraceLab receipt contract; result artifacts stay in Postgres."""

    job_id: str
    mission_id: str
    status: Literal["complete"] = "complete"

    def body(self) -> bytes:
        """Return stable JSON bytes used for both transport and HMAC signing."""

        return json.dumps(
            {
                "job_id": self.job_id,
                "mission_id": self.mission_id,
                "status": self.status,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ReceiptDeliveryResult:
    """Redacted result of one bounded receipt-delivery sequence."""

    success: bool
    attempts: int
    retryable: bool
    final_status_code: int | None = None
    error: str | None = None


class CompletionReceiptClient:
    """Deliver signed receipts with bounded 429/5xx/transport retries."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        secret: str | None = None,
        timeout: float = DEFAULT_RECEIPT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_RECEIPT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_RECEIPT_BACKOFF_BASE_SECONDS,
    ) -> None:
        resolved_base_url = (
            base_url if base_url is not None else os.getenv("TRACELAB_BASE_URL", "")
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.secret = (
            secret
            if secret is not None
            else os.getenv(TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV, "")
        )
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.backoff_base = max(0.0, backoff_base)

    @property
    def is_configured(self) -> bool:
        """Return whether both destination and signing secret are available."""

        return bool(self.base_url and self.secret)

    async def deliver(self, receipt: CompletionReceipt) -> ReceiptDeliveryResult:
        """Deliver one receipt without ever raising into terminal persistence."""

        if not self.is_configured:
            return ReceiptDeliveryResult(
                success=False,
                attempts=0,
                retryable=False,
                error="TraceLab completion receipt transport is not configured",
            )
        if not is_secure_tracelab_url(self.base_url):
            return ReceiptDeliveryResult(
                success=False,
                attempts=0,
                retryable=False,
                error="TraceLab completion receipts require HTTPS outside localhost",
            )

        body = receipt.body()
        digest = hmac.new(
            self.secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-DeepSearch-Signature": f"sha256={digest}",
        }
        url = f"{self.base_url}/api/v1/webhooks/deepsearch"
        final_status: int | None = None
        error: str | None = None
        attempts = 0

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for attempt in range(1, self.max_attempts + 1):
                    attempts = attempt
                    try:
                        response = await client.post(url, content=body, headers=headers)
                        final_status = response.status_code
                        if 200 <= response.status_code < 300:
                            return ReceiptDeliveryResult(
                                success=True,
                                attempts=attempt,
                                retryable=False,
                                final_status_code=response.status_code,
                            )

                        retryable = response.status_code == 429 or response.status_code >= 500
                        error = f"TraceLab returned HTTP {response.status_code}"
                        if not retryable:
                            return ReceiptDeliveryResult(
                                success=False,
                                attempts=attempt,
                                retryable=False,
                                final_status_code=response.status_code,
                                error=error,
                            )
                    except (httpx.TimeoutException, httpx.RequestError) as exc:
                        final_status = None
                        error = f"{type(exc).__name__} during receipt delivery"

                    if attempt < self.max_attempts:
                        await asyncio.sleep(self.backoff_base * (2 ** (attempt - 1)))
        except Exception as exc:  # noqa: BLE001 - receipt failure cannot rewrite result
            error = f"{type(exc).__name__} during receipt delivery"

        result = ReceiptDeliveryResult(
            success=False,
            attempts=attempts,
            retryable=True,
            final_status_code=final_status,
            error=error or "Unknown receipt delivery failure",
        )
        _transport_logger.error(
            "TraceLab completion receipt failed for mission %s after %d attempts: %s",
            receipt.mission_id,
            result.attempts,
            result.error,
        )
        return result


__all__ = [
    "CompletionReceipt",
    "CompletionReceiptClient",
    "ReceiptDeliveryResult",
]
