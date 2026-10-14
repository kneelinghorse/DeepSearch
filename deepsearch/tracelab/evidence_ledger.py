"""Authenticated, trigger-only Evidence Ledger delivery for TraceLab."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, urlparse
from uuid import UUID

import httpx

from .auth import TraceLabAuth, TraceLabAuthError, is_secure_tracelab_url

_transport_logger = logging.getLogger("tracelab.evidence_ledger")

DEFAULT_LEDGER_TIMEOUT_SECONDS = 10.0
DEFAULT_LEDGER_MAX_ATTEMPTS = 4
DEFAULT_LEDGER_BACKOFF_BASE_SECONDS = 1.0
_MAX_LEDGER_TIMEOUT_SECONDS = 30.0
_MAX_LEDGER_ATTEMPTS = 4
_MAX_LEDGER_BACKOFF_BASE_SECONDS = 5.0
_TRACELAB_JOB_ID_MAX_LENGTH = 100

EvidenceLedgerDeliveryCategory = Literal[
    "accepted",
    "not_configured",
    "api_key_required",
    "invalid_destination",
    "invalid_mission_uuid",
    "invalid_job_id",
    "authentication",
    "authorization",
    "identity_mismatch",
    "client_rejected",
    "redirect_rejected",
    "rate_limited",
    "server_error",
    "transport_error",
    "internal_error",
]


@dataclass(frozen=True, slots=True)
class EvidenceLedgerTrigger:
    """Identity-only request for TraceLab to project persisted evidence."""

    mission_uuid: str
    deepsearch_job_id: str

    def body(self) -> bytes:
        """Return the complete schema-version-1 trigger body."""

        return json.dumps(
            {
                "schema_version": 1,
                "deepsearch_job_id": self.deepsearch_job_id,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EvidenceLedgerDeliveryResult:
    """Redacted result of one bounded Evidence Ledger delivery sequence."""

    success: bool
    attempts: int
    retryable: bool
    category: EvidenceLedgerDeliveryCategory
    final_status_code: int | None = None


class EvidenceLedgerClient:
    """Deliver minimal Evidence Ledger triggers with service API-key auth."""

    def __init__(
        self,
        *,
        auth: TraceLabAuth | None = None,
        timeout: float = DEFAULT_LEDGER_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_LEDGER_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_LEDGER_BACKOFF_BASE_SECONDS,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= _MAX_LEDGER_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "timeout must be finite and between 0 and 30 seconds"
            )
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= _MAX_LEDGER_ATTEMPTS
        ):
            raise ValueError("max_attempts must be an integer between 1 and 4")
        if (
            isinstance(backoff_base, bool)
            or not isinstance(backoff_base, (int, float))
            or not math.isfinite(backoff_base)
            or not 0 <= backoff_base <= _MAX_LEDGER_BACKOFF_BASE_SECONDS
        ):
            raise ValueError(
                "backoff_base must be finite and between 0 and 5 seconds"
            )
        self.auth = auth or TraceLabAuth()
        self.timeout = float(timeout)
        self.max_attempts = max_attempts
        self.backoff_base = float(backoff_base)

    @property
    def base_url(self) -> str:
        """Return the credential-owned TraceLab origin."""

        return self.auth.base_url.rstrip("/")

    @property
    def is_configured(self) -> bool:
        """Return whether a service API key and strict destination are ready."""

        return bool(
            self.auth.is_configured
            and self.auth.uses_api_key
            and _is_strict_tracelab_origin(self.base_url)
        )

    async def deliver(
        self,
        trigger: EvidenceLedgerTrigger,
    ) -> EvidenceLedgerDeliveryResult:
        """Deliver one trigger without raising into terminal persistence."""

        prerequisite_failure = self._validate_prerequisites(trigger)
        if prerequisite_failure is not None:
            return prerequisite_failure

        mission_segment = quote(trigger.mission_uuid, safe="")
        url = f"{self.base_url}/api/v1/missions/{mission_segment}/evidence"
        body = trigger.body()
        final_status: int | None = None
        category: EvidenceLedgerDeliveryCategory = "internal_error"
        attempts = 0

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,
            ) as client:
                for attempt in range(1, self.max_attempts + 1):
                    attempts = attempt
                    try:
                        await asyncio.to_thread(self.auth.require_service_role)
                        headers = {
                            **self.auth.get_headers(),
                            "Content-Type": "application/json",
                        }
                        response = await client.post(
                            url,
                            content=body,
                            headers=headers,
                        )
                        final_status = response.status_code
                        category, retryable = _classify_status(response.status_code)
                        if category == "accepted":
                            return EvidenceLedgerDeliveryResult(
                                success=True,
                                attempts=attempt,
                                retryable=False,
                                category=category,
                                final_status_code=response.status_code,
                            )
                        if not retryable:
                            return EvidenceLedgerDeliveryResult(
                                success=False,
                                attempts=attempt,
                                retryable=False,
                                category=category,
                                final_status_code=response.status_code,
                            )
                    except TraceLabAuthError as exc:
                        final_status = exc.status_code
                        category, retryable = _classify_auth_error(exc)
                        if not retryable:
                            return EvidenceLedgerDeliveryResult(
                                success=False,
                                attempts=attempt,
                                retryable=False,
                                category=category,
                                final_status_code=final_status,
                            )
                    except (httpx.TimeoutException, httpx.RequestError):
                        final_status = None
                        category = "transport_error"
                        retryable = True

                    if attempt < self.max_attempts:
                        await asyncio.sleep(self.backoff_base * (2 ** (attempt - 1)))
        except Exception:  # noqa: BLE001 - transport failure cannot rewrite a result
            result = EvidenceLedgerDeliveryResult(
                success=False,
                attempts=attempts,
                retryable=False,
                category="internal_error",
                final_status_code=final_status,
            )
            _log_failure(result)
            return result

        result = EvidenceLedgerDeliveryResult(
            success=False,
            attempts=attempts,
            retryable=True,
            category=category,
            final_status_code=final_status,
        )
        _log_failure(result)
        return result

    def _validate_prerequisites(
        self,
        trigger: EvidenceLedgerTrigger,
    ) -> EvidenceLedgerDeliveryResult | None:
        """Reject configuration and identity defects before authentication."""

        if not self.auth.is_configured:
            return _permanent_failure("not_configured")
        if not self.auth.uses_api_key:
            return _permanent_failure("api_key_required")
        if not _is_strict_tracelab_origin(self.base_url):
            return _permanent_failure("invalid_destination")
        try:
            canonical_uuid = str(UUID(trigger.mission_uuid))
        except (AttributeError, TypeError, ValueError):
            return _permanent_failure("invalid_mission_uuid")
        if trigger.mission_uuid != canonical_uuid:
            return _permanent_failure("invalid_mission_uuid")
        job_id = trigger.deepsearch_job_id
        if (
            not isinstance(job_id, str)
            or not job_id
            or job_id != job_id.strip()
            or len(job_id) > _TRACELAB_JOB_ID_MAX_LENGTH
        ):
            return _permanent_failure("invalid_job_id")
        return None


def _is_strict_tracelab_origin(base_url: str) -> bool:
    """Require the shared secure policy plus a credential-safe bare origin."""

    if not is_secure_tracelab_url(base_url):
        return False
    try:
        parsed = urlparse(base_url)
        parsed.port
    except ValueError:
        return False
    return bool(
        parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and parsed.path in ("", "/")
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )


def _classify_status(
    status_code: int,
) -> tuple[EvidenceLedgerDeliveryCategory, bool]:
    """Classify one response without retaining its body."""

    if 200 <= status_code < 300:
        return "accepted", False
    if status_code == 429:
        return "rate_limited", True
    if 500 <= status_code < 600:
        return "server_error", True
    if status_code == 401:
        return "authentication", False
    if status_code == 403:
        return "authorization", False
    if status_code == 409:
        return "identity_mismatch", False
    if 400 <= status_code < 500:
        return "client_rejected", False
    if 300 <= status_code < 400:
        return "redirect_rejected", False
    return "client_rejected", False


def _classify_auth_error(
    exc: TraceLabAuthError,
) -> tuple[EvidenceLedgerDeliveryCategory, bool]:
    """Classify redacted service-role failures under the same retry policy."""

    status_code = exc.status_code
    if status_code == 429:
        return "rate_limited", True
    if status_code is not None and 500 <= status_code < 600:
        return "server_error", True
    if status_code == 403:
        return "authorization", False
    if _has_transport_cause(exc):
        return "transport_error", True
    return "authentication", False


def _has_transport_cause(exc: BaseException) -> bool:
    """Return whether a sanitized auth failure wraps an HTTP transport error."""

    cause = exc.__cause__
    while cause is not None:
        if isinstance(cause, (httpx.TimeoutException, httpx.RequestError)):
            return True
        cause = cause.__cause__
    return False


def _permanent_failure(
    category: EvidenceLedgerDeliveryCategory,
) -> EvidenceLedgerDeliveryResult:
    """Build a pre-network permanent failure."""

    return EvidenceLedgerDeliveryResult(
        success=False,
        attempts=0,
        retryable=False,
        category=category,
    )


def _log_failure(result: EvidenceLedgerDeliveryResult) -> None:
    """Log a redacted transport failure without request identifiers."""

    _transport_logger.error(
        "TraceLab Evidence Ledger delivery failed after %d attempts "
        "(category=%s, status=%s)",
        result.attempts,
        result.category,
        result.final_status_code,
    )


__all__ = [
    "EvidenceLedgerClient",
    "EvidenceLedgerDeliveryCategory",
    "EvidenceLedgerDeliveryResult",
    "EvidenceLedgerTrigger",
]
