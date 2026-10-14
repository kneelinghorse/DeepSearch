"""Symmetric HMAC validation for TraceLab-facing API routes."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re

from fastapi import HTTPException, Request

from deepsearch.config import TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV

logger = logging.getLogger(__name__)

TRACE_LAB_SIGNATURE_HEADER = "X-TraceLab-Signature"
TRACE_LAB_TIMESTAMP_HEADER = "X-TraceLab-Timestamp"
_SIGNATURE_PATTERN = re.compile(r"^sha256=(?P<digest>[0-9a-fA-F]{64})$")


class HMACValidationError(Exception):
    """Raised when a TraceLab HMAC header fails validation."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _signed_payload(payload_body: bytes, timestamp: str | None) -> bytes:
    """Return the HMAC message bytes for the request payload."""

    if timestamp:
        return timestamp.encode("utf-8") + b"." + payload_body
    return payload_body


def validate_signature(
    payload_body: bytes,
    signature: str | None,
    timestamp: str | None = None,
) -> bool:
    """Validate a TraceLab request signature against the shared secret."""

    secret = os.environ.get(TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV)
    if not secret:
        logger.warning(
            "Skipping TraceLab signature validation because %s is unset.",
            TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV,
        )
        return True

    if not signature:
        raise HMACValidationError("missing_signature")

    match = _SIGNATURE_PATTERN.fullmatch(signature.strip())
    if match is None:
        raise HMACValidationError("invalid_signature")

    expected_digest = hmac.new(
        secret.encode("utf-8"),
        _signed_payload(payload_body, timestamp),
        hashlib.sha256,
    ).hexdigest()
    provided_digest = match.group("digest").lower()

    if not hmac.compare_digest(provided_digest, expected_digest):
        raise HMACValidationError("invalid_signature")

    return True


async def require_tracelab_signature(request: Request) -> bytes:
    """Require a valid TraceLab HMAC signature for the current request."""

    # Starlette caches request.body(), so validating here does not prevent the
    # route's Pydantic body parsing from reading the same payload afterward.
    payload_body = await request.body()
    request.state.raw_body = payload_body

    try:
        validate_signature(
            payload_body,
            request.headers.get(TRACE_LAB_SIGNATURE_HEADER),
            request.headers.get(TRACE_LAB_TIMESTAMP_HEADER),
        )
    except HMACValidationError as exc:
        raise HTTPException(status_code=401, detail={"error": exc.detail}) from exc

    return payload_body
