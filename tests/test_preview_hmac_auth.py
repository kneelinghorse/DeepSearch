"""Unit tests for preview endpoint HMAC auth."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import deepsearch.api.auth.hmac as preview_hmac
from deepsearch.api.app import create_app
from deepsearch.api.queue import JobQueue
from deepsearch.config import TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV

pytest.importorskip("fastapi")


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def client(temp_db_path):
    """Create test client with queue initialized."""

    app = create_app()
    app.state.queue = JobQueue(db_path=temp_db_path)
    return TestClient(app)


@pytest.fixture
def preview_payload():
    """Return a representative authored mission_context payload."""

    return {
        "missionId": "PREVIEW.1",
        "title": "TraceLab preview payload",
        "objective": "Compare Vite, esbuild, Webpack, Rollup, and Turbopack.",
        "successCriteria": [
            "Compare Vite, esbuild, Webpack, Rollup, and Turbopack architectures",
            "Include a comparison matrix",
        ],
        "deliverables": [
            "Annotated bibliography with at least 5 entries",
        ],
        "background": "Use the authored preview to verify compiler parity.",
        "requiredEntities": ["Rspack", "Vite"],
        "excludedEntities": ["Rspack"],
        "expectedOutputSchema": {
            "kind": "classification_matrix",
            "title": "Bundler Classification Matrix",
            "format": "table",
            "columns": ["Bundler", "Architecture", "Evidence"],
            "required_sections": ["Reviewer Notes"],
            "entity_extraction_required": True,
            "entity_types": ["author_year", "acronym"],
            "minimum_citation_count": 5,
        },
        "coverageThresholds": {"objective": 0.7},
        "validationThresholds": {"slot": 0.5},
        "researchDepth": "baseline",
        "maxLoops": 3,
    }


@pytest.fixture
def service_secret(monkeypatch):
    """Configure the shared preview secret for auth tests."""

    secret = "preview-shared-secret"
    monkeypatch.setenv(TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV, secret)
    return secret


@pytest.fixture
def generate_valid_signature():
    """Return a helper that signs preview request bodies."""

    def _generate(body: bytes, secret: str, timestamp: str | None = None) -> str:
        payload = body if timestamp is None else timestamp.encode("utf-8") + b"." + body
        digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    return _generate


def _request_body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def test_preview_rejects_missing_signature(client, preview_payload, service_secret):
    """Requests must include a signature when the shared secret is set."""

    response = client.post(
        "/api/v1/missions/preview",
        content=_request_body(preview_payload),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == {"error": "missing_signature"}


def test_preview_rejects_malformed_signature(client, preview_payload, service_secret):
    """Requests with malformed signature headers should be rejected."""

    response = client.post(
        "/api/v1/missions/preview",
        content=_request_body(preview_payload),
        headers={
            "content-type": "application/json",
            preview_hmac.TRACE_LAB_SIGNATURE_HEADER: "not-a-sha256-digest",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == {"error": "invalid_signature"}


def test_preview_accepts_valid_signature(
    client,
    preview_payload,
    service_secret,
    generate_valid_signature,
):
    """Requests with a valid body signature should succeed."""

    body = _request_body(preview_payload)
    response = client.post(
        "/api/v1/missions/preview",
        content=body,
        headers={
            "content-type": "application/json",
            preview_hmac.TRACE_LAB_SIGNATURE_HEADER: generate_valid_signature(
                body, service_secret
            ),
        },
    )

    assert response.status_code == 200
    assert "Vite" in response.json()["named_entities"]


def test_preview_accepts_valid_signature_with_timestamp(
    client,
    preview_payload,
    service_secret,
    generate_valid_signature,
):
    """Timestamped signatures should validate against timestamp.body."""

    body = _request_body(preview_payload)
    timestamp = "2026-04-22T19:00:00Z"
    response = client.post(
        "/api/v1/missions/preview",
        content=body,
        headers={
            "content-type": "application/json",
            preview_hmac.TRACE_LAB_SIGNATURE_HEADER: generate_valid_signature(
                body,
                service_secret,
                timestamp,
            ),
            preview_hmac.TRACE_LAB_TIMESTAMP_HEADER: timestamp,
        },
    )

    assert response.status_code == 200
    assert "Vite" in response.json()["named_entities"]


def test_preview_rejects_signature_signed_with_wrong_secret(
    client,
    preview_payload,
    service_secret,
    generate_valid_signature,
):
    """Signatures created with a different secret must be rejected."""

    body = _request_body(preview_payload)
    response = client.post(
        "/api/v1/missions/preview",
        content=body,
        headers={
            "content-type": "application/json",
            preview_hmac.TRACE_LAB_SIGNATURE_HEADER: generate_valid_signature(
                body,
                "different-secret",
            ),
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == {"error": "invalid_signature"}


def test_preview_dev_fallback_skips_signature_validation(
    client,
    preview_payload,
    monkeypatch,
    caplog,
):
    """Unset secret should skip validation and log a warning."""

    monkeypatch.delenv(TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV, raising=False)

    response = client.post(
        "/api/v1/missions/preview",
        content=_request_body(preview_payload),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV in caplog.text
    assert "Skipping TraceLab signature validation" in caplog.text


def test_preview_validation_uses_compare_digest(
    client,
    preview_payload,
    service_secret,
    generate_valid_signature,
    monkeypatch,
):
    """Signature validation should use hmac.compare_digest."""

    original_compare_digest = preview_hmac.hmac.compare_digest
    called = {"used": False}

    def _tracking_compare_digest(left: str, right: str) -> bool:
        called["used"] = True
        return original_compare_digest(left, right)

    monkeypatch.setattr(preview_hmac.hmac, "compare_digest", _tracking_compare_digest)

    body = _request_body(preview_payload)
    response = client.post(
        "/api/v1/missions/preview",
        content=body,
        headers={
            "content-type": "application/json",
            preview_hmac.TRACE_LAB_SIGNATURE_HEADER: generate_valid_signature(
                body, service_secret
            ),
        },
    )

    assert response.status_code == 200
    assert called["used"] is True
