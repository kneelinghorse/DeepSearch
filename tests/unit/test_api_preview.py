"""Unit tests for mission contract preview endpoint."""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture(autouse=True)
def preview_auth_dev_mode(monkeypatch):
    """Keep legacy preview tests in dev-fallback mode."""

    monkeypatch.delenv(TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV, raising=False)


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


def test_preview_endpoint_preserves_legacy_fields(client, preview_payload):
    """Compiler provenance is additive to the established preview response."""

    response = client.post("/api/v1/missions/preview", json=preview_payload)

    assert response.status_code == 200
    payload = response.json()
    assert "Rspack" not in payload["named_entities"]
    assert "Vite" in payload["named_entities"]
    assert payload["coverage_thresholds"] == {"objective": 0.7}
    assert payload["validation_thresholds"] == {"slot": 0.5}
    assert payload["objectives"]
    assert payload["evidence_slots"]
    assert payload["acceptance_checks"]
    assert payload["deliverable_schemas"]


def test_preview_endpoint_returns_structured_compile_error(client):
    """Invalid authored schema should surface field-scoped compile errors."""

    response = client.post(
        "/api/v1/missions/preview",
        json={
            "title": "Broken preview payload",
            "objective": "Validate compiler error reporting",
            "expectedOutputSchema": {"title": "Broken", "format": "table"},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "error": "Mission contract preview failed",
        "field": "mission_context.expected_output_schema[1].kind",
        "message": "is required",
    }


def test_preview_endpoint_present_in_openapi_schema(client):
    """OpenAPI should publish the preview route for TraceLab discovery."""

    response = client.get("/openapi.json")
    data = response.json()

    assert "/api/v1/missions/preview" in data["paths"]
    assert "post" in data["paths"]["/api/v1/missions/preview"]
