"""Unit tests for health endpoints."""

import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from deepsearch.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client):
        """Test that /health returns status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_timestamp(self, client):
        """Test that /health includes a timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
        assert len(data["timestamp"]) > 0

    def test_health_includes_version(self, client):
        """Test that /health includes version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"


class TestReadyEndpoint:
    """Tests for /ready endpoint."""

    def test_ready_returns_ready(self, client):
        """Test that /ready returns ready status."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_ready_includes_timestamp(self, client):
        """Test that /ready includes a timestamp."""
        response = client.get("/ready")
        data = response.json()
        assert "timestamp" in data

    def test_ready_includes_checks(self, client):
        """Test that /ready includes dependency checks."""
        response = client.get("/ready")
        data = response.json()
        assert "checks" in data
        assert isinstance(data["checks"], dict)
        assert data["checks"]["database"] == "ok"


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_preflight_request(self, client):
        """Test that CORS preflight works for allowed origin."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_configured_origin(self, client):
        """Test that configured origin is allowed."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


class TestOpenAPI:
    """Tests for OpenAPI docs."""

    def test_openapi_docs_accessible(self, client):
        """Test that /docs is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_accessible(self, client):
        """Test that OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "DeepSearch API"
        assert "/health" in data["paths"]
        assert "/ready" in data["paths"]
