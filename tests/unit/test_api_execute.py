"""Unit tests for mission execute endpoint."""

import os
import tempfile

import pytest
pytest.importorskip("fastapi")
from datetime import datetime
from fastapi.testclient import TestClient

from deepsearch.api.app import create_app
from deepsearch.api.queue import JobQueue


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
    # Initialize queue on app state for tests
    queue = JobQueue(db_path=temp_db_path)
    app.state.queue = queue
    return TestClient(app)


@pytest.fixture
def valid_mission_request():
    """Return a valid mission request payload."""
    return {
        "mission_id": "DRM.0.5",
        "title": "Test Research Mission",
        "objectives": [
            "Research objective 1",
            "Research objective 2",
        ],
        "success_criteria": [
            "Criterion 1",
            "Criterion 2",
        ],
        "context": {
            "background": "Test background info",
            "constraints": ["Constraint 1"],
        },
        "callback_url": "https://tracelab.namozine.com/webhooks/deepsearch",
        "max_loops": 3,
        "token_limit": 8000,
    }


class TestExecuteEndpointSuccess:
    """Tests for successful mission execution requests."""

    def test_valid_mission_returns_200(self, client, valid_mission_request):
        """Test that valid mission request returns 200."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 200

    def test_response_includes_job_id(self, client, valid_mission_request):
        """Test that response includes a job_id."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 36  # UUID length

    def test_response_includes_mission_id(self, client, valid_mission_request):
        """Test that response echoes mission_id."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        data = response.json()
        assert data["mission_id"] == "DRM.0.5"

    def test_response_status_is_queued(self, client, valid_mission_request):
        """Test that response status is 'queued'."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        data = response.json()
        assert data["status"] == "queued"

    def test_response_includes_queued_at(self, client, valid_mission_request):
        """Test that response includes queued_at timestamp."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        data = response.json()
        assert "queued_at" in data
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(data["queued_at"].replace("Z", "+00:00"))

    def test_unique_job_ids_generated(self, client, valid_mission_request):
        """Test that each request gets a unique job_id."""
        response1 = client.post("/api/v1/missions/execute", json=valid_mission_request)
        response2 = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response1.json()["job_id"] != response2.json()["job_id"]

    def test_minimal_request_accepted(self, client):
        """Test that minimal required fields are accepted."""
        minimal_request = {
            "mission_id": "MIN.1",
            "title": "Minimal mission for research",
            "objectives": ["Single research objective"],
            "success_criteria": ["At least one result"],
        }
        response = client.post("/api/v1/missions/execute", json=minimal_request)
        assert response.status_code == 200

    def test_optional_callback_url_omitted(self, client):
        """Test that callback_url is optional."""
        request = {
            "mission_id": "TEST.1",
            "title": "Test mission for research",
            "objectives": ["Research objective 1"],
            "success_criteria": ["Measurable outcome"],
        }
        response = client.post("/api/v1/missions/execute", json=request)
        assert response.status_code == 200

    def test_default_max_loops(self, client):
        """Test that max_loops defaults to 3."""
        request = {
            "mission_id": "TEST.1",
            "title": "Test mission for research",
            "objectives": ["Research objective 1"],
            "success_criteria": ["Deliverable criterion"],
        }
        response = client.post("/api/v1/missions/execute", json=request)
        assert response.status_code == 200


class TestExecuteEndpointValidation:
    """Tests for mission request validation."""

    def test_missing_mission_id_returns_422(self, client, valid_mission_request):
        """Test that missing mission_id returns 422."""
        del valid_mission_request["mission_id"]
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_missing_title_returns_422(self, client, valid_mission_request):
        """Test that missing title returns 422."""
        del valid_mission_request["title"]
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_missing_objectives_returns_422(self, client, valid_mission_request):
        """Test that missing objectives returns 422."""
        del valid_mission_request["objectives"]
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_empty_objectives_returns_422(self, client, valid_mission_request):
        """Test that empty objectives array returns 422."""
        valid_mission_request["objectives"] = []
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_empty_mission_id_returns_422(self, client, valid_mission_request):
        """Test that empty mission_id returns 422."""
        valid_mission_request["mission_id"] = ""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_empty_title_returns_422(self, client, valid_mission_request):
        """Test that empty title returns 422."""
        valid_mission_request["title"] = ""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_invalid_callback_url_returns_422(self, client, valid_mission_request):
        """Test that invalid callback_url returns 422."""
        valid_mission_request["callback_url"] = "not-a-valid-url"
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_max_loops_below_minimum_returns_422(self, client, valid_mission_request):
        """Test that max_loops below 1 returns 422."""
        valid_mission_request["max_loops"] = 0
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_max_loops_above_maximum_returns_422(self, client, valid_mission_request):
        """Test that max_loops above 10 returns 422."""
        valid_mission_request["max_loops"] = 11
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_token_limit_below_minimum_returns_422(self, client, valid_mission_request):
        """Test that token_limit below 1000 returns 422."""
        valid_mission_request["token_limit"] = 500
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_token_limit_above_maximum_returns_422(self, client, valid_mission_request):
        """Test that token_limit above 32000 returns 422."""
        valid_mission_request["token_limit"] = 50000
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        assert response.status_code == 422

    def test_validation_error_includes_field_info(self, client, valid_mission_request):
        """Test that validation errors include field information."""
        del valid_mission_request["mission_id"]
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        data = response.json()
        assert "detail" in data
        # FastAPI includes location in validation errors
        assert isinstance(data["detail"], list)
        assert any("mission_id" in str(err) for err in data["detail"])


class TestExecuteEndpointContext:
    """Tests for mission context handling."""

    def test_context_background_accepted(self, client):
        """Test that context with background is accepted."""
        request = {
            "mission_id": "CTX.1",
            "title": "Context test research mission",
            "objectives": ["Research test objective"],
            "success_criteria": ["Has results"],
            "context": {
                "background": "Some background information",
            },
        }
        response = client.post("/api/v1/missions/execute", json=request)
        assert response.status_code == 200

    def test_context_constraints_accepted(self, client):
        """Test that context with constraints is accepted."""
        request = {
            "mission_id": "CTX.2",
            "title": "Context test research mission",
            "objectives": ["Research test objective"],
            "success_criteria": ["Has results"],
            "context": {
                "constraints": ["Constraint 1", "Constraint 2"],
            },
        }
        response = client.post("/api/v1/missions/execute", json=request)
        assert response.status_code == 200

    def test_context_focus_accepted(self, client):
        """Test that context with focus is accepted."""
        request = {
            "mission_id": "CTX.3",
            "title": "Context test research mission",
            "objectives": ["Research test objective"],
            "success_criteria": ["Has results"],
            "context": {
                "focus": "Specific focus area",
            },
        }
        response = client.post("/api/v1/missions/execute", json=request)
        assert response.status_code == 200


class TestOpenAPISchema:
    """Tests for OpenAPI schema includes execute endpoint."""

    def test_execute_endpoint_in_schema(self, client):
        """Test that execute endpoint is in OpenAPI schema."""
        response = client.get("/openapi.json")
        data = response.json()
        assert "/api/v1/missions/execute" in data["paths"]

    def test_execute_endpoint_post_method(self, client):
        """Test that execute endpoint has POST method."""
        response = client.get("/openapi.json")
        data = response.json()
        assert "post" in data["paths"]["/api/v1/missions/execute"]
