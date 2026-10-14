"""Unit tests for job status and result endpoints."""

import json
import os
import tempfile
from datetime import datetime

import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from deepsearch.api.app import create_app
from deepsearch.api.queue import JobQueue, JobStatus


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
    queue = JobQueue(db_path=temp_db_path)
    app.state.queue = queue
    return TestClient(app)


@pytest.fixture
def valid_mission_request():
    """Return a valid mission request payload."""
    return {
        "mission_id": "TEST.1",
        "title": "Test Research Mission",
        "objectives": ["Research objective 1"],
        "success_criteria": ["Deliverable output"],
        "max_loops": 3,
    }


class TestStatusEndpointQueued:
    """Tests for status of queued jobs."""

    def test_queued_job_returns_200(self, client, valid_mission_request):
        """Test that queued job status returns 200."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        status_response = client.get(f"/api/v1/jobs/{job_id}/status")
        assert status_response.status_code == 200

    def test_queued_job_status_is_queued(self, client, valid_mission_request):
        """Test that status is 'queued' for queued job."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["status"] == "queued"

    def test_queued_job_has_no_progress(self, client, valid_mission_request):
        """Test that queued job has no progress info."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["progress"] is None

    def test_queued_job_has_no_result_url(self, client, valid_mission_request):
        """Test that queued job has no result_url."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["result_url"] is None

    def test_queued_job_includes_mission_id(self, client, valid_mission_request):
        """Test that response includes mission_id."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["mission_id"] == "TEST.1"

    def test_queued_job_includes_created_at(self, client, valid_mission_request):
        """Test that response includes created_at timestamp."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert "created_at" in data
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))


class TestStatusEndpointRunning:
    """Tests for status of running jobs."""

    def test_running_job_status(self, client, valid_mission_request):
        """Test that running job has status 'running'."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        # Manually update status to running
        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_progress(job_id, current_loop=2, findings_count=5)

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["status"] == "running"

    def test_running_job_has_progress(self, client, valid_mission_request):
        """Test that running job includes progress info."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_progress(job_id, current_loop=2, findings_count=5)

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["progress"] is not None
        assert data["progress"]["current_loop"] == 2
        assert data["progress"]["max_loops"] == 3
        assert data["progress"]["findings_count"] == 5

    def test_running_job_progress_percentage(self, client, valid_mission_request):
        """Test that progress percentage is calculated correctly."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_progress(job_id, current_loop=2, findings_count=5)

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        # 2/3 = 66%
        assert data["progress"]["percentage"] == 66

    def test_running_job_has_started_at(self, client, valid_mission_request):
        """Test that running job has started_at timestamp."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["started_at"] is not None

    def test_running_job_has_no_result_url(self, client, valid_mission_request):
        """Test that running job has no result_url."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["result_url"] is None


class TestStatusEndpointComplete:
    """Tests for status of completed jobs."""

    def test_complete_job_status(self, client, valid_mission_request, tmp_path):
        """Test that complete job has status 'complete'."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        # Create report file
        report_path = tmp_path / "report.md"
        report_path.write_text("# Test Report\nContent here.")

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.COMPLETE, result_path=str(report_path))

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["status"] == "complete"

    def test_complete_job_has_result_url(self, client, valid_mission_request, tmp_path):
        """Test that complete job includes result_url."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        report_path = tmp_path / "report.md"
        report_path.write_text("# Test Report")

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.COMPLETE, result_path=str(report_path))

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["result_url"] == f"/api/v1/jobs/{job_id}/result"

    def test_complete_job_has_completed_at(self, client, valid_mission_request, tmp_path):
        """Test that complete job has completed_at timestamp."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        report_path = tmp_path / "report.md"
        report_path.write_text("# Test Report")

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.COMPLETE, result_path=str(report_path))

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["completed_at"] is not None

    def test_complete_job_no_progress(self, client, valid_mission_request, tmp_path):
        """Test that complete job has no progress info."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        report_path = tmp_path / "report.md"
        report_path.write_text("# Test Report")

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.COMPLETE, result_path=str(report_path))

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["progress"] is None


class TestStatusEndpointFailed:
    """Tests for status of failed jobs."""

    def test_failed_job_status(self, client, valid_mission_request):
        """Test that failed job has status 'failed'."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.FAILED, error="Test error message")

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["status"] == "failed"

    def test_failed_job_has_error(self, client, valid_mission_request):
        """Test that failed job includes error message."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.FAILED, error="Test error message")

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["error"] == "Test error message"

    def test_failed_job_has_no_result_url(self, client, valid_mission_request):
        """Test that failed job has no result_url."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.FAILED, error="Test error")

        data = client.get(f"/api/v1/jobs/{job_id}/status").json()
        assert data["result_url"] is None


class TestStatusEndpoint404:
    """Tests for 404 responses."""

    def test_unknown_job_returns_404(self, client):
        """Test that unknown job_id returns 404."""
        response = client.get("/api/v1/jobs/unknown-job-id/status")
        assert response.status_code == 404

    def test_404_includes_detail(self, client):
        """Test that 404 includes error detail."""
        response = client.get("/api/v1/jobs/unknown-job-id/status")
        data = response.json()
        assert "detail" in data
        assert "unknown-job-id" in data["detail"]


class TestResultEndpoint:
    """Tests for result retrieval endpoint."""

    def test_result_returns_report_markdown(self, client, valid_mission_request, tmp_path):
        """Test that result includes report markdown."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        # Create report file
        report_path = tmp_path / "report.md"
        report_path.write_text("# Test Report\n\nThis is test content.")

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_progress(job_id, current_loop=3, findings_count=10)
        queue.update_status(job_id, JobStatus.COMPLETE, result_path=str(report_path))

        data = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert data["report_markdown"] == "# Test Report\n\nThis is test content."

    def test_result_returns_protocol_json(self, client, valid_mission_request, tmp_path):
        """Test that result includes protocol JSON."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        # Create report and protocol files
        report_path = tmp_path / "report.md"
        report_path.write_text("# Test Report")

        protocol_path = tmp_path / "report_protocol.json"
        protocol_data = {
            "evidence": [
                {"url": "https://example.com", "title": "Example"},
            ],
            "findings": ["Finding 1", "Finding 2"],
        }
        protocol_path.write_text(json.dumps(protocol_data))

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.COMPLETE, result_path=str(report_path))

        data = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert data["protocol_json"]["evidence"][0]["url"] == "https://example.com"
        assert data["protocol_json"]["findings"] == ["Finding 1", "Finding 2"]

    def test_result_includes_execution_metadata(
        self, client, valid_mission_request, tmp_path
    ):
        """Test that result includes execution metadata."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        report_path = tmp_path / "report.md"
        report_path.write_text("# Test Report")

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_progress(job_id, current_loop=3, findings_count=8)
        queue.update_status(job_id, JobStatus.COMPLETE, result_path=str(report_path))

        data = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert "execution_metadata" in data
        assert data["execution_metadata"]["loops_completed"] == 3
        assert data["execution_metadata"]["findings_count"] == 8

    def test_result_handles_missing_protocol(
        self, client, valid_mission_request, tmp_path
    ):
        """Test that result works without protocol file."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        report_path = tmp_path / "report.md"
        report_path.write_text("# Test Report")

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)
        queue.update_status(job_id, JobStatus.COMPLETE, result_path=str(report_path))

        data = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert data["protocol_json"] == {}


class TestResultEndpointErrors:
    """Tests for result endpoint error cases."""

    def test_result_404_unknown_job(self, client):
        """Test that unknown job_id returns 404."""
        response = client.get("/api/v1/jobs/unknown-job-id/result")
        assert response.status_code == 404

    def test_result_400_job_not_complete(self, client, valid_mission_request):
        """Test that incomplete job returns 400."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        # Job is queued, not complete
        response = client.get(f"/api/v1/jobs/{job_id}/result")
        assert response.status_code == 400
        assert "not complete" in response.json()["detail"]

    def test_result_400_running_job(self, client, valid_mission_request):
        """Test that running job returns 400."""
        response = client.post("/api/v1/missions/execute", json=valid_mission_request)
        job_id = response.json()["job_id"]

        queue = client.app.state.queue
        queue.update_status(job_id, JobStatus.RUNNING)

        response = client.get(f"/api/v1/jobs/{job_id}/result")
        assert response.status_code == 400


class TestOpenAPISchema:
    """Tests for OpenAPI schema includes status endpoints."""

    def test_status_endpoint_in_schema(self, client):
        """Test that status endpoint is in OpenAPI schema."""
        response = client.get("/openapi.json")
        data = response.json()
        # FastAPI uses path parameter syntax
        status_path = "/api/v1/jobs/{job_id}/status"
        assert status_path in data["paths"]

    def test_result_endpoint_in_schema(self, client):
        """Test that result endpoint is in OpenAPI schema."""
        response = client.get("/openapi.json")
        data = response.json()
        result_path = "/api/v1/jobs/{job_id}/result"
        assert result_path in data["paths"]

    def test_endpoints_have_get_method(self, client):
        """Test that endpoints have GET method."""
        response = client.get("/openapi.json")
        data = response.json()
        assert "get" in data["paths"]["/api/v1/jobs/{job_id}/status"]
        assert "get" in data["paths"]["/api/v1/jobs/{job_id}/result"]
