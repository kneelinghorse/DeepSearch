"""Unit tests for webhook delivery."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
pytest.importorskip("httpx")
import httpx

from deepsearch.api.webhook import (
    WebhookClient,
    WebhookPayload,
    WebhookResult,
    build_webhook_payload,
)


@pytest.fixture
def sample_payload():
    """Create a sample webhook payload."""
    return WebhookPayload(
        job_id="test-job-123",
        mission_id="DRM.0.1",
        status="complete",
        completed_at=datetime.now(timezone.utc),
        execution_metadata={
            "duration_seconds": 128.5,
            "loops_completed": 3,
            "findings_count": 9,
            "sources_count": 15,
        },
        report_markdown="# Test Report\n\nThis is a test report.",
        protocol={
            "research_statement": {"topic": "Test", "objective": "Testing"},
            "key_questions": [],
            "evidence": [],
            "synthesis": {},
            "quality_checkpoints": [],
        },
    )


@pytest.fixture
def temp_payloads_dir():
    """Create a temporary directory for failed payloads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestWebhookPayload:
    """Tests for WebhookPayload model."""

    def test_payload_serialization(self, sample_payload):
        """Payload serializes correctly to JSON."""
        data = sample_payload.model_dump(mode="json")
        assert data["job_id"] == "test-job-123"
        assert data["mission_id"] == "DRM.0.1"
        assert data["status"] == "complete"
        assert "completed_at" in data
        assert data["execution_metadata"]["loops_completed"] == 3

    def test_payload_structure_matches_contract(self, sample_payload):
        """Payload matches TraceLab expected contract."""
        data = sample_payload.model_dump(mode="json")

        # All required fields present
        required_fields = [
            "job_id",
            "mission_id",
            "status",
            "completed_at",
            "execution_metadata",
            "report_markdown",
            "protocol",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"


class TestWebhookClient:
    """Tests for WebhookClient delivery."""

    @pytest.mark.asyncio
    async def test_successful_delivery(self, sample_payload, temp_payloads_dir):
        """200 response returns success=True, attempts=1."""
        client = WebhookClient(
            max_retries=3,
            failed_payloads_dir=temp_payloads_dir,
        )

        with patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_post.return_value = mock_response

            result = await client.deliver(
                "https://api.example.com/webhook",
                sample_payload,
            )

            assert result.success is True
            assert result.attempts == 1
            assert result.final_status_code == 200
            assert result.error is None

    @pytest.mark.asyncio
    async def test_retry_on_5xx(self, sample_payload, temp_payloads_dir):
        """500 error triggers retry with backoff."""
        client = WebhookClient(
            max_retries=3,
            backoff_base=0.01,  # Fast backoff for tests
            failed_payloads_dir=temp_payloads_dir,
        )

        with patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
        ) as mock_post:
            # First two calls return 500, third succeeds
            mock_500 = AsyncMock()
            mock_500.status_code = 500
            mock_500.is_success = False

            mock_200 = AsyncMock()
            mock_200.status_code = 200
            mock_200.is_success = True

            mock_post.side_effect = [mock_500, mock_500, mock_200]

            result = await client.deliver(
                "https://api.example.com/webhook",
                sample_payload,
            )

            assert result.success is True
            assert result.attempts == 3
            assert result.final_status_code == 200

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self, sample_payload, temp_payloads_dir):
        """400 error returns immediately, no retry."""
        client = WebhookClient(
            max_retries=3,
            failed_payloads_dir=temp_payloads_dir,
        )

        with patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 400
            mock_response.is_success = False
            mock_response.text = "Bad Request"
            mock_post.return_value = mock_response

            result = await client.deliver(
                "https://api.example.com/webhook",
                sample_payload,
            )

            assert result.success is False
            assert result.attempts == 1  # No retries
            assert result.final_status_code == 400
            assert "Client error" in result.error

    @pytest.mark.asyncio
    async def test_failed_payload_saved(self, sample_payload, temp_payloads_dir):
        """After max retries, payload saved to disk."""
        client = WebhookClient(
            max_retries=2,
            backoff_base=0.01,
            failed_payloads_dir=temp_payloads_dir,
        )

        with patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.is_success = False
            mock_post.return_value = mock_response

            result = await client.deliver(
                "https://api.example.com/webhook",
                sample_payload,
            )

            assert result.success is False
            assert result.attempts == 2

            # Check payload was saved
            saved_files = list(Path(temp_payloads_dir).glob("*.json"))
            assert len(saved_files) == 1

            # Verify saved content
            saved_data = json.loads(saved_files[0].read_text())
            assert saved_data["callback_url"] == "https://api.example.com/webhook"
            assert saved_data["payload"]["job_id"] == "test-job-123"
            assert "error" in saved_data
            assert "failed_at" in saved_data


class TestBuildWebhookPayload:
    """Tests for build_webhook_payload helper."""

    def test_build_from_files(self):
        """Builds payload from report and protocol files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            report_path = Path(tmpdir) / "report.md"
            report_path.write_text("# Test Report\n\nFindings here.")

            protocol_path = Path(tmpdir) / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "research_statement": {"topic": "Test"},
                        "key_questions": [],
                        "evidence": [],
                    }
                )
            )

            payload = build_webhook_payload(
                job_id="job-456",
                mission_id="DRM.0.2",
                report_path=str(report_path),
                protocol_path=str(protocol_path),
                execution_metadata={"loops_completed": 2},
            )

            assert payload.job_id == "job-456"
            assert payload.mission_id == "DRM.0.2"
            assert payload.status == "complete"
            assert payload.report_markdown == "# Test Report\n\nFindings here."
            assert payload.protocol["research_statement"]["topic"] == "Test"
            assert payload.execution_metadata["loops_completed"] == 2

    def test_build_missing_report_raises(self):
        """Missing report file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol.json"
            protocol_path.write_text("{}")

            with pytest.raises(FileNotFoundError):
                build_webhook_payload(
                    job_id="job-789",
                    mission_id="DRM.0.3",
                    report_path="/nonexistent/report.md",
                    protocol_path=str(protocol_path),
                    execution_metadata={},
                )
