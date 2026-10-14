"""Integration tests for JWT authentication and auto-upload functionality.

Tests cover:
- JWT login flow (TraceLabAuth)
- Token refresh and expiry handling
- Auto-upload triggered from runner
- 401 retry behavior
- Mock mode for CI
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from deepsearch.report.protocol import MissionProtocolComplete


class TestTraceLabAuth:
    """Tests for JWT authentication."""

    @pytest.fixture(autouse=True)
    def _isolate_jwt_fallback_from_live_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """These cases intentionally exercise JWT, regardless of host config."""

        monkeypatch.delenv("TRACELAB_API_KEY", raising=False)

    def test_auth_requires_credentials(self) -> None:
        """Auth should raise error when credentials missing."""
        from deepsearch.tracelab.auth import TraceLabAuth, TraceLabAuthError

        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            auth = TraceLabAuth()
            assert auth.is_configured is False

            with pytest.raises(TraceLabAuthError) as exc_info:
                auth.get_token()

            assert "TRACELAB_BASE_URL" in str(exc_info.value)

    def test_auth_configured_from_env(self) -> None:
        """Auth should read credentials from environment."""
        from deepsearch.tracelab.auth import TraceLabAuth

        env = {
            "TRACELAB_BASE_URL": "https://test.example.com",
            "TRACELAB_USERNAME": "testuser",
            "TRACELAB_PASSWORD": "testpass",
        }
        with patch.dict(os.environ, env, clear=True):
            auth = TraceLabAuth()

            assert auth.is_configured is True
            assert auth.base_url == "https://test.example.com"
            assert auth.username == "testuser"
            assert auth.password == "testpass"

    def test_auth_configured_from_args(self) -> None:
        """Auth should accept explicit credentials."""
        from deepsearch.tracelab.auth import TraceLabAuth

        auth = TraceLabAuth(
            base_url="https://explicit.example.com",
            username="explicit_user",
            password="explicit_pass",
        )

        assert auth.is_configured is True
        assert auth.base_url == "https://explicit.example.com"

    def test_auth_strips_trailing_slash(self) -> None:
        """Auth should strip trailing slashes from base_url."""
        from deepsearch.tracelab.auth import TraceLabAuth

        auth = TraceLabAuth(
            base_url="https://example.com/",
            username="user",
            password="pass",
        )

        assert auth.base_url == "https://example.com"

    def test_token_info_expiry(self) -> None:
        """TokenInfo should track expiry correctly."""
        from deepsearch.tracelab.auth import TokenInfo

        # Token that expires in 10 minutes
        token = TokenInfo(
            access_token="test-token",
            expires_at=time.time() + 600,  # 10 minutes
        )

        assert token.is_expired is False
        assert token.time_until_expiry > 0
        assert token.time_until_expiry <= 600

    def test_token_info_expired_within_buffer(self) -> None:
        """TokenInfo should be considered expired within buffer period."""
        from deepsearch.tracelab.auth import TokenInfo, TOKEN_REFRESH_BUFFER_SECONDS

        # Token that expires in 4 minutes (within 5 minute buffer)
        token = TokenInfo(
            access_token="test-token",
            expires_at=time.time() + 240,  # 4 minutes
        )

        assert token.is_expired is True  # Should be considered expired

    @patch("httpx.post")
    def test_auth_login_success(self, mock_post: MagicMock) -> None:
        """Auth should successfully login and cache token."""
        from deepsearch.tracelab.auth import TraceLabAuth

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test-jwt-token",
            "token_type": "bearer",
            "expires_in": 86400,
        }
        mock_post.return_value = mock_response

        auth = TraceLabAuth(
            base_url="https://test.example.com",
            username="testuser",
            password="testpass",
        )

        token = auth.get_token()

        assert token == "test-jwt-token"
        assert auth._token is not None
        assert auth._token.access_token == "test-jwt-token"
        assert auth._login_count == 1

        # Second call should use cached token
        token2 = auth.get_token()
        assert token2 == "test-jwt-token"
        assert auth._login_count == 1  # No additional login

    @patch("httpx.post")
    def test_auth_login_401_error(self, mock_post: MagicMock) -> None:
        """Auth should raise error on 401 (invalid credentials)."""
        from deepsearch.tracelab.auth import TraceLabAuth, TraceLabAuthError

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        auth = TraceLabAuth(
            base_url="https://test.example.com",
            username="baduser",
            password="badpass",
        )

        with pytest.raises(TraceLabAuthError) as exc_info:
            auth.get_token()

        assert exc_info.value.status_code == 401
        assert "Invalid credentials" in str(exc_info.value)

    @patch("httpx.post")
    def test_auth_get_headers(self, mock_post: MagicMock) -> None:
        """Auth should return Authorization headers."""
        from deepsearch.tracelab.auth import TraceLabAuth

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test-jwt-token",
            "expires_in": 86400,
        }
        mock_post.return_value = mock_response

        auth = TraceLabAuth(
            base_url="https://test.example.com",
            username="testuser",
            password="testpass",
        )

        headers = auth.get_headers()

        assert headers["Authorization"] == "Bearer test-jwt-token"

    @patch("httpx.post")
    def test_auth_refresh_success(self, mock_post: MagicMock) -> None:
        """Auth should refresh token successfully."""
        from deepsearch.tracelab.auth import TraceLabAuth

        # First call is login, second is refresh
        login_response = MagicMock()
        login_response.status_code = 200
        login_response.json.return_value = {
            "access_token": "original-token",
            "expires_in": 86400,
        }

        refresh_response = MagicMock()
        refresh_response.status_code = 200
        refresh_response.json.return_value = {
            "access_token": "refreshed-token",
            "expires_in": 86400,
        }

        mock_post.side_effect = [login_response, refresh_response]

        auth = TraceLabAuth(
            base_url="https://test.example.com",
            username="testuser",
            password="testpass",
        )

        # Initial login
        auth.get_token()
        assert auth._token.access_token == "original-token"
        assert auth._refresh_count == 0

        # Refresh
        auth.refresh()
        assert auth._token.access_token == "refreshed-token"
        assert auth._refresh_count == 1

    @patch("httpx.post")
    def test_auth_refresh_fallback_to_login(self, mock_post: MagicMock) -> None:
        """Auth should fall back to login on refresh 401."""
        from deepsearch.tracelab.auth import TraceLabAuth

        # First is login, second is refresh failure (401), third is login again
        login_response = MagicMock()
        login_response.status_code = 200
        login_response.json.return_value = {
            "access_token": "original-token",
            "expires_in": 86400,
        }

        refresh_401 = MagicMock()
        refresh_401.status_code = 401

        relogin_response = MagicMock()
        relogin_response.status_code = 200
        relogin_response.json.return_value = {
            "access_token": "new-token-after-relogin",
            "expires_in": 86400,
        }

        mock_post.side_effect = [login_response, refresh_401, relogin_response]

        auth = TraceLabAuth(
            base_url="https://test.example.com",
            username="testuser",
            password="testpass",
        )

        # Initial login
        auth.get_token()
        assert auth._login_count == 1

        # Refresh should fall back to login
        auth.refresh()
        assert auth._login_count == 2  # Did a full login
        assert auth._token.access_token == "new-token-after-relogin"

    def test_auth_invalidate(self) -> None:
        """Auth should invalidate cached token."""
        from deepsearch.tracelab.auth import TraceLabAuth, TokenInfo

        auth = TraceLabAuth(
            base_url="https://test.example.com",
            username="testuser",
            password="testpass",
        )
        auth._token = TokenInfo(
            access_token="cached-token",
            expires_at=time.time() + 3600,
        )

        auth.invalidate()

        assert auth._token is None

    def test_auth_telemetry(self) -> None:
        """Auth should track telemetry."""
        from deepsearch.tracelab.auth import TraceLabAuth, TokenInfo

        auth = TraceLabAuth(
            base_url="https://test.example.com",
            username="testuser",
            password="testpass",
        )
        auth._login_count = 2
        auth._refresh_count = 3
        auth._last_login_at = time.time() - 100
        auth._token = TokenInfo(
            access_token="test",
            expires_at=time.time() + 3600,
        )

        telemetry = auth.get_telemetry()

        assert telemetry["login_count"] == 2
        assert telemetry["refresh_count"] == 3
        assert telemetry["last_login_at"] is not None
        assert telemetry["token_valid"] is True


class TestUploaderWithAuth:
    """Tests for TraceLabUploader with JWT auth."""

    @pytest.fixture
    def sample_protocol(self) -> Dict[str, Any]:
        """Create a valid protocol dict."""
        return {
            "mission_id": "TEST.0.1",
            "version": "1.0.0",
            "title": "Test Mission",
            "summary": "Test",
            "status": "complete",
            "research_statement": {
                "topic": "Test",
                "objective": "Test",
                "scope": "Test",
            },
            "key_questions": [],
            "synthesis": {
                "key_insights": ["Test"],
                "recommendations": ["Test"],
                "next_steps": ["Test"],
            },
            "evidence": [
                {
                    "evidence_id": "EV-001",
                    "source": "test",
                    "summary": "test",
                }
            ],
            "quality_checkpoints": [],
            "tags": [],
        }

    def test_uploader_uses_auth(self, sample_protocol: Dict[str, Any], tmp_path: Path) -> None:
        """Uploader should use TraceLabAuth for authentication."""
        from deepsearch.tracelab.uploader import TraceLabUploader
        from deepsearch.tracelab.auth import TraceLabAuth

        auth = TraceLabAuth(
            base_url="https://test.example.com",
            username="testuser",
            password="testpass",
        )
        uploader = TraceLabUploader(
            auth=auth,
            telemetry_path=tmp_path / "telemetry.jsonl",
        )

        assert uploader.auth is auth
        assert uploader._configured is True

    def test_uploader_dry_run_still_works(
        self, sample_protocol: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Dry-run should work without auth."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)

            def model_dump_json(self) -> str:
                return json.dumps(self._data)

        uploader = TraceLabUploader(telemetry_path=tmp_path / "telemetry.jsonl")
        protocol = ProtocolWrapper(sample_protocol)
        result = uploader.upload(protocol, dry_run=True)

        assert result.success is True
        assert result.mission_id == "TEST.0.1"

    @patch("httpx.post")
    @patch("httpx.Client")
    def test_uploader_retries_on_401(
        self,
        mock_client_class: MagicMock,
        mock_auth_post: MagicMock,
        sample_protocol: Dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Uploader should retry once on 401 error."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        # Auth responses
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json.return_value = {
            "access_token": "refreshed-token",
            "expires_in": 86400,
        }
        mock_auth_post.return_value = auth_response

        # First upload returns 401, second succeeds
        first_response = MagicMock()
        first_response.status_code = 401
        first_response.json.return_value = {"detail": "Unauthorized"}

        second_response = MagicMock()
        second_response.status_code = 201
        second_response.json.return_value = {
            "mission_uuid": "test-uuid",
            "mission_id": "TEST.0.1",
            "project_id": "proj-123",
            "quality_gates_passed": True,
            "auto_linking": {},
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = [first_response, second_response]
        mock_client_class.return_value = mock_client

        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)

            def model_dump_json(self) -> str:
                return json.dumps(self._data)

        uploader = TraceLabUploader(
            base_url="https://test.example.com",
            username="testuser",
            password="testpass",
            telemetry_path=tmp_path / "telemetry.jsonl",
        )
        protocol = ProtocolWrapper(sample_protocol)
        result = uploader.upload(protocol)

        assert result.success is True
        assert result.mission_uuid == "test-uuid"
        # Should have called post twice (first 401, then retry)
        assert mock_client.post.call_count == 2

    def test_uploader_missing_credentials_fails_gracefully(
        self, sample_protocol: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Upload without credentials should fail with clear error."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)

            def model_dump_json(self) -> str:
                return json.dumps(self._data)

        # Create uploader inside cleared env so _configured=False
        with patch.dict(os.environ, {}, clear=True):
            uploader = TraceLabUploader(telemetry_path=tmp_path / "telemetry.jsonl")
            protocol = ProtocolWrapper(sample_protocol)
            result = uploader.upload(protocol)

        assert result.success is False
        assert any("TRACELAB" in e for e in result.errors)

    def test_upload_logs_auth_telemetry(
        self, sample_protocol: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Upload should include auth telemetry in logs."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)

            def model_dump_json(self) -> str:
                return json.dumps(self._data)

        telemetry_path = tmp_path / "telemetry.jsonl"
        uploader = TraceLabUploader(telemetry_path=telemetry_path)
        protocol = ProtocolWrapper(sample_protocol)

        # Dry run
        uploader.upload(protocol, dry_run=True)

        assert telemetry_path.exists()
        content = telemetry_path.read_text()
        log_entry = json.loads(content.strip())

        assert log_entry["mission_id"] == "TEST.0.1"
        assert log_entry["action"] == "dry_run"


class TestAutoUploadIntegration:
    """Tests for auto-upload in runner."""

    @pytest.fixture
    def mock_state(self) -> Dict[str, Any]:
        """Create a mock agent state."""
        return {
            "mission_id": "AUTO.0.1",
            "mission_context": {"title": "Auto Upload Test"},
            "sources_found": [
                {"url": "https://example.com", "title": "Test Source", "snippet": "Test"},
            ],
            "findings": ["Finding 1", "Finding 2"],
            "reflection_logs": [],
            "report_content": "# Test Report\n\nTest content.",
            "research_loop_count": 2,
        }

    def test_maybe_upload_skips_on_failing_gates(self, mock_state: Dict[str, Any]) -> None:
        """Auto-upload should skip when quality gates fail."""
        import asyncio
        from deepsearch.runner import _maybe_upload_to_tracelab
        from deepsearch.validation import QualityThresholds

        # Create a mock validation result with quality_scores dict
        validation_result = MagicMock()
        validation_result.quality_scores = {
            "faithfulness": 0.5,  # Failing
            "relevance": 0.5,  # Failing
            "citation_rate": 0.5,  # Failing
        }
        validation_result.thresholds = QualityThresholds()

        result, error = asyncio.run(
            _maybe_upload_to_tracelab(mock_state, "AUTO.0.1", validation_result)
        )

        assert result is not None
        assert result["skipped"] is True
        assert result["reason"] == "quality_gates_failed"
        assert len(result["failing_gates"]) == 3
        assert error is None

    @patch("deepsearch.report.protocol.generate_mission_protocol")
    @patch("deepsearch.tracelab.uploader.upload_protocol")
    def test_maybe_upload_succeeds_on_passing_gates(
        self,
        mock_upload: MagicMock,
        mock_generate: MagicMock,
        mock_state: Dict[str, Any],
    ) -> None:
        """Auto-upload should proceed when quality gates pass."""
        import asyncio
        from deepsearch.runner import _maybe_upload_to_tracelab
        from deepsearch.validation import QualityThresholds
        from deepsearch.tracelab.uploader import UploadResult

        mock_generate.return_value = MagicMock(spec=MissionProtocolComplete, mission_id="AUTO.0.1")
        mock_upload.return_value = UploadResult(
            success=True,
            mission_uuid="test-uuid",
            mission_id="AUTO.0.1",
            quality_gates_passed=True,
        )

        # Create mock validation result with passing scores
        validation_result = MagicMock()
        validation_result.quality_scores = {
            "faithfulness": 0.95,
            "relevance": 0.85,
            "citation_rate": 0.85,
        }
        validation_result.thresholds = QualityThresholds()

        result, error = asyncio.run(
            _maybe_upload_to_tracelab(mock_state, "AUTO.0.1", validation_result)
        )

        assert result is not None
        assert result["success"] is True
        assert result["mission_uuid"] == "test-uuid"
        assert error is None

    @patch("deepsearch.report.protocol.generate_mission_protocol")
    def test_maybe_upload_handles_exception(
        self,
        mock_generate: MagicMock,
        mock_state: Dict[str, Any],
    ) -> None:
        """Auto-upload should handle exceptions gracefully."""
        import asyncio
        from deepsearch.runner import _maybe_upload_to_tracelab

        mock_generate.side_effect = Exception("Protocol generation failed")

        result, error = asyncio.run(
            _maybe_upload_to_tracelab(mock_state, "AUTO.0.1", None)
        )

        assert result is None
        assert error is not None
        assert "Protocol generation failed" in error

    @patch("deepsearch.runner._save_failed_submission")
    @patch("deepsearch.report.protocol.generate_mission_protocol")
    @patch("deepsearch.tracelab.uploader.upload_protocol")
    def test_maybe_upload_saves_failed_submission(
        self,
        mock_upload: MagicMock,
        mock_generate: MagicMock,
        mock_save: MagicMock,
        mock_state: Dict[str, Any],
    ) -> None:
        """Auto-upload should save failed submissions."""
        import asyncio
        from deepsearch.runner import _maybe_upload_to_tracelab
        from deepsearch.tracelab.uploader import UploadResult

        mock_protocol = MagicMock(spec=MissionProtocolComplete, mission_id="AUTO.0.1")
        mock_generate.return_value = mock_protocol
        mock_upload.return_value = UploadResult(
            success=False,
            mission_id="AUTO.0.1",
            errors=["Network error"],
        )

        result, error = asyncio.run(
            _maybe_upload_to_tracelab(mock_state, "AUTO.0.1", None)
        )

        assert result is not None
        assert result["success"] is False
        mock_save.assert_called_once()

    def test_cli_auto_upload_flag(self) -> None:
        """CLI should have --auto-upload flag."""
        from deepsearch.cli import build_parser

        parser = build_parser()

        # Parse with --auto-upload
        args = parser.parse_args(["test.yaml", "--auto-upload"])
        assert args.auto_upload is True

        # Parse with --no-auto-upload
        args = parser.parse_args(["test.yaml", "--no-auto-upload"])
        assert args.auto_upload is False

        # Parse without flag (default False)
        args = parser.parse_args(["test.yaml"])
        assert args.auto_upload is False


class TestDefaultAuth:
    """Tests for module-level auth singleton."""

    def test_get_default_auth_returns_singleton(self) -> None:
        """get_default_auth should return same instance."""
        from deepsearch.tracelab.auth import get_default_auth, reset_default_auth

        reset_default_auth()

        auth1 = get_default_auth()
        auth2 = get_default_auth()

        assert auth1 is auth2

    def test_reset_default_auth_clears_singleton(self) -> None:
        """reset_default_auth should clear singleton."""
        from deepsearch.tracelab.auth import get_default_auth, reset_default_auth

        auth1 = get_default_auth()
        reset_default_auth()
        auth2 = get_default_auth()

        assert auth1 is not auth2
