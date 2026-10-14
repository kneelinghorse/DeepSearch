"""TraceLab protocol uploader for MissionProtocolComplete payloads.

This module integrates with TraceLab's /api/v1/deepsearch/ingest endpoint
to upload completed mission protocols. Features:
- JWT authentication with automatic refresh
- Mock mode for testing without network
- Dry-run mode for validation without upload
- Error parsing for 422 and 400 responses with MissionActionableError (S29)
- Automatic retry on 401 with re-authentication
- Retry on transient EMBEDDING_FAILED / QDRANT_ERROR (S29)
- Human-readable mission IDs for GET/submit endpoints (S29)
- Create-and-submit convenience endpoint (S29)
- Telemetry logging to JSONL
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deepsearch.data_paths import data_path

from .auth import TraceLabAuth, TraceLabAuthError
from .schemas import MissionActionableError

logger = logging.getLogger(__name__)

# TraceLab S29 retryable error codes — transient infrastructure issues
RETRYABLE_ERROR_CODES = {"EMBEDDING_FAILED", "QDRANT_ERROR"}
MAX_RETRYABLE_ATTEMPTS = 2


@dataclass
class UploadResult:
    """Result of a TraceLab upload attempt."""

    success: bool
    mission_uuid: Optional[str] = None
    mission_id: Optional[str] = None
    project_id: Optional[str] = None
    quality_gates_passed: bool = False
    auto_linking: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "mission_uuid": self.mission_uuid,
            "mission_id": self.mission_id,
            "project_id": self.project_id,
            "quality_gates_passed": self.quality_gates_passed,
            "auto_linking": self.auto_linking,
            "errors": self.errors,
        }


class TraceLabUploader:
    """Client for uploading protocols to TraceLab.

    Supports the /api/v1/deepsearch/ingest endpoint with:
    - JWT authentication with automatic refresh
    - Project auto-creation
    - Evidence auto-linking
    - Quality gate validation
    - Automatic retry on 401 errors

    Args:
        base_url: TraceLab API base URL (default: from TRACELAB_BASE_URL env var)
        username: Username for JWT auth (default: from TRACELAB_USERNAME env var)
        password: Password for JWT auth (default: from TRACELAB_PASSWORD env var)
        timeout: HTTP timeout in seconds (default: 30.0)
        telemetry_path: Path for JSONL telemetry logs
        auth: Optional pre-configured TraceLabAuth instance
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
        telemetry_path: Optional[Path] = None,
        auth: Optional[TraceLabAuth] = None,
    ):
        resolved_base_url = base_url or os.getenv("TRACELAB_BASE_URL", "")
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout
        if telemetry_path is not None:
            self.telemetry_path = Path(telemetry_path)
        else:
            telemetry_dir = Path(os.getenv("DEEPSEARCH_TELEMETRY_DIR") or data_path("telemetry"))
            self.telemetry_path = telemetry_dir.expanduser() / "tracelab-uploads.jsonl"

        # Use provided auth instance or create new one
        if auth is not None:
            self.auth = auth
            self.base_url = auth.base_url
        else:
            self.auth = TraceLabAuth(
                base_url=self.base_url,
                username=username,
                password=password,
                timeout=timeout,
            )

        # Validate configuration (only when not in mock/dry-run mode)
        self._configured = self.auth.is_configured

    def upload(
        self,
        protocol: Any,
        *,
        project_id: Optional[str] = None,
        auto_create_project: bool = True,
        project_name: str = "DeepSearch Research",
        similarity_threshold: float = 0.78,
        dry_run: bool = False,
    ) -> UploadResult:
        """Upload a mission protocol to TraceLab.

        Args:
            protocol: MissionProtocolComplete instance
            project_id: Existing project UUID (optional)
            auto_create_project: Create project if not specified
            project_name: Name for auto-created project
            similarity_threshold: Evidence auto-linking threshold (B13.2: 0.75→0.78 for S29)
            dry_run: Validate only, don't persist

        Returns:
            UploadResult with success status and details
        """
        # Extract mission_id for telemetry
        mission_id = getattr(protocol, "mission_id", None) or "unknown"

        if dry_run:
            result = self._validate_locally(protocol)
            self._log_telemetry(mission_id, "dry_run", result)
            return result

        # Check configuration
        if not self._configured:
            try:
                # This will raise TraceLabAuthError with details
                self.auth.get_token()
            except TraceLabAuthError as exc:
                result = UploadResult(
                    success=False, mission_id=mission_id, errors=[str(exc)]
                )
                self._log_telemetry(mission_id, "upload", result)
                return result

        # Serialize protocol
        if hasattr(protocol, "model_dump_json"):
            mission_data = json.loads(protocol.model_dump_json())
        elif hasattr(protocol, "to_dict"):
            mission_data = protocol.to_dict()
        else:
            mission_data = dict(protocol)

        # Build request payload
        payload = {
            "project_id": project_id,
            "auto_create_project": auto_create_project,
            "project_name": project_name,
            "similarity_threshold": similarity_threshold,
            "mission": mission_data,
        }

        try:
            import httpx
        except ImportError:
            result = UploadResult(
                success=False,
                mission_id=mission_id,
                errors=["httpx not installed - run: pip install httpx"],
            )
            self._log_telemetry(mission_id, "upload", result)
            return result

        try:
            # Get auth headers (performs login if needed)
            headers = {
                **self.auth.get_headers(),
                "Content-Type": "application/json",
            }

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/v1/deepsearch/ingest",
                    json=payload,
                    headers=headers,
                )

                # Retry once on 401 with re-authentication
                if response.status_code == 401:
                    logger.info("Received 401, refreshing auth and retrying")
                    self.auth.refresh()
                    headers = {
                        **self.auth.get_headers(),
                        "Content-Type": "application/json",
                    }
                    response = client.post(
                        f"{self.base_url}/api/v1/deepsearch/ingest",
                        json=payload,
                        headers=headers,
                    )

                # B13.2: Retry on transient S29 error codes (EMBEDDING_FAILED, QDRANT_ERROR)
                result = self._parse_response(response, mission_id)
                if not result.success and self._is_retryable_error(result):
                    for attempt in range(1, MAX_RETRYABLE_ATTEMPTS + 1):
                        logger.info(
                            "Retryable error detected (%s), attempt %d/%d",
                            self._get_error_code(result),
                            attempt,
                            MAX_RETRYABLE_ATTEMPTS,
                        )
                        response = client.post(
                            f"{self.base_url}/api/v1/deepsearch/ingest",
                            json=payload,
                            headers=headers,
                        )
                        result = self._parse_response(response, mission_id)
                        if result.success or not self._is_retryable_error(result):
                            break

            self._log_telemetry(mission_id, "upload", result, auth_telemetry=self.auth.get_telemetry())
            return result

        except TraceLabAuthError as exc:
            result = UploadResult(
                success=False,
                mission_id=mission_id,
                errors=[f"Authentication failed: {exc}"],
            )
            self._log_telemetry(mission_id, "upload", result)
            return result

        except Exception as exc:
            result = UploadResult(
                success=False,
                mission_id=mission_id,
                errors=[f"Network error: {exc}"],
            )
            self._log_telemetry(mission_id, "upload", result)
            return result

    def create_and_submit(
        self,
        protocol: Any,
        *,
        project_name: str = "DeepSearch Research",
        similarity_threshold: float = 0.78,
    ) -> UploadResult:
        """Create a project and submit a protocol in a single request.

        B13.3: TraceLab S29 convenience endpoint that combines project creation
        and protocol submission. Uses human-readable mission IDs directly
        (no UUID lookup needed).

        Args:
            protocol: MissionProtocolComplete instance
            project_name: Name for the project
            similarity_threshold: Evidence auto-linking threshold

        Returns:
            UploadResult with success status and details
        """
        return self.upload(
            protocol,
            auto_create_project=True,
            project_name=project_name,
            similarity_threshold=similarity_threshold,
        )

    def get_mission(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a mission by its human-readable ID.

        B13.3: TraceLab S29 supports human-readable mission IDs directly
        in GET endpoints, eliminating the need for UUID lookup.

        Args:
            mission_id: Human-readable mission ID (e.g., "DRM.0.5")

        Returns:
            Mission data dictionary, or None if not found or on error.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed - cannot fetch mission")
            return None

        if not self._configured:
            return None

        try:
            headers = self.auth.get_headers()
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/api/v1/missions/{mission_id}",
                    headers=headers,
                )
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception as exc:
            logger.warning("Failed to fetch mission %s: %s", mission_id, exc)
            return None

    def _validate_locally(self, protocol: Any) -> UploadResult:
        """Validate protocol without uploading."""
        mission_id = getattr(protocol, "mission_id", None) or "unknown"

        try:
            # Check required fields
            errors = []

            if not hasattr(protocol, "mission_id") or not protocol.mission_id:
                errors.append("mission_id is required")
            if not hasattr(protocol, "research_statement") or not protocol.research_statement:
                errors.append("research_statement is required")
            if not hasattr(protocol, "evidence") or not protocol.evidence:
                errors.append("evidence is required (at least 1 item)")
            if not hasattr(protocol, "synthesis") or not protocol.synthesis:
                errors.append("synthesis is required")

            # Check quality gates if present. S84 m04: 'skip' means the gate
            # was NOT evaluated (no validator exists) — consistent with
            # all_gates_pass(), only a hard 'fail' fails local validation.
            if hasattr(protocol, "quality_checkpoints") and protocol.quality_checkpoints:
                failing_gates = [
                    cp.gate
                    for cp in protocol.quality_checkpoints
                    if hasattr(cp, "status") and cp.status == "fail"
                ]
                if failing_gates:
                    errors.append(f"Failing gates: {', '.join(failing_gates)}")

            if errors:
                return UploadResult(
                    success=False,
                    mission_id=mission_id,
                    errors=errors,
                )

            return UploadResult(
                success=True,
                mission_id=mission_id,
                quality_gates_passed=True,
                errors=["DRY RUN: Validation passed, not uploaded"],
            )

        except Exception as exc:
            return UploadResult(
                success=False,
                mission_id=mission_id,
                errors=[f"Validation failed: {exc}"],
            )

    def _parse_response(self, response: Any, mission_id: str) -> UploadResult:
        """Parse TraceLab API response."""
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            return UploadResult(
                success=False,
                mission_id=mission_id,
                errors=[f"Invalid JSON response: {response.text[:200]}"],
            )

        if response.status_code == 201:
            return UploadResult(
                success=True,
                mission_uuid=data.get("mission_uuid"),
                mission_id=data.get("mission_id") or mission_id,
                project_id=data.get("project_id"),
                quality_gates_passed=data.get("quality_gates_passed", True),
                auto_linking=data.get("auto_linking", {}),
                raw_response=data,
            )

        elif response.status_code == 422:
            errors = self._parse_422_errors(data)
            return UploadResult(
                success=False,
                mission_id=mission_id,
                errors=errors,
                raw_response=data,
            )

        elif response.status_code == 400:
            errors = self._parse_400_errors(data)
            return UploadResult(
                success=False,
                mission_id=mission_id,
                errors=errors,
                raw_response=data,
            )

        elif response.status_code == 401:
            return UploadResult(
                success=False,
                mission_id=mission_id,
                errors=["Authentication failed: Invalid or expired API key"],
                raw_response=data,
            )

        elif response.status_code == 404:
            return UploadResult(
                success=False,
                mission_id=mission_id,
                errors=["Project not found: Specify a valid project_id or enable auto_create_project"],
                raw_response=data,
            )

        else:
            return UploadResult(
                success=False,
                mission_id=mission_id,
                errors=[f"HTTP {response.status_code}: {response.text[:200]}"],
                raw_response=data if isinstance(data, dict) else None,
            )

    def _parse_422_errors(self, data: Dict[str, Any]) -> List[str]:
        """Parse FastAPI 422 validation errors into actionable messages."""
        errors = []
        for detail in data.get("detail", []):
            # Extract field location path
            loc = detail.get("loc", [])
            if loc:
                # Skip "body" prefix if present
                path_parts = [str(x) for x in loc if x != "body"]
                path = ".".join(path_parts)
            else:
                path = "unknown"

            msg = detail.get("msg", "Unknown error")
            error_type = detail.get("type", "")

            # Format actionable message
            if error_type == "missing":
                errors.append(f"{path}: Required field is missing")
            elif error_type == "string_too_short":
                errors.append(f"{path}: Value is too short - {msg}")
            elif error_type == "string_type":
                errors.append(f"{path}: Expected string, got different type")
            else:
                errors.append(f"{path}: {msg}")

        return errors or ["Validation error (no details provided)"]

    def _parse_400_errors(self, data: Dict[str, Any]) -> List[str]:
        """Parse quality gate failure errors into actionable messages.

        B13.3: Now parses MissionActionableError structure from S29 which
        includes a 'suggestion' field for recovery guidance.
        """
        errors = []

        # B13.3: Try parsing as MissionActionableError first (S29 format)
        error_obj = data.get("error", {})
        try:
            actionable = MissionActionableError(**error_obj)
            code = actionable.error_code or "QUALITY_GATE_FAILED"
            errors.append(f"{code}: {actionable.message}")
            if actionable.suggestion:
                errors.append(f"  Suggestion: {actionable.suggestion}")
            if actionable.current_status:
                errors.append(f"  Current status: {actionable.current_status}")
        except Exception:
            # Fallback to legacy parsing
            code = error_obj.get("code", "QUALITY_GATE_FAILED")
            message = error_obj.get("message", "Quality gate check failed")
            errors.append(f"{code}: {message}")

        # B13.2: Handle S29 transient infrastructure errors
        if code == "EMBEDDING_FAILED":
            errors.append("  Hint: Transient embedding service error — will retry automatically")
        elif code == "QDRANT_ERROR":
            errors.append("  Hint: Transient vector store error — will retry automatically")

        # List individual failing gates
        details = error_obj.get("details", {})
        failing_gates = details.get("failing_gates", [])
        for gate in failing_gates:
            errors.append(f"  - Gate failed: {gate}")

        # Add remediation hints
        if "research_statement" in failing_gates:
            errors.append("  Hint: Ensure research_statement has topic, objective, and scope")
        if "evidence_links" in failing_gates:
            errors.append("  Hint: At least 1 Evidence item is required")
        if "synthesis_quality" in failing_gates:
            errors.append("  Hint: Synthesis needs key_insights, recommendations, next_steps")

        return errors

    @staticmethod
    def _is_retryable_error(result: UploadResult) -> bool:
        """Check if an upload failure contains a retryable S29 error code."""
        if result.success:
            return False
        for err in result.errors:
            for code in RETRYABLE_ERROR_CODES:
                if code in err:
                    return True
        return False

    @staticmethod
    def _get_error_code(result: UploadResult) -> str:
        """Extract the error code from a failed upload result for logging."""
        for err in result.errors:
            for code in RETRYABLE_ERROR_CODES:
                if code in err:
                    return code
        return "UNKNOWN"

    def _log_telemetry(
        self,
        mission_id: str,
        action: str,
        result: UploadResult,
        auth_telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log upload attempt to telemetry JSONL file."""
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "mission_id": mission_id,
                "action": action,
                "success": result.success,
                "quality_gates_passed": result.quality_gates_passed,
                "errors": result.errors,
                "auto_linking": result.auto_linking,
            }

            # Include auth telemetry if provided
            if auth_telemetry:
                entry["auth"] = auth_telemetry

            with self.telemetry_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.warning("Failed to write telemetry: %s", e)


def upload_protocol(
    protocol: Any,
    **options: Any,
) -> UploadResult:
    """Convenience function for uploading protocols.

    Args:
        protocol: MissionProtocolComplete instance
        **options: Options passed to TraceLabUploader.upload()
            - project_id: Existing project UUID
            - auto_create_project: Create project if not specified (default: True)
            - project_name: Name for auto-created project
            - similarity_threshold: Evidence auto-linking threshold (default: 0.78)
            - dry_run: Validate only, don't persist
            - mock: Return simulated success without network call

    Returns:
        UploadResult with success status and details
    """
    mock = options.pop("mock", False)
    mission_id = getattr(protocol, "mission_id", None) or "unknown"

    if mock:
        result = UploadResult(
            success=True,
            mission_id=mission_id,
            mission_uuid="mock-uuid-12345",
            quality_gates_passed=True,
            errors=["MOCK: Simulated successful upload"],
        )

        # Still log telemetry for mock uploads
        uploader = TraceLabUploader()
        uploader._log_telemetry(mission_id, "mock", result)

        return result

    uploader = TraceLabUploader()
    return uploader.upload(protocol, **options)


def load_protocol_from_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a protocol JSON file.

    Args:
        path: Path to protocol JSON file

    Returns:
        Protocol dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "TraceLabUploader",
    "UploadResult",
    "upload_protocol",
    "load_protocol_from_file",
]
