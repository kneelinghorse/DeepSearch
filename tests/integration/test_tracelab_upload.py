"""Integration tests for DeepSearch's TraceLab upload path.

Tests cover the validation quality gate and the protocol uploader in
``deepsearch.tracelab``. Tests for the separate CMOS-side upload client
(``cmos.tracelab.uploader``) live with that code in
``cmos/tests/integration/test_cmos_tracelab_upload.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepsearch.validation.gate import GateResult, evaluate
from deepsearch.validation.hybrid_evaluator import QualityScores, QualityThresholds


class TestQualityGate:
    """Tests for the quality gate evaluation."""

    def test_gate_passes_when_all_thresholds_met(self) -> None:
        """Gate should pass when all scores meet thresholds."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.95,
                "relevance": 0.85,
                "citation_rate": 0.82,
            }
        }
        result = evaluate(metrics)

        assert result.passed is True
        assert result.score_checks["faithfulness"] is True
        assert result.score_checks["relevance"] is True
        assert result.score_checks["citation_rate"] is True
        assert len(result.failure_reasons) == 0

    def test_gate_fails_on_low_faithfulness(self) -> None:
        """Gate should fail when faithfulness is below threshold."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.85,  # Below 0.9 threshold
                "relevance": 0.85,
                "citation_rate": 0.82,
            }
        }
        result = evaluate(metrics)

        assert result.passed is False
        assert result.score_checks["faithfulness"] is False
        assert result.score_checks["relevance"] is True
        assert result.score_checks["citation_rate"] is True
        assert any("Faithfulness" in r for r in result.failure_reasons)

    def test_gate_fails_on_low_relevance(self) -> None:
        """Gate should fail when relevance is below threshold."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.95,
                "relevance": 0.70,  # Below 0.8 threshold
                "citation_rate": 0.82,
            }
        }
        result = evaluate(metrics)

        assert result.passed is False
        assert result.score_checks["faithfulness"] is True
        assert result.score_checks["relevance"] is False
        assert result.score_checks["citation_rate"] is True
        assert any("Relevance" in r for r in result.failure_reasons)

    def test_gate_fails_on_low_citation_rate(self) -> None:
        """Gate should fail when citation rate is below threshold."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.95,
                "relevance": 0.85,
                "citation_rate": 0.50,  # Below 0.8 threshold
            }
        }
        result = evaluate(metrics)

        assert result.passed is False
        assert result.score_checks["faithfulness"] is True
        assert result.score_checks["relevance"] is True
        assert result.score_checks["citation_rate"] is False
        assert any("Citation rate" in r for r in result.failure_reasons)

    def test_gate_fails_on_multiple_issues(self) -> None:
        """Gate should report all failing thresholds."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.50,
                "relevance": 0.50,
                "citation_rate": 0.50,
            }
        }
        result = evaluate(metrics)

        assert result.passed is False
        assert all(v is False for v in result.score_checks.values())
        assert len(result.failure_reasons) == 3

    def test_gate_warns_on_zero_scores(self) -> None:
        """Gate should warn when scores are zero (likely missing data)."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "citation_rate": 0.0,
            }
        }
        result = evaluate(metrics)

        assert len(result.warnings) >= 3

    def test_gate_handles_flat_metrics_structure(self) -> None:
        """Gate should handle metrics at root level (not nested)."""
        metrics = {
            "faithfulness": 0.95,
            "relevance": 0.85,
            "citation_rate": 0.82,
        }
        result = evaluate(metrics)

        assert result.passed is True

    def test_gate_with_custom_thresholds(self) -> None:
        """Gate should respect custom thresholds."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.75,
                "relevance": 0.65,
                "citation_rate": 0.55,
            }
        }
        custom_thresholds = QualityThresholds(
            faithfulness=0.7,
            relevance=0.6,
            citation_rate=0.5,
        )
        result = evaluate(metrics, thresholds=custom_thresholds)

        assert result.passed is True

    def test_gate_requires_telemetry_when_specified(self) -> None:
        """Gate should fail when telemetry is required but missing."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.95,
                "relevance": 0.85,
                "citation_rate": 0.82,
            }
        }
        result = evaluate(metrics, require_telemetry=True)

        assert result.passed is False
        assert any("Telemetry" in r for r in result.failure_reasons)

    def test_gate_passes_with_valid_telemetry(self) -> None:
        """Gate should pass when valid telemetry is provided."""
        metrics = {
            "quality_scores": {
                "faithfulness": 0.95,
                "relevance": 0.85,
                "citation_rate": 0.82,
            }
        }
        telemetry = {
            "mission_id": "DRM.0.1",
            "loops_executed": 3,
            "sources_count": 5,
        }
        result = evaluate(metrics, telemetry=telemetry, require_telemetry=True)

        assert result.passed is True


# ===== Protocol Uploader Tests (B4.8) =====

class TestProtocolUploader:
    """Tests for the new protocol-based TraceLab uploader (deepsearch.tracelab)."""

    @pytest.fixture
    def sample_protocol(self) -> Dict[str, Any]:
        """Create a valid MissionProtocolComplete-compatible dict."""
        return {
            "mission_id": "TEST.0.1",
            "version": "1.0.0",
            "title": "Test Research Mission",
            "summary": "Autonomous research mission for testing",
            "status": "complete",
            "research_statement": {
                "topic": "Protocol testing",
                "objective": "Validate upload functionality",
                "scope": "Unit tests",
            },
            "key_questions": [
                {
                    "question_id": "Q1",
                    "question": "Does upload work?",
                    "status": "answered",
                    "answer": "Yes, it works!",
                }
            ],
            "synthesis": {
                "key_insights": ["Upload works correctly"],
                "recommendations": ["Continue testing"],
                "next_steps": ["Deploy to production"],
            },
            "evidence": [
                {
                    "evidence_id": "EV-001",
                    "source": "Unit tests",
                    "summary": "Tests passed",
                    "source_type": "test",
                }
            ],
            "quality_checkpoints": [
                {"gate": "research_statement", "status": "pass"},
                {"gate": "evidence_links", "status": "pass"},
                {"gate": "synthesis_quality", "status": "pass"},
                {"gate": "traceability", "status": "pass"},
                {"gate": "contradictions_resolved", "status": "pass"},
            ],
            "tags": ["test", "deepsearch"],
        }

    @pytest.fixture
    def protocol_file(self, tmp_path: Path, sample_protocol: Dict[str, Any]) -> Path:
        """Create a protocol JSON file."""
        path = tmp_path / "protocol.json"
        path.write_text(json.dumps(sample_protocol), encoding="utf-8")
        return path

    def test_protocol_mock_upload(self, sample_protocol: Dict[str, Any], tmp_path: Path) -> None:
        """Mock mode should return simulated success."""
        from deepsearch.tracelab.uploader import upload_protocol

        # Create a simple wrapper
        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)

            def model_dump_json(self) -> str:
                return json.dumps(self._data)

        protocol = ProtocolWrapper(sample_protocol)
        result = upload_protocol(protocol, mock=True)

        assert result.success is True
        assert result.mission_id == "TEST.0.1"
        assert result.mission_uuid == "mock-uuid-12345"
        assert "MOCK" in result.errors[0]

    def test_protocol_dry_run(self, sample_protocol: Dict[str, Any], tmp_path: Path) -> None:
        """Dry-run should validate without uploading."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)

        uploader = TraceLabUploader(telemetry_path=tmp_path / "telemetry.jsonl")
        protocol = ProtocolWrapper(sample_protocol)
        result = uploader.upload(protocol, dry_run=True)

        assert result.success is True
        assert result.mission_id == "TEST.0.1"
        assert "DRY RUN" in result.errors[0]
        assert result.quality_gates_passed is True

    def test_protocol_dry_run_validation_failure(self, tmp_path: Path) -> None:
        """Dry-run should catch missing required fields."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        # Protocol missing required fields
        bad_protocol = {
            "mission_id": "BAD.0.1",
            # Missing: research_statement, evidence, synthesis
        }

        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)

        uploader = TraceLabUploader(telemetry_path=tmp_path / "telemetry.jsonl")
        protocol = ProtocolWrapper(bad_protocol)
        result = uploader.upload(protocol, dry_run=True)

        assert result.success is False
        assert any("research_statement" in e for e in result.errors)
        assert any("evidence" in e for e in result.errors)
        assert any("synthesis" in e for e in result.errors)

    def test_protocol_telemetry_logging(
        self, sample_protocol: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Upload should log to telemetry file."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        telemetry_path = tmp_path / "telemetry.jsonl"

        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)

        uploader = TraceLabUploader(telemetry_path=telemetry_path)
        protocol = ProtocolWrapper(sample_protocol)
        result = uploader.upload(protocol, dry_run=True)

        assert telemetry_path.exists()
        content = telemetry_path.read_text(encoding="utf-8")
        log_entry = json.loads(content.strip())

        assert log_entry["mission_id"] == "TEST.0.1"
        assert log_entry["action"] == "dry_run"
        assert log_entry["success"] is True

    def test_protocol_failing_gates(self, tmp_path: Path) -> None:
        """Dry-run should detect failing quality gates."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        protocol_with_failing_gate = {
            "mission_id": "FAIL.0.1",
            "research_statement": {"topic": "Test", "objective": "Test", "scope": "Test"},
            "evidence": [{"evidence_id": "EV-001", "source": "test", "summary": "test"}],
            "synthesis": {
                "key_insights": ["insight"],
                "recommendations": ["rec"],
                "next_steps": ["step"],
            },
            "quality_checkpoints": [
                {"gate": "research_statement", "status": "pass"},
                {"gate": "evidence_links", "status": "fail"},  # Failing gate
                {"gate": "synthesis_quality", "status": "pass"},
                {"gate": "traceability", "status": "pass"},
                {"gate": "contradictions_resolved", "status": "pass"},
            ],
        }

        class ProtocolWrapper:
            def __init__(self, data: dict):
                self._data = data
                for key, value in data.items():
                    setattr(self, key, value)
                # Handle quality_checkpoints specially
                self.quality_checkpoints = [
                    type("CP", (), cp)() for cp in data.get("quality_checkpoints", [])
                ]

        uploader = TraceLabUploader(telemetry_path=tmp_path / "telemetry.jsonl")
        protocol = ProtocolWrapper(protocol_with_failing_gate)
        result = uploader.upload(protocol, dry_run=True)

        assert result.success is False
        assert any("evidence_links" in e for e in result.errors)

    def test_load_protocol_from_file(self, protocol_file: Path) -> None:
        """Should load protocol from JSON file."""
        from deepsearch.tracelab.uploader import load_protocol_from_file

        data = load_protocol_from_file(protocol_file)

        assert data["mission_id"] == "TEST.0.1"
        assert data["version"] == "1.0.0"
        assert len(data["evidence"]) >= 1

    def test_load_protocol_from_missing_file(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        from deepsearch.tracelab.uploader import load_protocol_from_file

        with pytest.raises(FileNotFoundError):
            load_protocol_from_file(tmp_path / "nonexistent.json")

    def test_upload_without_credentials_fails(
        self, sample_protocol: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Upload without credentials should fail gracefully."""
        from deepsearch.tracelab.uploader import TraceLabUploader
        import os

        # Ensure env vars are not set
        old_url = os.environ.pop("TRACELAB_BASE_URL", None)
        old_key = os.environ.pop("TRACELAB_API_KEY", None)

        try:
            class ProtocolWrapper:
                def __init__(self, data: dict):
                    self._data = data
                    for key, value in data.items():
                        setattr(self, key, value)

            uploader = TraceLabUploader(telemetry_path=tmp_path / "telemetry.jsonl")
            protocol = ProtocolWrapper(sample_protocol)
            result = uploader.upload(protocol)  # Real upload without credentials

            assert result.success is False
            assert any("TRACELAB_BASE_URL" in e or "TRACELAB_API_KEY" in e for e in result.errors)
        finally:
            # Restore env vars
            if old_url:
                os.environ["TRACELAB_BASE_URL"] = old_url
            if old_key:
                os.environ["TRACELAB_API_KEY"] = old_key


class TestProtocolUploadErrorParsing:
    """Tests for error response parsing."""

    def test_parse_422_errors(self) -> None:
        """Should parse FastAPI 422 validation errors."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        uploader = TraceLabUploader()

        data_422 = {
            "detail": [
                {"loc": ["body", "mission", "research_statement"], "msg": "field required", "type": "missing"},
                {"loc": ["body", "mission", "evidence"], "msg": "field required", "type": "missing"},
            ]
        }

        errors = uploader._parse_422_errors(data_422)

        assert len(errors) == 2
        assert "mission.research_statement" in errors[0]
        assert "mission.evidence" in errors[1]

    def test_parse_400_errors(self) -> None:
        """Should parse quality gate failure errors."""
        from deepsearch.tracelab.uploader import TraceLabUploader

        uploader = TraceLabUploader()

        data_400 = {
            "success": False,
            "error": {
                "code": "QUALITY_GATE_FAILED",
                "message": "Quality check failed",
                "details": {
                    "failing_gates": ["evidence_links", "synthesis_quality"],
                },
            },
        }

        errors = uploader._parse_400_errors(data_400)

        assert "QUALITY_GATE_FAILED" in errors[0]
        assert any("evidence_links" in e for e in errors)
        assert any("synthesis_quality" in e for e in errors)
        # Check for hints
        assert any("Hint" in e for e in errors)
