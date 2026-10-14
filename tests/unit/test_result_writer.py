"""Unit tests for ResultWriter."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepsearch.agent.outcome import MissionOutcome
from deepsearch.worker.lease import LeaseAttempt
from deepsearch.worker.writer import (
    ExecutionMetadata,
    ResultWriter,
    StaleLeaseError,
    WriterError,
    build_execution_metadata,
)


class TestExecutionMetadata:
    """Tests for ExecutionMetadata dataclass."""

    def test_basic_creation(self):
        """Test creating metadata with required fields."""
        metadata = ExecutionMetadata(
            total_tokens=5000,
            duration_seconds=45.5,
        )
        assert metadata.total_tokens == 5000
        assert metadata.duration_seconds == 45.5
        assert metadata.agent_version == "deepsearch-1.0"
        assert metadata.convergence_reason is None

    def test_full_creation(self):
        """Test creating metadata with all fields."""
        metadata = ExecutionMetadata(
            total_tokens=12000,
            duration_seconds=120.789,
            agent_version="deepsearch-2.0",
            convergence_reason="coverage_threshold_met",
            contract_id="contract-123",
            contract_path="checkpoints/contracts/mission-foo-contract-123.json",
        )
        assert metadata.total_tokens == 12000
        assert metadata.duration_seconds == 120.789
        assert metadata.agent_version == "deepsearch-2.0"
        assert metadata.convergence_reason == "coverage_threshold_met"
        assert metadata.contract_id == "contract-123"

    def test_to_dict_basic(self):
        """Test serialization without optional fields."""
        metadata = ExecutionMetadata(
            total_tokens=3000,
            duration_seconds=30.123456,
        )
        result = metadata.to_dict()

        assert result["total_tokens"] == 3000
        assert result["duration_seconds"] == 30.12  # Rounded to 2 decimals
        assert result["agent_version"] == "deepsearch-1.0"
        assert "convergence_reason" not in result
        # S75 m01: the 3 lying fields are stripped from emission
        assert "loops_executed" not in result
        assert "sources_count" not in result
        assert "quality_gates_passed" not in result

    def test_to_dict_with_convergence(self):
        """Test serialization with convergence reason."""
        metadata = ExecutionMetadata(
            total_tokens=8000,
            duration_seconds=60.0,
            convergence_reason="max_loops_reached",
            contract_id="contract-xyz",
            telemetry_path="checkpoints/telemetry/mission-foo.jsonl",
            runtime_identity={"execution_path": "yaml"},
            config_fingerprint="fp-123",
            route_summary={"next_action": "synthesize"},
            finding_counts={
                "raw_findings": 10,
                "unique_concepts": 6,
                "cluster_count": 6,
                "deduped_findings": 6,
            },
            source_counts={"sources_before_filtering": 20, "sources_presented_to_llm": 8},
            final_outcome={"outcome": "validation_failed"},
        )
        result = metadata.to_dict()

        assert result["convergence_reason"] == "max_loops_reached"
        assert result["contract_id"] == "contract-xyz"
        assert result["telemetry_path"].endswith("mission-foo.jsonl")
        assert result["runtime_identity"]["execution_path"] == "yaml"
        assert result["config_fingerprint"] == "fp-123"
        assert result["route_summary"]["next_action"] == "synthesize"
        assert result["finding_counts"]["unique_concepts"] == 6
        assert result["finding_counts"]["deduped_findings"] == 6
        assert result["source_counts"]["sources_presented_to_llm"] == 8
        assert result["final_outcome"]["outcome"] == "validation_failed"

    def test_to_dict_is_json_serializable(self):
        """Test that to_dict output can be JSON serialized."""
        metadata = ExecutionMetadata(
            total_tokens=5000,
            duration_seconds=45.0,
            convergence_reason="all_objectives_met",
        )
        # Should not raise
        json_str = json.dumps(metadata.to_dict())
        parsed = json.loads(json_str)
        assert parsed["total_tokens"] == 5000
        assert parsed["convergence_reason"] == "all_objectives_met"

    def test_lease_audit_never_serializes_opaque_token(self) -> None:
        """Persisted attempt telemetry identifies ownership without its proof."""
        now = datetime(2026, 7, 18, tzinfo=timezone.utc)
        lease = LeaseAttempt(
            owner="worker-a",
            token="never-persist-this-token",
            leased_at=now,
            heartbeat_at=now,
            expires_at=now,
            attempt_count=2,
        )
        metadata = ExecutionMetadata(total_tokens=1, duration_seconds=1.0)
        metadata.lease_attempt = lease.audit_dict("mission-123")

        serialized = json.dumps(metadata.to_dict())
        assert "never-persist-this-token" not in serialized
        assert metadata.to_dict()["lease_attempt"]["attempt_count"] == 2
        assert len(metadata.to_dict()["lease_attempt"]["token_fingerprint"]) == 12


class TestBuildExecutionMetadata:
    """Tests for build_execution_metadata function."""

    def test_basic_state(self):
        """Test building metadata from basic state.

        S75 m01: ``loops_executed`` / ``sources_count`` were stripped from
        ``ExecutionMetadata`` emission. Only the still-truthful surfaces
        (token usage, runtime, agent_version) are asserted now.
        """
        state = {
            "research_loop_count": 3,
            "telemetry": {"token_usage": {"total_tokens": 5000}},
            "sources_found": ["source1", "source2", "source3"],
        }
        metadata = build_execution_metadata(state, 45.5)

        assert metadata.total_tokens == 5000
        assert metadata.duration_seconds == 45.5

    def test_empty_state(self):
        """Test building metadata from empty state."""
        metadata = build_execution_metadata({}, 10.0)

        assert metadata.total_tokens == 0
        assert metadata.duration_seconds == 10.0

    def test_token_usage_fallback(self):
        """Test extracting tokens from telemetry token_usage dict."""
        state = {
            "research_loop_count": 2,
            "telemetry": {"token_usage": {"total_tokens": 4500}},
            "sources_found": [],
        }
        metadata = build_execution_metadata(state, 30.0)

        assert metadata.total_tokens == 4500

    def test_with_convergence_reason(self):
        """Test extracting termination reason."""
        state = {
            "research_loop_count": 5,
            "telemetry": {"token_usage": {"total_tokens": 10000}},
            "sources_found": ["a", "b"],
            "termination_reason": "all_objectives_stuck",
        }
        metadata = build_execution_metadata(state, 120.0)

        assert metadata.convergence_reason == "all_objectives_stuck"

    def test_non_dict_telemetry_falls_back_cleanly(self):
        """Non-dict telemetry must not crash the builder.

        S75 m01 replacement for the old ``test_sources_not_list``: the
        ``sources_count`` field was stripped from emission, but we still want
        to confirm the builder handles malformed state without raising.
        """
        state = {
            "telemetry": "not a dict",
            "sources_found": "not a list",
        }
        metadata = build_execution_metadata(state, 20.0)
        assert metadata.total_tokens == 0
        assert metadata.duration_seconds == 20.0


class TestResultWriter:
    """Tests for ResultWriter class."""

    @pytest.fixture
    def mock_db_client(self):
        """Create a mock database client."""
        client = MagicMock()
        client.fetchval = AsyncMock()
        return client

    @pytest.fixture
    def writer(self, mock_db_client):
        """Create a ResultWriter with mock client."""
        return ResultWriter(mock_db_client)

    @pytest.fixture
    def fixed_time(self):
        """Fixed timestamp for testing."""
        return datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def lease(self, fixed_time):
        return LeaseAttempt(
            owner="worker-a",
            token="opaque-token-a",
            leased_at=fixed_time,
            heartbeat_at=fixed_time,
            expires_at=fixed_time,
            attempt_count=2,
        )


class TestMarkCompleted(TestResultWriter):
    """Tests for mark_completed method."""

    @pytest.fixture
    def sample_protocol(self):
        """Sample protocol dict."""
        return {
            "mission_id": "mission-123",
            "title": "Test Research",
            "evidence": [{"source": "test", "summary": "Found data"}],
            "quality_checkpoints": [{"gate": "evidence_links", "status": "pass"}],
        }

    @pytest.fixture
    def sample_metadata(self):
        """Sample execution metadata."""
        return ExecutionMetadata(
            total_tokens=5000,
            duration_seconds=45.0,
        )

    @pytest.mark.asyncio
    async def test_mark_completed_success(
        self, writer, mock_db_client, sample_protocol, sample_metadata, fixed_time, lease
    ):
        """Test successful mark_completed."""
        mock_db_client.fetchval.return_value = "mission-123"

        result = await writer.mark_completed(
            mission_id="mission-123",
            result_markdown="# Research Report\n\nFindings here.",
            result_protocol=sample_protocol,
            execution_metadata=sample_metadata,
            lease=lease,
            job_id="job-123",
            completed_at=fixed_time,
        )

        assert result is True
        mock_db_client.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_completed_serializes_jsonb(
        self, writer, mock_db_client, sample_protocol, sample_metadata, lease
    ):
        """Test that protocol and metadata are JSON serialized."""
        mock_db_client.fetchval.return_value = "mission-123"

        await writer.mark_completed(
            mission_id="mission-123",
            result_markdown="# Report",
            result_protocol=sample_protocol,
            execution_metadata=sample_metadata,
            lease=lease,
            job_id="job-123",
        )

        call_args = mock_db_client.fetchval.call_args[0]
        protocol_json = call_args[4]  # 5th arg
        metadata_json = call_args[5]  # 6th arg

        # Verify they're valid JSON strings
        parsed_protocol = json.loads(protocol_json)
        parsed_metadata = json.loads(metadata_json)

        assert parsed_protocol["mission_id"] == "mission-123"
        assert parsed_metadata["total_tokens"] == 5000
        # S75 m01: lying fields stripped
        assert "loops_executed" not in parsed_metadata
        assert "sources_count" not in parsed_metadata
        assert "quality_gates_passed" not in parsed_metadata

    @pytest.mark.asyncio
    async def test_mark_completed_rejects_stale_lease(
        self, writer, mock_db_client, sample_protocol, sample_metadata, lease
    ):
        """Test mark_completed when mission not found."""
        mock_db_client.fetchval.return_value = None

        with pytest.raises(StaleLeaseError):
            await writer.mark_completed(
                mission_id="nonexistent",
                result_markdown="# Report",
                result_protocol=sample_protocol,
                execution_metadata=sample_metadata,
                lease=lease,
                job_id="job-123",
            )

    @pytest.mark.asyncio
    async def test_mark_completed_db_error(
        self, writer, mock_db_client, sample_protocol, sample_metadata, lease
    ):
        """Test mark_completed handles database errors."""
        mock_db_client.fetchval.side_effect = Exception("Disk full")

        with pytest.raises(WriterError) as exc:
            await writer.mark_completed(
                mission_id="mission-123",
                result_markdown="# Report",
                result_protocol=sample_protocol,
                execution_metadata=sample_metadata,
                lease=lease,
                job_id="job-123",
            )

        assert "Disk full" in str(exc.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "result_markdown",
        ("# Report\r\n\r\nClaim.", "# Report\n\nClaim\x00."),
        ids=("crlf", "nul"),
    )
    async def test_mark_completed_rejects_noncanonical_report_without_db_write(
        self,
        writer,
        mock_db_client,
        sample_protocol,
        sample_metadata,
        lease,
        result_markdown,
    ) -> None:
        """Persistence cannot move citation coordinates after they are assigned."""

        with pytest.raises(WriterError, match="only LF newlines and no U\\+0000"):
            await writer.mark_completed(
                mission_id="mission-123",
                result_markdown=result_markdown,
                result_protocol=sample_protocol,
                execution_metadata=sample_metadata,
                lease=lease,
                job_id="job-123",
            )

        mock_db_client.fetchval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mark_completed_rejects_job_id_that_tracelab_cannot_store(
        self, writer, mock_db_client, sample_protocol, sample_metadata, lease
    ) -> None:
        """Correlation must fit TraceLab's nullable varchar(100) boundary."""

        with pytest.raises(WriterError, match="100-character limit"):
            await writer.mark_completed(
                mission_id="mission-123",
                result_markdown="# Report",
                result_protocol=sample_protocol,
                execution_metadata=sample_metadata,
                lease=lease,
                job_id="j" * 101,
            )

        mock_db_client.fetchval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_identical_completion_retry_reuses_one_result_key(
        self, writer, mock_db_client, sample_protocol, sample_metadata, lease
    ) -> None:
        """A retry addresses the same read-only idempotency branch."""
        mock_db_client.fetchval.return_value = "mission-123"

        for _ in range(2):
            assert await writer.mark_completed(
                mission_id="mission-123",
                result_markdown="# Report",
                result_protocol=sample_protocol,
                execution_metadata=sample_metadata,
                lease=lease,
                job_id="job-123",
            )

        first_key = mock_db_client.fetchval.call_args_list[0].args[-2]
        second_key = mock_db_client.fetchval.call_args_list[1].args[-2]
        assert first_key == second_key == lease.persistence_key("mission-123")
        assert mock_db_client.fetchval.call_args_list[0].args[-1] == "job-123"
        assert ResultWriter.MARK_COMPLETED_SQL.count("UPDATE missions") == 1
        assert "UNION ALL" in ResultWriter.MARK_COMPLETED_SQL
        assert "deepsearch_result_key = $9" in ResultWriter.MARK_COMPLETED_SQL

    def test_each_attempt_has_a_distinct_result_key(
        self,
        fixed_time,
        lease,
    ) -> None:
        """A recovered attempt cannot be mistaken for the prior attempt."""
        recovered = LeaseAttempt(
            owner="worker-b",
            token="opaque-token-b",
            leased_at=fixed_time,
            heartbeat_at=fixed_time,
            expires_at=fixed_time,
            attempt_count=lease.attempt_count + 1,
        )

        assert recovered.persistence_key("mission-123") != lease.persistence_key(
            "mission-123"
        )


class TestMarkFailed(TestResultWriter):
    """Tests for mark_failed method."""

    @pytest.mark.asyncio
    async def test_mark_failed_success(self, writer, mock_db_client, fixed_time, lease):
        """Test successful mark_failed."""
        mock_db_client.fetchval.return_value = "mission-123"

        result = await writer.mark_failed(
            mission_id="mission-123",
            error_message="Research timed out after 5 minutes",
            lease=lease,
            job_id="job-123",
            completed_at=fixed_time,
        )

        assert result is True
        mock_db_client.fetchval.assert_called_once()
        call_args = mock_db_client.fetchval.call_args[0]
        assert "mission-123" in call_args
        assert "Research timed out" in call_args[3]

    @pytest.mark.asyncio
    async def test_mark_failed_rejects_stale_lease(self, writer, mock_db_client, lease):
        """A missing/changed ownership proof must reject terminal persistence."""
        mock_db_client.fetchval.return_value = None

        with pytest.raises(StaleLeaseError):
            await writer.mark_failed(
                mission_id="nonexistent",
                error_message="Error occurred",
                lease=lease,
                job_id="job-123",
            )

    @pytest.mark.asyncio
    async def test_mark_failed_truncates_long_error(self, writer, mock_db_client, lease):
        """Test that very long error messages are truncated."""
        mock_db_client.fetchval.return_value = "mission-123"
        long_error = "x" * 5000  # Very long error

        await writer.mark_failed(
            mission_id="mission-123",
            error_message=long_error,
            lease=lease,
            job_id="job-123",
        )

        call_args = mock_db_client.fetchval.call_args[0]
        error_in_call = call_args[3]
        assert len(error_in_call) <= 4000
        assert error_in_call.endswith("...")

    @pytest.mark.asyncio
    async def test_mark_failed_db_error(self, writer, mock_db_client, lease):
        """Test mark_failed handles database errors."""
        mock_db_client.fetchval.side_effect = Exception("Network error")

        with pytest.raises(WriterError) as exc:
            await writer.mark_failed(
                mission_id="mission-123",
                error_message="Original error",
                lease=lease,
                job_id="job-123",
            )

        assert "Network error" in str(exc.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "result_markdown",
        ("# Report\r\n\r\nClaim.", "# Report\n\nClaim\x00."),
        ids=("crlf", "nul"),
    )
    async def test_validation_failed_rejects_noncanonical_report_without_db_write(
        self,
        writer,
        mock_db_client,
        lease,
        result_markdown,
    ) -> None:
        """Ledger-enrolled validation failures obey the lossless report contract."""

        with pytest.raises(WriterError, match="only LF newlines and no U\\+0000"):
            await writer.mark_failed(
                mission_id="mission-123",
                error_message="validation failed",
                outcome=MissionOutcome.VALIDATION_FAILED,
                lease=lease,
                job_id="job-123",
                result_markdown=result_markdown,
            )

        mock_db_client.fetchval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_blocked_retains_diagnostic_nul_sanitation(
        self,
        writer,
        mock_db_client,
        lease,
    ) -> None:
        """BLOCKED cannot enroll the outbox and must still land diagnostics."""

        mock_db_client.fetchval.return_value = "mission-123"
        await writer.mark_failed(
            mission_id="mission-123",
            error_message="identity violation",
            outcome=MissionOutcome.BLOCKED,
            lease=lease,
            job_id="job-123",
            result_markdown="# Report\n\nBad\x00 artifact",
        )

        assert mock_db_client.fetchval.await_args.args[4] == (
            "# Report\n\nBad artifact"
        )


class TestSerializeJsonb:
    """Tests for _serialize_jsonb helper."""

    def test_serialize_simple_dict(self):
        """Test serializing a simple dict."""
        data = {"key": "value", "number": 42}
        result = ResultWriter._serialize_jsonb(data, "test_field")
        parsed = json.loads(result)
        assert parsed == data

    def test_serialize_nested_dict(self):
        """Test serializing nested structures."""
        data = {
            "outer": {"inner": {"deep": "value"}},
            "list": [1, 2, 3],
        }
        result = ResultWriter._serialize_jsonb(data, "test_field")
        parsed = json.loads(result)
        assert parsed["outer"]["inner"]["deep"] == "value"

    def test_serialize_with_datetime(self):
        """Test serializing dict with datetime (uses default=str)."""
        dt = datetime(2024, 1, 15, 12, 0, 0)
        data = {"timestamp": dt}
        result = ResultWriter._serialize_jsonb(data, "test_field")
        parsed = json.loads(result)
        assert "2024-01-15" in parsed["timestamp"]

    def test_serialize_unicode(self):
        """Test serializing unicode characters."""
        data = {"text": "Hello 世界 🌍"}
        result = ResultWriter._serialize_jsonb(data, "test_field")
        assert "世界" in result
        assert "🌍" in result

    def test_serialize_empty_dict(self):
        """Test serializing empty dict."""
        result = ResultWriter._serialize_jsonb({}, "test_field")
        assert result == "{}"


class TestSqlQueries:
    """Tests to verify SQL query structure."""

    def test_mark_completed_sql_has_parameterized_placeholders(self):
        """Verify SQL uses parameterized queries."""
        sql = ResultWriter.MARK_COMPLETED_SQL
        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql
        assert "$4" in sql
        assert "$5" in sql
        assert "$10" in sql
        assert "status = 'completed'" in sql
        assert "result_markdown" in sql
        assert "result_protocol" in sql
        assert "execution_metadata" in sql
        assert "deepsearch_lease_token = $7" in sql
        assert "deepsearch_result_key = $9" in sql
        assert "deepsearch_job_id = $10" in sql

    def test_mark_failed_sql_has_parameterized_placeholders(self):
        """Verify SQL uses parameterized queries."""
        sql = ResultWriter.MARK_FAILED_SQL
        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql
        assert "$11" in sql
        assert "status = 'blocked'" in sql
        assert "error_message" in sql
        assert "deepsearch_lease_token = $8" in sql
        assert "deepsearch_job_id = $11" in sql

    def test_mark_validation_failed_sql_has_parameterized_placeholders(self):
        """Validation-failed terminal writes should have a dedicated SQL path."""
        sql = ResultWriter.MARK_VALIDATION_FAILED_SQL
        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql
        assert "$4" in sql
        assert "$5" in sql
        assert "$6" in sql
        assert "$11" in sql
        assert "status = 'validation_failed'" in sql
        assert "result_markdown" in sql
        assert "execution_metadata" in sql
        assert "deepsearch_job_id = $11" in sql
