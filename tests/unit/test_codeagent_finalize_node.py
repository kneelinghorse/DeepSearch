"""Unit tests for `deepsearch.agent.codeagent_finalize_node` — S64 live-retrieval path.

Verifies the node populates the same downstream-readable state contract that
the legacy ``finalize_node`` populated, so the worker (`deepsearch.worker.main`)
needs no changes after the seam is wired.

Post-S64: the upstream pool harvest is gone — the node no longer blocks on
empty ``sources_found`` because the CodeAgent retrieves cold via its live
tools. Block paths now key off prompt-construction errors, runner errors
(e.g. missing ``LINKUP_API_KEY``), agent errors, and empty agent output.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.execution_control import (
    ExecutionCancelledError,
    bind_execution_cancel_event,
)
from deepsearch.agent.outcome import GATE_REASON_ZERO_CITATIONS, MissionOutcome
from deepsearch.error_safety import safe_error_from_parts
from deepsearch.report.text import canonicalize_report_text


def _assert_safe_error(
    value: str | None,
    *,
    category: str,
    exception_class: str,
) -> None:
    """Pin the bounded category/class telemetry contract."""

    assert value is not None
    assert json.loads(value) == {
        "category": category,
        "exception_class": exception_class,
    }


def _critique_passthrough() -> CritiqueResult:
    """Default mocked critique: passes the draft through unchanged."""

    return CritiqueResult(
        output_markdown="# Report\n\nBody text with citation [1].",
        claim_count=2,
        gap_notes_inserted=0,
        runtime_seconds=2.5,
        token_usage={"input": 500, "output": 200, "total": 700},
        error=None,
    )


def _state(*, sources=None, mission_id="OODS-FIGMA-HOST-01"):
    return {
        "mission_id": mission_id,
        "mission_objectives": ["criterion 1", "criterion 2"],
        "mission_context": {
            "title": "Test Mission",
            "objective": "compare X vs Y",
            "background": "context blurb",
            "required_entities": ["X", "Y"],
            "excluded_entities": ["Z"],
            "deliverables": [],
        },
        "sources_found": sources if sources is not None else [],
        "telemetry": {},
    }


def _good_synthesis_result():
    return SynthesisResult(
        output_markdown="# Report\n\nBody text with citation [1].",
        step_count=4,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=42.0,
        token_usage={"input": 1000, "output": 2000, "total": 3000},
        tool_call_summary={"total_calls": 7, "by_tool": {"web_search": 3}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        error=None,
    )


@pytest.fixture()
def report_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    return tmp_path


def test_complete_path_populates_finalize_contract(report_dir):
    state = _state(
        sources=[
            {
                "url": "https://docs.aws.amazon.com/lambda/scaling",
                "title": "lambda scaling",
                "summary": "scaling text",
            }
        ]
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_good_synthesis_result(),
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_passthrough(),
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert result["routing_reason"] == ""
    assert result["synthesis_used"] is True
    assert "report_path" in result
    assert "protocol_path" in result
    assert Path(result["report_path"]).exists()
    assert Path(result["protocol_path"]).exists()
    assert "Body text with citation [1]" in Path(result["report_path"]).read_text()
    assert result["report_markdown"] == Path(result["report_path"]).read_text()
    assert result["protocol_payload"] == json.loads(
        Path(result["protocol_path"]).read_text()
    )

    s_tel = result["synthesis_telemetry"]
    assert s_tel["engine"] == "codeagent"
    assert s_tel["model_id"] == "deepseek-v4-pro"
    # S84 m04 — the framework label derives from the installed package
    # (a hardcoded version literal lied on every dependency bump); in this
    # test env smolagents is conftest-faked so the version-less fallback is
    # also legal.
    assert s_tel["framework"].startswith("smolagents")
    assert s_tel["framework"].endswith("CodeAgent")
    assert s_tel["step_count"] == 4
    assert s_tel["self_terminated"] is True
    assert s_tel["token_usage"]["total"] == 3000
    # Post-S64: no pool_size in telemetry — agent retrieves cold via live tools.
    assert "pool_size" not in s_tel
    # S66: critique pass observed and merged into telemetry.
    assert s_tel["critique_skipped"] is False
    assert s_tel["critique_telemetry"]["claim_count"] == 2
    assert s_tel["critique_telemetry"]["gap_notes_inserted"] == 0
    assert s_tel["critique_telemetry"]["error"] is None


def test_protocol_staging_failure_preserves_exact_inline_pair(report_dir) -> None:
    """A local protocol write failure cannot destroy the terminal artifacts."""

    async def run():
        with (
            patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=_good_synthesis_result(),
            ),
            patch.object(
                fnode,
                "run_critique",
                return_value=_critique_passthrough(),
            ),
            patch.object(
                fnode,
                "save_protocol",
                side_effect=OSError("protocol write failed"),
            ),
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    assert result["synthesis_used"] is True
    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert "report_path" in result
    assert "protocol_path" not in result
    assert result["report_markdown"] == Path(result["report_path"]).read_text()
    assert result["protocol_payload"]["mission_id"] == "OODS-FIGMA-HOST-01"
    assert len(result["generation_errors"]) == 1


def test_report_staging_failure_preserves_exact_inline_pair(report_dir) -> None:
    """A local report write failure falls back to the exact rendered bytes."""

    async def run():
        with (
            patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=_good_synthesis_result(),
            ),
            patch.object(
                fnode,
                "run_critique",
                return_value=_critique_passthrough(),
            ),
            patch.object(
                fnode,
                "save_report",
                side_effect=OSError("report write failed"),
            ),
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert "report_path" not in result
    assert result["report_markdown"].startswith(
        "# Research Report: OODS-FIGMA-HOST-01"
    )
    assert result["protocol_payload"] == json.loads(
        Path(result["protocol_path"]).read_text()
    )
    assert len(result["generation_errors"]) == 1


def test_both_staging_failures_preserve_exact_inline_pair(report_dir) -> None:
    """Attempt-local filesystem loss cannot erase either terminal artifact."""

    async def run():
        with (
            patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=_good_synthesis_result(),
            ),
            patch.object(
                fnode,
                "run_critique",
                return_value=_critique_passthrough(),
            ),
            patch.object(
                fnode,
                "save_report",
                side_effect=OSError("report write failed"),
            ),
            patch.object(
                fnode,
                "save_protocol",
                side_effect=OSError("protocol write failed"),
            ),
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert "report_path" not in result
    assert "protocol_path" not in result
    assert result["report_markdown"].startswith(
        "# Research Report: OODS-FIGMA-HOST-01"
    )
    assert result["protocol_payload"]["mission_id"] == "OODS-FIGMA-HOST-01"
    assert len(result["generation_errors"]) == 2


def test_report_construction_failure_preserves_scored_draft_inline(
    report_dir,
) -> None:
    """A deterministic renderer fault cannot erase paid synthesis."""

    async def run():
        with (
            patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=_good_synthesis_result(),
            ),
            patch.object(
                fnode,
                "run_critique",
                return_value=_critique_passthrough(),
            ),
            patch.object(
                fnode,
                "_render_report_document",
                side_effect=RuntimeError("renderer failed"),
            ),
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    expected = canonicalize_report_text(_critique_passthrough().output_markdown)
    assert result["mission_outcome"] == MissionOutcome.BLOCKED.value
    assert result["routing_reason"] == "report_construction_failed"
    assert result["report_markdown"] == expected
    assert "protocol_payload" not in result
    assert result["synthesis_telemetry"]["artifact_preservation"] == {
        "report": "scored_draft_inline",
        "protocol": "unavailable",
        "reason": "report_construction_failed",
    }


def test_protocol_construction_failure_preserves_rendered_report(
    report_dir,
) -> None:
    """A protocol schema fault cannot erase the already-rendered report."""

    async def run():
        with (
            patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=_good_synthesis_result(),
            ),
            patch.object(
                fnode,
                "run_critique",
                return_value=_critique_passthrough(),
            ),
            patch.object(
                fnode,
                "generate_mission_protocol",
                side_effect=RuntimeError("protocol failed"),
            ),
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.BLOCKED.value
    assert result["routing_reason"] == "protocol_construction_failed"
    assert result["report_markdown"] == Path(result["report_path"]).read_text()
    assert "protocol_payload" not in result


def test_nonempty_synthesis_error_preserves_report_as_validation_failed(
    report_dir,
) -> None:
    """An execution error cannot erase a substantive partial synthesis."""

    partial = _good_synthesis_result()
    partial.error = safe_error_from_parts("execution", "RuntimeError").to_text()

    async def run():
        with (
            patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=partial,
            ),
            patch.object(
                fnode,
                "run_critique",
                return_value=_critique_passthrough(),
            ),
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert result["routing_reason"] == "codeagent_partial_output_preserved"
    assert result["synthesis_telemetry"]["partial_output_preserved"] is True
    assert result["report_markdown"]
    assert result["protocol_payload"]


def test_lease_cancellation_after_sync_synthesis_prevents_artifact_writes(
    report_dir,
) -> None:
    """A completed executor call cannot persist after its lease fence was lost."""

    cancel_event = threading.Event()

    def _synthesis(**_kwargs):
        cancel_event.set()
        return _good_synthesis_result()

    async def run():
        with (
            bind_execution_cancel_event(cancel_event),
            patch.object(fnode, "run_codeagent_synthesis", side_effect=_synthesis),
            patch.object(fnode, "save_report") as save_report,
            patch.object(fnode, "save_protocol") as save_protocol,
        ):
            with pytest.raises(ExecutionCancelledError, match="lease loss"):
                await fnode.codeagent_finalize_node(_state())
        save_report.assert_not_called()
        save_protocol.assert_not_called()

    asyncio.run(run())


def test_lease_loss_after_report_write_prevents_protocol_write(report_dir) -> None:
    """The sequential persistence fence blocks the second canonical artifact."""

    cancel_event = threading.Event()
    original_save_report = fnode.save_report

    def save_then_lose_lease(*args, **kwargs):
        path = original_save_report(*args, **kwargs)
        cancel_event.set()
        return path

    async def run() -> None:
        with (
            bind_execution_cancel_event(cancel_event),
            patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=_good_synthesis_result(),
            ),
            patch.object(
                fnode,
                "run_critique",
                return_value=_critique_passthrough(),
            ),
            patch.object(
                fnode,
                "save_report",
                side_effect=save_then_lose_lease,
            ) as save_report,
            patch.object(fnode, "save_protocol") as save_protocol,
        ):
            with pytest.raises(ExecutionCancelledError, match="lease loss"):
                await fnode.codeagent_finalize_node(_state())
        assert save_report.call_count == 1
        save_protocol.assert_not_called()

    asyncio.run(run())
    assert len(list(report_dir.glob("*.md"))) == 1
    assert list(report_dir.glob("*.json")) == []


def test_empty_sources_no_longer_blocks(report_dir):
    """S64: empty sources_found is normal — the agent retrieves live."""

    state = _state(sources=[])

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_good_synthesis_result(),
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_passthrough(),
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert result["synthesis_used"] is True


def test_runner_error_blocks(report_dir, caplog):
    """Runner exception content cannot cross finalize logs or telemetry."""

    signed_url = "https://user:pass@example.test/private?signature=finalize-secret"
    body_marker = "HOSTILE_FINALIZE_RESPONSE_BODY"
    state = _state(
        sources=[{"url": "https://docs.aws.amazon.com/x", "title": "x"}]
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            side_effect=fnode.CodeAgentRunnerError(
                f"{signed_url} {body_marker}"
            ),
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.BLOCKED.value
    assert result["routing_reason"] == "codeagent_runner_error"
    assert result["synthesis_used"] is True
    assert "report_markdown" not in result
    assert "protocol_payload" not in result
    _assert_safe_error(
        result["synthesis_telemetry"]["synthesis_error"],
        category="execution",
        exception_class="CodeAgentRunnerError",
    )
    emitted = json.dumps(result, default=str) + caplog.text
    assert signed_url not in emitted
    assert body_marker not in emitted


def test_agent_error_blocks(report_dir):
    state = _state(
        sources=[{"url": "https://docs.aws.amazon.com/x", "title": "x"}]
    )
    failing = SynthesisResult(
        output_markdown="",
        step_count=2,
        self_terminated=False,
        hit_max_steps=False,
        runtime_seconds=10.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        tool_call_summary={"total_calls": 0, "by_tool": {}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        error=safe_error_from_parts("execution", "RuntimeError").to_text(),
    )

    async def run():
        with patch.object(fnode, "run_codeagent_synthesis", return_value=failing):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.BLOCKED.value
    assert result["routing_reason"] == "codeagent_synthesis_failed"
    assert result["synthesis_used"] is True
    _assert_safe_error(
        result["synthesis_telemetry"]["synthesis_error"],
        category="execution",
        exception_class="RuntimeError",
    )


def test_empty_output_blocks(report_dir):
    state = _state(
        sources=[{"url": "https://docs.aws.amazon.com/x", "title": "x"}]
    )
    empty = SynthesisResult(
        output_markdown="   \n\n  ",
        step_count=3,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=12.0,
        token_usage={"input": 100, "output": 0, "total": 100},
        tool_call_summary={"total_calls": 1, "by_tool": {}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        error=None,
    )

    async def run():
        with patch.object(fnode, "run_codeagent_synthesis", return_value=empty):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.BLOCKED.value
    assert result["routing_reason"] == "codeagent_empty_output"
    assert result["synthesis_used"] is True


def test_generation_recovery_telemetry_preserves_safe_failure_evidence() -> None:
    recovered = _good_synthesis_result()
    recovered.generation_failure_error = safe_error_from_parts(
        "execution", "AgentGenerationError"
    ).to_text()
    recovered.generation_failure_reason = "timeout"
    recovered.generation_failure_reason_source = "status"
    recovered.generation_failure_root_class = "APITimeoutError"
    recovered.generation_failure_http_status = 408
    recovered.generation_failure_transient = True
    recovered.generation_failure_step = 21
    recovered.generation_failure_pass = 0
    recovered.generation_failure_pass_count = 1
    recovered.generation_failure_consumed_tokens = 617_469
    recovered.generation_failure_retry_eligible = False
    recovered.generation_failure_retry_suppressed_reason = "paid_progress"
    recovered.generation_recovery_attempted = True
    recovered.generation_recovery_result = "success"
    recovered.generation_recovery_finish_reason = "stop"
    recovered.generation_error_recovered = True

    telemetry = fnode._build_synthesis_telemetry(recovered)

    _assert_safe_error(
        telemetry["generation_failure_error"],
        category="execution",
        exception_class="AgentGenerationError",
    )
    assert telemetry["generation_failure_reason"] == "timeout"
    assert telemetry["generation_failure_reason_source"] == "status"
    assert telemetry["generation_failure_root_class"] == "APITimeoutError"
    assert telemetry["generation_failure_http_status"] == 408
    assert telemetry["generation_failure_transient"] is True
    assert telemetry["generation_failure_step"] == 21
    assert telemetry["generation_failure_pass"] == 0
    assert telemetry["generation_failure_pass_count"] == 1
    assert telemetry["generation_failure_consumed_tokens"] == 617_469
    assert telemetry["generation_failure_retry_eligible"] is False
    assert (
        telemetry["generation_failure_retry_suppressed_reason"]
        == "paid_progress"
    )
    assert telemetry["generation_recovery_attempted"] is True
    assert telemetry["generation_recovery_result"] == "success"
    assert telemetry["generation_recovery_finish_reason"] == "stop"
    assert telemetry["generation_error_recovered"] is True


def test_bounded_recovery_combine_does_not_erase_generation_failure_warning() -> None:
    first = _good_synthesis_result()
    first.generation_failure_error = safe_error_from_parts(
        "execution", "AgentGenerationError"
    ).to_text()
    first.generation_failure_reason = "connection"
    first.generation_failure_reason_source = "class"
    first.generation_failure_root_class = "ConnectionError"
    first.generation_failure_transient = True
    first.generation_failure_step = 21
    first.generation_failure_pass_count = 1
    first.generation_failure_consumed_tokens = 617_469
    first.generation_failure_retry_eligible = False
    first.generation_failure_retry_suppressed_reason = "paid_progress"
    first.generation_recovery_attempted = True
    first.generation_recovery_result = "success"
    first.generation_recovery_finish_reason = "stop"
    first.generation_error_recovered = True
    first.recovered_from_scratchpad = True
    recovery = _good_synthesis_result()

    combined = fnode._combine_synthesis_results(first, recovery)

    assert combined.error is None
    assert combined.generation_failure_error == first.generation_failure_error
    assert combined.generation_failure_reason == "connection"
    assert combined.generation_failure_reason_source == "class"
    assert combined.generation_failure_root_class == "ConnectionError"
    assert combined.generation_failure_step == 21
    assert combined.generation_failure_pass == 0
    assert combined.generation_failure_pass_count == 1
    assert combined.generation_failure_consumed_tokens == 617_469
    assert combined.generation_failure_retry_eligible is False
    assert combined.generation_failure_retry_suppressed_reason == "paid_progress"
    assert combined.generation_recovery_result == "success"
    assert combined.generation_recovery_finish_reason == "stop"
    assert combined.generation_error_recovered is True
    assert combined.recovered_from_scratchpad is False

    telemetry = fnode._build_synthesis_telemetry(combined)
    assert telemetry["generation_failure_pass"] == 0
    assert telemetry["generation_failure_pass_count"] == 1


def test_qualified_generation_recovery_persists_exact_reviewable_pair(
    report_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cited provider-stop recovery survives the complete finalize path."""

    del report_dir
    from deepsearch.agent import citation_liveness as cl_mod

    url = "https://example.test/source"
    synthesis = _synthesis_with_sources(
        sources=[{"url": url, "title": "Primary source", "snippet": "evidence"}],
        markdown=f"# Report\n\nSupported claim. [cite:{url}]",
    )
    synthesis.generation_failure_error = safe_error_from_parts(
        "execution", "AgentGenerationError"
    ).to_text()
    synthesis.generation_failure_reason = "timeout"
    synthesis.generation_failure_reason_source = "class"
    synthesis.generation_failure_root_class = "TimeoutError"
    synthesis.generation_failure_transient = True
    synthesis.generation_failure_step = 4
    synthesis.generation_failure_pass_count = 1
    synthesis.generation_failure_consumed_tokens = 3_000
    synthesis.generation_failure_retry_suppressed_reason = "paid_progress"
    synthesis.generation_recovery_attempted = True
    synthesis.generation_recovery_result = "success"
    synthesis.generation_recovery_finish_reason = "stop"
    synthesis.generation_error_recovered = True
    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda value, timeout_s=5.0: {
            "alive": True,
            "status_code": 200,
            "final_url": value,
        },
    )

    async def run() -> dict:
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(
                "# Report\n\nSupported claim.", claim_count=1
            ),
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert result["delivery_quality"] == "usable_with_warning"
    assert (
        "generation_error_recovered_from_paid_memory"
        in result["quality_gate_failures"]
    )
    assert result["report_markdown"] == Path(result["report_path"]).read_text()
    assert result["protocol_payload"] == json.loads(
        Path(result["protocol_path"]).read_text()
    )
    citations = result["protocol_payload"]["citations"]
    assert len(citations) == 1
    citation = citations[0]
    assert result["report_markdown"][
        citation["start_index"] : citation["end_index"]
    ] == "Supported claim."


def test_unqualified_generation_partial_persists_as_validation_failed(
    report_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error-retaining cited partial is reviewable but never COMPLETE."""

    del report_dir
    from deepsearch.agent import citation_liveness as cl_mod

    url = "https://example.test/source"
    synthesis = _synthesis_with_sources(
        sources=[{"url": url, "title": "Primary source", "snippet": "evidence"}],
        markdown=f"# Partial report\n\nSupported claim. [cite:{url}]",
    )
    failure = safe_error_from_parts("execution", "AgentGenerationError").to_text()
    synthesis.error = failure
    synthesis.generation_failure_error = failure
    synthesis.generation_failure_reason = "connection"
    synthesis.generation_failure_reason_source = "message_marker"
    synthesis.generation_failure_root_class = "AgentGenerationError"
    synthesis.generation_failure_transient = True
    synthesis.generation_failure_step = 4
    synthesis.generation_failure_pass_count = 1
    synthesis.generation_failure_consumed_tokens = 3_000
    synthesis.generation_failure_retry_suppressed_reason = "paid_progress"
    synthesis.generation_recovery_attempted = True
    synthesis.generation_recovery_result = "failed"
    synthesis.recovered_from_scratchpad = True
    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda value, timeout_s=5.0: {
            "alive": True,
            "status_code": 200,
            "final_url": value,
        },
    )

    async def run() -> dict:
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(
                "# Partial report\n\nSupported claim.", claim_count=1
            ),
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert result["routing_reason"] == "generation_failure_unrecovered"
    assert result["delivery_quality"] == "failed"
    assert result["report_markdown"] == Path(result["report_path"]).read_text()
    assert result["protocol_payload"] == json.loads(
        Path(result["protocol_path"]).read_text()
    )
    assert len(result["protocol_payload"]["citations"]) == 1


# ---------------------------------------------------------------------------
# S66 critique wiring — telemetry, fallthrough, output replacement.
# ---------------------------------------------------------------------------


def test_critique_output_replaces_synthesis_in_report(report_dir):
    """The persisted markdown is the critique's output, not the raw synthesis."""

    state = _state()
    critiqued = CritiqueResult(
        output_markdown=(
            "# Report\n\nClaim with citation [1].\n\n"
            "Unsupported claim [Data unavailable in primary sources]."
        ),
        claim_count=2,
        gap_notes_inserted=1,
        runtime_seconds=2.1,
        token_usage={"input": 800, "output": 400, "total": 1200},
        error=None,
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_good_synthesis_result(),
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    # Half the claims required critique intervention, which crosses the S87
    # material-damage failure band even though the artifact remains persisted.
    assert result["mission_outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert result["delivery_quality"] == "failed"
    persisted = Path(result["report_path"]).read_text()
    # The critique's gap-noted body must show up on disk.
    assert "[Data unavailable in primary sources]" in persisted
    s_tel = result["synthesis_telemetry"]
    assert s_tel["critique_skipped"] is False
    assert s_tel["critique_telemetry"]["gap_notes_inserted"] == 1


def test_critique_llm_error_falls_through_with_uncritiqued_draft(report_dir):
    """Critique error persists the draft but fails outcome truth."""

    state = _state()
    fallthrough = CritiqueResult(
        output_markdown=_good_synthesis_result().output_markdown,
        claim_count=0,
        gap_notes_inserted=0,
        runtime_seconds=0.5,
        token_usage={"input": 0, "output": 0, "total": 0},
        error=safe_error_from_parts("execution", "RuntimeError").to_text(),
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_good_synthesis_result(),
        ), patch.object(
            fnode,
            "run_critique",
            return_value=fallthrough,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    # S66 fallthrough still preserves the artifact; S87 no longer reports an
    # uncritiqued artifact as a successful terminal outcome.
    assert result["mission_outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert result["delivery_quality"] == "failed"
    s_tel = result["synthesis_telemetry"]
    assert s_tel["critique_skipped"] is True
    _assert_safe_error(
        s_tel["critique_skip_reason"],
        category="execution",
        exception_class="RuntimeError",
    )
    # Persisted markdown is the un-critiqued draft.
    persisted = Path(result["report_path"]).read_text()
    assert "Body text with citation [1]" in persisted
    # Hard-error path is unaffected — synthesis_error stays None.
    assert s_tel["synthesis_error"] is None


def test_critique_unexpected_exception_falls_through(report_dir):
    """Unexpected critique errors persist the draft but fail outcome truth."""

    state = _state()

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_good_synthesis_result(),
        ), patch.object(
            fnode,
            "run_critique",
            side_effect=ValueError("unexpected critique bug"),
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert result["delivery_quality"] == "failed"
    s_tel = result["synthesis_telemetry"]
    assert s_tel["critique_skipped"] is True
    _assert_safe_error(
        s_tel["critique_skip_reason"],
        category="execution",
        exception_class="ValueError",
    )
    persisted = Path(result["report_path"]).read_text()
    # Un-critiqued draft persisted.
    assert "Body text with citation [1]" in persisted


def test_blocked_paths_skip_critique_entirely(report_dir):
    """BLOCKED short-circuits before critique — run_critique must NOT be called."""

    state = _state()

    async def run_runner_error():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            side_effect=fnode.CodeAgentRunnerError("LINKUP_API_KEY not set"),
        ), patch.object(fnode, "run_critique") as mc1:
            r = await fnode.codeagent_finalize_node(state)
            return r, mc1.call_count

    async def run_agent_error():
        failing = SynthesisResult(
            output_markdown="",
            step_count=2,
            self_terminated=False,
            hit_max_steps=False,
            runtime_seconds=10.0,
            token_usage={"input": 0, "output": 0, "total": 0},
            tool_call_summary={"total_calls": 0, "by_tool": {}},
            authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
            pool_size=0,
            step_trace=[],
            error=safe_error_from_parts("execution", "RuntimeError").to_text(),
        )
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=failing
        ), patch.object(fnode, "run_critique") as mc2:
            r = await fnode.codeagent_finalize_node(state)
            return r, mc2.call_count

    result_a, calls_a = asyncio.run(run_runner_error())
    result_b, calls_b = asyncio.run(run_agent_error())

    assert result_a["mission_outcome"] == MissionOutcome.BLOCKED.value
    assert result_b["mission_outcome"] == MissionOutcome.BLOCKED.value
    assert calls_a == 0, "critique must NOT run after CodeAgentRunnerError"
    assert calls_b == 0, "critique must NOT run after a synthesis error"


# ---------------------------------------------------------------------------
# S67 citation-extraction wiring — telemetry, stripped markdown on disk,
# protocol.citations populated, orphan-marker counting, title resolution.
# ---------------------------------------------------------------------------


def _critique_with_markers(markdown: str, *, claim_count: int = 2) -> CritiqueResult:
    return CritiqueResult(
        output_markdown=markdown,
        claim_count=claim_count,
        gap_notes_inserted=0,
        runtime_seconds=2.5,
        token_usage={"input": 500, "output": 200, "total": 700},
        error=None,
    )


def _synthesis_with_sources(*, sources, markdown="# Report\n\nDraft body."):
    return SynthesisResult(
        output_markdown=markdown,
        step_count=4,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=42.0,
        token_usage={"input": 1000, "output": 2000, "total": 3000},
        tool_call_summary={"total_calls": 7, "by_tool": {"web_search": 3}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=sources,
        error=None,
    )


@pytest.mark.parametrize(
    ("sources", "reason", "kept_count"),
    (
        (
            [
                {
                    "url": "HTTPS://Example.COM:443/a/../canonical",
                    "title": "First title",
                },
                {
                    "url": "https://example.com/canonical",
                    "title": "Conflicting title",
                },
            ],
            "canonical_metadata_conflict",
            1,
        ),
        (
            [
                {
                    "url": "https://user:password@example.com/private",
                    "title": "Unsafe source",
                }
            ],
            "invalid_source_record",
            0,
        ),
    ),
    ids=("canonical_metadata_conflict", "userinfo"),
)
def test_invalid_collected_source_is_quarantined_without_losing_artifacts(
    report_dir,
    sources,
    reason,
    kept_count,
) -> None:
    """One unsafe source projection cannot destroy a synthesized report."""

    synthesis = _synthesis_with_sources(sources=sources)

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            side_effect=lambda draft, *_args, **_kwargs: _critique_with_markers(
                draft
            ),
        ):
            result = await fnode.codeagent_finalize_node(
                _state(mission_id="INVALID-LEDGER-SOURCE")
            )
        return result

    result = asyncio.run(run())

    assert result["mission_outcome"] != MissionOutcome.BLOCKED.value
    assert Path(result["report_path"]).exists()
    assert Path(result["protocol_path"]).exists()
    projection = result["synthesis_telemetry"]["source_projection"]
    assert projection["reasons"] == {reason: 1}
    assert projection["kept_count"] == kept_count
    assert len(result["protocol_payload"]["sources_collected"]) == kept_count
    assert "password" not in repr(result["synthesis_telemetry"])


def test_protocol_identity_is_computed_after_nul_canonicalization(
    report_dir,
) -> None:
    """The identity-bound inline/file protocol must match stored JSONB bytes."""

    source_url = "https://example.test/nul-source"
    synthesis = _synthesis_with_sources(
        sources=[
            {
                "url": source_url,
                "title": "NUL source",
                "body": "before\x00after",
                "body_source": "source_fetch",
                "content_quality": "readable_body",
                "authority_tier": "medium",
            }
        ]
    )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            side_effect=lambda draft, *_args, **_kwargs: _critique_with_markers(
                draft
            ),
        ):
            return await fnode.codeagent_finalize_node(
                _state(mission_id="NUL-PROTOCOL")
            )

    result = asyncio.run(run())

    protocol_file = json.loads(Path(result["protocol_path"]).read_text())
    assert "\x00" not in repr(result["protocol_payload"])
    assert result["protocol_payload"] == protocol_file
    assert result["protocol_payload"]["sources_collected"][0]["body"] == (
        "beforeafter"
    )
    assert result["synthesis_telemetry"]["result_identity"][
        "protocol_sha256"
    ] == fnode._protocol_identity_sha256(protocol_file)


def test_canonical_equivalent_citation_and_critique_urls_persist_canonically(
    report_dir,
) -> None:
    """Every URL-bearing TraceLab projection surface shares one identity."""

    import json

    from deepsearch.agent.citation_liveness import LivenessTelemetry
    from deepsearch.agent.critique import CritiqueAnnotation

    raw_url = "HTTPS://Example.COM:443/a/../canonical?view=full"
    canonical_url = "https://example.com/canonical?view=full"
    claim = "A canonical source supports this claim."
    synthesis = _synthesis_with_sources(
        sources=[{"url": canonical_url, "title": "Canonical source"}],
        markdown=f"# Report\n\n{claim} [cite:{raw_url}]",
    )

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=1,
            live=1,
            distinct_urls_checked=1,
        )

    def critique(draft_markdown, _sources, _context, *, citations, scope_admission=None):
        assert [citation.url for citation in citations] == [canonical_url]
        return CritiqueResult(
            output_markdown=draft_markdown,
            claim_count=1,
            gap_notes_inserted=0,
            runtime_seconds=0.1,
            annotations=[
                CritiqueAnnotation(
                    anchor=claim,
                    verdict="supported",
                    citation_urls=(raw_url,),
                    applied=False,
                    reason="supported",
                )
            ],
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ), patch.object(
            fnode, "run_critique", side_effect=critique
        ):
            return await fnode.codeagent_finalize_node(
                _state(mission_id="CANONICAL-LEDGER-URL")
            )

    result = asyncio.run(run())
    protocol = json.loads(Path(result["protocol_path"]).read_text())

    assert protocol["sources_collected"][0]["url"] == canonical_url
    assert protocol["citations"][0]["url"] == canonical_url
    critique_urls = result["synthesis_telemetry"]["critique_telemetry"][
        "annotations"
    ][0]["citation_urls"]
    assert critique_urls == [canonical_url]


def test_persisted_protocol_citation_slices_match_tracelab_exact_claims(
    report_dir,
) -> None:
    """Indented Markdown citations persist trimmed, bounded exact slices."""

    import json

    from deepsearch.agent.citation_liveness import LivenessTelemetry

    first_url = "https://example.test/nested"
    second_url = "https://example.test/quote"
    synthesis = _synthesis_with_sources(
        sources=[
            {"url": first_url, "title": "Nested source"},
            {"url": second_url, "title": "Quoted source"},
        ],
        markdown=(
            "# Report\n\n"
            f"  - Nested cited claim. [cite:{first_url}]\n"
            f"  > Quoted cited claim. [cite:{second_url}]"
        ),
    )

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=2,
            live=2,
            distinct_urls_checked=2,
        )

    def keep_body(draft_markdown, _sources, _context, *, citations):
        return CritiqueResult(
            output_markdown=draft_markdown,
            claim_count=2,
            gap_notes_inserted=0,
            runtime_seconds=0.1,
        )

    async def run():
        with (
            patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=synthesis,
            ),
            patch.object(
                fnode,
                "check_citations_liveness",
                new=keep_live,
            ),
            patch.object(fnode, "run_critique", side_effect=keep_body),
        ):
            return await fnode.codeagent_finalize_node(
                _state(mission_id="TRIMMED-CITATION-SPANS")
            )

    result = asyncio.run(run())
    report = Path(result["report_path"]).read_text()
    protocol = json.loads(Path(result["protocol_path"]).read_text())
    claims = [
        report[item["start_index"] : item["end_index"]]
        for item in protocol["citations"]
    ]

    assert claims == ["- Nested cited claim.", "> Quoted cited claim."]
    assert all(claim and len(claim) <= 20_000 for claim in claims)
    assert all(claim == claim.strip() for claim in claims)


def test_citation_userinfo_is_quarantined_without_losing_artifacts(
    report_dir,
) -> None:
    """An unsafe citation is dropped while the report remains reviewable."""

    unsafe_url = "https://user:password@example.com/private"
    synthesis = _synthesis_with_sources(
        sources=[],
        markdown=f"# Report\n\nUnsafe citation. [cite:{unsafe_url}]",
    )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            side_effect=lambda draft, *_args, **_kwargs: _critique_with_markers(
                draft
            ),
        ):
            result = await fnode.codeagent_finalize_node(
                _state(mission_id="INVALID-LEDGER-CITATION")
            )
        return result

    result = asyncio.run(run())

    assert result["mission_outcome"] != MissionOutcome.BLOCKED.value
    assert Path(result["report_path"]).exists()
    assert Path(result["protocol_path"]).exists()
    assert result["protocol_payload"]["citations"] == []
    assert result["synthesis_telemetry"]["citation_url_projection"] == {
        "input_count": 1,
        "kept_count": 0,
        "dropped_count": 1,
        "reasons": {"invalid_url": 1},
    }
    assert "password" not in repr(result["synthesis_telemetry"])
    assert "password" not in result["report_markdown"]


def test_critique_url_quarantine_retains_prior_liveness_attempt(
    report_dir,
    monkeypatch,
) -> None:
    """A post-probe structural block cannot erase the shared HTTP audit."""

    from deepsearch.agent import citation_liveness as cl_mod
    from deepsearch.agent.critique import CritiqueAnnotation

    safe_url = "https://evidence.example.test/source"
    synthesis = _synthesis_with_sources(
        sources=[{"url": safe_url, "title": "Evidence"}],
        markdown=f"# Report\n\nSupported claim. [cite:{safe_url}]",
    )
    synthesis.tool_outcomes = {
        "ledger_records": [],
        "ledger_records_truncated": 0,
    }
    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda url, **kwargs: {
            "alive": True,
            "status_code": 200,
            "final_url": url,
            "error": None,
        },
    )

    critique = CritiqueResult(
        output_markdown="# Report\n\nSupported claim.",
        claim_count=1,
        gap_notes_inserted=0,
        runtime_seconds=0.1,
        annotations=[
            CritiqueAnnotation(
                anchor="Supported claim.",
                verdict="supported",
                citation_urls=("https://user:password@example.test/private",),
                applied=False,
                reason="supported",
            )
        ],
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critique,
        ):
            output = await fnode.codeagent_finalize_node(
                _state(mission_id="INVALID-CRITIQUE-URL-AFTER-LIVENESS")
            )
        return output

    result = asyncio.run(run())
    outcomes = result["synthesis_telemetry"]["tool_outcomes"]

    assert result["mission_outcome"] != MissionOutcome.BLOCKED.value
    assert result["synthesis_telemetry"]["critique_url_projection"][
        "dropped_count"
    ] == 1
    assert outcomes["ledger_records"] == [
        {
            "tool": "url_liveness",
            "url": safe_url,
            "status": "ok",
            "status_code": 200,
            "error_category": None,
            "alive": True,
        }
    ]
    assert "password" not in repr(result["synthesis_telemetry"])


def test_extract_citations_called_exactly_once_per_completion(report_dir):
    """The success path must run canonical extraction exactly once.

    Runtime call-count guard mirrors the S66 single-critique invariant. If a
    future change accidentally calls the extractor in a loop or adds a second
    callsite, this assertion fails. S68: extraction now precedes critique,
    so the markers live on the synthesis output rather than the critique
    output. S101 m02: the reader-number renderer reads the pure extractor
    once before it, without publishing annotations.
    """

    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[{"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0"}],
        markdown="# Report\n\nClaim. [cite:https://docs.aws.amazon.com/lambda/0]",
    )
    critiqued = _critique_with_markers("# Report\n\nClaim.")

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ), patch.object(
            fnode,
            "extract_citations",
            wraps=fnode.extract_citations,
        ) as ext_mock, patch.object(
            fnode,
            "_extract_and_resolve_citations",
            wraps=fnode._extract_and_resolve_citations,
        ) as canonical_mock:
            r = await fnode.codeagent_finalize_node(state)
            return r, ext_mock.call_count, canonical_mock.call_count

    result, ext_calls, canonical_calls = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert (ext_calls, canonical_calls) == (2, 1)


def test_citation_telemetry_always_present_on_success_path(report_dir):
    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[
            {"url": "https://a.example", "title": "A"},
            {"url": "https://b.example", "title": "B"},
        ],
        markdown=(
            "# Report\n\nFirst claim. [cite:https://a.example] "
            "Second claim. [cite:https://b.example]"
        ),
    )
    critiqued = _critique_with_markers(
        "# Report\n\nFirst claim. Second claim."
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ), patch.object(
            fnode,
            "check_citations_liveness",
            wraps=fnode.check_citations_liveness,
        ) as live_mock:
            r = await fnode.codeagent_finalize_node(state)
            return r, live_mock

    result, _ = asyncio.run(run())
    s_tel = result["synthesis_telemetry"]
    assert "citation_telemetry" in s_tel
    ct = s_tel["citation_telemetry"]
    assert ct["citations_extracted"] == 2
    assert ct["distinct_urls_cited"] == 2
    assert isinstance(ct["runtime_seconds"], float)
    assert ct["runtime_seconds"] >= 0


def test_citation_telemetry_present_when_zero_markers(report_dir):
    """A draft with no `[cite:URL]` markers still emits telemetry."""

    state = _state()
    critiqued = _critique_with_markers("# Report\n\nNo citation markers here at all.")

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_good_synthesis_result(),
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    ct = result["synthesis_telemetry"]["citation_telemetry"]
    assert ct["citations_extracted"] == 0
    assert ct["orphan_marker_count"] == 0
    assert ct["distinct_urls_cited"] == 0


def test_persisted_markdown_has_no_cite_markers(report_dir):
    """`result_markdown` ships the stripped version; `[cite:` must not appear.

    S68 reorder: markers live on the synthesis output and are stripped by
    `extract_citations` before critique sees the draft. The persisted
    text is the critique output, but since critique receives the stripped
    text and rule 7 forbids inventing markers, no `[cite:` substring can
    appear in the final markdown.
    """

    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[
            {"url": "https://a.example", "title": "A"},
            {"url": "https://b.example", "title": "B"},
        ],
        markdown=(
            "# Report\n\nFirst claim. [cite:https://a.example]\n"
            "Second claim. [cite:https://b.example]"
        ),
    )
    critiqued = _critique_with_markers(
        "# Report\n\nFirst claim.\nSecond claim."
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    persisted = Path(result["report_path"]).read_text()
    assert "[cite:" not in persisted
    # The cited claims themselves still appear.
    assert "First claim" in persisted
    assert "Second claim" in persisted


def test_protocol_citations_field_populated(report_dir):
    """`generate_mission_protocol` output gets `citations` filled by m02 wiring."""

    import json

    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[{"url": "https://docs.aws.amazon.com/lambda", "title": "L"}],
        markdown="# Report\n\nClaim text. [cite:https://docs.aws.amazon.com/lambda]",
    )
    critiqued = _critique_with_markers("# Report\n\nClaim text.")

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert "protocol_path" in result
    persisted_protocol = json.loads(Path(result["protocol_path"]).read_text())
    assert "citations" in persisted_protocol
    assert len(persisted_protocol["citations"]) == 1
    cite = persisted_protocol["citations"][0]
    assert cite["type"] == "url_citation"
    assert cite["url"] == "https://docs.aws.amazon.com/lambda"
    assert isinstance(cite["start_index"], int)
    assert isinstance(cite["end_index"], int)
    assert cite["end_index"] >= cite["start_index"]


def test_protocol_citation_spans_index_exact_persisted_report(report_dir):
    """Wrapper and run preamble cannot leave protocol spans body-relative."""

    import json

    from deepsearch.agent.citation_liveness import LivenessTelemetry

    first_claim = "RFC 9110 defines 307 as Temporary Redirect."
    second_claim = "The control endpoint deliberately returns 404."
    first_url = "https://www.rfc-editor.org/rfc/rfc9110"
    second_url = "https://example.test/deliberate-404"
    synthesis_markdown = (
        "# Agent-authored duplicate H1\n\n"
        f"{first_claim} [cite:{first_url}]\n\n"
        f"{second_claim} [cite:{second_url}]\n\n"
        "## References\n\n"
        f"[1] RFC 9110 — {first_url}\n"
        f"[2] Deliberate control — {second_url}\n"
    )
    stripped_markdown = synthesis_markdown.replace(
        f" [cite:{first_url}]", ""
    ).replace(f" [cite:{second_url}]", "")
    synthesis = _synthesis_with_sources(
        sources=[
            {"url": first_url, "title": "RFC 9110"},
            {"url": second_url, "title": "Deliberate control"},
        ],
        markdown=synthesis_markdown,
    )
    critique = _critique_with_markers(stripped_markdown)

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=len(citations),
            live=len(citations),
            distinct_urls_checked=len(citations),
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode, "run_critique", return_value=critique
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ):
            return await fnode.codeagent_finalize_node(
                _state(mission_id="LEDGER-2-SPAN-E2E")
            )

    result = asyncio.run(run())
    report = Path(result["report_path"]).read_text()
    protocol = json.loads(Path(result["protocol_path"]).read_text())
    citations = protocol["citations"]

    assert report.startswith("# Research Report: LEDGER-2-SPAN-E2E")
    assert "**Sources collected:** 2" in report
    assert [entry["url"] for entry in citations] == [first_url, second_url]
    assert [
        report[entry["start_index"] : entry["end_index"]]
        for entry in citations
    ] == [first_claim, second_claim]
    identity = result["synthesis_telemetry"]["result_identity"]
    assert identity["span_reference_len"] == len(report)


@pytest.mark.parametrize(
    ("case", "prefix"),
    (
        ("crlf_before_claim", "# Agent H1\r\n\r\nPrelude.\r\n\r\n"),
        ("nul_before_claim", "# Agent H1\n\nPre\x00lude.\n\n"),
    ),
)
def test_actor_text_is_canonical_before_claim_span_assignment(
    report_dir,
    case,
    prefix,
):
    """CRLF/NUL before a marker cannot shift persisted claim coordinates."""

    import json

    from deepsearch.agent.citation_liveness import LivenessTelemetry

    claim = "RFC 9110 requires method preservation for a 307 response."
    url = "https://www.rfc-editor.org/rfc/rfc9110#section-15.4.8"
    actor_markdown = (
        f"{prefix}{claim} [cite:{url}]\n\n"
        f"## References\n\n[1] RFC 9110 — {url}\n"
    )
    canonical_stripped = canonicalize_report_text(actor_markdown).replace(
        f" [cite:{url}]",
        "",
    )
    synthesis = _synthesis_with_sources(
        sources=[{"url": url, "title": "RFC 9110"}],
        markdown=actor_markdown,
    )

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=len(citations),
            live=len(citations),
            distinct_urls_checked=len(citations),
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(canonical_stripped),
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ):
            return await fnode.codeagent_finalize_node(
                _state(mission_id=f"CANONICAL-{case}")
            )

    result = asyncio.run(run())
    report_path = Path(result["report_path"])
    report = report_path.read_text(encoding="utf-8")
    protocol = json.loads(Path(result["protocol_path"]).read_text())
    citation = protocol["citations"][0]

    assert b"\r" not in report_path.read_bytes()
    assert b"\x00" not in report_path.read_bytes()
    assert report[citation["start_index"] : citation["end_index"]] == claim
    identity = result["synthesis_telemetry"]["result_identity"]
    assert identity["report_len"] == len(report)
    assert identity["span_reference_len"] == len(report)


@pytest.mark.parametrize(
    ("case", "background", "expected_background"),
    (
        (
            "context_crlf",
            "Context line one.\r\nContext line two.",
            "Context line one.\nContext line two.",
        ),
        (
            "context_nul",
            "Context before\x00after.",
            "Context beforeafter.",
        ),
    ),
)
def test_generated_wrapper_context_is_canonical_before_rebase_and_save(
    report_dir,
    case,
    background,
    expected_background,
):
    """Non-actor wrapper text cannot drift persisted spans or identity."""

    import json

    from deepsearch.agent.citation_liveness import LivenessTelemetry

    claim = "The cited actor claim remains byte-exact."
    url = "https://example.test/context-canonicalization"
    actor_markdown = (
        f"# Actor H1\n\n{claim} [cite:{url}]\n\n"
        f"## References\n\n[1] Context source — {url}\n"
    )
    stripped_markdown = actor_markdown.replace(f" [cite:{url}]", "")
    synthesis = _synthesis_with_sources(
        sources=[{"url": url, "title": "Context source"}],
        markdown=actor_markdown,
    )
    state = _state(mission_id=f"CANONICAL-{case}")
    state["mission_context"]["background"] = background

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=len(citations),
            live=len(citations),
            distinct_urls_checked=len(citations),
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(stripped_markdown),
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    report_path = Path(result["report_path"])
    report_bytes = report_path.read_bytes()
    report = report_bytes.decode("utf-8")
    protocol = json.loads(Path(result["protocol_path"]).read_text())
    citation = protocol["citations"][0]

    assert b"\r" not in report_bytes
    assert b"\x00" not in report_bytes
    assert expected_background in report
    assert report[citation["start_index"] : citation["end_index"]] == claim
    identity = result["synthesis_telemetry"]["result_identity"]
    assert identity["report_sha256"] == fnode._identity_sha256(report)
    assert identity["report_len"] == len(report)
    assert identity["span_reference_len"] == len(report)


def test_repeated_claim_text_binds_in_document_order(report_dir):
    """Repeated exact claim text is ordinal, not ambiguous.

    S92 m06 regression pin. Requiring a globally unique match made a report
    whose own body repeats a sentence unshippable: CAT-R006/R008/R009-S89 each
    ran a full paid mission and were destroyed at this exact call. The Nth
    occurrence in document order belongs to the Nth citation, and the report
    ships.
    """

    from deepsearch.agent.citation_liveness import LivenessTelemetry

    claim = "A repeated claim cannot be bound by guessing."
    url = "https://example.test/source"
    synthesis_markdown = f"# Report\n\n{claim} [cite:{url}]\n\n{claim}\n"
    stripped_markdown = synthesis_markdown.replace(f" [cite:{url}]", "")
    synthesis = _synthesis_with_sources(
        sources=[{"url": url, "title": "Source"}],
        markdown=synthesis_markdown,
    )
    captured = {}
    original_generate = fnode.generate_markdown_report

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=1,
            live=1,
            distinct_urls_checked=1,
        )

    def capture_generated_report(*args, **kwargs):
        report = original_generate(*args, **kwargs)
        captured["report"] = report
        return report

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(stripped_markdown),
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ), patch.object(
            fnode,
            "generate_markdown_report",
            side_effect=capture_generated_report,
        ), patch.object(
            fnode, "save_report"
        ) as save_report, patch.object(
            fnode, "save_protocol"
        ) as save_protocol:
            result = await fnode.codeagent_finalize_node(
                _state(mission_id="AMBIGUOUS-SPAN")
            )
        save_report.assert_called_once()
        save_protocol.assert_called_once()
        return result, save_protocol.call_args[0][0]

    result, protocol = asyncio.run(run())

    assert result["mission_outcome"] != MissionOutcome.BLOCKED.value
    assert result["routing_reason"] != "citation_span_rebase_failed"
    report = captured["report"]
    assert report.count(claim) == 2, "fixture must prove the claim repeats"
    assert len(protocol.citations) == 1
    citation = protocol.citations[0]
    assert report[citation.start_index : citation.end_index] == claim
    assert citation.start_index == report.index(claim), "first occurrence binds"
    assert result["synthesis_telemetry"]["citations_unmappable"] == 0


def test_critique_removed_cited_claim_drops_only_that_citation(report_dir):
    """A removed slice costs its own citation, never the whole report.

    S92 m06. The old policy raised here and the worker converted that into a
    hard `blocked` with no artifacts, discarding a completed paid run over one
    stale offset. Degrade instead: the unmappable coordinate is dropped and
    counted, and the synthesized report is still persisted and reviewable.
    """

    from deepsearch.agent.citation_liveness import LivenessTelemetry

    claim = "This cited claim is removed by critique."
    url = "https://example.test/source"
    synthesis = _synthesis_with_sources(
        sources=[{"url": url, "title": "Source"}],
        markdown=f"# Report\n\n{claim} [cite:{url}]",
    )

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=1,
            live=1,
            distinct_urls_checked=1,
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(
                "# Report\n\nA different claim survived critique."
            ),
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ), patch.object(
            fnode, "save_report"
        ) as save_report, patch.object(
            fnode, "save_protocol"
        ) as save_protocol:
            result = await fnode.codeagent_finalize_node(
                _state(mission_id="MISSING-SPAN")
            )
        save_report.assert_called_once()
        save_protocol.assert_called_once()
        return result, save_protocol.call_args[0][0]

    result, protocol = asyncio.run(run())

    assert result["mission_outcome"] != MissionOutcome.BLOCKED.value
    assert result["routing_reason"] != "citation_span_rebase_failed"
    assert protocol.citations == []
    rebase = result["synthesis_telemetry"]["citation_rebase"]
    assert rebase["dropped_count"] == 1
    assert rebase["reasons"] == {"source_span_missing_from_persisted_segment": 1}
    assert "anchor_error" not in rebase


def test_all_rebased_citations_lost_regrades_long_report_validation_failed(
    report_dir,
) -> None:
    """The terminal gate grades persisted coordinates, not pre-rebase markers."""

    from deepsearch.agent.citation_liveness import LivenessTelemetry

    source_url = "https://example.test/source"
    cited_claim = "This claim is present only before critique."
    surviving_body = " ".join(["substantiated"] * 240)
    synthesis = _synthesis_with_sources(
        sources=[{"url": source_url, "title": "Source"}],
        markdown=(
            f"# Report\n\n{cited_claim} [cite:{source_url}]\n\n"
            f"## Analysis\n\n{surviving_body}"
        ),
    )
    post_critique = f"# Report\n\n## Analysis\n\n{surviving_body}"

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        del budget_seconds
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=1,
            live=1,
            distinct_urls_checked=1,
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(post_critique),
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ):
            return await fnode.codeagent_finalize_node(
                _state(mission_id="ALL-CITATIONS-LOST")
            )

    result = asyncio.run(run())

    assert result["mission_outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert result["routing_reason"] == GATE_REASON_ZERO_CITATIONS
    assert result["protocol_payload"]["citations"] == []
    citation_telemetry = result["synthesis_telemetry"]["citation_telemetry"]
    assert citation_telemetry["citations_extracted"] == 0
    assert citation_telemetry["distinct_urls_cited"] == 0
    assert citation_telemetry["citations_extracted_pre_rebase"] == 1
    assert citation_telemetry["distinct_urls_cited_pre_rebase"] == 1
    assert Path(result["report_path"]).exists()
    assert Path(result["protocol_path"]).exists()


@pytest.mark.parametrize(
    ("case", "cited_text", "synthesis_markdown"),
    [
        (
            "stripped_h1",
            "# Shared cited heading",
            "# Shared cited heading [cite:https://example.test/source]\n\n"
            "A body claim remains after H1 stripping.",
        ),
        (
            "cleaner_removed_line",
            "Here is the cleaner-removed cited claim.",
            "# Report\n\nHere is the cleaner-removed cited claim. "
            "[cite:https://example.test/source]\n\n"
            "A body claim remains after synthesis cleaning.",
        ),
    ],
)
def test_removed_citation_text_cannot_rebind_to_wrapper_or_context(
    report_dir,
    case,
    cited_text,
    synthesis_markdown,
):
    """The exact inserted segment, not a same-text report decoy, is binding.

    This anti-gaming invariant is unchanged by S92 m06 and must never relax:
    text a transform removed from the body may not silently re-bind to
    byte-identical text in the wrapper title or Mission Context. What changed
    is only the consequence -- the citation is dropped rather than the report.
    """

    from deepsearch.agent.citation_liveness import LivenessTelemetry

    url = "https://example.test/source"
    stripped_markdown = synthesis_markdown.replace(f" [cite:{url}]", "")
    state = _state(mission_id=f"SPAN-DECOY-{case}")
    if case == "stripped_h1":
        # normalize_mission_title retains the leading hash, so the removed
        # cited H1 appears byte-identically inside the wrapper title.
        state["mission_title"] = cited_text
        state["mission_context"]["title"] = cited_text
    else:
        # The cleaner removes the cited line from synthesized content while
        # Mission Context renders the same bytes later in the report.
        state["mission_context"]["background"] = cited_text

    synthesis = _synthesis_with_sources(
        sources=[{"url": url, "title": "Source"}],
        markdown=synthesis_markdown,
    )
    captured = {}
    original_generate = fnode.generate_markdown_report

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=1,
            live=1,
            distinct_urls_checked=1,
        )

    def capture_generated_report(*args, **kwargs):
        report = original_generate(*args, **kwargs)
        captured["report"] = report
        return report

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(stripped_markdown),
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ), patch.object(
            fnode,
            "generate_markdown_report",
            side_effect=capture_generated_report,
        ), patch.object(
            fnode, "save_report"
        ) as save_report, patch.object(
            fnode, "save_protocol"
        ) as save_protocol:
            result = await fnode.codeagent_finalize_node(state)
        save_report.assert_called_once()
        save_protocol.assert_called_once()
        return result, save_protocol.call_args[0][0]

    result, protocol = asyncio.run(run())

    assert cited_text in captured["report"], "fixture must prove the decoy exists"
    # The decoy is never bound: the citation is dropped, not re-pointed at it.
    assert protocol.citations == []
    decoy_start = captured["report"].index(cited_text)
    assert all(
        citation.start_index != decoy_start for citation in protocol.citations
    )
    assert result["mission_outcome"] != MissionOutcome.BLOCKED.value
    assert result["synthesis_telemetry"]["citation_rebase"]["reasons"] == {
        "source_span_missing_from_persisted_segment": 1
    }


def test_orphan_marker_count_reflects_sources_collected(report_dir):
    """Orphan = URL on a marker that's not in `sources_collected`."""

    state = _state()
    sources = [
        {"url": "https://docs.aws.amazon.com/lambda/0", "title": "Lambda 0", "snippet": "x"},
        {"url": "https://docs.aws.amazon.com/lambda/1", "title": "Lambda 1", "snippet": "y"},
    ]
    synthesis_md = (
        "# Report\n\n"
        "Grounded claim. [cite:https://docs.aws.amazon.com/lambda/0]\n"
        "Also grounded. [cite:https://docs.aws.amazon.com/lambda/1]\n"
        "Orphan claim. [cite:https://orphan.example]\n"
    )
    critiqued = _critique_with_markers(
        "# Report\n\nGrounded claim.\nAlso grounded.\nOrphan claim."
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_synthesis_with_sources(sources=sources, markdown=synthesis_md),
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    ct = result["synthesis_telemetry"]["citation_telemetry"]
    assert ct["citations_extracted"] == 3
    assert ct["distinct_urls_cited"] == 3
    assert ct["orphan_marker_count"] == 1


def test_titles_resolved_from_sources_collected(report_dir):
    """Best-effort title lookup populates `CitationAnnotation.title`."""

    import json

    state = _state()
    sources = [
        {
            "url": "https://docs.aws.amazon.com/lambda/0",
            "title": "AWS Lambda Concurrency",
            "snippet": "x",
        }
    ]
    synthesis_md = "# Report\n\nClaim. [cite:https://docs.aws.amazon.com/lambda/0]"
    critiqued = _critique_with_markers("# Report\n\nClaim.")

    # S82 m02: the cited URL is in sources_collected, so a failing liveness
    # probe would classify it dead and the quarantine would (correctly) drop
    # it. This test is about TITLE RESOLUTION, so stub the probe alive to keep
    # the citation in the shipped set.
    from deepsearch.agent import citation_liveness as cl_mod

    with patch.object(
        cl_mod,
        "url_liveness",
        lambda url, timeout_s=5.0, **kwargs: {"alive": True, "status_code": 200, "final_url": url},
    ):
        async def run():
            with patch.object(
                fnode,
                "run_codeagent_synthesis",
                return_value=_synthesis_with_sources(sources=sources, markdown=synthesis_md),
            ), patch.object(
                fnode,
                "run_critique",
                return_value=critiqued,
            ):
                return await fnode.codeagent_finalize_node(state)

        result = asyncio.run(run())
    persisted = json.loads(Path(result["protocol_path"]).read_text())
    assert persisted["citations"][0]["title"] == "AWS Lambda Concurrency"


def test_orphan_marker_title_stays_none(report_dir, monkeypatch):
    """An orphan marker (URL not in sources_collected) gets no title.

    S83 m03: a TRUE orphan is reachable-but-uncollected (live=True), so it
    SURVIVES the quarantine and ships with no resolved title. (A live=False
    not-in-sources URL is now hallucinated and dropped — covered by
    test_s83_m03_invariants.py.) Mock the probe so the orphan resolves live.
    """

    import json

    from deepsearch.agent import citation_liveness as cl_mod

    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda url, timeout_s=5.0, **kwargs: {"alive": True, "status_code": 200, "final_url": url},
    )

    state = _state()
    sources = [
        {"url": "https://known", "title": "Known", "snippet": "x"},
    ]
    synthesis_md = "# Report\n\nOrphan. [cite:https://unknown]"
    critiqued = _critique_with_markers("# Report\n\nOrphan.")

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_synthesis_with_sources(sources=sources, markdown=synthesis_md),
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    persisted = json.loads(Path(result["protocol_path"]).read_text())
    cite = persisted["citations"][0]
    # `save_protocol` uses Pydantic's `model_dump_json` (not the custom
    # `to_dict()`), so a `None` title round-trips as `null` rather than being
    # omitted. The semantic invariant we lock here is that the orphan marker
    # has no resolved title.
    assert cite.get("title") is None


# ---------------------------------------------------------------------------
# S68 citation liveness wiring — runtime guards. Static AST guards live in
# tests/unit/test_liveness_invariants.py; this block locks the *runtime*
# behaviour: the gate is invoked once, telemetry is the 10-field PD7 shape,
# annotations are mutated in place before critique sees them, the protocol
# round-trips the live field, and the defense-in-depth wrapper preserves
# annotations on a contract-violating raise.
# ---------------------------------------------------------------------------


_LIVENESS_TELEMETRY_KEYS = {
    "checked",
    "live",
    "dead",
    "orphan",
    "unknown",
    "hallucinated_url_count",
    "distinct_urls_checked",
    "wall_clock_ms",
    "cache_hits",
    "budget_exhausted",
}


def test_main_validation_liveness_attempt_enters_lossless_ledger(
    report_dir,
    monkeypatch,
) -> None:
    """The post-synthesis HTTP probe is a projection input, not telemetry only."""

    from deepsearch.agent import citation_liveness as cl_mod

    raw_url = "HTTPS://Evidence.Example.TEST:443/a/../source"
    canonical_url = "https://evidence.example.test/source"
    synthesis = _synthesis_with_sources(
        sources=[{"url": canonical_url, "title": "Evidence"}],
        markdown=f"# Report\n\nSupported claim. [cite:{raw_url}]",
    )
    synthesis.tool_outcomes = {
        "ledger_records": [],
        "ledger_records_truncated": 0,
    }

    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda url, **kwargs: {
            "alive": True,
            "status_code": 200,
            "final_url": url,
            "error": None,
        },
    )

    def keep_draft(draft_markdown, _sources, _context, *, citations, scope_admission=None):
        assert citations[0].url == canonical_url
        return _critique_with_markers(draft_markdown, claim_count=1)

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(fnode, "run_critique", side_effect=keep_draft):
            return await fnode.codeagent_finalize_node(
                _state(mission_id="LIVENESS-LEDGER-MAIN")
            )

    result = asyncio.run(run())
    outcomes = result["synthesis_telemetry"]["tool_outcomes"]

    assert outcomes["ledger_records"] == [
        {
            "tool": "url_liveness",
            "url": canonical_url,
            "status": "ok",
            "status_code": 200,
            "error_category": None,
            "alive": True,
        }
    ]
    assert outcomes["ledger_records_truncated"] == 0


def test_check_citations_liveness_invoked_exactly_once(report_dir, monkeypatch):
    state = _state()
    sources = [{"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0"}]
    synthesis = _synthesis_with_sources(
        sources=sources,
        markdown="# Report\n\nClaim. [cite:https://docs.aws.amazon.com/lambda/0]",
    )
    critiqued = _critique_with_markers("# Report\n\nClaim.")

    # Stub the underlying URL probe so no real network HEAD is issued.
    from deepsearch.agent import citation_liveness as cl_mod

    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda url, timeout_s=5.0, **kwargs: {"alive": True, "status_code": 200, "final_url": url},
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ), patch.object(
            fnode,
            "check_citations_liveness",
            wraps=fnode.check_citations_liveness,
        ) as live_mock:
            r = await fnode.codeagent_finalize_node(state)
            return r, live_mock.call_count

    result, live_calls = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert live_calls == 1, (
        f"Expected check_citations_liveness called exactly once, got {live_calls}"
    )


def test_liveness_telemetry_always_present_pd7_shape(report_dir, monkeypatch):
    state = _state()
    sources = [{"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0"}]
    synthesis = _synthesis_with_sources(
        sources=sources,
        markdown="# Report\n\nClaim. [cite:https://docs.aws.amazon.com/lambda/0]",
    )
    critiqued = _critique_with_markers("# Report\n\nClaim.")

    from deepsearch.agent import citation_liveness as cl_mod

    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda url, timeout_s=5.0, **kwargs: {"alive": True, "status_code": 200, "final_url": url},
    )

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    s_tel = result["synthesis_telemetry"]
    assert "liveness_telemetry" in s_tel, (
        "synthesis_telemetry['liveness_telemetry'] must be present on the "
        "success path (PD7)"
    )
    lt = s_tel["liveness_telemetry"]
    assert set(lt.keys()) == _LIVENESS_TELEMETRY_KEYS, (
        f"liveness_telemetry keys diverge from PD7 shape: "
        f"missing {_LIVENESS_TELEMETRY_KEYS - set(lt.keys())} / "
        f"extra {set(lt.keys()) - _LIVENESS_TELEMETRY_KEYS}"
    )
    assert lt["checked"] == 1
    assert lt["live"] == 1
    assert lt["budget_exhausted"] is False
    # hallucinated_url_count is the load-bearing local metric (S73 baseline).
    assert lt["hallucinated_url_count"] == lt["orphan"]


def test_liveness_telemetry_zero_valued_when_zero_citations(report_dir):
    """Empty marker draft → zero-valued telemetry (always present per PD7)."""

    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[],
        markdown="# Report\n\nNo markers here at all.",
    )
    critiqued = _critique_with_markers("# Report\n\nNo markers here at all.")

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    lt = result["synthesis_telemetry"]["liveness_telemetry"]
    assert set(lt.keys()) == _LIVENESS_TELEMETRY_KEYS
    assert lt["checked"] == 0
    assert lt["live"] == lt["dead"] == lt["orphan"] == lt["unknown"] == 0
    assert lt["budget_exhausted"] is False


def test_run_critique_receives_live_populated_annotations(report_dir, monkeypatch):
    """The critique pass receives `citations=` with `live` already populated.

    Locks the PD2 ordering at the runtime layer: liveness check fires
    before run_critique sees the annotations, and the kwarg is the channel.
    """

    state = _state()
    sources = [{"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0"}]
    synthesis = _synthesis_with_sources(
        sources=sources,
        markdown=(
            "# Report\n\n"
            "Grounded claim. [cite:https://docs.aws.amazon.com/lambda/0]\n"
            "Orphan claim. [cite:https://orphan.example]\n"
        ),
    )
    critiqued = _critique_with_markers(
        "# Report\n\nGrounded claim.\nOrphan claim."
    )

    from deepsearch.agent import citation_liveness as cl_mod

    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda url, timeout_s=5.0, **kwargs: {"alive": True, "status_code": 200, "final_url": url},
    )

    captured: dict = {}

    def _capture(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return critiqued

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(fnode, "run_critique", side_effect=_capture):
            return await fnode.codeagent_finalize_node(state)

    asyncio.run(run())

    citations_kw = captured["kwargs"].get("citations")
    assert citations_kw is not None, (
        "run_critique must be called with citations= kwarg (S68 wiring)"
    )
    citations_kw = list(citations_kw)
    assert len(citations_kw) == 2
    # Both URLs are probed (S82 m05 removed the orphan short-circuit). Under the
    # universal-alive mock both resolve: the grounded URL → live (in sources),
    # the not-in-sources URL → live=True too (orphan-live, not hallucinated).
    by_url = {c.url: c for c in citations_kw}
    assert by_url["https://docs.aws.amazon.com/lambda/0"].live is True
    assert by_url["https://orphan.example/"].live is True  # canonical root URL


def test_protocol_citations_round_trip_live_field(report_dir, monkeypatch):
    """Persisted protocol carries `live` as bool|null on each citation."""

    import json

    state = _state()
    sources = [{"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0"}]
    synthesis = _synthesis_with_sources(
        sources=sources,
        markdown=(
            "# Report\n\n"
            "Grounded. [cite:https://docs.aws.amazon.com/lambda/0]\n"
            "Orphan. [cite:https://orphan.example]\n"
        ),
    )
    critiqued = _critique_with_markers("# Report\n\nGrounded.\nOrphan.")

    from deepsearch.agent import citation_liveness as cl_mod

    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda url, timeout_s=5.0, **kwargs: {"alive": True, "status_code": 200, "final_url": url},
    )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(fnode, "run_critique", return_value=critiqued):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    persisted = json.loads(Path(result["protocol_path"]).read_text())
    by_url = {c["url"]: c for c in persisted["citations"]}
    # `live` is True | False | null in the persisted JSON; never absent and
    # never a string.
    for c in persisted["citations"]:
        assert "live" in c, "every persisted citation carries a live field"
        assert c["live"] in (True, False, None)
    assert by_url["https://docs.aws.amazon.com/lambda/0"]["live"] is True
    # S82 m05: orphan URL is now probed and resolves → live=True (orphan-live).
    assert by_url["https://orphan.example/"]["live"] is True


def test_liveness_gate_unexpected_exception_falls_through(
    report_dir,
    monkeypatch,
    caplog,
):
    """Defense-in-depth: a contract-violating raise from the gate still completes.

    `check_citations_liveness`'s contract is no-raise; this test patches it
    to raise and asserts the finalize node logs at ERROR, leaves
    annotations un-live (live=None), preserves the public telemetry shape,
    and marks the internal Ledger envelope incomplete without leaking the
    exception text.
    """

    state = _state()
    sources = [{"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0"}]
    synthesis = _synthesis_with_sources(
        sources=sources,
        markdown="# Report\n\nClaim. [cite:https://docs.aws.amazon.com/lambda/0]",
    )
    synthesis.tool_outcomes = {
        "ledger_records": [],
        "ledger_records_truncated": 0,
    }
    critiqued = _critique_with_markers("# Report\n\nClaim.")

    async def _raising(*args, **kwargs):
        raise RuntimeError("credential=must-not-leak")

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(fnode, "run_critique", return_value=critiqued), patch.object(
            fnode, "check_citations_liveness", side_effect=_raising
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    lt = result["synthesis_telemetry"]["liveness_telemetry"]
    assert set(lt.keys()) == _LIVENESS_TELEMETRY_KEYS, (
        "telemetry shape preserved on fallthrough"
    )
    assert lt["checked"] == 1  # we know the citation count
    assert lt["unknown"] == 1
    assert lt["live"] == 0
    assert lt["dead"] == 0
    assert lt["orphan"] == 0
    outcomes = result["synthesis_telemetry"]["tool_outcomes"]
    assert outcomes["ledger_records"] == []
    assert outcomes["ledger_records_truncated"] == 1
    assert "credential" not in caplog.text


def test_extract_citations_runs_on_critique_skip_path(report_dir):
    """Extraction must run on the un-critiqued draft when critique fails.

    Per the plan-doc decision #518, if critique skips and the draft passes
    through unchanged, we still ship the stripped markdown and the populated
    citation list — the extractor input falls back to the synthesis draft.
    """

    state = _state()
    fallthrough = CritiqueResult(
        output_markdown=_good_synthesis_result().output_markdown,
        claim_count=0,
        gap_notes_inserted=0,
        runtime_seconds=0.5,
        token_usage={"input": 0, "output": 0, "total": 0},
        error=safe_error_from_parts("execution", "RuntimeError").to_text(),
    )
    # The synthesis result here carries `[cite:URL]` markers in its output.
    synthesis = SynthesisResult(
        output_markdown=(
            "# Report\n\nDraft claim with marker. [cite:https://drafted.example]"
        ),
        step_count=4,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=42.0,
        token_usage={"input": 1000, "output": 2000, "total": 3000},
        tool_call_summary={"total_calls": 7, "by_tool": {"web_search": 3}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=[
            {"url": "https://drafted.example", "title": "Drafted", "snippet": "x"}
        ],
        error=None,
    )
    # Re-bind the fallthrough output to the synthesis output so the seam
    # treats it as critique-skipped-with-passthrough.
    fallthrough.output_markdown = synthesis.output_markdown

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=fallthrough,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.VALIDATION_FAILED.value
    assert result["delivery_quality"] == "failed"
    assert result["synthesis_telemetry"]["critique_skipped"] is True
    ct = result["synthesis_telemetry"]["citation_telemetry"]
    assert ct["citations_extracted"] == 1
    persisted = Path(result["report_path"]).read_text()
    assert "[cite:" not in persisted


# ---------------------------------------------------------------------------
# S71 Path A — sources_collected persistence on MissionProtocolComplete.
#
# The runner-side accumulator (``SynthesisResult.sources_collected``) is the
# canonical retrieval-payload shape produced by web_search / source_fetch /
# url_liveness. The finalize node must thread it onto the persisted protocol
# so the FActScore harness has substrate to score against (closes S70 m02
# substrate blocker (2): empty sources_collected[] short-circuits PD4 and
# every probe tallies as unknown).
# ---------------------------------------------------------------------------


def test_protocol_sources_collected_populated_when_retrieval_occurred(report_dir):
    """Multi-source synthesis run → persisted protocol carries every URL."""

    import json

    state = _state()
    sources = [
        {
            "url": "https://docs.aws.amazon.com/lambda/0",
            "title": "Lambda Docs",
            "snippet": "scaling overview",
            "score": 0.9,
            "authority_tier": "high",
        },
        {
            "url": "https://aws.amazon.com/blogs/lambda-tips",
            "title": "Lambda Tips",
            "snippet": "blog post",
            "score": 0.5,
            "authority_tier": "medium",
            "alive": True,
        },
        {
            "url": "https://stackoverflow.com/q/123",
            "title": "SO question",
            "snippet": "discussion",
            "score": 0.2,
            "authority_tier": "low",
        },
    ]
    synthesis = _synthesis_with_sources(
        sources=sources,
        markdown="# Report\n\nGrounded body.",
    )
    critiqued = _critique_with_markers("# Report\n\nGrounded body.")

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value
    assert "protocol_path" in result

    persisted_protocol = json.loads(Path(result["protocol_path"]).read_text())
    assert "sources_collected" in persisted_protocol
    persisted_sources = persisted_protocol["sources_collected"]
    assert len(persisted_sources) == 3
    persisted_urls = {entry["url"] for entry in persisted_sources}
    assert persisted_urls == {entry["url"] for entry in sources}

    # The runtime SourceRecord shape must round-trip verbatim — no schema
    # remap. Each entry carries the keys agent.tools.web_search produced.
    by_url = {entry["url"]: entry for entry in persisted_sources}
    high_tier = by_url["https://docs.aws.amazon.com/lambda/0"]
    assert high_tier["title"] == "Lambda Docs"
    assert high_tier["authority_tier"] == "high"
    assert high_tier["score"] == 0.9

    # The ``alive`` field (populated by url_liveness calls) round-trips when
    # present and is absent when not — runtime polymorphism preserved.
    assert by_url["https://aws.amazon.com/blogs/lambda-tips"]["alive"] is True
    assert "alive" not in by_url["https://stackoverflow.com/q/123"]


def test_protocol_sources_collected_empty_list_when_no_retrieval(report_dir):
    """Cold-cache synthesis (no tool calls) → field is empty list, not absent.

    Distinguishing absent-vs-empty matters for the FActScore harness — PD4
    classifies probes by sources_collected presence, and dropping the field
    entirely would change the harness branch.
    """

    import json

    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[],  # No tools were invoked
        markdown="# Report\n\nUngrounded body.",
    )
    critiqued = _critique_with_markers("# Report\n\nUngrounded body.")

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    assert result["mission_outcome"] == MissionOutcome.COMPLETE.value

    persisted_protocol = json.loads(Path(result["protocol_path"]).read_text())
    assert "sources_collected" in persisted_protocol, (
        "sources_collected must be present (even as []) in the persisted "
        "protocol so FActScore PD4 can distinguish 'no retrieval' from "
        "'field dropped'."
    )
    assert persisted_protocol["sources_collected"] == []


def test_protocol_sources_collected_independent_of_state_sources_found(report_dir):
    """The persisted field must come from synthesis_result.sources_collected,
    NOT from state['sources_found']. The latter is a legacy LangGraph-stage
    surface; the former is the live-tool runner-side accumulator. They can
    diverge — sources_found may be stale from the old graph path while
    sources_collected reflects what the CodeAgent actually retrieved.
    """

    import json

    # state[sources_found] holds a single legacy URL; the synthesis run
    # collected two different URLs via the live tools.
    state = _state(
        sources=[{"url": "https://legacy.example/stale", "title": "Stale"}]
    )
    runner_sources = [
        {
            "url": "https://docs.aws.amazon.com/lambda/A",
            "title": "Lambda A",
            "snippet": "a",
            "score": 0.8,
            "authority_tier": "high",
        },
        {
            "url": "https://docs.aws.amazon.com/lambda/B",
            "title": "Lambda B",
            "snippet": "b",
            "score": 0.7,
            "authority_tier": "high",
        },
    ]
    synthesis = _synthesis_with_sources(
        sources=runner_sources,
        markdown="# Report\n\nBody.",
    )
    critiqued = _critique_with_markers("# Report\n\nBody.")

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=synthesis,
        ), patch.object(
            fnode,
            "run_critique",
            return_value=critiqued,
        ):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    persisted_protocol = json.loads(Path(result["protocol_path"]).read_text())
    persisted_urls = {entry["url"] for entry in persisted_protocol["sources_collected"]}

    assert persisted_urls == {
        "https://docs.aws.amazon.com/lambda/A",
        "https://docs.aws.amazon.com/lambda/B",
    }, (
        "protocol.sources_collected must come from the runner accumulator, "
        "not from state['sources_found']. Anything else loses the URL "
        "deduplication and shape that the FActScore harness expects."
    )
    assert "https://legacy.example/stale" not in persisted_urls


# ---------------------------------------------------------------------------
# S72 PD22 — ReportMetadata builder + at-a-glance markdown preamble.
#
# Locks the run-integrity surface that replaces the legacy execution_metadata
# fields (sources_count, loops_executed, coverage, quality_gates_passed)
# scheduled for source-level deletion at s72-m02. The block is single-source-
# of-truth: the markdown preamble and protocol.report_metadata both derive
# from the same SynthesisResult + synthesis_telemetry signals so a header /
# protocol drift is structurally impossible.
# ---------------------------------------------------------------------------


def test_build_report_metadata_populates_at_a_glance_fields():
    """The 4 PD22 header fields read off SynthesisResult + citation_telemetry."""

    from deepsearch.report.citation import CitationAnnotation

    synthesis = SynthesisResult(
        output_markdown="ignored",
        step_count=12,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=143.7,
        token_usage={"input": 1000, "output": 2000, "total": 3000},
        tool_call_summary={"total_calls": 18, "by_tool": {"web_search": 6, "source_fetch": 6}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        sources_collected=[
            {"url": "https://a.example/x", "title": "A", "snippet": "1"},
            {"url": "https://b.example/y", "title": "B", "snippet": "2"},
            {"url": "https://c.example/z", "title": "C", "snippet": "3"},
        ],
        error=None,
    )
    annotations = [
        CitationAnnotation(
            type="url_citation",
            url="https://a.example/x",
            title="A",
            start_index=0,
            end_index=5,
            live=True,
        ),
        CitationAnnotation(
            type="url_citation",
            url="https://b.example/y",
            title="B",
            start_index=6,
            end_index=11,
            live=True,
        ),
    ]
    synthesis_telemetry = {
        "citation_telemetry": {
            "citations_extracted": 2,
            "orphan_marker_count": 0,
            "distinct_urls_cited": 2,
            "runtime_seconds": 0.001,
        },
        "liveness_telemetry": {"checked": 2, "live": 2, "dead": 0, "unknown": 0},
        "critique_telemetry": {"claim_count": 2, "gap_notes_inserted": 0, "runtime_seconds": 1.0},
    }
    telemetry = {"runtime_identity": {"build_hash": "abcdef0123456789"}}

    meta = fnode._build_report_metadata(
        synthesis_result=synthesis,
        synthesis_telemetry=synthesis_telemetry,
        citation_annotations=annotations,
        telemetry=telemetry,
    )

    assert meta.sources_collected == 3
    assert meta.references == 2  # post-rebase distinct live protocol URLs
    assert meta.agent_steps_used == 12
    # max == DEFAULT_MAX_STEPS so a future bump in the runner constant
    # propagates here automatically; the at-a-glance "12 / 40" gut-check
    # always reflects the runner's actual ceiling.
    from deepsearch.agent.codeagent_runner import DEFAULT_MAX_STEPS

    assert meta.agent_steps_max == DEFAULT_MAX_STEPS
    assert meta.runtime_seconds == 143.7
    # Forensic block carries everything that needed context per PD22.
    assert meta.forensic["build_hash"] == "abcdef0123456789"
    assert meta.forensic["distinct_domains"] == 3
    assert meta.forensic["self_terminated"] is True
    assert meta.forensic["hit_max_steps"] is False
    assert meta.forensic["liveness_telemetry"]["live"] == 2
    assert meta.forensic["citation_telemetry"]["distinct_urls_cited"] == 2
    assert meta.forensic["token_usage"]["total"] == 3000
    assert meta.forensic["tool_call_summary"]["total_calls"] == 18


def test_build_report_metadata_uses_post_rebase_live_annotations():
    """References count distinct live annotations that actually persist."""

    from deepsearch.report.citation import CitationAnnotation

    synthesis = SynthesisResult(
        output_markdown="",
        step_count=0,
        self_terminated=False,
        hit_max_steps=False,
        runtime_seconds=0.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        tool_call_summary={"total_calls": 0, "by_tool": {}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        sources_collected=[],
        error=None,
    )
    annotations = [
        CitationAnnotation(
            type="url_citation",
            url="https://a/",
            title=None,
            start_index=0,
            end_index=1,
            live=True,
        ),
        CitationAnnotation(
            type="url_citation",
            url="https://a/",  # duplicate URL — distinct count is 1
            title=None,
            start_index=2,
            end_index=3,
            live=True,
        ),
        CitationAnnotation(
            type="url_citation",
            url="https://b/",
            title=None,
            start_index=4,
            end_index=5,
            live=True,
        ),
    ]

    meta = fnode._build_report_metadata(
        synthesis_result=synthesis,
        synthesis_telemetry={},
        citation_annotations=annotations,
        telemetry={},
    )
    assert meta.references == 2


def test_render_report_metadata_preamble_has_four_at_a_glance_fields():
    """The preamble is exactly the PD22 4-line gut-check shape — no decoration."""

    from deepsearch.report.protocol import ReportMetadata

    meta = ReportMetadata(
        sources_collected=47,
        references=23,
        agent_steps_used=18,
        agent_steps_max=25,
        runtime_seconds=142.4,
        forensic={"build_hash": "deadbeef", "distinct_domains": 12},
    )

    preamble = fnode._render_report_metadata_preamble(meta)

    # PD22 — exactly 4 plain lines, bold-key + plain-value, no heading,
    # no separator, no bullet markers.
    assert "## Run Integrity" not in preamble
    assert "---" not in preamble
    assert "- **" not in preamble
    assert "**Sources collected:** 47" in preamble
    assert "**Verified references:** 23" in preamble  # S82 m05: live-only header
    assert "**Agent steps:** 18 / 25" in preamble
    # Sub-minute runtime renders as plain seconds.
    assert "**Runtime:** 142.4s" in preamble or "**Runtime:** 2m 22s" in preamble
    # Forensic-only fields must NOT leak into the at-a-glance header.
    assert "deadbeef" not in preamble
    assert "distinct_domains" not in preamble
    assert "build_hash" not in preamble


def test_render_report_metadata_preamble_formats_runtime_in_minutes_and_seconds():
    """Runtime ≥60s renders as ``Xm Ys`` per PD22 spec example (``7m 42s``)."""

    from deepsearch.report.protocol import ReportMetadata

    meta = ReportMetadata(
        sources_collected=1,
        references=1,
        agent_steps_used=1,
        agent_steps_max=25,
        runtime_seconds=625.1,
        forensic={},
    )

    preamble = fnode._render_report_metadata_preamble(meta)
    assert "**Runtime:** 10m 25s" in preamble
    assert "625.1s" not in preamble


def test_persisted_protocol_carries_report_metadata_block(report_dir, monkeypatch):
    """End-to-end: protocol.report_metadata persists with PD22 4 + forensic."""

    import json

    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[
            {"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0", "snippet": "x"},
            {"url": "https://docs.aws.amazon.com/lambda/1", "title": "L1", "snippet": "y"},
        ],
        markdown=(
            "# Report\n\n"
            "Claim A. [cite:https://docs.aws.amazon.com/lambda/0]\n"
            "Claim B. [cite:https://docs.aws.amazon.com/lambda/1]\n"
        ),
    )
    critiqued = _critique_with_markers("# Report\n\nClaim A.\nClaim B.")

    # S82 m05: the header `references` is now the VERIFIED-LIVE count
    # (liveness_telemetry.live). Stub the probe alive so both in-sources URLs
    # are live → references == 2 (otherwise the no-network probe fails them).
    from deepsearch.agent import citation_liveness as cl_mod

    monkeypatch.setattr(
        cl_mod,
        "url_liveness",
        lambda url, timeout_s=5.0, **kwargs: {"alive": True, "status_code": 200, "final_url": url},
    )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(fnode, "run_critique", return_value=critiqued):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    persisted = json.loads(Path(result["protocol_path"]).read_text())

    assert "report_metadata" in persisted, (
        "S72 PD22 report_metadata must persist on the saved protocol JSON; "
        "this is the single truth surface that supersedes execution_metadata."
    )
    rm = persisted["report_metadata"]
    assert rm["sources_collected"] == 2
    assert rm["references"] == 2
    assert rm["agent_steps_used"] == 4
    # S79 m01.d / PD96 — DEFAULT_MAX_STEPS bumped to 40.
    assert rm["agent_steps_max"] == 40
    assert rm["runtime_seconds"] == 42.0
    assert "forensic" in rm
    assert rm["forensic"]["schema_version"].startswith("s72-")
    assert rm["forensic"]["distinct_domains"] >= 1
    assert "liveness_telemetry" in rm["forensic"]
    assert "citation_telemetry" in rm["forensic"]


def test_persisted_markdown_carries_run_integrity_preamble(report_dir):
    """The 4-line preamble lands in the rendered markdown before the synthesis body."""

    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[{"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0", "snippet": "x"}],
        markdown="# Report\n\nClaim A. [cite:https://docs.aws.amazon.com/lambda/0]",
    )
    critiqued = _critique_with_markers("# Report\n\nClaim A.")

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(fnode, "run_critique", return_value=critiqued):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    persisted_md = Path(result["report_path"]).read_text()

    # PD22 — 4 plain lines, no heading. The preamble is the bold-key block.
    # S79 m01.d / PD96 — DEFAULT_MAX_STEPS bumped to 40.
    assert "## Run Integrity" not in persisted_md
    assert "**Sources collected:** 1" in persisted_md
    assert "**Agent steps:** 4 / 40" in persisted_md
    assert "**Runtime:** 42.0s" in persisted_md
    # Preamble must precede the synthesis body so the user sees the gut-check
    # before the executive summary.
    integrity_idx = persisted_md.index("**Sources collected:**")
    body_idx = persisted_md.index("Claim A.")
    assert integrity_idx < body_idx


# ---------------------------------------------------------------------------
# S80 m03 (PD108) — `_strip_leading_h1` parametrized cases.
#
# The strip helper deletes the agent-emitted leading H1 BEFORE preamble
# concatenation so the wrapper H1 (`# Research Report: <id> - <title>`)
# emitted by ``build_report_header`` is the one and only top-level heading
# in the rendered markdown. WRAPPER WINS — locked S79 review session
# 2026-05-02 per user direction.
# ---------------------------------------------------------------------------


class TestStripLeadingH1:
    """Parametrized cases pinning the strip semantics."""

    @pytest.mark.parametrize(
        "label,markdown_in,expected_out",
        [
            (
                "leading_h1_stripped",
                "# Agent Topic\n\nBody starts here.",
                "Body starts here.",
            ),
            (
                "leading_blank_lines_then_h1_stripped",
                # Blank lines preceding the H1 are also dropped.
                "\n\n# Agent Topic\n\nBody starts here.",
                "Body starts here.",
            ),
            (
                "blank_lines_after_h1_collapsed",
                "# Agent Topic\n\n\nBody.",
                "Body.",
            ),
            (
                "input_starts_with_h2_unchanged",
                "## Section\n\nBody.",
                "## Section\n\nBody.",
            ),
            (
                "h1_only_mid_body_unchanged",
                "## Section\n\n# Mid-Body H1\n\nMore body.",
                "## Section\n\n# Mid-Body H1\n\nMore body.",
            ),
            (
                "no_h1_unchanged",
                "Plain prose with no leading heading.",
                "Plain prose with no leading heading.",
            ),
            (
                "multiple_consecutive_leading_h1s_only_first_stripped",
                # Rare but defined — pinning the behavior so it cannot drift.
                "# First\n# Second\n\nBody.",
                "# Second\n\nBody.",
            ),
            (
                "h1_with_extra_whitespace_after_hash_stripped",
                # `#  Title` (two spaces after hash) — startswith('# ') is True.
                "#  Padded Title\n\nBody.",
                "Body.",
            ),
            (
                "indented_h1_stripped",
                # Markdown allows up to 3 spaces of indentation before a heading.
                "   # Indented H1\n\nBody.",
                "Body.",
            ),
            (
                "hash_without_space_unchanged",
                # `#Title` (no space after hash) is NOT a markdown H1.
                "#Title\n\nBody.",
                "#Title\n\nBody.",
            ),
            (
                "lone_hash_only_unchanged",
                # `#` alone (no trailing space + content) is not stripped.
                "#\nBody.",
                "#\nBody.",
            ),
            (
                "empty_string_unchanged",
                "",
                "",
            ),
            (
                "all_whitespace_unchanged",
                "\n\n   \n",
                "\n\n   \n",
            ),
        ],
    )
    def test_strip_helper_cases(self, label, markdown_in, expected_out) -> None:
        result = fnode._strip_leading_h1(markdown_in)
        assert result == expected_out, (
            f"[{label}] _strip_leading_h1 returned {result!r}; "
            f"expected {expected_out!r}"
        )


def test_persisted_markdown_has_only_one_top_level_h1(report_dir):
    """Integration — agent body H1 stripped, wrapper H1 is the only one.

    Synthesizes a fixture with a leading body H1, runs through the full
    finalize-node code path (mocked critique + synthesis), and asserts the
    rendered markdown has exactly ONE leading-position top-level `# ` line
    after the preamble (the wrapper). Pre-S80 the document carried both
    the wrapper H1 and the agent-emitted H1; post-S80 only the wrapper
    survives.
    """
    state = _state()
    synthesis = _synthesis_with_sources(
        sources=[{"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0", "snippet": "x"}],
        markdown=(
            "# Agent Topic Heading\n\n"
            "Claim with citation [cite:https://docs.aws.amazon.com/lambda/0]"
        ),
    )
    critiqued = _critique_with_markers(
        "# Agent Topic Heading\n\nClaim with citation"
    )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(fnode, "run_critique", return_value=critiqued):
            return await fnode.codeagent_finalize_node(state)

    result = asyncio.run(run())
    persisted_md = Path(result["report_path"]).read_text()

    # The wrapper H1 must be present — that's the canonical identity line.
    wrapper_h1_count = sum(
        1 for line in persisted_md.splitlines()
        if line.startswith("# Research Report:")
    )
    assert wrapper_h1_count == 1, (
        f"Wrapper H1 must appear exactly once in the rendered markdown; "
        f"found {wrapper_h1_count}. Persisted markdown head:\n"
        f"{persisted_md[:400]}"
    )

    # The agent body H1 must NOT survive — the strip helper deleted it
    # before preamble concatenation.
    assert "# Agent Topic Heading" not in persisted_md, (
        "Agent body H1 (`# Agent Topic Heading`) must be stripped by "
        "_strip_leading_h1 before preamble concatenation (S80 m03 PD108). "
        "Wrapper H1 must be the only top-level heading."
    )


def test_segment_anchor_failure_ships_report_and_downgrades_quality(report_dir):
    """A broken segment anchor costs the coordinates, not the report.

    S92 m06. This is the one surviving raise inside the rebase: the persisted
    artifact no longer contains the segment its spans were measured against, so
    no coordinate is trustworthy. The report body is still intact and worth
    reviewing, so it is persisted with zero citations, the loss is recorded in
    telemetry, and a 'clean' verdict is downgraded one tier rather than the
    mission being blocked with nothing.
    """

    from deepsearch.agent.citation_liveness import LivenessTelemetry
    from deepsearch.report.citation import CitationSpanRebaseError

    claim = "An anchored claim the artifact should still ship."
    url = "https://example.test/source"
    synthesis_markdown = f"# Report\n\n{claim} [cite:{url}]\n"
    stripped_markdown = synthesis_markdown.replace(f" [cite:{url}]", "")
    synthesis = _synthesis_with_sources(
        sources=[{"url": url, "title": "Source"}],
        markdown=synthesis_markdown,
    )

    async def keep_live(citations, _sources, budget_seconds=45.0, *, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=1,
            live=1,
            distinct_urls_checked=1,
        )

    def broken_anchor(**_kwargs):
        raise CitationSpanRebaseError(
            annotation_index=0,
            reason="persisted_segment_missing_from_persisted_markdown",
            match_count=0,
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_with_markers(stripped_markdown),
        ), patch.object(
            fnode, "check_citations_liveness", new=keep_live
        ), patch.object(
            fnode, "rebase_citation_annotations", side_effect=broken_anchor
        ), patch.object(
            fnode, "save_report"
        ) as save_report, patch.object(
            fnode, "save_protocol"
        ) as save_protocol:
            result = await fnode.codeagent_finalize_node(_state(mission_id="ANCHOR-BREAK"))
        save_report.assert_called_once()
        save_protocol.assert_called_once()
        return result, save_protocol.call_args[0][0]

    result, protocol = asyncio.run(run())

    # The report ships; only the coordinates are lost.
    assert result["mission_outcome"] != MissionOutcome.BLOCKED.value
    assert "report_path" in result
    assert protocol.citations == []
    assert protocol.report_metadata is not None
    assert protocol.report_metadata.references == 0
    rebase = result["synthesis_telemetry"]["citation_rebase"]
    assert rebase["anchor_error"] == "persisted_segment_missing_from_persisted_markdown"
    assert rebase["dropped_count"] == 1
    # Truthful verdict: never still 'clean' once every coordinate is gone.
    assert result["synthesis_telemetry"]["delivery_quality"] != "clean"
    assert protocol.report_metadata.forensic["citation_rebase"] == rebase
    assert protocol.report_metadata.forensic["quality_record"][
        "delivery_quality"
    ] == result["delivery_quality"]
    assert len(result["generation_errors"]) == 1
