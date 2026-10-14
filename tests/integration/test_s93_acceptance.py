"""S93 real-instance persistence contracts; no claims about LLM behavior."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.config import DeepSearchSettings
from deepsearch.eval.cost import collect_attempt_events, estimate_all_attempts
from deepsearch.llm.runtime import resolve_effective_model
from deepsearch.llm.usage import usage_event
from deepsearch.report.protocol import MissionProtocolComplete
from deepsearch.worker.writer import assert_result_identity, build_execution_metadata


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
@pytest.mark.parametrize("staging_failure", [False, True], ids=["disk", "inline-fallback"])
def test_provider_quality_and_paid_attempts_survive_exact_artifact_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    staging_failure: bool,
) -> None:
    """Warnings, missing usage and a disk fault must never erase a paid report."""
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    config = resolve_effective_model(DeepSearchSettings(
        llm_backend=provider, deepseek_model="deepseek-flash", openai_model="gpt-5.6-luna",
        deepseek_api_key="offline-key", openai_api_key="offline-key",
    )).public_metadata()

    def observed(stage: str, input_tokens: int, output_tokens: int) -> dict:
        return usage_event(
            {"model": config["requested_model"], "choices": [{"finish_reason": "stop"}],
             "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}},
            config=config, stage=stage, elapsed=1, started_at="2026-09-19T12:00:00+00:00",
        )

    compiler_failure = usage_event(
        None, config=config, stage="compiler", elapsed=1,
        started_at="2026-09-19T12:00:00+00:00", error=TimeoutError("sensitive request"),
    )
    compiler_events = [compiler_failure, observed("compiler", 20, 5)]
    url = "https://primary.example/evidence"
    claim = "The stored evidence supports preserving this report."
    draft = (
        f"# Acceptance\n\n{claim} [cite:{url}] [1].\n\n"
        "## Recommendations\nKeep the report and expose incomplete assessment [16].\n\n"
        f"## References\n[1] Primary evidence — {url}\n"
    )
    synthesis = SynthesisResult(
        output_markdown=draft, step_count=30, self_terminated=True, hit_max_steps=False,
        runtime_seconds=1, token_usage={"input": 100, "output": 20, "total": 120},
        token_usage_complete=True, tool_call_summary={"total_calls": 0, "by_tool": {}},
        authority_tier_distribution={"high": 1}, pool_size=0, step_trace=[],
        sources_collected=[
            {"url": url, "title": "Primary evidence", "body": claim,
             "content_quality": "readable_body", "body_source": "source_fetch",
             "body_truncated": False},
            {"url": "https://user:password@example.test/unsafe", "title": "Unsafe"},
        ],
        model_id=config["requested_model"], effective_config=config,
        provider_attempts=[observed("synthesis", 100, 20)],
    )
    assessment = {
        "version": 1, "status": "partial", "assessed": 1, "supported": 1,
        "contradicted": 0, "unassessed": 1, "declared_context": 0,
        "report_claim_count": None, "coverage_ratio": None, "accuracy_pass": None,
    }

    def recorded_critique(markdown: str, *_args: object, **_kwargs: object) -> CritiqueResult:
        return CritiqueResult(
            output_markdown=markdown, claim_count=2, gap_notes_inserted=0, runtime_seconds=1,
            token_usage={"input": 30, "output": 10, "total": 40}, token_usage_complete=True,
            assessment=assessment, provider_attempts=[observed("critique", 30, 10)],
        )

    async def live(citations: list, *_args: object, **_kwargs: object) -> tuple:
        for citation in citations:
            citation.live = True
        return citations, LivenessTelemetry(checked=len(citations), live=len(citations))

    state = {
        "mission_id": "S93-ACCEPTANCE", "mission_uuid": "93000008-0000-4000-8000-000000000008",
        "mission_title": "S93 acceptance", "mission_objectives": ["Preserve paid artifacts"],
        "mission_context": {"title": "S93 acceptance", "objective": "Preserve paid artifacts",
                            "deliverables": ["Markdown report", "Recommendations"]},
        "mission_contract": {"entity_extraction_decisions": [
            {"entity": "", "decision": "provider_usage", "reason": json.dumps(event)}
            for event in compiler_events
        ]},
        "sources_found": [], "telemetry": {},
    }
    if staging_failure:
        def fail_staging(*_args: object, **_kwargs: object) -> None:
            raise OSError("staging unavailable")

        monkeypatch.setattr(fnode, "save_protocol", fail_staging)
    with patch.object(fnode, "run_codeagent_synthesis", return_value=synthesis) as synth_call, \
            patch.object(fnode, "run_critique", side_effect=recorded_critique), \
            patch.object(fnode, "check_citations_liveness", new=live):
        result = asyncio.run(fnode.codeagent_finalize_node(state))

    assert synth_call.call_count == 1  # A format label is not a paid recovery target.
    assert result["mission_outcome"] == "complete"
    assert result["delivery_quality"] != "clean"
    report = result["report_markdown"]
    assert claim in report and "[16]" in report
    assert Path(result["report_path"]).read_text() == report
    if staging_failure:
        assert "protocol_path" not in result and result["generation_errors"]
    else:
        assert json.loads(Path(result["protocol_path"]).read_text()) == result["protocol_payload"]
    protocol = MissionProtocolComplete.model_validate_json(json.dumps(result["protocol_payload"]))
    assert len(protocol.sources_collected) == len(protocol.citations) == 1
    assert claim in report[protocol.citations[0].start_index:protocol.citations[0].end_index]
    telemetry = result["synthesis_telemetry"]
    assert telemetry["result_identity"]["report_sha256"] == hashlib.sha256(
        report.encode()
    ).hexdigest()
    assert_result_identity(
        row_mission_id=state["mission_id"], row_title=state["mission_title"],
        result_markdown=report, result_protocol=protocol.model_dump(mode="json"),
        synthesis_telemetry=telemetry, markdown_from_artifact=True,
    )
    metadata = json.loads(json.dumps(build_execution_metadata({**state, **result}, 3).to_dict()))
    assert metadata["runtime_identity"]["model"] == config["requested_model"]
    assert metadata["runtime_identity"]["llm_backend"] == provider
    assert metadata["total_tokens"] == 160  # Legacy synthesis + critique boundary remains stable.
    assert metadata["synthesis_telemetry"] == {
        **telemetry, "compiler_provider_attempts": compiler_events,
    }
    assert protocol.report_metadata is not None
    forensic = protocol.report_metadata.forensic
    assert forensic["quality_record"] == metadata["quality_record"] == telemetry["quality_record"]
    assert forensic["critique_telemetry"]["assessment"] == assessment
    assert forensic["source_projection"]["reasons"] == {"invalid_source_record": 1}
    assert telemetry["deliverable_coverage"]["missing"] == []
    assert not telemetry["citation_consistency"]["consistent"]
    assert forensic["quality_record"]["claim_assessment"]["accuracy_pass"] is None
    quality = forensic["quality_record"]
    assert quality["schema_version"] == 5
    assert quality["verification"]["scope"] == "sampled"
    assert quality["verification"]["report_wide_accuracy"] == "unknown"
    assert quality["evidence_dependencies"]["readable_cited_urls"] == 1
    assert quality["citation_surface"]["actionable"] is True  # unresolved reader [16]
    assert "critique_unassessed_claims_1" in result["quality_gate_failures"]
    assert "critique_assessment_partial" not in result["quality_gate_failures"]
    accounting = forensic["attempt_accounting"]
    assert accounting["model_accounting"]["observed"]["total"] == 185
    assert not accounting["model_accounting"]["usage_complete"]
    assert accounting["model_accounting"]["reconciliation_delta"] == 0
    events = collect_attempt_events(metadata)
    assert len(events) == 4  # The compiler events appear twice in metadata; never bill twice.
    assert estimate_all_attempts(events)["model_usd_max"] is None
    assert "password" not in json.dumps(metadata)
    assert "sensitive request" not in json.dumps(metadata)
