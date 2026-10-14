"""Sprint 87 m02 retrieval-truth and bounded-grounding telemetry guards."""

from __future__ import annotations

from deepsearch.agent.codeagent_finalize_node import (
    _build_synthesis_telemetry,
    _critique_telemetry_dict,
)
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult


def _synthesis_result() -> SynthesisResult:
    return SynthesisResult(
        output_markdown="# Report",
        step_count=2,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=1.5,
        token_usage={"input": 10, "output": 20, "total": 30},
        token_usage_complete=True,
        tool_call_summary={"total_calls": 3, "by_tool": {}},
        authority_tier_distribution={"high": 1, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        sources_collected=[{"url": "https://example.com/source"}],
        sources_discovered=[{"url": "https://example.com/source"}],
        sources_fetched=[{"url": "https://example.com/source", "body": "evidence"}],
        fetch_attempts=[{"url": "https://example.com/source", "status": "ok"}],
        probes=[{"url": "https://example.com/probe", "alive": False}],
        retrieval_state={
            "discovered": 1,
            "fetched_readable": 1,
            "fetch_attempts": 1,
            "probes": 1,
            "grounding_sources": 1,
        },
    )


def test_finalize_telemetry_keeps_retrieval_states_separate() -> None:
    telemetry = _build_synthesis_telemetry(_synthesis_result())

    assert telemetry["sources_collected"] == 1
    assert telemetry["token_usage_complete"] is True
    assert telemetry["retrieval_state"] == {
        "discovered": 1,
        "fetched_readable": 1,
        "fetch_attempts": 1,
        "probes": 1,
        "grounding_sources": 1,
    }


def test_finalize_telemetry_persists_exact_critique_grounding_counts() -> None:
    result = CritiqueResult(
        output_markdown="# Report",
        claim_count=2,
        gap_notes_inserted=1,
        runtime_seconds=0.5,
        grounding_telemetry={
            "sources_available": 9,
            "sources_included": 4,
            "sources_dropped": 5,
            "sources_truncated": 2,
            "cited_live_included": 3,
            "context_chars": 120_000,
            "context_budget_chars": 450_000,
        },
    )

    telemetry = _critique_telemetry_dict(result)

    assert telemetry["grounding"]["sources_included"] == 4
    assert telemetry["grounding"]["sources_dropped"] == 5
    assert telemetry["grounding"]["sources_truncated"] == 2
    assert telemetry["grounding"]["cited_live_included"] == 3
    assert telemetry["token_usage_complete"] is False
