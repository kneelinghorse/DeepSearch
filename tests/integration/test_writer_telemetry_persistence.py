"""S68.5 HD4 — round-trip integration test for synthesis_telemetry persistence.

Locks the worker→TraceLab serialisation path that S64+S65+S66+S67+S68 all
needed but never had. Exercises the full chain:

    AgentState (with populated synthesis_telemetry) →
    build_execution_metadata(state, duration) →
    ExecutionMetadata.to_dict() →
    json.dumps/json.loads (JSONB-safe round-trip)

If a future change accidentally drops the ``synthesis_telemetry`` field
from :class:`ExecutionMetadata`, stops reading ``state["synthesis_telemetry"]``
in :func:`build_execution_metadata`, or breaks the HD3 runtime_identity.model
truthfulness override, these tests fail loudly at CI rather than at the next
free TraceLab dispatch — closing the DEFERRED-DISPATCH-MASKS-PERSISTENCE-DEFECTS
learning from session ``PS-2026-04-28-012``.

The tests do not require any database, network, or LLM — the round-trip is
pure Python: dataclass → dict → JSON string → dict.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from deepsearch.worker.writer import ExecutionMetadata, build_execution_metadata

# ---------------------------------------------------------------------------
# Realistic AgentState fixture — mirrors the shape codeagent_finalize_node
# writes onto state at S68 close (10 PD7 liveness fields + nested critique +
# citation envelopes + S64 root telemetry fields).
# ---------------------------------------------------------------------------


def _liveness_telemetry_pd7() -> Dict[str, Any]:
    """The 10 PD7 fields of synthesis_telemetry["liveness_telemetry"]."""
    return {
        "checked": 75,
        "live": 72,
        "dead": 0,
        "orphan": 3,
        "unknown": 0,
        "hallucinated_url_count": 3,
        "distinct_urls_checked": 75,
        "wall_clock_ms": 4280,
        "cache_hits": 0,
        "budget_exhausted": False,
    }


def _critique_telemetry() -> Dict[str, Any]:
    """The S66 critique_telemetry envelope shape."""
    return {
        "claim_count": 18,
        "gap_notes_inserted": 1,
        "runtime_seconds": 24.7,
        "token_usage": {"input": 12450, "output": 3120, "total": 15570},
        "error": None,
    }


def _citation_telemetry() -> Dict[str, Any]:
    """The S67 citation_telemetry envelope shape (4 fields)."""
    return {
        "citations_extracted": 75,
        "orphan_marker_count": 3,
        "distinct_urls_cited": 75,
        "runtime_seconds": 0.012,
    }


def _synthesis_telemetry_full() -> Dict[str, Any]:
    """A populated synthesis_telemetry dict matching the post-S68 shape."""
    return {
        "engine": "codeagent",
        "model_id": "deepseek-v4-pro",
        "framework": "smolagents 1.24.0 CodeAgent",
        "step_count": 18,
        "self_terminated": True,
        "hit_max_steps": False,
        "runtime_seconds": 612.4,
        "token_usage": {"input": 184230, "output": 14820, "total": 199050},
        "tool_call_summary": {
            "web_search": 22,
            "source_fetch": 30,
            "url_liveness": 0,
            "scratchpad_write": 4,
            "scratchpad_read": 2,
        },
        "synthesis_error": None,
        "critique_telemetry": _critique_telemetry(),
        "citation_telemetry": _citation_telemetry(),
        "liveness_telemetry": _liveness_telemetry_pd7(),
        "critique_skipped": False,
    }


def _legacy_telemetry() -> Dict[str, Any]:
    """The legacy state["telemetry"] envelope (LangGraph counters + identity).

    HD3 contract: the model field here is the legacy LLM_BACKEND env probe
    value. When synthesis_telemetry["model_id"] is populated (the smolagents
    binding self-reports), the build path overrides this with the truthful
    value. Keep both populated in the realistic fixture so we can assert the
    override fires.
    """
    return {
        "token_usage": {"total_tokens": 199050},
        "runtime_identity": {
            "build_hash": "550f1fc0808d6e2af0f9ab767ae35826bfc47599e701132f628539ae64c6bdd7",
            "execution_path": "tracelab",
            "llm_backend": "gemini",
            "model": "gemini-2.5-pro",
        },
        "config_fingerprint": "6e4c3a2146d6528183de3df8d13fdca2",
        "routing_trace": [
            {
                "judge_decision": "finalize",
                "judge_reason": "codeagent_complete",
                "next_action": "complete",
                "reason": "codeagent_complete",
            }
        ],
        "finding_counts": {"raw_findings": 0, "unique_concepts": 0},
        "source_counts": {"sources_before_filtering": 0, "sources_presented_to_llm": 0},
        "final_outcome": {
            "outcome": "complete",
            "routing_reason": "codeagent_complete",
        },
        "telemetry_artifact_path": "checkpoints/telemetry/mission-S685.jsonl",
    }


def _realistic_state() -> Dict[str, Any]:
    """A full AgentState snapshot as the worker reads it post-codeagent_finalize."""
    return {
        "mission_id": "OODS-FIGMA-HOST-01-S68-5-TELEMETRY-FIX",
        "mission_outcome": "complete",
        "report_path": "checkpoints/reports/OODS-FIGMA-HOST-01-S68-5-TELEMETRY-FIX.md",
        "protocol_path": "checkpoints/protocols/OODS-FIGMA-HOST-01-S68-5-TELEMETRY-FIX.json",
        "research_loop_count": 0,
        "sources_found": [],
        "findings": [],
        "synthesis_telemetry": _synthesis_telemetry_full(),
        "telemetry": _legacy_telemetry(),
        "depth_config": {
            "min_total_findings": 1,
            "min_total_sources": 1,
            "min_unique_domains_gate": 1,
        },
    }


# ---------------------------------------------------------------------------
# 1. HD2 — synthesis_telemetry survives ExecutionMetadata.to_dict round-trip.
# ---------------------------------------------------------------------------


def test_synthesis_telemetry_persists_through_to_dict() -> None:
    state = _realistic_state()
    metadata = build_execution_metadata(state, 612.4)

    assert metadata.synthesis_telemetry, (
        "synthesis_telemetry envelope was dropped between AgentState and "
        "ExecutionMetadata; HD2 read in build_execution_metadata is missing."
    )

    payload = metadata.to_dict()

    assert "synthesis_telemetry" in payload, (
        "ExecutionMetadata.to_dict() omitted synthesis_telemetry; the HD2 "
        "conditional clause is missing or the field is incorrectly typed."
    )
    assert payload["synthesis_telemetry"]["engine"] == "codeagent"
    assert payload["synthesis_telemetry"]["model_id"] == "deepseek-v4-pro"
    assert payload["synthesis_telemetry"]["framework"] == "smolagents 1.24.0 CodeAgent"
    assert payload["synthesis_telemetry"]["step_count"] == 18


def test_all_attempt_token_accounting_overrides_selected_pass_totals() -> None:
    """Rejected recovery work is paid work and must be counted exactly once."""

    state = _realistic_state()
    state["synthesis_telemetry"]["recovery"] = {
        "selected_artifact": "first_pass",
        "attempt_accounting": {
            "schema_version": 1,
            "synthesis_passes": 2,
            "critique_passes": 2,
            "token_usage": {"input": 300_000, "output": 33_333, "total": 333_333},
            "token_usage_complete": False,
        },
    }

    metadata = build_execution_metadata(state, 1_415.91)

    assert metadata.total_tokens == 333_333
    assert metadata.duration_seconds == 1_415.91
    assert (
        metadata.synthesis_telemetry["recovery"]["attempt_accounting"][
            "token_usage_complete"
        ]
        is False
    )


def test_nested_critique_citation_liveness_telemetry_survive_round_trip() -> None:
    """Nested envelopes (critique/citation/liveness) survive verbatim."""

    state = _realistic_state()
    metadata = build_execution_metadata(state, 612.4)
    payload = metadata.to_dict()
    synth = payload["synthesis_telemetry"]

    # S66 critique envelope
    assert synth["critique_telemetry"]["claim_count"] == 18
    assert synth["critique_telemetry"]["gap_notes_inserted"] == 1
    assert synth["critique_telemetry"]["error"] is None
    assert synth["critique_skipped"] is False

    # S67 citation envelope (4 fields)
    citation = synth["citation_telemetry"]
    assert set(citation.keys()) == {
        "citations_extracted",
        "orphan_marker_count",
        "distinct_urls_cited",
        "runtime_seconds",
    }
    assert citation["citations_extracted"] == 75
    assert citation["orphan_marker_count"] == 3

    # S68 liveness envelope (PD7's 10 fields — load-bearing for S73 paid smoke)
    liveness = synth["liveness_telemetry"]
    expected_pd7_fields = {
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
    assert set(liveness.keys()) == expected_pd7_fields, (
        f"liveness_telemetry must carry all 10 PD7 fields; missing: "
        f"{expected_pd7_fields - set(liveness.keys())}"
    )
    assert liveness["hallucinated_url_count"] == 3, (
        "hallucinated_url_count is the load-bearing metric for the S73 "
        "DRBench 3–13% URL hallucination comparison; it must round-trip."
    )
    assert liveness["budget_exhausted"] is False


# ---------------------------------------------------------------------------
# 2. HD3 — runtime_identity.model truthfulness override.
# ---------------------------------------------------------------------------


def test_runtime_identity_model_overridden_from_synthesis_telemetry() -> None:
    """When synthesis_telemetry["model_id"] is set, runtime_identity.model
    reflects it (overriding the legacy LLM_BACKEND env probe).
    """

    state = _realistic_state()
    metadata = build_execution_metadata(state, 612.4)

    assert metadata.runtime_identity["model"] == "deepseek-v4-pro", (
        "HD3 truthfulness override failed: runtime_identity.model still "
        "reports the legacy probe value rather than the smolagents binding."
    )
    assert metadata.runtime_identity["llm_backend"] == "deepseek", (
        "HD3 derives llm_backend from the model_id prefix when the agent "
        "self-reports its binding."
    )
    # Non-overridden fields stay verbatim from the legacy probe
    assert metadata.runtime_identity["execution_path"] == "tracelab"
    assert metadata.runtime_identity["build_hash"].startswith("550f1fc0808d")


def test_runtime_identity_unchanged_when_synthesis_telemetry_absent() -> None:
    """Backward compatibility: legacy missions without synthesis_telemetry
    keep whatever the legacy probe wrote into runtime_identity.
    """

    state = {
        "research_loop_count": 2,
        "telemetry": {
            "token_usage": {"total_tokens": 12000},
            "runtime_identity": {
                "build_hash": "abc123",
                "execution_path": "tracelab",
                "llm_backend": "gemini",
                "model": "gemini-2.5-pro",
            },
        },
        "sources_found": [{"url": "https://example.com"}],
        "findings": ["F1", "F2"],
    }

    metadata = build_execution_metadata(state, 30.0)

    assert metadata.runtime_identity["model"] == "gemini-2.5-pro"
    assert metadata.runtime_identity["llm_backend"] == "gemini"
    assert metadata.synthesis_telemetry == {}


def test_runtime_identity_unchanged_when_model_id_missing() -> None:
    """If synthesis_telemetry is populated but lacks model_id, the legacy
    probe value stays — HD3 only fires on a positive self-report.
    """

    state = _realistic_state()
    state["synthesis_telemetry"] = {
        "engine": "codeagent",
        "synthesis_skipped": True,
        "synthesis_skip_reason": "codeagent_runner_error",
        "synthesis_error": "RuntimeError(...)",
    }

    metadata = build_execution_metadata(state, 612.4)

    assert metadata.runtime_identity["model"] == "gemini-2.5-pro"
    assert metadata.runtime_identity["llm_backend"] == "gemini"
    # The truncated synthesis_telemetry still survives (HD2 is independent)
    assert metadata.synthesis_telemetry["engine"] == "codeagent"
    assert metadata.synthesis_telemetry["synthesis_skipped"] is True


# ---------------------------------------------------------------------------
# 3. JSONB-safety — the full payload survives json.dumps/json.loads.
# ---------------------------------------------------------------------------


def test_full_metadata_round_trips_through_json_dumps() -> None:
    """The serialised payload must be JSONB-safe — no datetime, set, or
    other non-JSON types leak through. Mirrors the path
    ResultWriter._serialize_jsonb takes for the missions.execution_metadata
    column.
    """

    state = _realistic_state()
    metadata = build_execution_metadata(state, 612.4)
    payload = metadata.to_dict()

    encoded = json.dumps(payload, default=str, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert decoded["synthesis_telemetry"]["engine"] == "codeagent"
    assert decoded["synthesis_telemetry"]["liveness_telemetry"]["checked"] == 75
    assert (
        decoded["synthesis_telemetry"]["liveness_telemetry"]["hallucinated_url_count"]
        == 3
    )
    assert decoded["runtime_identity"]["model"] == "deepseek-v4-pro"


# ---------------------------------------------------------------------------
# 4. Backward compatibility — empty synthesis_telemetry is omitted from the
#    payload (matches the existing if-self.X pattern for optional dicts).
# ---------------------------------------------------------------------------


def test_to_dict_omits_synthesis_telemetry_when_empty() -> None:
    metadata = ExecutionMetadata(
        total_tokens=4500,
        duration_seconds=22.0,
    )

    payload = metadata.to_dict()

    assert "synthesis_telemetry" not in payload, (
        "Empty synthesis_telemetry must be omitted from to_dict() to match "
        "the existing if-self.X pattern for optional dict fields."
    )


def test_legacy_state_without_synthesis_telemetry_emits_no_key() -> None:
    """A pre-S64 state shape (no synthesis_telemetry on AgentState) flows
    through build_execution_metadata → to_dict() with no synthesis_telemetry
    key in the output.
    """

    state = {
        "research_loop_count": 3,
        "telemetry": {"token_usage": {"total_tokens": 8000}},
        "sources_found": [
            {"url": "https://a.example.com/1"},
            {"url": "https://b.example.org/2"},
        ],
        "findings": ["F1", "F2", "F3"],
    }

    metadata = build_execution_metadata(state, 45.0)
    payload = metadata.to_dict()

    assert metadata.synthesis_telemetry == {}
    assert "synthesis_telemetry" not in payload
