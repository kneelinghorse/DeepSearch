"""Unit tests for execution metadata degradation fields (B15.3)."""

from __future__ import annotations

from deepsearch.worker.writer import build_execution_metadata


def test_build_execution_metadata_ignores_dead_degradation_state() -> None:
    """S84 m04: degraded_mode/degradation_policy were DELETED — their only
    writer was the empty_state default, so they persisted a constant
    not-degraded verdict for retired machinery on every mission. State keys
    are ignored; the payload carries neither. The live tier is
    final_outcome.delivery_quality."""

    state = {
        "research_loop_count": 4,
        "telemetry": {"token_usage": {"total_tokens": 9000}},
        "sources_found": [
            {"url": "https://a.example.com/1"},
            {"url": "https://b.example.org/2"},
            {"url": "https://c.example.net/3"},
        ],
        "findings": ["F1", "F2", "F3", "F4", "F5"],
        "degraded_mode": True,
        "degradation_policy": "skip_web_search",
        "depth_config": {
            "min_total_findings": 5,
            "min_total_sources": 3,
            "min_unique_domains_gate": 2,
        },
    }

    metadata = build_execution_metadata(state, 75.0)
    payload = metadata.to_dict()

    assert not hasattr(metadata, "degraded_mode")
    assert not hasattr(metadata, "degradation_policy")
    assert "degraded_mode" not in payload
    assert "degradation_policy" not in payload
    # S75 m01: quality_gates_passed stripped from ExecutionMetadata
    assert "quality_gates_passed" not in payload


def test_build_execution_metadata_non_degraded_state_defaults() -> None:
    """S84 m04: no degradation keys in the payload regardless of state."""

    state = {
        "research_loop_count": 2,
        "telemetry": {"token_usage": {"total_tokens": 3000}},
        "sources_found": [
            {"url": "https://docs.python.org/3/"},
            {"url": "https://developer.mozilla.org/en-US/docs/Web"},
            {"url": "https://kubernetes.io/docs/"},
        ],
        "findings": ["F1", "F2", "F3", "F4", "F5"],
        "depth_config": {
            "min_total_findings": 5,
            "min_total_sources": 3,
            "min_unique_domains_gate": 2,
        },
    }

    metadata = build_execution_metadata(state, 25.0)
    payload = metadata.to_dict()

    assert "degraded_mode" not in payload
    assert "degradation_policy" not in payload
    # S75 m01: quality_gates_passed stripped from ExecutionMetadata
    assert "quality_gates_passed" not in payload


def test_build_execution_metadata_includes_contract_reference() -> None:
    """Compiled contract identity should flow into worker execution metadata."""

    state = {
        "research_loop_count": 1,
        "telemetry": {"token_usage": {"total_tokens": 1200}},
        "sources_found": [{"url": "https://example.com"}],
        "mission_contract_id": "abc123contract",
        "mission_contract_path": "checkpoints/contracts/mission-test-abc123contract.json",
        "depth_config": {
            "min_total_findings": 1,
            "min_total_sources": 1,
            "min_unique_domains_gate": 1,
        },
        "findings": ["F1"],
    }

    metadata = build_execution_metadata(state, 12.0)
    payload = metadata.to_dict()

    assert metadata.contract_id == "abc123contract"
    assert payload["contract_id"] == "abc123contract"
    assert payload["contract_path"].endswith("abc123contract.json")


def test_build_execution_metadata_includes_runtime_route_and_count_context() -> None:
    """Execution metadata should expose runtime identity and routing truth."""

    state = {
        "research_loop_count": 2,
        "telemetry": {
            "token_usage": {"total_tokens": 2400},
            "runtime_identity": {
                "build_hash": "337dde6bc5d722ea701eeb9b2aeaea7f68453e26a232b59421a6b38d6409839a",
                "execution_path": "tracelab",
                "llm_backend": "gemini",
                "model": "gemini-2.5-pro",
            },
            "config_fingerprint": "6e4c3a2146d6528183de3df8d13fdca2",
            "routing_trace": [
                {
                    "judge_decision": "finalize",
                    "judge_reason": "gated_finalization",
                    "next_action": "continue",
                    "reason": "not_yet_converged",
                }
            ],
            "finding_counts": {
                "raw_findings": 6,
                "unique_concepts": 4,
                "cluster_count": 4,
                "deduped_findings": 4,
                "dedup_ratio": 0.3333,
            },
            "source_counts": {"sources_before_filtering": 12, "sources_presented_to_llm": 5},
            "final_outcome": {
                "outcome": "complete",
                "routing_reason": "converged_with_coverage",
            },
            "telemetry_artifact_path": "checkpoints/telemetry/mission-B27.2.jsonl",
        },
        "sources_found": [{"url": "https://example.com"}],
        "findings": ["F1", "F2"],
        "depth_config": {
            "min_total_findings": 1,
            "min_total_sources": 1,
            "min_unique_domains_gate": 1,
        },
    }

    metadata = build_execution_metadata(state, 18.0)
    payload = metadata.to_dict()

    assert payload["telemetry_path"].endswith("mission-B27.2.jsonl")
    assert payload["runtime_identity"]["execution_path"] == "tracelab"
    assert payload["config_fingerprint"] == "6e4c3a2146d6528183de3df8d13fdca2"
    assert payload["route_summary"]["judge_decision"] == "finalize"
    assert payload["finding_counts"]["unique_concepts"] == 4
    assert payload["finding_counts"]["deduped_findings"] == 4
    assert payload["source_counts"]["sources_presented_to_llm"] == 5
    assert payload["final_outcome"]["outcome"] == "complete"
