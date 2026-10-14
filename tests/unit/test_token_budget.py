"""Unit tests for source prompt token budgeting."""

from __future__ import annotations

from types import SimpleNamespace

from deepsearch.agent.token_budget import apply_source_token_budget


def test_apply_source_token_budget_trims_large_source_batches() -> None:
    sources = [
        {
            "title": f"Source {index}",
            "url": f"https://example.com/{index}",
            "snippet": "Long supporting evidence " * 80,
        }
        for index in range(120)
    ]
    telemetry = {"llm_calls": []}
    llm_config = SimpleNamespace(max_output_tokens=64, thinking_budget=0)

    def prompt_factory(source_subset):
        joined = "\n".join(str(source.get("snippet") or "") for source in source_subset)
        return "System prompt", joined

    trimmed_sources, _system_prompt, _human_prompt, _prompt_tokens = apply_source_token_budget(
        sources=sources,
        prompt_factory=prompt_factory,
        telemetry=telemetry,
        loop_index=0,
        llm_config=llm_config,
        max_tokens_per_loop=220,
        node_name="unit_test",
    )

    assert len(trimmed_sources) < len(sources)
    assert telemetry["token_budget_events"][0]["sources_before"] == 120
    assert telemetry["token_budget_events"][0]["sources_after"] == len(trimmed_sources)
