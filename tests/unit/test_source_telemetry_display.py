"""Sprint 58.1 — `select_displayed_research_sources` cap-removal regressions.

Pins the post-S58.1 contract:

* Normal path returns the full ranked pool (no 10/20 hard ceiling).
* Ranking is by composite authority + relevance, best-first.
* `compact_mode` retains its deliberate ≤ 4 degraded-mode tuning.
* `apply_source_token_budget` is the single boundary that may trim sources.
"""

from __future__ import annotations

from typing import Any, Dict, List

from deepsearch.agent.source_telemetry import (
    rank_sources_by_authority_and_relevance,
    select_displayed_research_sources,
)
from deepsearch.agent.token_budget import apply_source_token_budget


def _make_source(index: int, *, authority: float = 0.0, relevance: float = 0.0) -> Dict[str, Any]:
    return {
        "title": f"Source {index}",
        "url": f"https://example.com/source-{index}",
        "snippet": "A" * 600,
        "authority_score": authority,
        "relevance_score": relevance,
    }


def test_normal_path_returns_full_pool_uncapped() -> None:
    """A 100-source pool round-trips intact through the display selector."""

    pool = [_make_source(idx, authority=0.5, relevance=0.5) for idx in range(100)]

    displayed = select_displayed_research_sources(pool)

    assert len(displayed) == 100, "S58.1 cap removal: full pool must reach the prompt"
    assert {entry["url"] for entry in displayed} == {entry["url"] for entry in pool}


def test_normal_path_returns_more_than_legacy_cap_at_baseline_budget() -> None:
    """100 sources at a BASELINE-shaped 16k budget must yield >20 displayed.

    Mirrors the S58.1 success criterion: the legacy ceiling was 20 even at
    ALPHA; this test pins that the budget — not a count — is what drives
    final size, and that 100 short snippets fit well above the legacy cap.
    """

    pool = [_make_source(idx, authority=0.5, relevance=0.5) for idx in range(100)]

    displayed = select_displayed_research_sources(pool)

    def prompt_factory(subset: List[Dict[str, Any]]) -> tuple[str, str]:
        snippet = "\n".join(
            f"[{idx}] {entry['title']}: {entry['url']}" for idx, entry in enumerate(subset, 1)
        )
        return ("system", snippet)

    class StubLLM:
        max_output_tokens = 0
        thinking_budget = 0

    trimmed, _system, _human, _tokens = apply_source_token_budget(
        sources=displayed,
        prompt_factory=prompt_factory,
        telemetry={},
        loop_index=0,
        llm_config=StubLLM(),
        max_tokens_per_loop=16_000,
        node_name="research_node",
    )

    assert len(trimmed) > 20, "S58.1: cap removal must let > 20 sources reach a 16k budget"
    assert len(trimmed) <= len(pool)


def test_normal_path_orders_high_authority_first() -> None:
    """Mixed-authority pools must come out ranked best-first."""

    pool = [
        _make_source(0, authority=0.10, relevance=0.10),
        _make_source(1, authority=0.90, relevance=0.80),
        _make_source(2, authority=0.50, relevance=0.50),
        _make_source(3, authority=0.20, relevance=0.20),
        _make_source(4, authority=0.95, relevance=0.95),
    ]

    displayed = select_displayed_research_sources(pool)
    indices = [int(entry["url"].rsplit("-", 1)[-1]) for entry in displayed]

    assert indices == [4, 1, 2, 3, 0], (
        "Composite authority+relevance must drive ordering; insertion order should not."
    )


def test_compact_mode_caps_at_four() -> None:
    """compact_mode preserves the ≤ 4 degraded-mode tuning."""

    pool = [_make_source(idx, authority=0.5, relevance=0.5) for idx in range(20)]

    displayed = select_displayed_research_sources(pool, compact_mode=True)

    assert len(displayed) == 4
    assert displayed == select_displayed_research_sources(pool, compact_mode=True)


def test_compact_mode_still_ranked_best_first() -> None:
    """Within the ≤ 4 cap, compact_mode also returns best-ranked first."""

    pool = [
        _make_source(0, authority=0.05, relevance=0.05),
        _make_source(1, authority=0.95, relevance=0.95),
        _make_source(2, authority=0.50, relevance=0.50),
        _make_source(3, authority=0.20, relevance=0.20),
        _make_source(4, authority=0.80, relevance=0.80),
        _make_source(5, authority=0.10, relevance=0.10),
    ]

    displayed = select_displayed_research_sources(pool, compact_mode=True)
    indices = [int(entry["url"].rsplit("-", 1)[-1]) for entry in displayed]

    assert indices == [1, 4, 2, 3]


def test_explicit_max_display_still_honored_for_callers_that_need_it() -> None:
    """`max_display=N` still slices to N for backward compatibility."""

    pool = [_make_source(idx, authority=0.5, relevance=0.5) for idx in range(20)]

    displayed = select_displayed_research_sources(pool, max_display=5)

    assert len(displayed) == 5


def test_rank_sources_is_stable_for_unscored_pools() -> None:
    """When no source has authority/relevance, insertion order is preserved."""

    pool = [_make_source(idx) for idx in range(10)]

    ranked = rank_sources_by_authority_and_relevance(pool)
    indices = [int(entry["url"].rsplit("-", 1)[-1]) for entry in ranked]

    assert indices == list(range(10))


def test_empty_pool_returns_empty() -> None:
    """Edge case: empty input yields empty output, no exceptions."""

    assert select_displayed_research_sources([]) == []
    assert select_displayed_research_sources([], compact_mode=True) == []
    assert rank_sources_by_authority_and_relevance([]) == []
