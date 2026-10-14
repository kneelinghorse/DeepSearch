"""S60.2 — per-entity displayed-source floor regressions.

Pins the post-S60.2 contract:

* When ``required_entities`` is empty / blank-only / floor<=0, the function
  is a strict no-op against ``rank_sources_by_authority_and_relevance``.
* When declared entities are present, the top ``per_entity_floor`` sources
  bound to each declared entity are placed at the front of the ordering;
  the remainder fills by global rank.
* Multi-bound sources (a single source bound to multiple declared entities)
  count toward each entity's floor but appear only once in output.
* Reserved sources retain their global rank order at the front; the
  remainder retains its global rank order behind them.
* Sources bound to non-declared entities are NOT promoted by the floor.

These properties together guarantee that the OODS-FIGMA-HOST-01 funnel-
collapse defect (vendor docs for one platform crowding out the others
under global rank) is closed at the displayed-source selection layer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from deepsearch.agent.source_telemetry import (
    rank_sources_by_authority_and_relevance,
    rank_with_per_entity_floor,
)


def _make_source(
    index: int,
    *,
    authority: float = 0.0,
    relevance: float = 0.0,
    entities: list[str] | None = None,
) -> Dict[str, Any]:
    return {
        "title": f"Source {index}",
        "url": f"https://example.com/source-{index}",
        "authority_score": authority,
        "relevance_score": relevance,
        "contract_entity_matches": list(entities or []),
    }


def test_no_op_when_required_entities_empty() -> None:
    """Empty declared list returns the global ranking unchanged."""

    pool = [
        _make_source(idx, authority=(idx % 5) / 10.0, relevance=(idx % 7) / 10.0)
        for idx in range(20)
    ]

    floor_ranked = rank_with_per_entity_floor(
        pool, required_entities=[], per_entity_floor=9
    )
    global_ranked = rank_sources_by_authority_and_relevance(pool)

    assert [s["url"] for s in floor_ranked] == [s["url"] for s in global_ranked]


def test_no_op_when_floor_is_zero() -> None:
    """``per_entity_floor=0`` disables the floor regardless of declared list."""

    pool = [
        _make_source(idx, authority=0.5, relevance=0.5, entities=["AWS Lambda"])
        for idx in range(5)
    ]

    floor_ranked = rank_with_per_entity_floor(
        pool, required_entities=["AWS Lambda", "Railway"], per_entity_floor=0
    )
    global_ranked = rank_sources_by_authority_and_relevance(pool)

    assert floor_ranked == global_ranked


def test_no_op_when_declared_list_blank_only() -> None:
    """Whitespace-only declared entries do not trigger reservation."""

    pool = [
        _make_source(idx, authority=0.5, relevance=0.5, entities=["AWS Lambda"])
        for idx in range(3)
    ]
    floor_ranked = rank_with_per_entity_floor(
        pool, required_entities=["", "  ", None], per_entity_floor=9  # type: ignore[list-item]
    )
    global_ranked = rank_sources_by_authority_and_relevance(pool)
    assert floor_ranked == global_ranked


def test_per_entity_floor_promotes_underrepresented_entity_above_dominant_lane() -> None:
    """The motivating OODS-FIGMA-HOST-01 case at unit scale.

    20 high-authority AWS Lambda sources + 2 medium-authority Vercel
    sources + 2 medium-authority Fly.io sources. Global rank would push
    AWS Lambda to the top-20 slots, leaving Vercel/Fly.io 21st+. With a
    K=2 per-entity floor, Vercel and Fly.io each reserve their best 2
    sources at the front; the remaining slots fill by global rank.
    """

    aws_pool = [
        _make_source(
            f"aws-{i}", authority=0.95, relevance=0.95, entities=["AWS Lambda"]
        )
        for i in range(20)
    ]
    vercel_pool = [
        _make_source(
            f"vercel-{i}",
            authority=0.7,
            relevance=0.7,
            entities=["Vercel Functions"],
        )
        for i in range(2)
    ]
    fly_pool = [
        _make_source(
            f"fly-{i}", authority=0.7, relevance=0.7, entities=["Fly.io"]
        )
        for i in range(2)
    ]
    pool = aws_pool + vercel_pool + fly_pool

    declared = ["AWS Lambda", "Vercel Functions", "Fly.io"]
    floor_ranked = rank_with_per_entity_floor(
        pool, required_entities=declared, per_entity_floor=2
    )

    # First 6 reserved slots: 2 AWS + 2 Vercel + 2 Fly.io. (AWS reserves
    # its top 2 by global rank — index ordering across same-score is
    # stable; Vercel and Fly.io reserve all of theirs.)
    front_titles = [s["title"] for s in floor_ranked[:6]]
    aws_in_front = [t for t in front_titles if t.startswith("Source aws-")]
    vercel_in_front = [t for t in front_titles if t.startswith("Source vercel-")]
    fly_in_front = [t for t in front_titles if t.startswith("Source fly-")]
    assert len(aws_in_front) == 2
    assert len(vercel_in_front) == 2
    assert len(fly_in_front) == 2

    # Remainder of AWS (18 sources) follows in global-rank order; no
    # Vercel/Fly remaining beyond the 2 each.
    tail = floor_ranked[6:]
    assert all(s["title"].startswith("Source aws-") for s in tail), (
        f"unexpected non-AWS source in tail: {[s['title'] for s in tail]}"
    )
    assert len(tail) == 18


def test_per_entity_floor_does_not_duplicate_multi_bound_sources() -> None:
    """A source bound to two declared entities counts toward both floors but
    appears once in output."""

    pool = [
        _make_source(
            "shared",
            authority=0.9,
            relevance=0.9,
            entities=["AWS Lambda", "Google Cloud Run"],
        ),
        _make_source(
            "aws-only", authority=0.8, relevance=0.8, entities=["AWS Lambda"]
        ),
        _make_source(
            "gcr-only", authority=0.8, relevance=0.8, entities=["Google Cloud Run"]
        ),
    ]

    floor_ranked = rank_with_per_entity_floor(
        pool,
        required_entities=["AWS Lambda", "Google Cloud Run"],
        per_entity_floor=1,
    )

    urls = [s["url"] for s in floor_ranked]
    # No duplicates.
    assert len(urls) == len(set(urls)) == 3
    # Shared source must appear (it covers both entity floors at once).
    assert any("shared" in u for u in urls)


def test_per_entity_floor_preserves_global_rank_within_reserved_block() -> None:
    """Reserved sources stay in the order they had in the global ranking."""

    pool = [
        _make_source("a", authority=0.95, relevance=0.95, entities=["X"]),
        _make_source("b", authority=0.90, relevance=0.90, entities=["X"]),
        _make_source("c", authority=0.85, relevance=0.85, entities=["X"]),
        _make_source("d", authority=0.80, relevance=0.80, entities=["Y"]),
        _make_source("e", authority=0.70, relevance=0.70, entities=["Y"]),
    ]
    floor_ranked = rank_with_per_entity_floor(
        pool, required_entities=["X", "Y"], per_entity_floor=2
    )

    # Reserved set: top 2 of X (a, b) + top 2 of Y (d, e). Sorted by
    # original global-rank index → a, b, d, e.
    front_titles = [s["title"] for s in floor_ranked[:4]]
    assert front_titles == ["Source a", "Source b", "Source d", "Source e"]
    # Tail: c (the remaining X source).
    assert [s["title"] for s in floor_ranked[4:]] == ["Source c"]


def test_per_entity_floor_ignores_sources_bound_to_non_declared_entities() -> None:
    """A high-rank source bound only to a non-declared entity stays in the
    tail; declared-entity sources at lower rank still reserve."""

    pool = [
        _make_source("noise", authority=0.99, relevance=0.99, entities=["Heroku"]),
        _make_source(
            "declared-low", authority=0.4, relevance=0.4, entities=["Railway"]
        ),
    ]
    floor_ranked = rank_with_per_entity_floor(
        pool, required_entities=["Railway"], per_entity_floor=1
    )

    # Reserved front: Railway source. Tail: Heroku noise.
    assert [s["title"] for s in floor_ranked] == ["Source declared-low", "Source noise"]


def test_per_entity_floor_handles_entity_with_no_admitted_sources() -> None:
    """A declared entity with zero admitted sources gets nothing reserved;
    other declared entities' floors still apply."""

    pool = [
        _make_source(
            "v1", authority=0.7, relevance=0.7, entities=["Vercel Functions"]
        ),
        _make_source(
            "v2", authority=0.6, relevance=0.6, entities=["Vercel Functions"]
        ),
    ]
    # "Fly.io" has no sources in the admitted pool.
    floor_ranked = rank_with_per_entity_floor(
        pool,
        required_entities=["Vercel Functions", "Fly.io"],
        per_entity_floor=2,
    )

    assert len(floor_ranked) == 2
    titles = [s["title"] for s in floor_ranked]
    # Vercel sources still reserved at the front in rank order.
    assert titles == ["Source v1", "Source v2"]


def test_per_entity_floor_case_insensitive_match() -> None:
    """Declared-list and contract_entity_matches casing differences resolve."""

    pool = [
        _make_source("a", authority=0.9, relevance=0.9, entities=["aws lambda"]),
        _make_source("b", authority=0.85, relevance=0.85, entities=["AWS LAMBDA"]),
        _make_source("c", authority=0.8, relevance=0.8, entities=["Heroku"]),
    ]
    floor_ranked = rank_with_per_entity_floor(
        pool, required_entities=["AWS Lambda"], per_entity_floor=2
    )
    titles = [s["title"] for s in floor_ranked]
    # Both AWS variants reserve; Heroku stays in tail.
    assert titles == ["Source a", "Source b", "Source c"]


def test_per_entity_floor_reserved_count_is_capped_by_admitted_pool() -> None:
    """When per_entity_floor > admitted sources for an entity, only the
    available sources are reserved (no padding, no duplicates)."""

    pool = [
        _make_source("a", authority=0.9, relevance=0.9, entities=["Railway"]),
    ]
    floor_ranked = rank_with_per_entity_floor(
        pool, required_entities=["Railway"], per_entity_floor=9
    )
    assert [s["title"] for s in floor_ranked] == ["Source a"]
