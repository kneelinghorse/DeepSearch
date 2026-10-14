"""B35.3: synthesis evidence selection prefers high-authority sources.

When two working-set sources cover the same claim equivalently (same keyword
overlap or near-tied semantic similarity), the citation anchor must be the
high-authority source. The mission's success-criterion is locked here as a
deterministic test against the keyword-fallback path used when the semantic
matcher is unavailable, which is the production path most reports take.

Authority is a *tie-breaker*, not a primary signal — a low-authority source
that genuinely matches the claim better than a high-authority source still
wins. These tests cover both directions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from deepsearch.report.embeddings import _authority_for
from deepsearch.report.templates import (
    _find_best_matching_source,
    _inject_citation,
    _source_authority_score,
)


def _source(
    url: str,
    title: str,
    snippet: str,
    *,
    authority_score: float | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"url": url, "title": title, "snippet": snippet}
    if authority_score is not None:
        payload["authority_score"] = authority_score
    return payload


# ---------------------------------------------------------------------------
# Success criterion: high-authority wins when keyword scores tie
# ---------------------------------------------------------------------------


class TestAuthorityBreaksKeywordTies:
    """Same finding, two sources tied on keyword overlap, different authority."""

    def test_high_authority_wins_over_blog_on_keyword_tie(self) -> None:
        finding = "BM25Retriever and MultiQueryRetriever differ in lexical recall."
        sources = [
            _source(
                "https://medium.com/@user/blog-post",
                "BM25Retriever and MultiQueryRetriever",
                "Comparing BM25Retriever to MultiQueryRetriever for lexical recall",
                authority_score=0.5,
            ),
            _source(
                "https://arxiv.org/abs/2024.42",
                "BM25Retriever and MultiQueryRetriever",
                "Comparing BM25Retriever to MultiQueryRetriever for lexical recall",
                authority_score=0.95,
            ),
        ]
        chosen = _find_best_matching_source(finding, sources, cited_urls=set())
        assert chosen is not None
        assert chosen["url"] == "https://arxiv.org/abs/2024.42"

    def test_high_authority_wins_when_listed_after_blog(self) -> None:
        # Insertion order is the previous tiebreaker; assert authority overrides
        # it when a later high-authority source ties on keyword overlap.
        finding = "Transformers learn long-range dependencies via self-attention."
        sources = [
            _source(
                "https://dev.to/user/transformers-101",
                "Transformers and self-attention",
                "Transformers learn long-range dependencies via self-attention",
                authority_score=0.5,
            ),
            _source(
                "https://medium.com/@u/blog",
                "Transformers and self-attention",
                "Transformers learn long-range dependencies via self-attention",
                authority_score=0.5,
            ),
            _source(
                "https://nature.com/article/transformer-survey",
                "Transformers and self-attention",
                "Transformers learn long-range dependencies via self-attention",
                authority_score=0.9,
            ),
        ]
        chosen = _find_best_matching_source(finding, sources, cited_urls=set())
        assert chosen is not None
        assert chosen["url"] == "https://nature.com/article/transformer-survey"

    def test_three_way_authority_tie_break(self) -> None:
        finding = "PostgreSQL supports JSONB indexing for semistructured data."
        sources = [
            _source(
                "https://stackoverflow.com/q/jsonb",
                "PostgreSQL JSONB indexing",
                "PostgreSQL supports JSONB indexing for semistructured data",
                authority_score=0.5,
            ),
            _source(
                "https://github.com/postgres/repo",
                "PostgreSQL JSONB indexing",
                "PostgreSQL supports JSONB indexing for semistructured data",
                authority_score=0.6,
            ),
            _source(
                "https://postgresql.org/docs/current/datatype-json.html",
                "PostgreSQL JSONB indexing",
                "PostgreSQL supports JSONB indexing for semistructured data",
                authority_score=0.9,
            ),
        ]
        chosen = _find_best_matching_source(finding, sources, cited_urls=set())
        assert chosen is not None
        assert chosen["url"] == "https://postgresql.org/docs/current/datatype-json.html"


# ---------------------------------------------------------------------------
# Authority is a tie-breaker, not a primary signal
# ---------------------------------------------------------------------------


class TestAuthorityIsTieBreakerNotOverride:
    """A low-authority source with a clearly-better keyword match still wins."""

    def test_better_keyword_match_beats_higher_authority(self) -> None:
        finding = "MultiQueryRetriever paraphrases the user query into variants."
        sources = [
            _source(
                "https://arxiv.org/abs/9999.0001",
                # Generic title with no MultiQueryRetriever reference
                "Information retrieval survey",
                "An overview of dense retrieval methods",
                authority_score=0.95,
            ),
            _source(
                "https://medium.com/@user/multiqueryretriever-deep-dive",
                "MultiQueryRetriever paraphrases user query",
                "MultiQueryRetriever paraphrases the user query into variants for "
                "improved recall in retrieval pipelines",
                authority_score=0.5,
            ),
        ]
        chosen = _find_best_matching_source(finding, sources, cited_urls=set())
        assert chosen is not None
        # Medium post wins despite lower authority — its keyword overlap is
        # materially higher (multiqueryretriever appears in both title+snippet,
        # plus paraphrases, query, variants).
        assert chosen["url"] == "https://medium.com/@user/multiqueryretriever-deep-dive"


# ---------------------------------------------------------------------------
# Backward compatibility — sources without authority_score field
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Sources missing authority_score (legacy fixtures) default to 0.0."""

    def test_missing_authority_field_defaults_to_zero(self) -> None:
        finding = "Foo bar baz qux."
        sources = [
            _source("https://a.com/page", "Foo bar baz qux", "Foo bar baz qux."),
            _source(
                "https://arxiv.org/abs/x",
                "Foo bar baz qux",
                "Foo bar baz qux.",
                authority_score=0.9,
            ),
        ]
        chosen = _find_best_matching_source(finding, sources, cited_urls=set())
        assert chosen is not None
        # The source with explicit authority wins the tie; the unscored one
        # is treated as 0.0.
        assert chosen["url"] == "https://arxiv.org/abs/x"

    def test_all_sources_without_authority_falls_back_to_insertion_order(self) -> None:
        finding = "Foo bar baz qux."
        sources = [
            _source("https://first.com/page", "Foo bar baz qux", "Foo bar baz qux."),
            _source("https://second.com/page", "Foo bar baz qux", "Foo bar baz qux."),
        ]
        chosen = _find_best_matching_source(finding, sources, cited_urls=set())
        assert chosen is not None
        # Both authority=0.0; insertion order breaks the tie.
        assert chosen["url"] == "https://first.com/page"

    def test_malformed_authority_value_treated_as_zero(self) -> None:
        sources = [{"authority_score": "not-a-number"}, {"authority_score": None}]
        for source in sources:
            assert _source_authority_score(source) == 0.0
            assert _authority_for(source) == 0.0


# ---------------------------------------------------------------------------
# End-to-end through _inject_citation (templates path used in finalized
# reports when no LLM synthesis is available)
# ---------------------------------------------------------------------------


class TestInjectCitationUsesAuthorityTieBreak:
    """_inject_citation must surface the high-authority anchor in markdown."""

    def test_emitted_citation_url_is_high_authority(self) -> None:
        finding = "BM25Retriever and MultiQueryRetriever differ in lexical recall."
        sources: List[Dict[str, Any]] = [
            _source(
                "https://medium.com/@u/post",
                "BM25Retriever and MultiQueryRetriever",
                "Comparing BM25Retriever to MultiQueryRetriever for lexical recall",
                authority_score=0.5,
            ),
            _source(
                "https://arxiv.org/abs/2024.42",
                "BM25Retriever and MultiQueryRetriever",
                "Comparing BM25Retriever to MultiQueryRetriever for lexical recall",
                authority_score=0.95,
            ),
        ]
        cited: set[str] = set()
        # No matcher => keyword fallback path; B35.3 tie-break applies.
        rendered = _inject_citation(finding, sources, cited)
        assert "https://arxiv.org/abs/2024.42" in rendered
        assert "https://medium.com/@u/post" not in rendered
        assert "https://arxiv.org/abs/2024.42" in cited
