"""Unit tests for ``deepsearch.agent.tools.scholar_search`` — S79 m01.a NEW tool.

Coverage matches the test plan in ``cmos/reports/sprint_79_approach_and_plan.md``
§4 m01.a: contract on success path, OpenAlex → Semantic Scholar fallback chain,
error sentinel on both-backends-failed, tier defaulting to ``high`` for academic
results unless URL is in ``_LOW_HOSTS``, sourceless entries dropped, validation
on empty query.

Both backend clients are patched at the import boundary; no real network calls.
``rerank`` is patched to a passthrough so the test does not require the
fastembed ONNX model to be loaded — rerank itself is covered by its own test.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepsearch.agent.tools import scholar_search as ss_mod
from deepsearch.tools.openalex_search import OpenAlexError, OpenAlexResult
from deepsearch.tools.semantic_scholar_search import (
    SemanticScholarError,
    SemanticScholarResult,
)


# ---------------------------------------------------------------------------
# Rerank passthrough — keep these tests independent of fastembed model load.
# ---------------------------------------------------------------------------


def _passthrough_rerank(
    query: str,
    raw,
    top_k: int,
    *,
    high_authority_domains=None,
    default_authority_tier: str = "low",
):  # noqa: ARG001
    """Mirror rerank's enrichment shape without invoking BM25/dense models.

    Sets ``score=0.5`` (neutral) and ``authority_tier`` from
    ``derive_authority_tier`` — same behavior as the real rerank's tier pass.
    """
    from deepsearch.agent.codeagent_runner import derive_authority_tier

    out = []
    for item in raw[:top_k]:
        copy = dict(item)
        copy.setdefault("score", 0.5)
        copy["authority_tier"] = derive_authority_tier(
            copy.get("url", "") or "",
            high_authority_domains=high_authority_domains,
            default_tier=default_authority_tier,
        )
        out.append(copy)
    return out


@pytest.fixture(autouse=True)
def _patch_rerank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ss_mod, "rerank", _passthrough_rerank)


# ---------------------------------------------------------------------------
# Validation surface.
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_query_returns_error_sentinel(self):
        out = ss_mod.scholar_search("")
        assert len(out) == 1
        assert out[0]["error"] == "validation"
        assert out[0]["url"] == ""

    def test_whitespace_query_returns_error_sentinel(self):
        out = ss_mod.scholar_search("   \n\t  ")
        assert out[0]["error"] == "validation"

    def test_max_results_coerced_from_string(self):
        # Should accept "5" via int() coercion.
        with patch("deepsearch.agent.tools.scholar_search.OpenAlexClient") as MC:
            MC.return_value.search = _async_returning([])
            with patch(
                "deepsearch.agent.tools.scholar_search.SemanticScholarClient"
            ) as MS:
                MS.return_value.search = _async_returning([])
                out = ss_mod.scholar_search("query", max_results="5")  # type: ignore[arg-type]
        assert isinstance(out, list)


# ---------------------------------------------------------------------------
# OpenAlex primary path.
# ---------------------------------------------------------------------------


def _async_returning(value):
    """Build an async stub returning ``value``."""

    async def stub(*args, **kwargs):  # noqa: ARG001
        return value

    return stub


def _async_raising(exc: Exception):
    async def stub(*args, **kwargs):  # noqa: ARG001
        raise exc

    return stub


class TestOpenAlexPrimaryPath:
    def test_openalex_results_returned_when_non_empty(self, monkeypatch):
        results = [
            OpenAlexResult(
                url="https://arxiv.org/abs/2104.04471",
                title="A study",
                snippet="Snippet text",
            ),
            OpenAlexResult(
                url="https://doi.org/10.1234/ai",
                title="AI paper",
                snippet="More snippet",
            ),
        ]

        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex, patch(
            "deepsearch.agent.tools.scholar_search.SemanticScholarClient"
        ) as MSemanticScholar:
            MOpenAlex.return_value.search = _async_returning(results)

            out = ss_mod.scholar_search("PKCE OAuth 2.0")

        # Semantic Scholar should NOT be invoked when OpenAlex succeeds.
        MSemanticScholar.return_value.search.assert_not_called() if hasattr(
            MSemanticScholar.return_value.search, "assert_not_called"
        ) else None

        assert len(out) == 2
        urls = [item["url"] for item in out]
        assert "https://arxiv.org/abs/2104.04471" in urls
        # arxiv.org is on the m01.b high allowlist.
        arxiv_item = next(i for i in out if "arxiv.org" in i["url"])
        assert arxiv_item["authority_tier"] == "high"
        # doi.org is NOT on the explicit allowlist BUT the academic-tier
        # promotion in scholar_search defaults non-low hosts to high.
        doi_item = next(i for i in out if "doi.org" in i["url"])
        assert doi_item["authority_tier"] == "high"

    def test_sourceless_results_are_dropped_during_normalisation(self, monkeypatch):
        results = [
            OpenAlexResult(url="", title="empty url", snippet="x"),
            OpenAlexResult(
                url="https://arxiv.org/abs/x", title="ok", snippet="y"
            ),
        ]
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex:
            MOpenAlex.return_value.search = _async_returning(results)
            out = ss_mod.scholar_search("query")
        assert len(out) == 1
        assert out[0]["url"] == "https://arxiv.org/abs/x"


# ---------------------------------------------------------------------------
# Fallback path: Semantic Scholar runs when OpenAlex fails OR returns empty.
# ---------------------------------------------------------------------------


class TestSemanticScholarFallback:
    def test_fallback_on_openalex_empty(self):
        ss_results = [
            SemanticScholarResult(
                url="https://doi.org/10.5678/example",
                title="Fallback hit",
                snippet="Fallback snippet",
                year=2024,
            ),
        ]
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex, patch(
            "deepsearch.agent.tools.scholar_search.SemanticScholarClient"
        ) as MSemanticScholar:
            MOpenAlex.return_value.search = _async_returning([])
            MSemanticScholar.return_value.search = _async_returning(ss_results)

            out = ss_mod.scholar_search("query")

        assert len(out) == 1
        assert out[0]["url"] == "https://doi.org/10.5678/example"

    def test_fallback_on_openalex_error(self):
        ss_results = [
            SemanticScholarResult(
                url="https://arxiv.org/abs/2106.99999",
                title="Fallback OK",
                snippet="snippet",
            ),
        ]
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex, patch(
            "deepsearch.agent.tools.scholar_search.SemanticScholarClient"
        ) as MSemanticScholar:
            MOpenAlex.return_value.search = _async_raising(
                OpenAlexError("rate limited", category="network")
            )
            MSemanticScholar.return_value.search = _async_returning(ss_results)

            out = ss_mod.scholar_search("query")

        assert len(out) == 1
        assert out[0]["url"] == "https://arxiv.org/abs/2106.99999"

    def test_both_backends_fail_returns_error_sentinel(self):
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex, patch(
            "deepsearch.agent.tools.scholar_search.SemanticScholarClient"
        ) as MSemanticScholar:
            MOpenAlex.return_value.search = _async_raising(
                OpenAlexError("api down", category="network")
            )
            MSemanticScholar.return_value.search = _async_raising(
                SemanticScholarError("api down", category="network")
            )

            out = ss_mod.scholar_search("query")

        assert len(out) == 1
        # Primary error is preserved (OpenAlex was tried first).
        assert out[0]["error"] == "network"
        assert "api down" in out[0]["message"]

    def test_both_backends_empty_returns_empty_list(self):
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex, patch(
            "deepsearch.agent.tools.scholar_search.SemanticScholarClient"
        ) as MSemanticScholar:
            MOpenAlex.return_value.search = _async_returning([])
            MSemanticScholar.return_value.search = _async_returning([])

            out = ss_mod.scholar_search("nothing matches")

        assert out == []

    def test_malformed_primary_then_empty_fallback_returns_explicit_sentinel(self):
        """Malformed must not collapse into the valid both-backends-empty shape."""

        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex, patch(
            "deepsearch.agent.tools.scholar_search.SemanticScholarClient"
        ) as MSemanticScholar:
            MOpenAlex.return_value.search = _async_returning(None)
            MSemanticScholar.return_value.search = _async_returning([])

            out = ss_mod.scholar_search("query")

        assert out[0]["error"] == "malformed_payload"
        assert "expected list" in out[0]["message"]

    def test_malformed_fallback_item_returns_explicit_sentinel(self):
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex, patch(
            "deepsearch.agent.tools.scholar_search.SemanticScholarClient"
        ) as MSemanticScholar:
            MOpenAlex.return_value.search = _async_returning([])
            MSemanticScholar.return_value.search = _async_returning([{"url": "x"}])

            out = ss_mod.scholar_search("query")

        assert out[0]["error"] == "malformed_payload"


# ---------------------------------------------------------------------------
# Tier promotion — academic default high, low hosts stay low.
# ---------------------------------------------------------------------------


class TestAcademicTierPromotion:
    def test_low_host_stays_low_even_after_academic_search(self):
        # A scholar-search backend that surfaces a forum/aggregator link still
        # has that link demoted via the _LOW_HOSTS short-circuit.
        results = [
            OpenAlexResult(
                url="https://stackoverflow.com/questions/12345",
                title="SO discussion of paper",
                snippet="…",
            ),
        ]
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex:
            MOpenAlex.return_value.search = _async_returning(results)
            out = ss_mod.scholar_search("query")

        assert len(out) == 1
        assert out[0]["authority_tier"] == "low"

    def test_unknown_host_promoted_to_high(self):
        # A URL not on any allowlist (and not in _LOW_HOSTS) should be
        # promoted to high under the academic-default rule.
        results = [
            OpenAlexResult(
                url="https://example-publisher.org/paper-12345",
                title="A paper",
                snippet="…",
            ),
        ]
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex:
            MOpenAlex.return_value.search = _async_returning(results)
            out = ss_mod.scholar_search("query")

        assert out[0]["authority_tier"] == "high"

    def test_already_high_unchanged(self):
        # arxiv.org is high under derive_authority_tier post-m01.b — promotion
        # is a no-op.
        results = [
            OpenAlexResult(
                url="https://arxiv.org/abs/2104.04471",
                title="paper",
                snippet="…",
            ),
        ]
        with patch(
            "deepsearch.agent.tools.scholar_search.OpenAlexClient"
        ) as MOpenAlex:
            MOpenAlex.return_value.search = _async_returning(results)
            out = ss_mod.scholar_search("query")

        assert out[0]["authority_tier"] == "high"
