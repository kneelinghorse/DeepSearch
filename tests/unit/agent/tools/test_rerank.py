"""Unit tests for ``deepsearch.agent.tools.rerank``.

The dense-embedding model is replaced with a deterministic fake in every test
except ``test_rerank_dense_model_lazy_load`` so the suite never downloads the
real BGE ONNX model in CI. The lazy-load test patches ``TextEmbedding`` itself
to verify the module-level singleton instantiates exactly once across multiple
``rerank()`` calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import deepsearch.agent.tools.rerank as rerank_mod
from deepsearch.agent.tools.rerank import rerank


class _FakeEmbed:
    """Deterministic 8-dim 'embedding' driven by token overlap with a vocabulary."""

    _VOCAB = (
        "lambda",
        "serverless",
        "container",
        "browser",
        "automation",
        "cold",
        "start",
        "concurrency",
    )

    def embed(self, documents, **_: Any):
        for doc in documents:
            tokens = doc.lower().split()
            vec = np.array(
                [1.0 if word in tokens else 0.0 for word in self._VOCAB],
                dtype=float,
            )
            if vec.sum() == 0.0:
                # Tiny fallback so cosine sim is well-defined for ascii-only docs
                vec = np.full(len(self._VOCAB), 0.01, dtype=float)
            yield vec


@pytest.fixture
def fake_dense(monkeypatch: pytest.MonkeyPatch):
    """Replace the lazy singleton with a pre-built fake embedding model."""

    monkeypatch.setattr(rerank_mod, "_DENSE_MODEL", _FakeEmbed())
    yield


def test_rerank_happy_path(fake_dense):
    """Two strong matches and three weak matches; top-K returns the two strong."""

    results = [
        {
            "title": "AWS Lambda concurrency",
            "url": "https://docs.aws.amazon.com/lambda/concurrency",
            "snippet": "AWS Lambda serverless concurrency limits and cold start behavior.",
        },
        {
            "title": "Cloud Run container concurrency",
            "url": "https://cloud.google.com/run/docs/concurrency",
            "snippet": "Cloud Run serverless container concurrency model and cold start.",
        },
        {
            "title": "Random blog about cooking",
            "url": "https://example.com/cooking",
            "snippet": "Pasta recipes you can make in 20 minutes.",
        },
        {
            "title": "Stack Overflow about jQuery",
            "url": "https://stackoverflow.com/questions/1/jquery",
            "snippet": "How to bind a click event in jQuery.",
        },
        {
            "title": "Docker hub homepage",
            "url": "https://hub.docker.com/_/random",
            "snippet": "Browse Docker images.",
        },
    ]

    out = rerank("serverless concurrency cold start lambda", results, top_k=2)

    assert len(out) == 2
    top_urls = {r["url"] for r in out}
    assert "https://docs.aws.amazon.com/lambda/concurrency" in top_urls
    assert "https://cloud.google.com/run/docs/concurrency" in top_urls
    for r in out:
        assert "score" in r and isinstance(r["score"], float)
        assert "authority_tier" in r
        assert r["authority_tier"] in {"high", "medium", "low"}


def test_rerank_ties_broken_by_authority_tier(fake_dense):
    """Two results with identical text → identical RRF; high-tier URL wins."""

    snippet = "Cloud Run serverless container concurrency cold start lambda automation"
    results = [
        {
            "title": "Stack Overflow mirror",
            "url": "https://stackoverflow.com/questions/42/cloud-run",
            "snippet": snippet,
        },
        {
            "title": "Cloud Run docs",
            "url": "https://cloud.google.com/run/docs/cloud-run",
            "snippet": snippet,
        },
    ]

    out = rerank("serverless concurrency", results, top_k=2)

    assert len(out) == 2
    # Cloud Run docs (high tier) beats Stack Overflow (low tier) on identical text
    assert out[0]["url"].startswith("https://cloud.google.com/")
    assert out[0]["authority_tier"] == "high"
    assert out[1]["authority_tier"] == "low"
    assert out[0]["score"] > out[1]["score"]


def test_rerank_all_low_tier(fake_dense):
    """All low-tier URLs still rank cleanly (uniform 0.7 multiplier doesn't degenerate)."""

    results = [
        {
            "title": "Reddit thread",
            "url": "https://reddit.com/r/aws/lambda",
            "snippet": "AWS Lambda cold start in serverless workloads explained.",
        },
        {
            "title": "Medium post",
            "url": "https://medium.com/post/cold-start",
            "snippet": "Some discussion of cold start in containers without much depth.",
        },
        {
            "title": "Random blog",
            "url": "https://example.com/blog",
            "snippet": "Off-topic essay about gardening.",
        },
    ]

    out = rerank("aws lambda serverless cold start", results, top_k=3)

    assert len(out) == 3
    assert {r["authority_tier"] for r in out} == {"low"}
    # Reddit's snippet matches the query best → should rank first
    assert out[0]["url"].startswith("https://reddit.com/")
    # Scores must be strictly descending
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)


def test_rerank_empty_input(fake_dense):
    out = rerank("any query", [], top_k=10)
    assert out == []


def test_rerank_empty_query(fake_dense):
    """Empty/whitespace query returns [] without computing embeddings."""

    results = [
        {"title": "x", "url": "https://docs.aws.amazon.com/lambda", "snippet": "..."}
    ]
    assert rerank("", results, top_k=5) == []
    assert rerank("   ", results, top_k=5) == []


def test_rerank_short_snippets(fake_dense):
    """1-3 word snippets do not crash bm25s and still produce a sensible blend."""

    results = [
        {
            "title": "AWS Lambda",
            "url": "https://docs.aws.amazon.com/lambda",
            "snippet": "serverless",
        },
        {
            "title": "Cloud Run docs",
            "url": "https://cloud.google.com/run/docs",
            "snippet": "container runtime",
        },
        {
            "title": "Reddit",
            "url": "https://reddit.com/r/aws",
            "snippet": "cool",
        },
    ]

    out = rerank("serverless container", results, top_k=3)

    assert len(out) == 3
    for r in out:
        assert isinstance(r["score"], float)
        assert r["authority_tier"] in {"high", "medium", "low"}
    # Either AWS Lambda or Cloud Run should outrank the bare Reddit snippet
    assert out[0]["authority_tier"] in {"high"}
    assert out[-1]["authority_tier"] == "low"


def test_rerank_top_k_larger_than_results(fake_dense):
    results = [
        {
            "title": f"r{i}",
            "url": f"https://docs.aws.amazon.com/lambda/{i}",
            "snippet": f"snippet number {i} talking about lambda serverless",
        }
        for i in range(5)
    ]

    out = rerank("lambda serverless", results, top_k=20)

    assert len(out) == 5


def test_rerank_dense_model_lazy_load(monkeypatch: pytest.MonkeyPatch):
    """First rerank loads the model; subsequent calls reuse the singleton."""

    instantiation_count = {"n": 0}

    class _CountingEmbed:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            instantiation_count["n"] += 1

        def embed(self, documents, **_: Any):
            for d in documents:
                yield np.array([float(len(d) % 7), 1.0], dtype=float)

    # Reset singleton + replace the constructor symbol the lazy-loader pulls in
    monkeypatch.setattr(rerank_mod, "_DENSE_MODEL", None)
    monkeypatch.setattr(rerank_mod, "TextEmbedding", _CountingEmbed)

    results = [
        {"title": "r1", "url": "https://example.com/1", "snippet": "snip number one"},
        {"title": "r2", "url": "https://example.com/2", "snippet": "snip number two"},
    ]

    rerank("query a", results, top_k=2)
    rerank("query b", results, top_k=2)
    rerank("query c", results, top_k=2)

    assert instantiation_count["n"] == 1


def test_rerank_score_keys_in_output(fake_dense):
    """Original keys preserved; score and authority_tier appended."""

    results = [
        {
            "title": "AWS Lambda docs",
            "url": "https://docs.aws.amazon.com/lambda",
            "snippet": "AWS Lambda serverless documentation talks about concurrency.",
        }
    ]

    out = rerank("lambda concurrency", results, top_k=1)

    assert len(out) == 1
    item = out[0]
    assert item["title"] == "AWS Lambda docs"
    assert item["url"] == "https://docs.aws.amazon.com/lambda"
    assert "AWS Lambda" in item["snippet"]
    assert isinstance(item["score"], float)
    assert item["authority_tier"] == "high"
