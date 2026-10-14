"""S65 deterministic reranker — BM25 + dense embeddings + authority-tier filter.

Slots between Linkup's raw response and the CodeAgent's view of search
results. Replaces the deleted LLM-as-judge admission filter (named anti-pattern
per ``cmos/foundational-docs/sota_architecture_evidence.md`` §2 — too slow,
cost-prohibitive, high false-rejection rates) with a deterministic ML rerank
pipeline:

1. **BM25 lexical** via ``bm25s>=0.3.6`` — fast, no training.
2. **Dense cosine** via ``fastembed`` with ``BAAI/bge-small-en-v1.5``
   (~33MB ONNX, no PyTorch dep). Lazy-loaded once per worker process.
3. **Reciprocal Rank Fusion** (Cormack et al. 2009, k=60) blends the two
   ranked lists. RRF is robust to score-scale mismatch between BM25 raw scores
   and cosine similarity.
4. **Authority-tier modifier** multiplies the RRF score by ``1.3`` for ``high``,
   ``1.0`` for ``medium``, ``0.7`` for ``low`` per
   ``derive_authority_tier`` in ``deepsearch/agent/codeagent_runner.py``.

Snippet text used for both scoring layers: prefer ``snippet`` when it has at
least ~20 chars; otherwise concatenate ``title + " " + snippet`` so very
terse Linkup payloads still produce some BM25 signal.

Public surface: ``rerank(query, results, top_k) -> list[dict]``. Output
preserves the input keys (``title``, ``url``, ``snippet``) and appends two
new keys per item: ``score`` (float) and ``authority_tier`` (str).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Sequence

import bm25s
import numpy as np
from fastembed import TextEmbedding

from deepsearch.agent.codeagent_runner import derive_authority_tier

logger = logging.getLogger(__name__)


_DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_DENSE_MODEL: TextEmbedding | None = None  # lazy singleton, set on first use

_RRF_K = 60
_TIER_MULTIPLIER: dict[str, float] = {"high": 1.3, "medium": 1.0, "low": 0.7}
_SHORT_SNIPPET_THRESHOLD = 20


def _get_dense_model() -> TextEmbedding:
    """Return the shared BGE-small ONNX model, loading on first call.

    Subsequent calls reuse the cached instance — model load is the expensive
    cold-start path; on Railway the first call surfaces the duration in logs
    so cold-start latency is observable.
    """

    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        t0 = time.monotonic()
        logger.info("rerank: loading dense model %s", _DENSE_MODEL_NAME)
        _DENSE_MODEL = TextEmbedding(model_name=_DENSE_MODEL_NAME)
        logger.info(
            "rerank: dense model loaded in %.2fs", time.monotonic() - t0
        )
    return _DENSE_MODEL


def _extract_text(result: Mapping[str, Any]) -> str:
    snippet = str(result.get("snippet") or "").strip()
    title = str(result.get("title") or "").strip()
    if len(snippet) >= _SHORT_SNIPPET_THRESHOLD:
        return snippet
    if title and snippet:
        return f"{title} {snippet}"
    return snippet or title


def _bm25_scores(query: str, docs: Sequence[str]) -> np.ndarray:
    """Run BM25 over ``docs`` for ``query``; return raw per-doc scores aligned to docs."""

    if not docs:
        return np.zeros(0, dtype=float)
    # bm25s.tokenize returns a list of token-lists when return_ids=False
    tokenized_corpus = bm25s.tokenize(list(docs), return_ids=False, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus, show_progress=False)
    tokenized_query = bm25s.tokenize([query], return_ids=False, show_progress=False)
    scores = retriever.get_scores(tokenized_query[0])
    return np.asarray(scores, dtype=float)


def _dense_scores(query: str, docs: Sequence[str]) -> np.ndarray:
    """Cosine similarity between query and each doc using the BGE small ONNX model."""

    if not docs:
        return np.zeros(0, dtype=float)
    model = _get_dense_model()
    query_emb = np.asarray(list(model.embed([query]))[0], dtype=float)
    doc_embs = np.asarray(list(model.embed(list(docs))), dtype=float)

    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return v / norm

    q = _normalize(query_emb)
    d = _normalize(doc_embs)
    return d @ q


def _ranks_from_scores(scores: np.ndarray) -> list[int]:
    """Return 1-based descending ranks (rank 1 = highest score). Stable on ties."""

    n = scores.shape[0]
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    ranks = [0] * n
    for rank_pos, idx in enumerate(order, start=1):
        ranks[idx] = rank_pos
    return ranks


def _rrf_combine(
    bm25_ranks: Sequence[int], dense_ranks: Sequence[int], k: int = _RRF_K
) -> list[float]:
    """Reciprocal Rank Fusion (Cormack 2009): per-doc sum of 1/(k + rank_i)."""

    return [1.0 / (k + br) + 1.0 / (k + dr) for br, dr in zip(bm25_ranks, dense_ranks)]


def _tier_multiplier(tier: str) -> float:
    return _TIER_MULTIPLIER.get(tier, _TIER_MULTIPLIER["medium"])


def rerank(
    query: str,
    results: Sequence[Mapping[str, Any]],
    top_k: int,
    *,
    high_authority_domains: Sequence[str] | None = None,
    default_authority_tier: str = "low",
) -> list[dict[str, Any]]:
    """Rerank Linkup-shaped results by BM25 + dense + authority-tier blend.

    Args:
        query: The user-issued search query (what the agent passed to ``web_search``).
        results: Linkup-shaped ``[{title, url, snippet}, ...]``. Other keys
            (e.g. an ``error`` sentinel) are ignored — callers should route
            error sentinels around rerank, not through it.
        top_k: How many top-ranked results to return. If ``top_k > len(results)``,
            all results are returned.
        high_authority_domains: Mission-local primary domains promoted only
            for this scoring call.
        default_authority_tier: Fallback tier after global and mission-scoped
            rules. Scholar retrieval passes ``"high"`` so academic promotion
            occurs before the multiplier instead of mutating output afterward.

    Returns:
        ``list[dict]`` of length ``min(top_k, len(results))``. Each dict has
        the original keys plus ``score`` (final RRF × tier multiplier) and
        ``authority_tier`` (``"high" | "medium" | "low"``). Empty list on
        empty input or empty/whitespace query.
    """

    if not isinstance(query, str) or not query.strip():
        return []
    if not results:
        return []

    tiers = [
        derive_authority_tier(
            str(result.get("url") or ""),
            high_authority_domains=high_authority_domains,
            default_tier=default_authority_tier,
        )
        for result in results
    ]
    docs = [_extract_text(r) for r in results]
    bm25_raw = _bm25_scores(query, docs)
    dense_raw = _dense_scores(query, docs)
    bm25_ranks = _ranks_from_scores(bm25_raw)
    dense_ranks = _ranks_from_scores(dense_raw)
    rrf = _rrf_combine(bm25_ranks, dense_ranks)

    enriched: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        tier = tiers[i]
        final_score = float(rrf[i]) * _tier_multiplier(tier)
        item = dict(r)
        item["score"] = final_score
        item["authority_tier"] = tier
        enriched.append(item)

    enriched.sort(key=lambda item: item["score"], reverse=True)
    if top_k <= 0:
        return []
    return enriched[: min(top_k, len(enriched))]


__all__ = ["rerank"]
