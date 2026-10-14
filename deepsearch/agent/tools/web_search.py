"""web_search — single-lane Linkup wrapper + S65 reranker for the CodeAgent.

Wraps ``deepsearch.tools.linkup_search.LinkupSearchClient`` and runs every
non-empty success result list through ``deepsearch.agent.tools.rerank.rerank``
before returning to the CodeAgent. Return shape on success items:

- ``{"title", "url", "snippet", "score", "authority_tier"}``

The error sentinel path (rate-limit / 5xx / malformed payload / validation)
is unchanged — sentinels are routed around rerank, never through it. An
empty Linkup response is *not* an error: ``[]`` passes through rerank
trivially and ``[]`` is returned to the agent.

S65 carries forward the S64 single-lane shape (Exa stayed dropped per S60
retrieval-waste evidence) and adds the deterministic rerank that replaces
the deleted LLM-as-judge admission filter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from deepsearch.domain_policy import domain_matches
from deepsearch.tools.linkup_search import LinkupSearchClient, LinkupSearchError

from .rerank import rerank

logger = logging.getLogger(__name__)

MAX_RESULTS_HARD_CAP = 50


def _error_sentinel(category: str, message: str) -> list[dict[str, Any]]:
    return [
        {
            "error": category,
            "message": message,
            "title": "",
            "url": "",
            "snippet": "",
        }
    ]


def web_search(
    query: str,
    max_results: int = 20,
    *,
    excluded_domains: Sequence[str] | None = None,
    high_authority_domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Search the web via Linkup, single-lane.

    Returns up to ``max_results`` items (hard-capped at 50). On rate-limit, 5xx,
    or malformed-payload, returns a single-item sentinel list with ``error``
    populated rather than raising — the CodeAgent reacts via ReAct.

    Args:
        query: Natural-language or keyword search query.
        max_results: Soft cap on returned items (1..50).
        excluded_domains: Mission-local domains removed before reranking.
        high_authority_domains: Mission-local domains promoted during scoring.
    """

    if not isinstance(query, str) or not query.strip():
        return _error_sentinel("validation", "query must be a non-empty string")
    if not isinstance(max_results, int):
        try:
            max_results = int(max_results)
        except Exception:
            return _error_sentinel("validation", "max_results must be an int")
    capped = max(1, min(max_results, MAX_RESULTS_HARD_CAP))

    client = LinkupSearchClient()
    try:
        results = asyncio.run(client.search(query=query, max_results=capped))
    except LinkupSearchError as exc:
        return _error_sentinel(getattr(exc, "category", "unknown") or "unknown", str(exc))
    except Exception as exc:
        logger.warning("web_search unexpected error: %r", exc)
        return _error_sentinel("unknown", f"{type(exc).__name__}: {exc}")

    raw: list[dict[str, Any]] = []
    for r in results:
        if domain_matches(r.url, excluded_domains):
            continue
        raw.append(
            {
                "title": r.title or "",
                "url": r.url or "",
                "snippet": r.snippet or "",
            }
        )
    if high_authority_domains:
        return rerank(
            query,
            raw,
            top_k=capped,
            high_authority_domains=high_authority_domains,
        )
    return rerank(query, raw, top_k=capped)


__all__ = ["MAX_RESULTS_HARD_CAP", "web_search"]
