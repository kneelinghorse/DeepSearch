"""scholar_search — academic literature backend (OpenAlex primary, SS fallback).

Mirrors the ``web_search.py`` wrapper pattern: thin shim over the existing
low-level clients in ``deepsearch.tools.openalex_search`` and
``deepsearch.tools.semantic_scholar_search``, then routes results through
the S65 deterministic reranker. Return shape matches ``web_search``:

- ``{"title", "url", "snippet", "score", "authority_tier"}``

S79 m01.a addition (PD93). Closes the universe-of-retrieval failure mode
surfaced by the post-S78 mission pair (PLUGIN-BACKEND-AUTH-01 +
STRUCTURED-OUTPUT-VIEWER-01) — both produced 0 academic / 0 standards / 0 RFC
sources because the agent only had the single-lane Linkup retrieval surface.
This module is the second lane.

Scholar results default to ``authority_tier="high"`` because academic
literature is presumed high-quality; ``derive_authority_tier`` still demotes
items whose URL is on the explicit low-host short-circuit list (``_LOW_HOSTS``)
— e.g., a forum cross-link that surfaced in an academic search remains low.

Backend selection:
- Primary: OpenAlex via ``OpenAlexClient``. Free polite-pool tier with email;
  ``OPENALEX_API_KEY`` (used as the polite-pool email) upgrades to premium tier.
- Fallback: Semantic Scholar via ``SemanticScholarClient``.
  ``SEMANTIC_SCHOLAR_API_KEY`` (sent as ``x-api-key`` header) improves rate limits.

The error-sentinel pattern matches ``web_search.py``: rate-limit / 5xx /
malformed-payload from BOTH backends returns a single-item ``error`` sentinel
rather than raising. If the primary returns an empty list, the fallback runs;
if both succeed with empty lists, an empty list is returned (``[]`` is not an
error).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Sequence, cast

from deepsearch.domain_policy import domain_matches
from deepsearch.tools.openalex_search import OpenAlexClient, OpenAlexError
from deepsearch.tools.semantic_scholar_search import (
    SemanticScholarClient,
    SemanticScholarError,
)

from .rerank import rerank

logger = logging.getLogger(__name__)

MAX_RESULTS_HARD_CAP = 50


class _MalformedScholarResponseError(ValueError):
    """Backend payload cannot be distinguished safely from a valid empty set."""


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


def _normalise_results(
    results: Any,
    excluded_domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Coerce ``OpenAlexResult`` / ``SemanticScholarResult`` dataclasses to the
    ``{title, url, snippet}`` shape rerank expects. Sourceless entries are dropped.
    """
    if not isinstance(results, list):
        raise _MalformedScholarResponseError(
            f"expected list, got {type(results).__name__}"
        )
    raw: list[dict[str, Any]] = []
    for r in results:
        if not all(hasattr(r, field) for field in ("url", "title", "snippet")):
            raise _MalformedScholarResponseError(
                f"result item missing required fields: {type(r).__name__}"
            )
        url = (getattr(r, "url", "") or "").strip()
        if not url or domain_matches(url, excluded_domains):
            continue
        raw.append(
            {
                "title": (getattr(r, "title", "") or "").strip(),
                "url": url,
                "snippet": (getattr(r, "snippet", "") or "").strip(),
            }
        )
    return raw


def scholar_search(
    query: str,
    max_results: int = 20,
    *,
    excluded_domains: Sequence[str] | None = None,
    high_authority_domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Search academic literature: OpenAlex primary, Semantic Scholar fallback.

    Returns up to ``max_results`` items (hard-capped at 50). The primary
    backend is tried first; on error or empty result the fallback runs.
    On rate-limit / 5xx / malformed-payload from BOTH backends, returns a
    single-item sentinel list with ``error`` populated rather than raising —
    the CodeAgent reacts via ReAct.

    Args:
        query: Natural-language or keyword search query.
        max_results: Soft cap on returned items (1..50).
        excluded_domains: Mission-local domains removed before reranking.
        high_authority_domains: Mission-local primary domains promoted before
            scoring.
    """

    if not isinstance(query, str) or not query.strip():
        return _error_sentinel("validation", "query must be a non-empty string")
    if not isinstance(max_results, int):
        try:
            max_results = int(max_results)
        except Exception:
            return _error_sentinel("validation", "max_results must be an int")
    capped = max(1, min(max_results, MAX_RESULTS_HARD_CAP))

    primary_error: tuple[str, str] | None = None

    openalex = OpenAlexClient(api_key=os.environ.get("OPENALEX_API_KEY"))
    try:
        oa_results = asyncio.run(openalex.search(query, max_results=capped))
        oa_normalised = _normalise_results(oa_results, excluded_domains)
        if oa_normalised:
            return cast(
                list[dict[str, Any]],
                rerank(
                    query,
                    oa_normalised,
                    top_k=capped,
                    high_authority_domains=high_authority_domains,
                    default_authority_tier="high",
                ),
            )
    except _MalformedScholarResponseError as exc:
        primary_error = ("malformed_payload", str(exc))
        logger.warning("scholar_search OpenAlex malformed response: %s", exc)
    except OpenAlexError as exc:
        primary_error = (
            getattr(exc, "category", "unknown") or "unknown",
            str(exc),
        )
        logger.info(
            "scholar_search OpenAlex failed (%s); trying Semantic Scholar fallback",
            exc,
        )
    except Exception as exc:  # noqa: BLE001 — protect the agent from network surprises
        primary_error = ("unknown", f"{type(exc).__name__}: {exc}")
        logger.warning("scholar_search OpenAlex unexpected error: %r", exc)

    semantic = SemanticScholarClient(
        api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    )
    try:
        ss_results = asyncio.run(semantic.search(query, max_results=capped))
        ss_normalised = _normalise_results(ss_results, excluded_domains)
    except _MalformedScholarResponseError as exc:
        logger.warning("scholar_search Semantic Scholar malformed response: %s", exc)
        return _error_sentinel("malformed_payload", str(exc))
    except SemanticScholarError as exc:
        if primary_error is None:
            primary_error = (
                getattr(exc, "category", "unknown") or "unknown",
                str(exc),
            )
        return _error_sentinel(*primary_error)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scholar_search Semantic Scholar unexpected error: %r", exc)
        if primary_error is None:
            primary_error = ("unknown", f"{type(exc).__name__}: {exc}")
        return _error_sentinel(*primary_error)

    if (
        not ss_normalised
        and primary_error is not None
        and primary_error[0] == "malformed_payload"
    ):
        return _error_sentinel(*primary_error)
    return cast(
        list[dict[str, Any]],
        rerank(
            query,
            ss_normalised,
            top_k=capped,
            high_authority_domains=high_authority_domains,
            default_authority_tier="high",
        ),
    )


__all__ = ["MAX_RESULTS_HARD_CAP", "scholar_search"]
