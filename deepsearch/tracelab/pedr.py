"""PEDR client for pre-flight knowledge queries.

This module implements the "check before research" pattern by querying TraceLab's
PEDR (Protocol-Enhanced Deep Research) preflight endpoint. Before launching web
searches, DeepSearch can check if high-quality existing research already covers
the topic.

The preflight query returns one of three action recommendations:
- reuse: High-quality match found (similarity >= 82%, gates >= 4) - skip web search
- review: Moderate match found (similarity >= 65%) - check existing first
- proceed: No qualifying matches - launch new research

Note: TraceLab S29 fixed PEDR score compounding, producing more accurate
(generally lower) similarity scores. Thresholds were recalibrated in B13.2.

Environment Variables:
    TRACELAB_BASE_URL: Base URL for TraceLab API
    TRACELAB_USERNAME: Username for authentication
    TRACELAB_PASSWORD: Password for authentication
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .auth import TraceLabAuth, TraceLabAuthError
from .schemas import GraphSearchRequest, GraphSearchResponse, ScoreFusion

logger = logging.getLogger(__name__)


def _safe_exception_diagnostic(
    exc: BaseException,
    status_code: Optional[int] = None,
) -> str:
    """Describe an exception without persisting its potentially sensitive text."""

    diagnostic = type(exc).__name__
    if status_code is not None:
        diagnostic = f"{diagnostic}, HTTP {status_code}"
    return diagnostic


class PreflightAction(str, Enum):
    """Action recommendation from preflight query."""

    REUSE = "reuse"  # High-quality match, skip web search
    REVIEW = "review"  # Moderate match, check before proceeding
    PROCEED = "proceed"  # No match, launch new research


@dataclass
class PreflightMatch:
    """A matching mission from PEDR."""

    mission_id: str
    mission_uuid: str
    title: str
    objective: str
    similarity_score: float
    quality_gates_passed: int
    key_insights: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    score_fusion: Optional[ScoreFusion] = None


@dataclass
class PreflightResult:
    """Result of a PEDR preflight query."""

    action: PreflightAction
    summary: str
    top_score: float
    match_count: int
    matches: List[PreflightMatch]
    query: str
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def has_reusable_match(self) -> bool:
        """Check if result has a high-quality reusable match."""
        return self.action == PreflightAction.REUSE and len(self.matches) > 0

    @property
    def has_relevant_match(self) -> bool:
        """Check if result has any relevant match (reuse or review)."""
        return (
            self.action in (PreflightAction.REUSE, PreflightAction.REVIEW)
            and len(self.matches) > 0
        )


class PEDRError(Exception):
    """Raised when PEDR operations fail."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class PEDRClient:
    """Client for PEDR preflight queries.

    Enables the "check before research" pattern by querying TraceLab's
    existing research repository before launching new web searches.

    Args:
        auth: TraceLabAuth instance for JWT authentication.
              If not provided, creates one from environment variables.
        timeout: HTTP timeout in seconds for requests.

    Example:
        >>> pedr = PEDRClient()
        >>> result = pedr.preflight("passwordless authentication patterns")
        >>> if result.action == PreflightAction.REUSE:
        ...     # Use existing research instead of web search
        ...     evidence = pedr.get_reusable_evidence(result)
    """

    # B13.3: Differentiated timeouts per TraceLab S29 latency requirements
    STANDARD_QUERY_TIMEOUT: float = 10.0  # 500ms expected, generous buffer
    GRAPH_QUERY_TIMEOUT: float = 15.0  # 1000ms expected, generous buffer

    def __init__(
        self,
        auth: Optional[TraceLabAuth] = None,
        timeout: float = 30.0,
        standard_timeout: Optional[float] = None,
        graph_timeout: Optional[float] = None,
    ):
        self.auth = auth or TraceLabAuth()
        self.timeout = timeout
        self.standard_timeout = standard_timeout or self.STANDARD_QUERY_TIMEOUT
        self.graph_timeout = graph_timeout or self.GRAPH_QUERY_TIMEOUT

    @property
    def is_configured(self) -> bool:
        """Check if PEDR client is properly configured."""
        return self.auth.is_configured

    def preflight(
        self,
        query: str,
        *,
        project_id: Optional[str] = None,
        include_prior_missions: bool = True,
        source_origins: Optional[List[str]] = None,
        min_quality_gates: int = 4,
        status: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.65,
    ) -> PreflightResult:
        """Check if existing research covers the query.

        Args:
            query: Research question or objective to check.
            project_id: Optional project UUID to scope search.
            include_prior_missions: If True and source_origins not set, filters
                to "synthesized" sources (prior DeepSearch work). Default: True.
            source_origins: Filter by source origin. Use ["synthesized"] to
                prioritize prior DeepSearch missions. Overrides include_prior_missions.
            min_quality_gates: Minimum gates passed for match (default: 4).
            status: Mission status filter (default: ["complete"]).
            top_k: Maximum matches to return (default: 5).
            similarity_threshold: Minimum similarity score (default: 0.65).
                B13.2: Lowered from 0.70 to 0.65 to account for S29 score
                compounding fix producing more accurate (lower) scores.

        Returns:
            PreflightResult with action recommendation and matches.

        Note:
            On failure, returns a PreflightResult with action=PROCEED
            to gracefully degrade to web search.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed - PEDR preflight unavailable")
            return self._fallback_result(query, "httpx not installed")

        if status is None:
            status = ["complete"]

        # Determine source_origins filter
        effective_source_origins = source_origins
        if effective_source_origins is None and include_prior_missions:
            effective_source_origins = ["synthesized"]

        # NOTE: No trailing slash per TraceLab B12.2 workaround
        url = f"{self.auth.base_url}/api/v1/pedr/preflight"

        payload: Dict[str, Any] = {
            "query": query,
            "min_quality_gates": min_quality_gates,
            "status": status,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }

        # Add optional parameters
        if project_id is not None:
            payload["project_id"] = project_id
        if effective_source_origins is not None:
            payload["source_origins"] = effective_source_origins

        try:
            headers = self.auth.get_headers()
            resp = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.standard_timeout,  # B13.3: use standard timeout for preflight
            )
            resp.raise_for_status()
            data = resp.json()

            matches = [
                PreflightMatch(
                    mission_id=m["mission_id"],
                    mission_uuid=m.get("mission_uuid", ""),
                    title=m["title"],
                    objective=m.get("objective", ""),
                    similarity_score=m["similarity_score"],
                    quality_gates_passed=m["quality_gates_passed"],
                    key_insights=m.get("key_insights", []),
                    tags=m.get("tags", []),
                    score_fusion=ScoreFusion(**m["score_fusion"])
                    if m.get("score_fusion")
                    else None,
                )
                for m in data.get("matches", [])
            ]

            return PreflightResult(
                action=PreflightAction(data["action"]),
                summary=data.get("summary", ""),
                top_score=data.get("top_score", 0.0),
                match_count=data.get("match_count", len(matches)),
                matches=matches,
                query=data.get("query", query),
                latency_ms=data.get("latency_ms", 0.0),
            )

        except TraceLabAuthError as exc:
            diagnostic = _safe_exception_diagnostic(exc, exc.status_code)
            logger.warning("PEDR auth failed (%s)", diagnostic)
            return self._fallback_result(
                query,
                f"Authentication failed ({diagnostic})",
                exc.status_code,
            )

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            diagnostic = _safe_exception_diagnostic(exc, status_code)
            logger.warning(
                "PEDR preflight HTTP failure (%s)",
                diagnostic,
            )
            return self._fallback_result(
                query,
                f"HTTP failure ({diagnostic})",
                status_code,
            )

        except httpx.TimeoutException as exc:
            diagnostic = _safe_exception_diagnostic(exc)
            logger.warning("PEDR preflight timeout (%s)", diagnostic)
            return self._fallback_result(query, f"Timeout ({diagnostic})")

        except httpx.RequestError as exc:
            diagnostic = _safe_exception_diagnostic(exc)
            logger.warning("PEDR preflight request failure (%s)", diagnostic)
            return self._fallback_result(query, f"Request failure ({diagnostic})")

        except Exception as exc:
            diagnostic = _safe_exception_diagnostic(exc)
            logger.warning("PEDR preflight unexpected failure (%s)", diagnostic)
            return self._fallback_result(query, f"Unexpected failure ({diagnostic})")

    def search(
        self,
        query: str,
        *,
        project_id: Optional[str] = None,
        top_k: int = 5,
        enable_graph: bool = True,
        graph_depth: Optional[int] = None,
        graph_decay: Optional[float] = None,
        graph_edge_types: Optional[List[str]] = None,
        graph_weight: Optional[float] = None,
        include_related: bool = False,
        include_embeddings: bool = False,
        source_origin: Optional[str] = None,
    ) -> GraphSearchResponse:
        """Execute a PEDR graph search query.

        Args:
            query: Search query text.
            project_id: Optional project UUID to scope search.
            top_k: Maximum results to return (default: 5).
            enable_graph: Enable graph expansion layer (default: True).
            graph_depth: Graph expansion depth (L6).
            graph_decay: Graph score decay factor.
            graph_edge_types: Graph edge types to include.
            graph_weight: Weight of graph layer in ranking.
            include_related: Include related entities in results.
            include_embeddings: Include embedding vectors in results.
            source_origin: B13.3: Filter by source origin (e.g., "synthesized").

        Returns:
            GraphSearchResponse with results and metadata.

        Note:
            On failure, returns an empty GraphSearchResponse to degrade gracefully.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed - PEDR graph search unavailable")
            return self._fallback_search_response(query, "httpx not installed")

        query = (query or "").strip()
        if not query:
            return self._fallback_search_response(query, "Query cannot be empty")

        request = GraphSearchRequest(
            query=query,
            project_id=project_id,
            top_k=top_k,
            enable_graph=enable_graph,
            graph_depth=graph_depth,
            graph_decay=graph_decay,
            graph_edge_types=graph_edge_types or [],
            graph_weight=graph_weight,
            include_related=include_related,
            include_embeddings=include_embeddings,
            source_origin=source_origin,
        )
        payload = request.model_dump(exclude_none=True)

        url = f"{self.auth.base_url}/api/v1/pedr/search"

        try:
            # B13.3: Use graph_timeout for graph-enabled queries
            effective_timeout = self.graph_timeout if enable_graph else self.standard_timeout
            headers = self.auth.get_headers()
            resp = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=effective_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return GraphSearchResponse(**data)

        except TraceLabAuthError as exc:
            diagnostic = _safe_exception_diagnostic(exc, exc.status_code)
            logger.warning("PEDR auth failed (%s)", diagnostic)
            return self._fallback_search_response(
                query,
                f"Authentication failed ({diagnostic})",
                exc.status_code,
            )

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            diagnostic = _safe_exception_diagnostic(exc, status_code)
            logger.warning(
                "PEDR search HTTP failure (%s)",
                diagnostic,
            )
            return self._fallback_search_response(
                query,
                f"HTTP failure ({diagnostic})",
                status_code,
            )

        except httpx.TimeoutException as exc:
            diagnostic = _safe_exception_diagnostic(exc)
            logger.warning("PEDR search timeout (%s)", diagnostic)
            return self._fallback_search_response(query, f"Timeout ({diagnostic})")

        except httpx.RequestError as exc:
            diagnostic = _safe_exception_diagnostic(exc)
            logger.warning("PEDR search request failure (%s)", diagnostic)
            return self._fallback_search_response(
                query, f"Request failure ({diagnostic})"
            )

        except Exception as exc:
            diagnostic = _safe_exception_diagnostic(exc)
            logger.warning("PEDR search unexpected failure (%s)", diagnostic)
            return self._fallback_search_response(
                query, f"Unexpected failure ({diagnostic})"
            )

    def _fallback_result(
        self,
        query: str,
        error: str,
        status_code: Optional[int] = None,
    ) -> PreflightResult:
        """Create a fallback result that gracefully degrades to web search."""
        return PreflightResult(
            action=PreflightAction.PROCEED,
            summary=f"Preflight unavailable: {error}",
            top_score=0.0,
            match_count=0,
            matches=[],
            query=query,
            latency_ms=0.0,
            error=error,
        )

    def _fallback_search_response(
        self,
        query: str,
        error: str,
        status_code: Optional[int] = None,
    ) -> GraphSearchResponse:
        """Create a fallback response with empty results."""
        payload = {
            "query": query,
            "results": [],
            "metadata": {
                "graph_enabled": False,
                "graph_candidates_expanded": 0,
                "timings": {},
                "error": error,
            },
            "total": 0,
        }
        if status_code is not None:
            payload["metadata"]["status_code"] = status_code
        return GraphSearchResponse(**payload)
    def should_skip_web_search(self, result: PreflightResult) -> bool:
        """Determine if web search should be skipped based on preflight result.

        Returns True only when there's a high-quality reusable match
        (action=REUSE with similarity >= 85% and quality_gates >= 4).
        """
        return result.action == PreflightAction.REUSE and len(result.matches) > 0

    def get_reusable_evidence(self, result: PreflightResult) -> List[Dict[str, Any]]:
        """Extract evidence from matched missions for reuse.

        Converts key_insights from matched missions into evidence objects
        that can be added to the research state.

        Args:
            result: PreflightResult containing matches.

        Returns:
            List of evidence dictionaries with source, title, content,
            similarity score, and quality gates information.
        """
        evidence = []
        for match in result.matches:
            for insight in match.key_insights:
                insight_text = (
                    insight.get("text", "")
                    if isinstance(insight, dict)
                    else str(insight)
                )
                if not insight_text:
                    continue
                evidence.append({
                    "source": f"PEDR:{match.mission_id}",
                    "title": match.title,
                    "url": f"tracelab://mission/{match.mission_uuid or match.mission_id}",
                    "snippet": insight_text,
                    "content": insight_text,
                    "similarity": match.similarity_score,
                    "quality_gates": match.quality_gates_passed,
                    "reused_from": match.mission_id,
                })
        return evidence

    def get_supplementary_context(
        self, result: PreflightResult
    ) -> Dict[str, Any]:
        """Extract context from matched missions for review action.

        When action=REVIEW, existing research should supplement but not
        replace new web searches. This method extracts context information
        to inform the new research.

        Args:
            result: PreflightResult with review action.

        Returns:
            Dictionary with related_missions, key_topics, and suggested_focus.
        """
        if not result.matches:
            return {}

        related_missions = []
        all_tags = set()
        for match in result.matches:
            related_missions.append({
                "mission_id": match.mission_id,
                "title": match.title,
                "similarity": match.similarity_score,
                "objective": match.objective,
            })
            all_tags.update(match.tags)

        top_match = result.matches[0]
        return {
            "related_missions": related_missions,
            "key_topics": list(all_tags)[:10],
            "suggested_focus": (
                f"Related research found in {top_match.mission_id} ({top_match.title}). "
                f"Focus on aspects not covered by existing research."
            ),
            "top_similarity": result.top_score,
        }


# Module-level singleton for convenience
_default_client: Optional[PEDRClient] = None


def get_default_pedr_client() -> PEDRClient:
    """Get or create the default PEDRClient instance.

    Returns:
        Singleton PEDRClient instance configured from environment.
    """
    global _default_client
    if _default_client is None:
        _default_client = PEDRClient()
    return _default_client


def reset_default_pedr_client() -> None:
    """Reset the default PEDR client instance (useful for testing)."""
    global _default_client
    _default_client = None


__all__ = [
    "PEDRClient",
    "PEDRError",
    "PreflightAction",
    "PreflightMatch",
    "PreflightResult",
    "get_default_pedr_client",
    "reset_default_pedr_client",
]
