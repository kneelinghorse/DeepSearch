"""Integration tests for PEDR graph search with TraceLab."""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from deepsearch.tracelab.pedr import PEDRClient

httpx = pytest.importorskip("httpx", reason="httpx required for PEDR integration tests")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a dev dependency
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _env_ready() -> bool:
    required = ("TRACELAB_BASE_URL", "TRACELAB_USERNAME", "TRACELAB_PASSWORD")
    return all(os.getenv(key) for key in required)


def test_pedr_graph_search_live() -> None:
    """Live graph search should return related entities and metadata."""
    _load_env()
    if os.getenv("RUN_TRACELAB_INTEGRATION", "").lower() != "true":
        pytest.skip("Set RUN_TRACELAB_INTEGRATION=true to enable live TraceLab tests")
    if not _env_ready():
        pytest.skip("TraceLab credentials missing from environment")

    query = os.getenv("TRACELAB_GRAPH_TEST_QUERY", "LangGraph")
    client = PEDRClient()

    response = client.search(
        query,
        enable_graph=True,
        include_related=True,
        include_embeddings=False,
        top_k=5,
    )

    assert response.metadata is not None
    assert response.metadata.graph_enabled is True
    assert response.metadata.graph_candidates_expanded is not None
    assert response.results

    related_entities = [
        entity
        for result in response.results
        for entity in result.related_entities
    ]
    assert related_entities, "Expected related_entities when include_related=True"


def test_pedr_graph_search_graceful_degradation() -> None:
    """Unreachable graph layer should return empty results."""

    class StubAuth:
        base_url = "http://127.0.0.1:9"

        def get_headers(self) -> Dict[str, str]:
            return {"Authorization": "Bearer test-token"}

    client = PEDRClient(auth=StubAuth(), timeout=1.0)
    response = client.search(
        "graph query",
        enable_graph=True,
        include_related=True,
        include_embeddings=False,
    )

    assert response.results == []
    assert response.metadata is not None
    assert response.metadata.graph_enabled is False
