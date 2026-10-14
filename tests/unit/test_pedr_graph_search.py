"""Unit tests for PEDRClient graph search."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from deepsearch.tracelab.pedr import PEDRClient

httpx = pytest.importorskip("httpx", reason="httpx required for PEDR graph search tests")


class DummyAuth:
    """Minimal auth stub for PEDR client tests."""

    base_url = "https://tracelab.example"

    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer test-token"}


def _build_response(payload: Dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    response.status_code = 200
    response.text = ""
    return response


def test_search_sends_graph_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}
    response_payload = {
        "query": "graph query",
        "results": [
            {
                "mission_id": "DRM.0.5",
                "title": "Graph Search Result",
                "rrf_score": 0.91,
            }
        ],
        "metadata": {"graph_enabled": True, "graph_candidates_expanded": 12},
        "total": 1,
    }
    response = _build_response(response_payload)

    def fake_post(url: str, **kwargs: Any) -> MagicMock:
        captured["url"] = url
        captured.update(kwargs)
        return response

    monkeypatch.setattr(httpx, "post", fake_post)

    client = PEDRClient(auth=DummyAuth(), timeout=12.5)
    result = client.search(
        "graph query",
        project_id="project-1",
        top_k=7,
        enable_graph=True,
        graph_depth=2,
        graph_decay=0.6,
        graph_edge_types=["cites"],
        graph_weight=0.3,
        include_related=True,
        include_embeddings=True,
    )

    assert captured["url"] == "https://tracelab.example/api/v1/pedr/search"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["json"]["query"] == "graph query"
    assert captured["json"]["project_id"] == "project-1"
    assert captured["json"]["top_k"] == 7
    assert captured["json"]["enable_graph"] is True
    assert captured["json"]["graph_depth"] == 2
    assert captured["json"]["graph_decay"] == 0.6
    assert captured["json"]["graph_edge_types"] == ["cites"]
    assert captured["json"]["graph_weight"] == 0.3
    assert captured["json"]["include_related"] is True
    assert captured["json"]["include_embeddings"] is True

    assert result.results[0].rrf_score == 0.91
    assert result.metadata is not None
    assert result.metadata.graph_enabled is True
    assert result.total == 1


def test_search_returns_empty_results_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx, "post", fake_post)

    client = PEDRClient(auth=DummyAuth())
    result = client.search("timeout query")

    assert result.results == []
    assert result.total == 0
    assert result.metadata is not None
    assert result.metadata.graph_enabled is False
