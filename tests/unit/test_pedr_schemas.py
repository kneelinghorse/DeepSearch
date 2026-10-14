"""Unit tests for PEDR graph search schemas."""

from __future__ import annotations

from deepsearch.tracelab.schemas import (
    GraphSearchRequest,
    GraphSearchResponse,
)


class TestGraphSearchRequest:
    """Tests for GraphSearchRequest schema."""

    def test_accepts_graph_params_camel_case(self) -> None:
        payload = {
            "query": "graph search test",
            "enableGraph": True,
            "graphDepth": 2,
            "graphDecay": 0.7,
            "graphEdgeTypes": ["cites", "mentions"],
            "graphWeight": 0.4,
            "includeRelated": True,
            "includeEmbeddings": True,
        }

        request = GraphSearchRequest(**payload)

        assert request.query == "graph search test"
        assert request.enable_graph is True
        assert request.graph_depth == 2
        assert request.graph_decay == 0.7
        assert request.graph_edge_types == ["cites", "mentions"]
        assert request.graph_weight == 0.4
        assert request.include_related is True
        assert request.include_embeddings is True

    def test_graph_edge_types_accepts_string(self) -> None:
        request = GraphSearchRequest(
            query="edge type test",
            graph_edge_types="cites",
        )
        assert request.graph_edge_types == ["cites"]


class TestGraphSearchResponse:
    """Tests for GraphSearchResponse schema."""

    def test_parses_graph_response_payload(self) -> None:
        payload = {
            "query": "graph search response",
            "results": [
                {
                    "mission_id": "DRM.0.5",
                    "title": "Graph search mission",
                    "objective": "Validate graph search schema models",
                    "rrf_score": 0.92,
                    "layer_ranks": {"embedding": 1, "keyword": 3, "graph": 2},
                    "relatedEntities": [
                        {
                            "entityId": "entity-01",
                            "label": "LangGraph",
                            "type": "framework",
                            "score": 0.88,
                            "missionId": "DRM.0.4",
                        }
                    ],
                    "embedding": [0.1, 0.2, 0.3],
                    "embedding_score": 0.73,
                    "embeddingModel": "text-embedding-3-small",
                }
            ],
            "metadata": {
                "graphEnabled": True,
                "graphCandidatesExpanded": 42,
                "timings": {"graph_ms": 12.5, "total_ms": 55.0},
            },
        }

        response = GraphSearchResponse(**payload)

        assert response.query == "graph search response"
        assert response.metadata is not None
        assert response.metadata.graph_enabled is True
        assert response.metadata.graph_candidates_expanded == 42
        assert response.metadata.timings["graph_ms"] == 12.5

        result = response.results[0]
        assert result.rrf_score == 0.92
        assert result.layer_ranks["graph"] == 2
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.embedding_score == 0.73
        assert result.embedding_model == "text-embedding-3-small"

        entity = result.related_entities[0]
        assert entity.entity_id == "entity-01"
        assert entity.label == "LangGraph"
        assert entity.entity_type == "framework"
        assert entity.score == 0.88
        assert entity.mission_id == "DRM.0.4"
