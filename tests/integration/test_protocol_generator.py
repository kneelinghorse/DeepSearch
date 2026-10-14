"""Integration tests for MissionProtocolComplete payload generator.

Tests cover end-to-end scenarios with realistic AgentState fixtures,
verifying the complete protocol generation workflow from state to
TraceLab-ready payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from deepsearch.agent.state import AgentState
from deepsearch.report.protocol import (
    MissionProtocolComplete,
    REQUIRED_COMPLETION_GATES,
    generate_mission_protocol,
)


class TestRealisticAgentState:
    """Tests with realistic AgentState from actual research missions."""

    @pytest.fixture
    def realistic_state(self) -> AgentState:
        """Create a realistic AgentState from a completed research mission."""
        return {
            "mission_id": "DRM.0.1",
            # S80 m02 (PD107): canonical title pinned at state["mission_title"].
            "mission_title": "Gemini Deep Search CLI Research",
            "project_id": "deepsearch",
            "mission_context": {
                "title": "Gemini Deep Search CLI Research",
                "objective": "Investigate Google's Deep Research CLI capabilities for macOS development",
                "focus": "macOS environment, CLI tools, API integration",
                "background": "As part of the DeepSearch project, we need to understand competing "
                "research automation tools, particularly Google's Gemini Deep Research feature.",
                "summary": "Comprehensive research into Google Gemini's CLI-based research capabilities "
                "and their potential integration patterns.",
            },
            "mission_objectives": [
                "Document available CLI commands and flags for Gemini Deep Research",
                "Catalog API endpoints and authentication methods",
                "Identify limitations and constraints of the CLI approach",
                "Compare capabilities with DeepSearch's approach",
            ],
            "findings": [
                "Google Gemini provides Deep Research through the web interface, with CLI access via the gemini CLI tool.",
                "The gemini CLI requires OAuth2 authentication through a Google Cloud project.",
                "Deep Research operations are rate-limited to prevent abuse, with tier-based quotas.",
                "Results are returned in JSON format with structured fields for findings and sources.",
                "The API supports both streaming and batch modes for large research queries.",
                "Maximum context window is 1 million tokens, allowing extensive document analysis.",
                "Integration with Google Workspace is seamless for enterprise users.",
            ],
            # S83 m06: evidence_links + traceability gates read citation evidence.
            "synthesis_telemetry": {
                "citation_telemetry": {"citations_extracted": 7, "distinct_urls_cited": 7}
            },
            "sources_found": [
                {
                    "url": "https://ai.google.dev/gemini-api/docs",
                    "title": "Gemini API Documentation",
                    "snippet": "Official documentation for the Gemini API including authentication and rate limits.",
                    "retrieved_at": "2024-12-04T10:30:00Z",
                },
                {
                    "url": "https://cloud.google.com/gemini/docs/cli",
                    "title": "Gemini CLI Reference",
                    "snippet": "Command-line interface reference for Google Gemini with examples and best practices.",
                    "retrieved_at": "2024-12-04T10:35:00Z",
                },
                {
                    "url": "https://developers.google.com/identity/protocols/oauth2",
                    "title": "OAuth 2.0 for Google APIs",
                    "snippet": "How to use OAuth 2.0 to access Google APIs including Gemini.",
                    "retrieved_at": "2024-12-04T10:40:00Z",
                },
                {
                    "url": "https://blog.google/technology/ai/gemini-deep-research/",
                    "title": "Introducing Gemini Deep Research",
                    "snippet": "Blog post announcing Deep Research feature with capabilities overview.",
                    "retrieved_at": "2024-12-04T10:45:00Z",
                },
            ],
            "reflection_result": {
                "decision": "finalize",
                "reasoning": "All four objectives have been addressed with supporting evidence. "
                "CLI commands documented, API endpoints identified, limitations cataloged, "
                "and comparison points noted.",
                "criteria_coverage": {
                    "OBJ-1": "complete",
                    "OBJ-2": "complete",
                    "OBJ-3": "complete",
                    "OBJ-4": "partial",
                },
            },
            "reflection_scores": {
                "overall_coverage_score": 0.85,
                "is_mission_complete": True,
                "per_objective_scores": [
                    {"objective_id": "OBJ-1", "score": 0.9},
                    {"objective_id": "OBJ-2", "score": 0.85},
                    {"objective_id": "OBJ-3", "score": 0.8},
                    {"objective_id": "OBJ-4", "score": 0.7},
                ],
            },
            "loop_count": 3,
            "max_loops": 5,
        }

    def test_generate_protocol_from_realistic_state(
        self, realistic_state: AgentState
    ) -> None:
        """Should generate valid protocol from realistic research state."""
        protocol = generate_mission_protocol(realistic_state)

        # Verify mission identification
        assert protocol.mission_id == "DRM.0.1"
        assert protocol.title == "Gemini Deep Search CLI Research"
        assert protocol.status == "complete"

        # Verify research statement
        assert protocol.research_statement.topic == "Gemini Deep Search CLI Research"
        assert "Investigate" in protocol.research_statement.objective
        assert protocol.research_statement.passes_quality_gate()

        # Verify key questions generated from objectives
        assert len(protocol.key_questions) >= 4
        answered_count = sum(
            1 for q in protocol.key_questions if q.status == "answered"
        )
        assert answered_count >= 1, "At least one question must be answered"

        # Verify synthesis quality
        assert len(protocol.synthesis.key_insights) >= 1
        assert len(protocol.synthesis.recommendations) >= 1
        assert len(protocol.synthesis.next_steps) >= 1
        assert protocol.synthesis.passes_quality_gate()

        # Verify evidence from sources (generate_evidence_list retained, decision B)
        assert len(protocol.evidence) == 4
        for ev in protocol.evidence:
            assert ev.evidence_id.startswith("EV-")
            assert ev.source_type == "web_search"

        # Verify all gates pass (evidence_links + traceability read citation_telemetry)
        assert protocol.all_gates_pass()

    def test_protocol_json_serialization(
        self, realistic_state: AgentState
    ) -> None:
        """Protocol should serialize to valid JSON."""
        protocol = generate_mission_protocol(realistic_state)
        data = protocol.to_dict()

        # Should not raise
        json_str = json.dumps(data, indent=2)
        assert json_str

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["mission_id"] == "DRM.0.1"
        assert len(parsed["evidence"]) == 4
        assert len(parsed["quality_checkpoints"]) == 5

    def test_protocol_pydantic_round_trip(
        self, realistic_state: AgentState
    ) -> None:
        """Protocol should validate through Pydantic round-trip."""
        protocol = generate_mission_protocol(realistic_state)

        # Dump and re-validate
        data = protocol.model_dump()
        validated = MissionProtocolComplete.model_validate(data)

        assert validated.mission_id == protocol.mission_id
        assert len(validated.evidence) == len(protocol.evidence)
        assert len(validated.key_questions) == len(protocol.key_questions)


class TestMinimalAgentState:
    """Tests with minimal AgentState to verify fallback behavior."""

    def test_minimal_state_produces_valid_protocol(self) -> None:
        """Minimal state should still produce a protocol.

        S80 m02 (PD107) — when ``state["mission_title"]`` is missing
        (hand-constructed AgentState bypassing ``empty_state()``),
        ``_derive_title`` falls back to ``state["mission_id"]``.
        """
        state: AgentState = {"mission_id": "MINIMAL-001"}

        protocol = generate_mission_protocol(state)

        assert protocol.mission_id == "MINIMAL-001"
        assert protocol.title == "MINIMAL-001"
        assert protocol.all_gates_pass() is False

        # Should have fallback evidence
        assert len(protocol.evidence) >= 1

        # Should have a fallback question. S84 m04: with zero findings it
        # stays honestly open — no fabricated placeholder answer.
        assert len(protocol.key_questions) >= 1
        assert all(q.status == "open" for q in protocol.key_questions)

        # Should have valid synthesis
        assert len(protocol.synthesis.key_insights) >= 1
        assert len(protocol.synthesis.recommendations) >= 1
        assert len(protocol.synthesis.next_steps) >= 1

    def test_empty_state_produces_valid_protocol(self) -> None:
        """Empty dict should still produce a protocol with defaults."""
        state: AgentState = {}

        protocol = generate_mission_protocol(state)

        assert protocol.mission_id == "UNKNOWN"
        assert protocol.all_gates_pass() is False


class TestQualityGateIntegration:
    """Tests verifying quality gate requirements are met."""

    def test_all_five_gates_present(self) -> None:
        """Protocol should have all 5 required quality gates."""
        state: AgentState = {"mission_id": "GATES-TEST"}

        protocol = generate_mission_protocol(state)

        gates = {cp.gate for cp in protocol.quality_checkpoints}
        expected_gates = set(REQUIRED_COMPLETION_GATES)

        assert gates == expected_gates

    def test_all_gates_pass_status(self) -> None:
        """All quality gates should have status='pass'."""
        state: AgentState = {
            "mission_id": "PASS-TEST",
            "mission_objectives": ["Validate gate wiring"],
            # S83 m06: evidence_links + traceability gates read citation evidence.
            "synthesis_telemetry": {
                "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4}
            },
        }

        protocol = generate_mission_protocol(state)

        # S84 m04: contradictions_resolved is honestly 'skip'; the four
        # evaluated gates pass and all_gates_pass() holds (skip != fail).
        for cp in protocol.quality_checkpoints:
            expected = "skip" if cp.gate == "contradictions_resolved" else "pass"
            assert cp.status == expected, f"Gate {cp.gate} should be {expected}"
        assert protocol.all_gates_pass() is True

    def test_research_statement_gate_satisfied(self) -> None:
        """Research statement gate requirements should be met."""
        state: AgentState = {
            "mission_id": "RS-GATE-TEST",
            "mission_context": {
                "title": "Test Research",
                "objective": "Investigate something important",
                "focus": "Specific area of focus",
            },
        }

        protocol = generate_mission_protocol(state)

        # Verify research_statement quality gate would pass
        rs = protocol.research_statement
        assert rs.topic and len(rs.topic.strip()) > 0
        assert rs.objective and len(rs.objective.strip()) > 0
        assert rs.scope and len(rs.scope.strip()) > 0
        assert rs.passes_quality_gate()

    def test_evidence_links_gate_satisfied(self) -> None:
        """Evidence links gate requires at least 1 evidence item."""
        state: AgentState = {
            "mission_id": "EV-GATE-TEST",
            "sources_found": [
                {
                    "url": "https://example.com",
                    "title": "Test Source",
                    "snippet": "Test content for evidence.",
                }
            ],
        }

        protocol = generate_mission_protocol(state)

        assert len(protocol.evidence) >= 1

    def test_synthesis_quality_gate_satisfied(self) -> None:
        """Synthesis quality gate requires key_insights, recommendations, next_steps."""
        state: AgentState = {
            "mission_id": "SYN-GATE-TEST",
            "findings": [
                "This is a substantial finding with enough characters to be selected as a key insight."
            ],
        }

        protocol = generate_mission_protocol(state)

        syn = protocol.synthesis
        assert len(syn.key_insights) >= 1
        assert len(syn.recommendations) >= 1
        assert len(syn.next_steps) >= 1
        assert syn.passes_quality_gate()


class TestTraceLabCompatibility:
    """Tests verifying TraceLab schema compatibility."""

    def test_protocol_schema_version(self) -> None:
        """Protocol should use version 1.0.0."""
        state: AgentState = {"mission_id": "VERSION-TEST"}

        protocol = generate_mission_protocol(state)

        assert protocol.version == "1.0.0"

    def test_protocol_owner_is_deepsearch(self) -> None:
        """Protocol owner should be deepsearch-agent."""
        state: AgentState = {"mission_id": "OWNER-TEST"}

        protocol = generate_mission_protocol(state)

        assert protocol.owner == "deepsearch-agent"

    def test_protocol_has_timestamps(self) -> None:
        """Protocol should have created_at and updated_at timestamps."""
        state: AgentState = {"mission_id": "TIMESTAMP-TEST"}

        protocol = generate_mission_protocol(state)

        assert protocol.created_at is not None
        assert protocol.updated_at is not None

    def test_protocol_tags_include_deepsearch(self) -> None:
        """Protocol tags should include 'deepsearch'."""
        state: AgentState = {"mission_id": "TAGS-TEST"}

        protocol = generate_mission_protocol(state)

        assert "deepsearch" in protocol.tags
        assert "autonomous-research" in protocol.tags


class TestEdgeCasesIntegration:
    """Integration tests for edge cases."""

    def test_very_long_findings(self) -> None:
        """Should handle very long findings gracefully."""
        long_finding = "X" * 5000
        state: AgentState = {
            "mission_id": "LONG-FINDINGS",
            "findings": [long_finding],
        }

        protocol = generate_mission_protocol(state)

        # Should not raise
        assert protocol.synthesis is not None

    def test_many_sources(self) -> None:
        """Should handle many sources efficiently."""
        sources = [
            {"url": f"https://example{i}.com", "title": f"Source {i}", "snippet": f"Content {i}"}
            for i in range(100)
        ]
        state: AgentState = {
            "mission_id": "MANY-SOURCES",
            "sources_found": sources,
        }

        protocol = generate_mission_protocol(state)

        assert len(protocol.evidence) == 100

    def test_unicode_content(self) -> None:
        """Should handle unicode content correctly."""
        state: AgentState = {
            "mission_id": "UNICODE-INT-TEST",
            # S80 m02 (PD107): canonical title pinned at state["mission_title"].
            "mission_title": "研究任务: API Documentation",
            "mission_objectives": ["Investigate unicode content"],
            "mission_context": {
                "title": "研究任务: API Documentation",
                "objective": "Investigate émojis and symbols",
            },
            "findings": [
                "Discovery: API supports UTF-8 encoding properly",
                "日本語のドキュメントも対応しています",
            ],
            # S83 m06: evidence_links + traceability gates read citation evidence.
            "synthesis_telemetry": {
                "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4}
            },
        }

        protocol = generate_mission_protocol(state)

        assert "研究任务" in protocol.title
        assert protocol.all_gates_pass()

        # Should serialize to JSON without errors
        json_str = json.dumps(protocol.to_dict(), ensure_ascii=False)
        assert "研究任务" in json_str

    def test_special_characters_in_urls(self) -> None:
        """Should handle special characters in source URLs."""
        state: AgentState = {
            "mission_id": "SPECIAL-URLS",
            "mission_objectives": ["Verify URL handling"],
            # S83 m06: evidence_links + traceability gates read citation evidence.
            "synthesis_telemetry": {
                "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4}
            },
            "sources_found": [
                {
                    "url": "https://example.com/path?query=value&other=test",
                    "title": "Query Parameters",
                    "snippet": "URL with query parameters.",
                },
                {
                    "url": "https://example.com/path#section",
                    "title": "Fragment",
                    "snippet": "URL with fragment.",
                },
                {
                    "url": "https://example.com/path%20with%20spaces",
                    "title": "Encoded",
                    "snippet": "URL with encoded spaces.",
                },
            ],
        }

        protocol = generate_mission_protocol(state)

        assert len(protocol.evidence) == 3
        assert protocol.all_gates_pass()
