"""Unit tests for MissionProtocolComplete payload generator."""

from __future__ import annotations

from datetime import datetime, timezone

from deepsearch.agent.state import AgentState
from deepsearch.report.evidence import Evidence
from deepsearch.report.protocol import (
    REQUIRED_COMPLETION_GATES,
    DepthConfigMeta,
    MissionProtocolComplete,
    QualityCheckpoint,
    _build_depth_config_meta,
    _build_quality_checkpoints,
    _derive_summary,
    _derive_title,
    _extract_tags,
    generate_mission_protocol,
)
from deepsearch.report.synthesis import ResearchStatement, Synthesis


class TestQualityCheckpointModel:
    """Tests for the QualityCheckpoint Pydantic model."""

    def test_quality_checkpoint_required_fields(self) -> None:
        """QualityCheckpoint should require gate field."""
        checkpoint = QualityCheckpoint(gate="research_statement")
        assert checkpoint.gate == "research_statement"
        assert checkpoint.status == "pass"  # default

    def test_quality_checkpoint_status_options(self) -> None:
        """QualityCheckpoint should accept pass, fail, skip statuses."""
        pass_cp = QualityCheckpoint(gate="test", status="pass")
        assert pass_cp.status == "pass"

        fail_cp = QualityCheckpoint(gate="test", status="fail")
        assert fail_cp.status == "fail"

        skip_cp = QualityCheckpoint(gate="test", status="skip")
        assert skip_cp.status == "skip"

    def test_quality_checkpoint_optional_fields(self) -> None:
        """QualityCheckpoint optional fields should default to None."""
        checkpoint = QualityCheckpoint(gate="test")
        assert checkpoint.notes is None
        assert checkpoint.validated_by is None
        assert checkpoint.validated_at is None

    def test_quality_checkpoint_with_all_fields(self) -> None:
        """QualityCheckpoint should accept all optional fields."""
        now = datetime.now(timezone.utc)
        checkpoint = QualityCheckpoint(
            gate="synthesis_quality",
            status="pass",
            notes="Validated successfully",
            validated_by="deepsearch-agent",
            validated_at=now,
        )
        assert checkpoint.notes == "Validated successfully"
        assert checkpoint.validated_by == "deepsearch-agent"
        assert checkpoint.validated_at == now

    def test_quality_checkpoint_to_dict(self) -> None:
        """to_dict should return dict with ISO timestamp."""
        now = datetime.now(timezone.utc)
        checkpoint = QualityCheckpoint(
            gate="test",
            status="pass",
            validated_at=now,
        )
        data = checkpoint.to_dict()

        assert data["gate"] == "test"
        assert data["status"] == "pass"
        assert data["validated_at"] == now.isoformat()


class TestMissionProtocolCompleteModel:
    """Tests for the MissionProtocolComplete Pydantic model."""

    def test_protocol_required_fields(self) -> None:
        """MissionProtocolComplete should require essential fields."""
        research_statement = ResearchStatement(
            topic="Test Topic",
            objective="Test Objective",
            scope="Test Scope",
        )
        synthesis = Synthesis(
            key_insights=["Test insight with enough characters here"],
            recommendations=["Test recommendation"],
            next_steps=["Test next step"],
        )

        protocol = MissionProtocolComplete(
            mission_id="TEST-001",
            title="Test Mission",
            research_statement=research_statement,
            synthesis=synthesis,
        )

        assert protocol.mission_id == "TEST-001"
        assert protocol.title == "Test Mission"
        assert protocol.version == "1.0.0"
        assert protocol.status == "complete"

    def test_protocol_optional_fields_default(self) -> None:
        """Optional fields should have sensible defaults."""
        research_statement = ResearchStatement(
            topic="T", objective="O", scope="S"
        )
        synthesis = Synthesis()

        protocol = MissionProtocolComplete(
            mission_id="TEST",
            title="Test",
            research_statement=research_statement,
            synthesis=synthesis,
        )

        assert protocol.summary is None
        assert protocol.project_id is None
        assert protocol.owner is None
        assert protocol.key_questions == []
        assert protocol.evidence == []
        assert protocol.quality_checkpoints == []
        assert protocol.tags == []
        assert protocol.created_at is None
        assert protocol.updated_at is None

    def test_protocol_all_gates_pass_true(self) -> None:
        """all_gates_pass should return True when all 5 gates pass."""
        checkpoints = [
            QualityCheckpoint(gate=gate, status="pass")
            for gate in REQUIRED_COMPLETION_GATES
        ]
        research_statement = ResearchStatement(
            topic="T", objective="O", scope="S"
        )
        synthesis = Synthesis()

        protocol = MissionProtocolComplete(
            mission_id="TEST",
            title="Test",
            research_statement=research_statement,
            synthesis=synthesis,
            quality_checkpoints=checkpoints,
        )

        assert protocol.all_gates_pass() is True

    def test_protocol_all_gates_pass_false_missing(self) -> None:
        """all_gates_pass should return False when gate is missing."""
        checkpoints = [
            QualityCheckpoint(gate="research_statement", status="pass"),
            # Missing other 4 gates
        ]
        research_statement = ResearchStatement(
            topic="T", objective="O", scope="S"
        )
        synthesis = Synthesis()

        protocol = MissionProtocolComplete(
            mission_id="TEST",
            title="Test",
            research_statement=research_statement,
            synthesis=synthesis,
            quality_checkpoints=checkpoints,
        )

        assert protocol.all_gates_pass() is False

    def test_protocol_all_gates_pass_false_failed(self) -> None:
        """all_gates_pass should return False when any gate fails."""
        checkpoints = [
            QualityCheckpoint(gate="research_statement", status="pass"),
            QualityCheckpoint(gate="evidence_links", status="fail"),  # Failed
            QualityCheckpoint(gate="synthesis_quality", status="pass"),
            QualityCheckpoint(gate="traceability", status="pass"),
            QualityCheckpoint(gate="contradictions_resolved", status="pass"),
        ]
        research_statement = ResearchStatement(
            topic="T", objective="O", scope="S"
        )
        synthesis = Synthesis()

        protocol = MissionProtocolComplete(
            mission_id="TEST",
            title="Test",
            research_statement=research_statement,
            synthesis=synthesis,
            quality_checkpoints=checkpoints,
        )

        assert protocol.all_gates_pass() is False

    def test_protocol_to_dict(self) -> None:
        """to_dict should return properly serialized dictionary."""
        now = datetime.now(timezone.utc)
        research_statement = ResearchStatement(
            topic="T", objective="O", scope="S"
        )
        synthesis = Synthesis()
        evidence = [
            Evidence(evidence_id="EV-001", source="test", summary="Test")
        ]
        checkpoints = [
            QualityCheckpoint(gate="test", status="pass")
        ]

        protocol = MissionProtocolComplete(
            mission_id="TEST-001",
            title="Test",
            research_statement=research_statement,
            synthesis=synthesis,
            evidence=evidence,
            quality_checkpoints=checkpoints,
            created_at=now,
            updated_at=now,
        )

        data = protocol.to_dict()

        assert data["mission_id"] == "TEST-001"
        assert data["created_at"] == now.isoformat()
        assert data["updated_at"] == now.isoformat()
        assert isinstance(data["research_statement"], dict)
        assert isinstance(data["synthesis"], dict)
        assert len(data["evidence"]) == 1
        assert len(data["quality_checkpoints"]) == 1


class TestRequiredCompletionGates:
    """Tests for the REQUIRED_COMPLETION_GATES constant."""

    def test_required_gates_count(self) -> None:
        """Should have exactly 5 required gates."""
        assert len(REQUIRED_COMPLETION_GATES) == 5

    def test_required_gates_names(self) -> None:
        """Should include all 5 TraceLab gate names."""
        expected = {
            "research_statement",
            "evidence_links",
            "synthesis_quality",
            "traceability",
            "contradictions_resolved",
        }
        assert set(REQUIRED_COMPLETION_GATES) == expected


class TestBuildQualityCheckpoints:
    """Tests for _build_quality_checkpoints helper function."""

    def test_build_checkpoints_count(self) -> None:
        """Should generate exactly 5 checkpoints."""
        checkpoints = _build_quality_checkpoints()
        assert len(checkpoints) == 5

    def test_build_checkpoints_all_pass(self) -> None:
        """All checkpoints should have status='pass'."""
        checkpoints = _build_quality_checkpoints(
            {
                "mission_objectives": ["Document the API surface"],
                # S83 m06: evidence_links + traceability gates repointed onto the
                # citation evidence the codeagent path produces.
                "synthesis_telemetry": {
                    "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4}
                },
            },
            research_statement=ResearchStatement(
                topic="API surface",
                objective="Document the API surface",
                scope="Public endpoints",
            ),
            synthesis=Synthesis(
                key_insights=["A sufficiently detailed insight that exceeds the minimum length."],
                recommendations=["Ship the documented protocol."],
                next_steps=["Validate the protocol in integration tests."],
            ),
        )
        # S84 m04: contradictions_resolved honestly reports 'skip' (no
        # validator exists — the old unconditional 'pass' was a placeholder
        # lie); every evaluated gate still passes.
        for cp in checkpoints:
            if cp.gate == "contradictions_resolved":
                assert cp.status == "skip"
            else:
                assert cp.status == "pass"

    def test_build_checkpoints_gates(self) -> None:
        """Should include all required gates."""
        checkpoints = _build_quality_checkpoints()
        gates = {cp.gate for cp in checkpoints}
        assert gates == set(REQUIRED_COMPLETION_GATES)

    def test_build_checkpoints_validated_by(self) -> None:
        """All checkpoints should be validated by deepsearch-agent."""
        checkpoints = _build_quality_checkpoints()
        for cp in checkpoints:
            assert cp.validated_by == "deepsearch-agent"

    def test_build_checkpoints_validated_at(self) -> None:
        """All checkpoints should have validated_at timestamp."""
        checkpoints = _build_quality_checkpoints()
        for cp in checkpoints:
            assert cp.validated_at is not None
            assert isinstance(cp.validated_at, datetime)

    def test_build_checkpoints_fail_synthesis_quality_on_structural_failure(self) -> None:
        """Structural validation failures should fail the synthesis_quality gate."""
        checkpoints = _build_quality_checkpoints(
            {
                "mission_objectives": ["Compare vendors"],
                "synthesis_telemetry": {
                    "structural_validation_passed": False,
                    "structural_validation_failed_schemas": ["classification_matrix"],
                    # S83 m06: repointed gates read citation_telemetry (the
                    # synthesis_quality gate still fails on the structural flag).
                    "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4},
                },
                "routing_reason": "structural_validation_failed",
            },
            research_statement=ResearchStatement(
                topic="Vendor comparison",
                objective="Compare vendors",
                scope="Current vendor landscape",
            ),
            synthesis=Synthesis(
                key_insights=["A sufficiently detailed insight that exceeds the minimum length."],
                recommendations=["Use the deterministic structural validator."],
                next_steps=["Retry synthesis with explicit remediation hints."],
            ),
        )
        gate_status = {checkpoint.gate: checkpoint.status for checkpoint in checkpoints}
        assert gate_status["synthesis_quality"] == "fail"


class TestDeriveTitle:
    """Tests for `_derive_title` helper function.

    S80 m02 (PD107) — the helper is now a thin reader of the canonical
    ``state["mission_title"]`` key pinned by ``empty_state()`` in
    deepsearch/agent/state.py. The independent title/objective fallback
    that lived here pre-S80 was the divergence point with
    ``codeagent_finalize_node`` (path 1 fell back to ``mission_id``;
    path 2 fell back to ``objective[:100]`` then to
    ``f"Research Mission: {mission_id}"``). Both readers now share a
    single source of truth, so ``_derive_title`` does not re-derive.
    """

    def test_derive_title_returns_canonical_mission_title_key(self) -> None:
        """The canonical ``state["mission_title"]`` is the only authority."""
        state: AgentState = {
            "mission_title": "API Documentation Research",
            "mission_context": {"title": "Stale legacy title"},
        }
        title = _derive_title(state)
        assert title == "API Documentation Research", (
            "_derive_title must read state['mission_title'] (S80 m02 PD107) "
            "and ignore any pre-S80 fallback resolution"
        )

    def test_derive_title_falls_back_to_mission_id_when_canonical_missing(self) -> None:
        """Hand-constructed states (test fixtures) bypass empty_state's pin.

        The S80 m02 contract documents this fallback as the second-tier reader
        for callers that build ``AgentState`` literals without going through
        ``empty_state()``. Production paths (loader/runner/worker) always
        pin ``mission_title``.
        """
        state: AgentState = {
            "mission_id": "DRM.0.1",
            "mission_context": {},
        }
        title = _derive_title(state)
        assert title == "DRM.0.1"

    def test_derive_title_falls_back_to_research_when_state_empty(self) -> None:
        """Last-ditch fallback when neither canonical nor mission_id is set."""
        state: AgentState = {}
        title = _derive_title(state)
        assert title == "Research"

    def test_derive_title_empty_canonical_falls_through(self) -> None:
        """Empty-string canonical falls through to mission_id (truthy check)."""
        state: AgentState = {
            "mission_title": "",
            "mission_id": "TEST-001",
        }
        title = _derive_title(state)
        assert title == "TEST-001"


class TestDeriveSummary:
    """Tests for _derive_summary helper function."""

    def test_derive_summary_from_context(self) -> None:
        """Should use summary from mission_context if present."""
        state: AgentState = {
            "mission_context": {
                "summary": "This research explores API patterns in depth."
            },
        }
        summary = _derive_summary(state)
        assert summary == "This research explores API patterns in depth."

    def test_derive_summary_truncates_long(self) -> None:
        """Should truncate long summaries to 500 chars."""
        long_summary = "A" * 600
        state: AgentState = {
            "mission_context": {"summary": long_summary},
        }
        summary = _derive_summary(state)
        assert len(summary) == 500

    # ---------------------------------------------------------------------
    # S81 m05 (PD126) — _derive_summary reroute through synthesis_telemetry.
    # Pre-S81 the synthesized branch read state['sources_found'] /
    # summarize_state_findings / reflection_scores — all dead post-S77.
    # Post-S81 the function reads synthesis_telemetry which the CodeAgent
    # finalize node populates with live counts.
    # ---------------------------------------------------------------------

    def test_derive_summary_reads_synthesis_telemetry(self) -> None:
        """Synthesized path reads from synthesis_telemetry post-S81 m05."""
        state: AgentState = {
            "mission_context": {},
            "synthesis_telemetry": {
                "sources_collected": 42,
                "citation_telemetry": {"citations_extracted": 75},
                "deliverable_coverage": {
                    "satisfied": ["Comparison matrix", "Pattern catalog"],
                    "missing": ["Recommendation"],
                    "shortfall_score": 0.333,
                },
            },
        }
        summary = _derive_summary(state)
        assert "75 citations" in summary
        assert "42 sources" in summary
        # 2 satisfied / 3 total → 67%.
        assert "Deliverables: 2/3 satisfied" in summary
        assert "67%" in summary

    def test_derive_summary_handles_telemetry_under_telemetry_key(self) -> None:
        """Some upstream paths nest synthesis_telemetry under state['telemetry']."""
        state: AgentState = {
            "mission_context": {},
            "telemetry": {
                "synthesis_telemetry": {
                    "sources_collected": 12,
                    "citation_telemetry": {"citations_extracted": 30},
                }
            },
        }
        summary = _derive_summary(state)
        assert "30 citations" in summary
        assert "12 sources" in summary

    def test_derive_summary_telemetry_without_coverage_emits_short_form(self) -> None:
        """When deliverable_coverage is absent, the short-form summary fires."""
        state: AgentState = {
            "mission_context": {},
            "synthesis_telemetry": {
                "sources_collected": 8,
                "citation_telemetry": {"citations_extracted": 17},
            },
        }
        summary = _derive_summary(state)
        # S84 m04 review fix: the semicolon phrasing avoids the false
        # provenance claim ("from" implied the citations came from all N
        # collected sources, which include never-opened SERP results).
        assert summary == (
            "Autonomous research mission with 17 citations; 8 sources collected."
        )

    def test_derive_summary_empty_state(self) -> None:
        """Empty state returns the non-empty defensive string post-S81."""
        state: AgentState = {}
        summary = _derive_summary(state)
        # Must NOT read as the pre-S81 zero-cue falsehood
        # ("0 unique concepts ... 0 sources. Coverage: 0%").
        assert summary
        assert "0 unique concepts" not in summary
        assert "Coverage: 0%" not in summary
        # Must be the documented defensive string.
        assert "telemetry not yet populated" in summary

    def test_derive_summary_skips_short(self) -> None:
        """Short explicit summary (<20 chars) falls through to telemetry path."""
        state: AgentState = {
            "mission_context": {"summary": "Short"},
            "synthesis_telemetry": {
                "sources_collected": 5,
                "citation_telemetry": {"citations_extracted": 12},
            },
        }
        summary = _derive_summary(state)
        # Falls through to synthesis_telemetry path (not the short literal).
        assert "Short" != summary
        assert "12 citations" in summary
        assert "5 sources" in summary

    def test_derive_summary_defensive_on_malformed_telemetry(self) -> None:
        """Non-Mapping telemetry / malformed coverage do not crash."""
        state: AgentState = {
            "mission_context": {},
            "synthesis_telemetry": {
                "sources_collected": "not_an_int",
                "citation_telemetry": "not_a_mapping",
                "deliverable_coverage": "not_a_mapping",
            },
        }
        summary = _derive_summary(state)
        # Falls through to the defensive string when all numeric fields
        # coerce to 0 and no deliverable_coverage.
        assert summary
        assert "telemetry not yet populated" in summary


class TestExtractTags:
    """Tests for _extract_tags helper function."""

    def test_extract_tags_base_tags(self) -> None:
        """Should always include base tags."""
        state: AgentState = {}
        tags = _extract_tags(state)
        assert "deepsearch" in tags
        assert "autonomous-research" in tags

    def test_extract_tags_mission_id_prefix(self) -> None:
        """Should extract mission_id prefix as tag."""
        state: AgentState = {"mission_id": "DRM.0.1"}
        tags = _extract_tags(state)
        assert "drm" in tags

    def test_extract_tags_no_mission_id(self) -> None:
        """Should handle missing mission_id."""
        state: AgentState = {}
        tags = _extract_tags(state)
        assert len(tags) == 2  # Only base tags

    def test_extract_tags_simple_mission_id(self) -> None:
        """Should handle mission_id without dots."""
        state: AgentState = {"mission_id": "SIMPLE"}
        tags = _extract_tags(state)
        assert "simple" in tags


class TestGenerateMissionProtocol:
    """Tests for generate_mission_protocol function."""

    def test_generate_protocol_normal(self) -> None:
        """Normal case with all fields populated."""
        state: AgentState = {
            "mission_id": "DRM.0.1",
            # S80 m02 (PD107): canonical title is pinned at state["mission_title"]
            # by empty_state(); the legacy mission_context["title"] read in
            # _derive_title was deleted.
            "mission_title": "API Research Mission",
            "mission_context": {
                "title": "API Research Mission",
                "objective": "Investigate REST patterns",
                "focus": "Python ecosystem",
            },
            "mission_objectives": [
                "Document API endpoints",
                "Catalog authentication methods",
            ],
            "findings": [
                "REST APIs use HTTP methods for CRUD operations",
                "OAuth 2.0 is the most common authentication standard",
            ],
            "sources_found": [
                {
                    "url": "https://restfulapi.net",
                    "title": "REST API Tutorial",
                    "snippet": "A comprehensive guide to REST API design.",
                },
            ],
            "reflection_scores": {
                "overall_coverage_score": 0.85,
            },
            "reflection_result": {
                "decision": "finalize",
                "reasoning": "All objectives addressed",
            },
            # S83 m06: evidence_links + traceability gates read citation evidence.
            "synthesis_telemetry": {
                "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4}
            },
        }

        protocol = generate_mission_protocol(state)

        assert protocol.mission_id == "DRM.0.1"
        assert protocol.title == "API Research Mission"
        assert protocol.status == "complete"
        assert protocol.owner == "deepsearch-agent"
        assert protocol.version == "1.0.0"
        assert protocol.all_gates_pass() is True

        # Verify nested objects
        assert protocol.research_statement.topic == "API Research Mission"
        assert len(protocol.key_questions) >= 1
        assert protocol.synthesis is not None
        assert len(protocol.evidence) >= 1

    def test_generate_protocol_includes_graph_provenance(self) -> None:
        """Graph context should be captured in protocol output."""
        state: AgentState = {
            "mission_id": "GRAPH-PROTOCOL",
            "pedr_result": {
                "graph_context": {
                    "query": "graph query",
                    "results": [
                        {
                            "mission_id": "DRM.0.8",
                            "title": "Graph Research",
                            "rrf_score": 0.91,
                        }
                    ],
                    "metadata": {"graph_enabled": True},
                },
            },
        }

        protocol = generate_mission_protocol(state)

        assert protocol.graph_provenance is not None
        assert protocol.graph_provenance["related_research"][0]["mission_id"] == "DRM.0.8"
        assert protocol.graph_provenance["metadata"]["graph_enabled"] is True

    def test_generate_protocol_includes_runtime_diagnostics_without_pedr(self) -> None:
        """Runtime diagnostics should surface in protocol metadata even without PEDR."""

        state: AgentState = {
            "mission_id": "DIAG-PROTOCOL",
            "slot_coverage": {
                "slots": [
                    {
                        "slot_id": "slot-deliverable-schema-1",
                        "kind": "deliverable_schema",
                        "status": "unaddressed",
                        "source_count": 0,
                        "target": {"schema_id": "deliverable-schema-1", "format": "table"},
                        "reasons": [],
                        "domains": [],
                    }
                ]
            },
            "synthesis_telemetry": {
                "report_render_path": "template_fallback",
                "report_render_reason": "primary_empty_response",
                "finalize_contract_present": True,
                "finalize_has_deliverable_schemas": True,
                "finalize_deliverable_schema_ids": ["deliverable-schema-1"],
            },
        }

        protocol = generate_mission_protocol(state)

        assert protocol.graph_provenance is not None
        diagnostics = protocol.graph_provenance["metadata"]["runtime_diagnostics"]
        assert diagnostics["finalize"]["report_render_path"] == "template_fallback"
        assert diagnostics["finalize"]["deliverable_schema_ids"] == ["deliverable-schema-1"]
        assert diagnostics["coverage"]["failed_slot_ids"] == ["slot-deliverable-schema-1"]
        assert diagnostics["coverage"]["failed_slot_kinds"] == ["deliverable_schema"]

    def test_generate_protocol_minimal_state(self) -> None:
        """Minimal state should still render a protocol.

        S80 m02 (PD107) — when ``state["mission_title"]`` is missing
        (hand-constructed AgentState bypassing ``empty_state()``),
        ``_derive_title`` falls back to ``state["mission_id"]``. The pre-S80
        prefixed form ``"Research Mission: <id>"`` was removed when the
        independent fallback chain was deleted.
        """
        state: AgentState = {
            "mission_id": "MINIMAL",
        }

        protocol = generate_mission_protocol(state)

        assert protocol.mission_id == "MINIMAL"
        assert protocol.title == "MINIMAL"
        assert protocol.all_gates_pass() is False
        # Should have fallback evidence
        assert len(protocol.evidence) >= 1
        # S84 m04: no fabricated placeholder answers — a minimal state with
        # zero findings ships honestly-open questions.
        assert protocol.key_questions
        assert all(q.status == "open" for q in protocol.key_questions)

    def test_generate_protocol_empty_sources(self) -> None:
        """Empty sources should generate fallback evidence."""
        state: AgentState = {
            "mission_id": "NO-SOURCES",
            "sources_found": [],
        }

        protocol = generate_mission_protocol(state)

        # Should have fallback evidence
        assert len(protocol.evidence) == 1
        assert protocol.evidence[0].evidence_id == "EV-001"
        assert protocol.evidence[0].source_type == "web_search"

    def test_generate_protocol_quality_checkpoints(self) -> None:
        """Should have all 5 quality checkpoints passing."""
        state: AgentState = {
            "mission_id": "GATES-TEST",
            "mission_objectives": ["Document protocol gates"],
            # S83 m06: evidence_links + traceability gates read citation evidence.
            "synthesis_telemetry": {
                "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4}
            },
        }

        protocol = generate_mission_protocol(state)

        assert len(protocol.quality_checkpoints) == 5
        # S84 m04: contradictions_resolved is honestly 'skip' (not
        # evaluated); the four real gates pass and all_gates_pass() holds
        # because skip is not a failure.
        for cp in protocol.quality_checkpoints:
            if cp.gate == "contradictions_resolved":
                assert cp.status == "skip"
            else:
                assert cp.status == "pass"
        assert protocol.all_gates_pass() is True

        gates = {cp.gate for cp in protocol.quality_checkpoints}
        assert gates == set(REQUIRED_COMPLETION_GATES)

    def test_generate_protocol_timestamps(self) -> None:
        """Should have created_at and updated_at timestamps."""
        state: AgentState = {
            "mission_id": "TIMESTAMP-TEST",
        }

        protocol = generate_mission_protocol(state)

        assert protocol.created_at is not None
        assert protocol.updated_at is not None
        assert isinstance(protocol.created_at, datetime)
        assert isinstance(protocol.updated_at, datetime)

    def test_generate_protocol_tags(self) -> None:
        """Should include appropriate tags."""
        state: AgentState = {
            "mission_id": "DRM.0.1",
        }

        protocol = generate_mission_protocol(state)

        assert "deepsearch" in protocol.tags
        assert "autonomous-research" in protocol.tags
        assert "drm" in protocol.tags

    def test_generate_protocol_schema_validation(self) -> None:
        """Generated protocol should pass Pydantic validation."""
        state: AgentState = {
            "mission_id": "VALIDATE-TEST",
            "mission_context": {"title": "Validation Test"},
            "findings": ["Test finding with content"],
        }

        protocol = generate_mission_protocol(state)

        # Should not raise
        validated = MissionProtocolComplete.model_validate(
            protocol.model_dump()
        )
        assert validated.mission_id == "VALIDATE-TEST"

    def test_generate_protocol_to_dict_json_serializable(self) -> None:
        """to_dict output should be JSON serializable."""
        import json

        state: AgentState = {
            "mission_id": "JSON-TEST",
        }

        protocol = generate_mission_protocol(state)
        data = protocol.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert json_str
        assert "JSON-TEST" in json_str


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_none_values_in_state(self) -> None:
        """Should handle None values gracefully."""
        state: AgentState = {
            "mission_id": "NONE-TEST",
            "mission_context": None,
            "findings": None,
            "sources_found": None,
            "reflection_scores": None,
        }

        protocol = generate_mission_protocol(state)

        assert protocol.mission_id == "NONE-TEST"
        assert protocol.all_gates_pass() is False

    def test_empty_lists_in_state(self) -> None:
        """Should handle empty lists gracefully."""
        state: AgentState = {
            "mission_id": "EMPTY-TEST",
            "mission_objectives": [],
            "findings": [],
            "sources_found": [],
        }

        protocol = generate_mission_protocol(state)

        assert protocol.mission_id == "EMPTY-TEST"
        # Should have fallback evidence
        assert len(protocol.evidence) >= 1
        # Should have fallback question
        assert len(protocol.key_questions) >= 1

    def test_unicode_content(self) -> None:
        """Should handle unicode content correctly."""
        state: AgentState = {
            "mission_id": "UNICODE-TEST",
            "mission_objectives": ["Investigate unicode support"],
            # S80 m02 (PD107): canonical title pinned at state["mission_title"].
            "mission_title": "研究任务 - API Documentation",
            "mission_context": {
                "title": "研究任务 - API Documentation",
                "objective": "Investigate émojis 🔍 and symbols",
            },
            "findings": [
                "Découverte: API supports UTF-8 encoding",
                "日本語のドキュメントも見つかりました",
            ],
            # S83 m06: evidence_links + traceability gates read citation evidence.
            "synthesis_telemetry": {
                "citation_telemetry": {"citations_extracted": 5, "distinct_urls_cited": 4}
            },
        }

        protocol = generate_mission_protocol(state)

        assert "研究任务" in protocol.title
        assert protocol.all_gates_pass() is True

    def test_very_long_findings(self) -> None:
        """Should handle very long findings."""
        long_finding = "A" * 2000
        state: AgentState = {
            "mission_id": "LONG-TEST",
            "findings": [long_finding],
        }

        protocol = generate_mission_protocol(state)

        # Should not raise, synthesis handles truncation
        assert protocol.synthesis is not None

    def test_special_characters_in_mission_id(self) -> None:
        """Should handle special characters in mission_id."""
        state: AgentState = {
            "mission_id": "TEST-001_alpha.v2",
        }

        protocol = generate_mission_protocol(state)

        assert protocol.mission_id == "TEST-001_alpha.v2"
        # Should extract prefix correctly
        assert "test-001_alpha" in protocol.tags


class TestIntegrationWithComponents:
    """Tests verifying integration with component generators."""

    def test_evidence_from_sources(self) -> None:
        """Evidence should be generated from sources_found.

        S83 m06 decision B: generate_evidence_list is a retained generic
        Evidence-builder (the one carved-out reader of the dead field), so it
        still derives Evidence from a supplied sources list.
        """
        state: AgentState = {
            "mission_id": "EVIDENCE-INT",
            "sources_found": [
                {
                    "url": "https://example.com/article1",
                    "title": "First Article",
                    "snippet": "First article content here.",
                },
                {
                    "url": "https://example.com/article2",
                    "title": "Second Article",
                    "snippet": "Second article content here.",
                },
            ],
        }

        protocol = generate_mission_protocol(state)

        assert len(protocol.evidence) == 2
        assert protocol.evidence[0].evidence_id == "EV-001"
        assert protocol.evidence[1].evidence_id == "EV-002"

    def test_key_questions_from_objectives(self) -> None:
        """KeyQuestions should be generated from mission_objectives."""
        state: AgentState = {
            "mission_id": "QUESTIONS-INT",
            "mission_objectives": [
                "Document API endpoints",
                "Identify authentication patterns",
            ],
            "findings": [
                "API uses REST endpoints for data access",
                "OAuth 2.0 authentication is recommended",
            ],
        }

        protocol = generate_mission_protocol(state)

        assert len(protocol.key_questions) >= 2
        # At least one should be answered
        answered = [q for q in protocol.key_questions if q.status == "answered"]
        assert len(answered) >= 1

    def test_synthesis_from_findings(self) -> None:
        """Synthesis should include key insights from findings."""
        state: AgentState = {
            "mission_id": "SYNTHESIS-INT",
            "findings": [
                "This is a key finding with enough content to be selected as an insight.",
            ],
        }

        protocol = generate_mission_protocol(state)

        assert protocol.synthesis is not None
        assert len(protocol.synthesis.key_insights) >= 1

    def test_research_statement_from_context(self) -> None:
        """ResearchStatement should be derived from mission_context."""
        state: AgentState = {
            "mission_id": "RS-INT",
            "mission_context": {
                "title": "API Research",
                "objective": "Investigate REST patterns",
                "focus": "Python ecosystem",
            },
        }

        protocol = generate_mission_protocol(state)

        assert protocol.research_statement.topic == "API Research"
        assert "Investigate" in protocol.research_statement.objective


# --- Depth Config Tests (B10.11) ---


class TestDepthConfigMetaModel:
    """Tests for the DepthConfigMeta Pydantic model."""

    def test_depth_config_meta_required_fields(self) -> None:
        """DepthConfigMeta should require all config fields."""
        config = DepthConfigMeta(
            tier="alpha",
            max_loops=6,
            min_loops=4,
            max_sources=25,
            convergence_threshold=0.03,
            source_quality_floor=0.6,
        )
        assert config.tier == "alpha"
        assert config.max_loops == 6
        assert config.min_loops == 4
        assert config.max_sources == 25
        assert config.convergence_threshold == 0.03
        assert config.source_quality_floor == 0.6

    def test_depth_config_meta_defaults(self) -> None:
        """DepthConfigMeta should have correct defaults."""
        config = DepthConfigMeta(
            tier="baseline",
            max_loops=3,
            min_loops=2,
            max_sources=15,
            convergence_threshold=0.05,
            source_quality_floor=0.5,
        )
        assert config.require_source_diversity is False
        assert config.enable_contradiction_detection is False

    def test_depth_config_meta_alpha_features(self) -> None:
        """DepthConfigMeta should capture Alpha-specific features."""
        config = DepthConfigMeta(
            tier="alpha",
            max_loops=6,
            min_loops=4,
            max_sources=25,
            convergence_threshold=0.03,
            source_quality_floor=0.6,
            require_source_diversity=True,
            enable_contradiction_detection=True,
        )
        assert config.require_source_diversity is True
        assert config.enable_contradiction_detection is True

    def test_depth_config_meta_to_dict(self) -> None:
        """to_dict should return properly serialized dictionary."""
        config = DepthConfigMeta(
            tier="deep",
            max_loops=5,
            min_loops=3,
            max_sources=20,
            convergence_threshold=0.04,
            source_quality_floor=0.55,
        )
        data = config.to_dict()

        assert data["tier"] == "deep"
        assert data["max_loops"] == 5
        assert data["source_quality_floor"] == 0.55


class TestBuildDepthConfigMeta:
    """Tests for _build_depth_config_meta helper function."""

    def test_build_depth_config_from_state(self) -> None:
        """Should build DepthConfigMeta from state correctly."""
        state: AgentState = {
            "research_depth": "alpha",
            "depth_config": {
                "max_loops": 6,
                "min_loops": 4,
                "max_sources": 25,
                "convergence_threshold": 0.03,
                "source_quality_floor": 0.6,
                "require_source_diversity": True,
                "enable_contradiction_detection": True,
            },
        }

        config = _build_depth_config_meta(state)

        assert config is not None
        assert config.tier == "alpha"
        assert config.max_loops == 6
        assert config.source_quality_floor == 0.6
        assert config.require_source_diversity is True
        assert config.enable_contradiction_detection is True

    def test_build_depth_config_none_when_missing(self) -> None:
        """Should return None when depth info is missing."""
        state: AgentState = {
            "mission_id": "TEST",
        }

        config = _build_depth_config_meta(state)

        assert config is None

    def test_build_depth_config_none_when_no_depth(self) -> None:
        """Should return None when research_depth is missing."""
        state: AgentState = {
            "depth_config": {"max_loops": 3},
        }

        config = _build_depth_config_meta(state)

        assert config is None

    def test_build_depth_config_baseline(self) -> None:
        """Should build DepthConfigMeta for baseline tier."""
        state: AgentState = {
            "research_depth": "baseline",
            "depth_config": {
                "max_loops": 3,
                "min_loops": 2,
                "max_sources": 15,
                "convergence_threshold": 0.05,
                "source_quality_floor": 0.5,
            },
        }

        config = _build_depth_config_meta(state)

        assert config is not None
        assert config.tier == "baseline"
        assert config.max_loops == 3
        assert config.require_source_diversity is False


class TestProtocolWithDepthConfig:
    """Tests for protocol generation with depth config."""

    def test_protocol_includes_depth_config_for_alpha(self) -> None:
        """Protocol should include depth_config for Alpha missions."""
        state: AgentState = {
            "mission_id": "DEPTH-ALPHA",
            "mission_context": {"title": "Alpha Research"},
            "research_depth": "alpha",
            "depth_config": {
                "max_loops": 6,
                "min_loops": 4,
                "max_sources": 25,
                "convergence_threshold": 0.03,
                "source_quality_floor": 0.6,
                "require_source_diversity": True,
                "enable_contradiction_detection": True,
            },
        }

        protocol = generate_mission_protocol(state)

        assert protocol.depth_config is not None
        assert protocol.depth_config.tier == "alpha"
        assert protocol.depth_config.enable_contradiction_detection is True

    def test_protocol_includes_depth_config_for_baseline(self) -> None:
        """Protocol should include depth_config for Baseline missions."""
        state: AgentState = {
            "mission_id": "DEPTH-BASELINE",
            "mission_context": {"title": "Baseline Research"},
            "research_depth": "baseline",
            "depth_config": {
                "max_loops": 3,
                "min_loops": 2,
                "max_sources": 15,
                "convergence_threshold": 0.05,
                "source_quality_floor": 0.5,
            },
        }

        protocol = generate_mission_protocol(state)

        assert protocol.depth_config is not None
        assert protocol.depth_config.tier == "baseline"
        assert protocol.depth_config.max_loops == 3

    def test_protocol_depth_config_none_when_missing(self) -> None:
        """Protocol should have None depth_config when info is missing."""
        state: AgentState = {
            "mission_id": "NO-DEPTH",
        }

        protocol = generate_mission_protocol(state)

        assert protocol.depth_config is None

    def test_protocol_to_dict_includes_depth_config(self) -> None:
        """to_dict should include depth_config when present."""
        state: AgentState = {
            "mission_id": "DICT-DEPTH",
            "research_depth": "deep",
            "depth_config": {
                "max_loops": 5,
                "min_loops": 3,
                "max_sources": 20,
                "convergence_threshold": 0.04,
                "source_quality_floor": 0.55,
            },
        }

        protocol = generate_mission_protocol(state)
        data = protocol.to_dict()

        assert "depth_config" in data
        assert data["depth_config"]["tier"] == "deep"
        assert data["depth_config"]["max_loops"] == 5

    def test_protocol_json_serializable_with_depth_config(self) -> None:
        """Protocol with depth_config should be JSON serializable."""
        import json

        state: AgentState = {
            "mission_id": "JSON-DEPTH",
            "research_depth": "alpha",
            "depth_config": {
                "max_loops": 6,
                "min_loops": 4,
                "max_sources": 25,
                "convergence_threshold": 0.03,
                "source_quality_floor": 0.6,
                "require_source_diversity": True,
                "enable_contradiction_detection": True,
            },
        }

        protocol = generate_mission_protocol(state)
        data = protocol.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert "alpha" in json_str
        assert "depth_config" in json_str
