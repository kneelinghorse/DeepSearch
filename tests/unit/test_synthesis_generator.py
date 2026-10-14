"""Unit tests for TraceLab Synthesis and ResearchStatement generation from AgentState."""

from __future__ import annotations

import pytest

from deepsearch.agent.state import AgentState
from deepsearch.report.synthesis import (
    MIN_INSIGHT_LENGTH,
    ResearchStatement,
    Synthesis,
    _derive_next_steps,
    _derive_recommendations,
    _derive_research_objective,
    _derive_research_scope,
    _derive_research_topic,
    _select_key_insights,
    generate_research_statement,
    generate_synthesis,
)


class TestSynthesisModel:
    """Tests for the Synthesis Pydantic model."""

    def test_synthesis_default_empty_lists(self) -> None:
        """Synthesis should default all list fields to empty."""
        synthesis = Synthesis()
        assert synthesis.key_insights == []
        assert synthesis.surprising_findings == []
        assert synthesis.contradictory_information == []
        assert synthesis.contradiction_resolutions == []
        assert synthesis.recommendations == []
        assert synthesis.next_steps == []

    def test_synthesis_with_all_fields(self) -> None:
        """Synthesis should accept all fields."""
        synthesis = Synthesis(
            key_insights=["Insight 1 with enough characters to pass the gate"],
            surprising_findings=["Surprising finding"],
            contradictory_information=["Contradiction"],
            contradiction_resolutions=["Resolution"],
            recommendations=["Recommendation"],
            next_steps=["Next step"],
        )
        assert len(synthesis.key_insights) == 1
        assert len(synthesis.surprising_findings) == 1
        assert len(synthesis.contradictory_information) == 1
        assert len(synthesis.contradiction_resolutions) == 1
        assert len(synthesis.recommendations) == 1
        assert len(synthesis.next_steps) == 1

    def test_synthesis_to_dict(self) -> None:
        """Synthesis to_dict should return all fields."""
        synthesis = Synthesis(
            key_insights=["Test insight with at least 40 characters for gate"],
            recommendations=["Test recommendation"],
            next_steps=["Test next step"],
        )
        result = synthesis.to_dict()
        assert "key_insights" in result
        assert "surprising_findings" in result
        assert "contradictory_information" in result
        assert "contradiction_resolutions" in result
        assert "recommendations" in result
        assert "next_steps" in result


class TestQualityGate:
    """Tests for Synthesis quality gate validation."""

    def test_passes_quality_gate_valid(self) -> None:
        """Synthesis with all requirements should pass gate."""
        synthesis = Synthesis(
            key_insights=["This insight has at least 40 characters as required"],
            recommendations=["A recommendation"],
            next_steps=["A next step"],
        )
        assert synthesis.passes_quality_gate() is True

    def test_fails_quality_gate_no_insights(self) -> None:
        """Synthesis without key_insights should fail gate."""
        synthesis = Synthesis(
            key_insights=[],
            recommendations=["Recommendation"],
            next_steps=["Next step"],
        )
        assert synthesis.passes_quality_gate() is False

    def test_fails_quality_gate_short_insight(self) -> None:
        """Synthesis with short first insight should fail gate."""
        synthesis = Synthesis(
            key_insights=["Too short"],  # Less than 40 chars
            recommendations=["Recommendation"],
            next_steps=["Next step"],
        )
        assert synthesis.passes_quality_gate() is False

    def test_fails_quality_gate_no_recommendations(self) -> None:
        """Synthesis without recommendations should fail gate."""
        synthesis = Synthesis(
            key_insights=["This insight has at least 40 characters as required"],
            recommendations=[],
            next_steps=["Next step"],
        )
        assert synthesis.passes_quality_gate() is False

    def test_fails_quality_gate_no_next_steps(self) -> None:
        """Synthesis without next_steps should fail gate."""
        synthesis = Synthesis(
            key_insights=["This insight has at least 40 characters as required"],
            recommendations=["Recommendation"],
            next_steps=[],
        )
        assert synthesis.passes_quality_gate() is False

    def test_passes_quality_gate_exactly_40_chars(self) -> None:
        """Insight with exactly 40 characters should pass."""
        # Create exactly 40 character insight
        insight = "A" * 40
        synthesis = Synthesis(
            key_insights=[insight],
            recommendations=["Recommendation"],
            next_steps=["Next step"],
        )
        assert len(synthesis.key_insights[0]) == 40
        assert synthesis.passes_quality_gate() is True


class TestSelectKeyInsights:
    """Tests for _select_key_insights function."""

    def test_empty_findings(self) -> None:
        """Empty findings should return empty list."""
        result = _select_key_insights([])
        assert result == []

    def test_filters_short_findings(self) -> None:
        """Findings shorter than MIN_INSIGHT_LENGTH should be filtered."""
        findings = [
            "Short",
            "A" * (MIN_INSIGHT_LENGTH - 1),
            "This finding has enough characters to pass the length filter",
        ]
        result = _select_key_insights(findings)
        assert len(result) == 1
        assert "enough characters" in result[0]

    def test_deduplicates_by_prefix(self) -> None:
        """Duplicate findings (by 50-char prefix) should be removed."""
        # Two findings with same 50-char prefix
        findings = [
            "This is a long finding that starts with these words and continues differently here",
            "This is a long finding that starts with these words but ends with something else",
        ]
        result = _select_key_insights(findings)
        assert len(result) == 1

    def test_emits_all_unique_findings_post_s75(self) -> None:
        """S75 m01 removed the legacy 5-cap; all unique findings flow through."""
        findings = [
            f"Finding {chr(ord('A') + i)}{i:02d} unique with enough chars for the gate "
            "to make sure we exceed the length threshold."
            for i in range(10)
        ]
        result = _select_key_insights(findings)
        assert len(result) == 10

    def test_preserves_order(self) -> None:
        """Earlier findings should be preferred (order preservation)."""
        findings = [
            "First finding with enough characters for the gate",
            "Second finding with enough characters for the gate",
            "Third finding with enough characters for the gate",
        ]
        result = _select_key_insights(findings)
        assert "First" in result[0]
        assert "Second" in result[1]
        assert "Third" in result[2]

    def test_handles_whitespace(self) -> None:
        """Should strip whitespace before length check."""
        findings = [
            "   Short   ",
            "   This finding has enough characters after trimming   ",
        ]
        result = _select_key_insights(findings)
        assert len(result) == 1
        assert result[0].strip() == result[0]

    def test_case_insensitive_dedup(self) -> None:
        """Deduplication should be case-insensitive."""
        # Must be >50 chars and share same first 50 chars (case-insensitive)
        findings = [
            "THIS IS A LONG FINDING WITH UPPERCASE START AND MORE TEXT THAT CONTINUES HERE",
            "this is a long finding with uppercase start and more text but lowercase version",
        ]
        result = _select_key_insights(findings)
        assert len(result) == 1


class TestDeriveRecommendations:
    """Tests for _derive_recommendations function."""

    def test_from_reasoning(self) -> None:
        """Should generate recommendation from reflection reasoning."""
        reflection_result = {"reasoning": "The research shows X because Y and Z"}
        reflection_scores = {"overall_coverage_score": 0.9}
        result = _derive_recommendations(reflection_result, reflection_scores)
        assert len(result) >= 1
        assert "Based on analysis" in result[0]

    def test_low_coverage_no_longer_emits_dead_reflection_claim(self) -> None:
        """S84 m04: the reflection-coverage branch was DELETED — its source
        is dead on the production path, so 'Consider additional research to
        improve coverage (current: 0%, ...)' shipped as a constant false
        claim on every mission. Coverage-based advice now comes only from
        the LIVE deliverable_coverage signal."""
        reflection_result = {}
        reflection_scores = {"overall_coverage_score": 0.5}
        result = _derive_recommendations(reflection_result, reflection_scores)
        assert not any("additional research" in r.lower() for r in result)
        # Explicit reflection data present and incomplete -> honest legacy default.
        assert any("remaining gaps" in r.lower() for r in result)

        # The LIVE path: unsatisfied deliverables drive the recommendation.
        live = _derive_recommendations(
            {},
            {},
            synthesis_telemetry={
                "deliverable_coverage": {"satisfied": ["A"], "missing": ["B", "C"]}
            },
        )
        assert any("unsatisfied deliverable" in r.lower() for r in live)
        assert any("B" in r for r in live)

    def test_high_coverage_no_extra_recommendation(self) -> None:
        """Should not recommend more research when coverage >= 80%."""
        reflection_result = {}
        reflection_scores = {"overall_coverage_score": 0.85, "is_mission_complete": True}
        result = _derive_recommendations(reflection_result, reflection_scores)
        assert not any("additional research" in r.lower() for r in result)

    def test_complete_mission_default(self) -> None:
        """Completed mission should recommend proceeding."""
        reflection_result = {}
        reflection_scores = {"overall_coverage_score": 0.9, "is_mission_complete": True}
        result = _derive_recommendations(reflection_result, reflection_scores)
        assert any("proceed" in r.lower() for r in result)

    def test_incomplete_mission_default(self) -> None:
        """Incomplete mission should recommend continuing."""
        reflection_result = {}
        reflection_scores = {"overall_coverage_score": 0.9, "is_mission_complete": False}
        result = _derive_recommendations(reflection_result, reflection_scores)
        assert any("continue" in r.lower() for r in result)

    def test_always_returns_at_least_one(self) -> None:
        """Should always return at least one recommendation."""
        result = _derive_recommendations({}, {})
        assert len(result) >= 1

    def test_truncates_long_reasoning(self) -> None:
        """Long reasoning should be truncated."""
        long_reasoning = "A" * 300
        reflection_result = {"reasoning": long_reasoning}
        result = _derive_recommendations(reflection_result, {})
        # Should be truncated with ellipsis
        assert len(result[0]) < 300


class TestDeriveNextSteps:
    """Tests for _derive_next_steps function."""

    def test_from_partial_coverage(self) -> None:
        """Should generate next steps for partial objectives."""
        objectives = ["Document API patterns", "Analyze performance"]
        criteria_coverage = {"OBJ-1": "complete", "OBJ-2": "partial"}
        result = _derive_next_steps(objectives, criteria_coverage)
        assert len(result) == 1
        assert "performance" in result[0].lower()

    def test_from_not_addressed(self) -> None:
        """Should generate next steps for not_addressed objectives."""
        objectives = ["Document X", "Analyze Y"]
        criteria_coverage = {"OBJ-1": "complete", "OBJ-2": "not_addressed"}
        result = _derive_next_steps(objectives, criteria_coverage)
        assert len(result) == 1
        assert "Analyze Y" in result[0]

    def test_all_complete(self) -> None:
        """All complete objectives should return default next step."""
        objectives = ["Document X", "Analyze Y"]
        criteria_coverage = {"OBJ-1": "complete", "OBJ-2": "complete"}
        result = _derive_next_steps(objectives, criteria_coverage)
        assert len(result) == 1
        assert "deliverables" in result[0].lower()

    def test_empty_objectives(self) -> None:
        """Empty objectives should return default next step."""
        result = _derive_next_steps([], {})
        assert len(result) == 1
        assert "deliverables" in result[0].lower()

    def test_limits_to_five(self) -> None:
        """Should return at most 5 next steps."""
        objectives = [f"Objective {i}" for i in range(10)]
        criteria_coverage = {f"OBJ-{i+1}": "partial" for i in range(10)}
        result = _derive_next_steps(objectives, criteria_coverage)
        assert len(result) == 5

    def test_empty_coverage_no_longer_fabricates_incomplete_objectives(self) -> None:
        """S84 m04: an EMPTY criteria_coverage (its only writer is retired)
        no longer defaults every objective to not_addressed — the pre-m04
        behavior emitted 'Complete research: <objective>' for fully
        satisfied objectives on every production mission. No signal ->
        honest default; live deliverable_coverage drives real next steps."""
        objectives = ["Objective 1"]
        result = _derive_next_steps(objectives, {})
        assert result == ["Review findings and prepare final deliverables"]

        # Explicit legacy coverage data is still honored verbatim.
        legacy = _derive_next_steps(objectives, {"OBJ-1": "partial"})
        assert "Objective 1" in legacy[0]

        # Live signal wins: unsatisfied deliverables become next steps.
        live = _derive_next_steps(
            objectives,
            {},
            deliverable_coverage={"satisfied": ["A"], "missing": ["Comparison Matrix"]},
        )
        assert live == ["Address unsatisfied deliverable: Comparison Matrix"]


class TestGenerateSynthesis:
    """Tests for the main generate_synthesis function."""

    def test_normal_synthesis(self) -> None:
        """Normal case with all data should generate valid synthesis."""
        state: AgentState = {
            "mission_id": "TEST-001",
            "findings": [
                "Finding one with enough characters to pass the 40-char gate",
                "Finding two with enough characters to pass the 40-char gate",
            ],
            "reflection_result": {
                "decision": "finalize",
                "reasoning": "All objectives covered",
                "criteria_coverage": {"OBJ-1": "complete"},
            },
            "reflection_scores": {
                "overall_coverage_score": 0.85,
                "is_mission_complete": True,
            },
            "mission_objectives": ["Document the API patterns"],
        }
        synthesis = generate_synthesis(state)
        assert synthesis.passes_quality_gate()
        assert len(synthesis.key_insights) >= 1
        assert len(synthesis.recommendations) >= 1
        assert len(synthesis.next_steps) >= 1

    def test_empty_findings(self) -> None:
        """Empty findings should still produce valid synthesis with defaults."""
        state: AgentState = {
            "mission_id": "EMPTY-001",
            "findings": [],
            "reflection_result": {},
            "reflection_scores": {"overall_coverage_score": 0.5},
            "mission_objectives": [],
        }
        synthesis = generate_synthesis(state)
        assert synthesis.passes_quality_gate()
        assert len(synthesis.key_insights) == 1
        assert "EMPTY-001" in synthesis.key_insights[0]

    def test_empty_state(self) -> None:
        """Completely empty state should produce valid synthesis."""
        state: AgentState = {}
        synthesis = generate_synthesis(state)
        assert synthesis.passes_quality_gate()
        # Should have default insight
        assert len(synthesis.key_insights) >= 1
        # Should have default recommendation
        assert len(synthesis.recommendations) >= 1
        # Should have default next step
        assert len(synthesis.next_steps) >= 1

    def test_partial_objectives(self) -> None:
        """Partial objective coverage should generate corresponding next steps."""
        state: AgentState = {
            "mission_id": "PARTIAL-001",
            "findings": [
                "A finding with enough characters to pass the quality gate check",
            ],
            "reflection_result": {
                "criteria_coverage": {"OBJ-1": "complete", "OBJ-2": "partial"},
            },
            "reflection_scores": {"overall_coverage_score": 0.6},
            "mission_objectives": ["First objective", "Second objective"],
        }
        synthesis = generate_synthesis(state)
        assert synthesis.passes_quality_gate()
        # Should have next step for partial objective
        assert any("Second objective" in step for step in synthesis.next_steps)

    def test_default_insight_reads_live_gate_output_not_dead_reflection(self) -> None:
        """S84 m04: the fallback insight derives from LIVE gate output —
        the pre-m04 boilerplate interpolated dead reflection_scores and
        shipped 'Coverage score: 0%' on every production mission."""
        state: AgentState = {
            "mission_id": "SCORE-001",
            "findings": [],
            "reflection_result": {},
            "reflection_scores": {"overall_coverage_score": 0.75},
            "mission_objectives": ["One", "Two"],
            "synthesis_telemetry": {
                "deliverable_coverage": {"satisfied": ["A", "B"], "missing": ["C"]},
                "citation_telemetry": {"citations_extracted": 12},
                "report_word_count": 3400,
            },
        }
        synthesis = generate_synthesis(state)
        insight = synthesis.key_insights[0]
        assert "Coverage score:" not in insight  # the dead-reflection lie
        assert "2 of 3 deliverables" in insight
        assert "12 citations" in insight
        assert "3400 words" in insight
        assert len(insight) >= 40  # passes_quality_gate floor

        # Without any live telemetry the fallback stays honest: it reports
        # what it knows and says coverage was not evaluated.
        bare = generate_synthesis(
            {
                "mission_id": "SCORE-002",
                "findings": [],
                "reflection_result": {},
                "reflection_scores": {},
                "mission_objectives": ["One"],
            }
        )
        assert "not evaluated" in bare.key_insights[0]
        assert "Coverage score:" not in bare.key_insights[0]

    def test_insights_deduplicated(self) -> None:
        """Duplicate findings should be deduplicated in key_insights."""
        state: AgentState = {
            "findings": [
                # These share same first 50 chars so second should be deduped
                "Same prefix finding that is long enough to share fifty characters AAAA",
                "Same prefix finding that is long enough to share fifty characters BBBB",
                "Different finding with unique start and more than enough length here",
            ],
            "reflection_result": {},
            "reflection_scores": {},
            "mission_objectives": [],
        }
        synthesis = generate_synthesis(state)
        # Should have only 2 unique insights (one duplicate removed)
        assert len(synthesis.key_insights) == 2

    def test_short_findings_filtered(self) -> None:
        """Short findings should not appear in key_insights."""
        state: AgentState = {
            "findings": [
                "Short",
                "Also short",
                "This finding has enough characters to pass the gate requirement",
            ],
            "reflection_result": {},
            "reflection_scores": {},
            "mission_objectives": [],
        }
        synthesis = generate_synthesis(state)
        # Only the long finding should be selected
        assert len(synthesis.key_insights) == 1
        assert "enough characters" in synthesis.key_insights[0]


class TestIntegration:
    """Integration tests verifying full workflow."""

    def test_synthesis_from_complete_research(self) -> None:
        """Complete research state should produce comprehensive synthesis."""
        state: AgentState = {
            "mission_id": "INT-001",
            "project_id": "proj-123",
            "mission_objectives": [
                "Document the authentication flow",
                "Analyze security patterns",
                "Identify optimization opportunities",
            ],
            "findings": [
                "The authentication uses OAuth 2.0 with PKCE flow for security",
                "JWT tokens are validated on each request with RSA signatures",
                "Rate limiting is implemented at the API gateway level",
                "Caching could be improved with Redis for session storage",
                "Security headers follow OWASP recommendations throughout",
            ],
            "reflection_result": {
                "decision": "finalize",
                "reasoning": "All objectives covered with sufficient depth",
                "criteria_coverage": {
                    "OBJ-1": "complete",
                    "OBJ-2": "complete",
                    "OBJ-3": "partial",
                },
            },
            "reflection_scores": {
                "overall_coverage_score": 0.85,
                "confidence_score": 0.9,
                "is_mission_complete": True,
            },
        }
        synthesis = generate_synthesis(state)

        # Quality gate should pass
        assert synthesis.passes_quality_gate()

        # Key insights should include top findings
        assert len(synthesis.key_insights) >= 1
        assert len(synthesis.key_insights) <= 5

        # Recommendations should be present
        assert len(synthesis.recommendations) >= 1

        # Next steps should include the partial objective
        assert any("optimization" in step.lower() for step in synthesis.next_steps)

    def test_synthesis_export_compatible(self) -> None:
        """Synthesis should export to dict compatible with JSON serialization."""
        state: AgentState = {
            "mission_id": "EXPORT-001",
            "findings": ["A finding with sufficient length for the quality gate"],
            "reflection_result": {"reasoning": "Test"},
            "reflection_scores": {"overall_coverage_score": 0.8},
            "mission_objectives": [],
        }
        synthesis = generate_synthesis(state)
        exported = synthesis.to_dict()

        # All required fields should be present
        assert "key_insights" in exported
        assert "surprising_findings" in exported
        assert "contradictory_information" in exported
        assert "contradiction_resolutions" in exported
        assert "recommendations" in exported
        assert "next_steps" in exported

        # All values should be serializable
        import json
        json_str = json.dumps(exported)
        assert len(json_str) > 0


# =============================================================================
# ResearchStatement Tests
# =============================================================================


class TestResearchStatementModel:
    """Tests for the ResearchStatement Pydantic model."""

    def test_research_statement_required_fields(self) -> None:
        """ResearchStatement requires topic, objective, scope."""
        rs = ResearchStatement(
            topic="Test Topic",
            objective="Test Objective",
            scope="Test Scope",
        )
        assert rs.topic == "Test Topic"
        assert rs.objective == "Test Objective"
        assert rs.scope == "Test Scope"

    def test_research_statement_optional_fields(self) -> None:
        """ResearchStatement optional fields default correctly."""
        rs = ResearchStatement(
            topic="Test Topic",
            objective="Test Objective",
            scope="Test Scope",
        )
        assert rs.audience is None
        assert rs.methodology is None
        assert rs.success_metrics == []
        assert rs.risks == []

    def test_research_statement_all_fields(self) -> None:
        """ResearchStatement should accept all fields."""
        rs = ResearchStatement(
            topic="Test Topic",
            objective="Test Objective",
            scope="Test Scope",
            audience="Developers",
            methodology="Web research",
            success_metrics=["Metric 1", "Metric 2"],
            risks=["Risk 1"],
        )
        assert rs.audience == "Developers"
        assert rs.methodology == "Web research"
        assert len(rs.success_metrics) == 2
        assert len(rs.risks) == 1

    def test_research_statement_to_dict(self) -> None:
        """ResearchStatement to_dict should return all fields."""
        rs = ResearchStatement(
            topic="Test Topic",
            objective="Test Objective",
            scope="Test Scope",
            success_metrics=["Metric"],
        )
        result = rs.to_dict()
        assert "topic" in result
        assert "objective" in result
        assert "scope" in result
        assert "audience" in result
        assert "methodology" in result
        assert "success_metrics" in result
        assert "risks" in result


class TestResearchStatementQualityGate:
    """Tests for ResearchStatement quality gate validation."""

    def test_passes_quality_gate_valid(self) -> None:
        """ResearchStatement with non-empty required fields should pass gate."""
        rs = ResearchStatement(
            topic="Valid Topic",
            objective="Valid Objective",
            scope="Valid Scope",
        )
        assert rs.passes_quality_gate() is True

    def test_fails_quality_gate_whitespace_only_topic(self) -> None:
        """ResearchStatement with whitespace-only topic should fail gate."""
        rs = ResearchStatement(
            topic="   ",
            objective="Valid Objective",
            scope="Valid Scope",
        )
        assert rs.passes_quality_gate() is False

    def test_fails_quality_gate_whitespace_only_objective(self) -> None:
        """ResearchStatement with whitespace-only objective should fail gate."""
        rs = ResearchStatement(
            topic="Valid Topic",
            objective="   ",
            scope="Valid Scope",
        )
        assert rs.passes_quality_gate() is False

    def test_fails_quality_gate_whitespace_only_scope(self) -> None:
        """ResearchStatement with whitespace-only scope should fail gate."""
        rs = ResearchStatement(
            topic="Valid Topic",
            objective="Valid Objective",
            scope="   ",
        )
        assert rs.passes_quality_gate() is False


class TestDeriveResearchTopic:
    """Tests for _derive_research_topic function."""

    def test_from_title(self) -> None:
        """Should derive topic from context.title."""
        context = {"title": "Gemini Deep Search Research"}
        result = _derive_research_topic(context, [], "DRM.0.1")
        assert result == "Gemini Deep Search Research"

    def test_from_objective_when_no_title(self) -> None:
        """Should derive topic from first objective when no title."""
        context = {}
        objectives = ["Catalog CLI commands and parameters"]
        result = _derive_research_topic(context, objectives, "DRM.0.1")
        assert result == "Catalog CLI commands and parameters"

    def test_truncates_long_objective(self) -> None:
        """Should truncate long objectives to 80 chars at word boundary."""
        context = {}
        objectives = ["A" * 50 + " " + "B" * 40]  # 91 chars total
        result = _derive_research_topic(context, objectives, "DRM.0.1")
        assert len(result) <= 80

    def test_fallback_to_mission_id(self) -> None:
        """Should fallback to mission_id when no title or objectives."""
        context = {}
        result = _derive_research_topic(context, [], "DRM.0.1")
        assert result == "Research mission DRM.0.1"

    def test_strips_whitespace(self) -> None:
        """Should strip whitespace from title."""
        context = {"title": "  Spaced Title  "}
        result = _derive_research_topic(context, [], "DRM.0.1")
        assert result == "Spaced Title"

    def test_short_title_falls_through(self) -> None:
        """Short title (<3 chars) should fall through to next fallback."""
        context = {"title": "AB"}
        objectives = ["Valid objective here"]
        result = _derive_research_topic(context, objectives, "DRM.0.1")
        assert result == "Valid objective here"


class TestDeriveResearchObjective:
    """Tests for _derive_research_objective function."""

    def test_from_explicit_objective(self) -> None:
        """Should derive from context.objective."""
        context = {"objective": "Investigate CLI capabilities and integration options"}
        result = _derive_research_objective(context, [])
        assert result == "Investigate CLI capabilities and integration options"

    def test_from_summary_when_no_objective(self) -> None:
        """Should derive from context.summary when no objective."""
        context = {"summary": "Research the Gemini API deep search functionality"}
        result = _derive_research_objective(context, [])
        assert result == "Research the Gemini API deep search functionality"

    def test_truncates_long_summary(self) -> None:
        """Should truncate long summary to ~200 chars."""
        context = {"summary": "A" * 100 + " " + "B" * 150}
        result = _derive_research_objective(context, [])
        assert len(result) <= 210  # ~200 + "..."

    def test_from_objectives_list(self) -> None:
        """Should synthesize from objectives list when no objective/summary."""
        context = {}
        objectives = ["Catalog CLI commands", "Document parameters", "Test integration"]
        result = _derive_research_objective(context, objectives)
        assert result.startswith("Investigate:")
        assert "Catalog CLI commands" in result
        assert "Document parameters" in result
        assert "Test integration" in result

    def test_limits_objectives_to_three(self) -> None:
        """Should only include first 3 objectives in synthesis."""
        context = {}
        objectives = ["Obj1", "Obj2", "Obj3", "Obj4", "Obj5"]
        result = _derive_research_objective(context, objectives)
        assert "Obj4" not in result
        assert "Obj5" not in result

    def test_default_fallback(self) -> None:
        """Should return default when no data available."""
        result = _derive_research_objective({}, [])
        assert result == "Conduct autonomous research and synthesize findings"

    def test_short_objective_falls_through(self) -> None:
        """Short objective (<10 chars) should fall through."""
        context = {"objective": "Short", "summary": "A longer summary that meets minimum"}
        result = _derive_research_objective(context, [])
        assert "longer summary" in result


class TestDeriveResearchScope:
    """Tests for _derive_research_scope function."""

    def test_from_focus(self) -> None:
        """Should derive scope from context.focus."""
        context = {"focus": "macOS environment with Python 3.11"}
        result = _derive_research_scope(context)
        assert result == "macOS environment with Python 3.11"

    def test_from_background(self) -> None:
        """Should derive scope from context.background first sentence."""
        context = {"background": "DeepSearch needs CLI support. More details follow."}
        result = _derive_research_scope(context)
        assert "DeepSearch needs CLI support" in result

    def test_default_fallback(self) -> None:
        """Should return default when no data available."""
        result = _derive_research_scope({})
        assert result == "General research scope - web sources and documentation"

    def test_strips_whitespace(self) -> None:
        """Should strip whitespace from focus."""
        context = {"focus": "  Trimmed Focus  "}
        result = _derive_research_scope(context)
        assert result == "Trimmed Focus"

    def test_short_focus_falls_through(self) -> None:
        """Short focus (<5 chars) should fall through."""
        context = {"focus": "Mac", "background": "A longer background description here."}
        result = _derive_research_scope(context)
        assert "longer background" in result


class TestGenerateResearchStatement:
    """Tests for the main generate_research_statement function."""

    def test_full_context_mapping(self) -> None:
        """Full context should map to all fields correctly."""
        state: AgentState = {
            "mission_id": "DRM.0.1",
            "mission_context": {
                "title": "Gemini Deep Search Research",
                "objective": "Investigate CLI capabilities and integration",
                "focus": "macOS environment",
                "audience": "Developers",
            },
            "mission_objectives": ["Catalog CLI commands", "Document parameters"],
        }
        rs = generate_research_statement(state)
        assert rs.passes_quality_gate()
        assert rs.topic == "Gemini Deep Search Research"
        assert rs.objective == "Investigate CLI capabilities and integration"
        assert rs.scope == "macOS environment"
        assert rs.audience == "Developers"
        assert rs.success_metrics == ["Catalog CLI commands", "Document parameters"]

    def test_minimal_context(self) -> None:
        """Minimal context should still produce valid ResearchStatement."""
        state: AgentState = {
            "mission_id": "TEST-001",
            "mission_context": {"title": "Test Research"},
            "mission_objectives": [],
        }
        rs = generate_research_statement(state)
        assert rs.passes_quality_gate()
        assert rs.topic == "Test Research"

    def test_empty_context(self) -> None:
        """Empty context should use all fallbacks."""
        state: AgentState = {
            "mission_id": "EMPTY-001",
            "mission_context": {},
            "mission_objectives": [],
        }
        rs = generate_research_statement(state)
        assert rs.passes_quality_gate()
        assert rs.topic == "Research mission EMPTY-001"
        assert "autonomous research" in rs.objective
        assert "General research scope" in rs.scope

    def test_completely_empty_state(self) -> None:
        """Completely empty state should still work."""
        state: AgentState = {}
        rs = generate_research_statement(state)
        assert rs.passes_quality_gate()

    def test_objectives_as_success_metrics(self) -> None:
        """Mission objectives should be used as success_metrics."""
        state: AgentState = {
            "mission_id": "METRIC-001",
            "mission_context": {"title": "Test"},
            "mission_objectives": ["Obj1", "Obj2", "Obj3"],
        }
        rs = generate_research_statement(state)
        assert rs.success_metrics == ["Obj1", "Obj2", "Obj3"]

    def test_limits_success_metrics_to_five(self) -> None:
        """Should limit success_metrics to first 5 objectives."""
        state: AgentState = {
            "mission_id": "LIMIT-001",
            "mission_context": {"title": "Test"},
            "mission_objectives": [f"Obj{i}" for i in range(10)],
        }
        rs = generate_research_statement(state)
        assert len(rs.success_metrics) == 5

    def test_default_methodology(self) -> None:
        """Should set default methodology."""
        state: AgentState = {
            "mission_id": "METHOD-001",
            "mission_context": {"title": "Test"},
            "mission_objectives": [],
        }
        rs = generate_research_statement(state)
        assert rs.methodology == "Autonomous web research with LLM synthesis"

    def test_export_compatible(self) -> None:
        """ResearchStatement should export to dict compatible with JSON."""
        state: AgentState = {
            "mission_id": "EXPORT-001",
            "mission_context": {
                "title": "Export Test",
                "objective": "Test JSON export",
                "focus": "Serialization",
            },
            "mission_objectives": ["Test export"],
        }
        rs = generate_research_statement(state)
        exported = rs.to_dict()

        import json
        json_str = json.dumps(exported)
        assert len(json_str) > 0
        # Verify roundtrip
        parsed = json.loads(json_str)
        assert parsed["topic"] == "Export Test"
