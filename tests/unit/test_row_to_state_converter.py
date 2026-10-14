"""Unit tests for row-to-state converter module.

Tests the conversion of TraceLab MissionRow objects to DeepSearch AgentState.
Updated to match TraceLab schema (Sprint 09):
- objective (TEXT) instead of objectives (JSONB array)
- success_criteria (JSONB) for additional objectives
- research_phases (JSONB) for config like max_loops
- No config, callback_url, or project_id fields
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from deepsearch.worker.converter import (
    DEFAULT_DELIVERABLE_FORMAT,
    DEFAULT_MAX_LOOPS,
    DEFAULT_MIN_LOOPS,
    ConversionError,
    _build_mission_context,
    _extract_deliverable_format,
    _extract_depth,
    _extract_max_loops,
    _extract_min_loops,
    _extract_objectives,
    convert_row_to_state,
)
from deepsearch.worker.lease import LeaseAttempt
from deepsearch.worker.poller import MissionRow


def _lease() -> LeaseAttempt:
    timestamp = datetime(2026, 7, 18, 4, 0, 0)
    return LeaseAttempt(
        owner="worker-test",
        token="opaque-test-token",
        leased_at=timestamp,
        heartbeat_at=timestamp,
        expires_at=timestamp,
        attempt_count=1,
    )


def create_mission_row(
    id: str = "uuid-123-456",
    mission_id: str = "B17.1",
    title: str = "Test Mission",
    objective: str = "Research the topic thoroughly",
    context: Optional[Dict[str, Any]] = None,
    status: str = "queued",
    queued_at: Optional[datetime] = None,
    success_criteria: Optional[List[str]] = None,
    deliverables: Optional[List[str]] = None,
    research_phases: Optional[Dict[str, Any]] = None,
    research_depth: Optional[str] = None,
    background: Optional[str] = None,
    focus: Optional[str] = None,
    references: Optional[List[Dict[str, Any]]] = None,
    required_entities: Optional[List[str]] = None,
    excluded_entities: Optional[List[str]] = None,
    expected_output_schema: Optional[Dict[str, Any]] = None,
    coverage_thresholds: Optional[Dict[str, Any]] = None,
    validation_thresholds: Optional[Dict[str, Any]] = None,
    deliverable_format: Optional[str] = None,
    max_loops: Optional[int] = None,
    min_loops: Optional[int] = None,
    constraints: Optional[List[str]] = None,
) -> MissionRow:
    """Factory function for creating test MissionRow instances."""
    return MissionRow(
        id=id,
        mission_id=mission_id,
        title=title,
        objective=objective,
        context=context or {},
        status=status,
        queued_at=queued_at or datetime(2025, 1, 15, 10, 30, 0),
        lease=_lease(),
        success_criteria=success_criteria,
        deliverables=deliverables,
        research_phases=research_phases,
        research_depth=research_depth,
        background=background,
        focus=focus,
        references=references,
        required_entities=required_entities,
        excluded_entities=excluded_entities,
        expected_output_schema=expected_output_schema,
        coverage_thresholds=coverage_thresholds,
        validation_thresholds=validation_thresholds,
        deliverable_format=deliverable_format,
        max_loops=max_loops,
        min_loops=min_loops,
        constraints=constraints,
    )


# =============================================================================
# convert_row_to_state tests
# =============================================================================

class TestConvertRowToState:
    """Tests for the main convert_row_to_state function."""

    def test_basic_conversion(self):
        """Test basic MissionRow to AgentState conversion."""
        row = create_mission_row()
        state = convert_row_to_state(row)

        assert state["mission_id"] == "B17.1"
        assert state["mission_objectives"] == ["Research the topic thoroughly"]
        assert state["max_loops"] == DEFAULT_MAX_LOOPS
        assert state["min_loops"] == DEFAULT_MIN_LOOPS
        assert state["deliverable_format"] == DEFAULT_DELIVERABLE_FORMAT

    def test_mission_context_includes_title(self):
        """Test that mission context includes title."""
        row = create_mission_row(title="Research Report")
        state = convert_row_to_state(row)

        assert state["mission_context"]["title"] == "Research Report"

    def test_mission_context_strips_prefixed_mission_id_from_title(self):
        """TraceLab titles should not duplicate the mission ID in runtime context."""
        row = create_mission_row(
            mission_id="FORCE-SMOKE-08",
            title="FORCE-SMOKE-08 - LangGraph Checkpointing Patterns",
        )
        state = convert_row_to_state(row)

        assert state["mission_context"]["title"] == "LangGraph Checkpointing Patterns"

    def test_objective_in_context(self):
        """Test objective is correctly mapped to context."""
        row = create_mission_row(objective="Main research goal")
        state = convert_row_to_state(row)

        assert state["mission_context"]["objective"] == "Main research goal"

    def test_tracelab_uuid_preserved(self):
        """Test TraceLab UUID is preserved in context."""
        row = create_mission_row(id="abc-123-uuid")
        state = convert_row_to_state(row)

        assert state["mission_context"]["tracelab_uuid"] == "abc-123-uuid"

    def test_missing_id_raises_error(self):
        """Test that missing mission ID raises ConversionError."""
        row = create_mission_row(id="")
        with pytest.raises(ConversionError, match="missing required 'id'"):
            convert_row_to_state(row)

    def test_missing_title_raises_error(self):
        """Test that missing title raises ConversionError."""
        row = create_mission_row(title="")
        with pytest.raises(ConversionError, match="missing required 'title'"):
            convert_row_to_state(row)

    def test_empty_objective_raises_error(self):
        """Test that empty objective raises ConversionError."""
        row = create_mission_row(objective="")
        with pytest.raises(ConversionError, match="has no objectives"):
            convert_row_to_state(row)

    def test_whitespace_only_objective_raises_error(self):
        """Test that whitespace-only objective raises ConversionError."""
        row = create_mission_row(objective="   \t\n  ")
        with pytest.raises(ConversionError, match="has no objectives"):
            convert_row_to_state(row)

    def test_research_phases_overrides_defaults(self):
        """Test that research_phases values override defaults."""
        row = create_mission_row(
            research_phases={"max_loops": 5, "min_loops": 3, "deliverable_format": "json"}
        )
        state = convert_row_to_state(row)

        assert state["max_loops"] == 5
        assert state["min_loops"] == 3
        assert state["deliverable_format"] == "json"

    def test_research_depth_extracted_to_state(self):
        """Research depth label flows from research_phases to state."""
        row = create_mission_row(
            research_phases={"depth": "deep"}
        )
        state = convert_row_to_state(row)

        assert state["research_depth"] == "deep"
        # Post-S76: depth_config is an empty dict (no in-process tier dispatch).
        assert state["depth_config"] == {}
        # max/min loops fall back to converter defaults when not on the row.
        assert state["max_loops"] == DEFAULT_MAX_LOOPS
        assert state["min_loops"] == DEFAULT_MIN_LOOPS

    def test_research_depth_alpha_label_passthrough(self):
        """Alpha label is recorded as a plain string post-S76."""
        row = create_mission_row(
            research_phases={"depth": "alpha"}
        )
        state = convert_row_to_state(row)

        assert state["research_depth"] == "alpha"
        assert state["depth_config"] == {}
        assert state["max_loops"] == DEFAULT_MAX_LOOPS
        assert state["min_loops"] == DEFAULT_MIN_LOOPS

    def test_research_depth_defaults_to_baseline(self):
        """Test depth defaults to baseline when not specified."""
        row = create_mission_row()
        state = convert_row_to_state(row)

        assert state["research_depth"] == "baseline"

    def test_success_criteria_added_to_objectives(self):
        """Success criteria should replace the umbrella objective at runtime."""
        row = create_mission_row(
            objective="Main goal",
            success_criteria=["Criterion 1", "Criterion 2"]
        )
        state = convert_row_to_state(row)

        assert len(state["mission_objectives"]) == 2
        assert "Criterion 1" in state["mission_objectives"]
        assert "Criterion 2" in state["mission_objectives"]

    def test_queued_at_in_context(self):
        """Test queued_at timestamp is preserved in context."""
        queued = datetime(2025, 6, 15, 12, 0, 0)
        row = create_mission_row(queued_at=queued)
        state = convert_row_to_state(row)

        assert state["mission_context"]["queued_at"] == "2025-06-15T12:00:00"

    def test_mission_id_in_context(self):
        """Test mission_id is preserved in context."""
        row = create_mission_row(mission_id="B17.3")
        state = convert_row_to_state(row)

        assert state["mission_context"]["mission_id"] == "B17.3"


# =============================================================================
# _extract_objectives tests
# =============================================================================

class TestExtractObjectives:
    """Tests for objective extraction from MissionRow."""

    def test_objective_extracted(self):
        """Test objective extracted from row.objective."""
        row = create_mission_row(objective="Main research objective")
        objectives = _extract_objectives(row)

        assert objectives == ["Main research objective"]

    def test_objective_with_whitespace(self):
        """Test whitespace is trimmed from objective."""
        row = create_mission_row(objective="  Trimmed objective  ")
        objectives = _extract_objectives(row)

        assert objectives == ["Trimmed objective"]

    def test_success_criteria_appended(self):
        """Explicit success criteria should become the runtime objectives."""
        row = create_mission_row(
            objective="Primary goal",
            success_criteria=["SC 1", "SC 2"]
        )
        objectives = _extract_objectives(row)

        assert objectives == ["SC 1", "SC 2"]

    def test_empty_success_criteria_filtered(self):
        """Test empty string criteria are filtered."""
        row = create_mission_row(
            objective="Goal",
            success_criteria=["Valid", "", "  ", "Also Valid"]
        )
        objectives = _extract_objectives(row)

        assert objectives == ["Valid", "Also Valid"]

    def test_fallback_to_context_success_criteria(self):
        """Test fallback to success_criteria in context."""
        row = create_mission_row(
            objective="",
            context={"success_criteria": ["Criterion 1", "Criterion 2"]}
        )
        objectives = _extract_objectives(row)

        assert objectives == ["Criterion 1", "Criterion 2"]

    def test_fallback_to_camelcase_success_criteria(self):
        """Test fallback to successCriteria (camelCase) in context."""
        row = create_mission_row(
            objective="",
            context={"successCriteria": ["Criteria A"]}
        )
        objectives = _extract_objectives(row)

        assert objectives == ["Criteria A"]

    def test_mixed_types_in_success_criteria(self):
        """Test non-string items in context success_criteria are converted."""
        row = create_mission_row(
            objective="",
            context={"success_criteria": ["String", 123, True]}
        )
        objectives = _extract_objectives(row)

        assert objectives == ["String", "123", "True"]


# =============================================================================
# _build_mission_context tests
# =============================================================================

class TestBuildMissionContext:
    """Tests for mission context building."""

    def test_merges_existing_context(self):
        """Test existing context values are preserved."""
        row = create_mission_row(
            context={"background": "Research background", "constraints": ["A", "B"]}
        )
        context = _build_mission_context(row)

        assert context["background"] == "Research background"
        assert context["constraints"] == ["A", "B"]

    def test_dedicated_authoring_columns_override_legacy_even_when_empty(self):
        """Explicit empty authoring values clear pre-column context values."""
        row = create_mission_row(
            context={
                "background": "legacy",
                "references": [{"title": "legacy"}],
                "expected_output_schema": {"kind": "legacy"},
                "coverage_thresholds": {"entity": 0.5},
                "constraints": ["legacy"],
            },
            background="",
            references=[],
            expected_output_schema={},
            coverage_thresholds={},
            constraints=[],
        )

        context = _build_mission_context(row)

        assert context["background"] == ""
        assert context["references"] == []
        assert context["expected_output_schema"] == {}
        assert context["coverage_thresholds"] == {}
        assert context["constraints"] == []

    def test_title_always_set(self):
        """Test title is always set in context."""
        row = create_mission_row(title="Mission Title")
        context = _build_mission_context(row)

        assert context["title"] == "Mission Title"

    def test_title_normalizes_prefixed_mission_id(self):
        """Mission-id prefixes should be stripped from TraceLab display titles."""
        row = create_mission_row(
            mission_id="FORCE-SMOKE-08",
            title="FORCE-SMOKE-08 - LangGraph Checkpointing Patterns",
            objective="",
        )
        context = _build_mission_context(row)

        assert context["title"] == "LangGraph Checkpointing Patterns"
        assert context["objective"] == "LangGraph Checkpointing Patterns"

    def test_objective_set_from_row(self):
        """Test objective from row is used."""
        row = create_mission_row(objective="Main research goal")
        context = _build_mission_context(row)

        assert context["objective"] == "Main research goal"

    def test_objective_fallback_to_title(self):
        """Test objective falls back to title when empty."""
        row = create_mission_row(
            title="Mission Title",
            objective=""
        )
        context = _build_mission_context(row)

        assert context["objective"] == "Mission Title"

    def test_research_phases_included_in_context(self):
        """Test research_phases is included in context."""
        row = create_mission_row(
            research_phases={"max_loops": 5, "depth": "deep"}
        )
        context = _build_mission_context(row)

        assert context["research_phases"]["max_loops"] == 5
        assert context["research_phases"]["depth"] == "deep"

    def test_success_criteria_included_in_context(self):
        """Test success_criteria is included in context."""
        row = create_mission_row(
            success_criteria=["SC 1", "SC 2"]
        )
        context = _build_mission_context(row)

        assert context["success_criteria"] == ["SC 1", "SC 2"]

    def test_deliverables_included_in_context(self):
        """Test deliverables are included in context."""
        row = create_mission_row(
            deliverables=["Annotated bibliography of 15 papers"]
        )
        context = _build_mission_context(row)

        assert context["deliverables"] == ["Annotated bibliography of 15 papers"]

    def test_empty_context_handled(self):
        """Test empty/None context is handled."""
        row = MissionRow(
            id="test-uuid",
            mission_id="B1.1",
            title="Test",
            objective="Obj",
            context={},
            status="queued",
            queued_at=datetime.now(),
            lease=_lease(),
        )
        context = _build_mission_context(row)

        assert "title" in context
        assert "objective" in context


# =============================================================================
# _extract_max_loops tests
# =============================================================================

class TestExtractMaxLoops:
    """Tests for max_loops extraction."""

    def test_from_research_phases_snake_case(self):
        """Test max_loops from research_phases (snake_case)."""
        row = create_mission_row(research_phases={"max_loops": 7})
        max_loops = _extract_max_loops(row)

        assert max_loops == 7

    def test_from_research_phases_camel_case(self):
        """Test maxLoops from research_phases (camelCase)."""
        row = create_mission_row(research_phases={"maxLoops": 8})
        max_loops = _extract_max_loops(row)

        assert max_loops == 8

    def test_from_context_fallback(self):
        """Test fallback to context when not in research_phases."""
        row = create_mission_row(
            research_phases={},
            context={"max_loops": 6}
        )
        max_loops = _extract_max_loops(row)

        assert max_loops == 6

    def test_research_phases_takes_precedence(self):
        """Test research_phases value takes precedence over context."""
        row = create_mission_row(
            research_phases={"max_loops": 10},
            context={"max_loops": 5}
        )
        max_loops = _extract_max_loops(row)

        assert max_loops == 10

    def test_dedicated_column_takes_precedence_over_legacy_budgets(self):
        """The first-class TraceLab budget is the execution authority."""
        row = create_mission_row(
            max_loops=12,
            research_phases={"max_loops": 10},
            context={"max_loops": 5},
        )

        assert _extract_max_loops(row) == 12

    def test_default_when_not_found(self):
        """Test default value when not found anywhere."""
        row = create_mission_row(research_phases={}, context={})
        max_loops = _extract_max_loops(row)

        assert max_loops == DEFAULT_MAX_LOOPS

    def test_string_value_parsed(self):
        """Test string value is parsed to int."""
        row = create_mission_row(research_phases={"max_loops": "4"})
        max_loops = _extract_max_loops(row)

        assert max_loops == 4

    def test_zero_uses_default(self):
        """Test zero value falls through to default."""
        row = create_mission_row(research_phases={"max_loops": 0})
        max_loops = _extract_max_loops(row)

        assert max_loops == DEFAULT_MAX_LOOPS

    def test_negative_uses_default(self):
        """Test negative value falls through to default."""
        row = create_mission_row(research_phases={"max_loops": -1})
        max_loops = _extract_max_loops(row)

        assert max_loops == DEFAULT_MAX_LOOPS


# =============================================================================
# _extract_min_loops tests
# =============================================================================

class TestExtractMinLoops:
    """Tests for min_loops extraction."""

    def test_from_research_phases(self):
        """Test min_loops from research_phases."""
        row = create_mission_row(research_phases={"min_loops": 3})
        min_loops = _extract_min_loops(row)

        assert min_loops == 3

    def test_from_research_phases_camel_case(self):
        """Test minLoops from research_phases (camelCase)."""
        row = create_mission_row(research_phases={"minLoops": 4})
        min_loops = _extract_min_loops(row)

        assert min_loops == 4

    def test_default_when_not_found(self):
        """Test default value when not found."""
        row = create_mission_row(research_phases={}, context={})
        min_loops = _extract_min_loops(row)

        assert min_loops == DEFAULT_MIN_LOOPS

    def test_context_fallback(self):
        """Test fallback to context."""
        row = create_mission_row(context={"min_loops": 1})
        min_loops = _extract_min_loops(row)

        assert min_loops == 1

    def test_dedicated_column_takes_precedence_over_legacy_budgets(self):
        """The first-class TraceLab minimum wins over nested configuration."""
        row = create_mission_row(
            min_loops=5,
            research_phases={"min_loops": 3},
            context={"min_loops": 1},
        )

        assert _extract_min_loops(row) == 5


# =============================================================================
# _extract_deliverable_format tests
# =============================================================================

class TestExtractDeliverableFormat:
    """Tests for deliverable_format extraction."""

    def test_from_research_phases(self):
        """Test deliverable_format from research_phases."""
        row = create_mission_row(research_phases={"deliverable_format": "json"})
        format = _extract_deliverable_format(row)

        assert format == "json"

    def test_from_research_phases_camel_case(self):
        """Test deliverableFormat from research_phases (camelCase)."""
        row = create_mission_row(research_phases={"deliverableFormat": "XML"})
        format = _extract_deliverable_format(row)

        assert format == "xml"  # Lowercased

    def test_format_key(self):
        """Test 'format' key in research_phases."""
        row = create_mission_row(research_phases={"format": "Html"})
        format = _extract_deliverable_format(row)

        assert format == "html"

    def test_from_context_fallback(self):
        """Test fallback to context."""
        row = create_mission_row(
            research_phases={},
            context={"deliverable_format": "pdf"}
        )
        format = _extract_deliverable_format(row)

        assert format == "pdf"

    def test_dedicated_column_takes_precedence_over_legacy_format(self):
        """The first-class format wins over research_phases and context."""
        row = create_mission_row(
            deliverable_format=" Evidence Matrix ",
            research_phases={"deliverable_format": "json"},
            context={"deliverable_format": "pdf"},
        )

        assert _extract_deliverable_format(row) == "evidence matrix"

    def test_explicit_empty_column_clears_legacy_format(self):
        """Empty is explicit, so the safe default replaces—not legacy—format."""
        row = create_mission_row(
            deliverable_format="",
            research_phases={"deliverable_format": "json"},
            context={"deliverable_format": "pdf"},
        )

        assert _extract_deliverable_format(row) == DEFAULT_DELIVERABLE_FORMAT

    def test_default_when_not_found(self):
        """Test default value when not found."""
        row = create_mission_row(research_phases={}, context={})
        format = _extract_deliverable_format(row)

        assert format == DEFAULT_DELIVERABLE_FORMAT

    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed."""
        row = create_mission_row(research_phases={"format": "  markdown  "})
        format = _extract_deliverable_format(row)

        assert format == "markdown"


# =============================================================================
# _extract_depth tests
# =============================================================================

class TestExtractDepth:
    """Tests for research depth extraction from MissionRow."""

    def test_from_direct_research_depth_column(self):
        """Test depth from direct research_depth column (highest priority)."""
        row = create_mission_row(research_depth="alpha")
        depth = _extract_depth(row)

        assert depth == "alpha"

    def test_direct_column_takes_precedence_over_research_phases(self):
        """Test direct column takes precedence over research_phases."""
        row = create_mission_row(
            research_depth="alpha",
            research_phases={"depth": "baseline"}
        )
        depth = _extract_depth(row)

        assert depth == "alpha"

    def test_direct_column_takes_precedence_over_context(self):
        """Test direct column takes precedence over context."""
        row = create_mission_row(
            research_depth="deep",
            context={"depth": "baseline"}
        )
        depth = _extract_depth(row)

        assert depth == "deep"

    def test_direct_column_case_insensitive(self):
        """Test direct column is case-insensitive."""
        row = create_mission_row(research_depth="ALPHA")
        depth = _extract_depth(row)

        assert depth == "alpha"

    def test_direct_column_whitespace_trimmed(self):
        """Test whitespace is trimmed from direct column."""
        row = create_mission_row(research_depth="  deep  ")
        depth = _extract_depth(row)

        assert depth == "deep"

    def test_empty_direct_column_falls_through(self):
        """Test empty direct column falls through to research_phases."""
        row = create_mission_row(
            research_depth="",
            research_phases={"depth": "alpha"}
        )
        depth = _extract_depth(row)

        assert depth == "alpha"

    def test_none_direct_column_falls_through(self):
        """Test None direct column falls through to research_phases."""
        row = create_mission_row(
            research_depth=None,
            research_phases={"depth": "deep"}
        )
        depth = _extract_depth(row)

        assert depth == "deep"

    def test_from_research_phases_depth(self):
        """Test depth from research_phases['depth']."""
        row = create_mission_row(research_phases={"depth": "deep"})
        depth = _extract_depth(row)

        assert depth == "deep"

    def test_from_research_phases_research_depth(self):
        """Test depth from research_phases['research_depth']."""
        row = create_mission_row(research_phases={"research_depth": "alpha"})
        depth = _extract_depth(row)

        assert depth == "alpha"

    def test_from_research_phases_camel_case(self):
        """Test depth from research_phases['researchDepth'] (camelCase)."""
        row = create_mission_row(research_phases={"researchDepth": "deep"})
        depth = _extract_depth(row)

        assert depth == "deep"

    def test_from_context_fallback(self):
        """Test fallback to context when not in research_phases."""
        row = create_mission_row(
            research_phases={},
            context={"depth": "alpha"}
        )
        depth = _extract_depth(row)

        assert depth == "alpha"

    def test_research_phases_takes_precedence(self):
        """Test research_phases value takes precedence over context."""
        row = create_mission_row(
            research_phases={"depth": "alpha"},
            context={"depth": "baseline"}
        )
        depth = _extract_depth(row)

        assert depth == "alpha"

    def test_default_when_not_found(self):
        """Test default to BASELINE when not found anywhere."""
        row = create_mission_row(research_phases={}, context={})
        depth = _extract_depth(row)

        assert depth == "baseline"

    def test_case_insensitive(self):
        """Test depth parsing is case-insensitive."""
        row = create_mission_row(research_phases={"depth": "DEEP"})
        depth = _extract_depth(row)

        assert depth == "deep"

    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed from depth value."""
        row = create_mission_row(research_phases={"depth": "  alpha  "})
        depth = _extract_depth(row)

        assert depth == "alpha"

    def test_invalid_depth_defaults_to_baseline(self):
        """Test invalid depth value falls back to BASELINE."""
        row = create_mission_row(research_phases={"depth": "invalid_tier"})
        depth = _extract_depth(row)

        assert depth == "baseline"

    def test_empty_string_defaults_to_baseline(self):
        """Test empty string depth falls back to BASELINE."""
        row = create_mission_row(research_phases={"depth": ""})
        depth = _extract_depth(row)

        assert depth == "baseline"

    def test_none_research_phases(self):
        """Test None research_phases handled gracefully."""
        row = create_mission_row(research_phases=None, context={"depth": "deep"})
        depth = _extract_depth(row)

        assert depth == "deep"

    def test_none_context(self):
        """Test None context handled gracefully."""
        # MissionRow requires context to be dict, but test the fallback chain
        row = create_mission_row(research_phases={"depth": "alpha"}, context={})
        depth = _extract_depth(row)

        assert depth == "alpha"


# =============================================================================
# Integration tests
# =============================================================================

class TestConverterIntegration:
    """Integration tests for full conversion workflow."""

    def test_full_tracelab_row(self):
        """Test conversion of a full TraceLab-style row."""
        row = MissionRow(
            id="uuid-2025-001",
            mission_id="B17.1",
            title="Competitive Analysis Report",
            objective="Analyze competitor product features and market positioning",
            context={
                "background": "Quarterly competitive analysis for strategic planning",
                "focus": "SaaS competitors in the project management space",
                "constraints": ["Use only public information", "Complete within 24 hours"],
            },
            status="queued",
            queued_at=datetime(2025, 1, 15, 9, 0, 0),
            lease=_lease(),
            success_criteria=[
                "Identify top 5 competitors",
                "Evaluate pricing models"
            ],
            research_phases={
                "max_loops": 5,
                "min_loops": 3,
                "deliverable_format": "markdown",
                "depth": "deep"
            }
        )

        state = convert_row_to_state(row)

        # Verify core fields
        assert state["mission_id"] == "B17.1"
        assert len(state["mission_objectives"]) == 2
        assert state["max_loops"] == 5
        assert state["min_loops"] == 3
        assert state["deliverable_format"] == "markdown"

        # Verify research depth label flows through (post-S76 boundary string).
        assert state["research_depth"] == "deep"
        assert state["depth_config"] == {}

        # Verify context preservation
        context = state["mission_context"]
        assert context["title"] == "Competitive Analysis Report"
        assert context["background"] == "Quarterly competitive analysis for strategic planning"
        assert context["focus"] == "SaaS competitors in the project management space"
        assert context["tracelab_uuid"] == "uuid-2025-001"
        assert context["mission_id"] == "B17.1"

        # Verify state is ready for research
        assert state["research_loop_count"] == 0
        # sources_found stripped at S78 m02 — converter no longer seeds it.
        assert "sources_found" not in state
        assert state["finding_records"] == []
        assert state["concept_clusters"] == []
        assert state["findings"] == []

    def test_minimal_row(self):
        """Test conversion of minimal required fields."""
        row = MissionRow(
            id="min-001",
            mission_id="B1.1",
            title="Minimal Mission",
            objective="Single objective",
            context={},
            status="queued",
            queued_at=datetime.now(),
            lease=_lease(),
        )

        state = convert_row_to_state(row)

        assert state["mission_id"] == "B1.1"
        assert state["mission_objectives"] == ["Single objective"]
        assert state["max_loops"] == DEFAULT_MAX_LOOPS
        assert state["deliverable_format"] == DEFAULT_DELIVERABLE_FORMAT

    def test_state_has_all_required_fields(self):
        """Test that returned state has all required AgentState fields."""
        row = create_mission_row()
        state = convert_row_to_state(row)

        # Required fields that must exist. sources_found stripped at
        # S78 m02 (PD71 conservative) — readers across runner.py / report/
        # use state.get(...) and degrade gracefully on absence.
        required_fields = [
            "mission_id",
            "mission_objectives",
            "mission_context",
            "max_loops",
            "min_loops",
            "research_loop_count",
            "finding_records",
            "concept_clusters",
            "findings",
            "reflection_result",
            "reflection_scores",
            "convergence_history",
            "previous_queries",
            "error_history",
            "research_depth",
            "depth_config",
        ]

        for field in required_fields:
            assert field in state, f"Missing required field: {field}"
