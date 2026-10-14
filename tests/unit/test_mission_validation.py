"""Unit tests for mission validation."""

import pytest
pytest.importorskip("fastapi")

from deepsearch.api.models.requests import ExecuteMissionRequest
from deepsearch.api.validators import (
    ValidationError,
    ValidationResult,
    validate_mission,
    _validate_objectives,
    _validate_success_criteria,
    _validate_title,
    _validate_limits,
)


def make_request(**kwargs) -> ExecuteMissionRequest:
    """Create a valid request with overrides."""
    defaults = {
        "mission_id": "TEST.0.1",
        "title": "Test research mission",
        "objectives": ["Compare framework A vs framework B"],
        "success_criteria": ["Comparison table with 5+ rows"],
        "max_loops": 3,
        "token_limit": 8000,
    }
    defaults.update(kwargs)
    return ExecuteMissionRequest(**defaults)


class TestValidateMission:
    """Tests for validate_mission function."""

    def test_valid_mission_passes(self):
        """A well-formed mission passes validation."""
        request = make_request()
        result = validate_mission(request)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_collects_all_errors(self):
        """Multiple errors are collected, not stopped at first."""
        request = make_request(
            title="Short",  # Too short
            objectives=["", "Compare X"],  # Empty first objective
            success_criteria=[],  # Empty criteria
        )
        result = validate_mission(request)
        assert result.valid is False
        # Should have: title error, objectives[0] error, success_criteria error
        assert len(result.errors) >= 3

    def test_returns_validation_result(self):
        """Returns proper ValidationResult type."""
        request = make_request()
        result = validate_mission(request)
        assert isinstance(result, ValidationResult)
        assert isinstance(result.errors, list)


class TestValidateObjectives:
    """Tests for objective validation."""

    def test_empty_objectives_rejected(self):
        """Empty objectives list is rejected."""
        errors = _validate_objectives([])
        assert len(errors) == 1
        assert errors[0].field == "objectives"
        assert "At least one" in errors[0].message
        assert errors[0].suggestion is not None

    def test_whitespace_only_rejected(self):
        """Whitespace-only objective is rejected."""
        errors = _validate_objectives(["   ", "Valid objective"])
        assert any(e.field == "objectives[0]" for e in errors)
        assert any("empty" in e.message.lower() for e in errors)

    def test_empty_string_rejected(self):
        """Empty string objective is rejected."""
        errors = _validate_objectives([""])
        assert len(errors) == 1
        assert errors[0].field == "objectives[0]"

    def test_non_researchable_this_report(self):
        """'this report' pattern is detected."""
        errors = _validate_objectives(["Summarize this report findings"])
        assert len(errors) == 1
        assert "non-researchable" in errors[0].message.lower()

    def test_non_researchable_this_document(self):
        """'this document' pattern is detected."""
        errors = _validate_objectives(["Analyze this document"])
        assert len(errors) == 1
        assert "non-researchable" in errors[0].message.lower()

    def test_non_researchable_file_url(self):
        """file:// URL is detected."""
        errors = _validate_objectives(["Read file:///path/to/doc"])
        assert len(errors) == 1

    def test_non_researchable_md_extension(self):
        """Markdown file reference is detected."""
        errors = _validate_objectives(["Process README.md"])
        assert len(errors) == 1

    def test_non_researchable_yaml_extension(self):
        """YAML file reference is detected."""
        errors = _validate_objectives(["Parse config.yaml"])
        assert len(errors) == 1

    def test_non_researchable_json_extension(self):
        """JSON file reference is detected."""
        errors = _validate_objectives(["Load data.json"])
        assert len(errors) == 1

    def test_non_researchable_cmos_path(self):
        """cmos/ path is detected."""
        errors = _validate_objectives(["Check cmos/missions file"])
        assert len(errors) == 1

    def test_non_researchable_deepsearch_path(self):
        """deepsearch/ path is detected."""
        errors = _validate_objectives(["Look at deepsearch/agent code"])
        assert len(errors) == 1

    def test_non_researchable_saved_to(self):
        """'saved to' pattern is detected."""
        errors = _validate_objectives(["Generate report saved to file"])
        assert len(errors) == 1

    def test_non_researchable_output_to(self):
        """'output to' pattern is detected."""
        errors = _validate_objectives(["Create data output to disk"])
        assert len(errors) == 1

    def test_duplicate_objectives_detected(self):
        """Duplicate objectives are flagged."""
        errors = _validate_objectives([
            "Compare React vs Vue",
            "Compare React vs Vue",
        ])
        assert len(errors) == 1
        assert "Duplicate" in errors[0].message
        assert errors[0].field == "objectives[1]"

    def test_case_insensitive_duplicates(self):
        """Duplicates are detected case-insensitively."""
        errors = _validate_objectives([
            "Compare REACT vs VUE",
            "compare react vs vue",
        ])
        assert len(errors) == 1
        assert "Duplicate" in errors[0].message

    def test_valid_objectives_pass(self):
        """Valid objectives pass validation."""
        errors = _validate_objectives([
            "Compare GraphQL vs REST for mobile apps",
            "Identify best practices for API design",
        ])
        assert len(errors) == 0


class TestValidateSuccessCriteria:
    """Tests for success criteria validation."""

    def test_empty_criteria_rejected(self):
        """Empty criteria list is rejected."""
        errors = _validate_success_criteria([])
        assert len(errors) == 1
        assert errors[0].field == "success_criteria"
        assert "At least one" in errors[0].message
        assert errors[0].suggestion is not None

    def test_whitespace_only_rejected(self):
        """Whitespace-only criterion is rejected."""
        errors = _validate_success_criteria(["   "])
        assert len(errors) == 1
        assert errors[0].field == "success_criteria[0]"

    def test_empty_string_rejected(self):
        """Empty string criterion is rejected."""
        errors = _validate_success_criteria([""])
        assert len(errors) == 1

    def test_non_researchable_detected(self):
        """Self-referential criteria are detected."""
        errors = _validate_success_criteria(["Save this report to file"])
        assert len(errors) == 1
        assert "non-researchable" in errors[0].message.lower()

    def test_file_output_detected(self):
        """File output references are detected."""
        errors = _validate_success_criteria(["Output to data.json"])
        assert len(errors) == 1

    def test_valid_criteria_pass(self):
        """Valid criteria pass validation."""
        errors = _validate_success_criteria([
            "Comparison table with pros and cons",
            "At least 3 code examples",
            "Summary of recommendations",
        ])
        assert len(errors) == 0


class TestValidateTitle:
    """Tests for title validation."""

    def test_empty_title_rejected(self):
        """Empty title is rejected."""
        errors = _validate_title("")
        assert len(errors) == 1
        assert errors[0].field == "title"
        assert "empty" in errors[0].message.lower()

    def test_whitespace_only_rejected(self):
        """Whitespace-only title is rejected."""
        errors = _validate_title("   ")
        assert len(errors) == 1
        assert "empty" in errors[0].message.lower()

    def test_short_title_rejected(self):
        """Title under 10 characters is rejected."""
        errors = _validate_title("Test")
        assert len(errors) == 1
        assert "too short" in errors[0].message.lower()
        assert "10 characters" in errors[0].message

    def test_minimum_length_passes(self):
        """Title with exactly 10 characters passes."""
        errors = _validate_title("1234567890")
        assert len(errors) == 0

    def test_valid_title_passes(self):
        """Descriptive title passes validation."""
        errors = _validate_title("Compare GraphQL vs REST for mobile APIs")
        assert len(errors) == 0


class TestValidateLimits:
    """Tests for limit validation."""

    def test_zero_loops_rejected(self):
        """Zero max_loops is rejected."""
        errors = _validate_limits(0, 8000)
        assert len(errors) == 1
        assert errors[0].field == "max_loops"
        assert "at least 1" in errors[0].message

    def test_negative_loops_rejected(self):
        """Negative max_loops is rejected."""
        errors = _validate_limits(-1, 8000)
        assert len(errors) == 1
        assert errors[0].field == "max_loops"

    def test_excessive_loops_rejected(self):
        """max_loops over 10 is rejected."""
        errors = _validate_limits(11, 8000)
        assert len(errors) == 1
        assert errors[0].field == "max_loops"
        assert "exceeds maximum" in errors[0].message

    def test_low_token_limit_rejected(self):
        """Token limit under 1000 is rejected."""
        errors = _validate_limits(3, 500)
        assert len(errors) == 1
        assert errors[0].field == "token_limit"
        assert "too low" in errors[0].message

    def test_excessive_token_limit_rejected(self):
        """Token limit over 32000 is rejected."""
        errors = _validate_limits(3, 50000)
        assert len(errors) == 1
        assert errors[0].field == "token_limit"
        assert "exceeds maximum" in errors[0].message

    def test_valid_limits_pass(self):
        """Valid limits pass validation."""
        errors = _validate_limits(3, 8000)
        assert len(errors) == 0

    def test_boundary_loops_pass(self):
        """Boundary values for loops pass."""
        assert len(_validate_limits(1, 8000)) == 0
        assert len(_validate_limits(10, 8000)) == 0

    def test_boundary_tokens_pass(self):
        """Boundary values for tokens pass."""
        assert len(_validate_limits(3, 1000)) == 0
        assert len(_validate_limits(3, 32000)) == 0


class TestActionableErrors:
    """Tests that all errors are actionable with suggestions."""

    def test_objective_error_has_suggestion(self):
        """Objective errors include suggestions."""
        errors = _validate_objectives([])
        assert errors[0].suggestion is not None
        assert len(errors[0].suggestion) > 0

    def test_criteria_error_has_suggestion(self):
        """Criteria errors include suggestions."""
        errors = _validate_success_criteria([])
        assert errors[0].suggestion is not None

    def test_title_error_has_suggestion(self):
        """Title errors include suggestions."""
        errors = _validate_title("")
        assert errors[0].suggestion is not None

    def test_limits_error_has_suggestion(self):
        """Limit errors include suggestions."""
        errors = _validate_limits(0, 8000)
        assert errors[0].suggestion is not None
        errors = _validate_limits(3, 500)
        assert errors[0].suggestion is not None
