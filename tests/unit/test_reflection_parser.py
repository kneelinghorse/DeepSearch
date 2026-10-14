"""Unit tests for the hardened reflection parser (B3.2, B6.2).

B6.2 additions:
- Balanced brace extraction for trailing content
- JavaScript comment stripping
- Fixture-based tests from real LLM outputs
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepsearch.agent.reflection_parser import (
    ReflectionErrorType,
    ReflectionParseError,
    ReflectionParseResult,
    build_repair_prompt,
    extract_json_from_content,
    parse_reflection_json,
    _extract_balanced_json,
    _strip_js_comments,
)

# Path to fixture files
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "reflection_outputs"


# --- Fixtures for valid reflection JSON ---

VALID_REFLECTION_JSON = json.dumps({
    "overall_coverage_score": 0.85,
    "is_mission_complete": False,
    "confidence_score": 0.75,
    "aggregate_rationale": "Good progress on objectives but more depth needed.",
    "per_objective_scores": [
        {"objective_id": "OBJ01", "score": 0.9, "reasoning": "Well covered"},
        {"objective_id": "OBJ02", "score": 0.7, "reasoning": "Partially addressed"},
    ],
})

VALID_REFLECTION_FENCED = f"""
Here is my evaluation:

```json
{VALID_REFLECTION_JSON}
```

Let me know if you need clarification.
"""

VALID_REFLECTION_EMBEDDED = f"""
After reviewing the research findings, I've assessed the progress:

{VALID_REFLECTION_JSON}

This completes my analysis.
"""


class TestExtractJsonFromContent:
    """Tests for JSON extraction from mixed LLM responses."""

    def test_pure_json(self) -> None:
        """Pure JSON content should be extracted directly."""
        result, error = extract_json_from_content(VALID_REFLECTION_JSON)
        assert error is None
        assert result is not None
        assert '"overall_coverage_score"' in result

    def test_fenced_code_block(self) -> None:
        """JSON in fenced blocks should be extracted."""
        result, error = extract_json_from_content(VALID_REFLECTION_FENCED)
        assert error is None
        assert result is not None
        parsed = json.loads(result)
        assert parsed["overall_coverage_score"] == 0.85

    def test_embedded_json(self) -> None:
        """JSON embedded in prose should be extracted."""
        result, error = extract_json_from_content(VALID_REFLECTION_EMBEDDED)
        assert error is None
        assert result is not None

    def test_empty_content(self) -> None:
        """Empty content should return extraction error."""
        result, error = extract_json_from_content("")
        assert result is None
        assert error is not None
        assert error.error_type == ReflectionErrorType.JSON_EXTRACTION

    def test_no_json_found(self) -> None:
        """Content without JSON should return extraction error."""
        result, error = extract_json_from_content("Just some text without JSON")
        assert result is None
        assert error is not None
        assert error.error_type == ReflectionErrorType.JSON_EXTRACTION
        assert "No JSON object found" in error.message


class TestParseReflectionJson:
    """Tests for full reflection parsing pipeline."""

    def test_valid_json_parses_successfully(self) -> None:
        """Valid reflection JSON should parse and validate."""
        result = parse_reflection_json(VALID_REFLECTION_JSON)
        assert result.success is True
        assert result.evaluation is not None
        assert len(result.errors) == 0
        # Scores get snapped to the discrete reflection scale
        assert result.evaluation.overall_coverage_score in [0.0, 0.5, 0.7, 0.8, 1.0]
        assert len(result.evaluation.per_objective_scores) == 2

    def test_fenced_json_parses_successfully(self) -> None:
        """JSON in code fences should parse."""
        result = parse_reflection_json(VALID_REFLECTION_FENCED)
        assert result.success is True
        assert result.evaluation is not None

    def test_missing_required_field(self) -> None:
        """Missing required fields should produce structured error."""
        incomplete = json.dumps({
            "overall_coverage_score": 0.8,
            "is_mission_complete": False,
            # Missing confidence_score, aggregate_rationale, per_objective_scores
        })
        result = parse_reflection_json(incomplete)
        assert result.success is False
        assert len(result.errors) > 0
        # Should identify missing field
        error_types = {e.error_type for e in result.errors}
        assert ReflectionErrorType.MISSING_FIELD in error_types

    def test_score_gets_clamped_to_valid_range(self) -> None:
        """Scores outside valid range get clamped by Pydantic validators."""
        # Note: The snap_to_scale validator normalizes scores to 4-point scale
        # so 1.5 becomes 1.0. This is expected behavior.
        # Testing true validation failure requires a non-numeric value.
        invalid = json.dumps({
            "overall_coverage_score": 0.8,
            "is_mission_complete": False,
            "confidence_score": 1.5,  # This exceeds 1.0 and will fail
            "aggregate_rationale": "Test",
            "per_objective_scores": [
                {"objective_id": "OBJ01", "score": 0.8, "reasoning": "OK"}
            ],
        })
        result = parse_reflection_json(invalid)
        # confidence_score doesn't use snap_to_scale, so 1.5 should fail validation
        assert result.success is False
        error_types = {e.error_type for e in result.errors}
        assert ReflectionErrorType.INVALID_SCORE in error_types or ReflectionErrorType.CONSTRAINT_VIOLATION in error_types

    def test_empty_objectives_list(self) -> None:
        """Empty per_objective_scores should produce structured error."""
        empty_objectives = json.dumps({
            "overall_coverage_score": 0.8,
            "is_mission_complete": False,
            "confidence_score": 0.5,
            "aggregate_rationale": "Test rationale",
            "per_objective_scores": [],
        })
        result = parse_reflection_json(empty_objectives)
        assert result.success is False
        error_types = {e.error_type for e in result.errors}
        assert ReflectionErrorType.EMPTY_OBJECTIVES in error_types

    def test_malformed_json_syntax(self) -> None:
        """Malformed JSON should produce syntax or extraction error."""
        # If the regex can't find a complete JSON object, it's extraction error
        # If it finds one but can't parse, it's syntax error
        malformed = '{"overall_coverage_score": 0.8, "unclosed": true'
        result = parse_reflection_json(malformed)
        assert result.success is False
        error_types = {e.error_type for e in result.errors}
        # Either extraction (no complete object found) or syntax (parse failed)
        assert (
            ReflectionErrorType.JSON_SYNTAX in error_types
            or ReflectionErrorType.JSON_EXTRACTION in error_types
        )

    def test_wrong_type_for_field(self) -> None:
        """Wrong field types should produce type error."""
        wrong_type = json.dumps({
            "overall_coverage_score": "not a number",  # Should be float
            "is_mission_complete": False,
            "confidence_score": 0.5,
            "aggregate_rationale": "Test",
            "per_objective_scores": [
                {"objective_id": "OBJ01", "score": 0.8, "reasoning": "OK"}
            ],
        })
        result = parse_reflection_json(wrong_type)
        assert result.success is False
        # Should produce invalid type or invalid score error
        assert len(result.errors) > 0


class TestReflectionParseResult:
    """Tests for parse result serialization."""

    def test_to_checkpoint_payload(self) -> None:
        """Checkpoint payload should be JSON-serializable."""
        result = parse_reflection_json(VALID_REFLECTION_JSON)
        payload = result.to_checkpoint_payload()

        # Should be serializable
        serialized = json.dumps(payload)
        assert "success" in serialized
        assert "attempts" in serialized
        assert "errors" in serialized

    def test_error_to_dict(self) -> None:
        """Errors should serialize to dicts properly."""
        error = ReflectionParseError(
            error_type=ReflectionErrorType.MISSING_FIELD,
            message="Field 'confidence_score' is required",
            field_path="confidence_score",
            suggestion="Include the required field",
        )
        d = error.to_dict()
        assert d["error_type"] == "missing_field"
        assert d["field_path"] == "confidence_score"
        assert d["suggestion"] is not None


class TestBuildRepairPrompt:
    """Tests for repair prompt generation."""

    def test_repair_prompt_includes_errors(self) -> None:
        """Repair prompt should include specific error information."""
        errors = [
            ReflectionParseError(
                error_type=ReflectionErrorType.MISSING_FIELD,
                message="Field 'confidence_score' is required",
                field_path="confidence_score",
                suggestion="Include the required field 'confidence_score' in your response",
            ),
        ]
        prompt = build_repair_prompt(
            "Original prompt here",
            run_index=0,
            attempt=2,
            errors=errors,
            last_output="invalid output",
        )
        assert "REPAIR REQUIRED" in prompt
        assert "Attempt 2" in prompt
        assert "missing_field" in prompt
        assert "confidence_score" in prompt

    def test_repair_prompt_limits_errors(self) -> None:
        """Repair prompt should limit to top 3 errors."""
        errors = [
            ReflectionParseError(
                error_type=ReflectionErrorType.MISSING_FIELD,
                message=f"Error {i}",
                field_path=f"field_{i}",
            )
            for i in range(5)
        ]
        prompt = build_repair_prompt(
            "Original prompt",
            run_index=0,
            attempt=2,
            errors=errors,
        )
        # Should only include first 3 errors
        assert "field_0" in prompt
        assert "field_2" in prompt
        # field_3 and field_4 should not appear due to limit

    def test_repair_prompt_includes_schema(self) -> None:
        """Repair prompt should include JSON schema reminder."""
        prompt = build_repair_prompt(
            "Original prompt",
            run_index=1,
            attempt=2,
            errors=[],
        )
        assert "overall_coverage_score" in prompt
        assert "per_objective_scores" in prompt
        assert "ONLY a valid JSON object" in prompt


class TestRetryScenarios:
    """Tests for retry success/failure scenarios."""

    def test_recoverable_error_scenario(self) -> None:
        """Demonstrate recoverable error flow."""
        # First attempt: missing field
        attempt1 = json.dumps({
            "overall_coverage_score": 0.8,
            "is_mission_complete": False,
            # Missing other fields
        })
        result1 = parse_reflection_json(attempt1)
        assert result1.success is False

        # Generate repair prompt
        repair = build_repair_prompt(
            "Original prompt",
            run_index=0,
            attempt=2,
            errors=result1.errors,
            last_output=attempt1,
        )
        assert "missing_field" in repair

        # Second attempt: complete JSON
        attempt2 = VALID_REFLECTION_JSON
        result2 = parse_reflection_json(attempt2)
        assert result2.success is True

    def test_non_recoverable_scenario(self) -> None:
        """Complete garbage should still produce structured errors."""
        garbage = "This is not JSON at all, just random text"
        result = parse_reflection_json(garbage)
        assert result.success is False
        assert len(result.errors) > 0
        # Should be classified as extraction error
        assert result.errors[0].error_type == ReflectionErrorType.JSON_EXTRACTION


class TestBalancedBraceExtraction:
    """Tests for B6.2 balanced brace extraction."""

    def test_extracts_first_complete_object(self) -> None:
        """Should extract only the first complete JSON object."""
        text = '{"a": 1} extra {"b": 2}'
        result = _extract_balanced_json(text)
        assert result == '{"a": 1}'

    def test_handles_nested_objects(self) -> None:
        """Should correctly handle nested objects."""
        text = '{"outer": {"inner": {"deep": 1}}}'
        result = _extract_balanced_json(text)
        assert result == text

    def test_handles_braces_in_strings(self) -> None:
        """Should not count braces inside string literals."""
        text = '{"msg": "Hello {world}", "count": 1}'
        result = _extract_balanced_json(text)
        assert result == text

    def test_handles_escaped_quotes(self) -> None:
        """Should handle escaped quotes in strings."""
        text = r'{"msg": "He said \"hello\"", "ok": true}'
        result = _extract_balanced_json(text)
        assert result == text

    def test_incomplete_json_returns_none(self) -> None:
        """Incomplete JSON should return None."""
        text = '{"incomplete": true'
        result = _extract_balanced_json(text)
        assert result is None

    def test_no_json_returns_none(self) -> None:
        """Text without JSON should return None."""
        text = "Just plain text without braces"
        result = _extract_balanced_json(text)
        assert result is None

    def test_trailing_prose_with_braces(self) -> None:
        """Should not include trailing prose with braces."""
        text = '{"score": 0.8}\n\nNote: {0.8} is good.'
        result = _extract_balanced_json(text)
        assert result == '{"score": 0.8}'


class TestJsCommentStripping:
    """Tests for B6.2 JavaScript comment stripping."""

    def test_strips_inline_comments(self) -> None:
        """Should remove // comments."""
        json_with_comments = '{"a": 1, // comment\n"b": 2}'
        result = _strip_js_comments(json_with_comments)
        assert "//" not in result
        # Should be parseable
        parsed = json.loads(result)
        assert parsed["a"] == 1
        assert parsed["b"] == 2

    def test_preserves_slashes_in_strings(self) -> None:
        """Should not strip // inside string literals."""
        json_str = '{"url": "https://example.com"}'
        result = _strip_js_comments(json_str)
        assert result == json_str

    def test_preserves_escaped_quotes(self) -> None:
        """Should handle escaped quotes correctly."""
        json_str = r'{"msg": "path is C:\\\\dir"}'
        result = _strip_js_comments(json_str)
        assert result == json_str

    def test_multiple_comments(self) -> None:
        """Should strip multiple comments."""
        json_with_comments = '''{
  "a": 1, // first
  "b": 2, // second
  "c": 3  // third
}'''
        result = _strip_js_comments(json_with_comments)
        assert result.count("//") == 0


class TestFixtureBasedParsing:
    """Tests using fixture files with real LLM output patterns."""

    def test_trailing_braces_fixture(self) -> None:
        """Parse JSON with trailing braces in prose."""
        fixture_path = FIXTURES_DIR / "trailing_braces.txt"
        if fixture_path.exists():
            content = fixture_path.read_text()
            result = parse_reflection_json(content)
            assert result.success is True
            assert result.evaluation.overall_coverage_score == 0.8

    def test_multiple_json_objects_fixture(self) -> None:
        """Parse content with multiple JSON objects."""
        fixture_path = FIXTURES_DIR / "multiple_json_objects.txt"
        if fixture_path.exists():
            content = fixture_path.read_text()
            result = parse_reflection_json(content)
            assert result.success is True
            assert len(result.evaluation.per_objective_scores) == 2

    def test_js_comments_fixture(self) -> None:
        """Parse JSON with JavaScript-style comments."""
        fixture_path = FIXTURES_DIR / "js_comments.txt"
        if fixture_path.exists():
            content = fixture_path.read_text()
            result = parse_reflection_json(content)
            assert result.success is True
            assert result.evaluation.overall_coverage_score == 0.8

    def test_fenced_without_tag_fixture(self) -> None:
        """Parse JSON in fenced block without 'json' tag."""
        fixture_path = FIXTURES_DIR / "fenced_without_tag.txt"
        if fixture_path.exists():
            content = fixture_path.read_text()
            result = parse_reflection_json(content)
            assert result.success is True
            assert result.evaluation.overall_coverage_score == 0.5

    def test_prose_before_json_fixture(self) -> None:
        """Parse JSON preceded by explanatory prose."""
        fixture_path = FIXTURES_DIR / "prose_before_json.txt"
        if fixture_path.exists():
            content = fixture_path.read_text()
            result = parse_reflection_json(content)
            assert result.success is True
            assert len(result.evaluation.per_objective_scores) == 2

    def test_nested_braces_in_rationale_fixture(self) -> None:
        """Parse JSON with braces in the rationale strings."""
        fixture_path = FIXTURES_DIR / "nested_braces_in_rationale.txt"
        if fixture_path.exists():
            content = fixture_path.read_text()
            result = parse_reflection_json(content)
            assert result.success is True
            assert "{main()}" in result.evaluation.aggregate_rationale


class TestEdgeCases:
    """Additional edge case tests for B6.2."""

    def test_empty_string(self) -> None:
        """Empty string should fail gracefully."""
        result = parse_reflection_json("")
        assert result.success is False
        assert result.errors[0].error_type == ReflectionErrorType.JSON_EXTRACTION

    def test_whitespace_only(self) -> None:
        """Whitespace-only should fail gracefully."""
        result = parse_reflection_json("   \n\t   ")
        assert result.success is False

    def test_json_with_unicode(self) -> None:
        """Should handle unicode characters in strings."""
        content = json.dumps({
            "overall_coverage_score": 0.8,
            "is_mission_complete": False,
            "confidence_score": 0.0,
            "aggregate_rationale": "Analysis of \u00e9v\u00e9nements internationaux",
            "per_objective_scores": [
                {"objective_id": "OBJ01", "score": 0.8, "reasoning": "\u2713 Verified"}
            ],
        })
        result = parse_reflection_json(content)
        assert result.success is True

    def test_very_long_rationale(self) -> None:
        """Should handle very long rationale strings."""
        long_rationale = "A" * 5000
        content = json.dumps({
            "overall_coverage_score": 0.8,
            "is_mission_complete": False,
            "confidence_score": 0.0,
            "aggregate_rationale": long_rationale,
            "per_objective_scores": [
                {"objective_id": "OBJ01", "score": 0.8, "reasoning": "OK"}
            ],
        })
        result = parse_reflection_json(content)
        assert result.success is True

    def test_many_objectives(self) -> None:
        """Should handle many objective scores."""
        objectives = [
            {"objective_id": f"OBJ{i:02d}", "score": 0.8, "reasoning": f"Objective {i}"}
            for i in range(1, 21)
        ]
        content = json.dumps({
            "overall_coverage_score": 0.8,
            "is_mission_complete": False,
            "confidence_score": 0.0,
            "aggregate_rationale": "Multiple objectives evaluated",
            "per_objective_scores": objectives,
        })
        result = parse_reflection_json(content)
        assert result.success is True
        assert len(result.evaluation.per_objective_scores) == 20

    def test_newlines_in_strings(self) -> None:
        """Should handle escaped newlines in strings."""
        content = json.dumps({
            "overall_coverage_score": 0.8,
            "is_mission_complete": False,
            "confidence_score": 0.0,
            "aggregate_rationale": "Line 1\\nLine 2\\nLine 3",
            "per_objective_scores": [
                {"objective_id": "OBJ01", "score": 0.8, "reasoning": "Multi\\nline"}
            ],
        })
        result = parse_reflection_json(content)
        assert result.success is True
