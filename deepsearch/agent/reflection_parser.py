"""Hardened reflection JSON parser with structured error handling (B3.2).

This module provides robust parsing of LLM reflection outputs with:
- Structured error taxonomy for debugging and telemetry
- JSON extraction from mixed content (fenced blocks, prose)
- Automatic repair prompts for recoverable errors
- Logging hooks for checkpoint integration
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from deepsearch.agent.evaluation import ResearchQualityEvaluation

__all__ = [
    "ReflectionParseError",
    "ReflectionErrorType",
    "ReflectionParseResult",
    "parse_reflection_json",
    "extract_json_from_content",
    "build_repair_prompt",
]

logger = logging.getLogger(__name__)


class ReflectionErrorType(str, Enum):
    """Taxonomy of reflection parsing errors for telemetry."""

    JSON_SYNTAX = "json_syntax"
    JSON_EXTRACTION = "json_extraction"
    MISSING_FIELD = "missing_field"
    INVALID_SCORE = "invalid_score"
    INVALID_TYPE = "invalid_type"
    EMPTY_OBJECTIVES = "empty_objectives"
    CONSTRAINT_VIOLATION = "constraint_violation"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ReflectionParseError:
    """Structured error payload for reflection parsing failures."""

    error_type: ReflectionErrorType
    message: str
    field_path: Optional[str] = None
    raw_value: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error for checkpoint logging."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "field_path": self.field_path,
            "raw_value": self.raw_value[:200] if self.raw_value else None,
            "suggestion": self.suggestion,
        }


@dataclass(slots=True)
class ReflectionParseResult:
    """Result container for reflection parsing attempts."""

    success: bool
    evaluation: Optional[ResearchQualityEvaluation] = None
    errors: List[ReflectionParseError] = field(default_factory=list)
    raw_content: Optional[str] = None
    extracted_json: Optional[str] = None
    attempts: int = 0

    def to_checkpoint_payload(self) -> Dict[str, Any]:
        """Generate checkpoint-compatible error log."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "errors": [err.to_dict() for err in self.errors],
            "extracted_json_length": len(self.extracted_json) if self.extracted_json else 0,
        }


# JSON extraction patterns
_JSON_FENCED_PATTERN = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", re.IGNORECASE)
_JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")
# Pattern to strip JavaScript-style comments from JSON
_JS_COMMENT_PATTERN = re.compile(r"//[^\n]*")


def _extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text using brace counting.

    This handles cases where the greedy regex would capture trailing braces
    from prose after the JSON object.
    """
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    brace_count = 0
    in_string = False
    escape_next = False
    end_idx = start_idx

    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                return text[start_idx:end_idx]

    # Brace count never reached zero - incomplete JSON
    return None


def _strip_js_comments(json_str: str) -> str:
    """Remove JavaScript-style // comments from JSON string.

    LLMs sometimes add inline comments which aren't valid JSON.
    This preserves // inside string literals.
    """
    result: List[str] = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == "\\":
            result.append(char)
            escape_next = True
            i += 1
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if not in_string and char == "/" and i + 1 < len(json_str) and json_str[i + 1] == "/":
            # Skip until end of line
            while i < len(json_str) and json_str[i] != "\n":
                i += 1
            continue

        result.append(char)
        i += 1

    return "".join(result)


def extract_json_from_content(content: str) -> Tuple[Optional[str], Optional[ReflectionParseError]]:
    """Extract JSON object from LLM response content.

    Handles common LLM response patterns:
    - Pure JSON responses
    - JSON in fenced code blocks
    - JSON embedded in explanatory text
    - JSON with trailing prose containing braces
    - JSON with JavaScript-style comments

    Args:
        content: Raw LLM response text.

    Returns:
        Tuple of (extracted_json, error). One will always be None.
    """
    if not content or not content.strip():
        return None, ReflectionParseError(
            error_type=ReflectionErrorType.JSON_EXTRACTION,
            message="Empty response content",
            suggestion="Ensure LLM returns a JSON object",
        )

    text = content.strip()

    # Check for fenced code blocks first (highest precedence)
    fenced_match = _JSON_FENCED_PATTERN.search(text)
    if fenced_match:
        inner = fenced_match.group(1).strip()
        if inner.startswith("{"):
            # Use balanced extraction within the fenced block
            extracted = _extract_balanced_json(inner)
            if extracted:
                # Strip JS comments if present
                cleaned = _strip_js_comments(extracted)
                return cleaned, None

    # Use balanced brace extraction (handles trailing content correctly)
    extracted = _extract_balanced_json(text)
    if extracted:
        # Strip JS comments if present
        cleaned = _strip_js_comments(extracted)
        return cleaned, None

    return None, ReflectionParseError(
        error_type=ReflectionErrorType.JSON_EXTRACTION,
        message="No JSON object found in response",
        raw_value=text[:500] if len(text) > 500 else text,
        suggestion="Wrap your response in a JSON object with curly braces",
    )


def _classify_validation_error(exc: ValidationError) -> List[ReflectionParseError]:
    """Convert Pydantic ValidationError to structured error list."""
    errors: List[ReflectionParseError] = []

    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", []))
        err_type = err.get("type", "")
        msg = err.get("msg", "Validation failed")

        if "missing" in err_type:
            error = ReflectionParseError(
                error_type=ReflectionErrorType.MISSING_FIELD,
                message=msg,
                field_path=loc,
                suggestion=f"Include the required field '{loc}' in your response",
            )
        elif "less_than" in err_type or "greater_than" in err_type or "bounds" in err_type.lower():
            error = ReflectionParseError(
                error_type=ReflectionErrorType.INVALID_SCORE,
                message=msg,
                field_path=loc,
                suggestion="Scores must be between 0.0 and 1.0",
            )
        elif "type" in err_type:
            error = ReflectionParseError(
                error_type=ReflectionErrorType.INVALID_TYPE,
                message=msg,
                field_path=loc,
                suggestion=f"Field '{loc}' has incorrect type",
            )
        elif "empty" in msg.lower() or "at least" in msg.lower():
            error = ReflectionParseError(
                error_type=ReflectionErrorType.EMPTY_OBJECTIVES,
                message=msg,
                field_path=loc,
                suggestion="per_objective_scores must contain at least one objective",
            )
        else:
            error = ReflectionParseError(
                error_type=ReflectionErrorType.CONSTRAINT_VIOLATION,
                message=msg,
                field_path=loc,
            )
        errors.append(error)

    return errors


def parse_reflection_json(content: str) -> ReflectionParseResult:
    """Parse and validate reflection JSON with structured error handling.

    This function attempts to:
    1. Extract JSON from mixed content
    2. Parse as standard JSON
    3. Validate against the Pydantic schema
    4. Recompute derived scores

    Args:
        content: Raw LLM response that should contain reflection JSON.

    Returns:
        ReflectionParseResult with either a valid evaluation or structured errors.
    """
    result = ReflectionParseResult(
        success=False,
        raw_content=content,
        attempts=1,
    )

    # Step 1: Extract JSON from content
    extracted, extraction_error = extract_json_from_content(content)
    if extraction_error:
        result.errors.append(extraction_error)
        logger.debug("JSON extraction failed: %s", extraction_error.message)
        return result

    result.extracted_json = extracted

    # Step 2: Parse JSON syntax
    try:
        data = json.loads(extracted)  # type: ignore[arg-type]
    except json.JSONDecodeError as exc:
        result.errors.append(
            ReflectionParseError(
                error_type=ReflectionErrorType.JSON_SYNTAX,
                message=f"Invalid JSON syntax at position {exc.pos}: {exc.msg}",
                raw_value=extracted[:500] if extracted and len(extracted) > 500 else extracted,
                suggestion="Fix JSON syntax errors (check quotes, commas, brackets)",
            )
        )
        logger.debug("JSON syntax error: %s", exc)
        return result

    # Step 3: Validate against Pydantic schema
    try:
        evaluation = ResearchQualityEvaluation.model_validate(data)
    except ValidationError as exc:
        result.errors.extend(_classify_validation_error(exc))
        logger.debug("Schema validation failed with %d errors", len(result.errors))
        return result

    # Step 4: Recompute derived scores
    evaluation.recompute_overall()

    result.success = True
    result.evaluation = evaluation
    return result


def build_repair_prompt(
    base_prompt: str,
    *,
    run_index: int,
    attempt: int,
    errors: List[ReflectionParseError],
    last_output: Optional[str] = None,
) -> str:
    """Generate a targeted repair prompt based on structured errors.

    Unlike the legacy approach that just says "invalid JSON", this function
    provides specific guidance based on the error taxonomy.

    Args:
        base_prompt: Original reflection prompt.
        run_index: Current consensus run number (0-indexed).
        attempt: Current retry attempt (1-indexed).
        errors: List of structured errors from previous attempt.
        last_output: Previous LLM response (truncated in prompt).

    Returns:
        Repair prompt with specific fix instructions.
    """
    error_details: List[str] = []
    for err in errors[:3]:  # Limit to top 3 errors
        if err.field_path:
            error_details.append(f"- {err.error_type.value}: {err.message} (field: {err.field_path})")
        else:
            error_details.append(f"- {err.error_type.value}: {err.message}")
        if err.suggestion:
            error_details.append(f"  Fix: {err.suggestion}")

    error_block = "\n".join(error_details)

    output_preview = ""
    if last_output:
        preview = last_output[:300] if len(last_output) > 300 else last_output
        output_preview = f"\n\nPrevious response (truncated):\n{preview}"

    repair_instructions = f"""
**REPAIR REQUIRED** (Attempt {attempt} for run {run_index + 1})

Your previous response could not be parsed. Specific errors:
{error_block}

Please respond with ONLY a valid JSON object matching this exact schema:
{{
  "overall_coverage_score": <float 0.0-1.0>,
  "is_mission_complete": <boolean>,
  "confidence_score": <float 0.0-1.0>,
  "aggregate_rationale": "<string explaining overall assessment>",
  "per_objective_scores": [
    {{"objective_id": "<string>", "score": <0.0|0.5|0.8|1.0>, "reasoning": "<string>"}}
  ]
}}

IMPORTANT: Do NOT include any text before or after the JSON object.
{output_preview}
"""

    return f"{base_prompt}\n\n{repair_instructions}"
