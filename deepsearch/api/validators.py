"""Mission validation for DeepSearch API."""

import re
from typing import Optional
from pydantic import BaseModel

from deepsearch.api.models.requests import ExecuteMissionRequest


class ValidationError(BaseModel):
    """A single validation error."""
    field: str
    message: str
    suggestion: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of mission validation."""
    valid: bool
    errors: list[ValidationError]


# Patterns that indicate non-researchable content
NON_RESEARCHABLE_PATTERNS = [
    (r"this report", "self-referential 'this report'"),
    (r"this document", "self-referential 'this document'"),
    (r"file://", "file:// URL"),
    (r"\.md$", ".md file extension"),
    (r"\.yaml$", ".yaml file extension"),
    (r"\.json$", ".json file extension"),
    (r"cmos/", "internal cmos/ path"),
    (r"deepsearch/", "internal deepsearch/ path"),
    (r"saved to", "'saved to' file output"),
    (r"output to", "'output to' file reference"),
]


def validate_mission(request: ExecuteMissionRequest) -> ValidationResult:
    """
    Validate mission request for semantic correctness.

    Returns ValidationResult with all errors found.
    """
    errors: list[ValidationError] = []

    # Validate objectives
    errors.extend(_validate_objectives(request.objectives))

    # Validate success criteria
    errors.extend(_validate_success_criteria(request.success_criteria))

    # Validate title
    errors.extend(_validate_title(request.title))

    # Validate limits
    errors.extend(_validate_limits(request.max_loops, request.token_limit))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
    )


def _validate_objectives(objectives: list[str]) -> list[ValidationError]:
    """Validate objectives are researchable."""
    errors = []

    if not objectives:
        errors.append(ValidationError(
            field="objectives",
            message="At least one objective is required",
            suggestion="Add a research objective like 'Compare X vs Y'",
        ))
        return errors

    for i, obj in enumerate(objectives):
        # Check for empty/whitespace
        if not obj or not obj.strip():
            errors.append(ValidationError(
                field=f"objectives[{i}]",
                message="Objective cannot be empty",
                suggestion="Provide a specific research question",
            ))
            continue

        # Check for non-researchable patterns
        obj_lower = obj.lower()
        for pattern, description in NON_RESEARCHABLE_PATTERNS:
            if re.search(pattern, obj_lower):
                errors.append(ValidationError(
                    field=f"objectives[{i}]",
                    message=f"Objective contains non-researchable content: {description}",
                    suggestion="Rephrase as a web-searchable question",
                ))
                break

    # Check for duplicates
    seen: dict[str, int] = {}
    for i, obj in enumerate(objectives):
        if not obj or not obj.strip():
            continue
        normalized = obj.strip().lower()
        if normalized in seen:
            errors.append(ValidationError(
                field=f"objectives[{i}]",
                message=f"Duplicate of objectives[{seen[normalized]}]",
                suggestion="Remove duplicate or rephrase",
            ))
        else:
            seen[normalized] = i

    return errors


def _validate_success_criteria(criteria: list[str]) -> list[ValidationError]:
    """Validate success criteria are measurable."""
    errors = []

    if not criteria:
        errors.append(ValidationError(
            field="success_criteria",
            message="At least one success criterion is required",
            suggestion="Add criteria like 'Comparison table with 5+ rows'",
        ))
        return errors

    for i, criterion in enumerate(criteria):
        if not criterion or not criterion.strip():
            errors.append(ValidationError(
                field=f"success_criteria[{i}]",
                message="Criterion cannot be empty",
                suggestion="Provide a measurable success criterion",
            ))
            continue

        # Check for self-referential criteria
        criterion_lower = criterion.lower()
        for pattern, description in NON_RESEARCHABLE_PATTERNS:
            if re.search(pattern, criterion_lower):
                errors.append(ValidationError(
                    field=f"success_criteria[{i}]",
                    message=f"Criterion contains non-researchable reference: {description}",
                    suggestion="Focus on deliverable content, not file output",
                ))
                break

    return errors


def _validate_title(title: str) -> list[ValidationError]:
    """Validate title is meaningful."""
    errors = []

    if not title or not title.strip():
        errors.append(ValidationError(
            field="title",
            message="Title cannot be empty",
            suggestion="Provide a descriptive research title",
        ))
    elif len(title.strip()) < 10:
        errors.append(ValidationError(
            field="title",
            message="Title too short (minimum 10 characters)",
            suggestion="Provide a more descriptive title",
        ))

    return errors


def _validate_limits(max_loops: int, token_limit: int) -> list[ValidationError]:
    """Validate research limits are reasonable."""
    errors = []

    if max_loops < 1:
        errors.append(ValidationError(
            field="max_loops",
            message="max_loops must be at least 1",
            suggestion="Use 2-5 loops for thorough research",
        ))
    elif max_loops > 10:
        errors.append(ValidationError(
            field="max_loops",
            message="max_loops exceeds maximum (10)",
            suggestion="Use 3-5 loops for most research",
        ))

    if token_limit < 1000:
        errors.append(ValidationError(
            field="token_limit",
            message="token_limit too low (minimum 1000)",
            suggestion="Use 4000-8000 for comprehensive research",
        ))
    elif token_limit > 32000:
        errors.append(ValidationError(
            field="token_limit",
            message="token_limit exceeds maximum (32000)",
            suggestion="Use 8000 for balanced cost/quality",
        ))

    return errors
