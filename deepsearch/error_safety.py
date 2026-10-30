"""Secret-safe exception projection for logs, telemetry, and IPC boundaries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final

SAFE_ERROR_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "authentication",
        "cancelled",
        "conversion",
        "execution",
        "internal",
        "persistence",
        "rate_limit",
        "timeout",
        "transport",
        "validation",
    }
)

_CATEGORY_BY_EXCEPTION_CLASS: Final[dict[str, str]] = {
    "AgentExecutionError": "execution",
    "AgentGenerationError": "execution",
    "AgentMaxStepsError": "execution",
    "APIConnectionError": "transport",
    "APITimeoutError": "timeout",
    "AuthenticationError": "authentication",
    "CancelledError": "cancelled",
    "CodeAgentRunnerError": "execution",
    "ConnectionError": "transport",
    "ConnectError": "transport",
    "ConversionError": "conversion",
    "DatabaseError": "persistence",
    "ExecutionCancelledError": "cancelled",
    "HTTPError": "transport",
    "IntegrityError": "persistence",
    "JSONDecodeError": "validation",
    "LeaseLostError": "cancelled",
    "PermissionDeniedError": "authentication",
    "PermissionError": "authentication",
    "ProcessTerminationError": "execution",
    "RateLimitError": "rate_limit",
    "ReadError": "transport",
    "RemoteExecutionError": "execution",
    "RequestError": "transport",
    "TimeoutError": "timeout",
    "TypeError": "validation",
    "ValidationError": "validation",
    "ValueError": "validation",
    "WriterError": "persistence",
}

_SAFE_EXCEPTION_CLASS_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
)


@dataclass(frozen=True, slots=True)
class SafeErrorDetails:
    """Bounded diagnostic shape that cannot carry exception text."""

    category: str
    exception_class: str

    def __post_init__(self) -> None:
        """Keep direct construction as safe as the public factory helpers."""

        if (
            not isinstance(self.category, str)
            or self.category not in SAFE_ERROR_CATEGORIES
        ):
            object.__setattr__(self, "category", "internal")
        if not (
            isinstance(self.exception_class, str)
            and _SAFE_EXCEPTION_CLASS_RE.fullmatch(self.exception_class)
        ):
            object.__setattr__(self, "exception_class", "Exception")

    def to_dict(self) -> dict[str, str]:
        """Return the canonical two-field boundary shape."""

        return {
            "category": self.category,
            "exception_class": self.exception_class,
        }

    def to_text(self) -> str:
        """Return the canonical compact JSON encoding."""

        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def safe_error_from_parts(
    category: object,
    exception_class: object,
    *,
    default_category: str = "internal",
) -> SafeErrorDetails:
    """Validate untrusted category/class values without retaining raw text."""

    resolved_default = (
        default_category
        if default_category in SAFE_ERROR_CATEGORIES
        else "internal"
    )
    resolved_category = (
        category if isinstance(category, str) and category in SAFE_ERROR_CATEGORIES
        else resolved_default
    )
    resolved_class = (
        exception_class
        if isinstance(exception_class, str)
        and _SAFE_EXCEPTION_CLASS_RE.fullmatch(exception_class)
        else "Exception"
    )
    return SafeErrorDetails(
        category=resolved_category,
        exception_class=resolved_class,
    )


def safe_exception_details(
    exc: BaseException,
    *,
    category: str | None = None,
) -> SafeErrorDetails:
    """Classify ``exc`` using only its type hierarchy, never its message."""

    exception_class = type(exc).__name__
    inferred_category = next(
        (
            _CATEGORY_BY_EXCEPTION_CLASS[class_type.__name__]
            for class_type in type(exc).__mro__
            if class_type.__name__ in _CATEGORY_BY_EXCEPTION_CLASS
        ),
        "internal",
    )
    return safe_error_from_parts(
        category if category is not None else inferred_category,
        exception_class,
    )


def safe_exception_text(
    exc: BaseException,
    *,
    category: str | None = None,
) -> str:
    """Return canonical category/class JSON without calling ``str`` or ``repr``."""

    return safe_exception_details(exc, category=category).to_text()


def normalize_safe_error_text(
    value: object,
    *,
    default_category: str = "internal",
    default_exception_class: str = "Exception",
) -> str | None:
    """Accept only the canonical shape; collapse all free text to safe defaults."""

    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and set(parsed) == {
            "category",
            "exception_class",
        }:
            return safe_error_from_parts(
                parsed.get("category"),
                parsed.get("exception_class"),
                default_category=default_category,
            ).to_text()
    return safe_error_from_parts(
        default_category,
        default_exception_class,
        default_category=default_category,
    ).to_text()


__all__ = [
    "SAFE_ERROR_CATEGORIES",
    "SafeErrorDetails",
    "normalize_safe_error_text",
    "safe_error_from_parts",
    "safe_exception_details",
    "safe_exception_text",
]
