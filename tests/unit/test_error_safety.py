"""Secret-safe exception projection invariants."""

from __future__ import annotations

import json

from deepsearch.error_safety import (
    SAFE_ERROR_CATEGORIES,
    normalize_safe_error_text,
    safe_error_from_parts,
    safe_exception_details,
    safe_exception_text,
)


def test_error_categories_are_finite_and_pinned() -> None:
    """External diagnostics cannot invent attacker-controlled categories."""

    assert SAFE_ERROR_CATEGORIES == {
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


def test_exception_projection_never_reads_message_or_repr() -> None:
    """Even hostile exception methods cannot expose a URL or response body."""

    class HostileProviderError(RuntimeError):
        def __str__(self) -> str:
            raise AssertionError("exception text must not be read")

        def __repr__(self) -> str:
            raise AssertionError("exception repr must not be read")

    exc = HostileProviderError(
        "https://user:pass@example.test/private?signature=secret "
        "HOSTILE_PROVIDER_BODY"
    )

    details = safe_exception_details(exc, category="transport")
    text = safe_exception_text(exc, category="transport")

    assert details.to_dict() == {
        "category": "transport",
        "exception_class": "HostileProviderError",
    }
    assert json.loads(text) == details.to_dict()
    assert "signature" not in text
    assert "HOSTILE_PROVIDER_BODY" not in text


def test_untrusted_parts_collapse_to_bounded_defaults() -> None:
    """Pipe payloads cannot smuggle text through category or class fields."""

    details = safe_error_from_parts(
        "https://example.test/?token=category-secret",
        "Bad/Class?body=HOSTILE",
        default_category="execution",
    )

    assert details.to_dict() == {
        "category": "execution",
        "exception_class": "Exception",
    }


def test_free_text_is_never_accepted_as_a_safe_error() -> None:
    """Legacy error strings are replaced, not redacted heuristically."""

    hostile = (
        "RuntimeError: https://user:pass@example.test/private?signature=secret "
        "HOSTILE_RESPONSE_BODY"
    )

    normalized = normalize_safe_error_text(
        hostile,
        default_category="execution",
    )

    assert normalized is not None
    assert json.loads(normalized) == {
        "category": "execution",
        "exception_class": "Exception",
    }
    assert hostile not in normalized
    assert "signature" not in normalized
    assert "HOSTILE_RESPONSE_BODY" not in normalized


def test_canonical_error_round_trips_without_new_fields() -> None:
    """The two-field encoding is the only accepted string representation."""

    canonical = safe_error_from_parts("timeout", "TimeoutError").to_text()

    assert normalize_safe_error_text(canonical) == canonical
