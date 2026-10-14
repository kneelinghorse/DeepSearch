"""Canonical text rules for persisted research reports."""

from __future__ import annotations


class NonCanonicalReportTextError(ValueError):
    """Raised when a persistence boundary receives mutable report text."""


def canonicalize_report_text(text: str) -> str:
    """Return the one report representation used for spans, hashes, and storage.

    Actor output can contain platform newlines or U+0000. Normalize those once,
    before citation extraction assigns codepoint coordinates. Persistence code
    must use :func:`require_canonical_report_text` instead of calling this helper
    so it can never move an already-assigned citation span silently.
    """

    if not isinstance(text, str):
        raise TypeError("report text must be a string")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def require_canonical_report_text(text: str) -> str:
    """Return ``text`` unchanged or reject it at a persistence boundary."""

    if canonicalize_report_text(text) != text:
        raise NonCanonicalReportTextError(
            "report text must contain only LF newlines and no U+0000"
        )
    return text


__all__ = [
    "NonCanonicalReportTextError",
    "canonicalize_report_text",
    "require_canonical_report_text",
]
