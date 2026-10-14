"""Lossless persisted-report text contract regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepsearch.report.generator import save_report
from deepsearch.report.text import (
    NonCanonicalReportTextError,
    canonicalize_report_text,
    require_canonical_report_text,
)


@pytest.mark.parametrize(
    ("actor_output", "expected"),
    (
        ("one\r\ntwo\rthree\n", "one\ntwo\nthree\n"),
        ("before\x00claim", "beforeclaim"),
        ("\r\n\x00claim\r", "\nclaim\n"),
    ),
)
def test_actor_output_is_canonicalized_once_before_span_assignment(
    actor_output: str,
    expected: str,
) -> None:
    assert canonicalize_report_text(actor_output) == expected


def test_require_canonical_returns_the_same_string_object() -> None:
    report = "# Report\n\nExact claim.\n"
    assert require_canonical_report_text(report) is report


@pytest.mark.parametrize(
    "report",
    ("one\r\ntwo", "one\rtwo", "one\x00two"),
    ids=("crlf", "cr", "nul"),
)
def test_persistence_guard_rejects_any_text_it_would_change(report: str) -> None:
    with pytest.raises(NonCanonicalReportTextError):
        require_canonical_report_text(report)


@pytest.mark.parametrize(
    "report",
    ("one\r\ntwo", "one\rtwo", "one\x00two"),
    ids=("crlf", "cr", "nul"),
)
def test_save_report_rejects_noncanonical_text_without_creating_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report: str,
) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))

    with pytest.raises(NonCanonicalReportTextError):
        save_report(report, mission_id="CANONICAL-01", mission_uuid="12345678")

    assert list(tmp_path.iterdir()) == []


def test_save_report_preserves_exact_utf8_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    report = "# Report\n\nCafé claim.\n"

    path = Path(
        save_report(
            report,
            mission_id="CANONICAL-01",
            mission_uuid="12345678-0000-0000-0000-000000000000",
        )
    )

    assert path.read_bytes() == report.encode("utf-8")
