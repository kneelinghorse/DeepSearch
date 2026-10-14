"""CI blocks every failure, error and skip; historical reds need explicit review."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import scripts.check_unit_failure_baseline as baseline
from scripts.check_unit_failure_baseline import EXPECTED_FAILURES, evaluate


def _write_report(
    path: Path,
    *,
    failures: set[str],
    skipped: set[str] | None = None,
) -> None:
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for case_id in sorted(failures | (skipped or set())):
        classname, name = case_id.split("::", 1)
        case = ET.SubElement(
            suite,
            "testcase",
            {"classname": classname, "name": name},
        )
        if case_id in failures:
            ET.SubElement(case, "failure")
        if case_id in (skipped or set()):
            ET.SubElement(case, "skipped")
    ET.ElementTree(root).write(path)


def test_clean_suite_is_green_with_no_historical_reds(tmp_path: Path) -> None:
    report = tmp_path / "unit.xml"
    _write_report(report, failures=set(), skipped=None)
    # A report needs at least one case; add a passing one.
    root = ET.parse(report).getroot()
    ET.SubElement(root.find("testsuite"), "testcase", {"classname": "tests.unit.x", "name": "ok"})
    ET.ElementTree(root).write(report)

    assert EXPECTED_FAILURES == frozenset()
    assert evaluate(report) == []


def test_any_failure_blocks_the_offline_gate(tmp_path: Path) -> None:
    report = tmp_path / "unit.xml"
    _write_report(report, failures={"tests.unit.test_new_regression::test_breaks"})

    assert evaluate(report) == [
        "unexpected failure: tests.unit.test_new_regression::test_breaks"
    ]


def test_resolved_pin_requires_baseline_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a pin is ever re-added, silently fixing it still demands review."""
    pin = "tests.unit.test_pin::test_frozen"
    monkeypatch.setattr(baseline, "EXPECTED_FAILURES", frozenset({pin}))
    report = tmp_path / "unit.xml"
    _write_report(report, failures={"tests.unit.test_new_regression::test_breaks"})

    violations = baseline.evaluate(report)

    assert any(item.startswith("unexpected failure:") for item in violations)
    assert f"historical-red baseline changed: {pin}" in violations


def test_skips_never_hide_work_from_the_offline_gate(tmp_path: Path) -> None:
    report = tmp_path / "unit.xml"
    skipped = "tests.unit.test_runtime_boundary::test_required"
    _write_report(report, failures=set(), skipped={skipped})

    assert f"skipped test: {skipped}" in evaluate(report)
