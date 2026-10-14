#!/usr/bin/env python3
"""Require the full unit suite to match the explicit historical-red baseline."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Historical-red pins that may fail without blocking CI. Empty since S97 retired
# the S87 CAT-readiness pins (decision 975) and corrected the S93 assertion. An entry
# requires an explicit, reviewed baseline change.
EXPECTED_FAILURES: frozenset[str] = frozenset()


def _case_id(case: ET.Element) -> str:
    return f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"


def evaluate(path: Path) -> list[str]:
    """Return every baseline violation found in one pytest JUnit report."""

    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failures = {
        _case_id(case)
        for case in cases
        if case.find("failure") is not None
    }
    errors = {
        _case_id(case)
        for case in cases
        if case.find("error") is not None
    }
    skipped = {
        _case_id(case)
        for case in cases
        if case.find("skipped") is not None
    }

    violations: list[str] = []
    if not cases:
        violations.append("JUnit report contains no test cases")
    for case_id in sorted(failures - EXPECTED_FAILURES):
        violations.append(f"unexpected failure: {case_id}")
    for case_id in sorted(EXPECTED_FAILURES - failures):
        violations.append(f"historical-red baseline changed: {case_id}")
    for case_id in sorted(errors):
        violations.append(f"test error: {case_id}")
    for case_id in sorted(skipped):
        violations.append(f"skipped test: {case_id}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml", type=Path)
    args = parser.parse_args(argv)

    try:
        violations = evaluate(args.junit_xml)
    except (ET.ParseError, OSError) as exc:
        print(f"invalid JUnit report: {exc}", file=sys.stderr)
        return 1
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print(
        "Unit suite matches the explicit baseline: "
        f"{len(EXPECTED_FAILURES)} historical failures, zero errors/skips."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
