"""Unit tests for ``deepsearch.agent.tools.scratchpad``.

Verbatim behavioral parity with the S62 in-runner scratchpad pattern: keys are
mission-scoped via the closure-captured dict, ``read`` returns ``""`` on miss,
``write`` overwrites prior values, two scratchpads from separate ``build_*``
calls do NOT share state.
"""

from __future__ import annotations

import pytest

from deepsearch.agent.tools.scratchpad import build_scratchpad


def test_read_after_write_hit():
    bundle = build_scratchpad()
    confirmation = bundle.write("aws_lambda_notes", "timeout = 15 minutes")
    assert "scratchpad['aws_lambda_notes']" in confirmation
    assert bundle.read("aws_lambda_notes") == "timeout = 15 minutes"


def test_read_on_empty_miss_returns_empty_string():
    bundle = build_scratchpad()
    assert bundle.read("never_written") == ""


def test_write_overwrites_prior_value():
    bundle = build_scratchpad()
    bundle.write("draft", "v1")
    bundle.write("draft", "v2 — revised")
    assert bundle.read("draft") == "v2 — revised"


def test_state_is_isolated_between_bundles():
    a = build_scratchpad()
    b = build_scratchpad()
    a.write("k", "from_a")
    assert b.read("k") == ""


def test_state_dict_reflects_writes():
    bundle = build_scratchpad()
    bundle.write("k1", "v1")
    bundle.write("k2", "v2")
    assert bundle.state == {"k1": "v1", "k2": "v2"}


def test_write_rejects_empty_key():
    bundle = build_scratchpad()
    with pytest.raises(ValueError):
        bundle.write("", "value")
    with pytest.raises(ValueError):
        bundle.write("   ", "value")


def test_write_rejects_non_string_value():
    bundle = build_scratchpad()
    with pytest.raises(ValueError):
        bundle.write("k", 123)  # type: ignore[arg-type]


def test_read_rejects_non_string_key():
    bundle = build_scratchpad()
    with pytest.raises(ValueError):
        bundle.read(123)  # type: ignore[arg-type]
