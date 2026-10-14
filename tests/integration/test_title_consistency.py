"""S80 m02 (PD107) — wrapper-H1 vs protocol-`.title` consistency.

Pre-S80 the mission title was independently re-derived in three sites:

* ``deepsearch/agent/codeagent_finalize_node.py:100`` — fed the local report
  filename and the wrapper-H1 path.
* ``deepsearch/report/generator.py:48`` — fed the wrapper H1 emitted by
  ``build_report_header``.
* ``deepsearch/report/protocol.py:_derive_title`` — fed the protocol JSON
  ``.title`` field returned to TraceLab.

When ``mission_context["title"]`` was empty, the three paths took
*different* fallback decisions: the wrapper rendered the objective; the
protocol rendered ``"Research Mission: <id>"``. Same execution, two
strings on disk and in TraceLab.

After S80 m02 the canonical title is pinned ONCE inside
``empty_state()`` to ``state["mission_title"]`` and all three readers
collapse to a single-key read. The integration tests below build a real
``AgentState`` via ``empty_state()`` from three fallback fixtures (full
title; empty title with objective; empty title and short objective) and
assert that the wrapper H1 and the protocol JSON ``.title`` field
produce IDENTICAL strings end-to-end.
"""

from __future__ import annotations

import re

import pytest

from deepsearch.agent.state import AgentState, empty_state
from deepsearch.report.generator import generate_markdown_report
from deepsearch.report.protocol import generate_mission_protocol


def _extract_wrapper_h1(report_content: str) -> str:
    """Return the wrapper H1 line (without the leading `# ` and trailing newline)."""
    match = re.search(r"^# (Research Report: .+)$", report_content, re.MULTILINE)
    assert match is not None, (
        "Wrapper H1 (`# Research Report: <id> - <title>`) must be present in "
        f"the rendered markdown. Got first 200 chars:\n{report_content[:200]!r}"
    )
    return match.group(1)


def _wrapper_title_only(wrapper_h1: str, mission_id: str) -> str:
    """Strip the `Research Report: <id> - ` prefix to return just the title."""
    prefix = f"Research Report: {mission_id} - "
    assert wrapper_h1.startswith(prefix), (
        f"Wrapper H1 must start with {prefix!r}; got {wrapper_h1!r}"
    )
    return wrapper_h1[len(prefix):]


@pytest.mark.parametrize(
    "fixture_name,mission_context,expected_title",
    [
        # Fixture 1 — full title present.
        (
            "full_title",
            {
                "title": "Backend Identity Model Research",
                "objective": "Investigate token rotation patterns",
            },
            "Backend Identity Model Research",
        ),
        # Fixture 2 — empty title; canonical falls back to mission_id (the
        # post-S80 fallback contract — empty_state() pins mission_title to
        # mission_id when context["title"] is empty/whitespace).
        (
            "empty_title_with_objective",
            {
                "title": "",
                "objective": "Investigate the impact of macroscopic critique on agent self-termination behavior",
            },
            "PLUGIN-BACKEND-AUTH-01",  # mission_id fallback
        ),
        # Fixture 3 — empty title, short objective. Same fallback semantics.
        (
            "empty_title_short_objective",
            {
                "title": "",
                "objective": "AB",
            },
            "PLUGIN-BACKEND-AUTH-01",  # mission_id fallback
        ),
    ],
)
def test_wrapper_h1_matches_protocol_title(
    fixture_name: str,
    mission_context: dict,
    expected_title: str,
) -> None:
    """End-to-end: empty_state pin -> generator wrapper H1 == protocol .title.

    PD114 verification expectation #2 — the local-testable assertion of the
    "single source of truth" contract closed at m02.
    """
    mission_id = "PLUGIN-BACKEND-AUTH-01"
    state: AgentState = empty_state(
        mission_id=mission_id,
        mission_objectives=["Document the contract"],
        mission_context=mission_context,
    )

    # Sanity — the canonical key is pinned by empty_state().
    assert state["mission_title"] == expected_title, (
        f"[{fixture_name}] empty_state must pin state['mission_title'] to "
        f"the canonical resolved value; got {state['mission_title']!r}, "
        f"expected {expected_title!r}"
    )

    # Path 1 — wrapper H1 emitted by generate_markdown_report.
    report_content = generate_markdown_report(state)
    wrapper_h1 = _extract_wrapper_h1(report_content)
    wrapper_title = _wrapper_title_only(wrapper_h1, mission_id)

    # Path 2 — protocol JSON `.title` field returned to TraceLab.
    protocol = generate_mission_protocol(state)
    protocol_title = protocol.title

    # The load-bearing assertion — wrapper and protocol agree.
    assert wrapper_title == protocol_title, (
        f"[{fixture_name}] S80 m02 PD107 — wrapper H1 title and protocol "
        f"`.title` must match. Wrapper={wrapper_title!r}, "
        f"protocol={protocol_title!r}. The single-source-of-truth contract "
        "is broken."
    )
    assert protocol_title == expected_title, (
        f"[{fixture_name}] both paths must return the canonical title "
        f"{expected_title!r}; got {protocol_title!r}"
    )


def test_full_title_round_trip_uses_normalized_string() -> None:
    """Loader's ``normalize_mission_title`` strips a duplicated `<id> - ` prefix.

    When the user-supplied title repeats the mission_id (`<id> - Real Title`),
    the loader normalizes to `Real Title`. The empty_state pin reads the
    already-normalized context["title"] value and must propagate the clean
    form to both readers identically.
    """
    mission_id = "DEMO-01"
    state: AgentState = empty_state(
        mission_id=mission_id,
        mission_objectives=["Validate normalization round trip"],
        mission_context={
            # Already-normalized form — empty_state simply reads it.
            "title": "Real Title After Normalize",
        },
    )

    report_content = generate_markdown_report(state)
    wrapper_h1 = _extract_wrapper_h1(report_content)
    wrapper_title = _wrapper_title_only(wrapper_h1, mission_id)

    protocol_title = generate_mission_protocol(state).title

    assert wrapper_title == protocol_title == "Real Title After Normalize"
