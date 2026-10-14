"""Integration: post-S81 coverage_gate truthfulness on real mission shapes.

S81 m01 (PD122 + PD116a). Two structural fixes on the same data path:

(a) ``_resolve_deliverables`` falls back to ``success_criteria`` when no
    explicit deliverables are declared. Replays the Alexander-mission
    shape (deliverables=null, success_criteria=[6 items]) and verifies
    completeness_check returns a non-empty satisfied/missing partition
    against a representative draft.

(b) ``_canonical_heading`` strips parentheticals before boundary-word
    truncation. Replays the PLUGIN-BACKEND-AUTH-01 parenthetical-bullet
    spec against the post-S80 markdown heading shape and verifies the
    matcher recovers (no false-missing).

The integration boundary is the same one ``codeagent_finalize_node``
uses post-synthesis to populate ``synthesis_telemetry.deliverable_coverage``:
``resolve_deliverables(state)`` → ``completeness_check(draft, spec)``.
"""

from __future__ import annotations

from deepsearch.agent.contract_to_prompt import resolve_deliverables
from deepsearch.agent.tools.completeness_check import completeness_check


# ---------------------------------------------------------------------------
# m01.a — Alexander mission shape (success_criteria-only).
# ---------------------------------------------------------------------------


def _alexander_state() -> dict:
    """Approximates the alexander-ooux-lineage mission as ingested.

    deliverables=null at the TraceLab boundary; success_criteria=[6 items].
    Pre-S81 this produced an empty deliverable spec and the agent's
    Required Deliverables prompt block read ``(none)``.
    """
    return {
        "mission_id": "alexander-ooux-lineage",
        "mission_objectives": [
            "Trace the OOUX lineage from Christopher Alexander's pattern "
            "language to modern object-oriented UX practice",
        ],
        "mission_context": {
            "title": "OOUX lineage from Christopher Alexander",
            "objective": (
                "Trace the OOUX lineage from Christopher Alexander's "
                "pattern language to modern object-oriented UX practice"
            ),
            "success_criteria": [
                "Comparison matrix across schools",
                "Pattern catalog covering each cited motif",
                "Per-school analysis covering provenance and methods",
                "Recommendation grounded in primary sources",
                "Executive summary 200-400 words",
                "References list with at least 8 distinct URLs",
            ],
        },
    }


def _alexander_draft_satisfies_some() -> str:
    """A draft that addresses exactly two of the six success-criteria headings."""
    return (
        "## Executive Summary\n\n"
        "A 200-word summary of the lineage.\n\n"
        "## Comparison Matrix across schools\n\n"
        "| School | Provenance |\n|---|---|\n| A | X |\n\n"
        "## Other Section\n\nbody body body.\n"
    )


def test_alexander_shape_resolves_deliverables_from_success_criteria() -> None:
    state = _alexander_state()
    spec = resolve_deliverables(state)
    assert len(spec) == 6, (
        "m01.a fallback must surface all 6 success_criteria entries as "
        "deliverables when the contract supplies no explicit deliverables."
    )
    assert spec[0].startswith("Comparison matrix")


def test_alexander_shape_completeness_check_reports_honest_shortfall() -> None:
    """The post-m01 telemetry surfaces honest satisfied/missing partition.

    Pre-S81: deliverable_coverage was always ``{satisfied: [], missing: [],
    shortfall_score: 0.0}`` on this shape because the spec was empty.
    Post-S81: spec contains 6 entries; the draft addresses 2; the matcher
    reports 4 missing.
    """
    state = _alexander_state()
    spec = resolve_deliverables(state)
    out = completeness_check(_alexander_draft_satisfies_some(), spec)
    assert out["missing"], "Matcher must honestly surface missing deliverables."
    assert len(out["missing"]) >= 4, (
        f"Draft addresses 2 of 6 success-criteria headings; expected at least "
        f"4 missing, got {len(out['missing'])}: {out['missing']}"
    )
    assert out["shortfall_score"] > 0.0


# ---------------------------------------------------------------------------
# m01.b — PLUGIN-BACKEND-AUTH-01 parenthetical bullet shape.
# ---------------------------------------------------------------------------


def _plugin_backend_auth_spec() -> list[str]:
    """The post-S80 PLUGIN-BACKEND-AUTH-01 deliverables verbatim.

    The parenthetical bullet at index 4 is the bullet that pre-S81
    produced a false-missing because the open paren orphaned the canonical
    heading.
    """
    return [
        "Executive summary 200-400 words",
        "Per-vendor analysis covering each named pattern",
        "Comparison matrix across vendors",
        "Recommendation grounded in vendor docs",
        (
            "Pattern catalogue (persistent-account / ephemeral-session / "
            "hybrid with optional account upgrade)"
        ),
        "References list with at least 12 distinct URLs",
    ]


def _plugin_backend_auth_draft() -> str:
    """Approximates the post-S80 mission's report shape, all 6 sections present."""
    return (
        "## 1. Executive Summary\n\nbody.\n\n"
        "## 2. Per-Vendor Analysis\n\nbody.\n\n"
        "## 3. Comparison Matrix\n\n| A | B |\n|---|---|\n\n"
        "## 4. Recommendation\n\nbody.\n\n"
        "## 5. Pattern / Practice Catalogue\n\nPattern catalogue covering "
        "each named pattern, grounded in vendor docs. See "
        "https://example.org/patterns for the catalogue.\n\n"
        "## 6. References\n\n[1] X — https://example.org/x\n"
    )


def test_plugin_backend_auth_01_matcher_recovers_after_parens_fix() -> None:
    """Post-m01.b: parens-bullet matches the report H2 with no false-missing.

    Pre-S81 the open paren orphaned the canonical heading and substring
    match failed against ``## 5. Pattern / Practice Catalogue``. Post-S81
    the canonical heading is ``Pattern catalogue`` and substring match
    against the body text ``Pattern catalogue covering each named pattern``
    succeeds.

    The test only asserts the parens-bullet recovery — other deliverable
    entries in the spec (e.g. ``Executive summary 200-400 words``) retain
    their pre-S81 substring-match semantics and are not what this guard
    pins; the contrived fixture's full coverage is incidental.
    """
    spec = _plugin_backend_auth_spec()
    draft = _plugin_backend_auth_draft()
    out = completeness_check(draft, spec)

    # The parens-bullet specifically must be in satisfied, not missing.
    assert any("Pattern catalogue" in heading for heading in out["satisfied"]), (
        f"Pattern catalogue heading must satisfy after m01.b parens fix. "
        f"Satisfied: {out['satisfied']}; missing: {out['missing']}"
    )

    # Pre-S81 the canonical heading carried an orphaned open paren; the
    # post-S81 matcher must never emit a heading with an unbalanced paren.
    for heading in out["satisfied"] + out["missing"]:
        assert heading.count("(") == heading.count(")"), (
            f"Canonical heading {heading!r} has unbalanced parens — "
            "regresses S81 m01.b PD116a fix."
        )
