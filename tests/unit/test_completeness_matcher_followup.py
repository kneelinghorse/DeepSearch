"""DSV-045 follow-up — heading-MATCHER false-MISSING fixes, validated by an
adversarial re-run of the real ``completeness_check`` (verdict DO_NOT_LAND on the
first, narrow attempt → SAFE_TO_LAND after this design).

The S83 m04 work fixed the reworded-heading and lead-noun-in-parenthetical classes.
The 2026-06-16 cross-store re-run + an adversarial verification pass surfaced a
BROADER report-killer class plus a plural gap:

FIX (c) — heading-budget floor (the report-killer class). The ``ceil(N/2)`` overlap
floor scales with the DELIVERABLE's token count, so a bullet with a long descriptive
tail and no boundary word — "Recommendations the team should adopt immediately"
(N=5, floor 3) — could not clear the floor against a heading that reproduces its lead
noun EXACTLY but supplies a single content token (``## Recommendations``, 1 overlap),
and a present, fully-substantive section scored MISSING (shortfall up to 1.0). This
is the exact recall-bias-contract violation (decision #820) DSV-045 set out to kill.
Fixed by capping the floor at the heading's own content-token budget:
``min(ceil(N/2), #heading_content_tokens)`` — a short heading carrying the lead noun
always binds, however verbose the bullet. (The first attempt — truncating the
canonical at ':' / '→' — was REVERTED: it only covered separator-delimited tails,
left the common plain-prose tail unfixed, and introduced a generic-"Noun:" over-match
the adversarial pass flagged. The budget floor subsumes it without that cost.)

FIX (b) — plural deliverable vs singular heading. ``_token_matches`` matched a needle
only as a substring of a heading token (one-directional): "recommendation" bound
"recommendations" but the reverse — plural "recommendations" vs singular
"Recommendation" — did not. Fixed by OR-ing in a singular-stem comparison
(``_depluralize``); the raw-substring clause is kept FIRST so the change only ADDS
matches (recall monotonically non-decreasing).

Both fixes stay AND-ed with ``_section_is_substantive`` — a reworded/plural/verbose
match over a stub body still scores MISSING (the S82 m04 truthfulness gain is intact).
"""

from __future__ import annotations

import pytest

from deepsearch.agent.tools.completeness_check import (
    _heading_content_tokens,
    _heading_matches,
    _overlap_floor,
    completeness_check,
)

_CITED = "See https://example.org/x for the grounding evidence and details."
# A genuinely substantive (>= MIN_SECTION_WORDS=40 non-placeholder words) body so a
# MISS can only be a heading-binding failure, never a substance failure.
_SUBSTANTIVE_BODY = (
    "We recommend the phased rollout described above, sequenced so the highest-impact "
    "and lowest-risk changes land first, with explicit owners, dated milestones, and a "
    "rollback plan for each phase, all grounded in the primary sources surveyed during "
    f"this engagement and cross-checked against vendor documentation. {_CITED}"
)


# ---------------------------------------------------------------------------
# FIX (c) — heading-budget floor closes the report-killer false-MISSING class.
# These are the adversarial pass's confirmed-critical cases, pinned as a gate.
# ---------------------------------------------------------------------------


def test_long_tail_no_boundary_word_binds_exact_lead_noun_heading() -> None:
    """The headline confirmed false-MISSING: a verbose bullet with no ':'/'→' and
    no boundary word binds a heading reproducing its lead noun verbatim."""
    assert (
        _heading_matches("Recommendations the team should adopt immediately", "9. Recommendations")
        is True
    )


@pytest.mark.parametrize(
    "deliverable",
    [
        "Recommendations the team should adopt immediately",
        "Recommendations should be adopted now",
        "Recommendations available to the leadership now",
        "Recommendations throughout the engagement period",
        "Recommendations based on the surveyed evidence",
    ],
)
def test_connective_sweep_present_section_scores_satisfied(deliverable: str) -> None:
    """The broad class, end-to-end: every natural-English connective tail used to
    false-MISS a present, substantive ``## Recommendations`` section. None may now."""
    draft = f"## 9. Recommendations\n\n{_SUBSTANTIVE_BODY}\n"
    out = completeness_check(draft, [deliverable])
    assert out["satisfied"] == [deliverable]
    assert out["missing"] == []
    assert out["shortfall_score"] == 0.0


def test_multi_token_deliverable_binds_short_lead_noun_heading() -> None:
    """A 4-token deliverable binds a 1-content-token heading carrying its lead noun:
    the budget caps the floor at 1, so the single 'risk' overlap binds. (A 2+-token
    synonym heading like '## Risk Management' that shares only 'risk' is an accepted
    residual MISS — the judge rated that looser class 'notable', not the critical
    exact-lead-noun class this floor fixes.)"""
    draft = f"## 5. Risks\n\n{_SUBSTANTIVE_BODY}\n"
    out = completeness_check(draft, ["Risk register and the mitigation plan"])
    assert out["missing"] == []
    assert out["satisfied"] == ["Risk register and the mitigation plan"]


def test_budget_floor_formula() -> None:
    """floor = min(ceil(N/2), #heading_content_tokens). A 1-content-token heading
    caps the floor at 1; section numbers and stopwords are not content tokens."""
    assert _heading_content_tokens("9. Recommendations") == ["recommendations"]
    assert _heading_content_tokens("5. Pattern / Practice Catalogue") == [
        "pattern",
        "practice",
        "catalogue",
    ]
    # N=5 deliverable, 1-token heading budget -> floor min(3, 1) == 1.
    assert _overlap_floor(5) == 3  # the per-N component is unchanged
    assert _heading_matches("Alpha beta gamma delta epsilon", "Alpha") is True


# ---------------------------------------------------------------------------
# FIX (b) — plural deliverable token binds a singular heading token.
# ---------------------------------------------------------------------------


def test_plural_needle_binds_singular_heading() -> None:
    """'Final recommendations' (plural) binds '9. Recommendation' (singular) — the
    reverse direction the one-directional substring match used to miss."""
    assert _heading_matches("Final recommendations", "9. Recommendation") is True


def test_plural_deliverable_now_satisfied() -> None:
    draft = f"## 9. Recommendation\n\n{_SUBSTANTIVE_BODY}\n"
    out = completeness_check(draft, ["Final recommendations"])
    assert out["satisfied"] == ["Final recommendations"]
    assert out["missing"] == []
    assert out["shortfall_score"] == 0.0


# ---------------------------------------------------------------------------
# Recall-bias guards — the relaxations must NOT introduce false-SATISFIED on a
# stub body, must NOT resurrect the pinned 0-overlap negatives, and the reverted
# colon truncation must NOT over-match a generic 'Noun:' deliverable.
# ---------------------------------------------------------------------------


def test_zero_overlap_negatives_preserved() -> None:
    """0 matches never clears a floor of >= 1, regardless of the budget cap."""
    assert _heading_matches("Comparison matrix", "Executive Summary") is False
    assert _heading_matches("Glossary", "Executive Summary") is False


def test_generic_colon_deliverable_does_not_over_match_unrelated_heading() -> None:
    """Regression for the reverted fix (a): the generic 'Plan: ...' / 'Cost: ...'
    deliverables must NOT bind unrelated substantive headings via single-token
    substring collisions (plan in exPLANation, cost in COSTa). With truncation
    reverted, N stays high enough that one collision cannot clear the floor."""
    draft = (
        f"## Detailed Explanation of Findings\n\n{_SUBSTANTIVE_BODY}\n\n"
        f"## Costa Rica Field Study\n\n{_SUBSTANTIVE_BODY}\n"
    )
    out = completeness_check(
        draft, ["Plan: the rollout schedule with phases", "Cost: total spend breakdown"]
    )
    assert out["satisfied"] == []
    assert len(out["missing"]) == 2


def test_plural_match_over_stub_body_still_missing() -> None:
    """FIX (b) stays AND-ed with substance: a plural-matched heading whose body is a
    placeholder stub still scores MISSING."""
    draft = "## 9. Recommendation\n\n[Data unavailable in primary sources] —\n"
    out = completeness_check(draft, ["Final recommendations"])
    assert out["missing"] == ["Final recommendations"]


def test_long_tail_match_over_stub_body_still_missing() -> None:
    """FIX (c) stays AND-ed with substance: the budget floor binds the heading, but a
    stub body still scores MISSING (no false-SATISFIED on an empty section)."""
    draft = "## 9. Recommendations\n\n—\n"
    out = completeness_check(draft, ["Recommendations the team should adopt immediately"])
    assert out["missing"] == ["Recommendations the team should adopt immediately"]
