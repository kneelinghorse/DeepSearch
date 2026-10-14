"""S83 m04 RD2 invariants — completeness_check heading-matcher false-MISSING fix.

Two reproduced live false-MISSING classes, fixed WITHOUT weakening the S82 m04
truthfulness gains (heading-scoped + substance-aware + entity-count):

CASE 1 (synonym/reword): the pre-S83 ``_heading_matches`` required EVERY
canonical content token to substring-match, so ``"A final recommendation"`` ->
``{final, recommendation}`` false-failed against ``"## 9. Recommendation"``
(``final`` absent) and a present substantive section scored MISSING. Fixed by a
token-overlap floor: ``N==1`` exact substring, ``N>=2`` ``>= ceil(N/2)``.

CASE 2 (lead-noun dropped into a parenthetical): ``"Risk register (likelihood /
impact / mitigation)"`` canonicalises to ``"Risk register"`` (parens stripped),
0 overlap against a reworded ``"## Likelihood and Impact Mitigation Table"``. A
pure overlap relaxation cannot fix this — it needs the deliverable's
parenthetical sub-terms as a second binding signal.

The fuzzier/paren matching stays AND-ed with ``_section_is_substantive`` so a
reworded heading over a stub body still scores MISSING; ``_canonical_heading``
is unchanged (the fix is in the matcher, not canonicalisation).
"""

from __future__ import annotations

import inspect
import re

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.tools.completeness_check import (
    MIN_SECTION_WORDS,
    _canonical_heading,
    _heading_matches,
    _overlap_floor,
    _parenthetical_tokens,
    completeness_check,
)

_CITED = "See https://example.org/x for the grounding evidence and details."


# ---------------------------------------------------------------------------
# CASE 1 — synonym / reworded heading binds under the overlap floor.
# ---------------------------------------------------------------------------


def test_case1_heading_matches_reworded_recommendation() -> None:
    """``_heading_matches('A final recommendation', '9. Recommendation')`` True.

    Pre-S83 this was False (``final`` absent from the heading). The N==2 floor
    is ceil(2/2)==1, so the single ``recommendation`` overlap now binds.
    """
    assert _heading_matches("A final recommendation", "9. Recommendation") is True


def test_case1_present_section_scores_satisfied() -> None:
    """End-to-end: a present substantive '## 9. Recommendation' section now
    satisfies the 'A final recommendation' deliverable (was a false MISSING)."""
    draft = (
        "## 9. Recommendation\n\n"
        "We recommend the phased rollout described above, grounded in the "
        f"primary sources surveyed. {_CITED}\n"
    )
    out = completeness_check(draft, ["A final recommendation"])
    assert out["satisfied"] == ["A final recommendation"]
    assert out["missing"] == []
    assert out["shortfall_score"] == 0.0


# ---------------------------------------------------------------------------
# CASE 2 — lead noun dropped; parenthetical sub-terms bind the heading.
# ---------------------------------------------------------------------------


def test_case2_canonical_heading_unchanged() -> None:
    """The fix is in the matcher, not canonicalisation: the canonical of the
    parenthetical bullet STILL strips to 'Risk register'."""
    assert (
        _canonical_heading("Risk register (likelihood / impact / mitigation)")
        == "Risk register"
    )


def test_case2_parenthetical_tokens_bind_reworded_heading() -> None:
    """'Risk register (likelihood / impact / mitigation)' binds a substantive
    '## Likelihood and Impact Mitigation Table' via parenthetical-token overlap
    even though the lead noun 'risk register' is absent from the heading."""
    spec = ["Risk register (likelihood / impact / mitigation)"]
    draft = (
        "## Likelihood and Impact Mitigation Table\n\n"
        "This section maps each risk to its likelihood, impact, and mitigation, "
        f"grounded in the surveyed sources. {_CITED}\n"
    )
    out = completeness_check(draft, spec)
    # satisfied carries the CANONICAL string, not the raw bullet.
    assert out["satisfied"] == ["Risk register"]
    assert out["missing"] == []
    # The canonical path alone does NOT match — the paren path is load-bearing.
    assert _heading_matches("Risk register", "Likelihood and Impact Mitigation Table") is False


# ---------------------------------------------------------------------------
# Threshold formula + trade-off — pinned as code; only the 0-overlap negative.
# ---------------------------------------------------------------------------


def test_overlap_floor_formula_pinned() -> None:
    """N==1 -> 1 (exact substring); N>=2 -> ceil(N/2)."""
    assert _overlap_floor(1) == 1
    assert _overlap_floor(2) == 1
    assert _overlap_floor(3) == 2
    assert _overlap_floor(4) == 2
    assert _overlap_floor(5) == 3


def test_zero_overlap_does_not_match() -> None:
    """The pinned negative: a 0-shared-token canonical never binds. (Single-
    content-token exclusivity is intentionally NOT pinned — it is in direct
    tension with the 2-token CASE-1 fix under any single overlap formula.)"""
    assert _heading_matches("Comparison matrix", "Executive Summary") is False


def test_single_token_deliverable_keeps_strict_substring() -> None:
    """N==1 is unchanged from pre-S83: the single token must be a substring of
    some heading token (so single-noun deliverables do not get fuzzier)."""
    assert _heading_matches("Recommendation", "6. Recommendations and Next Steps") is True
    assert _heading_matches("Glossary", "Executive Summary") is False


# ---------------------------------------------------------------------------
# Parenthetical-token plumbing — canonical strings + stable return shape.
# ---------------------------------------------------------------------------


def test_return_shape_and_canonical_entries_unchanged() -> None:
    """satisfied/missing still hold CANONICAL strings; the return dict keeps its
    compatible keys and entry types (outcome.py / protocol.py count consumers rely on
    this)."""
    spec = [
        "A final recommendation",  # satisfied (CASE 1)
        "Pattern catalogue",  # missing (no such heading)
    ]
    draft = (
        "## 9. Recommendation\n\n"
        f"A grounded recommendation section. {_CITED}\n"
    )
    out = completeness_check(draft, spec)
    assert {"satisfied", "missing", "shortfall_score"} <= set(out.keys())
    assert isinstance(out["shortfall_score"], float)
    assert all(isinstance(item, str) for item in out["satisfied"])
    assert all(isinstance(item, str) for item in out["missing"])
    assert out["satisfied"] == ["A final recommendation"]
    assert out["missing"] == ["Pattern catalogue"]
    assert out["shortfall_score"] == 0.5


# ---------------------------------------------------------------------------
# Parenthetical-noise negative — a generic-example paren cannot over-satisfy.
# ---------------------------------------------------------------------------


def test_parenthetical_noise_drops_single_char_tokens() -> None:
    """The 'e'/'g' of 'e.g.' are dropped so they cannot substring-match
    arbitrary heading tokens; only the meaningful names survive."""
    assert _parenthetical_tokens("Vendor list (e.g. Stripe, Square)") == ["stripe", "square"]


def test_parenthetical_noise_does_not_over_satisfy_unrelated_heading() -> None:
    """'Vendor list (e.g. Stripe, Square)' must NOT bind an unrelated reworded
    substantive heading via the paren-token path."""
    spec = ["Vendor list (e.g. Stripe, Square)"]
    draft = (
        "## Implementation Timeline\n\n"
        f"A substantive but unrelated section about scheduling. {_CITED}\n"
    )
    out = completeness_check(draft, spec)
    assert out["missing"] == ["Vendor list"]
    assert out["satisfied"] == []


# ---------------------------------------------------------------------------
# Substance AND is not dropped — reworded/paren match over a stub still misses.
# ---------------------------------------------------------------------------


def test_reworded_heading_over_stub_body_still_missing() -> None:
    """CASE-1 match path: a matching heading whose body is a placeholder stub
    (< MIN_SECTION_WORDS, no citation) STILL scores missing."""
    draft = "## 9. Recommendation\n\n[Data unavailable in primary sources] —\n"
    out = completeness_check(draft, ["A final recommendation"])
    assert out["missing"] == ["A final recommendation"]


def test_parenthetical_match_over_stub_body_still_missing() -> None:
    """CASE-2 paren path: a paren-token-matched heading over a stub body still
    scores missing (the paren path is AND-ed with substance too)."""
    draft = "## Likelihood and Impact Mitigation Table\n\n—\n"
    out = completeness_check(
        draft, ["Risk register (likelihood / impact / mitigation)"]
    )
    assert out["missing"] == ["Risk register"]


# ---------------------------------------------------------------------------
# RD2 structural guard.
# ---------------------------------------------------------------------------


def test_finalize_callsite_still_reads_final_markdown() -> None:
    """Defense-in-depth re-pin (duplicates test_s82_m04:228): the gate's
    deliverable_coverage is computed on the FINAL post-critique markdown."""
    src = inspect.getsource(fnode)
    assert re.search(r"completeness_check\(\s*final_markdown", src), (
        "deliverable_coverage must be recomputed on final_markdown"
    )


def test_substance_check_still_anded_in_non_quantity_branch() -> None:
    """The fuzzier/paren matcher must remain AND-ed with _section_is_substantive
    in the non-quantity branch — the over-counting fix from S82 m04 is preserved.
    """
    src = inspect.getsource(completeness_check)
    assert "_section_is_substantive(body)" in src, (
        "the substance check was dropped from the non-quantity branch"
    )
    # Both binding signals AND-ed against substance (the `and` is not severed):
    assert "_heading_matches(canonical" in src
    assert "_tokens_clear_floor(paren_tokens" in src
