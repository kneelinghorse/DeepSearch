"""Unit tests for ``deepsearch.agent.tools.completeness_check`` — S79 m02.a NEW tool.

Coverage matches the test plan in ``cmos/reports/sprint_79_approach_and_plan.md``
§5 m02.a: heading-presence detection on the draft markdown, defensive empty
shape on missing/empty spec, shortfall_score arithmetic, canonical-heading
extraction from deliverable bullets (drops trailing prepositional clauses).
"""

from __future__ import annotations

import pytest

from deepsearch.agent.tools.completeness_check import (
    _canonical_heading,
    completeness_check,
)


# ---------------------------------------------------------------------------
# Defensive empty shape — load-bearing for PD101 risk #6 (no false positives).
# ---------------------------------------------------------------------------


class TestDefensiveEmptyShape:
    def test_empty_deliverable_spec_returns_empty_result(self):
        out = completeness_check("# Report\n\nbody.")
        assert out["assessment_status"] == "unassessed"
        assert out["shortfall_score"] is None
        assert out["satisfied"] == out["missing"] == []

    def test_none_deliverable_spec_returns_empty_result(self):
        out = completeness_check("# Report", deliverable_spec=None)
        assert out["assessment_status"] == "unassessed"
        assert out["shortfall_score"] is None
        assert out["satisfied"] == out["missing"] == []

    def test_non_string_draft_returns_empty_result(self):
        out = completeness_check(12345, deliverable_spec=["X"])  # type: ignore[arg-type]
        assert out["assessment_status"] == "unassessed"
        assert out["shortfall_score"] is None
        assert out["satisfied"] == out["missing"] == []

    def test_spec_with_only_non_string_entries_returns_empty(self):
        out = completeness_check("body", deliverable_spec=[None, 42, {}])  # type: ignore[list-item]
        assert out["assessment_status"] == "unassessed"
        assert out["shortfall_score"] is None
        assert out["satisfied"] == out["missing"] == []


# ---------------------------------------------------------------------------
# Canonical heading extraction — strip list markers, emphasis, prep tails.
# ---------------------------------------------------------------------------


class TestCanonicalHeading:
    @pytest.mark.parametrize(
        "bullet,expected",
        [
            ("Comparison matrix", "Comparison matrix"),
            ("- Comparison matrix", "Comparison matrix"),
            ("1. Comparison matrix", "Comparison matrix"),
            ("**Comparison matrix**", "Comparison matrix"),
            (
                "Comparison matrix with rows for each entity",
                "Comparison matrix",
            ),
            (
                "Pattern catalog covering each named pattern",
                "Pattern catalog",
            ),
            (
                "Per-entity analysis covering pricing, scaling, and limits",
                "Per-entity analysis",
            ),
            ("Recommendation:", "Recommendation"),
        ],
    )
    def test_canonical_heading_extraction(self, bullet: str, expected: str):
        assert _canonical_heading(bullet) == expected


# ---------------------------------------------------------------------------
# Satisfied / missing classification + shortfall_score.
#
# S82 m04 (PD135): a deliverable is satisfied only when its canonical heading
# binds to a real parsed Markdown heading whose section is SUBSTANTIVE (clears
# the non-placeholder word floor OR carries an inline citation signal). These
# fixtures therefore give matched sections a citation signal (a URL) so they
# count — a bare ``body.`` is now (correctly) a stub.
# ---------------------------------------------------------------------------

_SUBSTANTIVE = "Substantive section grounded in evidence. See https://example.org/a for details."


class TestSatisfiedMissing:
    def test_all_present_zero_shortfall(self):
        spec = ["Executive Summary", "Per-Entity Analysis", "Comparison Matrix", "Recommendation"]
        draft = (
            "# Report\n\n"
            f"## Executive Summary\n{_SUBSTANTIVE}\n\n"
            f"## Per-Entity Analysis\n{_SUBSTANTIVE}\n\n"
            f"## Comparison Matrix\n{_SUBSTANTIVE}\n\n"
            f"## Recommendation\n{_SUBSTANTIVE}"
        )
        out = completeness_check(draft, deliverable_spec=spec)
        assert sorted(out["satisfied"]) == sorted(spec)
        assert out["missing"] == []
        assert out["shortfall_score"] == 0.0

    def test_partial_shortfall_renders_correct_score(self):
        spec = ["Executive Summary", "Comparison Matrix", "Recommendation", "Pattern Catalog"]
        draft = f"## Executive Summary\n{_SUBSTANTIVE}"  # 3 of 4 missing
        out = completeness_check(draft, deliverable_spec=spec)
        assert out["satisfied"] == ["Executive Summary"]
        assert sorted(out["missing"]) == sorted(["Comparison Matrix", "Recommendation", "Pattern Catalog"])
        assert out["shortfall_score"] == 0.75

    def test_full_shortfall_renders_one(self):
        # Realistic multi-word headings — single-letter spec items are
        # avoided because substring matching would trivially hit any prose.
        spec = ["Comparison Matrix", "Pattern Catalog", "Per-Entity Analysis"]
        draft = "Empty draft body with no headings."
        out = completeness_check(draft, deliverable_spec=spec)
        assert out["satisfied"] == []
        assert out["missing"] == ["Comparison Matrix", "Pattern Catalog", "Per-Entity Analysis"]
        assert out["shortfall_score"] == 1.0

    def test_case_insensitive_heading_match(self):
        # ``Comparison Matrix`` heading + ``COMPARISON MATRIX`` heading → match.
        spec = ["Comparison Matrix"]
        draft = f"## COMPARISON MATRIX\n\n{_SUBSTANTIVE}"
        out = completeness_check(draft, deliverable_spec=spec)
        assert out["satisfied"] == ["Comparison Matrix"]
        assert out["missing"] == []

    def test_prepositional_tail_stripped_before_match(self):
        # The bullet has a tail clause; only the leading noun phrase should
        # need to match a parsed heading.
        spec = ["Comparison matrix with rows for each entity"]
        draft = f"## Comparison Matrix\n\n{_SUBSTANTIVE}"
        out = completeness_check(draft, deliverable_spec=spec)
        assert "Comparison matrix" in out["satisfied"]
        assert out["missing"] == []

    def test_duplicate_bullets_collapsed(self):
        # Two bullets that reduce to the same heading don't double-count.
        spec = ["Comparison matrix", "Comparison matrix with rows"]
        draft = f"## Comparison Matrix\n{_SUBSTANTIVE}"
        out = completeness_check(draft, deliverable_spec=spec)
        assert out["satisfied"] == ["Comparison matrix"]
        assert out["missing"] == []
        assert out["shortfall_score"] == 0.0

    def test_stub_section_under_matched_heading_scores_missing(self):
        # S82 m04: a real heading whose body is a placeholder stub does NOT
        # count satisfied — the core over-counting fix.
        spec = ["Comparison Matrix"]
        draft = "## Comparison Matrix\n\n[Data unavailable in primary sources]"
        out = completeness_check(draft, deliverable_spec=spec)
        assert out["missing"] == ["Comparison Matrix"]
        assert out["shortfall_score"] == 1.0

    def test_prose_echo_without_heading_scores_missing(self):
        # S82 m04: the keyword appearing only in body prose (no matching
        # heading) must NOT satisfy — closes the prose-echo loophole.
        spec = ["Comparison Matrix"]
        draft = "## Summary\n\nWe present a comparison matrix below, see https://x.org/a."
        out = completeness_check(draft, deliverable_spec=spec)
        assert out["missing"] == ["Comparison Matrix"]


# ---------------------------------------------------------------------------
# S81 m01.b (PD116a) — _canonical_heading parens-handling fix.
#
# Before this fix the open paren in a parenthetical bullet orphaned the
# canonical heading, defeating substring match against the report's
# corresponding section heading. PLUGIN-BACKEND-AUTH-01 reported a false
# 5/6 shortfall against the post-S80 build because of this defect.
# ---------------------------------------------------------------------------


class TestParensHandling:
    @pytest.mark.parametrize(
        "bullet,expected",
        [
            # PLUGIN-BACKEND-AUTH-01 verbatim bullet that previously orphaned.
            (
                "Pattern catalogue (persistent-account / ephemeral-session / "
                "hybrid with optional account upgrade)",
                "Pattern catalogue",
            ),
            # Parens with trailing material after the close paren.
            (
                "OAuth flows (PKCE / device code / authorization code) "
                "comparison",
                "OAuth flows comparison",
            ),
            # Parens at the very end of the bullet.
            ("Comparison matrix (per-vendor)", "Comparison matrix"),
            # Parens followed by a prepositional clause that should still
            # be truncated post-paren-strip.
            (
                "Pattern catalogue (a / b / c) with rows for each entity",
                "Pattern catalogue",
            ),
            # Multiple disjoint parentheticals.
            (
                "Risk register (top 9) covering (a) cost (b) coverage",
                "Risk register",
            ),
            # Markdown emphasis + parens.
            (
                "**Pattern catalogue (persistent-account / ephemeral-session "
                "/ hybrid)**",
                "Pattern catalogue",
            ),
        ],
    )
    def test_parens_bullets_canonicalised(self, bullet: str, expected: str) -> None:
        assert _canonical_heading(bullet) == expected

    def test_plain_bullets_still_work(self) -> None:
        """Regression: S79-coverage canonical-heading cases keep passing."""
        assert _canonical_heading("Comparison matrix") == "Comparison matrix"
        assert (
            _canonical_heading(
                "Comparison matrix with rows for each entity"
            )
            == "Comparison matrix"
        )
        assert _canonical_heading("- Comparison matrix") == "Comparison matrix"
        assert _canonical_heading("**Comparison matrix**") == "Comparison matrix"
        assert _canonical_heading("Recommendation:") == "Recommendation"

    def test_parens_strip_does_not_double_whitespace(self) -> None:
        """Whitespace collapse must produce no doubled spaces."""
        out = _canonical_heading("Pattern catalogue (notes) with rows")
        assert "  " not in out

    def test_integration_plugin_backend_auth_01_parenthetical_satisfies(self) -> None:
        """Replay PLUGIN-BACKEND-AUTH-01 parenthetical bullet + report heading.

        Before m01.b: the open-paren orphan caused completeness_check to
        report this bullet as missing. After m01.b: the canonical heading
        is ``Pattern catalogue`` and substring matches the H2.
        """
        spec = [
            "Pattern catalogue (persistent-account / ephemeral-session / "
            "hybrid with optional account upgrade)"
        ]
        draft = (
            "## 5. Pattern / Practice Catalogue\n\n"
            "Pattern catalogue covering each named pattern, grounded in vendor "
            "documentation. See https://example.org/patterns for the catalogue.\n"
        )
        out = completeness_check(draft, deliverable_spec=spec)
        assert out["satisfied"] == ["Pattern catalogue"]
        assert out["missing"] == []
        assert out["shortfall_score"] == 0.0
