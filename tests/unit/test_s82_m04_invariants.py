"""S82 m04 (PD135) RD2 invariants — truthful deliverable coverage.

Hardens ``completeness_check`` so the S82 m01 outcome gate reads a coverage
signal grounded in TRUTH, not substring artifacts. Three guarantees:

1. Heading-scoped matching — a deliverable binds to a real parsed Markdown
   heading, not to a prose echo / inline bold label / footer restatement.
2. Substance — a matched section must clear the non-placeholder word floor OR
   carry an inline citation signal; placeholder-dominated stubs score missing.
3. Entity-count — "at least N" deliverables are scored by the actual count of
   structural entity units (table rows / H3 subsections / numbered entries),
   immune to prose claiming the threshold was met.

Re-validated against the R002 / R003 stored-draft STRUCTURES (the agent-
verified shapes from the Concordance triage; no dispatch): R003 echoed
"At least 30 entities have been verified ... meeting the deliverable threshold"
in prose while writing up only 20 entity sections; R002 delivered 23 vendor
rows against a "≥30 vendors" deliverable. Both must score the quantity
deliverable MISSING.
"""

from __future__ import annotations

import re

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.tools.completeness_check import (
    MIN_SECTION_WORDS,
    _count_entity_units,
    _heading_matches,
    completeness_check,
)

_CITED = "See https://example.org/e for spec, license SPDX, scope and format details."


# ---------------------------------------------------------------------------
# Heading-scoped matching — prose echoes / inline labels do NOT satisfy.
# ---------------------------------------------------------------------------


def test_prose_echo_does_not_satisfy() -> None:
    spec = ["Comparison Matrix"]
    draft = (
        "## Executive Summary\n\n"
        "This report includes a comparison matrix of all vendors, "
        "https://example.org/x meeting the deliverable.\n"
    )
    out = completeness_check(draft, spec)
    assert out["missing"] == ["Comparison Matrix"], (
        "a keyword echoed only in prose (no matching heading) must not satisfy"
    )


def test_inline_bold_label_does_not_satisfy() -> None:
    # R002 shape: "Ambiguity notes" exists only as an inline **bold label**,
    # not a real heading. OLD substring matched it; NEW must not.
    spec = ["Ambiguity notes"]
    draft = (
        "## 3.1 Stripe\n\n"
        "**Ambiguity notes:** ownership is shared with the Linux Foundation. "
        f"{_CITED}\n"
    )
    out = completeness_check(draft, spec)
    assert out["missing"] == ["Ambiguity notes"]


def test_heading_wording_drift_still_matches() -> None:
    # S81 m01.b case must survive: canonical 'Pattern catalogue' binds to a
    # heading worded '5. Pattern / Practice Catalogue'.
    assert _heading_matches("Pattern catalogue", "5. Pattern / Practice Catalogue")
    assert _heading_matches("Recommendation", "6. Recommendations and Next Steps")
    assert not _heading_matches("Comparison matrix", "Executive Summary")


# ---------------------------------------------------------------------------
# Substance — placeholder-dominated stubs do not count; word-rich placeholder
# sections DO (R002 had 8 such vendor sections).
# ---------------------------------------------------------------------------


def test_stub_section_scores_missing() -> None:
    spec = ["Comparison Matrix"]
    draft = "## Comparison Matrix\n\n[Data unavailable in primary sources] —\n"
    out = completeness_check(draft, spec)
    assert out["missing"] == ["Comparison Matrix"]


def test_word_rich_section_with_embedded_placeholders_still_satisfies() -> None:
    # R002 vendor-section shape: substantial prose carrying a couple of
    # "[Data unavailable ...]" tokens is NOT a stub.
    spec = ["Stripe"]
    body = (
        "Stripe owns the @stripe/* npm family and maintains stripe-node. "
        "License SPDX is MIT. Last updated date observed recently. "
        "[Data unavailable in primary sources] for the alt-IDL notes, but the "
        "core ownership chain is well evidenced across the developer portal "
        "and the GitHub organisation with named maintainers listed."
    )
    draft = f"## 3.1 Stripe\n\n{body}\n"
    out = completeness_check(draft, spec)
    assert out["satisfied"] == ["Stripe"], (
        "a word-rich section must not false-fail on embedded placeholders"
    )


def test_short_but_cited_section_satisfies() -> None:
    # The citation-OR-words rule: a short section with a citation signal counts.
    spec = ["Linear"]
    draft = "## 3.12 Linear\n\nLinear publishes a spec. https://linear.app/developers\n"
    out = completeness_check(draft, spec)
    assert out["satisfied"] == ["Linear"]


def test_min_section_words_constant_pinned() -> None:
    assert MIN_SECTION_WORDS == 40


# ---------------------------------------------------------------------------
# Entity-count — "at least N" scored by structural units, not prose.
# ---------------------------------------------------------------------------


def _r003_like_draft(num_entities: int) -> str:
    """R003 structure: prose claims the threshold; only num_entities sections."""
    parts = [
        "# Research Report: R003",
        "## 1. Executive Summary",
        # The prose echo that fooled the OLD substring check:
        "At least 30 entities have been verified with concrete spec URLs, "
        "meeting the deliverable threshold for this mission.",
        "## 3. Per-Entity Analyses",
    ]
    for i in range(1, num_entities + 1):
        parts.append(f"### 3.{i} Entity{i}")
        parts.append(
            f"Entity{i} publishes an OpenAPI spec with license metadata. {_CITED}"
        )
    return "\n\n".join(parts)


_R003_SPEC = [
    "At least 30 of the v3 §4.5.1 ~40 candidates verified with concrete spec URL "
    "OR concrete confirmed-absence",
    "Each verified entry has: vendor / spec URL / license SPDX / scope / format",
    "Confirmed-absence entries note where lookups were attempted",
    "Output is a single .md report with one table or sectioned listing",
    "No spec downloads — deliverable is location + license metadata",
]


def test_r003_under_count_scores_quantity_deliverable_missing() -> None:
    """20 entity sections + a prose threshold-claim → 'At least 30' MISSING."""
    out = completeness_check(_r003_like_draft(20), _R003_SPEC)
    assert "At least 30" in out["missing"], (
        "the quantity deliverable must fail on actual count (20 < 30) even "
        "though the Executive Summary prose claims the threshold was met"
    )
    # R003 stays catastrophic for the m01 gate: shortfall well over HARD (0.60).
    assert out["shortfall_score"] >= 0.60


def test_r003_meeting_count_flips_quantity_deliverable_satisfied() -> None:
    """Counter-fixture: 30 real entity sections → 'At least 30' satisfied.

    Proves the verdict is driven by the structural count, not the prose echo
    (identical in both fixtures)."""
    out = completeness_check(_r003_like_draft(30), _R003_SPEC)
    assert "At least 30" in out["satisfied"]


def _r002_like_draft(num_vendors: int) -> str:
    """R002 structure: per-vendor H3 sections + a comparison-matrix table."""
    parts = ["# Research Report: R002", "## 1. Executive Summary", f"Covers vendors. {_CITED}"]
    for i in range(1, num_vendors + 1):
        parts.append(f"### 3.{i} Vendor{i}")
        parts.append(
            f"**Canonical npm packages:** @vendor{i}/core. Ownership evidenced. {_CITED}"
        )
    rows = "\n".join(f"| Vendor{i} | yes |" for i in range(1, num_vendors + 1))
    parts.append("## 4. Comparison Matrix")
    parts.append(f"| Vendor | Owns |\n|---|---|\n{rows}")
    return "\n\n".join(parts)


_R002_SPEC = [
    "Markdown report: vendor → SDK-family ownership map. Per-vendor sections "
    "covering the v3 §3/§7.3 canonical anchors (≥30 vendors)."
]


def test_r002_under_count_single_deliverable_missing() -> None:
    """23 vendors against a ≥30 deliverable → the single deliverable MISSING.

    Under m01 this single-deliverable shortfall=1.0 DEGRADES (not fails)."""
    draft = _r002_like_draft(23)
    out = completeness_check(draft, _R002_SPEC)
    assert out["missing"], "≥30-vendor deliverable must fail on 23 actual vendors"
    assert out["shortfall_score"] == 1.0


def test_entity_unit_count_uses_max_of_rows_and_subsections() -> None:
    # R003 (20 H3, no table) -> 20; R002 (23 rows, 23 H3) -> 23.
    assert _count_entity_units(_r003_like_draft(20)) == 20
    assert _count_entity_units(_r002_like_draft(23)) == 23


def test_entity_count_ignores_bulleted_detail_lists() -> None:
    # Bulleted sub-lists (per-entity details) must NOT inflate the entity count.
    draft = (
        "## Inventory\n\n"
        "- detail one\n- detail two\n- detail three\n- detail four\n"
    )
    assert _count_entity_units(draft) == 0


# ---------------------------------------------------------------------------
# Wiring — finalize computes coverage on the FINAL post-critique markdown.
# ---------------------------------------------------------------------------


def test_finalize_computes_coverage_on_final_markdown() -> None:
    """Requirement (d): the deliverable_coverage the gate reads is computed on
    final_markdown (post-critique), not synthesis_result.output_markdown."""
    import inspect

    src = inspect.getsource(fnode)
    assert re.search(r"completeness_check\(\s*final_markdown", src), (
        "deliverable_coverage must be recomputed on final_markdown (m04 d)"
    )
    assert not re.search(
        r"completeness_check\(\s*synthesis_result\.output_markdown", src
    ), "the pre-critique raw-output coverage call must be gone"
