"""completeness_check subsection-rollup fix — DSV-045 false-MISSING class.

Live production class (TraceLab mission DSV-045, 2026-06-15): a deliverable
heading whose prose lives ENTIRELY under ``###`` children owned a 0-word direct
body, so ``_section_is_substantive`` false-failed it and the deliverable scored
MISSING. With 4 such deliverables the gate computed shortfall 1.000 (0/4) →
``deliverable_shortfall_exceeded`` → a spurious catastrophic ``validation_failed``
on a genuinely complete 3.4k-word report. Confirmed by re-running the current
matcher on the real report (reproduced 0/4 byte-identically).

The fix rolls a section's descendant subsection PROSE into its body for the
substance check, WITHOUT weakening the S82/S83 m04 truthfulness gains: the
heading match is still AND-ed with substance, and an all-placeholder or truly
empty section still scores MISSING (only descendant prose rolls up, never the
descendant heading labels, and the placeholder floor is unchanged). Only
strictly-deeper descendants roll up — an equal-or-higher sibling does not lend
its substance.
"""

from __future__ import annotations

from deepsearch.agent.tools.completeness_check import completeness_check

_CITED = "See https://example.org/x for the grounding evidence and details here."

# >= MIN_SECTION_WORDS (40) of real prose, for cases that need a genuinely
# substantive body/descendant.
_P40 = (
    "We recommend a three-tier pricing model with a free tier capped at one "
    "thousand monthly active users, a Pro tier at forty-nine dollars per seat "
    "each month for mid-market teams, and an Enterprise tier with custom "
    "pricing gated on single sign-on, audit logs, and a dedicated customer "
    "success manager across the entire account base described here in detail."
)


def _two_level_section() -> str:
    """A ``## deliverable`` whose prose lives only under ``###`` children — the
    DSV-045 shape (parent heading immediately followed by a child heading)."""
    return (
        "## 6. Achievable-Today vs Research-Frontier Split\n"
        "\n"
        "### Achievable Today (Production-Viable or Near-Production)\n"
        "Vega-Lite and ECharts are production-ready grammar-of-graphics "
        "renderers already used widely across deployed BI surfaces today. "
        f"{_CITED}\n"
        "\n"
        "### Research Frontier (2025-2026 and Beyond)\n"
        "Agentic multi-turn BI exploration remains experimental, with early "
        f"systems only beginning to close the analysis loop. {_CITED}\n"
    )


def test_nested_subsection_prose_satisfies_parent_deliverable() -> None:
    """The DSV-045 repro: a deliverable heading whose substance lives under its
    ``###`` children now SATISFIES (pre-fix it was a false MISSING because the
    parent's own body was empty)."""
    out = completeness_check(
        _two_level_section(),
        ["Achievable-today vs research-frontier split"],
    )
    assert out["satisfied"] == ["Achievable-today vs research-frontier split"]
    assert out["missing"] == []
    assert out["shortfall_score"] == 0.0


def test_all_four_dsv045_shape_deliverables_satisfied() -> None:
    """Four deliverables, each a ``##`` heading with prose only under ``###``
    children — all satisfied, no shortfall. Pre-fix this was the production 0/4
    (shortfall 1.000) catastrophic failure."""
    # Headings are the real DSV-045 section headings (verbatim shape); all prose
    # lives under the ``###`` children, exactly as the failed report did. Each
    # child carries >= MIN_SECTION_WORDS of real prose because the rollup path
    # is judged by the placeholder-stripped WORD floor (a borrowed citation
    # alone does not count) — mirroring the real report's word-rich subsections.
    p = (
        "Voyager, Draco, and LIDA recommend visualization encodings directly "
        "from the data profile and from perceptual-effectiveness constraints "
        "rather than from hand-authored rules, and they have been benchmarked "
        "against supervised baselines across several published studies that "
        "report consistent improvements on standard datasets and tasks today. "
    )
    draft = (
        "## 3. SOTA Report: The Generative/Agentic Data-Visualization Frontier\n\n"
        f"### 3.1 Automatic Visualization Recommendation\n{p}{_CITED}\n\n"
        "## 5. Technique Inventory: Each Technique to Maturity to How It Plugs into a "
        "JSON Grammar-of-Graphics Normalized Spec\n\n"
        f"### 5.1 Constraint-Based Recommendation\n{p}{_CITED}\n\n"
        "## 6. Achievable-Today vs Research-Frontier Split\n\n"
        f"### Achievable Today\n{p}{_CITED}\n\n"
        "## 7. Concrete Recommendations + Named Differentiators for an "
        "Agent-Native Deterministic Viz Engine\n\n"
        f"### Primary Architecture\n{p}{_CITED}\n"
    )
    spec = [
        "SOTA report (markdown) on the generative/agentic data-viz frontier",
        "Technique inventory: each technique maturity",
        "Achievable-today vs research-frontier split",
        "Concrete recommendations + named differentiators",
    ]
    out = completeness_check(draft, spec)
    assert out["missing"] == []
    assert out["shortfall_score"] == 0.0


# ---------------------------------------------------------------------------
# Truthfulness preserved — the rollup must NOT manufacture substance.
# ---------------------------------------------------------------------------


def test_all_placeholder_subsections_still_missing() -> None:
    """A deliverable heading whose every ``###`` child is a placeholder stub
    still scores MISSING: the rollup carries descendant prose but the
    placeholder floor strips it, so no substance is manufactured."""
    draft = (
        "## 6. Achievable-Today vs Research-Frontier Split\n\n"
        "### Achievable Today\n"
        "[Data unavailable in primary sources] —\n\n"
        "### Research Frontier\n"
        "[Data unavailable in primary sources] —\n"
    )
    out = completeness_check(
        draft, ["Achievable-today vs research-frontier split"]
    )
    assert out["missing"] == ["Achievable-today vs research-frontier split"]
    assert out["satisfied"] == []


def test_rollup_stops_at_equal_level_sibling() -> None:
    """Only strictly-deeper descendants roll up: a stub ``##`` followed by a
    SUBSTANTIVE sibling ``##`` (equal level) does not borrow the sibling's
    substance — the stub deliverable stays MISSING."""
    draft = (
        "## 6. Achievable-Today vs Research-Frontier Split\n"
        "—\n"
        "## 7. Concrete Recommendations\n"
        "A fully substantive and grounded recommendations section that easily "
        f"clears the non-placeholder word floor with real cited content. {_CITED}\n"
    )
    out = completeness_check(
        draft, ["Achievable-today vs research-frontier split"]
    )
    assert out["missing"] == ["Achievable-today vs research-frontier split"]


# ---------------------------------------------------------------------------
# Over-count: the rollup is deliberately RECALL-BIASED. The ONE guarded
# resurrection case is an explicit gap-note OWN body. The remaining over-counts
# are ACCEPTED (a coverage shortfall degrades, not fails — see outcome.py) and
# pinned as documented known limitations so the trade-off stays visible.
# ---------------------------------------------------------------------------


def test_gap_noted_own_body_is_not_rescued_by_descendants() -> None:
    """The guarded resurrection case: a deliverable whose OWN body is the
    canonical gap-note ('[Data unavailable ...]') declares its content absent and
    is NOT rescued by a descendant (here a References list with a URL) → MISSING."""
    draft = (
        "## 4. Regulatory Landscape Analysis\n\n"
        "[Data unavailable in primary sources]\n\n"
        "### References\n"
        "- https://regulator.example.gov/filing-index\n"
    )
    out = completeness_check(draft, ["Regulatory landscape analysis"])
    assert out["missing"] == ["Regulatory landscape analysis"]


def test_known_overcount_limitations_are_accepted() -> None:
    """DOCUMENTED known over-counts, ACCEPTED under recall-bias + degrade-not-fail
    (see the completeness_check module docstring and outcome.py). These pin the
    deliberate trade-off — if one flips, that is a behaviour change to review, not
    a silent regression. The rollup never false-MISSES (the costly error); it may
    over-count, which only softens a now-degrading (not failing) gate.

    (a) misnested PEER deliverable: a stub parent borrows a *different*
        deliverable nested under it.
    (b) citation borrow: an empty-bodied parent whose placeholder children carry
        a stray URL (contrast test_all_placeholder_subsections_still_missing,
        which has no URL and correctly stays MISSING)."""
    # (a) GTM stub borrows the Pricing deliverable misnested under it.
    misnested = (
        "# Market Entry Analysis\n\n"
        "## Go-To-Market Strategy\n\n"
        "*(Content pending further research.)*\n\n"
        "### Pricing and Packaging Recommendations\n"
        f"{_P40}\n"
    )
    out = completeness_check(
        misnested, ["Go-to-market strategy", "Pricing and packaging recommendations"]
    )
    assert out["satisfied"] == [
        "Go-to-market strategy",
        "Pricing and packaging recommendations",
    ]

    # (b) empty parent + placeholder children, one carrying a URL.
    cited_placeholders = (
        "## 6. Achievable-Today vs Research-Frontier Split\n\n"
        "### Achievable Today\n"
        "[Data unavailable in primary sources] — see https://example.org/y\n\n"
        "### Research Frontier\n"
        "[Data unavailable in primary sources] —\n"
    )
    out = completeness_check(
        cited_placeholders, ["Achievable-today vs research-frontier split"]
    )
    assert out["satisfied"] == ["Achievable-today vs research-frontier split"]


# ---------------------------------------------------------------------------
# Fenced-code-block awareness — a '#' inside ``` is not a heading.
# ---------------------------------------------------------------------------


def test_code_fence_hash_does_not_sever_rollup() -> None:
    """A ``# comment`` inside a fenced code block must NOT be parsed as a
    shallow heading that severs the descendant rollup — the fence-unaware parser
    false-MISSED this on technical reports that embed code examples."""
    draft = (
        "## Pricing Analysis\n\n"
        f"### Stripe\n{_P40}\n\n"
        "```python\n"
        "# Initialize the Stripe client with your secret key\n"
        "# See https://stripe.com/docs for the full reference\n"
        "import stripe\n"
        "```\n\n"
        f"### PayPal\n{_P40}\n"
    )
    out = completeness_check(draft, ["Pricing Analysis"])
    assert out["satisfied"] == ["Pricing Analysis"]


def test_code_fence_text_does_not_fake_a_section() -> None:
    """A deliverable heading that exists ONLY inside a fenced code block is not a
    real section; the deliverable scores MISSING (no false-SATISFIED from code
    text clearing the word floor)."""
    draft = (
        "## Intro\n\nPlaceholder intro line.\n\n"
        "```text\n"
        "# Vendor Comparison\n"
        f"{_P40}\n"
        "```\n"
    )
    out = completeness_check(draft, ["Vendor Comparison"])
    assert out["missing"] == ["Vendor Comparison"]


def test_fence_run_length_inner_backticks_are_literal() -> None:
    """A 3-backtick run INSIDE a 4-backtick fence is literal content, not a close
    (CommonMark run-length rule). The inner ``# Not A Real Heading`` stays inert
    and the real ``## Conclusion`` after the 4-backtick close is still seen."""
    draft = (
        f"## Configuration Reference\n\n{_P40}\n\n"
        "````markdown\n"
        "# Not A Real Heading\n"
        "```\n"
        "nested code sample\n"
        "```\n"
        "more documentation text inside the outer block\n"
        "````\n\n"
        f"## Conclusion\n{_P40}\n"
    )
    out = completeness_check(draft, ["Configuration Reference", "Conclusion"])
    assert out["satisfied"] == ["Configuration Reference", "Conclusion"]
    assert out["missing"] == []
