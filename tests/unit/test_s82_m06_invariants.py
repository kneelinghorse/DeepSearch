"""S82 m06 (PD135) RD2 invariants — prompt-incentive + early-stop discipline.

Four contract_to_prompt edits that reduce the model's fabrication propensity and
premature-stop behaviour, aligned with the enforcement S82 m01/m02/m05 landed:
  (a) the free-gamble fabrication framing is reframed — fabrication is no longer
      "replaced with a gap note anyway" (it now costs delivery quality);
  (b) the rigid ">=25 distinct URLs" floor is softened to "no minimum quota";
  (c) the out-of-scope/stub license is gated behind genuine retrieval attempts;
  (d) a ~75%-step-budget gap-note clause (reconciled into the m03 STEP BUDGET
      section).

The MANDATORY anti-fabrication + CITATION MARKERS + PRE-ACCEPTANCE COMPLETENESS
CHECK guardrails must be PRESERVED, and the retrieval-tool contracts + Output
Format section structure unchanged.
"""

from __future__ import annotations

from deepsearch.agent.contract_to_prompt import build_codeagent_prompts


def _system() -> str:
    system, _ = build_codeagent_prompts(
        {
            "mission_id": "M",
            "mission_context": {
                "title": "T",
                "objective": "O",
                "deliverables": ["Comparison Matrix"],
                "required_entities": ["X", "Y"],
            },
        }
    )
    return system


# ---------------------------------------------------------------------------
# (a) Free-gamble fabrication framing removed / reframed.
# ---------------------------------------------------------------------------


def test_free_gamble_framing_is_gone() -> None:
    sys = _system()
    assert "fabrication will be replaced with a gap note anyway" not in sys, (
        "the asymmetric free-gamble incentive must be removed"
    )
    # Reframed: fabrication now has a real cost (gate + quarantine).
    assert "is NOT free" in sys
    assert "delivery quality" in sys.lower()


# ---------------------------------------------------------------------------
# (b) Rigid >=25-URL floor softened.
# ---------------------------------------------------------------------------


def test_rigid_url_floor_softened() -> None:
    sys = _system()
    assert "at least 25 distinct URLs" not in sys, "the rigid floor must be gone"
    assert "no minimum url quota" in sys.lower()
    assert "12-URL report" in sys  # the honest-over-padded framing


# ---------------------------------------------------------------------------
# (c) Out-of-scope license gated behind genuine retrieval attempts.
# ---------------------------------------------------------------------------


def test_out_of_scope_requires_attempts_first() -> None:
    sys = _system()
    assert "do NOT burn retrieval steps trying to find unfindable content" not in sys, (
        "the pre-emptive stub license must be reframed"
    )
    assert "ONLY AFTER" in sys
    assert "retrieval attempts" in sys
    assert "do NOT declare out-of-scope pre-emptively" in sys


# ---------------------------------------------------------------------------
# (d) ~75%-step-budget gap-note clause (reconciled with m03).
# ---------------------------------------------------------------------------


def test_step_budget_gap_note_clause_present() -> None:
    sys = _system()
    assert "75% of your step budget" in sys
    # Still tied to the m03 finalize-before-exhaustion discipline + report_draft.
    assert "report_draft" in sys
    assert "STEP BUDGET & STRANDED-DRAFT SAFETY" in sys


# ---------------------------------------------------------------------------
# Guardrails preserved — m06 must not weaken anti-fabrication.
# ---------------------------------------------------------------------------


def test_anti_fabrication_guardrails_preserved() -> None:
    sys = _system()
    # MANDATORY anti-fabrication.
    assert "NEVER fabricate sources or URLs" in sys
    assert "CITATION MARKERS — MANDATORY" in sys
    assert "MUST be followed immediately by a `[cite:URL]` marker" in sys
    # PRE-ACCEPTANCE COMPLETENESS CHECK preserved.
    assert "PRE-ACCEPTANCE COMPLETENESS CHECK — MANDATORY" in sys
    assert "completeness_check(draft_markdown=<your draft>)" in sys


def test_retrieval_contracts_and_output_format_unchanged() -> None:
    sys = _system()
    # Retrieval-tool contracts intact.
    tools = (
        "web_search(",
        "source_fetch(",
        "url_liveness(",
        "scratchpad_write(",
        "completeness_check(",
    )
    for tool in tools:
        assert tool in sys, f"retrieval tool contract {tool} must be intact"
    # Output Format section structure intact.
    assert "# Output Format" in sys
    assert "## References" in sys
    assert "Executive Summary" in sys
