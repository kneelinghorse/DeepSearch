"""Unit tests for `deepsearch.agent.contract_to_prompt` adapter (S64).

Verifies the adapter produces complete CodeAgent prompts for the
OODS-FIGMA-HOST-01 fixture and degrades gracefully when contract fields
are missing. Post-S64 the adapter no longer takes a ``pool_size`` argument —
the CodeAgent retrieves live via its tool surface.
"""

from __future__ import annotations


import pytest

from deepsearch.agent.contract_to_prompt import (
    build_codeagent_prompts,
    resolve_deliverables,
)
















def test_minimal_state_uses_objective_as_fallback_objective_list():
    state = {
        "mission_id": "MIN-01",
        "mission_objectives": [],
        "mission_context": {
            "title": "Minimal Mission",
            "objective": "compare A vs B",
        },
    }
    system, user = build_codeagent_prompts(state)
    assert "Minimal Mission" in system
    obj_block = system.split("# Research Objectives", 1)[1].split("# Required Deliverables", 1)[0]
    assert "1. compare A vs B" in obj_block
    assert "MIN-01" in user


def test_handles_camelcase_keys():
    state = {
        "mission_id": "CC-01",
        "mission_objectives": ["o1", "o2"],
        "mission_context": {
            "title": "CamelCase Mission",
            "objective": "x vs y",
            "background": "context here",
            "requiredEntities": ["Alpha", "Beta"],
            "excludedEntities": ["Gamma"],
            "deliverables": ["Comparison Matrix"],
        },
    }
    system, _ = build_codeagent_prompts(state)
    assert "- Alpha" in system
    assert "- Beta" in system
    assert "- Gamma" in system  # in the excluded block


def test_background_falls_back_when_missing():
    state = {
        "mission_id": "NB-01",
        "mission_objectives": ["o1"],
        "mission_context": {
            "title": "NoBackground Mission",
            "objective": "compare X",
        },
    }
    system, _ = build_codeagent_prompts(state)
    bg_block = system.split("**Background:**", 1)[1].split("# Required Entities", 1)[0]
    assert "(no background provided)" in bg_block


def test_handles_no_required_entities():
    state = {
        "mission_id": "NR-01",
        "mission_objectives": ["o1"],
        "mission_context": {
            "title": "No-required Mission",
            "objective": "explore X",
        },
    }
    system, _ = build_codeagent_prompts(state)
    req_block = system.split("# Required Entities", 1)[1].split("# Out of scope", 1)[0]
    assert "(none)" in req_block


def test_no_pool_size_kwarg_accepted():
    """Regression: build_codeagent_prompts no longer takes a pool_size kwarg."""

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    with pytest.raises(TypeError):
        build_codeagent_prompts(state, pool_size=10)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# S66 — prompt template content audit (mandatory checklist subsection).
# Third consecutive sprint applying the rule (S64 m03 → S65 m02 → S66 m02).
# ---------------------------------------------------------------------------


def test_template_advertises_critique_pass():
    """The system prompt must advertise the post-draft critique pass.

    Per S66 plan-doc §"Prompt template content audit": adding the critique
    surface changes the post-draft pipeline shape, and the agent's prompt
    must reflect that so it can adapt its drafting strategy (cite or
    self-note a gap rather than fabricate). Three load-bearing phrases are
    asserted: that the critique runs as a single-shot pass, that fabrication
    will be replaced with a gap note, and that the critique does NOT trigger
    re-retrieval.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    assert "critique pass" in system
    assert "[Data unavailable in primary sources]" in system
    # The agent must be told the critique does not trigger re-retrieval and
    # is not part of its ReAct loop — this prevents the agent from over-
    # iterating defensively in anticipation of the critique.
    assert "does not trigger re-retrieval" in system
    assert "ReAct loop" in system


def test_template_critique_section_lands_in_correct_position():
    """Critique advertisement must follow the tool block and precede Output Format.

    The agent should see tools first (its loop), then the critique pass
    (post-draft), then the output format. Reversing the ordering would
    suggest the critique is part of the agent's loop — wrong, and would
    push the agent toward defensive re-retrieval.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    tool_block_idx = system.find("# Tools Available")
    critique_block_idx = system.find("# Post-Draft Critique Pass")
    output_format_idx = system.find("# Output Format")
    assert tool_block_idx != -1
    assert critique_block_idx != -1
    assert output_format_idx != -1
    assert tool_block_idx < critique_block_idx < output_format_idx


# ---------------------------------------------------------------------------
# S67 — prompt template content audit.
# Fourth consecutive sprint applying the rule (S64 m03 → S65 m02 → S66 m02 →
# S67 m02). Per the S66 retro learning, the audit subsection has been promoted
# from process learning to load-bearing unit-test assertion in S67. The S67
# layer adds the [cite:URL] marker convention.
# ---------------------------------------------------------------------------


def test_template_advertises_citation_markers():
    """The system prompt must advertise the inline `[cite:URL]` marker convention.

    The agent has to know that source-backed claims need an inline marker
    immediately after them, that the URL must come from its retrieval payload
    (no fabrication), and that the markers will not appear in the final
    report. Three load-bearing phrases are asserted.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    assert "[cite:URL]" in system
    # The marker must come from the retrieval payload, not parametric memory —
    # this is the structural anti-fabrication rule.
    assert "URL you did not retrieve" in system or "do not retrieve" in system.lower()
    # Multi-source chaining: the agent should know to write [cite:U1][cite:U2].
    assert "[cite:URL1][cite:URL2]" in system
    # The reader sees clean prose; markers are stripped before shipping.
    assert "stripped" in system.lower() or "extractor" in system.lower()


def test_template_citation_section_lands_in_correct_position():
    """Citation-marker block must follow Tools, precede Critique, precede Output Format.

    Ordering protects the agent from inferring that the marker convention is
    a critique-pass artefact. The agent must emit markers DURING drafting,
    not as a separate critique-time step.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    tool_block_idx = system.find("# Tools Available")
    citation_block_idx = system.find("# CITATION MARKERS — MANDATORY")
    critique_block_idx = system.find("# Post-Draft Critique Pass")
    output_format_idx = system.find("# Output Format")
    assert tool_block_idx != -1
    assert citation_block_idx != -1
    assert critique_block_idx != -1
    assert output_format_idx != -1
    assert (
        tool_block_idx
        < citation_block_idx
        < critique_block_idx
        < output_format_idx
    ), (
        "Section ordering must be: Tools → Citation Markers → Critique Pass → "
        "Output Format. The marker convention belongs to drafting, not "
        "critique."
    )


def test_template_advertises_liveness_check():
    """The system prompt must advertise the S68 downstream liveness check.

    Three load-bearing canonical phrases are asserted: (a) every marker
    will be checked against the retrieval payload (orphan check) AND via
    HTTP probe (liveness check); (b) FABRICATED markers will be removed +
    their claims replaced with a gap note, while dead-but-collected markers
    are kept flagged unverified (S84 m02 — the prompt must not threaten
    deletion the pipeline no longer performs, and must not promise
    fabrication is survivable); (c) "no marker is better than a fabricated
    marker" framing — emit a gap note instead. If a future prompt edit
    drops or softens these, the agent loses the structural incentive to
    NOT fabricate citations.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    # (a) downstream liveness + orphan check called out together.
    assert "checked downstream against your retrieval payload (orphan check)" in system
    assert "HTTP probe (liveness check)" in system
    # (b) fabricated markers removed + gap-note replacement; dead-but-collected
    # markers survive flagged (S84 m02 downgrade-not-delete truth).
    assert "Fabricated markers" in system
    assert "will be removed" in system
    assert "replaced with a gap note" in system
    assert "kept and flagged unverified" in system
    assert "Orphan and dead markers will be removed" not in system, (
        "stale pre-S84 deletion promise — the pipeline no longer deletes dead markers"
    )
    # (c) gap note > fabricated marker.
    assert "worse than no marker" in system


def test_template_liveness_amendment_lands_inside_citation_section():
    """The S68 amendment lives inside the existing # CITATION MARKERS — MANDATORY section.

    Position invariant: the new paragraph appears AFTER the existing S67
    canonical phrasing AND BEFORE the # Post-Draft Critique Pass heading
    that terminates the citation section. A future prompt edit that
    accidentally moves the amendment into a different section (or hoists
    it above the marker phrasing) breaks the load-bearing reading order
    the agent relies on.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    citation_section_start = system.index("# CITATION MARKERS — MANDATORY")
    critique_section_start = system.index("# Post-Draft Critique Pass")
    # Search for the `[cite:URL]` canonical phrasing inside the citation
    # section, not the first occurrence in the document — S78 m01.d added
    # an `# Objective Classification` clause that legitimately mentions
    # `[cite:URL]` earlier (referring back to the marker convention).
    s67_canonical_idx = system.index("[cite:URL]", citation_section_start)
    liveness_amendment_idx = system.index(
        "checked downstream against your retrieval payload"
    )

    assert citation_section_start < s67_canonical_idx, (
        "S67 marker phrasing must live inside the # CITATION MARKERS — MANDATORY section"
    )
    assert s67_canonical_idx < liveness_amendment_idx, (
        "S68 liveness amendment must follow the existing S67 marker "
        "phrasing — not precede it"
    )
    assert liveness_amendment_idx < critique_section_start, (
        "S68 liveness amendment must land inside the # CITATION MARKERS — MANDATORY "
        "section, before the # Post-Draft Critique Pass heading"
    )


def test_template_citation_block_warns_against_code_block_markers():
    """The agent must know that markers inside code blocks are NOT extracted.

    If the agent drops a `[cite:URL]` inside a fenced or inline code span,
    the extractor leaves it as literal text — the claim won't be grounded.
    The prompt has to call this out explicitly so the agent doesn't quote
    snippets containing markers.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    # The phrasing varies; assert the substantive content.
    citation_section = system.split("# CITATION MARKERS — MANDATORY", 1)[1].split(
        "# Post-Draft Critique Pass", 1
    )[0]
    assert "code block" in citation_section.lower() or "backtick" in citation_section.lower()


# ---------------------------------------------------------------------------
# S74 m01 PD32 Path 1 — promoted heading + explicit failure language +
# worked example. The S73-WIRING-02 dispatch produced a numbered References
# list with zero `[cite:URL]` markers. The prompt rule existed but was
# treated as low-priority; S74 lifts the heading to MANDATORY, names the
# failure mode, and shows the dual-marker pattern explicitly.
# ---------------------------------------------------------------------------


def test_template_uses_mandatory_uppercase_heading():
    """The S74 heading promotion is the load-bearing prominence change.

    `# CITATION MARKERS — MANDATORY` (uppercase) replaces the lowercase
    `# Citation Markers — Required Inline` heading. Heading prominence is
    the lever the agent reads first.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    assert "# CITATION MARKERS — MANDATORY" in system, (
        "S74 m01 promoted the heading to uppercase MANDATORY form. Drift "
        "here regresses the prompt-prominence fix and re-opens the "
        "S73-WIRING-02 References:0 failure mode."
    )
    # Defensive: the legacy lowercase form should NOT coexist (we replaced,
    # not duplicated).
    assert "# Citation Markers — Required Inline" not in system


def test_template_calls_out_failure_mode_explicitly():
    """The prompt names the consequence of skipping `[cite:URL]` markers.

    Two load-bearing phrases: (a) reports without markers are treated as
    ungrounded; (b) the critique stage rejects them. The S73-WIRING-02
    failure mode (numbered refs only, zero `[cite:URL]`) is exactly what
    this language warns against.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    assert "ungrounded" in system, (
        "Failure language must name the 'ungrounded' framing — the agent "
        "needs to associate skipping markers with a structural rejection, "
        "not a stylistic preference."
    )
    assert "rejected by the critique stage" in system, (
        "Failure language must name the rejection mechanism (critique "
        "stage). Without it the agent can read the rule as advisory."
    )


def test_template_carries_worked_example_with_dual_markers():
    """Positive example: claim + `[cite:URL]` + numbered `[1]` + References entry.

    The worked example is the load-bearing teaching moment: it shows the
    agent that BOTH numbered references AND inline `[cite:URL]` markers
    coexist on the same claim. The S73-WIRING-02 failure mode emerged
    because the agent emitted the human-readable surface (numbered refs)
    and skipped the machine-readable surface (`[cite:URL]`); the worked
    example removes that ambiguity.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    # The example must show a claim with BOTH a numbered reference
    # `[1]` and an inline `[cite:URL]` marker on the same line.
    citation_section = system.split("# CITATION MARKERS — MANDATORY", 1)[1].split(
        "# Post-Draft Critique Pass", 1
    )[0]
    assert "[1]" in citation_section, (
        "Worked example must include numbered reference syntax `[1]` so "
        "the agent sees the dual-marker pattern."
    )
    assert "[cite:https://" in citation_section, (
        "Worked example must include a concrete `[cite:URL]` marker so "
        "the agent sees the marker syntax in context."
    )
    assert "## References" in citation_section, (
        "Worked example must include a `## References` section so the "
        "agent sees how numbered refs and `[cite:URL]` markers coexist "
        "with the references list."
    )


def test_template_explicitly_calls_numbered_refs_independent_of_cite_markers():
    """The S73-WIRING-02 failure mode is named directly.

    The prompt must state that emitting numbered references alone (without
    `[cite:URL]` markers) is the failure mode — not a permissible
    abbreviation.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    citation_section = system.split("# CITATION MARKERS — MANDATORY", 1)[1].split(
        "# Post-Draft Critique Pass", 1
    )[0]
    # The independence rule must be stated explicitly so the agent cannot
    # infer that numbered refs replace `[cite:URL]` markers.
    assert "INDEPENDENT" in citation_section.upper() or "independent" in citation_section, (
        "Citation section must call out that numbered references are "
        "independent of `[cite:URL]` markers — both are required."
    )


# ---------------------------------------------------------------------------
# S78 m01.c (PD81) — `[cite:URL]` prompt strengthening.
# After S74 promoted the heading to MANDATORY, OODS-PLUGIN-REVIEW-01 still
# emitted zero markers under broad-scope/multi-objective pressure. m01.c
# adds two reinforcements: (1) the "marked ungrounded by automated review"
# framing per PD81; (2) an explicit "rule does not relax under pressure"
# clause + self-check-before-final_answer guidance.
# ---------------------------------------------------------------------------


def test_template_carries_automated_review_failure_framing():
    """The PD81 'marked ungrounded by automated review' phrase must appear.

    S74 already named "ungrounded" + "rejected by the critique stage". S78
    m01.c adds the PD81 framing that the verdict comes from automated
    review — making the consequence procedural rather than discretionary.
    Future prompt edits must keep this phrasing intact; softening it
    re-opens the OODS-PLUGIN-REVIEW-01 adherence-failure mode.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    citation_section = system.split("# CITATION MARKERS — MANDATORY", 1)[1].split(
        "# Post-Draft Critique Pass", 1
    )[0]
    assert "marked ungrounded by automated review" in citation_section, (
        "S78 m01.c language must use the PD81 'marked ungrounded by "
        "automated review' phrase — agents read procedural language as "
        "load-bearing in a way they don't read advisory language."
    )


def test_template_calls_out_rule_does_not_relax_under_pressure():
    """OODS-PLUGIN-REVIEW-01 emitted zero markers across a broad-scope mission.

    PD81 / m01.c counter the failure mode with an explicit clause that the
    citation-marker rule does not relax for long reports or broad topic
    coverage, plus a self-check pass before `final_answer`. Future prompt
    edits must preserve both cues.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    citation_section = system.split("# CITATION MARKERS — MANDATORY", 1)[1].split(
        "# Post-Draft Critique Pass", 1
    )[0]
    assert "does not relax" in citation_section, (
        "Citation section must contain the 'does not relax' framing — "
        "broad-scope adherence-failure mode (OODS-PLUGIN-REVIEW-01) "
        "depends on this cue."
    )
    assert "before you call `final_answer`" in citation_section.lower() or (
        "before `final_answer`" in citation_section.lower()
        and "scan" in citation_section.lower()
    ), (
        "Citation section must instruct the agent to self-check the "
        "draft for unmarked claims before `final_answer` — this is the "
        "behavioral lever for the broad-scope failure mode."
    )


# ---------------------------------------------------------------------------
# S78 m01.d (PD82) — synthesis-vs-retrieval objective-classification clause.
# OODS-PLUGIN-REVIEW-01 burned steps trying to retrieve unfindable content
# (private system policies) and produced gap notes for that section. m01.d
# adds a 3-category classification (public-research / synthesis-from-context
# / out-of-scope) so the agent routes appropriately on first read.
# ---------------------------------------------------------------------------


def test_template_carries_objective_classification_section():
    """The synthesis-vs-retrieval clause exists with a load-bearing heading.

    A future prompt edit that drops the section entirely re-opens the
    OODS-PLUGIN-REVIEW-01 failure mode where the agent burned steps on
    unretrievable content.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    assert "# Objective Classification" in system, (
        "Synthesis-vs-retrieval clause must live under a top-level "
        "'# Objective Classification' heading so the agent reads it as a "
        "first-class instruction."
    )


def test_template_objective_classification_names_three_categories():
    """All three categories must be named per PD82.

    (1) public-research — default retrieval path
    (2) synthesis-from-context — apply public findings to declared properties
    (3) out-of-scope — decline and explain, do not gap-note pad

    Asserting on the category labels (not paragraph content) keeps the test
    resilient to prose tweaks while pinning the load-bearing taxonomy.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    classification_section = system.split("# Objective Classification", 1)[1].split(
        "# Tools Available", 1
    )[0]
    # Three category labels must be present.
    assert "Public-research" in classification_section or "public-research" in classification_section, (
        "Category 1 'Public-research objective' must be named explicitly."
    )
    assert "Synthesis-from-context" in classification_section or "synthesis-from-context" in classification_section, (
        "Category 2 'Synthesis-from-context objective' must be named "
        "explicitly so the agent recognizes the apply-findings pattern."
    )
    assert "Out-of-scope" in classification_section or "out-of-scope" in classification_section, (
        "Category 3 'Out-of-scope objective' must be named so the agent "
        "knows to decline rather than burn steps on unfindable content."
    )


# ---------------------------------------------------------------------------
# S81 m01.a (PD122) — `_resolve_deliverables` additive success_criteria
# fallback. Closes the Alexander-mission coverage-gate blindness: missions
# that supply success_criteria but no explicit deliverables now feed the
# Required Deliverables prompt block + the completeness_check spec.
# ---------------------------------------------------------------------------


def test_resolve_deliverables_falls_back_to_success_criteria() -> None:
    """When deliverables=null and success_criteria=[...], fallback fires."""
    state = {
        "mission_id": "FALLBACK-01",
        "mission_objectives": [],
        "mission_context": {
            "title": "Alexander mission shape",
            "objective": "trace lineage",
            "success_criteria": [
                "Cite 8 primary sources",
                "Comparison matrix across decades",
                "Pattern catalog covering each motif",
            ],
        },
    }
    assert resolve_deliverables(state) == [
        "Cite 8 primary sources",
        "Comparison matrix across decades",
        "Pattern catalog covering each motif",
    ]


def test_resolve_deliverables_handles_camelcase_success_criteria() -> None:
    """The fallback path accepts ``successCriteria`` (camelCase) too."""
    state = {
        "mission_id": "FALLBACK-CC",
        "mission_objectives": [],
        "mission_context": {
            "title": "CC shape",
            "objective": "trace",
            "successCriteria": ["Matrix", "Catalog"],
        },
    }
    assert resolve_deliverables(state) == ["Matrix", "Catalog"]


def test_resolve_deliverables_explicit_takes_precedence_over_success_criteria() -> None:
    """Explicit deliverables retain precedence; fallback never overrides them."""
    state = {
        "mission_id": "PRECEDENCE-01",
        "mission_objectives": [],
        "mission_context": {
            "title": "Both present",
            "objective": "compare",
            "deliverables": ["Hosting comparison matrix", "Pattern catalog"],
            "success_criteria": ["DO NOT USE 1", "DO NOT USE 2"],
        },
    }
    assert resolve_deliverables(state) == [
        "Hosting comparison matrix",
        "Pattern catalog",
    ]


def test_resolve_deliverables_empty_when_neither_present() -> None:
    """When both deliverables and success_criteria are absent, returns []."""
    state = {
        "mission_id": "EMPTY-01",
        "mission_objectives": ["o1"],
        "mission_context": {
            "title": "Empty contract",
            "objective": "explore",
        },
    }
    assert resolve_deliverables(state) == []


def test_resolve_deliverables_logs_substitution(caplog) -> None:
    """The fallback path logs at INFO so the substitution is observable."""
    import logging

    state = {
        "mission_id": "LOG-01",
        "mission_objectives": [],
        "mission_context": {
            "title": "Log shape",
            "objective": "trace",
            "success_criteria": ["A", "B"],
        },
    }
    with caplog.at_level(logging.INFO, logger="deepsearch.agent.contract_to_prompt"):
        result = resolve_deliverables(state)
    assert result == ["A", "B"]
    assert any(
        "synthesising from" in rec.message and "success_criteria" in rec.message
        for rec in caplog.records
    ), f"Expected INFO log on fallback substitution. Logged: {[r.message for r in caplog.records]}"


def test_build_codeagent_prompts_renders_success_criteria_as_deliverables() -> None:
    """End-to-end: success_criteria flow through into Required Deliverables.

    Closes the Alexander-mission failure mode: the agent now sees the same
    list under both ``# Research Objectives`` (which already fell back) AND
    ``# Required Deliverables`` so completeness_check has something to
    check against.
    """
    state = {
        "mission_id": "ALEXANDER-LIKE",
        "mission_objectives": [],
        "mission_context": {
            "title": "Alexander mission shape",
            "objective": "trace lineage",
            "success_criteria": [
                "Comparison matrix across schools",
                "Pattern catalog covering each motif",
            ],
        },
    }
    system, _ = build_codeagent_prompts(state)
    deliverables_block = system.split("# Required Deliverables", 1)[1].split(
        "# Objective Classification", 1
    )[0]
    assert "1. Comparison matrix across schools" in deliverables_block
    assert "2. Pattern catalog covering each motif" in deliverables_block


def test_template_objective_classification_lands_before_tools():
    """The clause must precede `# Tools Available` so it shapes retrieval.

    If classification lands AFTER the tool block, the agent has already
    been primed to retrieve aggressively before being told to triage. The
    classification step is intentionally a first-read action.
    """

    state = {
        "mission_id": "X",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    objectives_idx = system.find("# Research Objectives")
    classification_idx = system.find("# Objective Classification")
    tools_idx = system.find("# Tools Available")
    assert objectives_idx != -1
    assert classification_idx != -1
    assert tools_idx != -1
    assert objectives_idx < classification_idx < tools_idx, (
        "Section order must be: Research Objectives → Objective "
        "Classification → Tools Available. Classification triages the "
        "objectives; it must read before the tool surface is described."
    )
