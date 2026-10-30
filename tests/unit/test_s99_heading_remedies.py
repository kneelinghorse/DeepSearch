"""S99-m02: heading words that always bind, and remedies the agent can act on.

Paid run STIG-002 delivered all seven deliverables, yet four scored missing:
their headings named the subject ("### 3.1 Khushiyant (arXiv:2512.10166)"),
not the deliverable ("Khushiyant adjudication"), and the agent was never told
which heading words bind. The agent-facing completeness_check now returns one
remedy per missing requirement. A heading made of the deliverable's lead phrase
binds it under any section number, because heading tokens drop numbering and
one-letter tokens just as deliverable tokens do. The finalize path and the
controller keep the default output, so persisted coverage and tiers are unchanged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from deepsearch.agent.codeagent_runner import build_codeagent_tools
from deepsearch.agent.completion_control import (
    COVERAGE_TOOL_TRAJECTORY_LIMIT,
    CompletionController,
)
from deepsearch.agent.contract_to_prompt import build_codeagent_prompts
from deepsearch.agent.tools.completeness_check import (
    HEADING_RULE,
    _heading_content_tokens,
    completeness_check,
)

ROOT = Path(__file__).resolve().parents[2]
BODY = (
    "This section carries the delivered analysis with enough substantive words to "
    "clear the section floor, grounded in the cited evidence and written out in "
    "full rather than as a placeholder. It names sources, gives figures, and "
    "explains the reasoning behind each conclusion in plain terms for readers."
)
UNRELATED = f"## Zzyzx\n\n{BODY}\n"
# The heading rule promises these numbering forms (plan, "Symmetric heading tokens").
PREFIXES = ["", "3 ", "3.1 ", "3.1a ", "1b. ", "11a. ", "IV. ", "B. ", "6) "]
# One-letter tokens, an abbreviation that a ". " cut would break, and a lead
# phrase with no content token of its own.
LEAD_PHRASES = {
    "U.S. regulatory landscape: federal and state rules": "U.S. regulatory landscape",
    "Reviewer's verdict: whether the claims hold up": "Reviewer's verdict",
    "Plan B: fallback strategy and triggers": "Plan B",
    "Dr. Smith's critique: whether the method holds": "Dr. Smith's critique",
    "Q&A: open questions from the review": "Q&A: open questions from the review",
}
MIXED_SPEC = [
    "A Markdown research report with citations",
    "At least 3 vendor profiles",
    "Comparison table with columns: Vendor, Price, Support",
    "Risk register",
    "Recommendations",
]
THIN_RISKS = "## Risk register\n\nTBD.\n"


@pytest.mark.parametrize(("bullet", "words"), LEAD_PHRASES.items())
def test_a_missing_deliverable_is_given_its_lead_phrase(bullet: str, words: str) -> None:
    [remedy] = completeness_check(UNRELATED, [bullet], remedies=True)["remedies"]
    assert (remedy["kind"], remedy["words"]) == ("heading", words)


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("bullet", list(LEAD_PHRASES))
def test_the_lead_phrase_binds_under_every_section_number(bullet: str, prefix: str) -> None:
    draft = f"## {prefix}{LEAD_PHRASES[bullet]}\n\n{BODY}\n"
    assert completeness_check(draft, [bullet])["missing"] == []


@pytest.mark.parametrize(("heading", "tokens"), [
    ("1b. Deliverable checklist", ["deliverable", "checklist"]),
    ("11a. Plan B", ["plan"]),
    ("IV. Civil remedies mix", ["civil", "remedies", "mix"]),
    ("B. Option analysis", ["option", "analysis"]),
    ("6) U.S. policy", ["policy"]),
    # Roman numerals are numbering only as a leading label, never elsewhere.
    ("Mix of civil IV fluids", ["mix", "civil", "iv", "fluids"]),
])
def test_numbering_and_one_letter_tokens_are_not_heading_content(
    heading: str, tokens: list[str],
) -> None:
    assert _heading_content_tokens(heading) == tokens


@pytest.mark.parametrize(("heading", "kept"), [
    ("**3.1a** Findings", "1a"), ("iv. Findings", "iv"), ("S1. Findings", "s1"),
])
def test_bold_and_lowercase_roman_numbering_stay_outside_the_rule(
    heading: str, kept: str,
) -> None:
    # The heading rule's examples do not promise these forms.
    assert kept in _heading_content_tokens(heading)


@pytest.mark.parametrize(("heading", "bullet"), [
    ("Option A", "Preferred option: cost and risk"),  # Bound this way before S99.
    ("Option B", "Preferred option: cost and risk"),
    ("Plan B", "Migration plan: phases and owners"),
    ("Model X", "Model comparison: accuracy and cost"),
])
def test_single_token_headings_bind_a_shared_word_as_a_documented_trade_off(
    heading: str, bullet: str,
) -> None:
    # Decision 820's accepted single-token trade-off, not intended matching: a
    # one-content-token heading binds any deliverable sharing that word. The
    # symmetry fix makes "Option B" behave as "Option A" always has.
    assert completeness_check(f"### {heading}\n\n{BODY}\n", [bullet])["missing"] == []


def test_a_zero_overlap_heading_still_never_binds() -> None:
    assert completeness_check(f"## 1b. Executive Summary\n\n{BODY}\n", [
        "Comparison matrix with rows for each entity",
    ])["missing"] == ["Comparison matrix"]


def test_a_thin_section_is_told_to_expand_not_to_retitle() -> None:
    [remedy] = completeness_check(THIN_RISKS, ["Risk register with owners"], remedies=True)[
        "remedies"]
    assert (remedy["kind"], remedy["heading"]) == ("expand", "Risk register")


def test_a_gap_note_section_is_told_to_expand() -> None:
    draft = (
        "## 4. Risk register\n\n[Data unavailable in primary sources]\n\n"
        f"### 4.1 Detail\n\n{BODY}\n"
    )
    coverage = completeness_check(draft, ["Risk register with owners"], remedies=True)
    assert coverage["missing"] == ["Risk register"]
    assert coverage["remedies"][0]["kind"] == "expand"
    assert coverage["remedies"][0]["heading"] == "4. Risk register"


def test_quantity_columns_citation_and_artifact_requirements_get_their_own_remedies() -> None:
    draft = THIN_RISKS + "".join(f"\n### Vendor {n}\n\nNotes.\n" for n in (1, 2))
    coverage = completeness_check(draft, MIXED_SPEC, remedies=True)
    assert coverage["missing"] == [
        "A Markdown research report", "citations", "At least 3 vendor profiles",
        "Comparison table", "Risk register", "Recommendations",
    ]
    remedies = coverage["remedies"]
    # Exactly one remedy per missing requirement, in missing order, artifacts first.
    assert [r["requirement"] for r in remedies] == coverage["missing"]
    assert [r["kind"] for r in remedies] == [
        "artifact", "citations", "count", "columns", "expand", "heading"]
    assert (remedies[2]["minimum"], remedies[2]["found"]) == (3, 2)
    assert remedies[3]["columns"] == ["Vendor", "Price", "Support"]
    assert remedies[3]["words"] == "Comparison table"
    assert remedies[5]["words"] == "Recommendations"
    assert all(isinstance(r["advice"], str) and r["advice"] for r in remedies)


def test_a_duplicate_canonical_keeps_its_first_bullets_remedy() -> None:
    columns = "Comparison table with columns: Vendor, Price"
    plain = "Comparison table for the board"
    for spec, kind in (([columns, plain], "columns"), ([plain, columns], "heading")):
        coverage = completeness_check(UNRELATED, spec, remedies=True)
        assert coverage["missing"] == ["Comparison table"]
        assert [r["kind"] for r in coverage["remedies"]] == [kind]


def test_the_flag_only_adds_remedies_and_the_heading_rule() -> None:
    default = completeness_check(THIN_RISKS, MIXED_SPEC)
    assert "remedies" not in default and "heading_rule" not in default
    flagged = completeness_check(THIN_RISKS, MIXED_SPEC, remedies=True)
    assert {k: v for k, v in flagged.items() if k not in {"remedies", "heading_rule"}} == default
    assert flagged["heading_rule"] == HEADING_RULE
    satisfied = completeness_check(f"## Recommendations\n\n{BODY}\n", ["Recommendations"],
                                   remedies=True)
    assert satisfied["missing"] == [] and satisfied["remedies"] == []


@pytest.mark.parametrize(("draft", "spec"), [(None, MIXED_SPEC), (THIN_RISKS, None)])
def test_unassessed_coverage_has_no_remedies(draft: object, spec: object) -> None:
    coverage = completeness_check(draft, spec, remedies=True)  # type: ignore[arg-type]
    assert coverage["assessment_status"] == "unassessed"
    assert coverage["remedies"] == [] and coverage["heading_rule"] == HEADING_RULE


def test_the_heading_rule_names_the_numbering_it_promises() -> None:
    assert HEADING_RULE == (
        "A content deliverable counts only under the Markdown heading of the section that "
        "delivers it. If a missing deliverable is already in the draft under another heading, "
        "retitle that heading with the words given. A section number such as 3, 3.1 or 1b is "
        "fine; any other added word, including a parenthetical, can stop the match."
    )


def test_only_the_agent_facing_tool_asks_for_remedies() -> None:
    callers: dict[str, list[ast.Call]] = {}
    for path in (ROOT / "deepsearch").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            name = getattr(getattr(node, "func", None), "id", None) or getattr(
                getattr(node, "func", None), "attr", None)
            if isinstance(node, ast.Call) and name in {
                "completeness_check", "_completeness_check_impl",
            }:
                callers.setdefault(path.relative_to(ROOT).as_posix(), []).append(node)
    # The finalize path and the controller compute coverage, with the default.
    for name in ("codeagent_finalize_node.py", "completion_control.py"):
        calls = callers[f"deepsearch/agent/{name}"]
        assert calls and all(kw.arg != "remedies" for call in calls for kw in call.keywords)
    flagged = [
        (file, kw.value) for file, calls in callers.items() for call in calls
        for kw in call.keywords if kw.arg == "remedies"
    ]
    assert [(file, ast.literal_eval(value)) for file, value in flagged] == [
        ("deepsearch/agent/codeagent_runner.py", True)]


def test_the_controller_keeps_its_own_coverage_without_remedies() -> None:
    control = CompletionController(40, deliverables=MIXED_SPEC)
    control.observe_draft(THIN_RISKS + "\nClaim [cite:https://example.org/a].\n")
    assert control.checks == 1 and "remedies" not in control.coverage


def test_the_agent_tool_returns_remedies_and_reports_indexes_to_the_controller() -> None:
    control = CompletionController(40, deliverables=MIXED_SPEC)
    control.completed_steps = 4
    tool = build_codeagent_tools(MIXED_SPEC, completion_controller=control).tools[-1]
    result = tool(THIN_RISKS)
    assert [r["kind"] for r in result["remedies"]] == [
        "artifact", "citations", "count", "columns", "expand", "heading"]
    assert control.telemetry("normal_final_answer")["coverage_tool_trajectory"] == {
        "version": 1, "limit": COVERAGE_TOOL_TRAJECTORY_LIMIT, "total": 1, "dropped": 0,
        "checks": [{
            "seq": 1, "step": 5, "draft_chars": len(THIN_RISKS),
            "missing": [0, 1, 2, 3, 4, 5], "heading_remedies": [5],
        }],
    }


def test_the_coverage_trajectory_is_capped_and_counts_what_it_drops() -> None:
    control = CompletionController(40)
    coverage = completeness_check(THIN_RISKS, MIXED_SPEC, remedies=True)
    for _ in range(COVERAGE_TOOL_TRAJECTORY_LIMIT + 3):
        control.record_coverage_check(THIN_RISKS, coverage)
    trajectory = control.telemetry("normal_final_answer")["coverage_tool_trajectory"]
    assert (trajectory["total"], trajectory["dropped"]) == (COVERAGE_TOOL_TRAJECTORY_LIMIT + 3, 3)
    assert [check["seq"] for check in trajectory["checks"]] == list(
        range(1, COVERAGE_TOOL_TRAJECTORY_LIMIT + 1))


def test_without_tool_checks_the_trajectory_is_explicitly_empty() -> None:
    trajectory = CompletionController(40).telemetry("normal_final_answer")[
        "coverage_tool_trajectory"]
    assert (trajectory["total"], trajectory["dropped"], trajectory["checks"]) == (0, 0, [])


def test_the_prompt_states_the_heading_rule() -> None:
    system, _ = build_codeagent_prompts({
        "mission_id": "S99-01", "mission_objectives": ["compare A vs B"],
        "mission_context": {"title": "Heading rule", "deliverables": ["Comparison matrix"]},
    })
    section = system.split("# PRE-ACCEPTANCE COMPLETENESS CHECK", 1)[1].split("\n# ", 1)[0]
    assert (
        "Each content deliverable is matched to the Markdown heading of the section that "
        "delivers it, so title that section with the deliverable's leading words."
    ) in section
