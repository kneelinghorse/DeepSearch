"""Static-AST regression guards for S81 m02 — References Regex Generality v2
+ Output-Format Prompt Nudge.

17th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD123:

* ``deepsearch/agent/citation_extraction.py`` ``_REFERENCES_HEADING_RE``
  trailing slot is the multi-word-with-prep-negation form: a paren-wrapped
  phrase OR ``(?!(?:of|for|about|from|by|on|in)\\b)[A-Za-z][^\\n]*?``. The
  S80 single-word slot is structurally retired by this evolution.
* The trailing slot carries the em-dash / en-dash / hyphen separator option
  ``(?:[—–\\-][ \\t]+)?`` so dash-separated trailing phrases match.
* The compiled regex matches the verbatim STRUCTURED-OUTPUT-VIEWER-01
  failure heading shape (``## 11. Formatted references list with [N]-style
  numbered citations``).
* The compiled regex rejects all of the prep-prefix-negated body-prose
  shapes (``## Sources of Bias``, ``## References for Internal Reviewers``,
  ``## Bibliography from Primary Texts``, ``## References by Topic``,
  ``## Sources on Methodology``, ``## References in Chronological Order``,
  ``## Sources about Bias Reduction``).
* ``deepsearch/agent/contract_to_prompt.py`` `_SYSTEM_INSTRUCTIONS_TEMPLATE`
  Output Format §7 contains the heading-shape nudge: the agent is
  instructed that the section heading must be exactly ``## References``
  (or ``## N. References``) without descriptive trailing decoration.

These guards fail loudly if a future sprint:

* Reverts the trailing slot to the S80 single-word-or-paren form.
* Removes the prep-prefix negation lookahead.
* Drops the em-dash / en-dash / hyphen separator.
* Removes the §7 heading-shape nudge from the prompt template.

Tests do not call any LLM, do not hit the network, and exercise only AST
inspection plus a regex-only runtime match check on the compiled pattern
and a prompt-rendering call on a minimal state.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CITATION_EXTRACTION_PATH = REPO_ROOT / "deepsearch" / "agent" / "citation_extraction.py"
CONTRACT_TO_PROMPT_PATH = REPO_ROOT / "deepsearch" / "agent" / "contract_to_prompt.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _references_heading_re_assignment_pattern() -> str:
    """Return the joined string-literal arguments of the top-level
    ``_REFERENCES_HEADING_RE = re.compile(...)`` assignment."""
    tree = _module_ast(CITATION_EXTRACTION_PATH)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "_REFERENCES_HEADING_RE"
            for t in node.targets
        ):
            continue
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "compile"
        ):
            raise AssertionError(
                "_REFERENCES_HEADING_RE must be assigned via re.compile(...)."
            )
        joined: list[str] = []
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                joined.append(arg.value)
        return "".join(joined)
    raise AssertionError("_REFERENCES_HEADING_RE assignment not found.")


# ---------------------------------------------------------------------------
# m02.a — regex shape pinning.
# ---------------------------------------------------------------------------


def test_references_heading_re_trailing_slot_carries_prep_negation() -> None:
    """The trailing slot must carry the prep-prefix lookahead negation."""
    pattern = _references_heading_re_assignment_pattern()
    assert r"(?!(?:of|for|about|from|by|on|in)\b)" in pattern, (
        "S81 m02.a (PD123) requires the multi-word trailing slot to carry a "
        "negative lookahead `(?!(?:of|for|about|from|by|on|in)\\b)` on the "
        "first token. Without it `## Sources of Bias` / `## References for "
        "Internal Reviewers` regress to false-positive matches."
    )


def test_references_heading_re_trailing_slot_supports_em_dash_separator() -> None:
    """The trailing slot must allow `[—–\\-]` separator before the phrase."""
    pattern = _references_heading_re_assignment_pattern()
    # The separator slot lives inside the trailing alternation.
    assert r"[—–\-]" in pattern, (
        "S81 m02.a (PD123) requires an em-dash / en-dash / hyphen separator "
        "in the trailing slot to match `## 12. References — Cited Throughout`."
    )


def test_references_heading_re_trailing_slot_supports_multi_word_phrase() -> None:
    """The trailing slot must allow `[A-Za-z][^\\n]*?` (multi-word) phrase."""
    pattern = _references_heading_re_assignment_pattern()
    assert r"[A-Za-z][^\n]*?" in pattern, (
        "S81 m02.a (PD123) requires a non-greedy multi-character trailing "
        "phrase pattern (`[A-Za-z][^\\n]*?`). Reverting to the S80 single-"
        "word `[A-Za-z]\\w*` regresses the STRUCTURED-OUTPUT-VIEWER-01 "
        "post-S80 verbatim failure case."
    )


def test_references_heading_re_retired_single_word_slot() -> None:
    """The S80 single-word-only trailing slot must NOT remain in the pattern.

    Detecting the literal substring ``[A-Za-z]\\w*))?`` would mean the S80
    trailing slot is still present, indicating an incomplete migration.
    """
    pattern = _references_heading_re_assignment_pattern()
    # The S80 single-word slot ended with `[A-Za-z]\w*))?`. After m02.a the
    # slot ends with `[A-Za-z][^\n]*?))?`.
    assert r"[A-Za-z]\w*))?" not in pattern, (
        "S81 m02.a (PD123) retires the S80 single-word-only trailing slot. "
        "Reverting re-introduces the STRUCTURED-OUTPUT-VIEWER-01 references=0 "
        "failure mode."
    )


# ---------------------------------------------------------------------------
# m02.a — runtime regex correctness on the load-bearing failure case.
# ---------------------------------------------------------------------------


S81_M02A_LOAD_BEARING_POSITIVES = (
    # The post-S80 STRUCTURED-OUTPUT-VIEWER-01 verbatim heading. This is the
    # canonical empirical anchor for the m02.a fix; if this rejects, the
    # post-S81 STRUCTURED-OUTPUT-VIEWER-01 dispatch will continue to report
    # references=0 in metadata.
    "## 11. Formatted references list with [N]-style numbered citations",
    # Em-dash trailing.
    "## 12. References — Cited Throughout",
    # En-dash trailing.
    "## References – Compiled From Primary Texts",
    # Hyphen trailing.
    "## References - Compiled Inline",
    # Multi-word trailing without prep prefix (Shape C path).
    "## References list compiled from primary sources",
    # S80 carry — Shape A combined leading+trailing.
    "## 9. Formatted References List",
    # Bare flat heading.
    "## References",
)


S81_M02A_PREP_NEGATION_NEGATIVES = (
    # S80 PD113 — `of <noun>` shape.
    "## Sources of Bias",
    "## Bibliography of Methods",
    # S81 PD123 — extended prep-prefix negation set.
    "## References for Internal Reviewers",
    "## Sources about Bias Reduction",
    "## Bibliography from Primary Texts",
    "## References by Topic",
    "## Sources on Methodology",
    "## References in Chronological Order",
)


def test_compiled_regex_matches_s81_load_bearing_positives() -> None:
    """PD130 verification expectation #5 — local positive coverage."""
    from deepsearch.agent.citation_extraction import _REFERENCES_HEADING_RE

    for heading in S81_M02A_LOAD_BEARING_POSITIVES:
        assert _REFERENCES_HEADING_RE.search(heading) is not None, (
            f"S81 m02.a (PD123) positive case {heading!r} must match the "
            "_REFERENCES_HEADING_RE multi-word trailing slot."
        )


def test_compiled_regex_rejects_prep_prefix_negatives() -> None:
    """PD130 verification expectation #6 — prep-prefix negation negative coverage."""
    from deepsearch.agent.citation_extraction import _REFERENCES_HEADING_RE

    for non_heading in S81_M02A_PREP_NEGATION_NEGATIVES:
        assert _REFERENCES_HEADING_RE.search(non_heading) is None, (
            f"S81 m02.a (PD123) negative case {non_heading!r} must NOT "
            "match — its trailing first token is in the prep-prefix "
            "negation set `(of|for|about|from|by|on|in)`."
        )


def test_pattern_remains_a_compiled_regex_with_flags() -> None:
    """Belt-and-suspenders — the runtime symbol must remain a compiled
    Pattern with IGNORECASE + MULTILINE flags."""
    from deepsearch.agent.citation_extraction import _REFERENCES_HEADING_RE

    assert isinstance(_REFERENCES_HEADING_RE, re.Pattern)
    assert _REFERENCES_HEADING_RE.flags & re.IGNORECASE
    assert _REFERENCES_HEADING_RE.flags & re.MULTILINE


# ---------------------------------------------------------------------------
# m02.b — Output Format §7 heading-shape nudge.
# ---------------------------------------------------------------------------


def test_system_instructions_template_carries_heading_shape_nudge() -> None:
    """The §7 prompt nudge must appear in `_SYSTEM_INSTRUCTIONS_TEMPLATE`."""
    source = CONTRACT_TO_PROMPT_PATH.read_text(encoding="utf-8")

    # The nudge contains a couple of load-bearing phrases that pin the
    # instruction's intent. Asserting on the substrings rather than the
    # whole sentence keeps the test resilient to minor copy edits while
    # pinning the agent-readable contract.
    assert "section heading MUST be exactly `## References`" in source, (
        "S81 m02.b (PD123) requires the Output Format §7 prompt nudge to "
        "instruct the agent that the section heading must be exactly "
        "`## References` (or `## N. References` when numbering). Removing "
        "the nudge regresses the structural cue that pairs with the m02.a "
        "regex relaxation."
    )
    assert "descriptive embellishment causes the references count to silently report zero" in source, (
        "S81 m02.b (PD123) requires the §7 nudge to explain WHY the agent "
        "must not decorate the heading — the agent reads cause-and-effect "
        "framing as load-bearing in a way it does not read advisory text."
    )


def test_rendered_prompt_contains_heading_shape_nudge() -> None:
    """End-to-end: the rendered system prompt carries the §7 nudge.

    Some prompt edits (e.g. a typo in the f-string interpolation) can
    silently break the rendered output even when the template source is
    correct. This guard renders the prompt on a minimal state and asserts
    the nudge appears in the runtime-emitted text the agent actually sees.
    """
    from deepsearch.agent.contract_to_prompt import build_codeagent_prompts

    state = {
        "mission_id": "S81-M02B-PROMPT-SNAPSHOT",
        "mission_objectives": ["o1"],
        "mission_context": {"title": "T", "objective": "o"},
    }
    system, _ = build_codeagent_prompts(state)

    output_format_block = system.split("# Output Format", 1)[1]
    references_section = output_format_block.split("# Citation Rules", 1)[0]

    assert "section heading MUST be exactly `## References`" in references_section, (
        "S81 m02.b — rendered system prompt must contain the heading-shape "
        "nudge under # Output Format §7."
    )
