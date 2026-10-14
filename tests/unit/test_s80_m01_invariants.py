"""Static-AST regression guards for the S80 m01 references-regex relaxation.

16th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD106:

* ``deepsearch/agent/citation_extraction.py`` defines ``_REFERENCES_HEADING_RE``
  via a top-level ``re.compile(...)`` assignment whose source contains the
  three-shape disjunctive structure (digit-prefix path, word-prefix path,
  bare-synonym path).
* The compiled regex matches all three S80 m01 positive shape exemplars and
  rejects the calibration-boundary negative shapes that the plan-doc §11
  risk register #1 + #2 enumerates.

These guards fail loudly if a future sprint:

* Collapses the three-shape disjunction back to a single shape (regressing
  either the leading-token or trailing-token tolerance closed at S80).
* Drops the word-boundary `\\b` after the synonym (would re-admit
  ``## Sourcesfoo`` / ``## Backsource`` partial matches).
* Removes the trailing single-word constraint (would re-admit
  ``## Sources of Bias`` / ``## References used in this study`` body-prose
  shapes as headings).
* Allows the word-prefix Shape B to also accept a trailing token (would
  re-admit ``## My References Workflow`` calibration-boundary false-positive).

Tests do not call any LLM, do not hit the network, and do not exercise the
runtime numbered-fallback path — pure AST inspection plus a regex-only runtime
match check on the compiled pattern.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CITATION_EXTRACTION_PATH = REPO_ROOT / "deepsearch" / "agent" / "citation_extraction.py"


def _module_ast() -> ast.Module:
    return ast.parse(CITATION_EXTRACTION_PATH.read_text(encoding="utf-8"))


def _references_heading_re_assignment_pattern() -> str:
    """Return the joined string-literal arguments of the top-level
    ``_REFERENCES_HEADING_RE = re.compile(...)`` assignment.

    The S80 m01 design splits the pattern across multiple raw-string segments
    for readability; the AST guard concatenates them so shape-level assertions
    can be made on the joined source.
    """
    tree = _module_ast()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "_REFERENCES_HEADING_RE"
            for t in node.targets
        ):
            continue
        call = node.value
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "compile"):
            raise AssertionError(
                "_REFERENCES_HEADING_RE must be assigned via re.compile(...) "
                "(S80 m01 contract preserves the S78 m01.a top-level shape)"
            )
        # Concatenate every string-literal positional argument (the pattern
        # is one logical regex split across many raw-string segments).
        joined: list[str] = []
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                joined.append(arg.value)
            else:
                raise AssertionError(
                    "_REFERENCES_HEADING_RE re.compile arguments must all be "
                    "string-literal segments (so the AST guard can audit the "
                    "shape statically)."
                )
        return "".join(joined)
    raise AssertionError(
        "Top-level `_REFERENCES_HEADING_RE = re.compile(...)` assignment not "
        "found in citation_extraction.py — S80 m01 RD2 invariant failure."
    )


# ---------------------------------------------------------------------------
# Definition-surface invariants (pure AST, no regex execution).
# ---------------------------------------------------------------------------


def test_references_heading_re_remains_top_level_constant() -> None:
    """The symbol must continue to live as a top-level module attribute.

    Required by ``test_extractor_module_carries_numbered_fallback_surface``
    in test_metadata_truthfulness_invariants.py. The AST guard mirrors that
    requirement without coupling to the metadata-truthfulness suite.
    """
    pattern = _references_heading_re_assignment_pattern()
    assert pattern, "Top-level pattern source must be non-empty"


def test_references_heading_re_carries_three_shape_structure() -> None:
    """PD106 contract — the regex must encode the three-shape disjunction.

    Each shape begins with the heading anchor ``^[ \\t]*#{1,6}[ \\t]+`` and
    ends with ``[ \\t]*$``. The S80 m01 relaxation requires THREE such
    anchored alternatives joined by ``|`` — collapsing them back to one
    shape regresses either the leading-token (Shape B) or trailing-token
    (Shape C) tolerance closed at this sprint.
    """
    pattern = _references_heading_re_assignment_pattern()
    anchor_count = pattern.count(r"^[ \t]*#{1,6}[ \t]+")
    assert anchor_count == 3, (
        "S80 m01 PD106 — _REFERENCES_HEADING_RE must contain exactly three "
        f"`^[ \\t]*#{{1,6}}[ \\t]+` heading anchors (Shape A/B/C); found "
        f"{anchor_count}."
    )
    # The disjunction marker between shapes.
    assert pattern.count(r"|") >= 2, (
        "S80 m01 PD106 — _REFERENCES_HEADING_RE must use disjunction `|` "
        "between Shape A/B/C alternatives."
    )


def test_references_heading_re_pins_word_boundary_after_synonym() -> None:
    """The synonym anchor must terminate at a word boundary `\\b`.

    Without `\\b`, ``## Sourcesfoo`` would partially match `Sources` followed
    by `foo` — which the trailing-token slot would then absorb. The boundary
    is the cheapest defense against that class of false-positive.
    """
    pattern = _references_heading_re_assignment_pattern()
    # Each of the three shapes has its own synonym occurrence; all three must
    # be followed by `\b`.
    synonym_with_boundary = pattern.count(
        r"(?:references|sources|bibliography)\b"
    )
    assert synonym_with_boundary == 3, (
        "S80 m01 PD106 — every synonym alternation must be followed by `\\b` "
        f"to pin the suffix boundary; found {synonym_with_boundary}/3."
    )


def test_references_heading_re_constrains_trailing_to_single_token() -> None:
    """The trailing slot must permit a paren-wrapped phrase OR a multi-word
    phrase whose first token is NOT a known English preposition.

    S80 m01 PD106 — original constraint was single-word OR paren-wrapped.
    S81 m02.a PD123 — widened to multi-word (the post-S80 STRUCTURED-OUTPUT-
    VIEWER-01 mission emitted ``## 11. Formatted references list with
    [N]-style numbered citations`` which the single-word slot rejected) but
    the multi-word path is gated by a prepositional-prefix lookahead
    negation so prose-shape false positives like ``## Sources of Bias`` /
    ``## Bibliography of Methods`` / ``## References for Internal
    Reviewers`` continue to reject. An optional em-dash / en-dash / hyphen
    separator handles ``## 12. References — Cited Throughout``.

    The exact trailing slot syntax for Shape A + Shape C is:
    ``(?:[ \\t]+(?:[—–\\-][ \\t]+)?(?:\\([^)\\n]+\\)|(?!(?:of|for|about|from|by|on|in)\\b)[A-Za-z][^\\n]*?))?``
    """
    pattern = _references_heading_re_assignment_pattern()
    # The exact S81 m02.a trailing slot syntax used in Shape A and Shape C.
    expected = (
        r"(?:[ \t]+(?:[—–\-][ \t]+)?(?:\([^)\n]+\)|"
        r"(?!(?:of|for|about|from|by|on|in)\b)[A-Za-z][^\n]*?))?"
    )
    occurrences = pattern.count(expected)
    assert occurrences == 2, (
        "S81 m02.a PD123 — trailing slot must be the multi-word-with-prep-"
        "negation form and must appear in Shape A and Shape C only (not "
        f"Shape B); found {occurrences}/2."
    )


# ---------------------------------------------------------------------------
# Runtime regex behavior (compiled pattern correctness).
# ---------------------------------------------------------------------------

S80_LOAD_BEARING_POSITIVE_HEADINGS = (
    # Shape A — digit + word combo (the verbatim post-S79 failing heading).
    "## 9. Formatted References List",
    # Shape A — digit only (pre-S80 baseline still passes).
    "## 12. Sources",
    # Shape B — single word + synonym, no trailing.
    "## Formatted References",
    "## Cited Sources",
    "## My Bibliography",
    # Shape B — `Word N:` shape.
    "## Section 9: References",
    # Shape C — bare synonym + single trailing word.
    "## References List",
    "## Sources Used",
    "## Bibliography Index",
    # Shape C — bare synonym + paren-wrapped trailing.
    "## References (Formatted)",
)

S80_CALIBRATION_BOUNDARY_NEGATIVE_HEADINGS = (
    # PD113 risk #1 — `synonym + of <noun>` body-prose shape.
    "## Sources of Bias",
    "## Bibliography of Methods",
    # PD113 risk #2 — word-prefix + trailing-word body-prose shape.
    "## My References Workflow",
    # Synonym shape mismatches.
    "## Reference Architecture",
    "## Backsource Notes",
    "## Bibliographic notes",
)


def test_compiled_regex_matches_all_load_bearing_positives() -> None:
    """PD114 verification expectation #1 — local positive coverage."""
    from deepsearch.agent.citation_extraction import _REFERENCES_HEADING_RE

    for heading in S80_LOAD_BEARING_POSITIVE_HEADINGS:
        assert _REFERENCES_HEADING_RE.search(heading) is not None, (
            f"S80 m01 PD106 positive case {heading!r} must match the "
            "_REFERENCES_HEADING_RE three-shape disjunction."
        )


def test_compiled_regex_rejects_calibration_boundary_negatives() -> None:
    """PD113 risk register — calibration-boundary negative coverage."""
    from deepsearch.agent.citation_extraction import _REFERENCES_HEADING_RE

    for non_heading in S80_CALIBRATION_BOUNDARY_NEGATIVE_HEADINGS:
        assert _REFERENCES_HEADING_RE.search(non_heading) is None, (
            f"S80 m01 PD106 negative case {non_heading!r} must NOT match "
            "the _REFERENCES_HEADING_RE three-shape disjunction "
            "(calibration boundary)."
        )


def test_pattern_remains_a_compiled_regex() -> None:
    """Belt-and-suspenders — the runtime symbol must be a compiled Pattern."""
    from deepsearch.agent.citation_extraction import _REFERENCES_HEADING_RE

    assert isinstance(_REFERENCES_HEADING_RE, re.Pattern), (
        "_REFERENCES_HEADING_RE must remain a compiled re.Pattern object."
    )
    # Case-insensitive AND multi-line flags are both load-bearing for the
    # `## REFERENCES` / `## Sources` / `## BIBLIOGRAPHY` cases AND for matching
    # the heading at the start of a line within a larger markdown document.
    assert _REFERENCES_HEADING_RE.flags & re.IGNORECASE, (
        "_REFERENCES_HEADING_RE must carry the IGNORECASE flag (S78 m01.a + "
        "S80 m01 case-insensitive synonym match)."
    )
    assert _REFERENCES_HEADING_RE.flags & re.MULTILINE, (
        "_REFERENCES_HEADING_RE must carry the MULTILINE flag (heading match "
        "anchored at line start within a multi-line markdown document)."
    )
