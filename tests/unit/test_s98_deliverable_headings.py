"""S98-m03: deliverable labels keep their noun, and real deliverables bind.

Three false-missing classes from the 33-run census (#1032, decision 821(a)):
a tail-boundary word cut "A mapping memo: ..." down to the article "A"; an
artifact phrased as "One concise Markdown comparison note" was scored as a
heading; and a table named by its required columns sat under an unrelated
heading. A possessive's one-letter token also let any heading containing "s"
bind a deliverable, so a deleted section could still read as delivered.
"""

from __future__ import annotations

import pytest

from deepsearch.agent.tools.completeness_check import (
    _canonical_heading,
    _canonical_tokens,
    _heading_matches,
    completeness_check,
)

BODY = (
    "This section carries the delivered analysis with enough substantive words to "
    "clear the section floor, grounded in the cited evidence and written out in "
    "full rather than as a placeholder. It names sources, gives figures, and "
    "explains the reasoning behind each conclusion in plain terms for readers."
)


@pytest.mark.parametrize(("bullet", "label"), [
    ("A mapping memo: under what conditions the mechanism licenses a prediction, "
     "and where the analogy breaks",
     "A mapping memo: under what conditions the mechanism licenses a prediction, "
     "and where the analogy breaks"),
    ("A mapping table. Required columns: Element, Answer, Confidence.",
     "A mapping table. Required columns: Element, Answer, Confidence"),
    ("The mapping of vendors to pricing tiers", "The mapping"),
    ("A final recommendation", "A final recommendation"),
    ("Comparison matrix with rows for each entity", "Comparison matrix"),
    ("Recommendation:", "Recommendation"),
])
def test_a_boundary_never_leaves_only_an_article(bullet: str, label: str) -> None:
    assert _canonical_heading(bullet) == label


def test_colon_and_sentence_ends_are_not_cut_points() -> None:
    assert _canonical_heading("Plan: the rollout schedule with phases") == (
        "Plan: the rollout schedule"
    )


def test_one_letter_tokens_never_bind_a_heading() -> None:
    deliverable = "A recommendations section addressed to the Librarian's design"
    assert "s" not in _canonical_tokens(_canonical_heading(deliverable))
    assert not _heading_matches(_canonical_heading(deliverable), "References")
    assert _heading_matches(_canonical_heading(deliverable), "6. Recommendations")


@pytest.mark.parametrize(("bullet", "artifact", "content"), [
    ("One concise Markdown comparison note with the requested table and worked timeline.",
     "One concise Markdown comparison note", "the requested table and worked timeline"),
    ("Markdown report with an options comparison table", "Markdown report",
     "an options comparison table"),
    ("A Markdown research report", "A Markdown research report", None),
])
def test_markdown_artifacts_split_from_their_content(
    bullet: str, artifact: str, content: str | None,
) -> None:
    draft = f"## Overview\n\n{BODY}\n"
    coverage = completeness_check(draft, [bullet])
    assert [r["requirement"] for r in coverage["artifact_requirements"]] == [artifact]
    contents = [r["requirement"] for r in coverage["content_requirements"]]
    assert contents == ([content] if content else [])


@pytest.mark.parametrize("bullet", [
    "Markdown report generation risks",
    "A Markdown table of monthly costs",
    "Five Markdown report sections",
])
def test_topics_that_mention_markdown_stay_content(bullet: str) -> None:
    coverage = completeness_check(f"## Overview\n\n{BODY}\n", [bullet])
    assert coverage["artifact_requirements"] == []


TABLE = (
    "## 4. Comparison Matrix\n\n"
    "| Shock element | Its historical answer | AI-as-context analogue | Confidence |\n"
    "|---|---|---|---|\n"
    "| Cheap images | Painting turned to colour | Cheap prose | Medium |\n"
)
SPEC = "A mapping table. Required columns: Shock element, Its historical answer, " \
       "AI-as-context analogue, Confidence"


def test_a_table_with_every_required_column_delivers_the_table() -> None:
    coverage = completeness_check(f"## Overview\n\n{BODY}\n\n{TABLE}", [SPEC])
    assert coverage["missing"] == []


def test_a_table_missing_a_required_column_does_not() -> None:
    table = TABLE.replace("| Confidence |", "| Likelihood |")
    coverage = completeness_check(f"## Overview\n\n{BODY}\n\n{table}", [SPEC])
    assert coverage["missing"] == [SPEC]


def test_a_header_without_data_rows_does_not() -> None:
    header_only = "\n".join(TABLE.splitlines()[:4]) + "\n"
    coverage = completeness_check(f"## Overview\n\n{BODY}\n\n{header_only}", [SPEC])
    assert coverage["missing"] == [SPEC]


def test_columns_listed_with_or_inside_a_name_stay_whole() -> None:
    spec = "Tool capability table with columns Tool, Scores or not, MCP or not"
    table = (
        "## 4. Comparison Matrix\n\n| Tool | Scores or not | MCP or not |\n|---|---|---|\n"
        "| Storybook | No | Yes |\n"
    )
    assert completeness_check(table, [spec])["missing"] == []
