"""S98-m08: a source-list deliverable is delivered by the report's reference list.

Paid run S98-STIG-ACCEPT-03 delivered "Full source list, every entry carrying an
evidence tier and an access date" as its References section, every entry tiered
and dated, yet scored it missing: "References" shares no content token with the
deliverable, and the false shortfall labelled an undamaged report degraded. A
deliverable whose lead noun phrase names the source list binds a section headed
References, Bibliography or Sources, the reference-list convention the report
prose filter already uses (decision 1015). Nothing else binds those headings by
this rule, and the section still needs substance.
"""

from __future__ import annotations

import pytest

from deepsearch.agent.tools.completeness_check import completeness_check

SOURCE_LIST = "Full source list, every entry carrying an evidence tier and an access date"
BODY = (
    "This section carries the delivered analysis with enough substantive words to "
    "clear the section floor, grounded in the cited evidence and written out in "
    "full rather than as a placeholder. It names sources, gives figures, and "
    "explains the reasoning behind each conclusion in plain terms for readers."
)
ENTRIES = (
    "[1] SwarmWorld — https://arxiv.org/abs/2608.26081 — T2 abstract-read — "
    "Access date: 2026-10-29\n"
    "[2] Emergent Collective Memory — https://arxiv.org/abs/2512.10166 — T2 abstract-read — "
    "Access date: 2026-10-29\n"
)


def draft(heading: str, body: str = ENTRIES) -> str:
    return f"## 1. Findings\n\n{BODY}\n\n## {heading}\n\n{body}"


@pytest.mark.parametrize("heading", [
    "References", "12. References", "Bibliography", "Sources",
    "Appendix: Full Source List (tier and access date per entry)",
])
def test_the_reference_list_section_delivers_a_source_list(heading: str) -> None:
    assert completeness_check(draft(heading), [SOURCE_LIST])["missing"] == []


@pytest.mark.parametrize("bullet", [
    "Dated sources list distinguishing vendor claims from independent evidence",
    "List of sources consulted, with access dates",
    "An annotated bibliography",
    "Works cited",
])
def test_each_way_of_naming_the_source_list_binds_it(bullet: str) -> None:
    assert completeness_check(draft("References"), [bullet])["missing"] == []


@pytest.mark.parametrize("bullet", [
    "Citation verification ledger: a table with one row per flagged citation",
    "Evidence and source list",
    "Sources of funding analysis",
    "Source quality assessment",
])
def test_other_deliverables_never_bind_the_reference_list(bullet: str) -> None:
    assert completeness_check(draft("References"), [bullet])["missing"] != []


def test_a_stub_reference_list_does_not_deliver_it() -> None:
    stub = draft("References", "[Data unavailable in primary sources]\n")
    assert completeness_check(stub, [SOURCE_LIST])["missing"] == [SOURCE_LIST]
