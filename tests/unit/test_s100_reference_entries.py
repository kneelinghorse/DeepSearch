"""S100-m02: one source's links are one References entry, in both parsers.

Paid run STIG-003 listed four papers as ``[N] Title — <arXiv abstract URL> (HTML:
<arXiv HTML URL>)``. The extractor and the finalize audit both treated any entry
with two URLs as an ambiguous number, so 61 markers resolved to nothing and three
papers the agent read in full left the protocol. A line whose links all name one
source (``arxiv_identity(url) or url``) now resolves to its first link. Links to
different sources, and lines that resolve to different sources, still leave the
number unresolved rather than guessed. The extractor also reads the outer
References section when a references-like subsection is nested inside it, as the
audit already did; the nested list's own labels are not body markers.
"""

from __future__ import annotations

from typing import Any

from deepsearch.agent.citation_extraction import extract_citations
from deepsearch.agent.codeagent_finalize_node import _annotate_references_and_check_consistency

ABS = "https://arxiv.org/abs/2606.24535"
HTML = "https://arxiv.org/html/2606.24535"
OTHER = "https://arxiv.org/abs/2609.33517"


def extract(text: str) -> tuple[list[Any], dict[str, Any]]:
    counts: dict[str, Any] = {}
    _, annotations = extract_citations(text, telemetry=counts)
    return annotations, counts


def audit(text: str, annotations: list[Any]) -> dict[str, Any]:
    _, consistency = _annotate_references_and_check_consistency(
        text, kept_annotations=annotations, downgraded_dead=[], quarantined=[],
        require_number_mapping=True,
    )
    return consistency


def test_an_entry_giving_a_papers_abstract_and_html_resolves_to_its_first_link() -> None:
    text = f"Claim [1].\n\n## References\n[1] Paper — {ABS} (HTML full text: {HTML}) — tier 1\n"
    annotations, counts = extract(text)
    assert [a.url for a in annotations] == [ABS]
    assert counts["reference_number_collisions"] == counts["unresolved_numbered_markers"] == 0
    consistency = audit(text, annotations)
    assert consistency["reference_number_collisions"] == 0
    assert consistency["unmapped_protocol_citations"] == 0
    # The audit keeps every link on the line, so the HTML form maps to [1] too.
    html = annotations[0].model_copy(update={"url": HTML})
    assert audit(text, [html])["unmapped_protocol_citations"] == 0


def test_a_doi_and_an_unrelated_url_stay_an_unresolved_collision() -> None:
    text = ("Claim [1].\n\n## References\n[1] Paper — https://doi.org/10.1145/3290605.3300857 "
            "(PDF: https://example.org/paper.pdf)\n")
    annotations, counts = extract(text)
    assert annotations == []
    assert counts["reference_number_collisions"] == counts["unresolved_numbered_markers"] == 1
    assert audit(text, annotations)["reference_number_collisions"] == 1


def test_two_lines_for_one_number_citing_different_papers_collide() -> None:
    text = f"Claim [3].\n\n## References\n[3] One — {ABS}\n[3] Two — {OTHER}\n"
    annotations, counts = extract(text)
    assert annotations == []
    assert counts["reference_number_collisions"] == counts["unresolved_numbered_markers"] == 1
    assert audit(text, annotations)["reference_number_collisions"] == 1


def test_two_lines_giving_one_papers_abstract_and_html_resolve_to_the_first() -> None:
    text = f"Claim [3].\n\n## References\n[3] Paper — {ABS}\n[3] Paper (HTML) — {HTML}\n"
    annotations, counts = extract(text)
    assert [a.url for a in annotations] == [ABS]
    assert counts["reference_number_collisions"] == 0
    assert audit(text, annotations)["reference_number_collisions"] == 0


def test_a_nested_list_restarting_at_one_collides_only_on_the_shared_number() -> None:
    text = ("First [1]. Second [2].\n\n## References\n"
            "[1] A — https://example.org/a\n[2] B — https://example.org/b\n\n"
            "### Sources retrieved but not cited\n[1] C — https://example.org/c\n")
    annotations, counts = extract(text)
    # Nested entries belong to the outer section, so [1] names two sources.
    assert [a.url for a in annotations] == ["https://example.org/b"]
    assert counts["reference_number_collisions"] == counts["unresolved_numbered_markers"] == 1
    assert audit(text, annotations)["reference_number_collisions"] == 1


def test_a_nested_sources_subsection_does_not_hide_the_outer_entries() -> None:
    # S98-STIG-ACCEPT-03's shape: the subsection lists uncited URLs as bullets.
    text = ("Claim [1]. Other [2].\n\n## References\n"
            "[1] A — https://example.org/a\n[2] B — https://example.org/b\n\n"
            "### Sources retrieved but not cited as evidence\n\n"
            "- https://example.org/c — mirror of [2]; not cited separately.\n")
    annotations, counts = extract(text)
    assert [a.url for a in annotations] == ["https://example.org/a", "https://example.org/b"]
    assert counts["unresolved_numbered_markers"] == counts["reference_number_collisions"] == 0


def test_a_mixed_claim_keeps_only_the_primary_citation_of_a_two_link_entry() -> None:
    one_line = f"[1] Paper — {ABS} (HTML: {HTML})\n"
    for entry in (one_line, f"[1] Paper — {ABS}\n[1] Paper — {HTML}\n"):
        text = f"Claim [1] [cite:{HTML}].\n\n## References\n{entry}"
        annotations, counts = extract(text)
        assert [(a.url, a._extraction_mode) for a in annotations] == [(HTML, "primary")], entry
        assert counts["numbered_annotations"] == counts["duplicates_removed"] == 1


def test_a_primary_citation_of_another_paper_keeps_the_numbered_one() -> None:
    text = f"Claim [1] [cite:{OTHER}].\n\n## References\n[1] Paper — {ABS} (HTML: {HTML})\n"
    annotations, counts = extract(text)
    assert sorted(a.url for a in annotations) == [ABS, OTHER]
    assert counts["duplicates_removed"] == 0


def test_a_one_link_entry_keeps_its_numbered_citation_beside_the_primary() -> None:
    # Already resolved before S100, so the identity dedup must not touch it.
    text = f"Claim [1] [cite:{HTML}].\n\n## References\n[1] Paper — {ABS}\n"
    annotations, counts = extract(text)
    assert sorted((a.url, a._extraction_mode) for a in annotations) == [
        (ABS, "numbered"), (HTML, "primary")]
    assert counts["duplicates_removed"] == 0
