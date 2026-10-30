"""S102-m04: a citation inside a References entry is audited against its own entry.

An agent can end a References entry with a ``[cite:URL]`` marker for that
entry's own URL. The audit binds a citation to the reader number at its
boundary, which for an entry is the next entry's label, so every such citation
counted unmapped. A citation whose start lies on a numbered entry line of the
audited text now maps when its URL is among that line's URLs, and is otherwise
unmapped; the entry's label is its reader number. Every other citation is
audited as before, and extraction, persisted annotations and rendering are
unchanged.
"""

from __future__ import annotations

from typing import Any

from deepsearch.agent.citation_extraction import extract_citations
from deepsearch.agent.codeagent_finalize_node import (
    _annotate_references_and_check_consistency,
)
from deepsearch.agent.quality_series import _citation_surface

A = "https://example.org/a"
B = "https://example.org/b"
C = "https://example.org/c"
BODY = f"# Report\n\nClaim one holds [1] [cite:{A}].\n\nClaim two holds [2] [cite:{B}].\n\n"


def references(*entries: str) -> str:
    return "## References\n\n" + "\n".join(entries) + "\n"


def audit(marked: str, *, prefix: str = "") -> dict[str, Any]:
    """Extract as finalize does, then run the final References audit.

    ``prefix`` shifts the audit's own markdown, as the pre-render pass audits
    the critique output against coordinates in the stripped draft.
    """
    stripped, annotations = extract_citations(marked)
    for annotation in annotations:
        annotation.live = True
    _, consistency = _annotate_references_and_check_consistency(
        prefix + stripped, kept_annotations=annotations, downgraded_dead=[], quarantined=[],
        citation_reference_markdown=stripped, require_number_mapping=True,
        annotate_flags=False, audit_quarantined_mapping=False,
    )
    return consistency


def fields(consistency: dict[str, Any]) -> tuple[int, int, int, int]:
    return (consistency["unmapped_protocol_citations"],
            consistency["annotations_without_reader_marker"],
            consistency["references_entry_citations"],
            consistency["references_entry_unmapped"])


def test_each_entry_citing_its_own_url_maps_by_its_label() -> None:
    marked = BODY + references(
        f"[1] Source A — {A} [cite:{A}]", f"[2] Source B — {B} [cite:{B}]",
        f"[3] Source C — {C} [cite:{C}]")
    consistency = audit(marked)
    assert fields(consistency) == (0, 0, 3, 0)
    # Entry [3] is never cited in the body: an accounting surplus, not a defect.
    assert consistency["unused_reference_numbers"] == [3]
    assert not _citation_surface({"citation_consistency": consistency})["actionable"]


def test_an_entry_citing_another_entrys_url_is_unmapped() -> None:
    marked = BODY + references(
        f"[1] Source A — {A}", f"[2] Source B — {B} [cite:{C}]", f"[3] Source C — {C}")
    consistency = audit(marked)
    assert fields(consistency) == (1, 0, 1, 1)
    assert _citation_surface({"citation_consistency": consistency})["actionable"]


def test_a_body_citation_just_before_the_references_heading_is_audited_as_before() -> None:
    # Its reader number trails it on the body line, so the boundary rule maps it.
    marked = BODY + f"Claim three closes the body [3] [cite:{C}].\n\n" + references(
        f"[1] Source A — {A}", f"[2] Source B — {B}", f"[3] Source C — {C}")
    assert fields(audit(marked)) == (0, 0, 0, 0)
    # Without a reader number it stays unmapped, exactly as before.
    marked = BODY + f"Claim three closes the body [cite:{C}].\n\n" + references(
        f"[1] Source A — {A}", f"[2] Source B — {B}", f"[3] Source C — {C}")
    assert fields(audit(marked)) == (1, 1, 0, 0)


def test_an_unnumbered_references_line_binds_nothing() -> None:
    marked = BODY + references(
        f"[1] Source A — {A}", f"[2] Source B — {B}", f"- Source C — {C} [cite:{C}]")
    # The bullet has no label, so the boundary rule applies: no reader number.
    assert fields(audit(marked)) == (1, 1, 0, 0)


def test_a_numbered_fallback_citation_on_an_entry_line_keeps_the_boundary_rule() -> None:
    # When a later sources heading repeats the list, the extractor reads the
    # audit's References labels as body markers: each numbered citation starts
    # on the previous entry's line and owns the label after it, as before m04.
    marked = ("Claim one [1]. Claim two [2]. Claim three [3].\n\n" + references(
        f"[1] Source A — {A}", f"[2] Source B — {B}", f"[3] Source C — {C}")
        + f"\n## Sources Consulted\n[1] Source A — {A}\n[2] Source B — {B}\n"
          f"[3] Source C — {C}\n")
    stripped, annotations = extract_citations(marked)
    on_entries = [a for a in annotations
                  if a.start_index > stripped.index("## References")
                  and a.start_index < stripped.index("## Sources Consulted")]
    assert on_entries and {a._extraction_mode for a in on_entries} == {"numbered"}
    consistency = audit(marked)
    assert fields(consistency) == (0, 0, 0, 0)
    assert consistency["consistent"] is True


def test_entry_lines_come_from_the_audited_text_on_both_passes() -> None:
    marked = BODY + references(f"[1] Source A — {A} [cite:{A}]", f"[2] Source B — {B}")
    # The pre-render pass audits a shifted markdown against the stripped draft.
    assert fields(audit(marked, prefix="A critique note.\n\n")) == (0, 0, 1, 0)
    assert fields(audit(marked)) == (0, 0, 1, 0)
