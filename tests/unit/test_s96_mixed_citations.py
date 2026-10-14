"""Mixed reader/machine citations must preserve ownership and quarantine."""

from __future__ import annotations

from typing import Any

import pytest

from deepsearch.agent.citation_extraction import extract_citations
from deepsearch.agent.citation_liveness import quarantine_dead_citations
from deepsearch.agent.codeagent_finalize_node import (
    _annotate_references_and_check_consistency,
    _canonicalize_citation_urls,
    _extract_and_resolve_citations,
)
from deepsearch.report.citation import rebase_citation_annotations

MIXED = (
    "Alpha claim [1][cite:https://example.org/a][2].\n"
    "Beta claim [3].\n\n## References\n"
    "[1] Alpha https://example.org/a\n[2] Other https://example.org/b\n"
    "[3] Beta https://example.org/c\n"
)


def test_mixed_ownership_survives_both_audits_and_persisted_rebase() -> None:
    telemetry: dict[str, Any] = {}
    text, annotations = _extract_and_resolve_citations(
        MIXED, sources_collected=[{"url": "https://example.org/a", "title": "Alpha"}],
        synthesis_telemetry=telemetry,
    )
    assert len(annotations) == 3
    assert [a._extraction_mode for a in annotations] == ["primary", "numbered", "numbered"]
    assert annotations[0].end_index == annotations[1].end_index
    counts = telemetry["citation_telemetry"]["reconciliation"]
    assert counts["primary_annotations"] == 1 and counts["numbered_annotations"] == 3
    assert counts["duplicates_removed"] == 1
    for a in annotations:
        a.live = True
    _, before = _annotate_references_and_check_consistency(
        text, kept_annotations=annotations, downgraded_dead=[], quarantined=[],
        require_number_mapping=True, annotate_flags=False,
    )
    assert before["unmapped_protocol_citations"] == 0
    # A deterministic critique transform may change unrelated prose; ownership
    # belongs to the annotation and must survive rebasing into the exact artifact.
    segment = text + "\nReview note.\n"
    persisted = "# Report wrapper\n\n" + segment
    rebased = rebase_citation_annotations(reference_markdown=text,
        persisted_segment_markdown=segment, persisted_markdown=persisted, annotations=annotations)
    assert not rebased.dropped
    assert [a._extraction_mode for a in rebased.annotations] == [
        a._extraction_mode for a in annotations
    ]
    _, after = _annotate_references_and_check_consistency(
        persisted, kept_annotations=rebased.annotations, downgraded_dead=[], quarantined=[],
        require_number_mapping=True, annotate_flags=False,
    )
    assert after["unmapped_protocol_citations"] == 0
    for old, new in zip(annotations, rebased.annotations):
        assert text[old.start_index:old.end_index] == persisted[new.start_index:new.end_index]
        assert "_extraction_mode" not in new.model_dump()
        assert set(new.model_dump()) == {"type", "start_index", "end_index", "url", "title", "live"}


def test_duplicate_representation_does_not_merge_distinct_claims_or_origins() -> None:
    text = (
        "Claim. [1][1][cite:https://example.org/a][cite:https://example.org/a][2].\n"
        "Another claim [1].\n\n## References\n[1] https://example.org/a\n"
        "[2] https://www.example.org/a\n"
    )
    counts: dict[str, Any] = {}
    stripped, citations = extract_citations(text, telemetry=counts)
    assert len(citations) == 3
    assert counts["duplicates_removed"] == 3
    assert len({a.url for a in citations}) == 2
    assert all(stripped[a.start_index:a.end_index].startswith(("Claim.", "Another"))
               for a in citations)


@pytest.mark.parametrize("label", ["[1]", "¹", "**[1]**", "1."])
def test_conflicting_reference_numbers_are_unresolved(label: str) -> None:
    text = ("Claim [1][cite:https://example.org/a].\n## References\n"
            f"[1] https://example.org/a\n{label} https://example.org/b\n")
    counts: dict[str, Any] = {}
    _, citations = extract_citations(text, telemetry=counts)
    assert len(citations) == 1 and citations[0]._extraction_mode == "primary"
    assert counts["reference_number_collisions"] == 1
    assert counts["unresolved_numbered_markers"] == 1


def test_mixed_code_fences_superscripts_missing_refs_and_direct_links() -> None:
    text = (
        "Real claim¹[cite:https://example.org/a]. Missing [9].\n"
        "`[1][cite:private]` and ``[1] `nested` `` and [1](https://example.org).\n"
        "~~~python\n[1][cite:private]\n~~~\n"
        "    [1][cite:private]\n"
        "## References\n¹ https://example.org/a\n"
        "```\n[2] https://private.test\n## References\n[1] https://private.test\n"
    )
    counts: dict[str, Any] = {}
    stripped, citations = extract_citations(text, telemetry=counts)
    assert len(citations) == 1 and citations[0].url == "https://example.org/a"
    assert counts["duplicates_removed"] == 1
    assert counts["unresolved_numbered_markers"] == 1
    assert stripped.count("[cite:private]") == 3


def test_recovered_numbered_links_still_obey_url_and_hallucination_quarantine() -> None:
    text = (
        "Collected [1][cite:https://example.org/a].\nInvented [2].\nUnsafe [3].\n"
        "## References\n[1] https://example.org/a\n[2] https://example.org/b\n"
        "[3] https://user:password@example.org/c\n"
    )
    _, citations = extract_citations(text)
    safe, projection = _canonicalize_citation_urls(citations)
    assert projection["dropped_count"] == 1
    for a in safe:
        a.live = False
    kept, quarantined = quarantine_dead_citations(safe, [{"url": "https://example.org/a"}])
    assert [a.url for a in kept] == ["https://example.org/a"]
    assert [a.url for a in quarantined] == ["https://example.org/b"]
    assert kept[0]._extraction_mode == "primary"
    assert quarantined[0]._extraction_mode == "numbered"




def test_conflicting_machine_and_reader_urls_remain_separate_and_visible() -> None:
    text = "Claim [1][cite:https://example.org/a].\n## References\n[1] https://example.org/b\n"
    stripped, citations = extract_citations(text)
    assert {a.url for a in citations} == {"https://example.org/a", "https://example.org/b"}
    _, audit = _annotate_references_and_check_consistency(
        stripped, kept_annotations=citations, downgraded_dead=[], quarantined=[],
        require_number_mapping=True, annotate_flags=False,
    )
    assert audit["unmapped_protocol_citations"] == 1
    assert audit["protocol_only"] == 1


def test_unparseable_reference_number_cannot_discard_the_report() -> None:
    text = "Claim [1].\n## References\n[" + "1" * 5000 + "] https://example.org/a\n"
    stripped, citations = extract_citations(text)
    assert stripped == text and citations == []
