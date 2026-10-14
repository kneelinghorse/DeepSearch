"""Reader resolution is independent of protocol presence and claim truth."""

from __future__ import annotations


import pytest

from deepsearch.agent.citation_surface import reader_markers
from deepsearch.agent.codeagent_finalize_node import _annotate_references_and_check_consistency
from deepsearch.report.citation import CitationAnnotation



def _audit(text: str, citations: list[CitationAnnotation] | None = None) -> dict:
    unchanged, result = _annotate_references_and_check_consistency(
        text,
        kept_annotations=citations or [],
        downgraded_dead=[],
        quarantined=[],
        require_number_mapping=True,
        annotate_flags=False,
    )
    assert unchanged == text
    return result


def test_reader_marker_without_any_annotation_or_references_is_not_silent() -> None:
    result = _audit("A visible claim [16].")
    assert result["dangling_reader_numbers"] == [16]
    assert result["unresolved_reader_markers"] == 1
    assert result["consistent"] is False


@pytest.mark.parametrize("label", ["**[1] Source**", "[1] Source", "1. Source", "1) Source"])
def test_supported_reference_labels_resolve_without_rewriting(label: str) -> None:
    text = f"Claim [1].\n\n## References\n\n{label} — https://example.org/a\n"
    annotation = CitationAnnotation(start_index=0, end_index=9, url="https://example.org/a")
    result = _audit(text, [annotation])
    assert result["consistent"]
    assert result["numbered_references"] == 1
    assert not result["dangling_reader_numbers"]


def test_nonreader_code_escaped_markers_and_direct_links_do_not_create_fake_defects() -> None:
    text = """Claim [1]¹.
`code [2]` and ``[3] `nested` `` and \\[4] and [5](https://example.org).
```python
x[6]
```
~~~
x[7]
~~~
    x[8]
"""
    assert reader_markers(text) == [1, 1]


def test_unused_unknown_shapes_and_bad_coordinates_remain_distinct() -> None:
    text = "Claim [1].\n\n## References\n[1] https://example.org/a\n[2] https://example.org/b\n(3) https://example.org/c\n"
    bad = CitationAnnotation(start_index=0, end_index=10000, url="https://example.org/a")
    result = _audit(text, [bad])
    assert result["unused_reference_numbers"] == [2]
    assert result["unsupported_reference_rows"] == 1
    assert result["annotation_coordinate_drift"] == 1
    assert not result["consistent"]


def test_distinct_www_origin_is_not_a_parser_repair() -> None:
    text = "Claim [1].\n\n## References\n**[1] Source** — https://example.org/a\n"
    annotation = CitationAnnotation(start_index=0, end_index=9, url="https://www.example.org/a")
    result = _audit(text, [annotation])
    assert result["protocol_only"] == result["references_only"] == 1
    assert result["unmapped_protocol_citations"] == 1




def test_year_in_source_title_is_visible_as_ambiguous_not_a_guessed_citation() -> None:
    result = _audit("Source title [2025].\n\n## References\nhttps://example.org/a\n")
    assert result["ambiguous_year_labels"] == [2025]
    assert result["dangling_reader_numbers"] == []
