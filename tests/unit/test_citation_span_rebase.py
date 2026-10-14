"""Persisted-coordinate citation span regression tests.

S92 m06 reset the failure policy encoded here. The rebase still refuses to
guess: a slice is bound only where it actually occurs inside the exact
persisted segment, never against wrapper or mission-context text. What changed
is the blast radius. One unmappable citation used to raise, and the worker
turned that raise into a hard ``blocked`` with no artifacts at all -- three
production missions (CAT-R006/R008/R009-S89) each burned a full paid run and
persisted nothing. Per-citation problems now degrade to a dropped annotation
with a structural reason; only a broken segment anchor still raises.
"""

from __future__ import annotations

import pytest

from deepsearch.report.citation import (
    CitationAnnotation,
    CitationSpanRebaseError,
    rebase_citation_annotations,
)


def _span(text: str, claim: str) -> tuple[int, int]:
    start = text.index(claim)
    return start, start + len(claim)


def _cite(text: str, claim: str, url: str) -> CitationAnnotation:
    start, end = _span(text, claim)
    return CitationAnnotation(start_index=start, end_index=end, url=url)


def test_rebases_multiple_citations_after_wrapper_and_preamble() -> None:
    """Every output slice indexes the exact wrapped artifact claim."""

    reference = (
        "# Agent draft\n\n"
        "RFC 9110 defines 307 as Temporary Redirect.\n\n"
        "The control endpoint deliberately returns 404.\n"
    )
    first_claim = "RFC 9110 defines 307 as Temporary Redirect."
    second_claim = "The control endpoint deliberately returns 404."
    first_start, first_end = _span(reference, first_claim)
    second_start, second_end = _span(reference, second_claim)
    annotations = [
        CitationAnnotation(
            start_index=first_start,
            end_index=first_end,
            url="https://www.rfc-editor.org/rfc/rfc9110",
            title="RFC 9110",
            live=True,
        ),
        CitationAnnotation(
            start_index=second_start,
            end_index=second_end,
            url="https://example.test/deliberate-404",
            live=False,
        ),
    ]
    segment = (
        "**Sources collected:** 2\n"
        "**Verified references:** 1\n"
        "**Agent steps:** 4 / 25\n"
        "**Runtime:** 8.0s\n\n"
        f"{first_claim}\n\n{second_claim}"
    )
    persisted = (
        "# Research Report: CANARY - Redirect behavior\n\n"
        f"{segment}\n\n"
        "---\n\n## Mission Context\n"
    )

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=annotations,
    )
    rebased = outcome.annotations

    assert outcome.dropped == []
    assert [entry.url for entry in rebased] == [entry.url for entry in annotations]
    assert [entry.title for entry in rebased] == ["RFC 9110", None]
    assert [entry.live for entry in rebased] == [True, False]
    assert rebased[0] is not annotations[0]
    assert persisted[rebased[0].start_index : rebased[0].end_index] == first_claim
    assert persisted[rebased[1].start_index : rebased[1].end_index] == second_claim


def test_same_claim_span_can_retain_multiple_source_citations() -> None:
    """Two sources on one uniquely rendered claim keep order and one span."""

    reference = "A single exact claim."
    annotations = [
        CitationAnnotation(start_index=0, end_index=len(reference), url="https://a"),
        CitationAnnotation(start_index=0, end_index=len(reference), url="https://b"),
    ]
    segment = f"Run preamble\n\n{reference}"
    persisted = f"# Wrapper\n\n{segment}\n"

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=annotations,
    )
    rebased = outcome.annotations

    assert outcome.dropped == []
    assert [entry.url for entry in rebased] == ["https://a", "https://b"]
    assert rebased[0].start_index == rebased[1].start_index
    assert rebased[0].end_index == rebased[1].end_index
    assert persisted[rebased[0].start_index : rebased[0].end_index] == reference


def test_same_start_nested_claim_spans_share_one_persisted_occurrence() -> None:
    """Clause and enclosing-sentence citations must not consume each other.

    The primary extractor emits this shape when one sentence carries a citation
    after an early clause and another after a later clause: the spans have the
    same source start but different ends. They describe one source occurrence,
    so both must rebase there even though their original spans are distinct.
    """

    short_claim = "Claim alpha"
    long_claim = "Claim alpha  and beta"
    reference = f"{long_claim}.\n"
    annotations = [
        CitationAnnotation(
            start_index=0,
            end_index=len(short_claim),
            url="https://a",
        ),
        CitationAnnotation(
            start_index=0,
            end_index=len(long_claim),
            url="https://b",
        ),
    ]
    segment = f"Run preamble\n\n{reference}"
    persisted = f"# Wrapper\n\n{segment}"

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=annotations,
    )

    assert outcome.dropped == []
    assert [entry.url for entry in outcome.annotations] == ["https://a", "https://b"]
    short, long = outcome.annotations
    assert short.start_index == long.start_index
    assert persisted[short.start_index : short.end_index] == short_claim
    assert persisted[long.start_index : long.end_index] == long_claim


def test_repeated_claim_text_binds_in_document_order() -> None:
    """Repeated claim text is ordinal, not ambiguous.

    A report that states a sentence twice and cites both occurrences is normal
    -- an inline References section makes it routine, which is exactly the
    shape that blocked CAT-R006/R008/R009-S89. The Nth source occurrence binds
    to the Nth persisted occurrence, so both citations survive with distinct,
    exact coordinates.
    """

    claim = "The response is cacheable by default."
    reference = f"{claim}\n\nSummary: {claim}"
    first = _span(reference, claim)
    second = (reference.rindex(claim), reference.rindex(claim) + len(claim))
    annotations = [
        CitationAnnotation(start_index=first[0], end_index=first[1], url="https://a"),
        CitationAnnotation(start_index=second[0], end_index=second[1], url="https://b"),
    ]
    segment = f"{claim}\n\nSummary: {claim}"
    persisted = f"# Wrapper\n\n{segment}\n\n## Context\n"

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=annotations,
    )

    assert outcome.dropped == []
    assert [entry.url for entry in outcome.annotations] == ["https://a", "https://b"]
    starts = [entry.start_index for entry in outcome.annotations]
    assert starts[0] < starts[1], "distinct occurrences, in document order"
    for entry in outcome.annotations:
        assert persisted[entry.start_index : entry.end_index] == claim


def test_removed_repeated_claim_does_not_reuse_consumed_occurrence() -> None:
    """A distinct later source span cannot collapse onto the first occurrence.

    When critique removes one of two identical claims, only one persisted
    occurrence remains. Rebinding both distinct source spans to that occurrence
    would attach the later citation to text whose source identity is no longer
    knowable. Keep the first ordinal match and drop the missing second one.
    """

    claim = "Same claim."
    reference = f"{claim}\n{claim}\n"
    first_start = reference.index(claim)
    second_start = reference.rindex(claim)
    annotations = [
        CitationAnnotation(
            start_index=first_start,
            end_index=first_start + len(claim),
            url="https://a",
        ),
        CitationAnnotation(
            start_index=second_start,
            end_index=second_start + len(claim),
            url="https://b",
        ),
    ]
    segment = f"{claim}\n"
    persisted = f"# Wrapper\n\n{segment}"

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=annotations,
    )

    assert [entry.url for entry in outcome.annotations] == ["https://a"]
    assert outcome.dropped == [
        {
            "annotation_index": 1,
            "reason": "source_span_missing_from_persisted_segment",
            "url": "https://b",
            "match_count": 0,
        }
    ]
    kept = outcome.annotations[0]
    assert persisted[kept.start_index : kept.end_index] == claim


def test_out_of_order_source_spans_still_bind_without_moving_the_cursor_back() -> None:
    """An annotation list emitted out of source order loses no citation."""

    later = "The later claim appears second in the body."
    earlier = "The earlier claim appears first in the body."
    reference = f"{earlier}\n\n{later}"
    annotations = [
        _cite(reference, later, "https://later"),
        _cite(reference, earlier, "https://earlier"),
    ]
    segment = reference
    persisted = f"# Wrapper\n\n{segment}\n"

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=annotations,
    )

    assert outcome.dropped == []
    assert [entry.url for entry in outcome.annotations] == [
        "https://later",
        "https://earlier",
    ]
    by_url = {entry.url: entry for entry in outcome.annotations}
    assert persisted[
        by_url["https://later"].start_index : by_url["https://later"].end_index
    ] == later
    assert persisted[
        by_url["https://earlier"].start_index : by_url["https://earlier"].end_index
    ] == earlier


def test_outer_whitespace_is_trimmed_rather_than_rejected() -> None:
    """A boundary space is an extractor artifact, so trim to the real claim.

    Rejecting it discarded whole reports for a defect that has an exactly
    correct repair: the trimmed slice is a *more* precise persisted coordinate,
    not a looser one.
    """

    claim = "Leading whitespace surrounds this claim."
    reference = f" {claim} "
    annotation = CitationAnnotation(
        start_index=0,
        end_index=len(reference),
        url="https://example.test/source",
    )
    segment = f"Preamble\n\n{claim}"
    persisted = f"# Wrapper\n\n{segment}\n"

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=[annotation],
    )

    assert outcome.dropped == []
    entry = outcome.annotations[0]
    assert persisted[entry.start_index : entry.end_index] == claim


def test_critique_removed_claim_drops_one_citation_and_keeps_the_rest() -> None:
    """A removed slice costs its own citation, never the surrounding report."""

    removed = "A claim the critique removed."
    kept = "A different supported claim."
    reference = f"{removed}\n\n{kept}"
    annotations = [
        _cite(reference, removed, "https://removed"),
        _cite(reference, kept, "https://kept"),
    ]
    segment = kept
    # Identical text elsewhere in the persisted wrapper must not be mistaken
    # for the removed body claim.
    persisted = f"# Wrapper: {removed}\n\n{kept}\n"

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=annotations,
    )

    assert [entry.url for entry in outcome.annotations] == ["https://kept"]
    assert outcome.dropped == [
        {
            "annotation_index": 0,
            "reason": "source_span_missing_from_persisted_segment",
            "url": "https://removed",
            "match_count": 0,
        }
    ]
    entry = outcome.annotations[0]
    assert persisted[entry.start_index : entry.end_index] == kept
    assert entry.start_index != persisted.index(removed)


@pytest.mark.parametrize(
    ("start", "end", "reason"),
    [
        (0, 0, "empty_source_span"),
        (0, 100, "source_span_out_of_bounds"),
    ],
)
def test_invalid_source_slice_is_dropped_with_its_reason(
    start: int, end: int, reason: str
) -> None:
    annotation = CitationAnnotation(
        start_index=start,
        end_index=end,
        url="https://example.test/source",
    )

    outcome = rebase_citation_annotations(
        reference_markdown="short",
        persisted_segment_markdown="short",
        persisted_markdown="# Wrapper\n\nshort\n",
        annotations=[annotation],
    )

    assert outcome.annotations == []
    assert outcome.dropped[0]["reason"] == reason
    assert outcome.dropped[0]["url"] == "https://example.test/source"


def test_overlong_claim_is_dropped_at_the_tracelab_boundary() -> None:
    """The 20k claim ceiling is a boundary contract, enforced by dropping."""

    claim = "x" * 20_001
    annotation = CitationAnnotation(
        start_index=0,
        end_index=len(claim),
        url="https://example.test/source",
    )

    outcome = rebase_citation_annotations(
        reference_markdown=claim,
        persisted_segment_markdown=claim,
        persisted_markdown=f"# Wrapper\n\n{claim}\n",
        annotations=[annotation],
    )

    assert outcome.annotations == []
    assert outcome.dropped[0]["reason"] == "source_span_too_long"


@pytest.mark.parametrize(
    ("segment", "persisted", "reason"),
    [
        ("", "# Wrapper\n", "persisted_segment_empty"),
        (
            "the exact segment",
            "# Wrapper\n\nsomething else entirely\n",
            "persisted_segment_missing_from_persisted_markdown",
        ),
        (
            "repeat",
            "# Wrapper\n\nrepeat\n\nrepeat\n",
            "persisted_segment_ambiguous_in_persisted_markdown",
        ),
    ],
)
def test_broken_segment_anchor_still_raises(
    segment: str, persisted: str, reason: str
) -> None:
    """No coordinate is trustworthy when the artifact lost its own segment.

    This is the one surviving raise. The caller handles it by shipping the
    report with zero citations rather than by discarding the report.
    """

    annotation = CitationAnnotation(
        start_index=0,
        end_index=6,
        url="https://example.test/source",
    )

    with pytest.raises(CitationSpanRebaseError) as exc:
        rebase_citation_annotations(
            reference_markdown="repeat",
            persisted_segment_markdown=segment,
            persisted_markdown=persisted,
            annotations=[annotation],
        )

    assert exc.value.reason == reason


def test_no_annotations_is_not_an_error() -> None:
    outcome = rebase_citation_annotations(
        reference_markdown="anything",
        persisted_segment_markdown="",
        persisted_markdown="",
        annotations=[],
    )

    assert outcome.annotations == []
    assert outcome.dropped == []


def test_report_scale_recurring_claim_ships_instead_of_dying() -> None:
    """One recurring sentence in a 40-citation report must not destroy it.

    Reports restate load-bearing sentences -- an executive summary that repeats
    a finding verbatim, a pitfalls register echoing a body claim. Under
    global-uniqueness matching the first such recurrence raised, and the worker
    converted that raise into a hard block with no artifact: CAT-R006/R008/
    R009-S89 each lost a completed 30+ minute paid run this way. Every citation
    must bind ordinally and the report must ship.
    """

    recurring = "Cutoff-time opacity is the dominant promise-credibility failure."
    body_claims = [
        f"Distributor {index} discloses a shipping cutoff time on the listing."
        for index in range(40)
    ]
    reference = (
        "# Draft\n\n"
        f"{recurring}\n\n"
        + "\n\n".join(body_claims)
        + f"\n\n{recurring}\n"
    )
    annotations = [
        CitationAnnotation(
            start_index=reference.index(recurring),
            end_index=reference.index(recurring) + len(recurring),
            url="https://example.test/summary",
        )
    ]
    annotations += [
        _cite(reference, claim, f"https://example.test/{index}")
        for index, claim in enumerate(body_claims)
    ]
    annotations.append(
        CitationAnnotation(
            start_index=reference.rindex(recurring),
            end_index=reference.rindex(recurring) + len(recurring),
            url="https://example.test/register",
        )
    )
    segment = reference.replace("# Draft\n\n", "Run preamble\n\n").strip()
    persisted = (
        f"# Research Report: CAT-R006-S89\n\n{segment}\n\n---\n\n## Mission Context\n"
    )

    assert segment.count(recurring) == 2, "fixture must prove the sentence recurs"

    outcome = rebase_citation_annotations(
        reference_markdown=reference,
        persisted_segment_markdown=segment,
        persisted_markdown=persisted,
        annotations=annotations,
    )

    assert outcome.dropped == [], "no citation may be lost in the normal shape"
    assert len(outcome.annotations) == len(annotations)
    for original, rebased in zip(annotations, outcome.annotations):
        expected = reference[original.start_index : original.end_index]
        assert persisted[rebased.start_index : rebased.end_index] == expected
    summary, register = outcome.annotations[0], outcome.annotations[-1]
    assert summary.start_index < register.start_index, "ordinal, not collapsed"
