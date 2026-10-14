"""S84 m02 invariants — quarantine downgrades, never deletes (decision #823).

WHY THIS FILE EXISTS: the S82/S83 quarantine was the AMPLIFIER of the CAT
damage chain — it deleted DEAD+HALLUCINATED citations and let critique rule 8
replace their claims with gap notes, stripping 45-91 markers per CAT run and
shipping heading-shell reports. m02 reverses the deletion half:

1. DEAD-with-collected-source citations are DOWNGRADED — kept in the body and
   in ``protocol.citations`` carrying ``live=False`` (the unverified flag),
   with their References entries annotated. Success criterion 2: a report
   with 11 false-dead citations retains all 11 claims flagged unverified,
   pinned end-to-end here.
2. Gap notes are reserved for claims with NO collected source at all
   (criterion 1 — no gap-note insertion keyed on DEAD verdicts with
   collected sources; the class guard lives inside strip_dead_cite_markers).
3. HALLUCINATED keeps hard quarantine; its References entry is annotated
   withdrawn so the three citation surfaces stay consistent (criterion 3 —
   the CAT-R004 defect class).
4. Everything new nests inside the already-declared ``synthesis_telemetry``
   channel — the S83-m02 declare-channel AST guard stays green by
   construction (criterion 5), pinned here.

New pins live in this NEW file: the six s82 invariant files are pinned at
exactly 70 collected cases by test_s83_m05_invariants.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.report.citation import CitationAnnotation


def _ann(url: str, live: bool | None) -> CitationAnnotation:
    return CitationAnnotation(
        type="url_citation", start_index=0, end_index=1, url=url, title=None, live=live
    )


# ---------------------------------------------------------------------------
# 1. The References annotation + consistency helper (unit level).
# ---------------------------------------------------------------------------


_BODY_WITH_REFS = (
    "# Report\n\n"
    "Claim about dead source [1]. Claim about live source [2].\n\n"
    "## References\n\n"
    "[1] Dead Title — https://dead.example/a\n"
    "[2] Live Title — https://live.example/c\n"
)


def test_downgraded_reference_entry_gets_unverified_flag() -> None:
    dead = _ann("https://dead.example/a", live=False)
    live = _ann("https://live.example/c", live=True)
    annotated, consistency = fnode._annotate_references_and_check_consistency(
        _BODY_WITH_REFS,
        kept_annotations=[dead, live],
        downgraded_dead=[dead],
        quarantined=[],
    )
    assert fnode._UNVERIFIED_REF_SUFFIX in annotated
    # Only the dead entry is flagged; the live entry is untouched.
    assert annotated.count(fnode._UNVERIFIED_REF_SUFFIX) == 1
    assert "[2] Live Title — https://live.example/c\n" in annotated
    assert consistency["consistent"] is True
    assert consistency["unverified_flagged"] == 1
    assert consistency["references_listed"] == 2
    assert consistency["protocol_citations"] == 2


def test_withdrawn_flag_for_quarantined_hallucinated_entry() -> None:
    """The hallucinated entry STAYS in the References list (body [n] anchors
    keep resolving — the CAT-R004 out-of-sync defect) but is annotated
    withdrawn, and the surfaces still count as consistent."""

    body = _BODY_WITH_REFS + "[3] Fabricated — https://fabricated.example/x\n"
    live = _ann("https://live.example/c", live=True)
    dead = _ann("https://dead.example/a", live=False)
    hallucinated = _ann("https://fabricated.example/x", live=False)
    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=[dead, live],
        downgraded_dead=[dead],
        quarantined=[hallucinated],
    )
    assert fnode._WITHDRAWN_REF_SUFFIX in annotated
    assert annotated.count(fnode._UNVERIFIED_REF_SUFFIX) == 1
    assert consistency["withdrawn_flagged"] == 1
    assert consistency["consistent"] is True, (
        "a withdrawn-annotated hallucinated entry is accounted for, not a mismatch"
    )


def test_annotation_is_idempotent() -> None:
    dead = _ann("https://dead.example/a", live=False)
    once, _ = fnode._annotate_references_and_check_consistency(
        _BODY_WITH_REFS, kept_annotations=[dead], downgraded_dead=[dead], quarantined=[]
    )
    twice, _ = fnode._annotate_references_and_check_consistency(
        once, kept_annotations=[dead], downgraded_dead=[dead], quarantined=[]
    )
    assert twice.count(fnode._UNVERIFIED_REF_SUFFIX) == 1


def test_missing_references_section_is_recorded_not_guessed() -> None:
    dead = _ann("https://dead.example/a", live=False)
    dead.title = "Dead Source"
    body = "# Report\n\nClaims without a references section.\n"
    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body, kept_annotations=[dead], downgraded_dead=[dead], quarantined=[]
    )
    assert annotated == body
    assert consistency["references_section_found"] is False
    assert consistency["consistent"] is False


def test_nested_annotated_source_list_belongs_to_references_section() -> None:
    """R2 shape: H3 under References must not hide the actual URL rows."""

    urls = [f"https://source.example/{index}" for index in range(30)]
    claims = "\n".join(
        f"Claim {index} is supported [{index}]."
        for index in range(1, 31)
    )
    body = (
        f"# Report\n\n{claims}\n\n## References\n### Annotated Source List\n\n"
        "- Primary and secondary sources follow.\n\n"
        + "\n".join(
            f"[{index}] Source {index} — {url}"
            for index, url in enumerate(urls, start=1)
        )
        + "\n"
    )
    annotations = []
    for index, url in enumerate(urls, start=1):
        start = body.index(f"Claim {index} is supported")
        marker = f"[{index}]"
        end = body.index(marker, start) + len(marker)
        annotations.append(
            CitationAnnotation(
                start_index=start,
                end_index=end,
                url=url,
                live=True,
            )
        )

    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=body,
        require_number_mapping=True,
    )

    assert annotated == body
    assert consistency["references_section_found"] is True
    assert consistency["references_listed"] == 30
    assert consistency["protocol_citations"] == 30
    assert consistency["protocol_only"] == 0
    assert consistency["numbered_references"] == 30
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is True


def test_exact_body_number_to_url_mapping_is_verified() -> None:
    body = (
        "# Report\n\nFirst claim [1]. Second claim [2].\n\n"
        "## References\n\n"
        "[1] First — https://source.example/first\n"
        "[2] Second — https://source.example/second\n"
    )
    first_start = body.index("First claim")
    first_end = body.index("[1]", first_start) + len("[1]")
    second_start = body.index("Second claim")
    second_end = body.index("[2]", second_start) + len("[2]")
    annotations = [
        CitationAnnotation(
            start_index=first_start,
            end_index=first_end,
            url="https://source.example/first",
            live=True,
        ),
        CitationAnnotation(
            start_index=second_start,
            end_index=second_end,
            url="https://source.example/second",
            live=True,
        ),
    ]

    _, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=body,
        require_number_mapping=True,
    )

    assert consistency["body_mapping_checked"] is True
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is True


def test_unnumbered_markdown_reference_uses_url_surface_consistency() -> None:
    """Machine citations need no invented numbered mapping when none exists."""

    url = "https://source.example/only"
    raw = (
        "# Report\n\nSupported claim [cite:https://source.example/only].\n\n"
        "## References\n\n"
        "- [Only source](https://source.example/only)\n"
    )
    stripped, annotations = fnode.extract_citations(raw)

    annotated, consistency = fnode._annotate_references_and_check_consistency(
        stripped,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=stripped,
        require_number_mapping=True,
    )

    assert annotated == stripped
    assert [annotation.url for annotation in annotations] == [url]
    assert consistency["numbered_references"] == 0
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is True


def test_swapped_body_number_mapping_fails_even_when_url_sets_match() -> None:
    body = (
        "# Report\n\nFirst claim [1]. Second claim [2].\n\n"
        "## References\n\n"
        "[1] Wrong first — https://source.example/second\n"
        "[2] Wrong second — https://source.example/first\n"
    )
    first_start = body.index("First claim")
    first_end = body.index("[1]", first_start) + len("[1]")
    second_start = body.index("Second claim")
    second_end = body.index("[2]", second_start) + len("[2]")
    annotations = [
        CitationAnnotation(
            start_index=first_start,
            end_index=first_end,
            url="https://source.example/first",
            live=True,
        ),
        CitationAnnotation(
            start_index=second_start,
            end_index=second_end,
            url="https://source.example/second",
            live=True,
        ),
    ]

    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=body,
        require_number_mapping=True,
    )

    assert annotated == body
    assert consistency["protocol_only"] == 0
    assert consistency["references_only"] == 0
    assert consistency["unmapped_protocol_citations"] == 2
    assert consistency["consistent"] is False


def test_post_rebase_audit_exposes_reference_without_surviving_annotation() -> None:
    """A dropped final annotation cannot leave stale green consistency."""

    body = (
        "# Report\n\nFirst claim [1]. Second claim [2].\n\n"
        "## References\n\n"
        "[1] First — https://source.example/first\n"
        "[2] Second — https://source.example/second\n"
    )
    first_start = body.index("First claim")
    first_end = body.index("[1]", first_start) + len("[1]")
    surviving = CitationAnnotation(
        start_index=first_start,
        end_index=first_end,
        url="https://source.example/first",
        live=True,
    )

    _, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=[surviving],
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=body,
        require_number_mapping=True,
    )

    assert consistency["protocol_only"] == 0
    assert consistency["references_only"] == 1
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is False


def test_same_sentence_mapping_uses_only_the_adjacent_number() -> None:
    """Earlier numbers in one annotation span cannot mask a swapped row."""

    body = (
        "# Report\n\n"
        "Alpha evidence [1], beta evidence [2], gamma evidence [3].\n\n"
        "## References\n\n"
        "[1] Alpha — https://source.example/alpha\n"
        "[2] Wrong beta — https://source.example/gamma\n"
        "[3] Wrong gamma — https://source.example/beta\n"
    )
    sentence_start = body.index("Alpha evidence")
    annotations = [
        CitationAnnotation(
            start_index=sentence_start,
            end_index=body.index(f"[{number}]", sentence_start) + 3,
            url=f"https://source.example/{name}",
            live=True,
        )
        for number, name in ((1, "alpha"), (2, "beta"), (3, "gamma"))
    ]

    _, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=body,
        require_number_mapping=True,
    )

    assert consistency["unmapped_protocol_citations"] == 2
    assert consistency["consistent"] is False


def test_superscript_reference_mapping_remains_supported() -> None:
    """The consistency audit accepts the extractor's Unicode fallback."""

    body = (
        "# Report\n\nFirst claim¹. Tenth claim¹⁰.\n\n"
        "## References\n\n"
        "¹ First — https://source.example/first\n"
        "¹⁰ Tenth — https://source.example/tenth\n"
    )
    _, annotations = fnode.extract_citations(body)

    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=body,
        require_number_mapping=True,
        numbered_fallback=True,
    )

    assert annotated == body
    assert len(annotations) == 2
    assert consistency["numbered_references"] == 2
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is True


def test_multiple_trailing_superscripts_share_primary_citation_coordinate() -> None:
    """Every source in ``claim¹ ² [cite:A][cite:B]`` maps exactly."""

    raw = (
        "# Report\n\nShared claim¹ ² "
        "[cite:https://source.example/first]"
        "[cite:https://source.example/second].\n\n"
        "## References\n\n"
        "¹ First — https://source.example/first\n"
        "² Second — https://source.example/second\n"
    )
    stripped, annotations = fnode.extract_citations(raw)

    _, consistency = fnode._annotate_references_and_check_consistency(
        stripped,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=stripped,
        require_number_mapping=True,
    )

    assert len(annotations) == 2
    assert annotations[0].end_index == annotations[1].end_index
    assert consistency["numbered_references"] == 2
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is True


def test_primary_citation_owns_trailing_number_before_adjacent_suffix() -> None:
    """A's [1] must win over B's adjacent [2] after machine-marker removal."""

    raw = (
        "# Report\n\nShared claim [1][cite:https://source.example/first]"
        "[2][cite:https://source.example/second].\n\n"
        "## References\n\n"
        "[1] First — https://source.example/first\n"
        "[2] Second — https://source.example/second\n"
    )
    stripped, annotations = fnode.extract_citations(raw)

    _, consistency = fnode._annotate_references_and_check_consistency(
        stripped,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=stripped,
        require_number_mapping=True,
    )

    assert stripped.startswith("# Report\n\nShared claim [1][2].")
    assert len(annotations) == 2
    assert stripped[annotations[0].start_index : annotations[0].end_index].endswith(
        "[1]"
    )
    assert stripped[annotations[0].end_index :].startswith("[2]")
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is True


def test_primary_trailing_number_still_rejects_swapped_adjacent_row() -> None:
    """Suffix adjacency cannot make a genuinely swapped first row look valid."""

    raw = (
        "# Report\n\nShared claim [1][cite:https://source.example/first]"
        "[2][cite:https://source.example/second].\n\n"
        "## References\n\n"
        "[1] Wrong first — https://source.example/second\n"
        "[2] Wrong second — https://source.example/first\n"
    )
    stripped, annotations = fnode.extract_citations(raw)

    _, consistency = fnode._annotate_references_and_check_consistency(
        stripped,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=stripped,
        require_number_mapping=True,
    )

    assert consistency["protocol_only"] == 0
    assert consistency["references_only"] == 0
    assert consistency["unmapped_protocol_citations"] == 1
    assert consistency["consistent"] is False


def test_primary_superscript_owns_trailing_number_before_adjacent_suffix() -> None:
    """Unicode reader markers follow the same primary ownership rule."""

    raw = (
        "# Report\n\nShared claim¹[cite:https://source.example/first]"
        " ²[cite:https://source.example/second].\n\n"
        "## References\n\n"
        "¹ First — https://source.example/first\n"
        "² Second — https://source.example/second\n"
    )
    stripped, annotations = fnode.extract_citations(raw)

    _, consistency = fnode._annotate_references_and_check_consistency(
        stripped,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=stripped,
        require_number_mapping=True,
    )

    assert stripped.startswith("# Report\n\nShared claim¹ ².")
    assert stripped[annotations[0].start_index : annotations[0].end_index].endswith(
        "¹"
    )
    assert stripped[annotations[0].end_index :].startswith(" ²")
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is True


def test_numbered_fallback_owns_suffix_after_prior_adjacent_marker() -> None:
    """Fallback B owns suffix [2], even though its span ends after A's [1]."""

    body = (
        "# Report\n\nShared claim [1][2].\n\n"
        "## References\n\n"
        "[1] First — https://source.example/first\n"
        "[2] Second — https://source.example/second\n"
    )
    _, annotations = fnode.extract_citations(body)

    _, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=annotations,
        downgraded_dead=[],
        quarantined=[],
        citation_reference_markdown=body,
        require_number_mapping=True,
        numbered_fallback=True,
    )

    assert len(annotations) == 2
    assert body[annotations[1].start_index : annotations[1].end_index].endswith(
        "[1]"
    )
    assert body[annotations[1].end_index :].startswith("[2]")
    assert consistency["unmapped_protocol_citations"] == 0
    assert consistency["consistent"] is True


def test_quarantined_rows_and_exact_mappings_are_required() -> None:
    """Quarantine is an audited surface, not an allowance for stale rows."""

    missing_body = "# Report\n\nWithdrawn claim [1].\n\n## References\n\n"
    missing = CitationAnnotation(
        start_index=missing_body.index("Withdrawn claim"),
        end_index=missing_body.index("[1]") + len("[1]"),
        url="https://fabricated.example/missing",
        live=False,
    )
    _, missing_consistency = fnode._annotate_references_and_check_consistency(
        missing_body,
        kept_annotations=[],
        downgraded_dead=[],
        quarantined=[missing],
        citation_reference_markdown=missing_body,
        require_number_mapping=True,
    )
    assert missing_consistency["quarantined_missing"] == 1
    assert missing_consistency["unmapped_quarantined_citations"] == 1
    assert missing_consistency["consistent"] is False

    swapped_body = (
        "# Report\n\nFirst withdrawn [1]. Second withdrawn [2].\n\n"
        "## References\n\n"
        "[1] Wrong first — https://fabricated.example/second\n"
        "[2] Wrong second — https://fabricated.example/first\n"
    )
    quarantined = [
        CitationAnnotation(
            start_index=swapped_body.index("First withdrawn"),
            end_index=swapped_body.index("[1]") + len("[1]"),
            url="https://fabricated.example/first",
            live=False,
        ),
        CitationAnnotation(
            start_index=swapped_body.index("Second withdrawn"),
            end_index=swapped_body.index("[2]") + len("[2]"),
            url="https://fabricated.example/second",
            live=False,
        ),
    ]
    _, swapped_consistency = fnode._annotate_references_and_check_consistency(
        swapped_body,
        kept_annotations=[],
        downgraded_dead=[],
        quarantined=quarantined,
        citation_reference_markdown=swapped_body,
        require_number_mapping=True,
    )
    assert swapped_consistency["quarantined_missing"] == 0
    assert swapped_consistency["unmapped_quarantined_citations"] == 2
    assert swapped_consistency["consistent"] is False


def test_read_only_audit_requires_persisted_reference_flags() -> None:
    """The final audit observes bytes; it cannot add hypothetical warnings."""

    body = (
        "# Report\n\nDead claim [1]. Withdrawn claim [2].\n\n"
        "## References\n\n"
        "[1] Dead — https://dead.example/a\n"
        "[2] Fabricated — https://fabricated.example/x\n"
    )
    dead = CitationAnnotation(
        start_index=body.index("Dead claim"),
        end_index=body.index("[1]") + len("[1]"),
        url="https://dead.example/a",
        live=False,
    )
    hallucinated = CitationAnnotation(
        start_index=body.index("Withdrawn claim"),
        end_index=body.index("[2]") + len("[2]"),
        url="https://fabricated.example/x",
        live=False,
    )

    audited, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=[dead],
        downgraded_dead=[dead],
        quarantined=[hallucinated],
        citation_reference_markdown=body,
        require_number_mapping=True,
        annotate_flags=False,
    )

    assert audited == body
    assert consistency["missing_unverified_flags"] == 1
    assert consistency["missing_withdrawn_flags"] == 1
    assert consistency["unverified_flagged"] == 0
    assert consistency["withdrawn_flagged"] == 0
    assert consistency["consistent"] is False


def test_annotation_stops_at_next_heading() -> None:
    body = (
        _BODY_WITH_REFS
        + "\n## Appendix\n\nMention of https://dead.example/a outside the refs list.\n"
    )
    dead = _ann("https://dead.example/a", live=False)
    annotated, _ = fnode._annotate_references_and_check_consistency(
        body, kept_annotations=[dead], downgraded_dead=[dead], quarantined=[]
    )
    appendix = annotated.split("## Appendix")[1]
    assert fnode._UNVERIFIED_REF_SUFFIX not in appendix


def test_parenthesised_url_reference_entry_still_gets_flagged() -> None:
    """ADVERSARIAL-REVIEW FIX PIN: a Wikipedia-style URL with balanced
    parentheses must survive capture intact — pre-fix the regex truncated at
    '(' exclusion, the keys never matched, and a downgraded-dead entry
    shipped with NO unverified flag plus a false consistent=False."""

    url = "https://en.wikipedia.org/wiki/Foo_(disambiguation)"
    body = (
        "# Report\n\nClaim [1].\n\n## References\n\n"
        f"[1] Wiki — {url}\n"
    )
    dead = _ann(url, live=False)
    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body, kept_annotations=[dead], downgraded_dead=[dead], quarantined=[]
    )
    assert annotated.count(fnode._UNVERIFIED_REF_SUFFIX) == 1
    assert consistency["consistent"] is True
    assert consistency["unverified_flagged"] == 1

    # And markdown-link syntax sheds its UNBALANCED closing paren.
    linked_body = (
        "# Report\n\nClaim [1].\n\n## References\n\n"
        "[1] [Wiki](https://plain.example.com/page)\n"
    )
    plain = _ann("https://plain.example.com/page", live=False)
    annotated2, consistency2 = fnode._annotate_references_and_check_consistency(
        linked_body, kept_annotations=[plain], downgraded_dead=[plain], quarantined=[]
    )
    assert annotated2.count(fnode._UNVERIFIED_REF_SUFFIX) == 1
    assert consistency2["consistent"] is True


def test_interleaved_wrapper_punctuation_tails_fully_unwrap() -> None:
    """RE-VERIFY HARDENING PIN: _clean_ref_url strips to a fixed point —
    bold-then-comma (**url**,) and punctuation-then-paren (url.)) interleaves
    fully unwrap; single-pass stripping left these entries unflagged with a
    false consistent=False."""

    assert fnode._clean_ref_url("https://x.com/p**,") == "https://x.com/p"
    assert fnode._clean_ref_url("https://x.com/p.)") == "https://x.com/p"
    assert (
        fnode._clean_ref_url("https://en.wikipedia.org/wiki/Foo_(bar)")
        == "https://en.wikipedia.org/wiki/Foo_(bar)"
    )

    url = "https://bold.example.com/page"
    body = (
        "# Report\n\nClaim [1].\n\n## References\n\n"
        f"[1] Bold — **{url}**,\n"
    )
    dead = _ann(url, live=False)
    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body, kept_annotations=[dead], downgraded_dead=[dead], quarantined=[]
    )
    assert annotated.count(fnode._UNVERIFIED_REF_SUFFIX) == 1
    assert consistency["consistent"] is True


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/search?q=what?",
        "https://en.wikipedia.org/wiki/Foo_(bar)",
    ],
)
def test_legal_terminal_url_characters_preserve_reference_identity(url: str) -> None:
    """Legal query punctuation and balanced path delimiters are not prose."""

    expected = {fnode._normalise_url_for_cache(url)}
    assert fnode._clean_ref_url(url, expected_urls=expected) == url

    body = (
        "# Report\n\nClaim [1].\n\n## References\n\n"
        f"[1] Exact reference — {url}\n"
    )
    annotation = _ann(url, live=True)
    _, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=[annotation],
        downgraded_dead=[],
        quarantined=[],
    )

    assert consistency["protocol_only"] == 0
    assert consistency["references_only"] == 0
    assert consistency["consistent"] is True


@pytest.mark.parametrize(
    ("reference_url", "protocol_url"),
    [
        ("https://example.com", "https://example.com/"),
        ("https://example.com:443/path", "https://example.com/path"),
        ("https://example.com/a/../path", "https://example.com/path"),
    ],
)
def test_reference_consistency_uses_tracelab_canonical_url_identity(
    reference_url: str,
    protocol_url: str,
) -> None:
    """Equivalent URL spellings cannot split protocol and reference surfaces."""

    body = (
        "# Report\n\nClaim [1].\n\n## References\n\n"
        f"[1] Equivalent reference — {reference_url}\n"
    )
    dead = _ann(protocol_url, live=False)
    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=[dead],
        downgraded_dead=[dead],
        quarantined=[],
    )

    assert annotated.count(fnode._UNVERIFIED_REF_SUFFIX) == 1
    assert consistency["protocol_only"] == 0
    assert consistency["references_only"] == 0
    assert consistency["consistent"] is True


def test_mixed_class_reference_line_gets_both_flags() -> None:
    """ADVERSARIAL-REVIEW FIX PIN: a nonstandard line carrying a downgraded
    AND a quarantined URL gets BOTH suffixes — the elif first-match-wins
    silently dropped the withdrawn flag."""

    body = (
        "# Report\n\nClaim [1].\n\n## References\n\n"
        "[1] Combo — https://dead.example/a (mirror: https://fabricated.example/x)\n"
    )
    dead = _ann("https://dead.example/a", live=False)
    hallucinated = _ann("https://fabricated.example/x", live=False)
    annotated, consistency = fnode._annotate_references_and_check_consistency(
        body,
        kept_annotations=[dead],
        downgraded_dead=[dead],
        quarantined=[hallucinated],
    )
    assert fnode._UNVERIFIED_REF_SUFFIX in annotated
    assert fnode._WITHDRAWN_REF_SUFFIX in annotated
    assert consistency["unverified_flagged"] == 1
    assert consistency["withdrawn_flagged"] == 1
    assert consistency["reference_number_collisions"] == 1
    assert consistency["consistent"] is False


def test_flag_suffixes_are_not_the_gap_note() -> None:
    """The References flags must never reuse the gap-note literal — gap notes
    are reserved for claims with no collected source at all (criterion 1)."""

    assert "[Data unavailable in primary sources]" not in fnode._UNVERIFIED_REF_SUFFIX
    assert "[Data unavailable in primary sources]" not in fnode._WITHDRAWN_REF_SUFFIX
    assert "unverified" in fnode._UNVERIFIED_REF_SUFFIX
    assert "withdrawn" in fnode._WITHDRAWN_REF_SUFFIX


# ---------------------------------------------------------------------------
# 2. End-to-end: 11 false-dead citations survive downgraded (criterion 2).
# ---------------------------------------------------------------------------

_ELEVEN_DEAD_URLS = [f"https://dead{i}.example/page" for i in range(11)]


def _state() -> dict:
    return {
        "mission_id": "S84-M02-E2E",
        "mission_objectives": ["c1"],
        "mission_context": {
            "title": "Downgrade Mission",
            "objective": "verify downgrade-not-delete",
            "deliverables": ["Comparison Matrix"],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _synthesis_eleven_dead() -> SynthesisResult:
    claims = " ".join(
        f"Claim {i} stands on collected evidence [{i + 1}][cite:{url}] with enough "
        "surrounding words to be a real sentence."
        for i, url in enumerate(_ELEVEN_DEAD_URLS)
    )
    refs = "\n".join(
        f"[{i + 1}] Source {i} — {url}" for i, url in enumerate(_ELEVEN_DEAD_URLS)
    )
    return SynthesisResult(
        output_markdown=f"# Report\n\n{claims}\n\n## References\n\n{refs}\n",
        step_count=5,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=10.0,
        token_usage={"input": 100, "output": 100, "total": 200},
        tool_call_summary={"total_calls": 3},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=[
            {"url": url, "title": f"S{i}", "snippet": "x"}
            for i, url in enumerate(_ELEVEN_DEAD_URLS)
        ],
        error=None,
    )


def _fake_liveness_all_dead():
    async def _run(citations, sources_collected, budget_seconds=45.0, scope_admission=None):
        for ann in citations:
            ann.live = False
        telemetry = LivenessTelemetry(
            checked=len(citations),
            live=0,
            dead=len(citations),
            orphan=0,
            unknown=0,
            hallucinated_url_count=0,
            distinct_urls_checked=len(citations),
        )
        return list(citations), telemetry

    return _run


def test_eleven_false_dead_citations_survive_downgraded(tmp_path, monkeypatch) -> None:
    """Success criterion 2 pinned end-to-end: a report whose ENTIRE citation
    set is marked dead ships all 11 claims, all 11 protocol citations
    (live=False = unverified), all 11 References entries flagged — and not
    one gap note. Pre-m02 this exact input shipped a husk."""

    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    captured: dict = {}

    def _capture_protocol(protocol, **kwargs):
        captured["citations"] = list(protocol.citations)
        return str(tmp_path / "protocol.json")

    def _capture_report(content, **kwargs):
        captured["report"] = content
        return str(tmp_path / "report.md")

    critique = CritiqueResult(
        output_markdown="",  # passthrough — final_markdown falls back to stripped
        claim_count=0,
        gap_notes_inserted=0,
        runtime_seconds=0.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        error=None,
    )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=_synthesis_eleven_dead()
        ), patch.object(fnode, "run_critique", return_value=critique), patch.object(
            fnode, "check_citations_liveness", new=_fake_liveness_all_dead()
        ), patch.object(
            fnode, "save_protocol", side_effect=_capture_protocol
        ), patch.object(
            fnode, "save_report", side_effect=_capture_report
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    # All 11 citations ship, every one carrying the unverified flag.
    shipped = captured["citations"]
    assert {c.url for c in shipped} == set(_ELEVEN_DEAD_URLS)
    assert all(c.live is False for c in shipped)

    # All 11 claims survive in the persisted body; zero gap notes.
    report = captured["report"]
    for i in range(11):
        assert f"Claim {i} stands on collected evidence" in report
    assert "[Data unavailable in primary sources]" not in report
    assert "[cite:" not in report

    # All 11 References entries flagged unverified.
    assert report.count(fnode._UNVERIFIED_REF_SUFFIX) == 11

    # Telemetry: downgraded count + consistency, nested in the declared channel.
    s_tel = result["synthesis_telemetry"]
    assert s_tel["markers_quarantined"] == 11  # downgraded, not deleted
    assert s_tel["hallucinated_quarantined"] == 0
    consistency = s_tel["citation_consistency"]
    assert consistency["consistent"] is True
    assert consistency["unverified_flagged"] == 11
    assert consistency["protocol_citations"] == 11
    assert consistency["references_listed"] == 11


def test_consistency_dict_is_nested_not_a_new_channel() -> None:
    """Criterion 5: everything m02 writes nests inside the already-declared
    synthesis_telemetry channel — no new top-level node-return key (the
    S83-m02 declare-channel AST guard enforces the general rule; this pins
    the m02-specific key placement)."""

    import inspect

    src = inspect.getsource(fnode)
    assert 'synthesis_telemetry["citation_consistency"]' in src
    assert 'result["citation_consistency"]' not in src


def test_finalize_passes_sources_to_annotate_mode_strip() -> None:
    """Criterion 1 wiring: the finalize strip call carries sources_collected,
    arming the class guard that makes gap-noting a collected-source URL
    structurally impossible."""

    import inspect

    src = inspect.getsource(fnode)
    strip_idx = src.index("strip_dead_cite_markers(")
    window = src[strip_idx : strip_idx + 220]
    assert "sources_collected" in window
