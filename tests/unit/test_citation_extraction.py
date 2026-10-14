"""Tests for :func:`deepsearch.agent.citation_extraction.extract_citations`.

Covers the success-criteria checklist from the S67 plan doc:

* happy path (3 markers / 3 distinct URLs)
* multi-marker single claim (identical end_index, distinct URLs)
* zero markers (empty list, markdown unchanged)
* marker at end of string (no trailing whitespace)
* multi-byte / non-Latin character (codepoint indexing correctness)
* malformed markers (left as literal text)
* marker inside fenced or inline code block (NOT extracted)
* round-trip stability (length conservation)

S67 plan-doc decisions #517–#519 lock the marker syntax, canonical text, and
indexing semantics — this file is the layered runtime guard for those locks.
"""

from __future__ import annotations

import pytest

from deepsearch.agent.citation_extraction import (
    _REFERENCES_HEADING_RE,
    extract_citations,
)


def _reconstruct_marker_len(url: str) -> int:
    return len(f"[cite:{url}]")


class TestHappyPath:
    def test_three_distinct_markers(self):
        md = (
            "First claim. [cite:https://a.example]\n"
            "Second claim. [cite:https://b.example]\n"
            "Third claim. [cite:https://c.example]"
        )
        stripped, annotations = extract_citations(md)
        assert len(annotations) == 3
        urls = [a.url for a in annotations]
        assert urls == [
            "https://a.example",
            "https://b.example",
            "https://c.example",
        ]
        # Each cited span should be the claim text, no trailing whitespace.
        assert stripped[annotations[0].start_index : annotations[0].end_index] == "First claim."
        assert stripped[annotations[1].start_index : annotations[1].end_index] == "Second claim."
        assert stripped[annotations[2].start_index : annotations[2].end_index] == "Third claim."

    def test_each_annotation_has_url_citation_type(self):
        stripped, annotations = extract_citations("claim. [cite:https://a]")
        assert len(annotations) == 1
        assert annotations[0].type == "url_citation"

    def test_titles_are_none_at_extraction_time(self):
        _, annotations = extract_citations("claim. [cite:https://a]")
        assert annotations[0].title is None


class TestMultiMarkerSingleClaim:
    def test_two_markers_back_to_back(self):
        md = "claim [cite:URL1][cite:URL2]."
        stripped, annotations = extract_citations(md)
        assert len(annotations) == 2
        # Both annotations cover the same claim span.
        assert annotations[0].start_index == annotations[1].start_index
        assert annotations[0].end_index == annotations[1].end_index
        # Distinct URLs preserved in order.
        assert [a.url for a in annotations] == ["URL1", "URL2"]
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        assert cited == "claim"

    def test_three_markers_back_to_back(self):
        md = "claim [cite:U1][cite:U2][cite:U3]."
        stripped, annotations = extract_citations(md)
        assert len(annotations) == 3
        ends = {a.end_index for a in annotations}
        assert len(ends) == 1


class TestZeroMarkers:
    def test_no_markers_returns_input_unchanged(self):
        md = "Just prose with no citations at all.\nA second paragraph."
        stripped, annotations = extract_citations(md)
        assert stripped == md
        assert annotations == []

    def test_empty_string(self):
        stripped, annotations = extract_citations("")
        assert stripped == ""
        assert annotations == []


class TestMarkerAtEndOfString:
    def test_trailing_marker_no_trailing_whitespace(self):
        md = "claim text [cite:https://x]"
        stripped, annotations = extract_citations(md)
        assert len(annotations) == 1
        # No trailing whitespace included in span.
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        assert cited == "claim text"
        assert not cited.endswith(" ")

    def test_trailing_marker_no_following_period(self):
        md = "Bare claim [cite:https://x]"
        stripped, annotations = extract_citations(md)
        assert stripped == "Bare claim "
        assert stripped[annotations[0].start_index : annotations[0].end_index] == "Bare claim"


class TestMultiByteCharacters:
    def test_emoji_in_claim(self):
        md = "Sky is blue 🌍 [cite:https://earth]"
        stripped, annotations = extract_citations(md)
        assert len(annotations) == 1
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        assert cited == "Sky is blue 🌍"

    def test_cjk_character_in_claim(self):
        md = "天空是蓝色的 [cite:https://sky]"
        stripped, annotations = extract_citations(md)
        assert len(annotations) == 1
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        assert cited == "天空是蓝色的"

    def test_emoji_and_cjk_combined(self):
        md = "Greeting 你好 🌟 [cite:https://hello]"
        stripped, annotations = extract_citations(md)
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        assert cited == "Greeting 你好 🌟"

    def test_codepoint_indexing_not_byte_indexing(self):
        # Each emoji + CJK char is multiple UTF-8 bytes but exactly one Python
        # str codepoint. Confirm indices are codepoint-based.
        md = "你好世界 [cite:https://cjk]"
        stripped, annotations = extract_citations(md)
        # "你好世界" is 4 codepoints.
        assert annotations[0].end_index == 4


class TestMalformedMarkers:
    def test_empty_url(self):
        md = "claim [cite:] more text"
        stripped, annotations = extract_citations(md)
        assert annotations == []
        assert stripped == md

    def test_url_with_whitespace(self):
        md = "claim [cite:not a url] more text"
        stripped, annotations = extract_citations(md)
        assert annotations == []
        assert stripped == md

    def test_unclosed_marker(self):
        md = "claim [cite:https://x more text"
        stripped, annotations = extract_citations(md)
        assert annotations == []
        assert stripped == md

    def test_open_bracket_inside_url(self):
        md = "claim [cite:https://[x] more text"
        stripped, annotations = extract_citations(md)
        # Bracket inside URL fails the regex; literal stays.
        assert annotations == []
        assert stripped == md


class TestCodeBlockProtection:
    def test_marker_inside_inline_code(self):
        md = "Use the syntax `[cite:URL]` to cite sources."
        stripped, annotations = extract_citations(md)
        assert annotations == []
        assert stripped == md

    def test_marker_inside_fenced_code(self):
        md = (
            "Example:\n```\nclaim [cite:https://x]\n```\nEnd."
        )
        stripped, annotations = extract_citations(md)
        assert annotations == []
        assert stripped == md

    def test_marker_outside_code_still_extracted(self):
        md = (
            "Real claim. [cite:https://real]\n"
            "Code: `[cite:https://fake]` not a real cite."
        )
        stripped, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://real"
        # The fake one inside backticks survives in the stripped output.
        assert "`[cite:https://fake]`" in stripped

    def test_fenced_with_language_tag(self):
        md = "Demo:\n```python\nclaim [cite:https://x]\n```\n"
        stripped, annotations = extract_citations(md)
        assert annotations == []
        assert stripped == md


class TestRoundTripStability:
    def test_length_conservation_simple(self):
        md = "First claim. [cite:https://a] Second claim. [cite:https://b]"
        stripped, annotations = extract_citations(md)
        marker_lengths = sum(_reconstruct_marker_len(a.url) for a in annotations)
        assert len(stripped) + marker_lengths == len(md)

    def test_length_conservation_multi_marker(self):
        md = "claim [cite:URL1][cite:URL2]."
        stripped, annotations = extract_citations(md)
        marker_lengths = sum(_reconstruct_marker_len(a.url) for a in annotations)
        assert len(stripped) + marker_lengths == len(md)

    def test_length_conservation_with_multibyte(self):
        md = "天空 🌍 [cite:URL1] 真的蓝 [cite:URL2]"
        stripped, annotations = extract_citations(md)
        marker_lengths = sum(_reconstruct_marker_len(a.url) for a in annotations)
        # len() on a str returns codepoint count, which is what we want.
        assert len(stripped) + marker_lengths == len(md)

    def test_each_annotation_span_resolves_in_stripped(self):
        md = "First claim. [cite:U1] Second claim. [cite:U2]"
        stripped, annotations = extract_citations(md)
        for ann in annotations:
            assert 0 <= ann.start_index <= ann.end_index <= len(stripped)


class TestSentenceBoundaryWalkBack:
    def test_two_sentences_only_second_is_cited(self):
        md = "First sentence. Second claim. [cite:https://x]"
        stripped, annotations = extract_citations(md)
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        # The walk-back logic stops at "First sentence. ", so only "Second
        # claim." is in the cited span.
        assert cited == "Second claim."

    def test_newline_resets_cited_span(self):
        md = "Paragraph one.\nParagraph two claim [cite:https://x]"
        stripped, annotations = extract_citations(md)
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        assert cited == "Paragraph two claim"

    def test_inline_span_skips_indent_but_keeps_markdown_bullet(self):
        md = "Prior line.\n  - Nested cited claim [cite:https://x]"
        stripped, annotations = extract_citations(md)

        cited = stripped[annotations[0].start_index : annotations[0].end_index]

        assert cited == "- Nested cited claim"
        assert cited == cited.strip()

    def test_question_mark_terminator(self):
        md = "First? Second claim [cite:https://x]"
        stripped, annotations = extract_citations(md)
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        assert cited == "Second claim"

    def test_exclamation_terminator(self):
        md = "Wow! Second claim [cite:https://x]"
        stripped, annotations = extract_citations(md)
        cited = stripped[annotations[0].start_index : annotations[0].end_index]
        assert cited == "Second claim"


class TestOrphanMarkers:
    def test_orphan_url_still_extracted(self):
        # The extractor itself does NOT compare URLs against sources_collected
        # (that is the finalize-node's responsibility for orphan-marker
        # telemetry). The extractor's job is structural — emit the annotation
        # and let the caller decide if the URL is grounded.
        md = "claim [cite:https://orphan-url-not-in-retrieval]"
        stripped, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://orphan-url-not-in-retrieval"


class TestPureFunctionGuarantees:
    def test_no_external_io_imports(self):
        # The extractor must not import I/O or LLM modules — re-retrieval
        # triggered by extraction is an explicit roadmap §3.7 anti-pattern.
        # If a future change adds such an import, this test fails fast.
        import deepsearch.agent.citation_extraction as mod

        forbidden_modules = {
            "requests",
            "httpx",
            "urllib.request",
            "deepsearch.agent.tools.web_search",
            "deepsearch.agent.tools.source_fetch",
            "deepsearch.agent.tools.url_liveness",
            "deepsearch.agent.codeagent_runner",
            "deepsearch.agent.critique",
        }
        module_globals = set(getattr(mod, "__dict__", {}).keys())
        for forbidden in forbidden_modules:
            top = forbidden.split(".")[0]
            assert top not in module_globals or forbidden not in str(
                getattr(mod, top, "")
            ), f"citation_extraction must not import {forbidden}"

    def test_idempotent_on_clean_markdown(self):
        md = "Just prose with no markers."
        stripped1, _ = extract_citations(md)
        stripped2, _ = extract_citations(stripped1)
        assert stripped1 == stripped2 == md

    def test_running_twice_on_already_stripped_is_noop(self):
        md = "claim. [cite:https://x]"
        stripped1, _ = extract_citations(md)
        stripped2, annotations2 = extract_citations(stripped1)
        assert stripped2 == stripped1
        assert annotations2 == []


class TestNumberedFallback:
    """S74 m01 PD32 Path 2 — numbered-references fallback path.

    Triggers ONLY when the primary `[cite:URL]` pass yields zero annotations
    AND the markdown carries a `## References` / `## Sources` section.

    Under fallback the returned markdown is **unchanged** (numbered markers
    stay visible to the human reader); annotation indices reference the input
    markdown directly.
    """

    def test_fallback_resolves_numbered_marker_to_url(self):
        md = (
            "Claim about Workers [1].\n\n"
            "## References\n\n"
            "[1] Workers limits — https://developers.cloudflare.com/workers/platform/limits/\n"
        )
        stripped, annotations = extract_citations(md)
        # Markdown is unchanged under fallback (numbered markers stay visible).
        assert stripped == md
        assert len(annotations) == 1
        assert annotations[0].url == "https://developers.cloudflare.com/workers/platform/limits/"
        assert annotations[0].type == "url_citation"

    def test_fallback_cited_span_walks_back_to_sentence_start(self):
        md = (
            "First claim. Second claim about X [1].\n\n"
            "## References\n\n"
            "[1] Source — https://example.com\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        cited = md[annotations[0].start_index : annotations[0].end_index]
        assert cited == "Second claim about X"

    def test_fallback_span_skips_indent_but_keeps_blockquote_marker(self):
        md = (
            "  > Nested numbered claim [1].\n\n"
            "## References\n\n"
            "[1] Source — https://example.com\n"
        )
        _, annotations = extract_citations(md)

        cited = md[annotations[0].start_index : annotations[0].end_index]

        assert cited == "> Nested numbered claim"
        assert cited == cited.strip()

    def test_fallback_back_to_back_markers_yield_distinct_annotations(self):
        md = (
            "Multi-cited claim [1][2].\n\n"
            "## References\n\n"
            "[1] One — https://a.example\n"
            "[2] Two — https://b.example\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 2
        urls = {a.url for a in annotations}
        assert urls == {"https://a.example", "https://b.example"}

    def test_fallback_handles_dash_separator(self):
        md = (
            "Claim [1].\n\n"
            "## References\n\n"
            "[1] Title - https://example.com\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://example.com"

    def test_fallback_handles_url_only_entry(self):
        md = (
            "Claim [1].\n\n"
            "## References\n\n"
            "[1] https://example.com\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://example.com"

    def test_fallback_handles_markdown_link_form(self):
        md = (
            "Claim [1].\n\n"
            "## References\n\n"
            "[1] [Title](https://example.com)\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://example.com"

    def test_fallback_accepts_sources_heading_alias(self):
        md = (
            "Claim [1].\n\n"
            "## Sources\n\n"
            "[1] Title — https://example.com\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://example.com"

    def test_fallback_heading_is_case_insensitive(self):
        md = (
            "Claim [1].\n\n"
            "## REFERENCES\n\n"
            "[1] Title — https://example.com\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://example.com"

    def test_fallback_silently_drops_orphan_numbered_marker(self):
        md = (
            "Claim [5].\n\n"
            "## References\n\n"
            "[1] Title — https://example.com\n"
        )
        _, annotations = extract_citations(md)
        # [5] resolves to nothing in the references map; dropped silently.
        assert annotations == []

    def test_fallback_skips_marker_inside_fenced_code(self):
        md = (
            "```python\n"
            "x = arr[1]\n"
            "```\n"
            "Claim outside code [1].\n\n"
            "## References\n\n"
            "[1] Title — https://example.com\n"
        )
        _, annotations = extract_citations(md)
        # Only the [1] outside the fenced block annotates.
        assert len(annotations) == 1

    def test_fallback_skips_marker_inside_inline_code(self):
        md = (
            "Inline `arr[1]` is code. Claim outside [1].\n\n"
            "## References\n\n"
            "[1] Title — https://example.com\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1

    def test_fallback_does_not_re_extract_from_references_section(self):
        md = (
            "Claim [1].\n\n"
            "## References\n\n"
            "[1] Title — https://a.example\n"
            "[2] Title — https://b.example\n"
        )
        _, annotations = extract_citations(md)
        # Only the body [1] generates an annotation. The [1]/[2] inside the
        # References list itself are NOT counted as citations.
        assert len(annotations) == 1

    def test_fallback_returns_empty_when_no_references_section(self):
        md = (
            "Claim with numbered marker [1] but no references list.\n"
            "Just narrative text."
        )
        stripped, annotations = extract_citations(md)
        assert stripped == md
        assert annotations == []

    def test_inline_and_numbered_markers_both_contribute(self):
        # S96: a primary marker cannot hide separately numbered claims.
        md = (
            "Inline cite [cite:https://a.example]. Numbered [1].\n\n"
            "## References\n\n"
            "[1] Other — https://b.example\n"
        )
        stripped, annotations = extract_citations(md)
        # Inline marker stripped; numbered marker stays in stripped output
        # (it's part of the human-readable References surface).
        assert "[cite:" not in stripped
        assert "[1]" in stripped
        assert len(annotations) == 2
        assert [a.url for a in annotations] == ["https://a.example", "https://b.example"]

    def test_fallback_under_zero_inline_markers_emits_annotation(self):
        # Defensive — confirm `references == 0` lying-preamble bug is closed
        # for the failure mode that surfaced on S73-WIRING-02 (agent emitted
        # numbered references but skipped [cite:URL] markers).
        md = (
            "## Per-Entity Analyses\n\n"
            "### Cloudflare\n\n"
            "Workers caps CPU at 30s [1].\n"
            "Cold starts under 5ms [2].\n\n"
            "## References\n\n"
            "[1] Workers limits — https://developers.cloudflare.com/workers/platform/limits/\n"
            "[2] No-Container Computing — https://blog.cloudflare.com/cloud-computing-without-containers/\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 2
        distinct_urls = {a.url for a in annotations}
        assert len(distinct_urls) == 2

    def test_conflicting_reference_numbers_are_not_guessed(self):
        md = (
            "Claim [1].\n\n"
            "## References\n\n"
            "[1] First — https://first.example\n"
            "[1] Duplicate — https://second.example\n"
        )
        _, annotations = extract_citations(md)
        assert annotations == []

    def test_fallback_strips_trailing_punctuation_from_url(self):
        md = (
            "Claim [1].\n\n"
            "## References\n\n"
            "[1] Title — https://example.com.\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://example.com"

    def test_fallback_uses_last_references_heading_when_multiple(self):
        # If `## References` appears earlier in prose (e.g. in a section
        # describing how to cite), the LAST occurrence anchors the parse.
        md = (
            "## Methodology\n\n"
            "We discuss `## References` style below. Claim [1].\n\n"
            "## References\n\n"
            "[1] Real — https://real.example\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://real.example"


# ---------------------------------------------------------------------------
# S78 m01.a (PD79) — heading-regex relaxation. Tolerate optional numbered
# prefix (`## 12. Sources`) + accept `bibliography` as a defensive synonym.
# Closes the OODS-PLUGIN-REVIEW-01 gap-note cascade where the prior regex
# silently rejected a numbered heading and routed zero citations into the
# critique grounding check.
# ---------------------------------------------------------------------------


class TestReferencesHeadingRegex:
    """Parametrized regex tests for `_REFERENCES_HEADING_RE`.

    Positive cases (must match): flat headings, numbered prefixes, deeper
    heading levels, all three synonyms (references / sources / bibliography),
    case-insensitive matches.

    Negative cases (must NOT match): body-text mentions, headings whose
    label only happens to contain the synonym word as a prefix
    (`## Source pages`, `## Reference materials`).
    """

    @pytest.mark.parametrize(
        "heading",
        [
            # Flat — pre-S78 baseline still tolerated.
            "## References",
            "## Sources",
            # S78 m01.a defensive synonym.
            "## Bibliography",
            # S78 m01.a numbered prefix (load-bearing for OODS-PLUGIN-REVIEW-01).
            "## 12. Sources",
            "### 5 References",
            "## 1. Bibliography",
            "## 7 Sources",  # numeric prefix without trailing dot
            # Deeper heading levels.
            "### Sources",
            "#### Sources",
            "###### References",
            # Case-insensitive (`(?im)`).
            "## REFERENCES",
            "## sources",
            "## BIBLIOGRAPHY",
            # Trailing whitespace tolerated.
            "## Sources   ",
            # Leading indentation tolerated.
            "  ## Sources",
            # ----- S80 m01 (PD106) — leading non-digit token tolerance (Shape B). -----
            "## Formatted References",
            "## Cited Sources",
            "## My Bibliography",
            "## Section 9: References",
            "## Section 9: Bibliography",
            # ----- S80 m01 (PD106) — trailing token tolerance (Shape C). -----
            "## References List",
            "## References (Formatted)",
            "## Sources Used",
            "## Bibliography Index",
            # ----- S80 m01 (PD106) — combined leading+trailing (Shape A). -----
            # The post-S79 PLUGIN-BACKEND-AUTH-01 re-run failure case verbatim.
            "## 9. Formatted References List",
            # ----- S81 m02.a (PD123) — multi-word trailing tolerance. -----
            # The post-S80 STRUCTURED-OUTPUT-VIEWER-01 re-run failure case
            # verbatim. Pre-S81 the single-word trailing slot rejected this
            # heading; 58 numbered entries in the body, references=0.
            "## 11. Formatted references list with [N]-style numbered citations",
            # Em-dash separator between synonym and trailing phrase.
            "## 12. References — Cited Throughout",
            # Hyphen separator.
            "## References - Compiled Inline",
            # Multi-word trailing, no leading numeric prefix (Shape C path).
            "## References list compiled from primary sources",
            # Sources + multi-word trailing.
            "## Sources reviewed during this analysis",
        ],
    )
    def test_positive_match(self, heading: str):
        assert _REFERENCES_HEADING_RE.search(heading) is not None, (
            f"Heading {heading!r} should be recognized as a "
            "references/sources/bibliography section anchor"
        )

    @pytest.mark.parametrize(
        "non_heading",
        [
            # Label contains the synonym only as a prefix word.
            "## Source pages",
            "## Reference materials",
            "## Sources of error",
            # S81 m02.a (PD123) calibration trade-off documented:
            # `## References used in this study` now MATCHES because "used"
            # is not in the prep-prefix negation set (PD129 risk #3). The
            # multi-word trailing slot was widened to close the post-S80
            # STRUCTURED-OUTPUT-VIEWER-01 references=0 failure mode; the
            # cost is that body-prose compound shapes with non-prep first
            # tokens now register as headings. The agent's prompt nudge
            # (m02.b) reduces this false-positive rate at the source.
            # Body-text mentions, not headings.
            "We provide References below.",
            "See sources cited in the appendix.",
            # Heading with no leading hash.
            "References",
            # Wrong synonym.
            "## Citations",
            "## Bibliographic notes",
            # Hash without space.
            "##References",
            # ----- S80 m01 (PD106) — calibration boundary negatives. -----
            # Singular `Reference` is a different concept (architecture / design).
            "## Reference Architecture",
            # `synonym + of <noun>` body-prose shape — must not absorb a refs heading.
            "## Sources of Bias",
            "## Bibliography of Methods",
            # Embedded synonym inside a larger word — `\b` and `[A-Za-z]+` separator
            # block this from matching as `## ... Sources/Source`.
            "## Backsource Notes",
            # Calibration boundary call (Shape B forbids trailing words after the
            # synonym when a leading word is present, blocking this body-prose
            # shape from masquerading as a heading).
            "## My References Workflow",
            # ----- S81 m02.a (PD123) — prep-prefix negation calibration. -----
            # All trailing phrases starting with `for`, `about`, `from`, `by`,
            # `on`, `in` (in addition to S80's `of`) must reject after the
            # multi-word trailing slot is widened.
            "## References for Internal Reviewers",
            "## Sources about Bias Reduction",
            "## Bibliography from Primary Texts",
            "## References by Topic",
            "## Sources on Methodology",
            "## References in Chronological Order",
        ],
    )
    def test_negative_no_match(self, non_heading: str):
        assert _REFERENCES_HEADING_RE.search(non_heading) is None, (
            f"String {non_heading!r} must not be matched as a "
            "references/sources/bibliography heading"
        )

    def test_post_s79_plugin_backend_auth_01_load_bearing_replay(self):
        """End-to-end load-bearing assertion for PD106.

        Verbatim heading shape from the post-S79 PLUGIN-BACKEND-AUTH-01 re-run
        protocol (build_hash b6fe9976566d39b4…). Pre-S80 the extractor silently
        produced ``citations_extracted == 0`` because the regex rejected both
        the leading non-digit token (``Formatted``) AND the trailing token
        (``List``). Post-S80 the numbered fallback resolves both body markers
        against the references map and emits >0 annotations.
        """
        md = (
            "Backend identity model overview [1].\n"
            "Token rotation guidance [2].\n\n"
            "## 9. Formatted References List\n\n"
            "[1] OAuth 2.0 RFC — https://www.rfc-editor.org/rfc/rfc6749\n"
            "[2] Token rotation guide — https://auth0.com/docs/secure/tokens/refresh-tokens/refresh-token-rotation\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 2, (
            "S80 m01 PD106 — references-extractor must emit >0 annotations on "
            "the post-S79 failure heading shape `## 9. Formatted References List`"
        )
        urls = {a.url for a in annotations}
        assert urls == {
            "https://www.rfc-editor.org/rfc/rfc6749",
            "https://auth0.com/docs/secure/tokens/refresh-tokens/refresh-token-rotation",
        }

    def test_numbered_heading_routes_into_fallback(self):
        # End-to-end: confirm the regex relaxation actually unlocks the
        # numbered-fallback path that previously dropped citations on
        # OODS-PLUGIN-REVIEW-01. This is the load-bearing assertion.
        md = (
            "## Per-Entity Analyses\n\n"
            "Workers caps CPU at 30s [1].\n"
            "Cold starts under 5ms [2].\n\n"
            "## 12. Sources\n\n"
            "[1] Workers limits — https://developers.cloudflare.com/workers/platform/limits/\n"
            "[2] No-Container Computing — https://blog.cloudflare.com/cloud-computing-without-containers/\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 2
        urls = {a.url for a in annotations}
        assert urls == {
            "https://developers.cloudflare.com/workers/platform/limits/",
            "https://blog.cloudflare.com/cloud-computing-without-containers/",
        }

    def test_bibliography_heading_routes_into_fallback(self):
        md = (
            "Claim [1].\n\n"
            "## Bibliography\n\n"
            "[1] Title — https://example.com\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://example.com"


# ---------------------------------------------------------------------------
# S79 m01.c (PD95) — Unicode superscript citation marker fallback.
#
# Mission 1 PLUGIN-BACKEND-AUTH-01 emitted ¹–²² markers in body prose AND in
# the references list (`¹ Source title — URL`). The pre-S79 extractor missed
# both shapes and dropped 22 references on that mission. Refs-section context
# only — body-text exponents (`T¹⁰`) must NOT be misinterpreted as citation
# markers. PD102 verification expectation #5.
# ---------------------------------------------------------------------------


class TestUnicodeSuperscriptCitations:
    def test_unicode_superscript_marker_resolves_against_refs_map(self):
        md = (
            "PKCE adds a code challenge to the OAuth 2.0 authorization "
            "request flow¹.\n\n"
            "## References\n\n"
            "¹ RFC 7636 — https://www.rfc-editor.org/rfc/rfc7636\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 1
        assert annotations[0].url == "https://www.rfc-editor.org/rfc/rfc7636"

    def test_unicode_superscript_two_digit_number(self):
        # ``¹⁰`` (10), ``²²`` (22) — the exact range Mission 1 emitted.
        md = (
            "Claim with ten¹⁰. Another with twenty-two²².\n\n"
            "## References\n\n"
            "¹⁰ Tenth ref — https://example.com/ten\n"
            "²² Twenty-second ref — https://example.com/twentytwo\n"
        )
        _, annotations = extract_citations(md)
        urls = sorted(a.url for a in annotations)
        assert urls == [
            "https://example.com/ten",
            "https://example.com/twentytwo",
        ]

    def test_full_digit_range_extracted_in_references(self):
        # Touch every Unicode superscript digit ⁰–⁹ in the references list to
        # ensure the character class covers the full range.
        md_lines = ["Body claim¹."]
        md_lines.append("\n## References\n")
        # Ref 1 + Ref 2 are sufficient body-side; refs section adds entries
        # for every digit so the character-class regression is observable.
        for marker, url in (
            ("¹", "https://example.com/1"),
            ("²", "https://example.com/2"),
            ("³", "https://example.com/3"),
            ("⁴", "https://example.com/4"),
            ("⁵", "https://example.com/5"),
            ("⁶", "https://example.com/6"),
            ("⁷", "https://example.com/7"),
            ("⁸", "https://example.com/8"),
            ("⁹", "https://example.com/9"),
            ("⁰", "https://example.com/0"),  # ⁰ alone (number 0) — edge case
        ):
            md_lines.append(f"{marker} Ref — {url}")
        md = "\n".join(md_lines)
        _, annotations = extract_citations(md)
        # At least one body annotation resolved (¹ → URL #1).
        assert any(
            a.url == "https://example.com/1" for a in annotations
        ), "Body marker ¹ must resolve against Unicode-prefixed refs entry"

    def test_body_text_exponent_does_not_match(self):
        # Negative case from PD95: ``T¹⁰`` in body prose is an exponent, NOT
        # a citation marker. The negative lookbehind ``(?<!\\w)`` blocks the
        # match because ``T`` is a word character.
        md = (
            "The temperature variable T¹⁰ is described.\n\n"
            "## References\n\n"
            "¹⁰ Some ref — https://example.com/ten\n"
        )
        _, annotations = extract_citations(md)
        # Refs entry is parseable but the body exponent does NOT resolve to
        # an annotation — there is no isolated ¹⁰ in the body.
        assert annotations == []

    def test_unicode_superscript_outside_refs_section_no_annotations(self):
        # No references section means no map; isolated body markers don't
        # become annotations.
        md = "Claim with marker¹. More prose with ²."
        stripped, annotations = extract_citations(md)
        assert stripped == md
        assert annotations == []

    def test_unicode_marker_in_code_block_not_extracted(self):
        # Mirror the [N]-style behavior: markers inside fenced/inline code
        # are ignored.
        md = (
            "Discussion of `T¹` and `²` in a snippet.\n"
            "Real claim¹ with marker.\n\n"
            "## References\n\n"
            "¹ Some ref — https://example.com/ref\n"
        )
        _, annotations = extract_citations(md)
        # Only the body claim¹ outside code → 1 annotation.
        # (Inline code spans are excluded; the explicit body claim resolves.)
        assert len(annotations) == 1
        assert annotations[0].url == "https://example.com/ref"

    def test_mixed_bracket_and_unicode_refs_in_same_section(self):
        # A references section containing both `[1]` style AND `¹` style
        # entries builds a single map covering both syntaxes.
        md = (
            "Bracket claim [1]. Unicode claim².\n\n"
            "## References\n\n"
            "[1] Bracket ref — https://example.com/bracket\n"
            "² Unicode ref — https://example.com/unicode\n"
        )
        _, annotations = extract_citations(md)
        urls = sorted(a.url for a in annotations)
        assert urls == [
            "https://example.com/bracket",
            "https://example.com/unicode",
        ]

    def test_unicode_marker_with_no_matching_refs_entry_dropped(self):
        # Body emits ⁵ but refs only define ¹ — orphan marker silently dropped.
        md = (
            "Claim with marker⁵.\n\n"
            "## References\n\n"
            "¹ Real ref — https://example.com/ref-1\n"
        )
        _, annotations = extract_citations(md)
        assert annotations == []

    def test_inline_and_superscript_markers_both_contribute(self):
        # S96: Unicode numbered claims survive alongside primary markers.
        md = (
            "Inline cite [cite:https://primary.example]. Unicode¹.\n\n"
            "## References\n\n"
            "¹ Other — https://fallback.example\n"
        )
        _, annotations = extract_citations(md)
        assert len(annotations) == 2
        assert {a.url for a in annotations} == {
            "https://primary.example", "https://fallback.example"
        }
