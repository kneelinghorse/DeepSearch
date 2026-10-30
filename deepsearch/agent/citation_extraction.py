"""Pure-function extractor for ``[cite:URL]`` inline citation markers.

S67 plan-doc decision #517 fixes the inline-marker syntax at ``[cite:URL]``.
This module exposes :func:`extract_citations`, which does a single regex pass
over the post-critique markdown, strips every well-formed marker, and returns
the cleaned markdown alongside a list of :class:`CitationAnnotation` whose
``start_index`` / ``end_index`` are Python-slice codepoint offsets into the
stripped output (decision #519).

S96 reconciles primary markers and numbered references on the same stripped
text. Reader markers remain unchanged. Duplicate claim-position/URL pairs
prefer primary coordinates; each annotation privately retains marker ownership.
Conflicting reference numbers are unresolved rather than guessed. S100 reads
an entry whose links all name one source (an arXiv abstract and its HTML page)
as one entry. S103 reads a superscript as a reader marker only in a report that
cites that way (:func:`reads_superscripts`).

No I/O, no LLM calls — the extractor is deliberately pure-function so that
re-retrieval triggered by extraction (roadmap §3.7 anti-pattern) is structurally
impossible.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from deepsearch.report.citation import CitationAnnotation

from .url_utils import arxiv_identity

# Marker regex: ``[cite:URL]`` where URL contains no whitespace and no brackets.
# Whitespace exclusion implicitly rejects ``[cite:not a url]`` (left as literal).
# ``+?`` requires ≥1 char, so ``[cite:]`` (empty URL) does not match.
# A missing closing ``]`` (``[cite:URL``) does not match either.
_CITE_MARKER_RE = re.compile(r"\[cite:([^\[\]\s]+?)\]")

# Matching delimiter runs also protect nested backticks inside inline code.
_INLINE_CODE_RE = re.compile(r"(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)")

# S74 numbered fallback. The agent's Output Format §7 requires a `## References`
# section (case-tolerated; `## Sources` accepted as an alias). Match the heading
# at the start of a line.
#
# S78 m01.a (PD79): tolerate an optional numbered prefix (`## 12. Sources`,
# `### 5 References`) and accept `bibliography` as a defensive synonym. The
# OODS-PLUGIN-REVIEW-01 forensic deep-dive (S77 m04) found the prior regex
# silently failed on numbered prefixes, routing zero citations into the
# critique grounding check and triggering an 11-gap-note cascade.
#
# S80 m01 (PD106): generalize the heading regex to tolerate optional leading
# word(s) AND optional trailing token around the synonym anchor.
#
# S81 m02.a (PD123): widen the Shape A + Shape C trailing slot to tolerate
# multi-word descriptive trailing phrases. STRUCTURED-OUTPUT-VIEWER-01
# post-S80 emitted `## 11. Formatted references list with [N]-style numbered
# citations` and the S80 single-word trailing slot rejected it (58 numbered
# entries in the body, references=0 in metadata). The trailing slot now
# accepts any token sequence not starting with a known prepositional
# preposition (`of`/`for`/`about`/`from`/`by`/`on`/`in`) so prose-shape
# false positives like `## Sources of Bias`, `## Bibliography of Methods`,
# and `## References for Internal Reviewers` continue to reject. An optional
# em-dash / en-dash / hyphen separator handles `## 12. References — Cited
# Throughout` style headings.
#
# Three explicit shapes keep the false-positive surface tight:
#
#   Shape A — digit prefix REQUIRED (with optional word), synonym, optional
#             trailing phrase (single token, multi-word, paren-wrapped, or
#             em-dash-separated)
#     matches: `## 12. Sources`, `### 5 References`,
#              `## 9. Formatted References List`,
#              `## 11. Formatted references list with [N]-style numbered citations`,
#              `## 12. References — Cited Throughout`
#
#   Shape B — word prefix REQUIRED (optional `Word N:`), synonym, NO trailing
#     matches: `## Formatted References`, `## My Bibliography`,
#              `## Cited Sources`, `## Section 9: References`
#
#   Shape C — no leading prefix, synonym, optional trailing phrase (single
#             token, multi-word, paren-wrapped, or em-dash-separated)
#     matches: `## References`, `## References List`, `## Sources Used`,
#              `## Bibliography Index`, `## References (Formatted)`
#
# Calibration boundary documented (PD113 risks #1 + #2 + S81 PD129 risk #3):
#  * Trailing `of <noun>` shape (`## Sources of Bias`, `## Bibliography of Methods`)
#    REJECTED — the multi-word trailing alternative carries a negative
#    lookahead `(?!(?:of|for|about|from|by|on|in)\b)` on the next token.
#  * `## References for Internal Reviewers` REJECTED — `for` is in the
#    prep-prefix negation set.
#  * `## My References Workflow` REJECTED — Shape B's word-prefix path
#    forbids trailing tokens, blocking the symmetric word + synonym + word
#    body-prose shape from masquerading as a heading.
#  * `## Reference Architecture`, `## Backsource Notes`, `## Bibliographic
#    notes` REJECTED — synonym shape mismatch (singular `reference`,
#    embedded `back-source`, `bibliographic` adjective). Word-boundary `\b`
#    after the synonym pins the suffix.
#  * Compound shapes whose trailing first token is NOT in the prep-prefix
#    negation set (`## References Across Time`) DO match. Documented
#    trade-off per S81 PD129 risk #3 — the negation set captures the
#    dominant English prepositions but is not exhaustive. Future calibration
#    extends the negation set if needed.
#
# Synonyms remain `(?:references|sources|bibliography)`. New synonyms (e.g.
# `works cited`) are deliberately deferred per PD112 — narrowing blast radius.
_REFERENCES_HEADING_RE = re.compile(
    r"(?im)"
    r"(?:"
    # Shape A — digit prefix REQUIRED, optional single word, optional trailing.
    r"^[ \t]*#{1,6}[ \t]+"
    r"\d+[.:]?[ \t]+(?:[A-Za-z]+[ \t]+)?"
    r"(?:references|sources|bibliography)\b"
    r"(?:[ \t]+(?:[—–\-][ \t]+)?(?:\([^)\n]+\)|(?!(?:of|for|about|from|by|on|in)\b)[A-Za-z][^\n]*?))?"
    r"[ \t]*$"
    r"|"
    # Shape B — word prefix REQUIRED (optional `Word N:` form), NO trailing.
    r"^[ \t]*#{1,6}[ \t]+"
    r"[A-Za-z]+[ \t]+(?:\d+:[ \t]+)?"
    r"(?:references|sources|bibliography)\b"
    r"[ \t]*$"
    r"|"
    # Shape C — no leading prefix, optional trailing (multi-word with prep-negation).
    r"^[ \t]*#{1,6}[ \t]+"
    r"(?:references|sources|bibliography)\b"
    r"(?:[ \t]+(?:[—–\-][ \t]+)?(?:\([^)\n]+\)|(?!(?:of|for|about|from|by|on|in)\b)[A-Za-z][^\n]*?))?"
    r"[ \t]*$"
    r")"
)

# A reference list entry: `[N] anything`. The URL is extracted separately via
# `_URL_RE` so we tolerate `[N] Title — URL`, `[N] Title - URL`, `[N] URL`,
# `[N] [Title](URL)`, etc. without listing every separator variant.
_REFERENCE_ENTRY_RE = re.compile(r"^[ \t]*\[(\d+)\][ \t]*(.+)$", re.MULTILINE)

# A bare URL inside a reference entry. Matches http(s) URLs through the first
# whitespace / bracket / unbalanced parenthesis. S98 m07: a balanced "(…)"
# segment is part of the URL (Wikipedia "Foo_(bar)", archive.org "/(57-59)%20…");
# a ")" closing a Markdown link or prose aside still ends it.
_URL_RE = re.compile(r"https?://(?:[^\s()\[\]]|\([^\s()\[\]]*\))+")

# Inline numbered citation marker: `[N]`. Caret-prefixed forms (`[^1]`,
# Markdown footnotes) and non-numeric forms (`[cite:…]`) do not match.
_NUMBERED_MARKER_RE = re.compile(r"\[(\d+)\]")

# S79 m01.c (PD95) — Unicode superscript citation markers.
#
# Mission 1 (PLUGIN-BACKEND-AUTH-01) emitted ¹–²² markers in body prose AND
# in the references list (`¹ Source title — URL`). The pre-S79 extractor
# recognised neither shape and silently dropped 22 references on that
# mission (see cmos/reports/sprint_79_approach_and_plan.md §1).
#
# Character set covers the full 0–9 superscript range:
#   ⁰ U+2070, ¹ U+00B9, ² U+00B2, ³ U+00B3,
#   ⁴ U+2074, ⁵ U+2075, ⁶ U+2076, ⁷ U+2077, ⁸ U+2078, ⁹ U+2079
#
# Body marker regex uses a lookbehind that REJECTS uppercase-letter
# prefixes (the standard math-variable shape) so ``T¹⁰`` (an exponent
# on variable ``T``) does NOT match. Lowercase prefixes like ``claim¹``
# are accepted because that is the dominant academic citation form.
# Lowercase single-letter variables (``y¹⁰``) remain a known false-positive
# risk; documented in PD95 risk #3 — the in-refs-section scope and the
# ``map.get(N)`` "drop on absence" semantics keep the blast radius small.
# The reference-entry regex anchors at line start (``^\s*…\s+``) so a
# superscript embedded inside title prose does NOT register as a new entry.
_UNICODE_SUPER_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_UNICODE_SUPER_TO_DIGIT = str.maketrans(
    {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
     "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
)
_UNICODE_SUPER_REFERENCE_ENTRY_RE = re.compile(
    r"^[ \t]*([" + _UNICODE_SUPER_DIGITS + r"]+)[ \t]+(.+)$",
    re.MULTILINE,
)
# S103 m02: a run after a superscript plus or minus sign is an exponent
# (``10⁻⁶``, ``e⁺²``) in every report.
_UNICODE_SUPER_SIGNS = "⁺⁻"
_UNICODE_SUPER_MARKER_RE = re.compile(
    # Lookbehind excludes uppercase letters AND any superscript char so that
    # a mid-run position (e.g. the ``⁰`` inside ``¹⁰``) does not trigger a
    # fresh match after the leading ``¹`` was rejected by the uppercase check.
    # A reader of a slice must match at its offset in the full text, so the
    # lookbehind sees the character before the run.
    r"(?<![A-Z" + _UNICODE_SUPER_DIGITS + _UNICODE_SUPER_SIGNS + r"])"
    r"([" + _UNICODE_SUPER_DIGITS + r"]+)"
)
_UNICODE_SUPER_DIGIT_RE = re.compile(r"[" + _UNICODE_SUPER_DIGITS + r"]")


def _super_to_int(token: str) -> int | None:
    """Convert a Unicode superscript run (``¹⁰``) to its integer value (10).

    Returns ``None`` if the token contains anything other than the 10
    Unicode superscript digits.
    """
    if not token or any(ch not in _UNICODE_SUPER_DIGITS for ch in token):
        return None
    try:
        return int(token.translate(_UNICODE_SUPER_TO_DIGIT))
    except ValueError:  # pragma: no cover - translation is total
        return None

# Sentence terminators: ``.``, ``!``, ``?`` followed by whitespace start a new
# sentence. ``\n`` also resets the cited-span start. Used by
# :func:`_find_span_start` when computing ``start_index``.
_SENTENCE_TERMINATORS = ".!?"


def _find_code_spans(markdown: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` ranges for fenced + inline code spans.

    Markers appearing inside one of these ranges are not extracted.
    """
    spans: List[Tuple[int, int]] = []
    fence: tuple[str, int, int] | None = None
    offset = 0
    for line in markdown.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= fence[1]:
                spans.append((fence[2], offset + len(line)))
                fence = None
        elif match:
            fence = (match[1][0], len(match[1]), offset)
        elif line.startswith(("    ", "\t")):
            spans.append((offset, offset + len(line)))
        offset += len(line)
    if fence is not None:
        spans.append((fence[2], len(markdown)))
    for match in _INLINE_CODE_RE.finditer(markdown):
        if any(s_start <= match.start() < s_end for s_start, s_end in spans):
            continue
        spans.append((match.start(), match.end()))
    spans.sort()
    return spans


def _is_inside_span(pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(s_start <= pos < s_end for s_start, s_end in spans)


def _find_span_start(stripped: str, end_index: int) -> int:
    """Walk back from ``end_index`` to the start of the cited claim.

    The cited claim begins immediately after the most recent sentence
    terminator (``. ``, ``! ``, ``? ``) or newline strictly before
    ``end_index``, or at position 0 if no boundary is found. The terminator's
    trailing whitespace must itself sit inside ``[0, end_index)``; if it
    doesn't, the terminator belongs to the cited claim (e.g. ``"claim."``
    immediately followed by the marker) and must not be treated as a previous
    boundary.
    """
    i = end_index - 1
    while i >= 0:
        ch = stripped[i]
        if ch == "\n":
            start = i + 1
            while start < end_index and stripped[start].isspace():
                start += 1
            return start
        if (
            ch in _SENTENCE_TERMINATORS
            and i + 1 < end_index
            and stripped[i + 1].isspace()
        ):
            start = i + 2
            while start < end_index and stripped[start].isspace():
                start += 1
            return start
        i -= 1
    start = 0
    while start < end_index and stripped[start].isspace():
        start += 1
    return start


def _source_identity(url: str) -> str:
    """The source a References link names: its arXiv paper, else the URL itself.

    S100 m02 extends decision 1002's paper identity from collection membership
    to reading References entries. Body mapping, readability and the probe
    cache keep exact URLs.
    """
    return arxiv_identity(url) or url


def _resolve_reference_line(urls: Sequence[str]) -> str | None:
    """Return a numbered line's first link when all its links name one source.

    A line whose links name different sources, such as a DOI and an unrelated
    mirror, is ambiguous and resolves to nothing.
    """
    return urls[0] if urls and len({_source_identity(url) for url in urls}) == 1 else None


def _section_end(markdown: str, heading: re.Match[str]) -> int:
    """Offset of the next heading at or above a references heading's level."""
    level = len(heading.group(0).lstrip().split()[0])
    offset = heading.end()
    for line in markdown[offset:].splitlines(keepends=True):
        next_heading = re.match(r"^\s*(#{1,6})\s", line)
        if next_heading and len(next_heading[1]) <= level:
            break
        offset += len(line)
    return offset


def _build_references_map(
    markdown: str, *, telemetry: dict[str, Any] | None = None,
    merged: set[int] | None = None,
) -> Tuple[Dict[int, str], int]:
    """Resolve only unambiguous numbered entries outside code, without I/O.

    The section is the last references heading not nested inside an earlier
    one; a nested subsection's entries belong to it, as in the finalize audit
    (S100 m02). A number resolves to its first line's link when every line
    resolves to one source, and is otherwise an unresolved collision.
    ``merged`` receives resolved numbers whose lines carry several links.
    """
    code_spans = _find_code_spans(markdown)
    headings = [m for m in _REFERENCES_HEADING_RE.finditer(markdown)
                if not _is_inside_span(m.start(), code_spans)]
    if not headings:
        return {}, len(markdown)
    heading, section_end = headings[0], _section_end(markdown, headings[0])
    for candidate in headings[1:]:
        if candidate.start() >= section_end:
            heading, section_end = candidate, _section_end(markdown, candidate)
    entries: dict[int, list[list[str]]] = {}
    prefix = re.compile(
        r"^\s*(?:[-*+]\s+)?(?:\*\*|__)?(?:\[(\d+)\]|(\d+)[.)])"
        r"(?:\*\*|__)?(?:\s|$)(.*)$"
    )
    offset = heading.end()
    for line in markdown[offset:section_end].splitlines(keepends=True):
        if not _is_inside_span(offset, code_spans):
            match = prefix.match(line)
            superscript = _UNICODE_SUPER_REFERENCE_ENTRY_RE.match(line)
            number = None
            text = ""
            if match:
                try:
                    number = int(match[1] or match[2])
                except ValueError:
                    number = None
                text = match[3]
            elif superscript:
                number = _super_to_int(superscript[1])
                text = superscript[2]
            if number is not None:
                urls = list(dict.fromkeys(
                    m.group(0).rstrip(".,;:") for m in _URL_RE.finditer(text)))
                if urls:
                    entries.setdefault(number, []).append(urls)
        offset += len(line)
    resolved: dict[int, str] = {}
    collisions = 0
    for number, lines in entries.items():
        links = [_resolve_reference_line(urls) for urls in lines]
        first = links[0]
        if first is None or any(
            link is None or _source_identity(link) != _source_identity(first) for link in links
        ):
            collisions += 1
            continue
        resolved[number] = first
        if merged is not None and len({url for urls in lines for url in urls}) > 1:
            merged.add(number)
    if telemetry is not None:
        telemetry["reference_number_collisions"] = collisions
    return resolved, heading.start()


def _marker_start(markdown: str, at: int, superscripts: bool) -> int | None:
    """Start of the ``[N]`` or superscript reader marker ending exactly at ``at``."""
    if at > 0 and markdown[at - 1] == "]":
        start = at - 2
        while start >= 0 and markdown[start].isdecimal():
            start -= 1
        return start if 0 <= start < at - 2 and markdown[start] == "[" else None
    if superscripts and at > 0 and markdown[at - 1] in _UNICODE_SUPER_DIGITS:
        start = at - 1
        while start > 0 and markdown[start - 1] in _UNICODE_SUPER_DIGITS:
            start -= 1
        return start if _UNICODE_SUPER_MARKER_RE.match(markdown, start) else None
    return None


def _claim_end(markdown: str, end: int, *, superscripts: bool) -> int:
    """Ignore a trailing reader-marker cluster only for claim identity/start.

    S103 m02 walks back from ``end`` over the cluster instead of searching the
    whole prefix, so the cost follows the cluster, not the report. The result
    equals the anchored search ``(?:\\s*(?:\\[\\d+\\]|<superscript marker>))+$``
    on ``markdown[:end]``, whose ``$`` also matches before a final newline.
    """
    at = end - 1 if end > 0 and markdown[end - 1] == "\n" else end
    cut = _marker_start(markdown, at, superscripts)
    while cut is not None:
        before = cut
        while before > 0 and markdown[before - 1].isspace():
            before -= 1
        earlier = _marker_start(markdown, before, superscripts)
        if earlier is None:
            break
        cut = earlier
    return len(markdown[:end if cut is None else cut].rstrip())


def _extract_numbered_fallback(
    markdown: str, *, superscripts: bool, telemetry: dict[str, Any] | None = None,
    merged: List[CitationAnnotation] | None = None,
) -> List[CitationAnnotation]:
    """Resolve inline `[N]` and Unicode-superscript markers against the
    trailing References section.

    Runs on the marker-stripped text alongside primary extraction.
    Returns annotations whose indices reference ``markdown`` directly (no
    stripping — numbered markers are part of the human-readable surface and
    stay visible to the reader). When the references section is absent,
    no map can be built and an empty list is returned. Markers whose number
    is not present in the map are silently dropped (orphan numbered markers
    are not preserved as guesses).

    S79 m01.c (PD95): Unicode superscript markers (``¹``, ``²``, …, ``¹⁰``)
    are scanned alongside the ``[N]`` style, scoped to the body region (up
    to ``body_end`` — the references heading offset). ``(?<!\\w)`` anchors
    the body match to avoid misclassifying exponents (``T¹⁰``). S103 m02 scans
    them only when ``superscripts`` says the report cites that way.
    """
    merged_numbers: set[int] = set()
    refs_map, body_end = _build_references_map(
        markdown, telemetry=telemetry, merged=merged_numbers,
    )

    code_spans = _find_code_spans(markdown[:body_end])
    annotations: List[CitationAnnotation] = []

    def _emit(match_start: int, number: int) -> None:
        url = refs_map.get(number)
        if not url:
            if telemetry is not None:
                telemetry["unresolved_numbered_markers"] = (
                    telemetry.get("unresolved_numbered_markers", 0) + 1
                )
            return
        end_index = match_start
        # Mirror the primary path: trim trailing whitespace so the cited span
        # excludes the space immediately preceding the marker.
        while end_index > 0 and markdown[end_index - 1].isspace():
            end_index -= 1
        start_index = _find_span_start(
            markdown, _claim_end(markdown, end_index, superscripts=superscripts))
        annotation = CitationAnnotation(
            type="url_citation",
            start_index=start_index,
            end_index=end_index,
            url=url,
            title=None,
        )
        annotations.append(annotation)
        if merged is not None and number in merged_numbers:
            merged.append(annotation)

    for match in _NUMBERED_MARKER_RE.finditer(markdown, 0, body_end):
        if (_is_inside_span(match.start(), code_spans)
                or markdown[match.end():].startswith("(")
                or (len(markdown[:match.start()])
                    - len(markdown[:match.start()].rstrip("\\"))) % 2):
            continue
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        _emit(match.start(), number)

    for match in (_UNICODE_SUPER_MARKER_RE.finditer(markdown, 0, body_end)
                  if superscripts else ()):
        if _is_inside_span(match.start(), code_spans):
            continue
        super_number = _super_to_int(match.group(1))
        if super_number is None:
            continue
        _emit(match.start(), super_number)

    for annotation in annotations:
        annotation._extraction_mode = "numbered"
    annotations.sort(key=lambda a: (a.end_index, a.start_index))
    return annotations


def _labels_in_superscripts(markdown: str) -> bool:
    """Whether a line outside code labels an entry with a superscript and a URL."""
    labels = [match.start(1) for match in _UNICODE_SUPER_REFERENCE_ENTRY_RE.finditer(markdown)
              if _URL_RE.search(match[2])]
    if not labels:
        return False
    code_spans = _find_code_spans(markdown)
    return any(not _is_inside_span(start, code_spans) for start in labels)


def _extract_in_style(
    markdown: str, telemetry: dict[str, Any] | None,
) -> Tuple[str, List[CitationAnnotation], bool]:
    """Extraction with the report's own superscript reading, and that reading."""
    if not _UNICODE_SUPER_DIGIT_RE.search(markdown):
        return (*_extract(markdown, telemetry, superscripts=False), False)
    if _labels_in_superscripts(markdown):
        return (*_extract(markdown, telemetry, superscripts=True), True)
    counts: dict[str, Any] = {}
    stripped, annotations = _extract(markdown, counts, superscripts=False)
    superscripts = False
    if not annotations:
        read_counts: dict[str, Any] = {}
        read_stripped, read = _extract(markdown, read_counts, superscripts=True)
        if read:
            stripped, annotations, counts, superscripts = read_stripped, read, read_counts, True
    if telemetry is not None:
        telemetry.update(counts)
    return stripped, annotations, superscripts


def reads_superscripts(markdown: str) -> bool:
    """Whether a superscript digit run in ``markdown`` can be a reader marker (S103 m02).

    A report reads superscripts when a line outside code labels an entry with one
    (a superscript run, whitespace, then text holding an http(s) URL), or when they
    are its only citations: extraction without them finds none and with them at
    least one. The whole text is read, because the extractor and the audit can
    disagree on where References starts. Every reader of superscript digits takes
    its answer from here; a run after an uppercase letter, a superscript digit or a
    superscript sign is never a marker either way.
    """
    if not _UNICODE_SUPER_DIGIT_RE.search(markdown):
        return False
    return _labels_in_superscripts(markdown) or _extract_in_style(markdown, None)[2]


def extract_citations(
    markdown: str, *, telemetry: dict[str, Any] | None = None,
) -> Tuple[str, List[CitationAnnotation]]:
    """Strip ``[cite:URL]`` markers and emit codepoint-indexed annotations.

    Parameters
    ----------
    markdown:
        Post-critique final markdown that may contain ``[cite:URL]`` tokens.

    Returns
    -------
    tuple
        ``(stripped_markdown, annotations)`` where ``stripped_markdown`` is
        the input with every well-formed marker removed (literal removal — no
        whitespace normalisation) and ``annotations`` is the list of
        :class:`CitationAnnotation` in left-to-right scan order. Indices on
        each annotation reference ``stripped_markdown`` directly:
        ``stripped_markdown[ann.start_index:ann.end_index]`` returns the cited
        span. ``title`` is always ``None`` at extraction time (resolution is
        the caller's responsibility).

        Both formats contribute. Deduplication uses claim start, claim end
        without trailing reader markers, and exact URL. Distinct URLs at the
        same boundary survive; repeated claims at other positions survive.
        Optional telemetry receives counts only and introduces no I/O dependency.
        Superscript markers are read only as :func:`reads_superscripts` decides.
    """
    stripped_markdown, annotations, _ = _extract_in_style(markdown, telemetry)
    return stripped_markdown, annotations


def _extract(
    markdown: str, telemetry: dict[str, Any] | None, *, superscripts: bool,
) -> Tuple[str, List[CitationAnnotation]]:
    counts = telemetry if telemetry is not None else {}
    counts.update(primary_annotations=0, numbered_annotations=0, duplicates_removed=0,
                  reference_number_collisions=0, unresolved_numbered_markers=0)
    if not markdown:
        return markdown, []

    code_spans = _find_code_spans(markdown)

    annotations: List[CitationAnnotation] = []
    stripped_parts: List[str] = []
    last_end = 0
    stripped_len = 0

    for match in _CITE_MARKER_RE.finditer(markdown):
        if _is_inside_span(match.start(), code_spans):
            continue

        url = match.group(1)
        # Append text between the previous cut and this marker.
        chunk = markdown[last_end : match.start()]
        stripped_parts.append(chunk)
        stripped_len += len(chunk)
        last_end = match.end()

        marker_pos = stripped_len  # codepoint position where marker would re-insert.

        # Back up over trailing whitespace so the cited span doesn't include
        # the space that preceded the marker.
        end_index = marker_pos
        # We need the stripped string so far to inspect characters; build lazily.
        stripped_so_far = "".join(stripped_parts)
        while end_index > 0 and stripped_so_far[end_index - 1].isspace():
            end_index -= 1

        start_index = _find_span_start(
            stripped_so_far, _claim_end(stripped_so_far, end_index, superscripts=superscripts))

        annotations.append(
            CitationAnnotation(
                type="url_citation",
                start_index=start_index,
                end_index=end_index,
                url=url,
                title=None,
            )
        )

    # Tail: append everything after the last marker.
    stripped_parts.append(markdown[last_end:])
    stripped_markdown = "".join(stripped_parts)

    counts["primary_annotations"] = len(annotations)
    for annotation in annotations:
        annotation._extraction_mode = "primary"
    merged: list[CitationAnnotation] = []
    numbered = _extract_numbered_fallback(
        stripped_markdown, superscripts=superscripts, telemetry=counts, merged=merged)
    counts["numbered_annotations"] = len(numbered)
    counts["numbered_distinct_urls"] = len({a.url for a in numbered})
    # S100 m02: a number that resolves only because its links name one source
    # must not add a second citation of that source to a primary-cited claim.
    primary_sources = {
        (a.start_index, _claim_end(stripped_markdown, a.end_index, superscripts=superscripts),
         _source_identity(a.url))
        for a in annotations
    }
    merged_ids = {id(a) for a in merged}
    reconciled: list[CitationAnnotation] = []
    seen: set[tuple[int, int, str]] = set()
    for annotation in [*annotations, *numbered]:
        claim_end = _claim_end(stripped_markdown, annotation.end_index, superscripts=superscripts)
        key = (annotation.start_index, claim_end, annotation.url)
        if key in seen or (id(annotation) in merged_ids and (
                annotation.start_index, claim_end, _source_identity(annotation.url),
        ) in primary_sources):
            counts["duplicates_removed"] += 1
            continue
        seen.add(key)
        reconciled.append(annotation)
    reconciled.sort(key=lambda a: (a.end_index, a.start_index))
    counts["reconciled_annotations"] = len(reconciled)
    return stripped_markdown, reconciled


def is_cited(text: object) -> bool:
    """Whether runtime text counts as cited (S100 m03).

    The runtime's one definition: the literal ``[cite:`` test used before S100,
    or at least one annotation from :func:`extract_citations`, so a numbered
    marker that resolves against the References section counts as it does in
    finalize. If extraction raises, the literal test's answer stands, so a paid
    step never fails on this check.
    """
    if not isinstance(text, str):
        return False
    if "[cite:" in text:
        return True
    try:
        return bool(extract_citations(text)[1])
    except Exception:  # noqa: BLE001 — advisory check; the literal answer stands
        return False


__all__ = ["extract_citations", "is_cited", "reads_superscripts"]
