"""completeness_check — pre-acceptance deliverable spec coverage.

S79 m02.a (PD97). Pure-function tool the CodeAgent calls before
``final_answer(...)`` to surface which named deliverables in the mission
spec are present vs missing in the current draft. Telemetry-only signal —
the agent decides whether to keep retrieving or finalize. No hard mechanism;
behavior-shaping clause in ``contract_to_prompt`` (m02.b) names this tool.

The same function is also called at the codeagent_finalize_node telemetry
build site (m02.c) to surface a ``deliverable_coverage`` field on
``synthesis_telemetry`` post-draft. Telemetry surface decoupled from gating
per PD97 Option B.

S82 m04 (PD135) hardened the matcher: a deliverable is satisfied only when its
canonical heading binds to a real PARSED Markdown heading whose section is
substantive (non-placeholder word floor OR inline citation signal), or — for
"at least N" quantity deliverables — when the draft carries >= N structural
entity units (table rows / H3 subsections / numbered entries). The pre-S82
full-body substring match over-counted: it scored a deliverable satisfied when
its keyword merely appeared in an executive-summary echo, a footer, or an
inline bold label. The leading-noun-phrase extraction (``_canonical_heading``)
is unchanged (e.g. ``"Comparison matrix with rows for each entity"`` →
``"Comparison matrix"``).

S93 separates artifact format from content requirements. Empty or malformed
specifications are explicitly unassessed (null shortfall), never fully covered.
Format labels do not require a matching content heading; attached content or
evidence requirements still apply. This is a structural coverage check, not
claim-level evidence validation.

DSV-045 (live, 2026-06-15) added a descendant rollup: a ``##`` deliverable
heading whose prose lives entirely under ``###`` children was scored MISSING
(its own body was empty), failing a complete 3.5k-word report 0/4. A matched
heading whose own body is not itself substantive is now SATISFIED when its
strictly-deeper descendant prose clears the citation-OR-words substance check.
The rollup is deliberately RECALL-BIASED: it never produces a false MISSING (the
costly, report-killing error) but it CAN over-count. This is acceptable because
a coverage shortfall now DEGRADES rather than catastrophically fails (see
``deepsearch/agent/outcome.py::evaluate_outcome_gate``) — an over-count only
softens the gate, the LLM critique and liveness signals remain, and the gate's
value is the FLAG not suppression. Known over-count limitations (a genuinely
absent/stub deliverable scored SATISFIED):

  * a stub heading (non-placeholder thin own body, e.g. "TODO") that has a
    misnested PEER deliverable or an off-topic word-rich subsection nested under
    it borrows that descendant's substance;
  * an empty-bodied heading whose only descendants are placeholder stubs that
    each carry a citation/URL (the borrowed citation satisfies the rollup).

A section whose OWN body is an explicit gap-note (``[Data unavailable ...]``) is
the one resurrection case guarded: it declares its own content absent and is NOT
rescued by descendants.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Words to drop when extracting the canonical heading from a deliverable
# bullet. ``"comparison matrix with rows for each entity"`` becomes
# ``"comparison matrix"`` after the prepositional tail is dropped.
_HEADING_TAIL_BOUNDARIES = re.compile(
    r"\s+(with|for|covering|that|including|listing|mapping|spanning|across|of|by|per)\b",
    re.IGNORECASE,
)
# Strip leading list markers (``- ``, ``* ``, ``1. ``) and Markdown
# emphasis (``**…**``, ``*…*``).
_LEADING_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
_MD_EMPHASIS_RE = re.compile(r"^\**(.*?)\**$")
# S81 m01.b (PD116a): strip parenthetical clauses BEFORE boundary-word
# truncation so an open paren never orphans the canonical heading. Before
# this fix ``"Pattern catalogue (persistent-account / ephemeral-session /
# hybrid with optional account upgrade)"`` reduced to ``"Pattern catalogue
# (persistent-account / ephemeral-session / hybrid"`` — open-paren orphan
# that defeats substring match against ``## 5. Pattern / Practice
# Catalogue``. PLUGIN-BACKEND-AUTH-01 reported 5/6 false-missing post-S80
# because of this. The canonical heading is for substring matching only,
# never for display; stripping parens here does not touch body content.
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*")
_WHITESPACE_RE = re.compile(r"\s+")


def _canonical_heading(bullet: str) -> str:
    """Reduce a deliverable-spec bullet to its leading noun phrase."""
    text = _LEADING_LIST_RE.sub("", bullet).strip()
    text = _MD_EMPHASIS_RE.sub(r"\1", text).strip()
    # S81 m01.b (PD116a): strip parentheticals BEFORE boundary-word
    # truncation. See _PARENTHETICAL_RE comment block above.
    text = _PARENTHETICAL_RE.sub(" ", text).strip()
    # Stop at the first prepositional / clausal boundary.
    head_match = _HEADING_TAIL_BOUNDARIES.search(text)
    if head_match:
        text = text[: head_match.start()].strip()
    # Strip trailing punctuation that would defeat substring matching.
    text = text.rstrip(".:;,—-")
    # Collapse interior double-whitespace introduced by paren stripping
    # (``"Foo  Bar"`` from ``"Foo (notes) Bar"`` → ``"Foo Bar"``).
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# S82 m04 (PD135/PD136) — heading-scoped, substance-aware coverage.
#
# The S79 substring matcher (``heading.lower() in draft.lower()``) scored a
# deliverable "satisfied" whenever its keyword appeared ANYWHERE in the body —
# in an Executive-Summary echo, a footer restating the spec, or an inline bold
# label. The Concordance triage (PD135) showed this over-counts: R003 echoed
# "At least 30 ... meeting the deliverable threshold" in prose while delivering
# only 20 of 30 entity sections; the substring matcher rewarded STATING the
# requirement, not MEETING it. That fed a false deliverable_coverage into the
# S82 m01 outcome gate.
#
# The hardened check requires a deliverable to bind to a real PARSED Markdown
# heading whose section is SUBSTANTIVE, and scores quantity deliverables
# ("at least N ...") by the actual count of structural entity units (table
# rows / H3 subsections / numbered entries), not by prose.
#
# Substance = a non-placeholder word floor OR an inline citation signal. The
# citation-OR-words rule is chosen because real reports cite via a References
# section / ``[N]`` markers rather than inline URLs, so requiring an inline
# citation ALONE would false-fail substantive sections (R002 had 8 word-rich
# vendor sections carrying "[Data unavailable ...]" placeholders). The stub
# test is therefore placeholder-DOMINATED (count words AFTER stripping
# placeholders), not raw phrase-presence.
#
# MIN_SECTION_WORDS is a conservative starting value (PD136: "tune from real
# data"); the behavior is pinned by tests/unit/test_s82_m04_invariants.py.
# ---------------------------------------------------------------------------

MIN_SECTION_WORDS = 40

# Gap-note / "no data" placeholders that must NOT count toward section
# substance — the canonical critique gap-note phrasing (bracketed or
# parenthesised), plus a bare "not found" and em-dash placeholders.
_PLACEHOLDER_RE = re.compile(
    r"[\[(]?\s*data unavailable(?:\s+in\s+primary\s+sources)?\s*[\])]?"
    r"|\bnot found\b",
    re.IGNORECASE,
)
_EM_DASH_PLACEHOLDER_RE = re.compile(r"(?:^|\s)[—–]{1,2}(?=\s|$)", re.MULTILINE)

# Inline citation signals that mark a section as evidence-bearing regardless of
# length: a ``[cite:URL]`` marker, a bare URL, or a markdown link to a URL.
_CITATION_SIGNAL_RE = re.compile(r"\[cite:|https?://", re.IGNORECASE)

# Markdown heading line (up to 3 leading spaces, 1-6 hashes, optional closing
# hashes). group(1) = hashes, group(2) = heading text.
_HEADING_LINE_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
# H3-or-deeper heading (``### ...``) — the per-entity subsection unit.
_H3_PLUS_RE = re.compile(r"^\s{0,3}#{3,6}\s+\S")
# Markdown table rows: any leading-pipe row, and the header-separator row.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s:|\-]+\|\s*$")
# Top-level numbered list item ("1. Stripe") — an entity-enumeration proxy.
_NUMBERED_ITEM_RE = re.compile(r"^\s{0,3}\d+\.\s+\S")
# Fenced-code-block delimiter (``` or ~~~, up to 3 leading spaces). Markdown
# structure inside a fence is inert, so a ``# comment`` / ``### x`` / table row /
# numbered line inside code must NOT be parsed as document structure.
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

# Minimum-entity-count phrasings in a deliverable bullet → quantity deliverable.
_MIN_COUNT_RES = [
    re.compile(r"(?:at least|no fewer than|minimum of|>=|≥)\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:\+|or more|or greater)", re.IGNORECASE),
    # "3-5 named case studies" → lower bound 3.
    re.compile(r"(\d+)\s*[-–]\s*\d+\s+(?:named|case|example)", re.IGNORECASE),
]

# Stopwords dropped from a canonical heading before token matching.
_HEADING_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "and", "or", "to", "in", "on", "by", "per", "for", "with"}
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _heading_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _canonical_tokens(canonical: str) -> list[str]:
    return [t for t in _heading_tokens(canonical) if t not in _HEADING_STOPWORDS]


def _overlap_floor(n: int) -> int:
    """Required token matches for an N-token signal.

    ``N == 1`` -> 1 (the single content token must be a substring of some
    heading token — the strict pre-S83 behaviour, so single-noun deliverables
    do NOT get fuzzier). ``N >= 2`` -> ``ceil(N/2)``.
    """
    if n <= 1:
        return 1
    return -(-n // 2)  # ceil(n / 2) without importing math


def _depluralize(token: str) -> str:
    """Strip a regular English plural suffix to a singular stem.

    DSV-045 follow-up: substring matching is one-directional, so a plural
    deliverable token (``"recommendations"``) never bound a singular heading
    token (``"recommendation"``) even though the reverse worked. Reducing both
    sides to a singular stem before comparison closes that gap. The length guards
    avoid creating sub-3-char stems (which would substring-match almost anything),
    and ``ss`` is excluded so ``"process"``/``"address"`` are left intact rather
    than becoming the spurious stems ``"proces"``/``"addres"``.

    KNOWN WART (over-count direction only, accepted under decision #820): a SINGULAR
    noun ending in ``-s`` is over-stemmed because there is no dictionary to tell it
    from a plural — ``"status"``->``"statu"``, ``"lens"``->``"len"``,
    ``"news"``->``"new"``. This is the mechanism behind the narrow, benign cross-word
    over-matches the adversarial pass found (``"Buses"`` binding ``"## Business …"``
    via the ``"bus"`` stem); they bind nothing in realistic multi-heading reports and
    only ever soften the gate, never harden it.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _token_matches(needle: str, heading_token: str) -> bool:
    """True iff ``needle`` matches ``heading_token`` by raw substring OR by
    singular-stem substring.

    The raw-substring clause is kept FIRST so every pair that matched before this
    change still matches: the singular-stem clause only ADDS matches, so recall is
    monotonically non-decreasing — the safe direction for a recall-biased matcher
    (decision #820: never introduce a guard that risks false-MISSING).
    """
    if needle in heading_token:
        return True
    return _depluralize(needle) in _depluralize(heading_token)


def _heading_content_tokens(heading_text: str) -> list[str]:
    """Heading tokens that carry meaning — drop stopwords and pure-digit section
    numbers (``"## 9. Recommendations"`` → ``["recommendations"]``)."""
    return [
        t
        for t in _heading_tokens(heading_text)
        if t not in _HEADING_STOPWORDS and not t.isdigit()
    ]


def _tokens_clear_floor(needles: list[str], heading_text: str) -> bool:
    """True iff ``needles`` clear the binding floor against the heading's content
    tokens.

    A needle matches when it is a substring of some heading content token, OR when
    their singular stems substring-match (so the plural ``"recommendations"`` binds
    the singular ``"recommendation"`` — DSV-045 follow-up b).

    The floor is the HEADING-BUDGET floor (DSV-045 follow-up c):
    ``min(ceil(N/2), #heading_content_tokens)``. Capping by the heading's own
    content-token count is the fix for the report-killer false-MISSING class the
    adversarial pass surfaced: a deliverable bullet with a long descriptive tail
    and no boundary word ("Recommendations the team should adopt immediately",
    N=5, ceil(N/2)=3) used to be unable to clear the floor against a heading that
    reproduces its lead noun EXACTLY but supplies only one content token
    (``## Recommendations``, 1 overlap). Pinning the floor to the heading's budget
    means a short heading carrying the deliverable's lead noun always binds,
    regardless of how verbose the bullet is. The cap only ever LOWERS the floor, so
    recall is monotonically non-decreasing (decision #820's safe direction); the
    pinned 0-overlap negatives are untouched because 0 matches never clears a floor
    of >= 1. Empty ``needles`` or a content-token-less heading never match.
    """
    if not needles:
        return False
    head_content = _heading_content_tokens(heading_text)
    if not head_content:
        return False
    matched = sum(
        1 for needle in needles if any(_token_matches(needle, ht) for ht in head_content)
    )
    floor = min(_overlap_floor(len(needles)), len(head_content))
    return matched >= floor


def _heading_matches(canonical: str, heading_text: str) -> bool:
    """True iff the canonical heading's content tokens clear the overlap floor
    against the parsed Markdown ``heading_text``.

    S83 m04 (CASE 1) relaxed the pre-S83 strict ALL-AND rule to a token-overlap
    floor; DSV-045 follow-up (c) then capped that floor by the heading's own
    content-token budget (``min(ceil(N/2), #heading_content_tokens)`` — see
    ``_tokens_clear_floor``). A present but reworded/verbose section is therefore
    no longer scored MISSING: canonical ``"A final recommendation"`` (N=2) and the
    long-tail ``"Recommendations the team should adopt immediately"`` (N=5) both
    bind ``"9. Recommendation"`` / ``"## Recommendations"`` on the single
    ``recommendation`` overlap, because the heading supplies one content token so
    the floor is 1. Heading-scoped + substance-AND (in ``completeness_check``)
    still kill prose-echo / inline-label / stub-body false positives.

    TRADE-OFF (documented, accepted per the S83 plan + decision #820): the floor
    can be 1, so deliverables that share a single content token with a heading can
    each bind it. The pinned invariant is the 0-overlap negative (``"Comparison
    matrix"`` !-> ``"Executive Summary"``), NOT single-token exclusivity — the
    latter is in direct tension with the recall-biased false-MISSING fix.
    """
    return _tokens_clear_floor(_canonical_tokens(canonical), heading_text)


def _parenthetical_tokens(bullet: str) -> list[str]:
    """Content tokens inside the deliverable bullet's parenthetical clause(s).

    S83 m04 (CASE 2): a pure canonical-token relaxation cannot bind a heading
    that drops the deliverable's LEAD noun — ``"Risk register (likelihood /
    impact / mitigation)"`` canonicalises to ``"Risk register"`` (parens are
    stripped by ``_canonical_heading``), which has 0 overlap with a reworded
    ``"## Likelihood and Impact Mitigation Table"``. The parenthetical sub-terms
    DO overlap, so they give a second binding signal. Reuses
    ``_PARENTHETICAL_RE`` / ``_TOKEN_RE`` / ``_HEADING_STOPWORDS``; this helper
    does NOT change ``_canonical_heading`` output (the canonical stays
    ``"Risk register"``).

    Single-character tokens are dropped so a generic-example parenthetical
    (``"(e.g. Stripe, Square)"`` -> the ``e``/``g`` of ``e.g.``) cannot
    substring-match arbitrary heading tokens (the parenthetical-noise guard).
    """
    tokens: list[str] = []
    for span in _PARENTHETICAL_RE.findall(bullet):
        inner = span.strip().strip("()")
        tokens.extend(
            tok
            for tok in _heading_tokens(inner)
            if len(tok) > 1 and tok not in _HEADING_STOPWORDS
        )
    return tokens


def _strip_placeholders(body: str) -> str:
    body = _PLACEHOLDER_RE.sub(" ", body)
    body = _EM_DASH_PLACEHOLDER_RE.sub(" ", body)
    return body


def _section_is_substantive(body: str) -> bool:
    """A section counts only if it carries an inline citation signal OR clears
    the non-placeholder word floor. Placeholder-dominated stub sections
    (em-dash / "Data unavailable") therefore do NOT count satisfied."""
    if _CITATION_SIGNAL_RE.search(body):
        return True
    return len(_strip_placeholders(body).split()) >= MIN_SECTION_WORDS


def _content_lines(draft_markdown: str):
    """Yield the draft's lines that are NOT inside a fenced code block.

    A ``# comment`` / ``### x`` / ``| row |`` / ``1. step`` line inside a ``` /
    ~~~ fence is code, not document structure. Treating it as a heading either
    severs the descendant rollup (a code comment becomes a spurious shallow
    heading that strands real prose -> false MISSING) or fakes a section out of
    code text (-> false SATISFIED). CommonMark renders fence contents inert, so
    we skip them here for both section parsing and entity counting.

    CommonMark closes a fence only with a same-char delimiter run AT LEAST AS
    LONG as the opener, so an inner ``` run inside a ```` four-backtick ````
    block is literal content, not a close. We track the opener's full run and
    honour that length rule (matching on the char alone mis-toggles such blocks).
    """
    fence_run = ""  # opener delimiter run (e.g. "```"); "" when not in a fence
    for line in draft_markdown.split("\n"):
        match = _FENCE_RE.match(line)
        if match:
            run = match.group(1)
            if not fence_run:
                fence_run = run
                continue
            if run[0] == fence_run[0] and len(run) >= len(fence_run):
                fence_run = ""
                continue
            # A shorter or different-char run while open is ordinary code content.
        if fence_run:
            continue
        yield line


def _parse_sections_struct(draft_markdown: str) -> list[tuple[int, str, str]]:
    """Split the draft into ``(level, heading_text, own_body)`` triples.

    ``own_body`` is ONLY the lines directly under the heading, up to the next
    heading of any depth (it does NOT roll up descendants — the rollup lives in
    ``completeness_check``). Text before the first heading is ignored —
    a deliverable must bind to a heading, not to preamble / executive-summary /
    footer prose (the prose-echo loophole the S82 m04 hardening closes). Fenced
    code blocks are skipped via ``_content_lines``.
    """
    sections: list[tuple[int, str, str]] = []
    level: int | None = None
    heading: str | None = None
    body_lines: list[str] = []
    for line in _content_lines(draft_markdown):
        match = _HEADING_LINE_RE.match(line)
        if match:
            if heading is not None:
                sections.append((level, heading, "\n".join(body_lines)))
            level = len(match.group(1))
            heading = match.group(2)
            body_lines = []
        elif heading is not None:
            body_lines.append(line)
    if heading is not None:
        sections.append((level, heading, "\n".join(body_lines)))
    return sections


def _count_entity_units(draft_markdown: str) -> int:
    """Count structural entity units for quantity ("at least N") deliverables.

    The max of (table data rows, H3+ subsections, top-level numbered items).
    Bulleted sub-lists are intentionally NOT counted — they enumerate
    per-entity details, not entities — so the count under- rather than
    over-reports. The header row of each table is removed by subtracting the
    separator-row count (one separator per table == one header per table).
    """
    non_sep_pipe = separators = h3_plus = numbered = 0
    for line in _content_lines(draft_markdown):
        if _TABLE_SEPARATOR_RE.match(line):
            separators += 1
        elif _TABLE_ROW_RE.match(line):
            non_sep_pipe += 1
        if _H3_PLUS_RE.match(line):
            h3_plus += 1
        if _NUMBERED_ITEM_RE.match(line):
            numbered += 1
    table_data_rows = max(0, non_sep_pipe - separators)
    return max(table_data_rows, h3_plus, numbered)


def _extract_min_entity_count(bullet: str) -> int | None:
    """Return the minimum entity count a deliverable bullet demands, or None.

    Operates on the RAW bullet (the count phrasing is usually past the
    boundary-word truncation that ``_canonical_heading`` applies).
    """
    for rx in _MIN_COUNT_RES:
        match = rx.search(bullet)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


# Only the declared artifact vocabulary is special. A topic such as
# "Report generation risks" remains an ordinary content requirement.
_ARTIFACT_RE = re.compile(
    r"^(?:a\s+)?Markdown(?:\s+research)?\s+report(?=$|[. :]|\s)", re.IGNORECASE
)
_ARTIFACT_TAIL_RE = re.compile(r"^\s*(?:with|including|covering|about|on)\s+", re.I)
_EVIDENCE_REQUIREMENT_RE = re.compile(
    r"^(?:resolving\s+)?citations?(?:\s+resolving)?(?:\s+to\s+evidence)?[.]?$", re.I
)


def _coverage_body(markdown: str) -> str:
    """Exclude the deterministic report envelope from content assessment."""
    text = markdown
    if text.lstrip().startswith("# Research Report:"):
        text = text.lstrip().split("\n", 1)[-1]
        text = re.sub(
            r"\A\s*\*\*Sources collected:\*\*[^\n]*\n"
            r"\*\*Verified references:\*\*[^\n]*\n"
            r"\*\*Agent steps:\*\*[^\n]*\n"
            r"\*\*Runtime:\*\*[^\n]*\n", "", text,
        )
    # Exact template boundary, not arbitrary words from a report's prose.
    return text.split("\n---\n\n**Mission Objectives:**", 1)[0].strip()


def _report_prose(markdown: str) -> str:
    """Return prose outside headings, references, and code fences."""
    lines: list[str] = []
    in_references = False
    fence: str | None = None
    for line in markdown.splitlines():
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = _HEADING_LINE_RE.match(line)
        if heading:
            in_references = bool(re.fullmatch(
                r"(?:\d+[.)]?\s*)?(?:references|bibliography|sources)",
                heading.group(2).strip(), re.I,
            ))
            continue
        if not in_references:
            lines.append(line)
    return "\n".join(lines)


def _has_evidence_signal(markdown: str) -> bool:
    """Check citation presence; URL resolution and claim support are separate audits."""
    prose = _report_prose(markdown)
    if _CITATION_SIGNAL_RE.search(prose):
        return True
    references = set(re.findall(r"^\s*\[(\d+)\].*https?://", markdown, re.M))
    return bool(references.intersection(re.findall(r"\[(\d+)\]", prose)))


def _coverage_result(
    artifacts: list[dict[str, Any]], content: list[dict[str, Any]],
) -> dict[str, Any]:
    requirements = artifacts + content
    satisfied = [r["requirement"] for r in requirements if r["satisfied"]]
    missing = [r["requirement"] for r in requirements if not r["satisfied"]]
    return {
        "semantics_version": 2,
        "assessment_status": "assessed" if requirements else "unassessed",
        "content_status": "assessed" if content else "unassessed",
        "artifact_requirements": artifacts,
        "content_requirements": content,
        "satisfied": satisfied,
        "missing": missing,
        "shortfall_score": round(len(missing) / len(requirements), 3) if requirements else None,
    }


def completeness_check(
    draft_markdown: str,
    deliverable_spec: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Surface which named deliverables are present vs missing in the draft.

    A deliverable is SATISFIED when:

    * it declares a minimum entity count ("at least N ...") and the draft
      contains >= N structural entity units (table rows / H3 subsections /
      numbered entries); OR
    * its canonical heading (a token-overlap match against the heading-budget
      floor ``min(ceil(N/2), #heading_content_tokens)`` — S83 m04 + DSV-045
      follow-up; plural/singular-tolerant) OR its parenthetical sub-terms bind to
      a real parsed Markdown heading whose section is substantive (clears the
      non-placeholder word floor OR carries an inline citation signal).

    Prose echoes, inline bold labels, and footer restatements of the spec do
    NOT satisfy — the match is heading-scoped, not full-body (S82 m04, PD135);
    a reworded heading over a stub body still scores missing (the overlap/paren
    relaxation is AND-ed with the substance check, S83 m04).

    Descendant rollup retains the recall-biased policy described above.
    Artifact-format requirements check report substance, not headings. Attached
    content clauses remain required, including citation presence (resolution is
    assessed by the separate liveness/citation audits).

    Args:
        draft_markdown: The current draft as a single Markdown string. At the
            ``codeagent_finalize_node`` telemetry site this is the FINAL
            post-critique markdown (S82 m04 (d)); in the CodeAgent loop it is
            the agent's working draft.
        deliverable_spec: deliverable / success-criteria bullets from the
            mission contract (resolved by ``resolve_deliverables``).

    Returns:
        Versioned artifact and content assessments, compatible satisfied/missing
        lists, and shortfall over the assessed requirements. Missing or malformed
        specifications/drafts return unassessed with a null shortfall.
    """
    if not isinstance(draft_markdown, str):
        logger.warning("completeness_check: invalid draft; coverage unassessed")
        return _coverage_result([], [])
    if not deliverable_spec or isinstance(deliverable_spec, (str, bytes)):
        return _coverage_result([], [])

    draft_markdown = _coverage_body(draft_markdown)
    artifacts: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    content_bullets: list[str] = []
    for bullet in deliverable_spec:
        if not isinstance(bullet, str) or not bullet.strip():
            continue
        clean = _MD_EMPHASIS_RE.sub(r"\1", _LEADING_LIST_RE.sub("", bullet).strip())
        artifact = _ARTIFACT_RE.match(clean)
        if artifact is None:
            content_bullets.append(bullet)
            continue
        tail = clean[artifact.end():].strip().rstrip(".")
        if tail and not _ARTIFACT_TAIL_RE.match(tail):
            content_bullets.append(bullet)
            continue
        requirement = artifact.group(0).strip()
        if not any(r["requirement"].lower() == requirement.lower() for r in artifacts):
            artifacts.append({
                "requirement": requirement, "format": "markdown",
                "satisfied": (
                    _section_is_substantive(_report_prose(draft_markdown))
                    or _has_evidence_signal(draft_markdown)
                ),
            })
        tail = _ARTIFACT_TAIL_RE.sub("", tail).strip()
        if tail:
            if _EVIDENCE_REQUIREMENT_RE.fullmatch(tail):
                evidence.append({
                    "requirement": tail, "kind": "citation_presence",
                    "satisfied": _has_evidence_signal(draft_markdown),
                    "resolution_status": "unassessed",
                })
            else:
                content_bullets.append(tail)

    # Build the (canonical heading, min_entity_count, parenthetical_tokens) spec,
    # deduped by canonical heading. The RAW bullet's parenthetical tokens are
    # threaded here (S83 m04 CASE 2) because the bullet is discarded after this
    # loop; satisfied/missing still carry the CANONICAL string (so the
    # outcome.py / protocol.py count consumers are unaffected).
    specs: list[tuple[str, int | None, list[str]]] = []
    seen: set[str] = set()
    for bullet in content_bullets:
        if not isinstance(bullet, str):
            continue
        canonical = _canonical_heading(bullet)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        specs.append(
            (canonical, _extract_min_entity_count(bullet), _parenthetical_tokens(bullet))
        )

    sections = _parse_sections_struct(draft_markdown)
    entity_count = _count_entity_units(draft_markdown)

    content = evidence
    for canonical, min_count, paren_tokens in specs:
        if min_count is not None:
            # Quantity deliverable — scored by actual structural entity count,
            # immune to prose claiming the threshold was met.
            ok = entity_count >= min_count
        else:
            ok = False
            for idx, (lvl, heading, body) in enumerate(sections):
                # Bind via canonical-token OR parenthetical-token overlap (S83
                # m04), AND-ed with substance so a reworded heading over a stub
                # body still scores missing.
                if not (
                    _heading_matches(canonical, heading)
                    or _tokens_clear_floor(paren_tokens, heading)
                ):
                    continue
                # The section's OWN body keeps the full citation-OR-words rule.
                if _section_is_substantive(body):
                    ok = True
                    break
                # An explicit gap-note own body (non-empty but all-placeholder,
                # e.g. "[Data unavailable in primary sources]") declares the
                # section's own content absent — it is NOT rescued by descendants.
                if body.strip() and not _strip_placeholders(body).strip():
                    continue
                # DSV-045 fix: a heading whose own body is not itself substantive
                # is still delivered when its strictly-deeper descendant prose
                # carries the section (a `## H` whose content lives under `### H.1`
                # children). Recall-biased on purpose — roll up ALL strictly-deeper
                # descendant prose and apply the same citation-OR-words rule. This
                # CAN over-count (a stub borrowing a misnested / off-topic
                # descendant's substance), which is ACCEPTED: a coverage shortfall
                # now DEGRADES rather than catastrophically fails (see
                # evaluate_outcome_gate), so an over-count only softens the gate —
                # it never kills a good report, whereas the inverse (a false
                # MISSING) is the costly error this rollup exists to prevent. The
                # over-count limitations are enumerated in the module docstring.
                rolled = body
                for desc_idx in range(idx + 1, len(sections)):
                    desc_lvl, _desc_heading, desc_body = sections[desc_idx]
                    if desc_lvl <= lvl:
                        break
                    rolled = f"{rolled}\n{desc_body}"
                if _section_is_substantive(rolled):
                    ok = True
                    break
        content.append({"requirement": canonical, "kind": "content", "satisfied": ok})

    return _coverage_result(artifacts, content)


__all__ = ["completeness_check"]
