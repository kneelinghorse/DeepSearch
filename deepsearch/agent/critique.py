"""S66 macroscopic critique stage — span-level annotation validator.

Per the rebuild roadmap §2 (S66 row) and the SOTA evidence base
(``sota_architecture_evidence.md`` §1 citation 8), the dual-actor pattern
inserts a bounded LLM critique call after CodeAgent synthesis that catches unsupported
claims and replaces them with explicit gap notes (``[Data unavailable in
primary sources]``) instead of triggering re-retrieval.

S85 m01 (decision #823) — CONTAINMENT of the whole-document rewrite SPOF.
The critique no longer round-trips the report body through the LLM. Instead:

- The LLM emits STRUCTURED SPAN ANNOTATIONS (a verbatim ``anchor`` substring
  of the draft + a ``verdict``); it never returns the report body.
- A DETERMINISTIC APPLIER renders gap notes onto the ORIGINAL ``draft_markdown``
  by exact-substring replacement. ``output_markdown`` is derived from the
  input draft, never the model's echo. An anchor that is not a unique verbatim
  quote is SKIPPED (recall-safe: the claim is preserved).
- Grounding reads the full ``body`` field of ``sources_collected`` (bounded by
  a per-source + aggregate budget), not the 300-char snippet that "lied" about
  what the fetched page actually said (decision #823, CAT-R002 evidence).
- A missing/unparseable annotations block is a HARD FAILURE — ``error`` is
  populated and the body is returned untouched (``output_markdown ==
  draft_markdown``). The old silent-truncation fallback that shipped a clipped
  LLM echo is gone.
- A pre/post STRUCTURAL DIFF GUARD rejects any transform that drops a heading,
  mutates a span outside an applied annotation, or hollows the report.

Contract:

- ``run_critique(draft_markdown, sources_collected, mission_context)`` is the
  public entry-point; it executes a single LLM call and returns a
  :class:`CritiqueResult`. The signature (3 positional params in this order,
  keyword-only ``model`` and ``citations``) is frozen — the finalize node
  calls it positionally.
- The model uses the same resolved provider/model/request contract as synthesis,
  built via ``codeagent_runner.build_default_model``. No stage-level token or
  temperature override can replace the effective configuration.
- The critique reads only from the ``sources_collected`` accumulator. Parametric
  memory is *never* consulted, and no retrieval tool is imported or called
  (DRM.0.6 §3; enforced by ``test_critique_invariants``).
- The function MUST NOT raise. Every failure mode (LLM error, unparseable
  output, guard rejection) returns a :class:`CritiqueResult` with
  ``output_markdown == draft_markdown``, ``error`` populated, and zero gap
  notes. The caller (finalize) flags ``critique_skipped=True`` and ships the
  pre-critique stripped markdown. S87 may invoke this function once more after
  one separate bounded recovery actor; each invocation remains retrieval-free.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from deepsearch.error_safety import safe_error_from_parts, safe_exception_text
from deepsearch.mission.scope import ScopeAdmission, compile_authored_scope
from deepsearch.report.citation import CitationAnnotation

from .citation_liveness import (
    CLASSIFICATION_DEAD,
    CLASSIFICATION_HALLUCINATED,
    _normalise_url_for_cache,
    build_source_url_set,
    classify_citation,
)
from .codeagent_runner import (
    build_default_model,
)
from .critique_evidence import select_critique_passages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output format — the LLM emits ONLY the delimiter followed by a single JSON
# object describing span annotations. The delimiter is a literal sentinel
# chosen to be unlikely to appear in research markdown. A missing delimiter or
# unparseable JSON is a hard failure (body preserved untouched).
# ---------------------------------------------------------------------------

CRITIQUE_OUTPUT_DELIMITER = "===CRITIQUE_METADATA==="

GAP_NOTE_DEFAULT = "[Data unavailable in primary sources]"

# Snippet preview cap (kept for backward-compat export; NO LONGER the grounding
# cap). The grounding block now reads the full ``body`` field, bounded below.
SOURCE_SNIPPET_MAX_CHARS = 300

# S85 m01 — full-body grounding. Each source contributes up to
# ``SOURCE_GROUNDING_MAX_CHARS`` of its stored ``body`` to the critique prompt,
# and the whole SOURCE LIST is capped at ``GROUNDING_TOTAL_BUDGET_CHARS`` so a
# run with hundreds of sources cannot overflow the model context. The per-source
# cap is reduced when the aggregate budget would otherwise be exceeded, but
# never below the snippet floor (every source keeps at least a preview).
SOURCE_GROUNDING_MAX_CHARS = 2400
GROUNDING_TOTAL_BUDGET_CHARS = 160_000
CRITIQUE_DRAFT_MAX_CHARS = 240_000
CRITIQUE_LIVENESS_MAX_CHARS = 40_000
CRITIQUE_CONTEXT_MAX_CHARS = 450_000
SOURCE_URL_MAX_CHARS = 2_048
SOURCE_TITLE_MAX_CHARS = 500
CRITIQUE_ANNOTATION_MAX_COUNT = 200
CRITIQUE_ANNOTATION_ANCHOR_MAX_CHARS = 2_000
CRITIQUE_ANNOTATION_NOTE_MAX_CHARS = 1_000
CRITIQUE_ANNOTATION_URL_MAX_COUNT = 20
CRITIQUE_ANNOTATION_URL_MAX_CHARS = 2_048

# An anchor must be at least this long to be applied — shorter strings are too
# likely to be non-unique or to match incidental prose. Non-unique anchors are
# skipped anyway; this is a cheap early guard.
MIN_ANCHOR_CHARS = 8

_VALID_VERDICTS = ("unsupported", "hallucinated")

# Heading detection covers every CommonMark heading form so the applier/guard
# can never anchor-delete a section title:
#   - ATX (optionally blockquoted): ``#..###### text`` / ``> ## text``
#   - Setext: a text line underlined by ``===`` (H1) or ``---`` (H2)
# Setext underlines are matched line-wise in ``_heading_line_indices``.
_ATX_HEADING_LINE_RE = re.compile(r"^[ \t]*>{0,3}[ \t]*#{1,6}[ \t]+\S")
_SETEXT_UNDERLINE_RE = re.compile(r"^[ \t]{0,3}(?:=+|-+)[ \t]*$")


CRITIQUE_SYSTEM_PROMPT = """You are a macroscopic critique stage in a deep-research pipeline.
You do NOT rewrite the report. You emit a structured list of SPAN ANNOTATIONS that identify
unsupported claims; a deterministic applier renders them onto the report.

You receive: a completed research DRAFT (markdown), a SOURCE LIST harvested by the upstream
researcher (URL + title + body text per source), the MISSION CONTEXT (title and objective),
and — when available — a per-citation context block.

Your job: find claims in the DRAFT that have NO support anywhere in the SOURCE LIST, and emit
one annotation per unsupported claim. You NEVER output the report body. You have no tools and
cannot retrieve — the source list is the only ground truth available.

Hard rules:
1. Do NOT re-retrieve and do NOT fabricate facts. The source list is the only ground truth.
2. Do NOT score the draft.
3. Bias toward PRESERVATION. Annotate a claim ONLY when it has no support anywhere in the
   source list. A claim with even partial source backing is left alone — do not annotate it.
   When in doubt, do not annotate.
4. Each annotation's "anchor" MUST be an EXACT, VERBATIM substring copied from the DRAFT —
   long enough to occur exactly once (quote the full sentence or claim). Copy it
   character-for-character. If your anchor is not a unique verbatim quote, the applier skips
   it and the claim is preserved.
5. The applier replaces each anchored claim with this exact gap-note form:
   [Data unavailable in primary sources]
6. Do NOT annotate headings, section titles, table header rows, or reference-list lines —
   only claim prose. Never annotate a span that would remove structure.
7. The per-citation context block (when present) pre-classifies the draft's citations. A
   citation annotated 'hallucinated' (fabricated URL, not in sources AND unreachable) means
   its backed claim is unsupported — annotate that claim with verdict "hallucinated".
   Citations annotated 'dead', 'orphan', 'live', or 'unknown' (live=None) are PRESERVED — do
   NOT annotate a claim solely because its citation is dead, orphan, or unknown; dead
   citations are flagged unverified downstream, never deleted. This rule fires off the
   per-citation context block, not your own guesses about liveness.
8. The SOURCE VISIBILITY notice tells you whether source rows or bodies were omitted from
   this bounded prompt. Missing text in an incomplete view is NOT evidence that a claim is
   unsupported. Preserve the claim unless the entire collected source corpus was fully visible.
   This does not weaken rule 7: a pre-classified hallucinated citation must still be annotated.

9. The declared author context is a statement of the user's own system, not public-source
   evidence. Do not call those facts unsupported because public pages cannot verify them.
   Record them as declared_context with a verbatim quote from C1 when relevant.
10. Assess claims using the labeled evidence passages. Report supported, contradicted,
    unassessed, or declared_context separately from destructive annotations. Each supported
    or contradicted judgment must cite visible evidence_ids and exact evidence_quotes.
    Selected passages cannot prove absence in a whole source or corpus. Contradictions are
    advisory findings; do not relabel them unsupported to force deletion. Never infer an
    accuracy pass from zero findings, zero evaluated claims, or missing evidence.

Output format (mandatory). Emit ONLY the delimiter line followed by a single JSON object — no
report body, no prose before or after the JSON:

===CRITIQUE_METADATA===
{"claim_count": <int — total distinct claims you evaluated>, "annotations": [{"anchor":
"<verbatim substring copied from the draft>", "verdict": "unsupported" or "hallucinated",
"note": "<short reason>", "citation_urls": ["<relevant URL from the citation context>"]}],
"assessments": [{"anchor": "<unique exact draft claim>", "verdict": "supported" or
"contradicted" or "unassessed" or "declared_context", "evidence_ids": ["E1.1"],
"evidence_quotes": {"E1.1": "<exact 8..600 character passage quote>"}, "note": "<reason>"}]}

The ===CRITIQUE_METADATA=== delimiter is literal; emit it exactly. If there are no unsupported
claims, emit an empty annotations list. The JSON must parse — a missing delimiter or
unparseable JSON is a hard failure that leaves the report untouched.
"""


# ---------------------------------------------------------------------------
# Public dataclass + errors.
# ---------------------------------------------------------------------------


class CritiqueTransformError(Exception):
    """Raised internally when the applier output fails the structural guard.

    Never propagates out of :func:`run_critique` — it is caught and converted
    into a ``CritiqueResult`` with ``error`` populated and the body untouched.
    """


@dataclass(slots=True)
class CritiqueAnnotation:
    """One bounded, auditable critique finding and deterministic disposition."""

    anchor: str
    verdict: str
    note: str = ""
    citation_urls: tuple[str, ...] = ()
    applied: bool = False
    reason: str = "not_processed"

    def to_dict(self) -> dict[str, Any]:
        """Return a stable telemetry shape."""

        return {
            "anchor": self.anchor,
            "verdict": self.verdict,
            "note": self.note,
            "citation_urls": list(self.citation_urls),
            "applied": self.applied,
            "reason": self.reason,
        }


@dataclass(slots=True)
class CritiqueResult:
    """Structured output of one critique pass."""

    output_markdown: str
    claim_count: int
    gap_notes_inserted: int
    runtime_seconds: float
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0, "total": 0}
    )
    # False means the recorded values are a lower bound because the provider
    # call returned no usable usage envelope (or failed after dispatch).
    token_usage_complete: bool = False
    error: Optional[str] = None
    grounding_telemetry: dict[str, int] = field(default_factory=dict)
    annotations: list[CritiqueAnnotation] = field(default_factory=list)
    annotations_dropped: int = 0
    provider_attempts: list[dict[str, Any]] = field(default_factory=list)
    assessment: dict[str, Any] = field(default_factory=dict)
    evidence_manifest: list[dict[str, Any]] = field(default_factory=list)

    def annotation_summary(self) -> dict[str, Any]:
        """Return exact counts derived from the bounded annotation ledger."""

        applied = sum(1 for annotation in self.annotations if annotation.applied)
        skipped = len(self.annotations) - applied
        by_reason: dict[str, int] = {}
        for annotation in self.annotations:
            by_reason[annotation.reason] = by_reason.get(annotation.reason, 0) + 1
        return {
            "total": len(self.annotations),
            "applied": applied,
            "skipped": skipped,
            "dropped": int(self.annotations_dropped),
            "by_reason": by_reason,
            "reconciled": applied == self.gap_notes_inserted,
        }


@dataclass(slots=True)
class _SpanAnnotation:
    """One LLM-emitted annotation targeting a claim in the draft."""

    anchor: str
    verdict: str
    note: str = ""
    citation_urls: tuple[str, ...] = ()
    parse_reason: str | None = None


@dataclass(slots=True)
class _GroundingVisibility:
    """Internal proof of which collected evidence the critic saw in full."""

    fully_visible_urls: set[str] = field(default_factory=set)
    corpus_complete: bool = True
    passages: list[dict[str, Any]] = field(default_factory=list)
    passage_text: dict[str, str] = field(default_factory=dict)
    draft_chars_visible: int | None = None
    author_context: str = ""
    declared_context_anchors: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Internal helpers — kept module-private; tests exercise them indirectly via
# ``run_critique`` (plus a few direct guard/applier unit tests) so the public
# surface stays small.
# ---------------------------------------------------------------------------


def _source_url_key(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.query,
                "",
            )
        )
    except Exception:  # noqa: BLE001
        return raw


def _format_source_row(index: int, source: Mapping[str, Any]) -> str:
    url = source.get("url", "")
    title = source.get("title", "") or "(untitled)"
    snippet = source.get("snippet", "") or "(no snippet)"
    return f"[{index}] {title}\n    URL: {url}\n    Content: {snippet}"


def _prepare_sources_with_telemetry(
    sources: Sequence[Mapping[str, Any]] | None,
    *,
    citations: Sequence[CitationAnnotation] | None = None,
    per_source_cap: int = SOURCE_GROUNDING_MAX_CHARS,
    total_budget: int = GROUNDING_TOTAL_BUDGET_CHARS,
    grounding_visibility: _GroundingVisibility | None = None,
    draft_markdown: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Prepare cited-first grounding rows under one exact rendered-char budget.

    S85 m01 — grounding reads the stored ``body`` (full fetched text) rather
    than the 300-char ``snippet`` that "lied" about page content. The whole
    block is HARD-bounded by ``total_budget``, including URL/title/rendering
    overhead. Live cited sources are ordered first, each source contributes at
    most ``per_source_cap`` content chars, and overflow rows are dropped.
    Entries without a URL are dropped. The output key remains ``snippet`` so
    :func:`_format_source_block` renders unchanged.
    """

    telemetry = {
        "sources_available": 0,
        "sources_invalid": 0,
        "sources_included": 0,
        "sources_dropped": 0,
        "sources_truncated": 0,
        "metadata_fields_truncated": 0,
        "cited_live_available": 0,
        "cited_live_included": 0,
        "cited_available": 0,
        "cited_included": 0,
        "cited_dropped": 0,
        "cited_missing_from_sources": 0,
        "source_block_chars": 0,
        "source_content_chars": 0,
        "source_budget_chars": total_budget,
        "estimated_source_tokens": 0,
        "source_urls_fully_visible": 0,
        "source_urls_incomplete": 0,
        "sources_provenance_incomplete": 0,
        "grounding_context_complete": 1,
    }
    if grounding_visibility is not None:
        grounding_visibility.fully_visible_urls.clear()
        grounding_visibility.passages.clear()
        grounding_visibility.passage_text.clear()
        grounding_visibility.corpus_complete = True
    live_keys: list[str] = []
    cited_keys: list[str] = []
    for citation in citations or []:
        key = _source_url_key(getattr(citation, "url", ""))
        if not key:
            continue
        if key not in cited_keys:
            cited_keys.append(key)
        if getattr(citation, "live", None) is True and key not in live_keys:
            live_keys.append(key)
    if not sources:
        telemetry["cited_missing_from_sources"] = len(cited_keys)
        if cited_keys:
            telemetry["grounding_context_complete"] = 0
            if grounding_visibility is not None:
                grounding_visibility.corpus_complete = False
        return [], telemetry
    valid = [
        src
        for src in sources
        if isinstance(src, Mapping) and str(src.get("url") or "").strip()
    ]
    telemetry["sources_available"] = len(valid)
    telemetry["sources_invalid"] = len(sources) - len(valid)
    if not valid:
        telemetry["cited_missing_from_sources"] = len(cited_keys)
        telemetry["grounding_context_complete"] = 0
        if grounding_visibility is not None:
            grounding_visibility.corpus_complete = False
        return [], telemetry

    source_keys = {_source_url_key(src.get("url")) for src in valid}
    provenance_incomplete_ids = {
        id(src)
        for src in valid
        if not (
            isinstance(src.get("body"), str)
            and bool(str(src.get("body") or "").strip())
        )
        or (
            bool(str(src.get("content_quality") or "").strip())
            and str(src.get("content_quality") or "").strip() != "readable_body"
        )
        or bool(src.get("body_truncated"))
    }
    telemetry["sources_provenance_incomplete"] = sum(
        1 for src in valid if id(src) in provenance_incomplete_ids
    )

    # S88 lever 1 — front ALL draft-cited sources (not only live==True) ahead of
    # uncited sources, live-cited first (they carry the fetched bodies). A claim
    # whose cited source is present-but-non-live (dead/unknown-with-body) must
    # still reach the critique's context so it is judged against its own evidence
    # rather than flagged for a packer eviction it never saw. This REORDERS
    # within the same ``total_budget`` (flat token cost) — the budget is NOT
    # raised. Net effect on false positives is empirical (fronting cited evicts
    # uncited corroborators from the tail); the ``cited_*`` telemetry makes every
    # eviction auditable and drives the measured re-run gate (cited_dropped == 0).
    telemetry["cited_live_available"] = sum(
        1 for src in valid if _source_url_key(src.get("url")) in live_keys
    )

    by_key = {_source_url_key(src.get("url")): src for src in valid}
    telemetry["cited_available"] = len([key for key in cited_keys if key in by_key])
    telemetry["cited_missing_from_sources"] = len(
        [key for key in cited_keys if key not in by_key]
    )
    # live-cited first, then the remaining cited (first-seen order), then uncited.
    front_keys = live_keys + [key for key in cited_keys if key not in live_keys]
    ordered: list[Mapping[str, Any]] = [
        by_key[key] for key in front_keys if key in by_key
    ]
    ordered_ids = {id(src) for src in ordered}
    ordered.extend(src for src in valid if id(src) not in ordered_ids)

    out: list[dict[str, Any]] = []
    block_chars = 0
    content_chars = 0
    included_keys: set[str] = set()
    incomplete_keys: set[str] = set()
    for src in ordered:
        raw_url = str(src.get("url") or "").strip()
        source_key = _source_url_key(raw_url)
        raw_title = str(src.get("title") or "").strip()
        url = raw_url[:SOURCE_URL_MAX_CHARS]
        title = raw_title[:SOURCE_TITLE_MAX_CHARS]
        telemetry["metadata_fields_truncated"] += int(len(raw_url) > len(url))
        telemetry["metadata_fields_truncated"] += int(len(raw_title) > len(title))
        body_raw = src.get("body")
        has_readable_body = isinstance(body_raw, str) and bool(body_raw.strip())
        if not has_readable_body:
            body_raw = src.get("snippet")
        if id(src) in provenance_incomplete_ids:
            # A search snippet, explicitly non-readable payload, or source_fetch
            # body already truncated upstream is not a complete negative-evidence
            # view even when its bounded representation fits this prompt exactly.
            incomplete_keys.add(source_key)
        original_text = str(body_raw or "")
        text = original_text.strip()
        original_text_chars = len(text)
        text = text[:per_source_cap]
        passages: list[dict[str, Any]] = []
        complete_body_visible = original_text_chars <= per_source_cap
        if draft_markdown is not None:
            claims = "\n".join(
                draft_markdown[max(0, c.start_index):min(len(draft_markdown), c.end_index)]
                for c in citations or [] if _source_url_key(c.url) == source_key
            )
            if not claims:
                claims = raw_title  # Uncited rows do not expand every draft claim.
            fixed = len(_format_source_row(len(out) + 1, {
                "url": url, "title": title, "snippet": ""
            }))
            available = min(per_source_cap, total_budget - block_chars - (2 if out else 0) - fixed)
            if available <= 0:
                break
            text, passages, complete_body_visible = select_critique_passages(
                original_text, claims, budget=available, source_index=len(out) + 1, url=raw_url,
            )
        row = {"url": url, "title": title, "snippet": text}
        separator_chars = 2 if out else 0
        rendered = _format_source_row(len(out) + 1, row)
        remaining = total_budget - block_chars - separator_chars
        if len(rendered) > remaining:
            fixed = len(_format_source_row(len(out) + 1, {**row, "snippet": ""}))
            available_text = max(0, remaining - fixed)
            if available_text <= 0:
                break
            row["snippet"] = text[:available_text]
            rendered = _format_source_row(len(out) + 1, row)
        if not complete_body_visible or (
            draft_markdown is None and original_text_chars > len(str(row["snippet"]))
        ):
            telemetry["sources_truncated"] += 1
            incomplete_keys.add(source_key)
        if len(raw_url) > len(url):
            # The critic cannot reliably associate a truncated URL with the
            # draft citation even when its body happened to fit.
            incomplete_keys.add(source_key)
        out.append(row)
        if grounding_visibility is not None:
            grounding_visibility.passages.extend(passages)
            grounding_visibility.passage_text.update({
                p["evidence_id"]: original_text[p["start"]:p["end"]] for p in passages
            })
        included_keys.add(source_key)
        block_chars += separator_chars + len(rendered)
        content_chars += len(str(row["snippet"]))

    # Rows after the first budget miss were never visible. A duplicate URL is
    # fully visible only when every collected row for that URL was included in
    # full; any omitted duplicate could contain the supporting passage.
    for src in ordered[len(out) :]:
        incomplete_keys.add(_source_url_key(src.get("url")))
    fully_visible_keys = (source_keys & included_keys) - incomplete_keys
    incomplete_source_keys = source_keys - fully_visible_keys
    telemetry["sources_included"] = len(out)
    telemetry["sources_dropped"] = len(valid) - len(out)
    telemetry["cited_live_included"] = sum(
        1 for key in live_keys if key in included_keys
    )
    telemetry["cited_included"] = len(
        [key for key in cited_keys if key in included_keys]
    )
    telemetry["cited_dropped"] = (
        telemetry["cited_available"] - telemetry["cited_included"]
    )
    telemetry["source_block_chars"] = block_chars
    telemetry["source_content_chars"] = content_chars
    telemetry["estimated_source_tokens"] = (block_chars + 3) // 4
    telemetry["source_urls_fully_visible"] = len(fully_visible_keys)
    telemetry["source_urls_incomplete"] = len(incomplete_source_keys)
    telemetry["grounding_context_complete"] = int(
        telemetry["sources_invalid"] == 0
        and telemetry["sources_dropped"] == 0
        and telemetry["sources_truncated"] == 0
        and telemetry["cited_missing_from_sources"] == 0
        and not incomplete_keys
    )
    if grounding_visibility is not None:
        grounding_visibility.fully_visible_urls.update(fully_visible_keys)
        grounding_visibility.corpus_complete = bool(
            telemetry["grounding_context_complete"]
        )
    return out, telemetry


def _prepare_sources(
    sources: Sequence[Mapping[str, Any]] | None,
    *,
    per_source_cap: int = SOURCE_GROUNDING_MAX_CHARS,
    total_budget: int = GROUNDING_TOTAL_BUDGET_CHARS,
) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only prepared grounding rows."""

    prepared, _ = _prepare_sources_with_telemetry(
        sources,
        per_source_cap=per_source_cap,
        total_budget=total_budget,
    )
    return prepared


def _format_source_block(sources: Sequence[Mapping[str, Any]]) -> str:
    """Render the source list as a numbered text block for the user prompt."""

    if not sources:
        return "(no sources collected)"
    lines: list[str] = []
    for i, src in enumerate(sources, start=1):
        lines.append(_format_source_row(i, src))
    return "\n\n".join(lines)


def _classify_citation_for_prompt(
    annotation: CitationAnnotation,
    sources_url_set: set[str],
) -> str:
    """Return the citation classification used in the liveness-context block.

    S82 m05 (PD135): delegates to the SHARED
    :func:`deepsearch.agent.citation_liveness.classify_citation` so the
    critique and the liveness gate agree on what "in sources" means — the
    ``sources_url_set`` MUST be built via ``build_source_url_set`` (normalised),
    which :func:`_build_critique_messages` now does. The classifier is 5-state:
    live / dead / orphan (reachable but uncollected) / hallucinated (unreachable
    AND uncollected) / unknown.
    """

    return classify_citation(
        live=annotation.live, url=annotation.url, source_url_set=sources_url_set
    )


_CLASSIFICATION_DESCRIPTIONS: dict[str, str] = {
    "live": "live",
    "dead": (
        "dead (URL in sources_collected but GET-confirmed gone — 404/410 or "
        "unresolvable host)"
    ),
    "orphan": "orphan (not in sources but reachable — uncollected/normalisation mismatch)",
    "hallucinated": "hallucinated (not in sources AND GET-confirmed unreachable — fabricated)",
    "unknown": (
        "unknown (probe inconclusive — bot-wall/rate-limit/timeout or budget "
        "exhausted; not dead evidence)"
    ),
}


def _format_citation_liveness_block(
    citations: Sequence[CitationAnnotation],
    sources_url_set: set[str],
) -> str:
    """Render the per-citation liveness context block for the critique prompt.

    Built deterministically from the populated annotations. Returns the
    section text (heading + bullets + rule reminder) ready to be appended
    to the user prompt body. Returns an empty string when ``citations`` is
    empty so callers don't have to special-case missing-vs-empty.
    """

    if not citations:
        return ""

    bullets: list[str] = []
    for ann in citations:
        classification = _classify_citation_for_prompt(ann, sources_url_set)
        descriptor = _CLASSIFICATION_DESCRIPTIONS[classification]
        bullets.append(f"- [cite:{ann.url}]  →  {descriptor}")

    return (
        "# Citation Liveness Context\n\n"
        "The following citations in the draft above have been pre-checked. "
        "Annotate a claim whose citation is 'hallucinated'; keep dead, orphan, "
        "live, and unknown citations' claims as-is:\n\n"
        + "\n".join(bullets)
        + "\n\nApply rule 7 to each hallucinated citation; preserve the rest."
    )


def _build_critique_messages(
    draft_markdown: str,
    sources_collected: Sequence[Mapping[str, Any]],
    mission_context: Mapping[str, Any],
    citations: Sequence[CitationAnnotation] | None = None,
    grounding_telemetry: dict[str, int] | None = None,
    grounding_visibility: _GroundingVisibility | None = None,
    scope_admission: ScopeAdmission | None = None,
) -> list[dict[str, str]]:
    """Build the ``[{role, content}]`` message list for the model call.

    ``citations`` is the S68 live-checked annotation list; when non-empty
    the user prompt gains a "# Citation Liveness Context" section that
    feeds the hallucinated-claim rule. ``None`` / empty omits the section
    (backward-compatible with direct test usage).
    """

    admission = scope_admission or ScopeAdmission(compile_authored_scope(mission_context))
    source_count_before_scope = len(sources_collected)
    sources_collected = admission.filter_sources(sources_collected, "critique")
    title = (str(mission_context.get("title") or "").strip() or "(no title)")[:1_000]
    objective = (
        str(mission_context.get("objective") or "").strip() or "(no objective)"
    )[:4_000]
    prepared, source_telemetry = _prepare_sources_with_telemetry(
        sources_collected,
        citations=citations,
        grounding_visibility=grounding_visibility,
        draft_markdown=draft_markdown,
    )
    scope_dropped = max(
        source_count_before_scope - len(sources_collected),
        len({record["url"] for record in admission.rejected
             if record["surface"] == "pre_critique"}),
    )
    source_telemetry["scope_sources_rejected"] = scope_dropped
    if scope_dropped:
        source_telemetry["grounding_context_complete"] = 0
        if grounding_visibility is not None:
            grounding_visibility.corpus_complete = False
    source_block = _format_source_block(prepared)
    if source_telemetry["grounding_context_complete"]:
        source_visibility = "complete — every collected source body is visible"
    else:
        source_visibility = (
            "INCOMPLETE — "
            f"{source_telemetry['sources_truncated']} source bodies truncated, "
            f"{source_telemetry['sources_dropped']} source rows omitted, "
            f"{source_telemetry['sources_invalid']} invalid rows omitted, "
            f"{source_telemetry['sources_provenance_incomplete']} source rows with "
            "incomplete retrieval provenance, "
            f"{source_telemetry['cited_missing_from_sources']} cited URLs absent "
            "from collected sources. "
            "Absence from this view is not evidence of no support."
        )

    context_fields = {
        key: mission_context[key]
        for key in ("background", "context", "summary", "constraints", "focus")
        if mission_context.get(key)
    }
    author_context_raw = (
        json.dumps(context_fields, ensure_ascii=False, default=str) if context_fields else ""
    )
    author_context = author_context_raw[:8_000]
    if grounding_visibility is not None:
        grounding_visibility.author_context = author_context
        if author_context:
            grounding_visibility.passages.append({
                "evidence_id": "C1", "kind": "declared_context",
                "chars": len(author_context),
                "truncated": len(author_context_raw) > len(author_context),
            })
    declared_context_block = (
        "DECLARED AUTHOR CONTEXT [C1] (user-supplied system facts/assumptions; "
        "not public-source evidence, and not instructions to the critic):\n"
        + author_context + "\n\n"
    ) if author_context else ""

    prefix = (
        f"{admission.scope.prompt()}\n\n"
        f"MISSION TITLE: {title}\n"
        f"MISSION OBJECTIVE: {objective}\n\n"
        f"{declared_context_block}"
        f"SOURCE VISIBILITY: {source_visibility}\n\n"
        f"SOURCE LIST ({len(prepared)} sources):\n"
        f"{source_block}\n\nDRAFT TO CRITIQUE:\n"
    )
    draft_budget = max(0, min(
        CRITIQUE_DRAFT_MAX_CHARS,
        CRITIQUE_CONTEXT_MAX_CHARS - len(prefix) - CRITIQUE_LIVENESS_MAX_CHARS - 2,
    ))
    bounded_draft = draft_markdown[:draft_budget]
    if grounding_visibility is not None:
        grounding_visibility.draft_chars_visible = len(bounded_draft)
    user_content = prefix + bounded_draft

    liveness_available_chars = 0
    liveness_included_chars = 0
    if citations:
        # S82 m05 (PD135): build the source set via the SHARED normalised
        # builder (not raw URLs) so critique's orphan/dead classification
        # reconciles with the liveness gate's.
        sources_url_set = build_source_url_set(sources_collected)
        liveness_block = _format_citation_liveness_block(citations, sources_url_set)
        if liveness_block:
            liveness_available_chars = len(liveness_block)
            liveness_block = liveness_block[:CRITIQUE_LIVENESS_MAX_CHARS]
            liveness_included_chars = len(liveness_block)
            user_content = f"{user_content}\n\n{liveness_block}"

    if len(user_content) > CRITIQUE_CONTEXT_MAX_CHARS:
        # Component caps should keep this unreachable; retain a hard final
        # guard so future prompt-label growth cannot silently break the bound.
        user_content = user_content[:CRITIQUE_CONTEXT_MAX_CHARS]

    if grounding_telemetry is not None:
        grounding_telemetry.update(source_telemetry)
        grounding_telemetry.update(
            {
                "author_context_chars_included": len(author_context),
                "author_context_chars_dropped": max(
                    0, len(author_context_raw) - len(author_context)
                ),
                "draft_chars_available": len(draft_markdown),
                "draft_chars_included": len(bounded_draft),
                "draft_chars_dropped": max(0, len(draft_markdown) - len(bounded_draft)),
                "liveness_chars_available": liveness_available_chars,
                "liveness_chars_included": liveness_included_chars,
                "liveness_chars_dropped": max(
                    0, liveness_available_chars - liveness_included_chars
                ),
                "context_chars": len(user_content),
                "context_budget_chars": CRITIQUE_CONTEXT_MAX_CHARS,
                "estimated_context_tokens": (len(user_content) + 3) // 4,
            }
        )

    return [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_response_content(response: Any) -> str:
    """Pull a string body out of whatever shape the model client returned."""

    if isinstance(response, str):
        return response
    for attr in ("content", "text"):
        if hasattr(response, attr):
            v = getattr(response, attr)
            if isinstance(v, str):
                return v
    if isinstance(response, Mapping):
        v = response.get("content") or response.get("text")
        if isinstance(v, str):
            return v
    return str(response)


def _extract_token_usage(response: Any) -> tuple[dict[str, int], bool]:
    """Return observed usage plus whether its total is provider-complete."""

    zero = {"input": 0, "output": 0, "total": 0}
    tu = getattr(response, "token_usage", None)
    if tu is None and isinstance(response, Mapping):
        tu = response.get("token_usage")
    if tu is None:
        return zero, False

    def _read(names: tuple[str, ...]) -> tuple[int, bool]:
        for name in names:
            if isinstance(tu, Mapping):
                if name not in tu or tu.get(name) is None:
                    continue
                raw = tu.get(name)
            else:
                if not hasattr(tu, name):
                    continue
                raw = getattr(tu, name)
                if raw is None:
                    continue
            if isinstance(raw, bool):
                return 0, False
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError):
                return 0, False
            return (value, True) if value >= 0 else (0, False)
        return 0, False

    in_t, has_input = _read(("input_tokens", "input"))
    out_t, has_output = _read(("output_tokens", "output"))
    total, has_total = _read(("total_tokens", "total"))
    component_total = in_t + out_t
    envelope_consistent = True
    if not has_total:
        total = component_total
    elif total < component_total:
        # Preserve the strongest observed lower bound, but never call a
        # contradictory provider envelope complete.
        total = component_total
        envelope_consistent = False
    complete = bool(
        envelope_consistent and (has_total or (has_input and has_output))
    )
    return {"input": in_t, "output": out_t, "total": total}, complete


def _extract_json_object(
    text: str,
    *,
    predicate: Any | None = None,
) -> Optional[dict[str, Any]]:
    """Extract the first balanced ``{...}`` JSON object satisfying ``predicate``.

    Tries ``json.loads`` on the whole (stripped) string first, then scans for
    brace-balanced regions (string-literal aware). ``predicate`` is a callable
    ``dict -> bool``; only a dict for which it returns True is accepted, so a
    stray leading object (e.g. one lacking an ``annotations`` list) is SKIPPED
    rather than shadowing the real block. Returns ``None`` when no qualifying
    object is found.
    """

    def _ok(obj: Any) -> bool:
        return isinstance(obj, dict) and (predicate is None or bool(predicate(obj)))

    def _as_dict(obj: Any) -> dict[str, Any]:
        return dict(obj)

    stripped = text.strip()
    if stripped:
        try:
            whole = json.loads(stripped)
        except (ValueError, TypeError):
            whole = None
        if _ok(whole):
            return _as_dict(whole)

    i = 0
    n = len(text)
    while True:
        start = text.find("{", i)
        if start == -1:
            return None
        depth = 0
        in_str = False
        escape = False
        end = -1
        for j in range(start, n):
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            # No balanced close from this '{' — advance one char and retry.
            i = start + 1
            continue
        try:
            obj = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            obj = None
        if _ok(obj):
            return _as_dict(obj)
        # Parsed-but-unqualified: skip the whole object; unparseable: step one.
        i = (end + 1) if obj is not None else (start + 1)


def _critique_payload(raw: str) -> dict[str, Any] | None:
    """Select the same structured object for findings and assessment facts."""
    if not isinstance(raw, str) or CRITIQUE_OUTPUT_DELIMITER not in raw:
        return None

    _, _, footer = raw.partition(CRITIQUE_OUTPUT_DELIMITER)
    # Find the annotations object, resisting stray leading dicts (adversarial
    # finding: the first balanced dict could shadow the real block and gap-note
    # the wrong span with error=None). Two-pass: prefer the first object with a
    # NON-EMPTY annotations list (so an empty ``{"annotations":[]}`` stray can't
    # shadow real findings), then fall back to the first list-annotations object
    # (a legitimate "no unsupported claims" result is a single empty list).
    obj = _extract_json_object(
        footer,
        predicate=lambda o: isinstance(o.get("annotations"), list) and bool(o["annotations"]),
    )
    if obj is None:
        obj = _extract_json_object(
            footer, predicate=lambda o: isinstance(o.get("annotations"), list)
        )
    return obj


def _parse_annotations(
    raw: str,
) -> Optional[tuple[list[_SpanAnnotation], int, int]]:
    """Parse the model output into findings, claim count, and dropped count.

    Returns ``None`` on a HARD FAILURE — the delimiter is absent, or the block
    after it does not parse into a JSON object with a list ``annotations`` key.
    A ``None`` return means the caller must leave the report body untouched and
    populate ``error`` (S85 m01 — no silent-truncation echo).
    """

    obj = _critique_payload(raw)
    if obj is None:
        return None

    anns_raw = obj["annotations"]

    annotations: list[_SpanAnnotation] = []
    invalid_count = 0
    bounded_raw = anns_raw[:CRITIQUE_ANNOTATION_MAX_COUNT]
    for item in bounded_raw:
        if not isinstance(item, Mapping):
            invalid_count += 1
            continue
        anchor = item.get("anchor")
        if not isinstance(anchor, str) or not anchor.strip():
            invalid_count += 1
            continue
        # TraceLab 043 requires applied anchors to be exact, trimmed claims.
        # Keep the prior recall-safe rule: outer whitespace means the model did
        # not quote a verbatim draft span, so retain a trimmed audit value but
        # mark it skipped rather than applying a shorter incidental substring.
        trimmed_anchor = anchor.strip()
        parse_reason = (
            "anchor_outer_whitespace" if anchor != trimmed_anchor else None
        )
        anchor = trimmed_anchor
        if len(anchor) > CRITIQUE_ANNOTATION_ANCHOR_MAX_CHARS:
            anchor = anchor[:CRITIQUE_ANNOTATION_ANCHOR_MAX_CHARS]
            parse_reason = parse_reason or "anchor_too_long"
        verdict_raw = item.get("verdict")
        verdict = verdict_raw if verdict_raw in _VALID_VERDICTS else "unsupported"
        if verdict_raw not in _VALID_VERDICTS:
            parse_reason = parse_reason or "invalid_verdict"
        note = str(item.get("note") or "")[:CRITIQUE_ANNOTATION_NOTE_MAX_CHARS]
        citation_urls_raw = item.get("citation_urls")
        citation_urls: list[str] = []
        if isinstance(citation_urls_raw, Sequence) and not isinstance(
            citation_urls_raw, (str, bytes)
        ):
            for value in citation_urls_raw:
                if not isinstance(value, str) or not value.startswith(("http://", "https://")):
                    continue
                bounded_url = value[:CRITIQUE_ANNOTATION_URL_MAX_CHARS]
                if bounded_url not in citation_urls:
                    citation_urls.append(bounded_url)
                if len(citation_urls) >= CRITIQUE_ANNOTATION_URL_MAX_COUNT:
                    break
        annotations.append(
            _SpanAnnotation(
                anchor=anchor,
                verdict=verdict,
                note=note,
                citation_urls=tuple(citation_urls),
                parse_reason=parse_reason,
            )
        )

    claim_count_raw = obj.get("claim_count")
    claim_count = len(annotations)
    if isinstance(claim_count_raw, int) and not isinstance(claim_count_raw, bool):
        claim_count = claim_count_raw
    elif isinstance(claim_count_raw, str):
        try:
            claim_count = int(claim_count_raw)
        except ValueError:
            pass
    if claim_count < 0:
        claim_count = len(annotations)

    claim_count = max(claim_count, len(annotations))
    dropped_count = max(0, len(anns_raw) - len(bounded_raw)) + invalid_count
    return annotations, claim_count, dropped_count


def _assessment_record(
    raw: str, draft: str, visibility: _GroundingVisibility, declared_count: int,
) -> dict[str, Any]:
    """Validate quoted evidence provenance while leaving semantic judging explicit."""
    payload = _critique_payload(raw) or {}
    reported = payload.get("assessments")
    reported = reported if isinstance(reported, list) else []
    manifest = {p["evidence_id"]: p for p in visibility.passages}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = max(0, len(reported) - CRITIQUE_ANNOTATION_MAX_COUNT)
    for item in reported[:CRITIQUE_ANNOTATION_MAX_COUNT]:
        if not isinstance(item, Mapping):
            dropped += 1
            continue
        anchor = item.get("anchor")
        if not isinstance(anchor, str) or not anchor or anchor in seen:
            dropped += 1
            continue
        seen.add(anchor)
        anchor = anchor[:CRITIQUE_ANNOTATION_ANCHOR_MAX_CHARS]
        loc, reason = _locate_anchor_with_reason(draft, anchor)
        verdict = item.get("verdict")
        if not isinstance(verdict, str) or verdict not in {
            "supported", "contradicted", "unassessed", "declared_context"
        }:
            verdict, reason = "unassessed", "invalid_verdict"
        ids = item.get("evidence_ids")
        ids = ids if isinstance(ids, list) else []
        quotes = item.get("evidence_quotes")
        quotes = quotes if isinstance(quotes, Mapping) else {}
        evidence: list[dict[str, Any]] = []
        for evidence_id in ids[:3]:
            if not isinstance(evidence_id, str) or evidence_id not in manifest:
                continue
            quote = quotes.get(evidence_id)
            passage = visibility.author_context if evidence_id == "C1" else (
                visibility.passage_text.get(evidence_id, "")
            )
            if not isinstance(quote, str) or not 8 <= len(quote) <= 600 or quote not in passage:
                continue
            entry = dict(manifest[evidence_id])
            entry["quote"] = quote
            if entry["kind"] == "source_passage":
                entry["quote_start"] = entry["start"] + passage.index(quote)
                entry["quote_end"] = entry["quote_start"] + len(quote)
            evidence.append(entry)
        required_kind = "declared_context" if verdict == "declared_context" else "source_passage"
        traceable = any(e["kind"] == required_kind for e in evidence)
        if loc is None or (loc and _range_touches_heading(draft, *loc)):
            verdict, reason = "unassessed", reason or "structure_overlap"
        elif visibility.draft_chars_visible is not None and loc[1] > visibility.draft_chars_visible:
            verdict, reason = "unassessed", "anchor_not_visible"
        elif len(item["anchor"]) > CRITIQUE_ANNOTATION_ANCHOR_MAX_CHARS:
            verdict, reason = "unassessed", "anchor_too_long"
        elif verdict != "unassessed" and (not traceable or len(evidence) != len(ids)):
            verdict, reason = "unassessed", "no_visible_quoted_evidence"
        elif verdict != "unassessed":
            reason = "traceable_model_judgment"
        if verdict == "declared_context":
            visibility.declared_context_anchors.add(anchor)
        rows.append({
            "anchor": anchor, "verdict": verdict, "reason": reason or "model_unassessed",
            "evidence": evidence, "note": str(item.get("note") or "")[:1000],
        })
    counts = {v: sum(row["verdict"] == v for row in rows)
              for v in ("supported", "contradicted", "unassessed", "declared_context")}
    assessed = counts["supported"] + counts["contradicted"]
    return {
        "version": 1, "status": "partial" if assessed else "unassessed",
        "scope": "model_selected_anchors", "assessed": assessed, **counts,
        "reported": len(rows), "dropped": dropped, "records": rows,
        "declared_claim_count": declared_count,
        "report_claim_count": None, "unreported_claims": "unknown", "coverage_ratio": None,
        "grounding_context_complete": visibility.corpus_complete, "accuracy_pass": None,
    }


def _heading_line_indices(text: str) -> set[int]:
    """Return the set of line indices that are part of ANY heading.

    Covers ATX (incl. blockquoted) heading lines, plus BOTH lines of a setext
    heading (the title line and its ``===``/``---`` underline). Over-inclusive
    by design — protecting a stray thematic-break/underline line from being
    anchor-deleted is the safe direction.
    """

    lines = text.split("\n")
    idx: set[int] = set()
    for i, line in enumerate(lines):
        if _ATX_HEADING_LINE_RE.match(line):
            idx.add(i)
    for i in range(1, len(lines)):
        if _SETEXT_UNDERLINE_RE.match(lines[i]):
            prev = lines[i - 1]
            if (
                prev.strip()
                and not _ATX_HEADING_LINE_RE.match(prev)
                and not _SETEXT_UNDERLINE_RE.match(prev)
            ):
                idx.add(i)       # underline line
                idx.add(i - 1)   # title line
    return idx


def _count_headings(text: str) -> int:
    """Count ATX + setext headings (see :func:`_heading_line_indices`)."""

    lines = text.split("\n")
    count = sum(1 for line in lines if _ATX_HEADING_LINE_RE.match(line))
    for i in range(1, len(lines)):
        if _SETEXT_UNDERLINE_RE.match(lines[i]):
            prev = lines[i - 1]
            if (
                prev.strip()
                and not _ATX_HEADING_LINE_RE.match(prev)
                and not _SETEXT_UNDERLINE_RE.match(prev)
            ):
                count += 1
    return count


def _range_touches_heading(text: str, start: int, end: int) -> bool:
    """True when the char range ``[start, end)`` intersects any heading line.

    Uses the comprehensive heading-line set so an anchor spanning a setext or
    blockquoted heading (not just an ATX ``#`` line) is caught.
    """

    heading_lines = _heading_line_indices(text)
    if not heading_lines:
        return False
    start_line = text.count("\n", 0, start)
    last_char = end - 1 if end > start else start
    end_line = text.count("\n", 0, max(start, last_char))
    return any(li in heading_lines for li in range(start_line, end_line + 1))


def _locate_anchor(draft: str, anchor: str) -> Optional[tuple[int, int]]:
    """Locate ``anchor`` as a UNIQUE, EXACT verbatim substring of ``draft``.

    Returns ``(start, end)`` only when the anchor occurs exactly once, verbatim.
    Zero matches, multiple matches, or an anchor shorter than
    :data:`MIN_ANCHOR_CHARS` return ``None`` (recall-safe: the claim is
    preserved). The parser rejects outer JSON-field whitespace before this
    helper runs; no whitespace normalization or fuzzy fallback occurs. A
    non-verbatim anchor is skipped rather than applied to a different span.
    """

    if len(anchor) < MIN_ANCHOR_CHARS:
        return None
    first = draft.find(anchor)
    if first == -1:
        return None
    if draft.find(anchor, first + 1) != -1:
        # non-unique — ambiguous, skip
        return None
    return first, first + len(anchor)


def _locate_anchor_with_reason(
    draft: str,
    anchor: str,
) -> tuple[Optional[tuple[int, int]], str]:
    """Locate an anchor and return a machine-readable skip reason."""

    if len(anchor) < MIN_ANCHOR_CHARS:
        return None, "anchor_too_short"
    first = draft.find(anchor)
    if first == -1:
        return None, "anchor_not_found"
    if draft.find(anchor, first + 1) != -1:
        return None, "anchor_ambiguous"
    return (first, first + len(anchor)), "located"


def _splice(text: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping ``(start, end, replacement)`` edits to ``text``."""

    if not edits:
        return text
    ordered = sorted(edits, key=lambda e: e[0])
    out: list[str] = []
    cursor = 0
    for start, end, replacement in ordered:
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _dead_empty_body_citation_urls(
    citations: Sequence[CitationAnnotation] | None,
    sources_collected: Sequence[Mapping[str, Any]] | None,
) -> set[str]:
    """Return the RAW citation URLs that are DEAD **and** carry no grounding text.

    S88 lever 2 — the deterministic rule-7 backstop's exactly-safe class. A URL
    qualifies only when its citation classifies as ``dead`` (``live is False``
    AND the URL is in ``sources_collected``) *and* the collected source row has
    an EMPTY body and snippet — i.e. the critique was shown no content for it and
    therefore had no basis to judge a claim resting on it.

    A DEAD citation whose source body is PRESENT was in the critique's context
    untruncated (anti-gaming state B): its damage must stand, so it is NOT
    returned. HALLUCINATED (``live is False`` but not in sources), UNKNOWN
    (``live is None``), ORPHAN, and LIVE citations are never returned — dropping
    a gap note on them would either defeat rule 7's hallucinated-annotate rule or
    silently bless a claim the downstream References dead-flag never covers. The
    ``dead`` classification here is the SAME one the finalize dead-flag path uses
    (:func:`classify_citation` over :func:`build_source_url_set`), so every URL
    returned is independently flagged unverified downstream — a suppressed claim
    still ships flagged-unverified, never silently blessed.
    """

    if not citations or not sources_collected:
        return set()
    source_url_set = build_source_url_set(sources_collected)
    empty_content_urls = {
        _normalise_url_for_cache(str(src.get("url") or ""))
        for src in sources_collected
        if isinstance(src, Mapping)
        and str(src.get("url") or "").strip()
        and not (
            str(src.get("body") or "").strip()
            or str(src.get("snippet") or "").strip()
        )
    }
    if not empty_content_urls:
        return set()
    dead_empty: set[str] = set()
    for citation in citations:
        url = getattr(citation, "url", "") or ""
        if not url:
            continue
        if (
            classify_citation(
                live=getattr(citation, "live", None),
                url=url,
                source_url_set=source_url_set,
            )
            == CLASSIFICATION_DEAD
            and _normalise_url_for_cache(url) in empty_content_urls
        ):
            dead_empty.add(url)
    return dead_empty


def _apply_span_annotations(
    draft: str,
    annotations: Sequence[_SpanAnnotation],
    citations: Sequence[CitationAnnotation] | None = None,
    *,
    dead_empty_body_urls: set[str] | None = None,
    hallucinated_citation_urls: set[str] | None = None,
    grounding_visibility: _GroundingVisibility | None = None,
) -> tuple[str, list[tuple[int, int, str]], list[CritiqueAnnotation]]:
    """Render gap notes onto ``draft`` for each locatable annotation.

    Returns ``(candidate, edits, outcomes)``. Each applied outcome replaces its
    uniquely-located, non-heading anchor span with :data:`GAP_NOTE_DEFAULT`.
    Annotations whose anchor is missing, non-unique, heading-touching, or
    overlapping a prior edit are represented as skipped outcomes (recall-safe).
    With no applicable annotations, ``candidate`` is byte-for-byte identical
    to ``draft``.

    S88 lever 2: an otherwise-applicable annotation is PRESERVED (not damaged)
    when EVERY citation overlapping its claim span is in ``dead_empty_body_urls``
    (dead with no collected grounding content). The critique had no basis to
    judge such a claim and frozen rule 7 requires preservation; the dead citation
    is flagged unverified in the References downstream, so the claim still ships
    flagged-unverified. Empty/None disables the backstop (byte-identical to the
    pre-lever behaviour for direct callers).

    S92 m07: an ``unsupported`` annotation is also preserved unless the entire
    collected source corpus was fully visible to the critic. A bounded excerpt
    cannot prove absence anywhere in the corpus. Pre-classified ``hallucinated``
    annotations remain actionable because that classification is independent of
    body visibility.
    """

    edits: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    outcomes: list[CritiqueAnnotation] = []
    known_urls = [citation.url for citation in citations or []]
    known_url_set = set(known_urls)
    blocked_urls: set[str] = dead_empty_body_urls or set()
    hallucinated_urls: set[str] = hallucinated_citation_urls or set()

    for ann in annotations:
        loc, reason = _locate_anchor_with_reason(draft, ann.anchor)
        requested_urls = [url for url in ann.citation_urls if url in known_url_set]
        overlapping_urls: list[str] = []
        if loc is not None:
            start, end = loc
            for citation in citations or []:
                if citation.start_index < end and citation.end_index > start:
                    if citation.url not in overlapping_urls:
                        overlapping_urls.append(citation.url)
        citation_urls = tuple(
            [
                url
                for url in known_urls
                if url in requested_urls or url in overlapping_urls
            ][:CRITIQUE_ANNOTATION_URL_MAX_COUNT]
        )
        if ann.parse_reason:
            outcomes.append(
                CritiqueAnnotation(
                    anchor=ann.anchor,
                    verdict=ann.verdict,
                    note=ann.note,
                    citation_urls=citation_urls,
                    applied=False,
                    reason=ann.parse_reason,
                )
            )
            continue
        if loc is None:
            outcomes.append(
                CritiqueAnnotation(
                    anchor=ann.anchor,
                    verdict=ann.verdict,
                    note=ann.note,
                    citation_urls=citation_urls,
                    applied=False,
                    reason=reason,
                )
            )
            continue
        start, end = loc
        if (grounding_visibility is not None
            and grounding_visibility.draft_chars_visible is not None
            and end > grounding_visibility.draft_chars_visible):
            outcomes.append(CritiqueAnnotation(
                anchor=ann.anchor, verdict=ann.verdict, note=ann.note,
                citation_urls=citation_urls, reason="preserved_incomplete_draft",
            ))
            continue
        if _range_touches_heading(draft, start, end):
            outcomes.append(
                CritiqueAnnotation(
                    anchor=ann.anchor,
                    verdict=ann.verdict,
                    note=ann.note,
                    citation_urls=citation_urls,
                    applied=False,
                    reason="heading_overlap",
                )
            )
            continue
        if any(not (end <= s or start >= e) for s, e in occupied):
            outcomes.append(
                CritiqueAnnotation(
                    anchor=ann.anchor,
                    verdict=ann.verdict,
                    note=ann.note,
                    citation_urls=citation_urls,
                    applied=False,
                    reason="annotation_overlap",
                )
            )
            continue
        # S88 lever 2 — deterministic rule-7 backstop. If EVERY citation whose
        # span overlaps this claim is dead-with-no-collected-content, the
        # critique had no basis to judge it; preserve the claim (it ships
        # flagged-unverified via the downstream References dead-flag). A claim
        # with no overlapping citation, or any live/orphan/unknown/hallucinated
        # overlap, is NOT preserved here — genuine judgment (state B) and
        # hallucinated annotations stand.
        if overlapping_urls and all(url in blocked_urls for url in overlapping_urls):
            outcomes.append(
                CritiqueAnnotation(
                    anchor=ann.anchor,
                    verdict=ann.verdict,
                    note=ann.note,
                    citation_urls=citation_urls,
                    applied=False,
                    reason="preserved_dead_no_basis",
                )
            )
            continue
        preclassified_hallucination = bool(overlapping_urls) and all(
            url in hallucinated_urls for url in overlapping_urls
        )
        if grounding_visibility is not None and not preclassified_hallucination and (
            ann.anchor in grounding_visibility.declared_context_anchors
            or (len(ann.anchor) >= MIN_ANCHOR_CHARS
                and ann.anchor in grounding_visibility.author_context)
        ):
            outcomes.append(CritiqueAnnotation(
                anchor=ann.anchor, verdict=ann.verdict, note=ann.note,
                citation_urls=citation_urls, reason="preserved_declared_context",
            ))
            continue
        if (
            grounding_visibility is not None
            and not preclassified_hallucination
            and not grounding_visibility.corpus_complete
        ):
            outcomes.append(
                CritiqueAnnotation(
                    anchor=ann.anchor,
                    verdict=ann.verdict,
                    note=ann.note,
                    citation_urls=citation_urls,
                    applied=False,
                    reason="preserved_incomplete_grounding",
                )
            )
            continue
        occupied.append((start, end))
        edits.append((start, end, GAP_NOTE_DEFAULT))
        outcomes.append(
            CritiqueAnnotation(
                anchor=ann.anchor,
                verdict=ann.verdict,
                note=ann.note,
                citation_urls=citation_urls,
                applied=True,
                reason="applied_gap_note",
            )
        )

    candidate = _splice(draft, edits)
    return candidate, edits, outcomes


def _demote_applied_annotations(
    annotations: Sequence[CritiqueAnnotation],
    reason: str,
) -> list[CritiqueAnnotation]:
    """Mark tentative applications skipped when the whole transform fails."""

    return [
        replace(annotation, applied=False, reason=reason)
        if annotation.applied
        else annotation
        for annotation in annotations
    ]


def _assert_bounded_transform(
    draft: str,
    candidate: str,
    edits: list[tuple[int, int, str]],
) -> None:
    """Structural diff guard — raise :class:`CritiqueTransformError` on breach.

    Verifies that ``candidate`` is nothing more than ``draft`` with each applied
    annotation replaced by a gap note:

    1. Reconstruction — ``candidate`` equals ``draft`` spliced with exactly
       these edits (so every span OUTSIDE an edit is byte-identical to ``draft``).
    2. Every edit replaces with :data:`GAP_NOTE_DEFAULT` and touches no heading
       line (no novel content injected, no structure removed).
    3. Heading count (ATX + setext) is non-decreasing.
    4. Length delta is bounded by the edits — the candidate cannot be longer
       than the draft plus one gap-note per edit (each edit only removes an
       anchored claim and inserts the fixed gap note). There is NO lower ratio
       floor: a legitimately gap-heavy critique may shrink the body, and
       reconstruction already proves nothing was removed beyond the edits
       (adversarial finding: a 0.5 floor wrongly rejected a single dominant-claim
       gap note, degrading a good report to critique_skipped).
    """

    if _splice(draft, edits) != candidate:
        raise CritiqueTransformError("applier reconstruction mismatch")

    for start, end, replacement in edits:
        if replacement != GAP_NOTE_DEFAULT:
            raise CritiqueTransformError(f"non-gap-note replacement: {replacement!r}")
        if _range_touches_heading(draft, start, end):
            raise CritiqueTransformError("edit intersects a heading line")

    draft_headings = _count_headings(draft)
    candidate_headings = _count_headings(candidate)
    if candidate_headings < draft_headings:
        raise CritiqueTransformError(
            f"heading count dropped {draft_headings} -> {candidate_headings}"
        )

    max_len = len(draft) + len(edits) * len(GAP_NOTE_DEFAULT)
    if len(candidate) > max_len:
        raise CritiqueTransformError(
            f"length grew beyond gap-note bound: {len(candidate)} > {max_len}"
        )


# ---------------------------------------------------------------------------
# Public entry-point.
# ---------------------------------------------------------------------------


def run_critique(
    draft_markdown: str,
    sources_collected: Sequence[Mapping[str, Any]],
    mission_context: Mapping[str, Any],
    *,
    model: Any | None = None,
    citations: Sequence[CitationAnnotation] | None = None,
    scope_admission: ScopeAdmission | None = None,
) -> CritiqueResult:
    """Run a single span-annotation critique pass over ``draft_markdown``.

    Args:
        draft_markdown: The synthesized draft to validate (in production this
            is the cite-stripped body — the anchor basis the applier matches).
        sources_collected: The retrieval-payload accumulator from the runner.
        mission_context: ``{title, objective, ...}`` mapping, including bounded
            declared author background/context/constraints distinct from sources.
        model: Optional pre-built smolagents ``OpenAIServerModel``. When
            ``None``, ``build_default_model()`` is invoked.
        citations: S68 — optional list of liveness-checked
            :class:`CitationAnnotation` records. When provided and non-empty,
            the critique user prompt gains a "# Citation Liveness Context"
            section that flags hallucinated-backed claims for annotation.

    Returns:
        :class:`CritiqueResult`. ``output_markdown`` is the draft with gap notes
        deterministically applied for each locatable unsupported-claim
        annotation. On ANY failure (model error, unparseable output, structural
        guard breach) ``output_markdown == draft_markdown`` unchanged, ``error``
        is populated, and ``gap_notes_inserted == 0``. **This function does not
        raise.**
    """

    t0 = time.monotonic()
    provider_attempts: list[dict[str, Any]] = []
    observation_start = 0
    grounding_telemetry: dict[str, int] = {}
    grounding_visibility = _GroundingVisibility()
    assessment = _assessment_record("", draft_markdown, grounding_visibility, 0)

    if not isinstance(draft_markdown, str) or not draft_markdown.strip():
        # Echo the input (I1 no-op byte identity): a whitespace-only draft must
        # come back unchanged, not blanked. Non-str drafts fall back to "".
        return CritiqueResult(
            output_markdown=draft_markdown if isinstance(draft_markdown, str) else "",
            claim_count=0,
            gap_notes_inserted=0,
            runtime_seconds=round(time.monotonic() - t0, 3),
            token_usage={"input": 0, "output": 0, "total": 0},
            token_usage_complete=True,
            error=None,
            grounding_telemetry=grounding_telemetry,
            provider_attempts=list(provider_attempts[observation_start:]),
            assessment=assessment,
            evidence_manifest=list(grounding_visibility.passages),
        )

    # Model construction + call + response extraction all inside the try — a
    # pathological model return (e.g. a .content property that raises, or a
    # non-numeric token_usage) must never propagate out (MUST-NEVER-RAISE).
    token_usage = {"input": 0, "output": 0, "total": 0}
    token_usage_complete = True
    provider_call_started = False
    provider_response_received = False
    try:
        if model is None:
            model = build_default_model()
        if hasattr(model, "provider_attempts"):
            model.usage_stage = "critique"
        observed = getattr(model, "provider_attempts", [])
        if isinstance(observed, list):
            provider_attempts = observed
            observation_start = len(observed)

        messages = _build_critique_messages(
            draft_markdown,
            sources_collected or [],
            mission_context or {},
            citations=citations,
            grounding_telemetry=grounding_telemetry,
            grounding_visibility=grounding_visibility,
            scope_admission=scope_admission,
        )

        assessment = _assessment_record("", draft_markdown, grounding_visibility, 0)
        provider_call_started = True
        response = model(
            messages=messages,
        )
        provider_response_received = True
        # A response proves dispatch, not that its usage envelope is readable.
        # Stay incomplete if accessing ``response.token_usage`` itself raises.
        token_usage_complete = False
        token_usage, token_usage_complete = _extract_token_usage(response)
        raw = _extract_response_content(response)
    except Exception as exc:  # noqa: BLE001 — fallthrough is the contract
        if provider_call_started and not provider_response_received:
            token_usage_complete = False
        return CritiqueResult(
            output_markdown=draft_markdown,
            claim_count=0,
            gap_notes_inserted=0,
            runtime_seconds=round(time.monotonic() - t0, 3),
            token_usage=token_usage,
            token_usage_complete=token_usage_complete,
            error=safe_exception_text(exc, category="execution"),
            grounding_telemetry=grounding_telemetry,
            provider_attempts=list(provider_attempts[observation_start:]),
            assessment=assessment,
            evidence_manifest=list(grounding_visibility.passages),
        )

    parsed = _parse_annotations(raw)
    if parsed is None:
        # HARD FAIL — no delimiter or unparseable annotations block. Leave the
        # report untouched (S85 m01: no silent clipped-echo ship).
        return CritiqueResult(
            output_markdown=draft_markdown,
            claim_count=0,
            gap_notes_inserted=0,
            runtime_seconds=round(time.monotonic() - t0, 3),
            token_usage=token_usage,
            token_usage_complete=token_usage_complete,
            error=safe_error_from_parts(
                "validation",
                "CritiqueOutputError",
            ).to_text(),
            grounding_telemetry=grounding_telemetry,
            provider_attempts=list(provider_attempts[observation_start:]),
            assessment=assessment,
            evidence_manifest=list(grounding_visibility.passages),
        )

    annotations, claim_count, annotations_dropped = parsed

    # Applier + structural guard inside a try — the guard raises
    # CritiqueTransformError on breach (degrade, body untouched); any other
    # unexpected applier fault is also caught so run_critique never raises.
    annotation_outcomes: list[CritiqueAnnotation] = []
    try:
        assessment = _assessment_record(raw, draft_markdown, grounding_visibility, claim_count)
        dead_empty_body_urls = _dead_empty_body_citation_urls(
            citations, sources_collected
        )
        source_url_set = build_source_url_set(sources_collected)
        hallucinated_citation_urls = {
            citation.url
            for citation in citations or []
            if classify_citation(
                live=getattr(citation, "live", None),
                url=getattr(citation, "url", "") or "",
                source_url_set=source_url_set,
            )
            == CLASSIFICATION_HALLUCINATED
        }
        candidate, edits, annotation_outcomes = _apply_span_annotations(
            draft_markdown,
            annotations,
            citations,
            dead_empty_body_urls=dead_empty_body_urls,
            hallucinated_citation_urls=hallucinated_citation_urls,
            grounding_visibility=grounding_visibility,
        )
        _assert_bounded_transform(draft_markdown, candidate, edits)
    except CritiqueTransformError as exc:
        annotation_outcomes = _demote_applied_annotations(
            annotation_outcomes,
            "transform_guard_failed",
        )
        return CritiqueResult(
            output_markdown=draft_markdown,
            claim_count=claim_count,
            gap_notes_inserted=0,
            runtime_seconds=round(time.monotonic() - t0, 3),
            token_usage=token_usage,
            token_usage_complete=token_usage_complete,
            error=safe_exception_text(exc, category="validation"),
            grounding_telemetry=grounding_telemetry,
            provider_attempts=list(provider_attempts[observation_start:]),
            assessment=assessment,
            evidence_manifest=list(grounding_visibility.passages),
            annotations=annotation_outcomes,
            annotations_dropped=annotations_dropped,
        )
    except Exception as exc:  # noqa: BLE001 — applier bug must never raise out
        annotation_outcomes = _demote_applied_annotations(
            annotation_outcomes,
            "transform_apply_failed",
        )
        return CritiqueResult(
            output_markdown=draft_markdown,
            claim_count=claim_count,
            gap_notes_inserted=0,
            runtime_seconds=round(time.monotonic() - t0, 3),
            token_usage=token_usage,
            token_usage_complete=token_usage_complete,
            error=safe_exception_text(exc, category="internal"),
            grounding_telemetry=grounding_telemetry,
            provider_attempts=list(provider_attempts[observation_start:]),
            assessment=assessment,
            evidence_manifest=list(grounding_visibility.passages),
            annotations=annotation_outcomes,
            annotations_dropped=annotations_dropped,
        )

    applied_count = sum(
        1 for annotation in annotation_outcomes if annotation.applied
    )
    return CritiqueResult(
        output_markdown=candidate,
        claim_count=claim_count,
        gap_notes_inserted=applied_count,
        runtime_seconds=round(time.monotonic() - t0, 3),
        token_usage=token_usage,
        token_usage_complete=token_usage_complete,
        error=None,
        grounding_telemetry=grounding_telemetry,
        provider_attempts=list(provider_attempts[observation_start:]),
        assessment=assessment,
        evidence_manifest=list(grounding_visibility.passages),
        annotations=annotation_outcomes,
        annotations_dropped=annotations_dropped,
    )


__all__ = [
    "CRITIQUE_CONTEXT_MAX_CHARS",
    "CRITIQUE_DRAFT_MAX_CHARS",
    "CRITIQUE_LIVENESS_MAX_CHARS",
    "CRITIQUE_ANNOTATION_MAX_COUNT",
    "CRITIQUE_OUTPUT_DELIMITER",
    "CRITIQUE_SYSTEM_PROMPT",
    "CritiqueAnnotation",
    "CritiqueResult",
    "CritiqueTransformError",
    "GAP_NOTE_DEFAULT",
    "GROUNDING_TOTAL_BUDGET_CHARS",
    "SOURCE_GROUNDING_MAX_CHARS",
    "SOURCE_SNIPPET_MAX_CHARS",
    "run_critique",
]
