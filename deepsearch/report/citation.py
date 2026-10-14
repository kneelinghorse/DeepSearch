"""Citation schema and persisted-coordinate rebasing helpers.

``CitationAnnotation`` mirrors OpenAI's ``url_citation`` annotation shape.
Before persistence, its codepoint offsets are rebased from the citation
extractor's marker-stripped reference text into the exact markdown artifact
shipped to the user. S68 (PD5/PD9) adds the additive ``live`` field for the URL
liveness gate.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, NamedTuple, Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator

CITATION_CLAIM_MAX_CHARS = 20_000


class CitationSpanRebaseError(ValueError):
    """The persisted artifact cannot anchor the segment its spans came from.

    Raised only for whole-segment structural breaks. Individual unmappable
    citations are reported through :class:`RebasedCitations.dropped` and never
    raise, because one citation must not be able to discard a synthesized
    report (S92 m06).

    The exception deliberately carries structural diagnostics only -- these
    fields are safe to log. Cited report text can contain sensitive mission
    content and must not be copied into logs merely because rebasing failed.
    """

    def __init__(
        self,
        *,
        annotation_index: int,
        reason: str,
        match_count: int | None = None,
    ) -> None:
        self.annotation_index = annotation_index
        self.reason = reason
        self.match_count = match_count
        detail = f"citation {annotation_index}: {reason}"
        if match_count is not None:
            detail += f" (matches={match_count})"
        super().__init__(detail)


class CitationAnnotation(BaseModel):
    """Per-claim citation annotation.

    Six fields. The first five match OpenAI's ``url_citation`` shape verbatim;
    ``live`` is the S68 additive field for URL-liveness gating.

    * ``type`` — discriminator, always ``"url_citation"``.
    * ``start_index`` — codepoint offset (≥0) into persisted ``result_markdown``
      where the cited span begins.
    * ``end_index`` — codepoint offset (≥ ``start_index``) into persisted
      ``result_markdown`` where the cited span ends, end-exclusive. In a
      persisted protocol, ``result_markdown[start_index:end_index]`` returns
      the cited span exactly.
    * ``url`` — the literal URL the agent retrieved (no normalisation in S67).
    * ``title`` — best-effort title from ``sources_collected``; ``None`` if not
      resolvable. Excluded from ``to_dict()`` output when ``None``.
    * ``live`` — S68 URL liveness probe outcome. ``None`` = unchecked
      (default; backward-compatible with pre-S68 protocols). ``True`` =
      checked + live. ``False`` = checked + dead-or-orphan. The persisted
      schema carries only the boolean; the internal classification
      (``orphan`` / ``dead`` / ``unknown``) is telemetry-only. Excluded from
      ``to_dict()`` output when ``None``.
    """

    # Internal ownership survives model_copy/rebase without extending protocol JSON.
    _extraction_mode: Literal["primary", "numbered"] | None = PrivateAttr(default=None)

    type: Literal["url_citation"] = Field(
        default="url_citation",
        description="Annotation discriminator (always 'url_citation').",
    )
    start_index: int = Field(
        ge=0,
        description=(
            "Codepoint offset into persisted result_markdown where the cited "
            "span begins."
        ),
    )
    end_index: int = Field(
        ge=0,
        description=(
            "Codepoint offset (end-exclusive) into persisted result_markdown "
            "where the cited span ends."
        ),
    )
    url: str = Field(
        min_length=1,
        description="URL the cited claim is backed by (literal value from retrieval payload).",
    )
    title: Optional[str] = Field(
        default=None,
        description="Best-effort source title from sources_collected; nullable.",
    )
    live: Optional[bool] = Field(
        default=None,
        description=(
            "URL liveness probe outcome (S68). None=unchecked (default; "
            "pre-S68 protocols), True=live, False=dead-or-orphan. Excluded "
            "from to_dict() when None."
        ),
    )

    @model_validator(mode="after")
    def _end_after_start(self) -> "CitationAnnotation":
        if self.end_index < self.start_index:
            raise ValueError(
                f"end_index ({self.end_index}) must be >= start_index ({self.start_index})"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict, excluding ``title`` when ``None``."""
        return self.model_dump(exclude_none=True)


class RebasedCitations(NamedTuple):
    """Outcome of rebasing one report's citation annotations.

    ``annotations`` are the mappable citations in their original order.
    ``dropped`` records the structural reason each unmappable citation was
    skipped, so telemetry can report the loss without carrying report text.
    """

    annotations: list["CitationAnnotation"]
    dropped: list[Dict[str, Any]]


def rebase_citation_annotations(
    *,
    reference_markdown: str,
    persisted_segment_markdown: str,
    persisted_markdown: str,
    annotations: list[CitationAnnotation],
) -> RebasedCitations:
    """Rebase citation slices into the exact persisted markdown coordinates.

    ``annotations`` initially index ``reference_markdown``, the marker-stripped
    text produced by citation extraction. ``persisted_segment_markdown`` is the
    exact cleaned synthesized segment inserted by the report generator after
    leading-H1 stripping, run-integrity preamble insertion, and synthesis
    cleaning. The full segment is anchored exactly once in ``persisted_markdown``;
    each cited slice is then located inside that segment. This segment-aware
    mapping prevents text removed by a transform from being falsely rebound to
    an identical string in the wrapper or mission context, without assuming any
    constant wrapper prefix.

    Two rules keep a normal report from being destroyed by one citation
    (S92 m06, production forensics on CAT-R006/R008/R009-S89):

    * **Document-order resolution.** Repeated claim text is not ambiguous, it is
      ordinal. Unique source spans are resolved in ascending source order
      against a monotonic cursor, so the Nth occurrence binds to the Nth
      citation. A report whose synthesis emits an inline References section --
      the exact shape that blocked three production missions -- routinely
      repeats claim text inside the segment and previously failed closed.
    * **Per-citation degradation.** A slice that still cannot be located is
      dropped and reported in ``dropped``; it never discards the surrounding
      report. Only a broken *segment* anchor -- persisted markdown that does not
      contain the segment it was built from -- still raises, because in that
      case no coordinate in the artifact can be trusted.

    Leading/trailing whitespace on a source slice is an extractor boundary
    artifact, not corruption: the span is trimmed inward to the cited text
    rather than rejected, which yields a *more* exact persisted coordinate.

    Multiple citations attached to the same original span are legal: they are
    mapped to the same persisted occurrence while retaining their original order
    and all non-coordinate fields.
    """

    rebased: list[CitationAnnotation] = []
    dropped: list[Dict[str, Any]] = []

    if not annotations:
        return RebasedCitations(annotations=rebased, dropped=dropped)
    if not persisted_segment_markdown:
        raise CitationSpanRebaseError(
            annotation_index=0,
            reason="persisted_segment_empty",
        )

    segment_match_count = _count_occurrences(
        persisted_markdown,
        persisted_segment_markdown,
    )
    if segment_match_count == 0:
        raise CitationSpanRebaseError(
            annotation_index=0,
            reason="persisted_segment_missing_from_persisted_markdown",
            match_count=0,
        )
    if segment_match_count > 1:
        raise CitationSpanRebaseError(
            annotation_index=0,
            reason="persisted_segment_ambiguous_in_persisted_markdown",
            match_count=segment_match_count,
        )
    segment_start = persisted_markdown.find(persisted_segment_markdown)

    # Resolve each distinct source span once, in ascending source order, so the
    # monotonic cursor below follows the reference document rather than the
    # order the extractor happened to emit annotations in.
    source_keys: list[tuple[int, int]] = []
    seen_keys: set[tuple[int, int]] = set()
    for annotation in annotations:
        key = (annotation.start_index, annotation.end_index)
        if key not in seen_keys:
            seen_keys.add(key)
            source_keys.append(key)

    resolved_spans: dict[tuple[int, int], tuple[int, int]] = {}
    rejected_keys: dict[tuple[int, int], tuple[str, int | None]] = {}
    cursor = 0

    # Resolve the longest same-start slice first. The extractor can attach a
    # citation to an early clause and another to the enclosing sentence, which
    # produces distinct but overlapping spans with one source start. Those
    # spans belong to the same document occurrence and must share its target
    # start; only a genuinely later source start consumes the next occurrence.
    active_source_start: int | None = None
    active_target_start: int | None = None
    for start_index, end_index in sorted(
        source_keys,
        key=lambda item: (item[0], -item[1]),
    ):
        key = (start_index, end_index)
        if end_index > len(reference_markdown):
            rejected_keys[key] = ("source_span_out_of_bounds", None)
            continue

        raw_text = reference_markdown[start_index:end_index]
        # Trim inward: a boundary space belongs to the document, not the claim.
        cited_text = raw_text.strip()
        if not cited_text:
            rejected_keys[key] = ("empty_source_span", None)
            continue
        if len(cited_text) > CITATION_CLAIM_MAX_CHARS:
            rejected_keys[key] = ("source_span_too_long", None)
            continue

        if start_index == active_source_start and active_target_start is not None:
            claim_start = active_target_start
            if not persisted_segment_markdown.startswith(cited_text, claim_start):
                rejected_keys[key] = (
                    "source_span_missing_from_persisted_segment",
                    0,
                )
                continue
        else:
            claim_start = persisted_segment_markdown.find(cited_text, cursor)
            if claim_start < 0:
                rejected_keys[key] = (
                    "source_span_missing_from_persisted_segment",
                    0,
                )
                continue
            active_source_start = start_index
            active_target_start = claim_start
            cursor = claim_start + 1

        target_start = segment_start + claim_start
        target_end = target_start + len(cited_text)
        if persisted_markdown[target_start:target_end] != cited_text:
            rejected_keys[key] = ("persisted_slice_mismatch", None)
            continue
        resolved_spans[key] = (target_start, target_end)

    for index, annotation in enumerate(annotations):
        key = (annotation.start_index, annotation.end_index)
        target_span: tuple[int, int] | None = resolved_spans.get(key)
        if target_span is None:
            reason, match_count = rejected_keys.get(key, ("unresolved_source_span", None))
            record: Dict[str, Any] = {
                "annotation_index": index,
                "reason": reason,
                "url": annotation.url,
            }
            if match_count is not None:
                record["match_count"] = match_count
            dropped.append(record)
            continue

        rebased.append(annotation.model_copy(update={
            "start_index": target_span[0], "end_index": target_span[1],
        }))

    return RebasedCitations(annotations=rebased, dropped=dropped)


def _count_occurrences(text: str, needle: str) -> int:
    """Count overlapping exact occurrences of a non-empty string."""

    count = 0
    search_from = 0
    while True:
        match = text.find(needle, search_from)
        if match < 0:
            return count
        count += 1
        search_from = match + 1


__all__ = [
    "CITATION_CLAIM_MAX_CHARS",
    "CitationAnnotation",
    "CitationSpanRebaseError",
    "RebasedCitations",
    "rebase_citation_annotations",
]
