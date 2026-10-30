"""Bounded passage selection for critique; selected text is never a full-source proof."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Sequence

_WORD = re.compile(r"[a-zA-Z][a-zA-Z_-]{3,}")
_STOP = frozenset("this that with from have will would should their which about into through"
                  " source sources report claim claims evidence using used also does only".split())
_CAVEAT_GROUPS = (
    re.compile(r"incomplete|truncat", re.I),
    re.compile(r"mistakes|semantic errors|incorrect values", re.I),
    re.compile(r"limitation|caveat|exception|not guarantee|cannot guarantee", re.I),
)
# A chunk starts every 600 characters of a paragraph and is scored on 700; a
# chosen chunk is shown cut to 500.
_STRIDE = 600


def _chunks(body: str) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    for paragraph in re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*", body):
        start, end = paragraph.span()
        for offset in range(start, end, _STRIDE):
            chunks.append((offset, min(end, offset + 700)))
    return chunks


def _rank(body: str, claim_text: str, chunks: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Chunk indices in rank order, and the at most four the selector chooses."""
    terms = set(_WORD.findall(claim_text.lower())) - _STOP
    term_sets = [set(_WORD.findall(body[a:b].lower())) - _STOP for a, b in chunks]
    frequency = Counter(term for terms_in_chunk in term_sets for term in terms_in_chunk)
    code_ranges: list[tuple[int, int]] = []
    fence_start: int | None = None
    for fence in re.finditer(r"(?m)^\s{0,3}```.*$", body):
        if fence_start is None:
            fence_start = fence.start()
        else:
            code_ranges.append((fence_start, fence.end()))
            fence_start = None
    if fence_start is not None:
        code_ranges.append((fence_start, len(body)))
    in_code = [any(a < end and b > start for start, end in code_ranges) for a, b in chunks]

    def score(index: int) -> float:
        a, b = chunks[index]
        overlap = terms & term_sets[index]
        lexical = sum(math.log1p(len(chunks) / frequency[word]) for word in overlap)
        # Normalize verbosity and prefer explanatory prose over repeated code.
        return lexical / math.sqrt(max(1, len(term_sets[index]))) - (30 if in_code[index] else 0)

    ranked = sorted(range(len(chunks)), key=lambda i: (-score(i), chunks[i][0]))
    chosen: list[int] = []
    # Diverse qualifications prevent one repeated topic from crowding out
    # truncation, semantic error, and documented capability limits.
    for caveat in _CAVEAT_GROUPS:
        match = next((i for i in ranked if not in_code[i] and terms & term_sets[i]
                      and caveat.search(body[chunks[i][0]:chunks[i][1]])), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    for i in ranked:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= 4:
            break
    return ranked, chosen


def select_critique_passages(
    body: str, claim_text: str, *, budget: int, source_index: int, url: str,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Select exact ranges within a rendered budget, including relevant caveats.

    Claim terms rank bounded paragraph windows; caveat terms prioritize contrary
    qualifications within a cited source. This is lexical retrieval, not an
    entailment judgment. Offsets refer to the original stored Unicode body.
    """
    source_id = hashlib.sha256(url.encode()).hexdigest()
    digest = hashlib.sha256(body.encode()).hexdigest()
    chunks = _chunks(body)
    if not chunks:
        return "", [], not body

    whole_header = f"[E{source_index}.1 chars=0:{len(body)} sha256={digest}]\n"
    if len(whole_header) + len(body) <= budget:
        ranges = [(0, len(body))]
    else:
        _, chosen = _rank(body, claim_text, chunks)
        ranges = []
        for i in chosen:
            a, b = chunks[i]
            b = min(b, a + 500)
            if not any(a < end and b > start for start, end in ranges):
                ranges.append((a, b))
        ranges.sort()

    rendered: list[str] = []
    manifest: list[dict[str, Any]] = []
    used = 0
    for start, end in ranges:
        evidence_id = f"E{source_index}.{len(manifest) + 1}"
        header = f"[{evidence_id} chars={start}:{end} sha256={digest}]\n"
        separator = 2 if rendered else 0
        available = budget - used - separator - len(header)
        if available <= 0:
            break
        end = min(end, start + available)
        header = f"[{evidence_id} chars={start}:{end} sha256={digest}]\n"
        text = body[start:end]
        rendered.append(header + text)
        used += separator + len(header) + len(text)
        manifest.append({
            "evidence_id": evidence_id, "source_id": source_id, "url": url,
            "body_sha256": digest, "start": start, "end": end,
            "body_chars": len(body), "kind": "source_passage",
        })
    complete = bool(manifest) and len(manifest) == 1 and (
        manifest[0]["start"] == 0 and manifest[0]["end"] == len(body)
    )
    return "\n\n".join(rendered), manifest, complete


def cited_passage_ranges(
    body: str, claim_text: str, shown: Sequence[tuple[int, int]], *, limit: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """S103 m03: a cut source's shown chunks in whole strides, then more of its chunks.

    ``shown`` are the source's passages as the selector showed them, each starting
    at its chunk. Each becomes its whole 600-character stride, clipped to its
    chunk's end. The selector's other chunks follow in its rank order as strides,
    skipping any that overlaps a range already taken, until ``limit`` chunks.
    Returns the widened shown ranges and the extra ones, in that order.
    """
    chunks = _chunks(body)
    chunk_end = dict(chunks)
    widened = [(start, max(end, min(start + _STRIDE, chunk_end[start]))) for start, end in shown]
    ranked, _ = _rank(body, claim_text, chunks)
    taken = list(widened)
    extra: list[tuple[int, int]] = []
    for index in ranked:
        if len(taken) >= limit:
            break
        start, end = chunks[index]
        stride = (start, min(start + _STRIDE, end))
        if not any(stride[0] < b and stride[1] > a for a, b in taken):
            taken.append(stride)
            extra.append(stride)
    return widened, extra


def render_passages(
    body: str, ranges: Sequence[tuple[int, int]], baseline: Sequence[tuple[int, int]], *,
    source_index: int, url: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Render ranges as the selector renders passages, touching ranges as one.

    Each manifest entry also lists the ``baseline`` ranges it contains, so the
    passages that were shown before the extension stay identifiable.
    """
    source_id = hashlib.sha256(url.encode()).hexdigest()
    digest = hashlib.sha256(body.encode()).hexdigest()
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    rendered: list[str] = []
    manifest: list[dict[str, Any]] = []
    for start, end in merged:
        evidence_id = f"E{source_index}.{len(manifest) + 1}"
        rendered.append(f"[{evidence_id} chars={start}:{end} sha256={digest}]\n{body[start:end]}")
        manifest.append({
            "evidence_id": evidence_id, "source_id": source_id, "url": url,
            "body_sha256": digest, "start": start, "end": end,
            "body_chars": len(body), "kind": "source_passage",
            "baseline_ranges": [[a, b] for a, b in sorted(baseline) if start <= a and b <= end],
        })
    return "\n\n".join(rendered), manifest
