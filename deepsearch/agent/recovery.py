"""Deterministic one-pass recovery policy and bounded recovery prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .critique import CritiqueAnnotation, CritiqueResult
from .critique_policy import CRITIQUE_RATIO_MIN_APPLIED

RECOVERY_APPLIED_COUNT_THRESHOLD = 10
RECOVERY_APPLIED_RATIO_THRESHOLD = 0.20
RECOVERY_MAX_STEPS = 10
RECOVERY_DRAFT_MAX_CHARS = 180_000
RECOVERY_CLAIMS_MAX_COUNT = 30
RECOVERY_CLAIM_MAX_CHARS = 1_000
RECOVERY_MISSING_SLOTS_MAX_COUNT = 20
RECOVERY_MISSING_SLOT_MAX_CHARS = 500
RECOVERY_SOURCE_CONTEXT_MAX_CHARS = 80_000
RECOVERY_SOURCE_CONTENT_MAX_CHARS = 2_000
RECOVERY_PROMPT_MAX_CHARS = 400_000


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Pure recovery-trigger result with auditable inputs and reasons."""

    triggered: bool
    reasons: tuple[str, ...]
    applied_count: int
    claim_count: int
    applied_ratio: float
    coverage_shortfall: float
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable telemetry representation."""

        return {
            "triggered": self.triggered,
            "reasons": list(self.reasons),
            "applied_count": self.applied_count,
            "claim_count": self.claim_count,
            "applied_ratio": self.applied_ratio,
            "coverage_shortfall": self.coverage_shortfall,
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True, slots=True)
class RecoveryPrompt:
    """Bounded recovery directive plus exact inclusion telemetry."""

    directive: str
    telemetry: dict[str, int]


COVERAGE_REGRESSION_EPSILON = 1e-9


def _coverage_shortfall_value(coverage: Mapping[str, Any] | None) -> float:
    """Return the clamped completeness shortfall (0.0 best .. 1.0 worst)."""

    raw_shortfall = (coverage or {}).get("shortfall_score", 0.0)
    try:
        return max(0.0, min(1.0, float(raw_shortfall or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def decide_recovery(
    critique_result: CritiqueResult,
    coverage: Mapping[str, Any] | None,
) -> RecoveryDecision:
    """Trigger once for material applied damage or any final coverage shortfall.

    Args:
        critique_result: Structured first-pass critique result.
        coverage: First-pass ``completeness_check`` output.

    Returns:
        A deterministic decision. Critique failures are ineligible because
        they provide no trustworthy damaged-claim ledger.
    """

    summary = critique_result.annotation_summary()
    applied_count = int(summary["applied"])
    claim_count = max(0, int(critique_result.claim_count))
    applied_ratio = round(applied_count / max(1, claim_count), 6)
    coverage_shortfall = _coverage_shortfall_value(coverage)

    if critique_result.error:
        return RecoveryDecision(
            triggered=False,
            reasons=(),
            applied_count=applied_count,
            claim_count=claim_count,
            applied_ratio=applied_ratio,
            coverage_shortfall=coverage_shortfall,
            blocked_reason="critique_error",
        )

    reasons: list[str] = []
    if applied_count >= RECOVERY_APPLIED_COUNT_THRESHOLD:
        reasons.append("applied_count_threshold")
    if (
        applied_count >= CRITIQUE_RATIO_MIN_APPLIED
        and applied_ratio >= RECOVERY_APPLIED_RATIO_THRESHOLD
    ):
        reasons.append("applied_ratio_threshold")
    if coverage_shortfall > 0.0:
        reasons.append("coverage_shortfall")

    return RecoveryDecision(
        triggered=bool(reasons),
        reasons=tuple(reasons),
        applied_count=applied_count,
        claim_count=claim_count,
        applied_ratio=applied_ratio,
        coverage_shortfall=coverage_shortfall,
    )


@dataclass(frozen=True, slots=True)
class RecoveryRegressionGuard:
    """Non-regression verdict comparing a recovered pass to the first pass.

    The bounded recovery pass runs a second full un-diffed whole-document
    synthesis (Decision #823). Selecting its output on the sole condition that
    its critique did not hard-error relocated that rewrite single-point-of-
    failure to the recovery stage: a recovered draft that drops supported
    coverage or live citations could still be selected, persisted, and graded
    COMPLETE/degraded. This guard keeps the reviewable first pass unless the
    recovered pass is non-regressive on BOTH dimensions.
    """

    non_regressive: bool
    first_shortfall: float
    recovered_shortfall: float
    first_live_citations: int
    recovered_live_citations: int
    first_applied: int
    recovered_applied: int
    regressed_dimensions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable telemetry representation."""

        return {
            "non_regressive": self.non_regressive,
            "first_shortfall": self.first_shortfall,
            "recovered_shortfall": self.recovered_shortfall,
            "first_live_citations": self.first_live_citations,
            "recovered_live_citations": self.recovered_live_citations,
            "first_applied": self.first_applied,
            "recovered_applied": self.recovered_applied,
            "regressed_dimensions": list(self.regressed_dimensions),
        }


def evaluate_recovery_regression(
    *,
    first_coverage: Mapping[str, Any] | None,
    recovered_coverage: Mapping[str, Any] | None,
    first_live_citations: int,
    recovered_live_citations: int,
    first_applied: int,
    recovered_applied: int,
) -> RecoveryRegressionGuard:
    """Reject a recovered pass that regresses coverage, live citations, or
    structured critique damage (``applied``).

    Non-regressive means the recovered pass does not INCREASE the completeness
    shortfall, does not DROP live citations, and does not ADD applied critique
    damage relative to the first pass. A recovery that improves or holds all
    dimensions is selected; anything that would gut the report or re-worsen the
    critique damage keeps the reviewable first pass. This is do-no-harm, not
    disable-recovery — a genuine repair (which lifts coverage, preserves
    grounding, and does not add damage) always passes.

    S88 lever 4: the ``applied`` term is additive and evaluated ONLY when
    coverage did not already regress. A coverage regression already rejects the
    pass, so the guard verdict is unchanged either way; scoping the term this
    way keeps ``regressed_dimensions`` clean and preserves the invariant that
    ``non_regressive = not (coverage OR live_citations OR applied)`` is an
    AND-of-negations (monotonic — the guard can only ever reject MORE). This
    closes the hole where the un-diffed recovery rewrite pushed applied 10->13
    (T2 is graded on the SELECTED pass), which the coverage/citation terms
    alone could not catch.
    """

    first_shortfall = _coverage_shortfall_value(first_coverage)
    recovered_shortfall = _coverage_shortfall_value(recovered_coverage)
    first_live = max(0, int(first_live_citations))
    recovered_live = max(0, int(recovered_live_citations))
    first_applied_count = max(0, int(first_applied))
    recovered_applied_count = max(0, int(recovered_applied))

    regressed: list[str] = []
    coverage_regressed = (
        recovered_shortfall > first_shortfall + COVERAGE_REGRESSION_EPSILON
    )
    if coverage_regressed:
        regressed.append("coverage")
    if recovered_live < first_live:
        regressed.append("live_citations")
    if not coverage_regressed and recovered_applied_count > first_applied_count:
        regressed.append("applied")

    return RecoveryRegressionGuard(
        non_regressive=not regressed,
        first_shortfall=first_shortfall,
        recovered_shortfall=recovered_shortfall,
        first_live_citations=first_live,
        recovered_live_citations=recovered_live,
        first_applied=first_applied_count,
        recovered_applied=recovered_applied_count,
        regressed_dimensions=tuple(regressed),
    )


def remaining_recovery_steps(*, total_steps: int, used_steps: int) -> int:
    """Return the one-pass step allowance under the original total ceiling."""

    remaining = max(0, int(total_steps) - max(0, int(used_steps)))
    return min(RECOVERY_MAX_STEPS, remaining)


def _bounded_live_source_block(
    sources: Sequence[Mapping[str, Any]],
    live_urls: set[str],
) -> tuple[str, dict[str, int]]:
    rows: list[str] = []
    candidates = [
        source
        for source in sources
        if str(source.get("url") or "").strip() in live_urls
    ]
    available = len(candidates)
    truncated = 0
    used_chars = 0
    for source in candidates:
        url = str(source.get("url") or "").strip()
        title = str(source.get("title") or "")[:500]
        content_raw = source.get("body") or source.get("snippet") or ""
        content = str(content_raw).strip()
        bounded_content = content[:RECOVERY_SOURCE_CONTENT_MAX_CHARS]
        truncated += int(len(content) > len(bounded_content))
        row = (
            f"- TITLE: {title}\n"
            f"  URL: {url[:2_048]}\n"
            f"  CONTENT: {bounded_content}"
        )
        separator = 2 if rows else 0
        if used_chars + separator + len(row) > RECOVERY_SOURCE_CONTEXT_MAX_CHARS:
            break
        rows.append(row)
        used_chars += separator + len(row)
    return "\n\n".join(rows) or "(no live source bodies available)", {
        "live_sources_available": available,
        "live_sources_included": len(rows),
        "live_sources_dropped": max(0, available - len(rows)),
        "live_sources_truncated": truncated,
        "live_source_context_chars": used_chars,
    }


def build_recovery_prompt(
    *,
    original_markdown: str,
    annotations: Sequence[CritiqueAnnotation],
    coverage: Mapping[str, Any] | None,
    sources: Sequence[Mapping[str, Any]],
    live_citation_urls: Sequence[str],
    remaining_steps: int,
) -> RecoveryPrompt:
    """Build the complete-report recovery directive under explicit budgets.

    Args:
        original_markdown: First synthesis draft, including citation markers.
        annotations: First-pass structured critique annotations.
        coverage: First-pass deliverable coverage output.
        sources: Existing run-local grounding sources.
        live_citation_urls: URLs whose first-pass citation liveness is ``True``.
        remaining_steps: Maximum steps the recovery CodeAgent may consume.

    Returns:
        Bounded prompt and exact inclusion/drop telemetry.
    """

    applied = [annotation for annotation in annotations if annotation.applied]
    bounded_annotations = applied[:RECOVERY_CLAIMS_MAX_COUNT]
    claim_rows = [
        json.dumps(
            {
                "anchor": annotation.anchor[:RECOVERY_CLAIM_MAX_CHARS],
                "verdict": annotation.verdict,
                "note": annotation.note[:300],
                "citation_urls": [
                    url[:500] for url in annotation.citation_urls[:3]
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for annotation in bounded_annotations
    ]

    missing_raw = (coverage or {}).get("missing", [])
    missing = (
        [str(item) for item in missing_raw]
        if isinstance(missing_raw, Sequence) and not isinstance(missing_raw, (str, bytes))
        else []
    )
    bounded_missing = [
        item[:RECOVERY_MISSING_SLOT_MAX_CHARS]
        for item in missing[:RECOVERY_MISSING_SLOTS_MAX_COUNT]
    ]
    source_block, source_telemetry = _bounded_live_source_block(
        sources,
        set(live_citation_urls),
    )
    bounded_draft = original_markdown[:RECOVERY_DRAFT_MAX_CHARS]

    directive = f"""RECOVERY PASS 1 OF 1 — COMPLETE REPORT REPAIR

You have at most {remaining_steps} CodeAgent steps. Return one complete revised Markdown report,
not a patch. Preserve supported content and report structure. Restore or replace only the damaged
claims and missing deliverables listed below. Reuse the existing live evidence first. You may use
targeted retrieval tools only when that evidence is insufficient. Never fabricate facts, URLs, or
citations. Every retained or repaired factual claim must use [cite:URL] markers. If evidence remains
unavailable, state the limitation without inventing support.

APPLIED CRITIQUE DAMAGE ({len(bounded_annotations)} included):
{chr(10).join(claim_rows) or "(none)"}

MISSING DELIVERABLE SLOTS ({len(bounded_missing)} included):
{chr(10).join(f'- {item}' for item in bounded_missing) or "(none)"}

EXISTING LIVE SOURCE CONTEXT:
{source_block}

ORIGINAL FIRST DRAFT:
{bounded_draft}
"""
    directive = directive[:RECOVERY_PROMPT_MAX_CHARS]
    telemetry = {
        "applied_claims_available": len(applied),
        "applied_claims_included": len(bounded_annotations),
        "applied_claims_dropped": max(0, len(applied) - len(bounded_annotations)),
        "missing_slots_available": len(missing),
        "missing_slots_included": len(bounded_missing),
        "missing_slots_dropped": max(0, len(missing) - len(bounded_missing)),
        "draft_chars_available": len(original_markdown),
        "draft_chars_included": len(bounded_draft),
        "draft_chars_dropped": max(0, len(original_markdown) - len(bounded_draft)),
        "prompt_chars": len(directive),
        "prompt_budget_chars": RECOVERY_PROMPT_MAX_CHARS,
        "estimated_prompt_tokens": (len(directive) + 3) // 4,
        **source_telemetry,
    }
    return RecoveryPrompt(directive=directive, telemetry=telemetry)


__all__ = [
    "RECOVERY_MAX_STEPS",
    "RECOVERY_PROMPT_MAX_CHARS",
    "RecoveryDecision",
    "RecoveryPrompt",
    "RecoveryRegressionGuard",
    "build_recovery_prompt",
    "decide_recovery",
    "evaluate_recovery_regression",
    "remaining_recovery_steps",
]
