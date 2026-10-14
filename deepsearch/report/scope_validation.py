"""Exact persisted-text authored-bound validation; never rewrites an artifact."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from deepsearch.mission.scope import ScopeAdmission, page_identity

WORD_COUNTER_VERSION = "unicode-whitespace-v1"
SCOPE_VALIDATION_VERSION = "persisted-scope-v1"
# Freeze Python str.split's Unicode whitespace set, including the four ASCII
# information separators which JavaScript \s does not include. No Markdown parser.
_WHITESPACE = re.compile(
    r"[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a"
    r"\u2028\u2029\u202f\u205f\u3000]+"
)


def count_report_words(text: str) -> int:
    """Count every nonempty whitespace-delimited token, with no markup exclusions."""
    return sum(bool(token) for token in _WHITESPACE.split(text))


def measured_text(text: str) -> dict[str, Any]:
    """Bind a word count to exact UTF-8 bytes and Unicode character length."""
    return {
        "words": count_report_words(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
    }


def validate_persisted_scope(
    *,
    admission: ScopeAdmission,
    pre_render: str,
    persisted: str,
    observed_sources: Sequence[Mapping[str, Any]],
    cited_urls: Sequence[str],
) -> dict[str, Any]:
    """Audit all consultation, not final citation count, against the persisted bytes.

    Unexpected pre-critique evidence remains part of the observed consultation
    set even when the new policy filtered it out of the critic's view. Rejected
    network actions do not count: the model never saw their source content.
    """
    scope = admission.scope
    consultation = set(admission.consulted)
    observed_urls = [str(s.get("url") or "") for s in observed_sources]
    observed_urls += [r["url"] for r in admission.rejected if r["surface"] == "pre_critique"]
    for url in observed_urls:
        try:
            consultation.add(page_identity(url))
        except ValueError:
            # Invalid records already carry source-projection warnings; do not
            # fabricate a distinct page or copy credential-bearing URL text.
            continue
    violations: list[str] = []
    actual = measured_text(persisted)
    if scope.min_words is not None and actual["words"] < scope.min_words:
        violations.append("scope_words_below_minimum")
    if scope.max_words is not None and actual["words"] > scope.max_words:
        violations.append("scope_words_above_maximum")
    if scope.max_sources is not None and len(consultation) > scope.max_sources:
        violations.append("scope_consulted_source_limit_exceeded")
    prohibited = sorted(url for url in consultation if not scope.allows(url))
    if prohibited:
        violations.append("scope_consulted_source_not_allowed")
    if any(not scope.allows(url) for url in cited_urls):
        violations.append("scope_citation_not_allowed")
    snapshot = admission.snapshot()
    warnings: list[str] = []
    source_bounded = scope.restriction != "unrestricted" or scope.max_sources is not None
    if source_bounded and snapshot["unavailable"]:
        warnings.append("scope_sources_unavailable")
    if source_bounded and not consultation:
        warnings.append("scope_no_consulted_evidence")
    return {
        "version": SCOPE_VALIDATION_VERSION,
        "counter_version": WORD_COUNTER_VERSION,
        "controlling_surface": "persisted_report",
        "policy_sha256": snapshot["policy_sha256"],
        "pre_render": measured_text(pre_render),
        "persisted": actual,
        "consulted_count": len(consultation),
        "consulted_urls": sorted(consultation),
        "prohibited_consulted_urls": prohibited,
        "unavailable": snapshot["unavailable"],
        "violations": violations,
        "warnings": warnings,
        "verdict": "violated" if violations else "partial" if warnings else "compliant",
    }
