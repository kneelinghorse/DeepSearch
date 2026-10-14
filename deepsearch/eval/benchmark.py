"""Forensic CAT benchmark harness with structured damage truth.

Encodes the 2026-07-16 fix-or-retire deep-dive grading (decision #823) as a
REPEATABLE, read-only harness over a persisted mission protocol + its report
markdown. It computes the four objective inputs the historical benchmark needs
and emits a per-mission scoresheet plus an acceptance rollup against four
review thresholds. A failed threshold creates evidence for bounded remediation;
it is not a retire or kill-switch instruction.

    1. >= 4/5 missions graded usable-or-better        (§4 rubric — human grade)
    2. < 10 applied structured critique gaps           (AUTOMATED, fail-closed)
    3. false-dead rate < 5% on a human-verified sample (checklist + sample sheet)
    4. persisted telemetry matches ground truth        (AUTOMATED diff)

Grading that needs expert judgment (usable vs weak) is NOT faked: the harness
emits the objective sub-checks + a suggested grade, and leaves ``reviewer_grade``
for a human to fill with recorded rationale (the deep dive itself flagged the
rubric as manual expert labour — encode the checklist, not a fake score).

Inputs are a protocol JSON (``MissionProtocolComplete.to_dict()`` shape) and the
report markdown body (the sibling ``<stem>.md`` file, or the Postgres
``result_markdown`` column). Read-only; NO paid calls; NO production-runtime
imports (PD10 — the eval package stays isolated, mirroring the few constants it
needs rather than importing ``deepsearch.agent.*``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# --- Mirrored constants (PD10 — do NOT import the production modules) ---------

# The canonical unavailability phrase is a broad TEXT DIAGNOSTIC only. It may
# include author-declared limitations as well as critique output, so it MUST NOT
# feed CAT T2. Exact T2 damage comes from persisted structured critique facts.
# A real marker is DELIMITED — the phrase is wrapped in ``[...]`` (critique's
# bracket form), ``(...)`` (the em-dash/paren table-cell form contract_to_prompt
# sanctions), or an em-dash — so a bare PROSE mention of the phrase is not
# counted (calibration on CAT-R001: 54 bracket markers + 2 paren table-cell gaps
# = 56, zero prose false-positives; the deep-dive's manual bracket-only count of
# 54 is reproduced by the bracket subset). Whitespace between words is tolerant
# (``\s+``) so a hard-wrapped / NBSP / double-spaced marker still counts once.
# Counted on a FENCE-STRIPPED body so a "marker format" example inside a code
# block is not counted; the sibling completeness_check.py is likewise fence-aware.
_GAP_MARKER_RE = re.compile(
    r"[\[(—]\s*data\s+unavailable\s+in\s+primary\s+sources", re.IGNORECASE
)

# The historical acceptance thresholds. S87 m04 preserves the T2 boundary but
# repairs its input; authorized post-S87 CAT runs are required before changing
# the numeric bar.
GAP_MARKER_THRESHOLD = 10          # threshold 2: < applied critique gaps
USABLE_FRACTION = 0.8              # threshold 1: >= 80% (>= 4/5) usable-or-better
FALSE_DEAD_RATE_THRESHOLD = 0.05   # threshold 3: < this false-dead rate
EXPECTED_CAT_COUNT = 5             # the benchmark is defined over exactly 5 CAT runs
MIN_SUBSTANTIVE_WORDS = 150        # below this a report body is treated as hollow

GRADES = ("strong", "usable", "weak", "unusable")
USABLE_OR_BETTER = ("strong", "usable")

_URL_RE = re.compile(r"https?://[^\s\])<>\"']+")
_FENCE_OPEN_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _strip_code(text: str) -> str:
    """Remove fenced code blocks and inline code spans from ``text``.

    Fence-aware line scan (honours the opener char + run-length, like
    completeness_check.py): everything from an opening ```/~~~ fence to its
    matching close is dropped, then inline `code` spans are removed. Used so
    gap-marker counting and body-URL harvesting ignore documentation/examples.
    """

    if not text:
        return ""
    out: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    for line in text.split("\n"):
        m = _FENCE_OPEN_RE.match(line)
        if fence_char is None:
            if m:
                fence_char = m.group(1)[0]
                fence_len = len(m.group(1))
                continue  # drop the opening fence line
            out.append(line)
        else:
            if m and m.group(1)[0] == fence_char and len(m.group(1)) >= fence_len:
                fence_char = None  # drop the closing fence line
            # any line while inside a fence is dropped
    return _INLINE_CODE_RE.sub("", "\n".join(out))


# ---------------------------------------------------------------------------
# Per-metric result containers.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TelemetryMismatch:
    """One persisted-vs-recomputed telemetry discrepancy."""

    field: str
    persisted: Any
    recomputed: Any
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "persisted": self.persisted,
            "recomputed": self.recomputed,
            "note": self.note,
        }


@dataclass(slots=True)
class BenchmarkReport:
    """The per-mission forensic scoresheet."""

    mission_id: str
    report_word_count: int
    critique_gap_count: Optional[int]
    critique_status: str
    legacy_critique_gap_count: Optional[int]
    declared_unavailable_count: int
    recovery: Dict[str, Any]
    # Citation liveness.
    citations_total: int
    citations_live: int
    citations_dead: int
    citations_unknown: int
    dead_url_sample: List[Dict[str, Any]]
    # Telemetry truthfulness (threshold 4).
    telemetry_mismatches: List[TelemetryMismatch]
    # Contract-vs-delivery slot satisfaction.
    contract_slots: List[Dict[str, Any]]
    # Grading.
    objective_subchecks: Dict[str, Any]
    suggested_grade: str
    reviewer_grade: Optional[str] = None
    reviewer_rationale: str = ""
    # Threshold outcomes for this single mission (2 + 4 are automatable here;
    # 1 + 3 need the human grade / verified sample, surfaced as pending).
    per_mission_thresholds: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "report_word_count": self.report_word_count,
            "critique_gap_count": self.critique_gap_count,
            "critique_status": self.critique_status,
            "legacy_critique_gap_count": self.legacy_critique_gap_count,
            "declared_unavailable_count": self.declared_unavailable_count,
            "recovery": self.recovery,
            "citations": {
                "total": self.citations_total,
                "live": self.citations_live,
                "dead": self.citations_dead,
                "unknown": self.citations_unknown,
            },
            "dead_url_sample": self.dead_url_sample,
            "telemetry_mismatches": [m.to_dict() for m in self.telemetry_mismatches],
            "contract_slots": self.contract_slots,
            "objective_subchecks": self.objective_subchecks,
            "suggested_grade": self.suggested_grade,
            "reviewer_grade": self.reviewer_grade,
            "reviewer_rationale": self.reviewer_rationale,
            "per_mission_thresholds": self.per_mission_thresholds,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Metric extractors — pure functions over (protocol dict, markdown body).
# ---------------------------------------------------------------------------


def count_gap_markers(markdown_body: str) -> int:
    """Count declared-unavailability text as a broad diagnostic.

    One occurrence of the canonical phrase == one marker, independent of the
    bracket/em-dash form and inter-word whitespace. Counted on a FENCE-STRIPPED
    body so a marker-format example in a code block is not counted. This is the
    quantity the deep-dive reported, but it is not attributable: an author may
    declare an unavailable field without critique applying a repair. CAT T2
    therefore uses :func:`structured_quality_facts`, never this count.
    """

    if not isinstance(markdown_body, str) or not markdown_body:
        return 0
    return len(_GAP_MARKER_RE.findall(_strip_code(markdown_body)))


def _exact_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def structured_quality_facts(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract exact critique/recovery facts without inventing legacy zeros.

    S87 protocols persist the canonical quality record in
    ``report_metadata.forensic.quality_record``. Older protocols retain their
    legacy ``gap_notes_inserted`` value for forensic comparison, but their T2
    exact count is ``None`` because no reconciled annotation ledger exists.
    """

    forensic = _forensic(protocol)
    quality_record = forensic.get("quality_record")
    if isinstance(quality_record, Mapping):
        critique = quality_record.get("critique")
        recovery = quality_record.get("recovery")
        critique_map = critique if isinstance(critique, Mapping) else {}
        recovery_map = recovery if isinstance(recovery, Mapping) else {}
        return {
            "critique_gap_count": _exact_nonnegative_int(
                critique_map.get("applied")
            ),
            "critique_status": str(
                critique_map.get("status") or "unknown"
            ),
            "legacy_critique_gap_count": _exact_nonnegative_int(
                critique_map.get("legacy_gap_notes_inserted")
            ),
            "recovery": {
                "passes": _exact_nonnegative_int(recovery_map.get("passes")),
                "outcome": str(recovery_map.get("outcome") or "unknown"),
                "selected_artifact": str(
                    recovery_map.get("selected_artifact") or "unknown"
                ),
                "triggered": (
                    recovery_map.get("triggered")
                    if isinstance(recovery_map.get("triggered"), bool)
                    else None
                ),
            },
        }

    critique_raw = forensic.get("critique_telemetry")
    critique = critique_raw if isinstance(critique_raw, Mapping) else {}
    summary_raw = critique.get("annotation_summary")
    summary = summary_raw if isinstance(summary_raw, Mapping) else {}
    error = str(critique.get("error") or "").strip()
    if error:
        status = "error"
        exact = None
    elif summary and summary.get("reconciled") is True:
        status = "ok"
        exact = _exact_nonnegative_int(summary.get("applied"))
        if exact is None:
            status = "error"
    elif critique:
        status = "legacy_unstructured"
        exact = None
    else:
        status = "unknown"
        exact = None
    return {
        "critique_gap_count": exact,
        "critique_status": status,
        "legacy_critique_gap_count": _exact_nonnegative_int(
            critique.get("gap_notes_inserted")
        ),
        "recovery": {
            "passes": None,
            "outcome": "unknown",
            "selected_artifact": "unknown",
            "triggered": None,
        },
    }


def _iter_citations(protocol: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if not isinstance(protocol, Mapping):
        return []
    cites = protocol.get("citations")
    return [c for c in cites if isinstance(c, Mapping)] if isinstance(cites, list) else []


def citation_liveness(
    protocol: Mapping[str, Any],
    *,
    sample_cap: int = 25,
) -> Dict[str, Any]:
    """Liveness rollup + a dead-URL SAMPLE SHEET for human false-dead review.

    ``live is True/False/None`` maps to live / dead / unknown. The dead sample
    (deduped by URL, capped) is what a human spot-checks for threshold 3 — the
    harness cannot itself decide whether a dead verdict is a *false* dead.
    """

    live = dead = unknown = 0
    seen_dead: set[str] = set()
    dead_sample: List[Dict[str, Any]] = []
    for ann in _iter_citations(protocol):
        alive = ann.get("live")
        if alive is True:
            live += 1
        elif alive is False:
            dead += 1
            url = str(ann.get("url") or "").strip()
            if url and url not in seen_dead:
                seen_dead.add(url)
                if len(dead_sample) < sample_cap:
                    dead_sample.append(
                        {
                            "url": url,
                            "title": str(ann.get("title") or ""),
                            "verified_alive": None,  # human fills: True => false-dead
                        }
                    )
        else:
            unknown += 1
    return {
        "total": live + dead + unknown,
        "live": live,
        "dead": dead,
        "unknown": unknown,
        "distinct_dead": len(seen_dead),
        "dead_sample": dead_sample,
        # Truncated only when the per-URL sample hit the cap (not merely because
        # dead citations share URLs — the sample is deduped by URL).
        "dead_sample_truncated": len(seen_dead) > len(dead_sample),
    }


def _distinct_body_urls(markdown_body: str) -> set[str]:
    if not markdown_body:
        return set()
    return {u.rstrip(".,);") for u in _URL_RE.findall(markdown_body)}


def _report_metadata(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(protocol, Mapping):
        return {}
    rm = protocol.get("report_metadata")
    return rm if isinstance(rm, Mapping) else {}


def _forensic(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    f = _report_metadata(protocol).get("forensic")
    return f if isinstance(f, Mapping) else {}


def telemetry_diff(protocol: Mapping[str, Any], markdown_body: str) -> List[TelemetryMismatch]:
    """Diff persisted telemetry against values recomputed from the artifacts.

    Threshold 4 ("persisted telemetry matches ground truth"). Targets the
    fields the deep-dive caught lying ("0 sources", "Coverage 0%", references
    == 0 despite a populated References list). An empty list == telemetry
    matches ground truth for the checked fields. Degrades to no-mismatch on a
    malformed/None protocol or body (never raises).
    """

    mismatches: List[TelemetryMismatch] = []
    body = markdown_body if isinstance(markdown_body, str) else ""
    rm = _report_metadata(protocol)
    forensic = _forensic(protocol)

    # (a) ``references`` truthfulness, keyed on the AUTHORITATIVE citation set
    # (protocol.citations) plus FENCE-STRIPPED body URLs — an example URL inside
    # a code block is not evidence that references should be > 0 (adversarial
    # finding: fence-blind harvesting false-flagged a truthful references==0).
    citations = _iter_citations(protocol)
    distinct_cited = len({str(c.get("url") or "").strip() for c in citations if c.get("url")})
    body_urls = _distinct_body_urls(_strip_code(body))
    persisted_refs = rm.get("references")
    if isinstance(persisted_refs, int) and persisted_refs == 0 and (
        distinct_cited > 0 or len(body_urls) > 0
    ):
        # The documented lie (references==0 despite a populated References list /
        # cited URLs). A NON-zero references that merely differs from the total
        # distinct-citation count is NOT flagged: references truthfully counts
        # distinct LIVE cited URLs in the final markdown, which legitimately
        # differs from protocol.citations (dead + orphan survive there post-S84);
        # ±(dead+orphan) is normal, not a lie (verified on real m03 data).
        mismatches.append(
            TelemetryMismatch(
                "references",
                0,
                max(distinct_cited, len(body_urls)),
                f"references==0 but {distinct_cited} cited URL(s) + "
                f"{len(body_urls)} non-fenced body URL(s) are present",
            )
        )

    # (b) citation_telemetry.citations_extracted. Extraction counts [cite:]
    # markers BEFORE quarantine drops hallucinated ones, so citations_extracted
    # is legitimately >= the surviving len(citations) (verified on real m03
    # data: R003 extracted 138, 127 survive after 11 hallucinated quarantined).
    # Only a gross lie (==0 despite surviving citations) or the impossible case
    # (extracted < survivors) is a mismatch.
    citation_telemetry = forensic.get("citation_telemetry")
    if isinstance(citation_telemetry, Mapping):
        extracted = citation_telemetry.get("citations_extracted")
        if isinstance(extracted, int) and (
            (extracted == 0 and len(citations) > 0) or extracted < len(citations)
        ):
            mismatches.append(
                TelemetryMismatch(
                    "citation_telemetry.citations_extracted",
                    extracted,
                    len(citations),
                    "citations_extracted==0 with citations present, or < surviving count",
                )
            )

    # (c) a non-empty report must not persist sources_collected == 0.
    persisted_sources = rm.get("sources_collected")
    if isinstance(persisted_sources, int) and persisted_sources == 0 and body.strip():
        mismatches.append(
            TelemetryMismatch(
                "sources_collected",
                0,
                ">0 expected",
                "sources_collected==0 on a non-empty report (the '0 sources' lie)",
            )
        )

    return mismatches


def _slot_satisfied(body_lower: str, slot: str) -> bool:
    """Heuristic: a contract slot is 'present' when its distinctive tokens
    appear in the report body. Deliberately lenient (recall-biased) — this is a
    reviewable signal, not a gate (the roadmap §8 anti-churn rule).
    """

    tokens = [t for t in re.split(r"[^a-z0-9]+", slot.lower()) if len(t) >= 4]
    if not tokens:
        return bool(slot.strip()) and slot.lower() in body_lower
    hits = sum(1 for t in tokens if t in body_lower)
    return hits >= max(1, (len(tokens) + 1) // 2)


def contract_slots(protocol: Mapping[str, Any], markdown_body: str) -> List[Dict[str, Any]]:
    """Contract-vs-delivery slot table.

    Slots are drawn from the contract surfaces present in the protocol:
    ``research_statement.success_metrics`` + the ``key_questions``. Each slot is
    marked satisfied/unsatisfied by the lenient token-presence heuristic. This
    is the 'contract slot table' input to the usable/weak grade, not a gate.
    """

    if not isinstance(protocol, Mapping):
        return []
    body_lower = (markdown_body or "").lower() if isinstance(markdown_body, str) else ""
    slots: List[Dict[str, Any]] = []

    rs = protocol.get("research_statement")
    if isinstance(rs, Mapping):
        metrics = rs.get("success_metrics")
        if isinstance(metrics, list):  # a scalar/None must not crash the iteration
            for metric in metrics:
                text = str(metric or "").strip()
                if text:
                    slots.append(
                        {
                            "kind": "success_metric",
                            "slot": text,
                            "satisfied": _slot_satisfied(body_lower, text),
                        }
                    )

    key_questions = protocol.get("key_questions")
    if isinstance(key_questions, list):
        for kq in key_questions:
            if not isinstance(kq, Mapping):
                continue
            q = str(kq.get("question") or "").strip()
            if not q:
                continue
            answered = str(kq.get("status") or "").lower() in ("answered", "complete", "resolved")
            slots.append(
                {
                    "kind": "key_question",
                    "slot": q,
                    "status": kq.get("status"),
                    "satisfied": answered or _slot_satisfied(body_lower, q),
                }
            )

    return slots


def _suggest_grade(
    *,
    critique_gap_count: Optional[int],
    critique_status: str,
    declared_unavailable_count: int,
    contract_slots_list: Sequence[Mapping[str, Any]],
    telemetry_mismatches: Sequence[Any],
    citations_total: int,
    report_word_count: int,
) -> tuple[str, Dict[str, Any]]:
    """Compute the objective sub-checks and a SUGGESTED grade (an ANCHOR only).

    The suggestion leans on the RELIABLE objective signals (word count, gap
    structured critique damage, telemetry, citations) and uses the fragile lexical slot ratio ONLY
    as a POSITIVE bonus toward ``strong`` — never as a negative that forces
    ``unusable``/``weak`` (adversarial finding: the lexical matcher false-LOWs a
    paraphrasing good report and false-HIGHs a contract-less one).

      - unusable: hollow/empty body (word count < MIN_SUBSTANTIVE_WORDS)
      - weak:     unknown/errored structured damage, heavy exact damage, or zero citations
      - strong:   substantive, < threshold exact damage, clean telemetry, citations
                  present, a contract present AND >= 90% slots satisfied
      - usable:   everything else (incl. a low lexical slot ratio — the reviewer
                  confirms coverage; a low ratio is NOT a downgrade)
    """

    total_slots = len(contract_slots_list)
    satisfied = sum(1 for s in contract_slots_list if s.get("satisfied"))
    slot_ratio = (satisfied / total_slots) if total_slots else None
    telemetry_clean = len(telemetry_mismatches) == 0
    critique_known = critique_status == "ok" and critique_gap_count is not None
    subchecks = {
        "report_word_count": report_word_count,
        "body_substantive": report_word_count >= MIN_SUBSTANTIVE_WORDS,
        "critique_damage_known": critique_known,
        "critique_status": critique_status,
        "critique_gaps_under_threshold": (
            critique_gap_count < GAP_MARKER_THRESHOLD
            if critique_status == "ok" and critique_gap_count is not None
            else None
        ),
        "critique_gap_count": critique_gap_count,
        "declared_unavailable_count": declared_unavailable_count,
        "contract_slots_total": total_slots,
        "contract_slots_satisfied": satisfied,
        "slot_satisfaction_ratio": round(slot_ratio, 3) if slot_ratio is not None else None,
        "telemetry_clean": telemetry_clean,
        "has_citations": citations_total > 0,
        "has_contract_surface": total_slots > 0,
    }

    if report_word_count < MIN_SUBSTANTIVE_WORDS:
        grade = "unusable"
    elif (
        not critique_known
        or (
            critique_gap_count is not None
            and critique_gap_count >= GAP_MARKER_THRESHOLD
        )
        or citations_total == 0
    ):
        grade = "weak"
    elif total_slots > 0 and slot_ratio is not None and slot_ratio >= 0.9 and telemetry_clean:
        grade = "strong"
    else:
        grade = "usable"
    return grade, subchecks


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------


def benchmark(
    mission_id: str,
    protocol: Mapping[str, Any],
    markdown_body: str,
) -> BenchmarkReport:
    """Build the full scoresheet for one mission from its protocol + body.

    Degrades to a scoresheet (never raises) on a malformed protocol / None body
    — every extractor guards its inputs.
    """

    body = markdown_body if isinstance(markdown_body, str) else ""
    declared_unavailable = count_gap_markers(body)
    quality = structured_quality_facts(protocol)
    critique_gap_count = quality["critique_gap_count"]
    critique_status = quality["critique_status"]
    live = citation_liveness(protocol)
    mismatches = telemetry_diff(protocol, body)
    slots = contract_slots(protocol, body)
    word_count = len(body.split())

    suggested, subchecks = _suggest_grade(
        critique_gap_count=critique_gap_count,
        critique_status=critique_status,
        declared_unavailable_count=declared_unavailable,
        contract_slots_list=slots,
        telemetry_mismatches=mismatches,
        citations_total=live["total"],
        report_word_count=word_count,
    )

    per_mission = {
        # threshold 2 — fully automatable here.
        "structured_critique_gaps_lt_10": {
            "value": critique_gap_count,
            "status": critique_status,
            "pass": (
                critique_status == "ok"
                and critique_gap_count is not None
                and critique_gap_count < GAP_MARKER_THRESHOLD
            ),
            "automated": True,
            "fail_closed": True,
        },
        "declared_unavailability_diagnostic": {
            "value": declared_unavailable,
            "automated": True,
            "note": "broad text count; never overwrites structured T2",
        },
        # threshold 4 — automatable here (empty diff == pass).
        "telemetry_matches_truth": {
            "mismatch_count": len(mismatches),
            "pass": len(mismatches) == 0,
            "automated": True,
        },
        # threshold 3 — needs the human-verified dead sample; pending until filled.
        "false_dead_lt_5pct": {
            "dead_total": live["dead"],
            "sample_size": len(live["dead_sample"]),
            "pass": None,
            "automated": False,
            "note": "fill dead_url_sample[].verified_alive, then re-run rollup",
        },
        # threshold 1 — needs the reviewer grade; pending until filled.
        "graded_usable_or_better": {
            "suggested_grade": suggested,
            "reviewer_grade": None,
            "pass": None,
            "automated": False,
        },
    }

    notes: List[str] = []
    if live["total"] == 0:
        notes.append(
            "0 citations in protocol — reviewer must confirm the report's sources by hand."
        )
    if not body.strip():
        notes.append(
            "report markdown body is EMPTY/unavailable — text + slot signals are blind."
        )
    if not subchecks["has_contract_surface"]:
        notes.append(
            "no contract surface in the protocol (no success_metrics / key_questions) — "
            "reviewer must confirm this is the RIGHT mission's report (wrong-mission class)."
        )
    if critique_status != "ok" or critique_gap_count is None:
        notes.append(
            "structured critique damage is unknown/errored — CAT T2 fails closed; "
            "declared-unavailability text remains diagnostic only."
        )
    if live["dead_sample_truncated"]:
        notes.append(
            f"dead-URL sample truncated at {len(live['dead_sample'])} of {live['dead']} "
            "dead citations — verify the sample, then extrapolate."
        )

    return BenchmarkReport(
        mission_id=mission_id,
        report_word_count=word_count,
        critique_gap_count=critique_gap_count,
        critique_status=critique_status,
        legacy_critique_gap_count=quality["legacy_critique_gap_count"],
        declared_unavailable_count=declared_unavailable,
        recovery=quality["recovery"],
        citations_total=live["total"],
        citations_live=live["live"],
        citations_dead=live["dead"],
        citations_unknown=live["unknown"],
        dead_url_sample=live["dead_sample"],
        telemetry_mismatches=mismatches,
        contract_slots=slots,
        objective_subchecks=subchecks,
        suggested_grade=suggested,
        per_mission_thresholds=per_mission,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Artifact loading (local files; the CLI also supports Postgres extraction).
# ---------------------------------------------------------------------------


def load_protocol(protocol_path: Path) -> Dict[str, Any]:
    data = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"protocol at {protocol_path} is not a JSON object")
    return data


def resolve_markdown_path(protocol_path: Path) -> Path:
    """``<stem>_protocol.json`` -> sibling ``<stem>.md`` (save_report convention)."""

    p = Path(protocol_path)
    stem = p.name[: -len("_protocol.json")] if p.name.endswith("_protocol.json") else p.stem
    return p.with_name(f"{stem}.md")


def load_markdown(protocol_path: Path) -> str:
    md = resolve_markdown_path(protocol_path)
    if md.exists():
        return md.read_text(encoding="utf-8")
    return ""


def benchmark_from_paths(mission_id: str, protocol_path: Path) -> BenchmarkReport:
    protocol = load_protocol(protocol_path)
    body = load_markdown(protocol_path)
    return benchmark(mission_id, protocol, body)


# ---------------------------------------------------------------------------
# Go/no-go rollup against the four thresholds (decision #823, roadmap §6).
# ---------------------------------------------------------------------------


def evaluate_go_no_go(reports: Sequence[BenchmarkReport]) -> Dict[str, Any]:
    """Roll per-mission scoresheets into a fail-closed acceptance verdict.

    Rules (decision #823 — the thresholds cannot be softened):
    - Reports are DEDUPED by mission_id first (a duplicate must never inflate a
      count) and warnings surface duplicates / off-cardinality (expected 5 CATs).
    - Thresholds 2 (gap) + 4 (telemetry) are automatic and always complete.
      Thresholds 1 (reviewer grade) + 3 (verified dead sample) need human input;
      threshold 3 is VACUOUSLY complete+pass when there are zero dead citations.
    - Verdict: FAIL the moment ANY complete threshold is missed (an already-missed
      automated gate can't be un-missed — never masked as PENDING). Otherwise
      PENDING while any human-gated threshold is incomplete. PASS only when all
      four are complete AND pass. NEVER auto-PASS on unfilled human inputs.
    """

    # Dedup by mission_id — duplicates are a caller error and must not inflate
    # the usable count (adversarial finding: a duplicated usable report flipped
    # a 3/4 FAIL to PASS).
    seen: Dict[str, BenchmarkReport] = {}
    duplicate_ids: List[str] = []
    for r in reports:
        if r.mission_id in seen:
            duplicate_ids.append(r.mission_id)
        else:
            seen[r.mission_id] = r
    distinct = list(seen.values())
    n = len(distinct)

    warnings: List[str] = []
    if duplicate_ids:
        warnings.append(f"duplicate mission_ids ignored: {sorted(set(duplicate_ids))}")
    if n != EXPECTED_CAT_COUNT:
        warnings.append(
            f"expected exactly {EXPECTED_CAT_COUNT} distinct CAT missions, got {n} — "
            f"threshold 1 (>= {USABLE_FRACTION:.0%} usable) assumes the full set"
        )

    # Threshold 2 — exact structured critique damage only. Unknown/legacy/error
    # data FAILS CLOSED; broad declared-unavailability text cannot substitute.
    known_t2 = [
        r
        for r in distinct
        if r.critique_status == "ok" and r.critique_gap_count is not None
    ]
    unknown_t2 = [r.mission_id for r in distinct if r not in known_t2]
    worst_gap = max(
        (r.critique_gap_count for r in known_t2 if r.critique_gap_count is not None),
        default=None,
    )
    t2_pass = (
        n > 0
        and not unknown_t2
        and all(
            r.critique_gap_count is not None
            and r.critique_gap_count < GAP_MARKER_THRESHOLD
            for r in distinct
        )
    )

    # Threshold 4 — every report's telemetry clean.
    total_mismatches = sum(len(r.telemetry_mismatches) for r in distinct)
    t4_pass = n > 0 and total_mismatches == 0

    # Threshold 1 — >= 80% (>= 4/5) reviewer-graded usable-or-better. Expressed
    # as a fraction of n so an off-cardinality list cannot pass below the bar.
    graded = [r for r in distinct if r.reviewer_grade in GRADES]
    usable = sum(1 for r in graded if r.reviewer_grade in USABLE_OR_BETTER)
    min_usable = ceil(USABLE_FRACTION * n) if n else 1
    t1_complete = n > 0 and len(graded) == n
    t1_pass = usable >= min_usable if t1_complete else None

    # Threshold 3 — false-dead rate over the human-verified sample. Zero dead
    # citations => vacuously 0% => complete + pass (the clean state m01 targets).
    fd = _false_dead_rate(distinct)
    total_dead = sum(r.citations_dead for r in distinct)
    if total_dead == 0:
        t3_complete = True
        t3_pass: Optional[bool] = True
    else:
        t3_complete = (
            fd["verified_total"] > 0
            and fd["unverified_remaining"] == 0
            and fd["invalid"] == 0
        )
        t3_pass = (fd["rate_raw"] < FALSE_DEAD_RATE_THRESHOLD) if t3_complete else None
    if fd["invalid"] > 0:
        warnings.append(
            f"{fd['invalid']} dead_url_sample entry(ies) have a non-bool verified_alive — "
            "treated as UNVERIFIED (use true/false); threshold 3 stays incomplete"
        )

    thresholds = {
        "t1_usable_or_better": {
            "required": f">= {min_usable}/{n} (>= {USABLE_FRACTION:.0%})",
            "graded": len(graded),
            "usable_or_better": usable,
            "complete": t1_complete,
            "pass": t1_pass,
        },
        "t2_critique_damage": {
            "required": f"< {GAP_MARKER_THRESHOLD} applied structured critique gaps per report",
            "worst": worst_gap,
            "unknown_or_errored": unknown_t2,
            "complete": n > 0,
            "pass": t2_pass if n > 0 else None,
        },
        "t3_false_dead": {
            "required": f"< {FALSE_DEAD_RATE_THRESHOLD:.0%} on verified sample",
            "rate": round(fd["rate_raw"], 4),
            "dead_total": total_dead,
            "sample_size": fd["verified_total"],
            "unverified_remaining": fd["unverified_remaining"],
            "complete": t3_complete,
            "pass": t3_pass,
        },
        "t4_telemetry_truth": {
            "required": "0 mismatches",
            "total_mismatches": total_mismatches,
            "complete": n > 0,
            "pass": t4_pass if n > 0 else None,
        },
    }

    # A complete-and-failed threshold => FAIL immediately (V2: no masking a
    # missed automated gate as PENDING). Then PENDING while incomplete; else PASS.
    checks = [
        (n > 0, t2_pass),          # t2 complete when n>0
        (n > 0, t4_pass),          # t4 complete when n>0
        (t1_complete, t1_pass),
        (t3_complete, t3_pass),
    ]
    any_failed = any(complete and (passed is False) for complete, passed in checks)
    all_complete = all(complete for complete, _ in checks) and n > 0
    if any_failed:
        verdict = "FAIL"
    elif not all_complete:
        verdict = "PENDING"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "mission_count": n,
        "thresholds": thresholds,
        "warnings": warnings,
        "note": (
            "ANY missed threshold => FAIL and creates evidence for the next bounded "
            "remediation; it is not a retire switch. PENDING means a human-gated "
            "threshold (reviewer_grade / verified dead sample) is unfilled — never "
            "read as PASS."
        ),
    }


_CAT_BASELINE_PATH = Path(__file__).with_name("cat_baseline.json")


def load_cat_baseline() -> Dict[str, Any]:
    """Load the 2026-07-16 CAT baseline scoresheet (the pre-fix numbers)."""

    data = json.loads(_CAT_BASELINE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("cat_baseline.json is not a JSON object")
    return data


def compare_to_baseline(reports: Sequence[BenchmarkReport]) -> List[Dict[str, Any]]:
    """Per-CAT delta of the re-dispatched results vs the pre-fix baseline.

    Matches each report to a baseline run by cat_id or mission UUID. The
    declared-unavailability delta remains a historical text diagnostic. Exact
    structured critique damage is reported separately and has no legacy
    baseline when the old protocol lacks a reconciled annotation ledger.
    """

    baseline = load_cat_baseline()
    by_id = {r["cat_id"]: r for r in baseline["runs"]}
    by_uuid = {r["mission_uuid"]: r for r in baseline["runs"]}
    rows: List[Dict[str, Any]] = []
    for rep in reports:
        b = by_id.get(rep.mission_id) or by_uuid.get(rep.mission_id)
        row: Dict[str, Any] = {
            "mission_id": rep.mission_id,
            "critique_gap_count": rep.critique_gap_count,
            "critique_status": rep.critique_status,
            "legacy_critique_gap_count": rep.legacy_critique_gap_count,
            "declared_unavailable_count": rep.declared_unavailable_count,
            "recovery": dict(rep.recovery),
            "suggested_grade": rep.suggested_grade,
            "reviewer_grade": rep.reviewer_grade,
        }
        if b is not None:
            base_gap = b["baseline_gap_markers"]
            row["baseline_grade"] = b["baseline_grade"]
            row["baseline_gap_markers"] = base_gap
            row["declared_unavailable_delta"] = (
                rep.declared_unavailable_count - base_gap
                if isinstance(base_gap, int)
                else None
            )
        rows.append(row)
    return rows


def _false_dead_rate(reports: Sequence[BenchmarkReport]) -> Dict[str, Any]:
    """False-dead rate over the human-verified dead sample.

    ``verified_alive`` MUST be a strict bool: ``True`` == confirmed false-dead
    (the URL is actually live), ``False`` == a genuine dead. Anything else
    (``None`` or, critically, a truthy non-bool like ``1`` / ``"yes"``) is
    UNVERIFIED — never silently read as a confirmed dead (adversarial finding: a
    truthy-non-bool inversion deflated a 100% false-dead rate to 0.0 and flipped
    the verdict to PASS). ``rate_raw`` is unrounded so a value just under 5% is
    not rounded up across the boundary into a FAIL.
    """

    false_dead = 0
    verified_total = 0
    unverified = 0
    invalid = 0
    for r in reports:
        for entry in r.dead_url_sample:
            v = entry.get("verified_alive")
            if v is True:
                verified_total += 1
                false_dead += 1
            elif v is False:
                verified_total += 1
            elif v is None:
                unverified += 1
            else:
                invalid += 1  # non-bool -> unverified + surfaced as a warning
    rate = (false_dead / verified_total) if verified_total else 0.0
    return {
        "rate_raw": rate,
        "verified_total": verified_total,
        "false_dead": false_dead,
        "unverified_remaining": unverified + invalid,
        "invalid": invalid,
    }


__all__ = [
    "BenchmarkReport",
    "TelemetryMismatch",
    "GAP_MARKER_THRESHOLD",
    "USABLE_FRACTION",
    "EXPECTED_CAT_COUNT",
    "MIN_SUBSTANTIVE_WORDS",
    "FALSE_DEAD_RATE_THRESHOLD",
    "GRADES",
    "USABLE_OR_BETTER",
    "count_gap_markers",
    "structured_quality_facts",
    "citation_liveness",
    "telemetry_diff",
    "contract_slots",
    "benchmark",
    "benchmark_from_paths",
    "load_protocol",
    "load_markdown",
    "resolve_markdown_path",
    "evaluate_go_no_go",
    "load_cat_baseline",
    "compare_to_baseline",
]
