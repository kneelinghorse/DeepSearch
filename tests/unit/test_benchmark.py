"""Unit tests for the CAT benchmark's structured quality definitions.

Validates the metric extractors (gap markers, citation liveness, telemetry-vs-
truth diff, contract slots), the suggested-grade mapping, and the go/no-go
rollup logic — deterministically over synthetic current-shape protocols plus a
real cached protocol. The end-to-end CAT calibration (reproduce 54/65/n-a/23/58)
is DATABASE_URL-gated (see sprint_85_benchmark_rubric.md §5); these tests pin the
harness *definitions* so it is trustworthy before the CAT protocols are pulled.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from deepsearch.eval import benchmark as B  # noqa: N812 — concise alias for the harness

# ---------------------------------------------------------------------------
# Fixtures — synthetic current-shape protocol + body.
# ---------------------------------------------------------------------------


def _quality_record(*, applied: int = 0, status: str = "ok") -> dict:
    return {
        "schema_version": 2,
        "critique": {
            "status": status,
            "applied": applied if status == "ok" else None,
            "skipped": 0 if status == "ok" else None,
            "dropped": 0 if status == "ok" else None,
            "errors": 0 if status == "ok" else None,
            "claim_count": 20,
            "applied_ratio": applied / 20 if status == "ok" else None,
            "legacy_gap_notes_inserted": applied,
        },
        "recovery": {
            "passes": 0,
            "outcome": "not_triggered",
            "selected_artifact": "first_pass",
        },
    }


def _protocol(**overrides):
    proto = {
        "mission_id": "SYN",
        "citations": [
            {"url": "https://a.com/1", "live": True, "title": "A"},
            {"url": "https://b.com/2", "live": False, "title": "B"},
            {"url": "https://c.com/3", "live": False, "title": "C"},
            {"url": "https://d.com/4", "live": None, "title": "D"},
        ],
        "report_metadata": {
            "references": 4,
            "sources_collected": 12,
            "forensic": {
                "citation_telemetry": {"citations_extracted": 4},
                "quality_record": _quality_record(),
            },
        },
        "research_statement": {"success_metrics": ["Compare tool A latency and tool B latency"]},
        "key_questions": [{"question": "What is tool A latency", "status": "answered"}],
    }
    proto.update(overrides)
    return proto


# A substantive (> MIN_SUBSTANTIVE_WORDS) body that satisfies the tool-A/B
# latency success metric, carries exactly one gap marker, and renders two URLs.
_BODY = (
    "# Comparative Latency Report\n\n"
    "## Overview\n\n"
    "This report compares tool A and tool B on request latency across several "
    "representative production workloads. We measured latency under identical "
    "sustained load, recorded the median and tail figures for each tool, and "
    "documented where every number came from so a reader can reproduce the "
    "comparison. The goal is to give a procurement team enough evidence to "
    "choose between the two platforms on latency grounds without re-running the "
    "benchmarks themselves. Each section below covers one tool, states the "
    "measured latency, and links to the raw capture used to derive it.\n\n"
    "## Tool A\n\n"
    "Tool A latency is about 5 ms at the median under sustained load, and we "
    "compare that figure against tool B throughout this section. The tail "
    "latency stayed under 20 ms across every workload we exercised [1].\n\n"
    "## Tool B\n\n"
    "Tool B latency for the enterprise tier is "
    "[Data unavailable in primary sources]; the vendor did not publish a "
    "comparable sustained-load figure and our probes could not establish one.\n\n"
    "See https://a.com/1 and https://b.com/2 for the raw latency measurements.\n"
)


# ---------------------------------------------------------------------------
# 1. Gap markers (threshold 2).
# ---------------------------------------------------------------------------


def test_count_gap_markers_counts_all_forms():
    body = (
        "Claim one [Data unavailable in primary sources].\n"
        "Claim two —Data unavailable in primary sources—\n"
        "Claim three (data unavailable in primary sources)\n"  # case-insensitive
        "A real sentence with no marker.\n"
    )
    assert B.count_gap_markers(body) == 3


def test_count_gap_markers_empty_and_nonstring():
    assert B.count_gap_markers("") == 0
    assert B.count_gap_markers(None) == 0  # type: ignore[arg-type]
    assert B.count_gap_markers("nothing here") == 0


# ---------------------------------------------------------------------------
# 2. Citation liveness (threshold 3 sample sheet).
# ---------------------------------------------------------------------------


def test_citation_liveness_counts_and_sample():
    live = B.citation_liveness(_protocol())
    # fixture: a=live, b=dead, c=dead, d=None(unknown)
    assert (live["live"], live["dead"], live["unknown"]) == (1, 2, 1)
    assert live["total"] == 4
    urls = {e["url"] for e in live["dead_sample"]}
    assert urls == {"https://b.com/2", "https://c.com/3"}
    assert all(e["verified_alive"] is None for e in live["dead_sample"])
    assert live["dead_sample_truncated"] is False


def test_citation_liveness_dedups_dead_urls_and_flags_cap():
    dupes = _protocol(citations=[{"url": "https://x.com", "live": False} for _ in range(5)])
    live = B.citation_liveness(dupes, sample_cap=25)
    assert live["dead"] == 5
    assert live["distinct_dead"] == 1
    assert len(live["dead_sample"]) == 1  # deduped by URL
    assert live["dead_sample_truncated"] is False  # dedup is not a cap-hit

    many = _protocol(citations=[{"url": f"https://x.com/{i}", "live": False} for i in range(30)])
    live2 = B.citation_liveness(many, sample_cap=25)
    assert len(live2["dead_sample"]) == 25
    assert live2["dead_sample_truncated"] is True


def test_citation_liveness_missing_citations_field():
    assert B.citation_liveness({})["total"] == 0
    assert B.citation_liveness({"citations": "not-a-list"})["total"] == 0


# ---------------------------------------------------------------------------
# 3. Telemetry-vs-truth diff (threshold 4).
# ---------------------------------------------------------------------------


def test_telemetry_diff_clean_protocol_no_mismatch():
    assert B.telemetry_diff(_protocol(), _BODY) == []


def test_telemetry_diff_flags_references_zero_lie():
    proto = _protocol()
    proto["report_metadata"]["references"] = 0
    ms = B.telemetry_diff(proto, _BODY)
    fields = {m.field for m in ms}
    assert "references" in fields
    # single references mismatch, not double-counted
    assert sum(1 for m in ms if m.field == "references") == 1


def test_telemetry_diff_citations_extracted_only_flags_real_lies():
    """extracted counts markers BEFORE hallucinated-quarantine, so extracted >=
    surviving len(citations) is legitimate (not flagged). Only extracted==0 with
    citations present, or extracted < survivors, is a lie."""

    # Legitimate: extracted (99) >= 4 survivors (95 quarantined) -> NOT flagged.
    p_ok = _protocol()
    p_ok["citations"] = [{"url": "https://a", "live": True}] * 4
    p_ok["report_metadata"]["forensic"]["citation_telemetry"]["citations_extracted"] = 99
    assert not any(m.field == "citation_telemetry.citations_extracted" for m in B.telemetry_diff(p_ok, _BODY))

    # Lie: extracted==0 despite surviving citations.
    p0 = _protocol()
    p0["report_metadata"]["forensic"]["citation_telemetry"]["citations_extracted"] = 0
    assert any(m.field == "citation_telemetry.citations_extracted" for m in B.telemetry_diff(p0, _BODY))

    # Impossible: extracted < survivors.
    plt = _protocol()  # 4 citations in fixture
    plt["report_metadata"]["forensic"]["citation_telemetry"]["citations_extracted"] = 2
    assert any(m.field == "citation_telemetry.citations_extracted" for m in B.telemetry_diff(plt, _BODY))


def test_telemetry_diff_flags_sources_collected_zero_on_nonempty_body():
    proto = _protocol()
    proto["report_metadata"]["sources_collected"] = 0
    ms = B.telemetry_diff(proto, _BODY)
    assert any(m.field == "sources_collected" for m in ms)


# ---------------------------------------------------------------------------
# 4. Contract slots.
# ---------------------------------------------------------------------------


def test_contract_slots_satisfied_and_unsatisfied():
    slots = B.contract_slots(_protocol(), _BODY)
    kinds = {s["kind"] for s in slots}
    assert "success_metric" in kinds and "key_question" in kinds
    # The success metric mentions "tool A latency"/"tool B latency" — present in body.
    sm = next(s for s in slots if s["kind"] == "success_metric")
    assert sm["satisfied"] is True

    # A slot whose distinctive tokens are absent is unsatisfied.
    proto = _protocol(research_statement={"success_metrics": ["Quantify kubernetes helm chart provenance signatures"]})
    slots2 = B.contract_slots(proto, _BODY)
    assert slots2[0]["satisfied"] is False


# ---------------------------------------------------------------------------
# 5. Suggested grade mapping.
# ---------------------------------------------------------------------------


def test_suggested_grade_strong():
    r = B.benchmark("SYN", _protocol(), _BODY)
    # 1 gap marker (< 10), slots satisfied, telemetry clean, citations present.
    assert r.suggested_grade in ("strong", "usable")


def test_suggested_grade_weak_on_heavy_gap_damage():
    body = _BODY + "\n".join(["Fact [Data unavailable in primary sources]."] * 12)
    proto = _protocol()
    proto["report_metadata"]["forensic"]["quality_record"] = _quality_record(
        applied=12
    )
    r = B.benchmark("SYN", proto, body)
    assert r.critique_gap_count == 12
    assert r.declared_unavailable_count >= 10
    assert r.suggested_grade == "weak"


def test_declared_unavailability_cannot_overwrite_exact_critique_damage():
    body = _BODY + "\n".join(
        ["Fact [Data unavailable in primary sources]."] * 20
    )
    report = B.benchmark("SYN", _protocol(), body)

    assert report.critique_gap_count == 0
    assert report.declared_unavailable_count >= 20
    assert report.per_mission_thresholds["structured_critique_gaps_lt_10"][
        "pass"
    ] is True


def test_count_gap_markers_requires_delimiter_not_bare_prose():
    """A real marker is delimited ([...]/(...)/em-dash); a bare prose mention of
    the phrase is NOT a gap marker (calibration on CAT-R001: 0 prose false-pos)."""

    assert B.count_gap_markers("We note that data unavailable in primary sources is our sentinel.") == 0
    assert B.count_gap_markers("cell | — (Data unavailable in primary sources) |") == 1
    assert B.count_gap_markers("y —Data unavailable in primary sources—") == 1


def test_suggested_grade_unusable_on_hollow_document():
    # No slot satisfied (body empty) -> ratio 0 -> unusable.
    proto = _protocol(
        research_statement={"success_metrics": ["alpha beta gamma delta", "epsilon zeta eta theta"]},
        key_questions=[],
    )
    r = B.benchmark("SYN", proto, "# Report\n\nunrelated content only.\n")
    assert r.suggested_grade == "unusable"


def test_zero_citations_is_weak_not_unusable():
    proto = _protocol(citations=[])
    # Substantive body, slots satisfied, but no citations -> weak (not unusable).
    r = B.benchmark("SYN", proto, _BODY)
    assert r.suggested_grade == "weak"
    assert any("0 citations" in n for n in r.notes)


# ---------------------------------------------------------------------------
# 6. Go/no-go rollup — PENDING / PASS / FAIL, never auto-PASS.
# ---------------------------------------------------------------------------


def _report(
    mission_id: str,
    *,
    grade=None,
    dead=0,
    dead_verified=None,
    gap=1,
    mismatch=0,
) -> B.BenchmarkReport:
    """A directly-constructed report with CONSISTENT citations_dead + dead sample
    + (clean-by-default) telemetry, so go/no-go tests exercise the real logic
    without protocol/telemetry inconsistencies leaking in.
    """

    dead_sample = []
    for i in range(dead):
        v = dead_verified[i] if (dead_verified is not None and i < len(dead_verified)) else None
        dead_sample.append({"url": f"https://d{mission_id}/{i}", "verified_alive": v})
    return B.BenchmarkReport(
        mission_id=mission_id,
        report_word_count=200,
        critique_gap_count=gap,
        critique_status="ok",
        legacy_critique_gap_count=None,
        declared_unavailable_count=0,
        recovery={
            "passes": 0,
            "outcome": "not_triggered",
            "selected_artifact": "first_pass",
        },
        citations_total=1 + dead,
        citations_live=1,
        citations_dead=dead,
        citations_unknown=0,
        dead_url_sample=dead_sample,
        telemetry_mismatches=[B.TelemetryMismatch("x", 1, 2)] * mismatch,
        contract_slots=[],
        objective_subchecks={},
        suggested_grade="usable",
        reviewer_grade=grade,
    )


def test_go_no_go_pending_until_human_grade_filled():
    # 0 dead => t3 vacuously complete; grades unfilled => t1 incomplete => PENDING.
    reports = [_report(f"m{i}", grade=None) for i in range(5)]
    rollup = B.evaluate_go_no_go(reports)
    assert rollup["verdict"] == "PENDING"
    assert rollup["thresholds"]["t2_critique_damage"]["complete"] is True
    assert rollup["thresholds"]["t1_usable_or_better"]["complete"] is False


def test_go_no_go_pass_when_all_four_met_zero_dead():
    reports = [_report(f"m{i}", grade="usable") for i in range(5)]  # 0 dead => t3 vacuous pass
    rollup = B.evaluate_go_no_go(reports)
    assert rollup["thresholds"]["t1_usable_or_better"]["pass"] is True
    assert rollup["thresholds"]["t2_critique_damage"]["pass"] is True
    assert rollup["thresholds"]["t3_false_dead"]["pass"] is True
    assert rollup["thresholds"]["t4_telemetry_truth"]["pass"] is True
    assert rollup["verdict"] == "PASS"


def test_go_no_go_fail_when_t1_missed():
    reports = [_report(f"m{i}", grade=("usable" if i < 3 else "weak")) for i in range(5)]  # 3/5
    rollup = B.evaluate_go_no_go(reports)
    assert rollup["thresholds"]["t1_usable_or_better"]["pass"] is False
    assert rollup["verdict"] == "FAIL"


def test_go_no_go_false_dead_rate_fails_threshold_3():
    # one report with 2 dead, both verified ALIVE (100% false-dead) => t3 FAIL.
    reports = [_report("m0", grade="usable", dead=2, dead_verified=[True, True])]
    reports += [_report(f"m{i}", grade="usable") for i in range(1, 5)]
    rollup = B.evaluate_go_no_go(reports)
    assert rollup["thresholds"]["t3_false_dead"]["complete"] is True
    assert rollup["thresholds"]["t3_false_dead"]["pass"] is False
    assert rollup["verdict"] == "FAIL"


def test_go_no_go_gap_marker_threshold_fails():
    reports = [_report(f"m{i}", grade="usable", gap=(50 if i == 0 else 1)) for i in range(5)]
    rollup = B.evaluate_go_no_go(reports)
    assert rollup["thresholds"]["t2_critique_damage"]["pass"] is False
    assert rollup["verdict"] == "FAIL"


# --- Adversarial regression pins (s85-m02 skeptic round) ---


def test_go_no_go_truthy_nonbool_verified_alive_never_auto_passes():
    """CRITICAL: a reviewer's 'alive' answer as a truthy non-bool (1/'yes'/'true')
    must NOT be read as a confirmed dead — it stays UNVERIFIED (PENDING/warn),
    never a false PASS on a threshold-3-failing build.
    """

    for badval in (1, "yes", "true"):
        reports = [_report("m0", grade="usable", dead=5, dead_verified=[badval] * 5)]
        reports += [_report(f"m{i}", grade="usable") for i in range(1, 5)]
        rollup = B.evaluate_go_no_go(reports)
        assert rollup["verdict"] != "PASS", badval
        assert rollup["thresholds"]["t3_false_dead"]["complete"] is False
        assert any("non-bool" in w for w in rollup["warnings"])


def test_go_no_go_failed_automated_threshold_short_circuits_to_fail():
    """An already-complete-and-missed automated gate (t2/t4) => FAIL, even with
    reviewer grades unfilled — never masked as PENDING (V2)."""

    reports = [_report(f"m{i}", grade=None, gap=(50 if i == 0 else 1)) for i in range(5)]
    rollup = B.evaluate_go_no_go(reports)
    assert rollup["verdict"] == "FAIL"


def test_go_no_go_duplicate_mission_id_does_not_inflate_usable():
    reps = [_report("R1", grade="usable"), _report("R2", grade="usable"),
            _report("R3", grade="usable"), _report("R4", grade="unusable")]
    assert B.evaluate_go_no_go(reps)["thresholds"]["t1_usable_or_better"]["usable_or_better"] == 3
    dup = B.evaluate_go_no_go(reps + [reps[0]])  # duplicate R1
    assert dup["thresholds"]["t1_usable_or_better"]["usable_or_better"] == 3
    assert any("duplicate" in w for w in dup["warnings"])


def test_go_no_go_threshold1_is_fractional_not_absolute_four():
    reports = [_report(f"m{i}", grade=("usable" if i < 4 else "unusable")) for i in range(8)]  # 4/8=50%
    rollup = B.evaluate_go_no_go(reports)
    assert rollup["thresholds"]["t1_usable_or_better"]["pass"] is False


# ---------------------------------------------------------------------------
# 7. CAT baseline fixture.
# ---------------------------------------------------------------------------


def test_load_cat_baseline_has_five_runs_and_expected_counts():
    baseline = B.load_cat_baseline()
    assert baseline["mission_count"] == 5
    assert baseline["baseline_usable_or_better"] == 1
    by_id = {r["cat_id"]: r for r in baseline["runs"]}
    assert by_id["CAT-R001"]["baseline_gap_markers"] == 54
    assert by_id["CAT-R002"]["baseline_gap_markers"] == 65
    assert by_id["CAT-R003"]["baseline_gap_markers"] is None  # wrong-mission report
    assert by_id["CAT-R004"]["baseline_gap_markers"] == 23
    assert by_id["CAT-R005"]["baseline_gap_markers"] == 58
    grades = {r["cat_id"]: r["baseline_grade"] for r in baseline["runs"]}
    assert grades == {
        "CAT-R001": "weak", "CAT-R002": "unusable", "CAT-R003": "unusable",
        "CAT-R004": "usable", "CAT-R005": "weak",
    }


def test_compare_to_baseline_computes_gap_delta():
    r = B.benchmark("CAT-R001", _protocol(citations=[{"url": "https://a", "live": True}]), _BODY)
    rows = B.compare_to_baseline([r])
    row = rows[0]
    assert row["baseline_gap_markers"] == 54
    assert row["baseline_grade"] == "weak"
    assert row["declared_unavailable_delta"] == r.declared_unavailable_count - 54


# ---------------------------------------------------------------------------
# 8. Path resolution + real cached protocol integration.
# ---------------------------------------------------------------------------


def test_resolve_markdown_path_convention():
    p = Path("cmos/research/FOO_protocol.json")
    assert B.resolve_markdown_path(p).name == "FOO.md"




# ---------------------------------------------------------------------------
# 9. PD10 — the eval harness must NOT import production-runtime modules.
# ---------------------------------------------------------------------------


def test_benchmark_module_has_no_production_runtime_imports():
    forbidden_prefixes = (
        "deepsearch.agent.codeagent_finalize_node",
        "deepsearch.agent.codeagent_runner",
        "deepsearch.agent.critique",
        "deepsearch.agent.citation_liveness",
        "deepsearch.agent.contract_to_prompt",
        "deepsearch.agent.tools",
        "deepsearch.worker.writer",
    )
    tree = ast.parse(Path(B.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    for mod in imported:
        assert not mod.startswith(forbidden_prefixes), f"PD10 violation: {mod}"


# ---------------------------------------------------------------------------
# 10. Adversarial regression pins — extractors + grade (s85-m02 skeptic round).
# ---------------------------------------------------------------------------


def test_gap_count_is_fence_aware():
    """A marker-format example inside a code fence must NOT be counted (over-count
    would false-FAIL threshold 2 and retire a good build)."""

    p = "[Data unavailable in primary sources]"
    body = "intro\n" + "\n".join(f"Fact {i}. {p}" for i in range(8))
    body += "\n\n```\nExample => " + p + "\nHeader => " + p + "\n```\n"
    assert B.count_gap_markers(body) == 8  # not 10


def test_gap_count_tolerates_wrapped_and_nbsp_whitespace():
    """A wrapped / NBSP / double-spaced marker must still count once (under-count
    would hide real gaps and PASS a bad build)."""

    assert B.count_gap_markers("x [Data unavailable in primary\nsources]") == 1
    assert B.count_gap_markers("x [Data unavailable in primary  sources]") == 1
    assert B.count_gap_markers("x [Data unavailable in primary sources]") == 1


def test_gap_count_ignores_inline_code_mention():
    assert B.count_gap_markers("The sentinel is `Data unavailable in primary sources`.") == 0


def test_telemetry_diff_truthful_references_zero_with_only_fenced_url_not_flagged():
    proto = {"citations": [], "report_metadata": {"references": 0}}
    body = "# Setup (no primary sources cited)\n```bash\ncurl https://api.example/ping\n```\n"
    assert B.telemetry_diff(proto, body) == []


def test_telemetry_diff_does_not_flag_nonzero_references_difference():
    """A non-zero references that merely differs from the total distinct-citation
    count is legitimate (references = distinct LIVE URLs; dead+orphan survive in
    protocol.citations post-S84). Only references==0 is the lie. Verified on real
    m03 data (CAT-R001: references=49 == 49 live, total distinct 51)."""

    proto = {
        "citations": [{"url": f"https://s{i}.example", "live": i < 3} for i in range(41)],
        "report_metadata": {"references": 3},  # 3 live, 41 total — legitimate
    }
    assert not any(m.field == "references" for m in B.telemetry_diff(proto, "body"))
    # references==0 with citations present IS still flagged.
    proto0 = {"citations": [{"url": "https://x", "live": True}], "report_metadata": {"references": 0}}
    assert any(m.field == "references" for m in B.telemetry_diff(proto0, "body"))


def test_telemetry_diff_none_body_does_not_raise():
    assert B.telemetry_diff({"report_metadata": {"sources_collected": 0}}, None) == []


@pytest.mark.parametrize("bad_protocol", ["notadict", [], None, 5, {"citations": "x"}])
def test_extractors_degrade_on_malformed_protocol(bad_protocol):
    """V5: no extractor may raise on a malformed/old-shape protocol."""

    assert B.citation_liveness(bad_protocol)["total"] == 0
    assert B.telemetry_diff(bad_protocol, "# body") == []
    assert B.contract_slots(bad_protocol, "# body") == []
    r = B.benchmark("M", bad_protocol, "# body " * 200)
    assert r.mission_id == "M"


def test_contract_slots_scalar_fields_do_not_crash():
    assert B.contract_slots({"research_statement": {"success_metrics": 5}}, "b") == []
    assert B.contract_slots({"key_questions": 7}, "b") == []


def test_benchmark_none_body_does_not_raise():
    r = B.benchmark("M", {"citations": []}, None)
    assert r.report_word_count == 0
    assert r.suggested_grade == "unusable"  # empty body is hollow


def test_grade_paraphrased_good_report_not_unusable():
    """A substantive report that paraphrases its success metrics must not be
    suggested `unusable` (false-LOW would anchor the reviewer toward retire)."""

    proto = {
        "citations": [{"url": "https://a", "live": True}],
        "research_statement": {"success_metrics": ["Quantify comparative throughput across platforms"]},
    }
    body = "We measured speed and bandwidth for each option and documented provenance. " * 30
    assert B.benchmark("GOOD", proto, body).suggested_grade != "unusable"


def test_grade_contractless_report_not_strong():
    """A hollow / contract-less report must not be suggested `strong` (false-HIGH
    would anchor toward shipping a wrong-mission build)."""

    proto = {"citations": [{"url": "https://x", "live": True}]}
    r = B.benchmark("H", proto, "one liner. see https://x")
    assert r.suggested_grade != "strong"
    assert any("contract surface" in n for n in r.notes)
