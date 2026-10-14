"""S84 m04 invariants — telemetry truth pass (decision #823).

WHY THIS FILE EXISTS: the persisted-field audit found the result protocol
lying on every production mission: the summary read "N citations from
0 sources" (the telemetry key had zero writers), a fabricated answered
question shipped with confidence=0.3, the synthesis boilerplate claimed
"Coverage score: 0%" and recommended more research regardless of outcome,
contradictions_resolved was a hardcoded pass for a validator that does not
exist, total_tokens persisted a constant 0, degraded_mode/degradation_policy/
loop_count were dead constants, and tool logging recorded inputs only — a
full Linkup outage persisted identically to a thin topic.

THE AUDIT TABLE (deliverable 1) — every persisted field the mission acted on:

  field                                     | verdict  | action
  ------------------------------------------|----------|-------------------
  protocol.summary (sources component)      | REPAIRED | finalize writes
      synthesis_telemetry.sources_collected (key had zero writers)
  protocol.status                           | REPAIRED | VALIDATION_FAILED
      maps to 'review'
  protocol.key_questions status/answer/conf | REPAIRED | no fabricated
      answers; open is honest
  protocol.synthesis.key_insights           | REPAIRED | live gate output,
      no dead-reflection coverage claim
  protocol.synthesis.recommendations        | REPAIRED | live
      deliverable_coverage / delivery_quality
  protocol.synthesis.next_steps             | REPAIRED | live missing
      deliverables only
  quality_checkpoints[contradictions]       | REPAIRED | 'skip' (not
      evaluated); all_gates_pass treats skip as non-failure
  execution_metadata.total_tokens           | REPAIRED | live synthesis +
      critique sums
  execution_metadata.degraded_mode          | DELETED  | guard below
  execution_metadata.degradation_policy    | DELETED  | guard below
  final_outcome.loop_count                  | DELETED  | guard below
  synthesis_telemetry.model_id              | REPAIRED | threaded from the
      bound model
  synthesis_telemetry.framework             | REPAIRED | derived from the
      installed package
  synthesis_telemetry.tool_outcomes         | NEW      | per-call outcomes
  execution_metadata.quality_record         | NEW      | PD137 persisted
  (evidence EV-001 fallback, structural_validation_failures always-[],
   route_summary/finding_counts/source_counts omit-when-empty, depth_config
   loop fields: documented residuals — absent-not-lying or deferred; see
   the s84-m04 CMOS completion record.)
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import patch

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import CodeAgentToolBundle, SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.outcome import build_final_outcome_payload
from deepsearch.worker.writer import build_execution_metadata

_REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Tool-outcome telemetry: outage vs thin topic (success criterion 3).
# ---------------------------------------------------------------------------


def _search_record(status: str, *, category: str | None, count: int) -> dict:
    return {
        "tool": "web_search",
        "query": "q",
        "max_results": 20,
        "status": status,
        "error_category": category,
        "sentinel": status == "error",
        "circuit_open": category == "circuit_open",
        "result_count": count,
    }


def test_linkup_outage_distinguishable_from_thin_topic_in_persisted_telemetry() -> None:
    """THE criterion-3 pin: the persisted rollup alone separates an outage
    (sentinel errors, circuit-open) from a thin topic (successful searches,
    few results). Pre-m04 both persisted as identical input-only counts."""

    outage = CodeAgentToolBundle(
        tools=[],
        call_log=[
            _search_record("error", category="rate_limit", count=0),
            _search_record("error", category="rate_limit", count=0),
            _search_record("error", category="circuit_open", count=0),
        ],
    ).tool_outcomes()

    thin = CodeAgentToolBundle(
        tools=[],
        call_log=[
            _search_record("ok", category=None, count=2),
            _search_record("ok", category=None, count=0),
            _search_record("ok", category=None, count=1),
        ],
    ).tool_outcomes()

    out_ws = outage["by_tool"]["web_search"]
    thin_ws = thin["by_tool"]["web_search"]

    # Outage signature: errors with categories, the circuit-open smoking gun.
    assert out_ws["error"] == 3 and out_ws["ok"] == 0
    assert out_ws["by_category"] == {"rate_limit": 2, "circuit_open": 1}
    assert out_ws["circuit_open"] == 1
    assert out_ws["results_total"] == 0

    # Thin-topic signature: successful calls, zero errors, few results.
    assert thin_ws["ok"] == 3 and thin_ws["error"] == 0
    assert thin_ws["by_category"] == {}
    assert thin_ws["results_total"] == 3

    # And the two are structurally distinguishable without any other signal.
    assert out_ws != thin_ws


def test_tool_outcomes_records_are_capped_loudly() -> None:
    """No silent caps: the per-call record list truncates at max_records and
    says so in records_truncated."""

    bundle = CodeAgentToolBundle(
        tools=[],
        call_log=[_search_record("ok", category=None, count=1) for _ in range(300)],
    )
    outcomes = bundle.tool_outcomes(max_records=250)
    assert len(outcomes["records"]) == 250
    assert outcomes["records_truncated"] == 50
    # The rollup still counts EVERY call, not just the recorded ones.
    assert outcomes["by_tool"]["web_search"]["calls"] == 300


def test_wrapper_records_outcomes_for_sentinel_and_ok(monkeypatch) -> None:
    """The web_search wrapper logs AFTER the impl with the outcome fields —
    driven through the real build_codeagent_tools wrapper."""

    from tests.unit.test_codeagent_runner import _install_fake_smolagents

    _install_fake_smolagents(monkeypatch)
    import deepsearch.agent.tools.web_search as ws_mod
    from deepsearch.agent import codeagent_runner as runner

    responses = [
        [{"error": "rate_limit", "message": "429", "title": "", "url": "", "snippet": ""}],
        [{"title": "T", "url": "https://x.example/a", "snippet": "s"}],
    ]
    calls = {"n": 0}

    def fake_web_search(query, max_results=20):
        out = responses[min(calls["n"], 1)]
        calls["n"] += 1
        return out

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)

    bundle = runner.build_codeagent_tools()
    web_search_tool = bundle.tools[0]
    web_search_tool("outage query")
    web_search_tool("ok query")

    first, second = bundle.call_log[0], bundle.call_log[1]
    assert first["status"] == "error"
    assert first["error_category"] == "rate_limit"
    assert first["sentinel"] is True
    assert first["result_count"] == 0
    assert second["status"] == "ok"
    assert second["error_category"] is None
    assert second["result_count"] == 1
    # The sentinel entry must NOT pollute sources_collected.
    assert all("x.example" in (s.get("url") or "") for s in bundle.sources_collected)


# ---------------------------------------------------------------------------
# 2. Ground-truth end-to-end (success criterion 1): every persisted field
# matches ground truth or is absent.
# ---------------------------------------------------------------------------


def _state() -> dict:
    return {
        "mission_id": "S84-M04-E2E",
        "mission_title": "Truth Mission",
        "mission_objectives": ["Objective one"],
        "mission_context": {
            "title": "Truth Mission",
            "objective": "verify telemetry truth",
            # This telemetry fixture is intentionally coverage-neutral. A
            # missing matrix would now trigger the separate S87 recovery actor
            # and double the synthesis telemetry this test is isolating.
            "deliverables": [],
        },
        "sources_found": [],
        "telemetry": {},
    }


def test_stubbed_end_to_end_persisted_fields_match_ground_truth(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    captured: dict = {}

    def _capture_protocol(protocol, **kwargs):
        captured["protocol"] = protocol
        return str(tmp_path / "protocol.json")

    fixture_outcomes = {
        "by_tool": {"web_search": {"calls": 3, "ok": 3, "error": 0,
                                    "results_total": 7, "circuit_open": 0,
                                    "by_category": {}}},
        "records": [],
        "records_truncated": 0,
        "ledger_records": [],
        "ledger_records_truncated": 0,
    }
    synthesis = SynthesisResult(
        output_markdown=(
            "# Report\n\nClaim [cite:https://live.example/c] with detail.\n\n"
            "## References\n\n[1] C — https://live.example/c\n"
        ),
        step_count=7,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=5.0,
        token_usage={"input": 900, "output": 600, "total": 1500},
        tool_call_summary={"total_calls": 3, "by_tool": {"web_search": 3},
                           "scratchpad_keys": []},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=[
            {"url": "https://live.example/c", "title": "C", "snippet": "z"},
            {"url": "https://live.example/d", "title": "D", "snippet": "y"},
        ],
        error=None,
        tool_outcomes=fixture_outcomes,
        model_id="deepseek-v4-pro-test",
    )
    critique = CritiqueResult(
        output_markdown="",
        claim_count=0,
        gap_notes_inserted=0,
        runtime_seconds=0.0,
        token_usage={"input": 100, "output": 50, "total": 150},
        error=None,
    )

    async def _fake_liveness(citations, sources_collected, budget_seconds=45.0, scope_admission=None):
        for ann in citations:
            ann.live = True
        return list(citations), LivenessTelemetry(
            checked=len(citations), live=len(citations),
            distinct_urls_checked=len(citations),
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(fnode, "run_critique", return_value=critique), patch.object(
            fnode, "check_citations_liveness", new=_fake_liveness
        ), patch.object(
            fnode, "save_protocol", side_effect=_capture_protocol
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())
    protocol = captured["protocol"]
    s_tel = result["synthesis_telemetry"]

    # Sources truth: the summary names the REAL collected-source count.
    assert s_tel["sources_collected"] == 2
    assert "2 sources collected" in protocol.summary
    assert "0 sources" not in protocol.summary
    assert "citations from" not in protocol.summary  # no false provenance claim

    # Model/framework truth: threaded, not hardcoded.
    assert s_tel["model_id"] == "deepseek-v4-pro-test"
    assert s_tel["framework"].startswith("smolagents")

    # Tool outcomes ride the declared channel verbatim.
    assert s_tel["tool_outcomes"] == fixture_outcomes

    # Questions truth: no findings exist -> honestly open, no placeholder.
    assert all(q.status == "open" for q in protocol.key_questions)
    assert all(q.answer is None for q in protocol.key_questions)
    assert all(q.confidence is None for q in protocol.key_questions)

    # Synthesis truth: live gate output, never the dead-reflection lie.
    joined = " ".join(protocol.synthesis.key_insights)
    assert "Coverage score:" not in joined
    assert "citations" in joined
    recs = " ".join(protocol.synthesis.recommendations)
    assert "Consider additional research to improve coverage" not in recs

    # Checkpoints truth: contradictions is skip; nothing failed.
    statuses = {cp.gate: cp.status for cp in protocol.quality_checkpoints}
    assert statuses["contradictions_resolved"] == "skip"
    assert "fail" not in {
        s for g, s in statuses.items() if g != "contradictions_resolved"
    }
    assert protocol.all_gates_pass() is True

    # Status truth: gate resolved COMPLETE -> 'complete'.
    assert protocol.status == "complete"


def test_execution_metadata_truth_repairs() -> None:
    """total_tokens sums the LIVE per-stage counters; quality_record is
    persisted; the deleted degradation keys are gone; final_outcome carries
    no loop_count."""

    state = {
        "telemetry": {
            "quality_record": {"word_count": 3400, "delivery_quality": "clean"},
        },
        "synthesis_telemetry": {
            "token_usage": {"input": 900, "output": 600, "total": 1500},
            "critique_telemetry": {"token_usage": {"total": 150}},
        },
    }
    metadata = build_execution_metadata(state, 42.0)
    payload = metadata.to_dict()

    assert payload["total_tokens"] == 1650  # 1500 synthesis + 150 critique
    assert payload["quality_record"] == {
        "word_count": 3400,
        "delivery_quality": "clean",
    }
    assert "degraded_mode" not in payload
    assert "degradation_policy" not in payload

    # A legacy fixture that explicitly populates the old chain still wins.
    legacy = build_execution_metadata(
        {"telemetry": {"token_usage": {"total_tokens": 9000}}}, 1.0
    )
    assert legacy.total_tokens == 9000


def test_final_outcome_payload_has_no_loop_count() -> None:
    from datetime import datetime, timezone

    payload = build_final_outcome_payload(
        outcome="complete",
        routing_reason="converged",
        coverage_gate_failures=[],
        quality_gate_failures=[],
        completed_at=datetime.now(timezone.utc),
    )
    assert "loop_count" not in payload
    assert payload["outcome"] == "complete"


def test_protocol_status_maps_failed_and_blocked_to_review(tmp_path, monkeypatch) -> None:
    """ADVERSARIAL-REVIEW FIX PIN: blocked missions reach this builder via
    the worker's protocol-regeneration fallback — they must not claim
    status='complete' either."""

    from deepsearch.report.protocol import generate_mission_protocol

    state = dict(_state())
    for outcome in ("validation_failed", "blocked"):
        state["mission_outcome"] = outcome
        assert generate_mission_protocol(state).status == "review"

    state["mission_outcome"] = "complete"
    assert generate_mission_protocol(state).status == "complete"
    state.pop("mission_outcome")
    assert generate_mission_protocol(state).status == "complete"  # legacy


def test_model_id_threads_from_the_bound_model(monkeypatch) -> None:
    """ADVERSARIAL-REVIEW FIX PIN: the production call site passes
    model=None; the runner must read model_id off the BOUND model it builds
    internally — reading the raw parameter left the field permanently empty
    and the telemetry fell back to the hardcoded literal."""

    import inspect

    from deepsearch.agent import codeagent_runner as runner

    src = inspect.getsource(runner.run_codeagent_synthesis)
    assert 'getattr(bound_model, "model_id"' in src
    assert 'getattr(model, "model_id"' not in src


def test_framework_label_derives_from_installed_version(monkeypatch) -> None:
    """ADVERSARIAL-REVIEW FIX PIN: prove derivation, not just shape — a
    revert to the hardcoded '1.24.0' literal must fail this test."""

    real_label = fnode._framework_label()
    assert real_label.startswith("smolagents") and real_label.endswith("CodeAgent")

    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda name: "9.9.9")
    assert fnode._framework_label() == "smolagents 9.9.9 CodeAgent"


def test_uploader_local_validation_tolerates_skip_gate() -> None:
    """ADVERSARIAL-REVIEW FIX PIN: the uploader's local validator agrees
    with all_gates_pass — 'skip' (not evaluated) is not a failing gate; only
    a hard 'fail' fails validation."""

    import inspect

    from deepsearch.tracelab import uploader

    src = inspect.getsource(uploader)
    assert 'cp.status == "fail"' in src
    assert 'cp.status != "pass"' not in src


# ---------------------------------------------------------------------------
# 3. Hard Rule 10 — AST consumer-scan guards for the deleted fields.
# ---------------------------------------------------------------------------

_DELETED_METADATA_FIELDS = {"degraded_mode", "degradation_policy"}
# Whitelist (S75 carve-out pattern — STATE-key readers, not consumers of the
# deleted ExecutionMetadata persistence fields; each is rationale-bearing):
#   state.py           — keeps the AgentState channel defaults (channel
#                        stability; no longer persisted).
#   worker/main.py:603 — logs a degraded warning off the state key; the
#                        default is False and nothing sets it (never fires).
#   report/generator.py:323 — legacy degraded-notice report block keyed on
#                        the state key (never fires post-S64).
#   runner.py:325      — local-runner log line, same never-fires state read.
# Removing the state channel + these readers is S86 worker-hardening/cleanup
# scope; the m04 deletion is the PERSISTED fields.
_DEGRADATION_WHITELIST = {
    "deepsearch/agent/state.py",
    "deepsearch/worker/main.py",
    "deepsearch/report/generator.py",
    "deepsearch/runner.py",
}


def _scan_for_reads(root: Path, names: set[str], whitelist: set[str]) -> list[str]:
    offenders: list[str] = []
    for path in sorted((root / "deepsearch").rglob("*.py")):
        rel = str(path.relative_to(root))
        if rel in whitelist:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in names:
                offenders.append(f"{rel}:{node.lineno} .{node.attr}")
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in names
                and isinstance(
                    getattr(node, "parent", None), (ast.Subscript,)
                )
            ):  # pragma: no cover — parent links absent in stdlib ast
                offenders.append(f"{rel}:{node.lineno} ['{node.value}']")
    return offenders


def test_hard_rule_10_no_degradation_field_reads() -> None:
    """Deleted ExecutionMetadata fields must not be read anywhere outside
    the whitelisted state defaults (Hard Rule 10 consumer scan)."""

    offenders = _scan_for_reads(_REPO, _DELETED_METADATA_FIELDS, _DEGRADATION_WHITELIST)
    # String-key reads: a plain source scan catches subscripts + get() calls.
    for path in sorted((_REPO / "deepsearch").rglob("*.py")):
        rel = str(path.relative_to(_REPO))
        if rel in _DEGRADATION_WHITELIST:
            continue
        text = path.read_text(encoding="utf-8")
        for name in _DELETED_METADATA_FIELDS:
            if f'"{name}"' in text or f"'{name}'" in text:
                offenders.append(f"{rel} literal {name}")
    assert offenders == [], f"deleted-field consumers remain: {offenders}"


def test_hard_rule_10_no_loop_count_in_outcome_or_worker() -> None:
    """final_outcome.loop_count was deleted: the outcome payload builder and
    the worker must not reference the key (evaluation.py's internal
    loop_count dataclass is a different, live surface and stays)."""

    for rel in ("deepsearch/agent/outcome.py", "deepsearch/worker"):
        target = _REPO / rel
        paths = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert '"loop_count"' not in text and "'loop_count'" not in text, (
                f"loop_count key reference remains in {path}"
            )


def test_state_degradation_defaults_still_initialised() -> None:
    """The whitelisted AgentState defaults remain (channel stability) —
    they are simply no longer persisted."""

    from deepsearch.agent.state import empty_state

    state = empty_state(mission_id="X")
    assert state.get("degraded_mode") is False
