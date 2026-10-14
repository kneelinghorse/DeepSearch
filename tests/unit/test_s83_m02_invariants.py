"""S83 m02 RD2 invariants — close the S42.2 dropped-channel bug *class*.

Two NEW local-only guards that close the class the post-deploy cross-store
re-dispatch (62eab1b0) twice surfaced: LangGraph 1.x ``StateGraph`` registers
channels from the ``AgentState`` TypedDict and **silently drops any node-return
key that is not a declared channel** on the node-return merge. S82 declared the
three gate keys by name (``delivery_quality`` / ``quality_gate_failures`` /
``coverage_gate_failures``), but the same shape was still open for two residual
leaks (``synthesis_used`` / ``generation_errors``), and — more importantly — no
test compiled the *real* graph, so the unit suite verified declaration set-
membership and a hand-built dict for ``resolve_final_outcome_payload``, both
bypassing the channel-merge that is the actual bug locus.

This file installs:

(a) **A generalized AST declare-channel guard** — scans every node-return
    ``StateUpdate`` accumulator in ``codeagent_finalize_node.py`` and asserts
    every key written onto it (``result["k"]=...`` Subscript-Store,
    ``result.setdefault("k", ...)``, ``result["k"] += ...`` AugAssign, and the
    initialiser dict literal) is a SUBSET of ``AgentState.__annotations__``.
    Reports the offending ``file:line:key`` on failure. Closes the whole class
    — it does NOT pin the current keys by name. A ``setdefault``-blind scan
    would miss ``generation_errors``; this guard collects that form explicitly.

    Falsifiability (do NOT commit either edit):
      * add ``result["bogus_undeclared_key"] = 1`` to ``codeagent_finalize_node``
        → guard fails naming ``codeagent_finalize_node.py:<line>:bogus_undeclared_key``.
      * remove ``synthesis_used`` (or ``generation_errors``) from ``AgentState``
        → guard fails naming ``codeagent_finalize_node.py:420:synthesis_used``
        (AC9).

(b) **A real-graph round-trip test** — compiles the actual ``StateGraph`` via
    ``deepsearch.agent.graph.build_workflow()`` and drives the compiled app's
    REAL ``.ainvoke`` (NO mock of ``build_workflow`` / ``StateGraph`` /
    ``.ainvoke`` — contrast the ``AsyncMock(return_value=state)`` pattern at
    ``test_worker_main.py:441-442``). A reviewable max-step fixture drives
    ``evaluate_outcome_gate`` to a non-clean verdict; the test asserts the gate
    channels SURVIVE the merge on the returned ``final_state``, then feeds it to
    ``resolve_final_outcome_payload`` and asserts the payload carries them.

    Falsifiability: transiently comment one of the ``state.py:136-138`` gate
    channel declarations → ``final_state`` loses that key → this test FAILS.

Tests make ZERO network/LLM calls (the reviewable markdown carries no
``[cite:`` markers, so the liveness gate is skipped on the empty annotation
set) and redirect ``DEEPSEARCH_REPORT_DIR`` to ``tmp_path``.
"""

from __future__ import annotations

import ast
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.graph import build_workflow
from deepsearch.agent.outcome import (
    DELIVERY_QUALITY_DEGRADED,
    MissionOutcome,
    resolve_final_outcome_payload,
)
from deepsearch.agent.state import empty_state

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
FINALIZE_PATH = DEEPSEARCH_ROOT / "agent" / "codeagent_finalize_node.py"
FINALIZE_REL = FINALIZE_PATH.name


# ===========================================================================
# (a) Generalized AST declare-channel guard.
# ===========================================================================


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _annotation_is_state_update(annotation: ast.AST | None) -> bool:
    return isinstance(annotation, ast.Name) and annotation.id == "StateUpdate"


def _state_update_var_names(func: ast.AST) -> set[str]:
    """Local-variable names in ``func`` that hold the returned node update.

    A name qualifies when the function (a) returns it as a bare ``Name``
    (``return result``), (b) annotates it ``StateUpdate``
    (``result: StateUpdate = {...}``), or (c) takes it as a parameter
    annotated ``StateUpdate`` (``_block_result(base: StateUpdate)``). This is
    how the guard generalises across the module rather than hard-coding
    ``result``: ``codeagent_finalize_node`` → ``{result}``, ``_block_result``
    → ``{base}`` (whose keys are all already declared — harmless, but scanned
    so a future leak there is caught too).
    """

    names: set[str] = set()
    args = func.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if _annotation_is_state_update(arg.annotation):
            names.add(arg.arg)
    for node in ast.walk(func):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _annotation_is_state_update(node.annotation)
        ):
            names.add(node.target.id)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
            names.add(node.value.id)
    return names


def _written_keys(func: ast.AST, var_names: set[str]) -> list[tuple[str, int]]:
    """Collect ``(key, lineno)`` for every key written onto a ``var_names`` accumulator.

    Covers FOUR write forms so the scan is not Subscript-Assign-only (which
    would miss the ``setdefault`` form that introduces ``generation_errors``):

      * ``result["k"] = ...``          — Subscript-Store assignment
      * ``result["k"] += ...``         — AugAssign on a Subscript
      * ``result.setdefault("k", ...)``— dict.setdefault key introduction
      * ``result = {"k": ...}``        — dict-literal initialiser
    """

    found: list[tuple[str, int]] = []

    def _record(key_node: ast.AST, lineno: int) -> None:
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            found.append((key_node.value, lineno))

    def _is_acc_subscript(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in var_names
        )

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if _is_acc_subscript(tgt) and isinstance(tgt.ctx, ast.Store):
                    _record(tgt.slice, tgt.lineno)
            if isinstance(node.value, ast.Dict):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in var_names:
                        for key in node.value.keys:
                            if key is not None:
                                _record(key, node.lineno)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id in var_names
                and isinstance(node.value, ast.Dict)
            ):
                for key in node.value.keys:
                    if key is not None:
                        _record(key, node.lineno)
        elif isinstance(node, ast.AugAssign):
            if _is_acc_subscript(node.target):
                _record(node.target.slice, node.target.lineno)
        elif isinstance(node, ast.Call):
            fn = node.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "setdefault"
                and isinstance(fn.value, ast.Name)
                and fn.value.id in var_names
                and node.args
            ):
                _record(node.args[0], node.lineno)

    return found


def test_finalize_node_return_keys_are_declared_channels() -> None:
    """Every key written onto a node-return ``StateUpdate`` MUST be a declared
    ``AgentState`` channel, or LangGraph drops it on the node-return merge.

    Asserts ``written_keys <= set(AgentState.__annotations__)`` (a SUBSET — not
    equality against a literal key set), so the guard closes the whole class
    and never needs editing when a new (declared) channel is added. See the
    module docstring for the two falsifiability edits.
    """

    from deepsearch.agent.state import AgentState

    declared = set(AgentState.__annotations__)
    tree = _module_ast(FINALIZE_PATH)

    scanned_accumulators: list[str] = []
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        var_names = _state_update_var_names(node)
        if not var_names:
            continue
        scanned_accumulators.append(node.name)
        for key, lineno in _written_keys(node, var_names):
            if key not in declared:
                offenders.append(f"{FINALIZE_REL}:{lineno}:{key}")

    assert "codeagent_finalize_node" in scanned_accumulators, (
        "guard found no node-return StateUpdate accumulator in "
        f"{FINALIZE_REL} — the scan heuristic broke; offenders cannot be trusted"
    )
    assert not offenders, (
        "node-return keys NOT declared as AgentState channels (LangGraph will "
        "silently drop them on the merge — the S42.2 class):\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\nDeclare each on AgentState (deepsearch/agent/state.py) and "
        "initialise it in empty_state()."
    )


def test_guard_collects_the_two_residual_leak_forms() -> None:
    """Meta-guard: the scanner itself must collect BOTH write forms that the
    S42.2 class hides — the Subscript-Store (``synthesis_used``) and the
    ``setdefault`` (``generation_errors``). A Subscript-only scanner would pass
    the subset assertion above while still being blind to ``generation_errors``;
    this pins that the scanner sees both, so the guard cannot silently weaken.
    """

    tree = _module_ast(FINALIZE_PATH)
    finalize = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "codeagent_finalize_node"
    )
    keys = {key for key, _ in _written_keys(finalize, _state_update_var_names(finalize))}
    assert "synthesis_used" in keys, "scanner missed the Subscript-Store form"
    assert "generation_errors" in keys, "scanner missed the setdefault form"


# ===========================================================================
# (b) Real-graph round-trip test.
# ===========================================================================


def _reviewable_max_steps_synthesis() -> SynthesisResult:
    """Non-empty, untruncated fallback that reached the max-step boundary.

    Post-S92 evidence grades a reviewable max-step fallback as COMPLETE with a
    non-clean delivery quality when it retains usable citations. The collected
    source carries a substantive body, so liveness uses the deterministic
    in-run body-trust path and the test remains provider-free.
    """

    source_url = "https://example.com/evidence"
    return SynthesisResult(
        output_markdown=(
            "# Report\n\nPartial draft, agent ran out of steps mid-assembly "
            f"but retained its evidence.[cite:{source_url}]"
        ),
        step_count=40,
        self_terminated=False,
        hit_max_steps=True,
        runtime_seconds=42.0,
        token_usage={"input": 1000, "output": 2000, "total": 3000},
        tool_call_summary={"total_calls": 7, "by_tool": {"web_search": 3}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        error=None,
        max_steps_fallback_complete=True,
        sources_collected=[
            {
                "url": source_url,
                "title": "Evidence",
                "snippet": "",
                "body": "Grounding evidence. " * 20,
                "body_source": "source_fetch",
                "content_quality": "readable_body",
                "score": 1.0,
                "authority_tier": "medium",
            }
        ],
    )


def _critique_passthrough() -> CritiqueResult:
    return CritiqueResult(
        output_markdown=(
            "# Report\n\nPartial draft, agent ran out of steps mid-assembly "
            "but retained its evidence."
        ),
        claim_count=1,
        gap_notes_inserted=0,
        runtime_seconds=1.0,
        token_usage={"input": 100, "output": 50, "total": 150},
        error=None,
    )


def test_gate_channels_survive_real_graph_roundtrip(tmp_path, monkeypatch) -> None:
    """The real ``StateGraph`` round-trip: gate channels survive ``ainvoke``.

    This is the test the S42.2 bug needed — it crosses the node-return merge
    boundary that the S82 unit tests bypassed. Compile ``build_workflow()`` and
    call the compiled app's REAL ``.ainvoke`` on a seed from ``empty_state`` with
    synthesis/critique patched to a non-clean verdict; assert the three gate
    channels survive on ``final_state``, then ``resolve_final_outcome_payload``
    carries them. Falsifiable by transiently commenting a ``state.py:136-138``
    declaration (the channel then vanishes from ``final_state``).
    """

    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))

    app = build_workflow()  # real compiled StateGraph — NOT mocked
    seed = empty_state(
        mission_id="S83-M02-ROUNDTRIP",
        mission_objectives=["c1", "c2"],
        mission_context={
            "title": "Round-trip Mission",
            "objective": "verify gate channels survive the graph merge",
            "deliverables": ["Comparison Matrix"],
        },
    )

    async def run() -> dict:
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_reviewable_max_steps_synthesis(),
        ), patch.object(fnode, "run_critique", return_value=_critique_passthrough()):
            return await app.ainvoke(seed)

    final_state = asyncio.run(run())

    # --- AC5: the three gate channels SURVIVE the node-return merge ---
    assert final_state.get("delivery_quality") == DELIVERY_QUALITY_DEGRADED, (
        "delivery_quality dropped on the node-return merge (S42.2 regression)"
    )
    assert isinstance(final_state.get("quality_gate_failures"), list)
    assert final_state["quality_gate_failures"], (
        "quality_gate_failures dropped on the node-return merge (S42.2 regression)"
    )
    assert any("hit_max_steps" in f for f in final_state["quality_gate_failures"])
    assert "coverage_gate_failures" in final_state, (
        "coverage_gate_failures channel dropped on the merge (S42.2 regression)"
    )
    assert isinstance(final_state["coverage_gate_failures"], list)
    # The two residual-leak channels declared by AC0 also survive the merge:
    assert final_state.get("synthesis_used") is True
    assert "generation_errors" in final_state
    # Exact inline artifacts are declared channels too; they must survive the
    # real node-return merge rather than disappearing before the worker seam.
    assert final_state["report_markdown"] == Path(final_state["report_path"]).read_text(
        encoding="utf-8"
    )
    assert final_state["protocol_payload"] == json.loads(
        Path(final_state["protocol_path"]).read_text(encoding="utf-8")
    )

    # --- AC6: the worker seam — final_state -> resolve_final_outcome_payload ---
    payload = resolve_final_outcome_payload(
        final_state, completed_at=datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)
    )
    assert payload["outcome"] == MissionOutcome.COMPLETE.value
    assert payload["delivery_quality"] == DELIVERY_QUALITY_DEGRADED
    assert payload["quality_gate_failures"], "payload quality list must be populated"
    assert "coverage_gate_failures" in payload
