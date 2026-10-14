"""S82 m03 (PD135) RD2 invariants — stranded-draft recovery + forced-finalize.

The cross-store failure class: the agent exhausts max_steps WITHOUT calling
final_answer, its assembled report stranded in the scratchpad, and the runner
ships an empty / marker-less result.output -> 0 citations -> a stub reported as
a success. m03 makes the runner SALVAGE the best [cite:URL]-bearing scratchpad
draft so a substantive cited artifact ships, while m01's gate still resolves
VALIDATION_FAILED on the max-steps-without-final_answer trigger (the recovered
draft is the reviewable artifact attached to that failed outcome).

Scope decision (user-approved 2026-05-28): the forced-finalize is implemented
on the two locally-validatable surfaces — runner-side scratchpad recovery and a
prompt-side recovery-key + finalize-discipline clause. The smolagents-internal
final_answer-template / reserve-N-steps override is DEFERRED to a future mission
that can validate it against a live dispatch (smolagents is not importable in
this environment; the stack is locked).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.codeagent_runner import (
    DEFAULT_API_BASE,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL_ID,
    DEFAULT_TEMPERATURE,
    SCRATCHPAD_RECOVERY_KEY,
    SynthesisResult,
    _has_cite_markers,
    _recover_stranded_draft,
)
from deepsearch.agent.contract_to_prompt import build_codeagent_prompts
from deepsearch.agent.outcome import MissionOutcome, evaluate_outcome_gate

# ---------------------------------------------------------------------------
# Pure recovery helpers.
# ---------------------------------------------------------------------------


def test_has_cite_markers() -> None:
    assert _has_cite_markers("a claim [cite:https://x.org/a].")
    assert not _has_cite_markers("a claim with no marker.")
    assert not _has_cite_markers("")


def test_recover_prefers_report_draft_when_cited() -> None:
    scratchpad = {
        "notes": "long uncited notes " * 50,
        SCRATCHPAD_RECOVERY_KEY: "# Report\n\nClaim [cite:https://x.org/a].",
        "misc": "Short cited [cite:https://y.org/b].",
    }
    assert _recover_stranded_draft(scratchpad) == scratchpad[SCRATCHPAD_RECOVERY_KEY]


def test_recover_falls_back_to_longest_cited_value() -> None:
    scratchpad = {
        "draft_a": "Short [cite:https://x.org/a].",
        "draft_b": "Longer assembled draft with a marker [cite:https://y.org/b]. " * 5,
        "uncited": "no markers here " * 100,  # longest overall but NOT cited
    }
    # report_draft absent -> longest CITED value wins (not the longest overall).
    assert _recover_stranded_draft(scratchpad) == scratchpad["draft_b"]


def test_recover_returns_none_when_no_cited_value() -> None:
    assert _recover_stranded_draft({"a": "no markers", "b": "still none"}) is None
    assert _recover_stranded_draft({}) is None


# ---------------------------------------------------------------------------
# Runner seam — recovery on hit_max_steps without a usable final_answer.
# ---------------------------------------------------------------------------


_CITED_DRAFT = (
    "# Cross-store Report\n\nForward-compat primitives are well evidenced "
    "[cite:https://docs.example.org/migrations]. Recommendation follows "
    "[cite:https://docs.example.org/recommendation]."
)


class ActionStep:
    """type().__name__ == 'ActionStep' — the runner counts steps by class name."""

    def __init__(self, *, is_final: bool) -> None:
        self.is_final_answer = is_final
        self.token_usage = None
        self.timing = None
        self.tool_calls = None


class _Memory:
    def __init__(self, steps):
        self.steps = steps


def _make_fake_agent(*, run_behavior, n_steps: int, last_is_final: bool):
    """Build a fake smolagents CodeAgent class with controlled memory + run()."""

    class _FakeCodeAgent:
        def __init__(
            self,
            *,
            tools,
            model,
            max_steps,
            instructions,
            verbosity_level,
            step_callbacks=None,
        ):
            self.tools = tools
            self.max_steps = max_steps
            steps = [ActionStep(is_final=False) for _ in range(n_steps - 1)]
            steps.append(ActionStep(is_final=last_is_final))
            self.memory = _Memory(steps)

        def run(self, user_directive, return_full_result=True):
            return run_behavior(self.tools)

    return _FakeCodeAgent


def _install_fake_smolagents(monkeypatch, agent_cls) -> None:
    fake = types.ModuleType("smolagents")
    fake.CodeAgent = agent_cls
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f  # passthrough @tool
    monkeypatch.setitem(sys.modules, "smolagents", fake)


def _scratchpad_writer(tools):
    """Return the scratchpad_write tool from the bundle by name."""
    for t in tools:
        if getattr(t, "__name__", "") == "scratchpad_write":
            return t
    raise AssertionError("scratchpad_write tool not found in bundle")


def _run(monkeypatch, *, run_behavior, last_is_final=False, n_steps=DEFAULT_MAX_STEPS):
    monkeypatch.setenv("LINKUP_API_KEY", "test-key")
    agent_cls = _make_fake_agent(
        run_behavior=run_behavior, n_steps=n_steps, last_is_final=last_is_final
    )
    _install_fake_smolagents(monkeypatch, agent_cls)
    return runner.run_codeagent_synthesis(
        system_instructions="sys",
        user_directive="do it",
        model=MagicMock(),  # skip build_default_model
    )


class _MockResult:
    """Minimal stand-in for the smolagents run result (output + token_usage)."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.token_usage = None


def test_recovers_stranded_cited_draft_on_max_steps(monkeypatch) -> None:
    """hit_max_steps + marker-less output + cited scratchpad -> recovery."""

    def behavior(tools):
        _scratchpad_writer(tools)("report_draft", _CITED_DRAFT)
        return _MockResult(output="")  # empty final output -> stranded

    result = _run(monkeypatch, run_behavior=behavior, last_is_final=False)

    assert result.hit_max_steps is True
    assert result.self_terminated is False
    assert result.recovered_from_scratchpad is True
    assert result.output_markdown == _CITED_DRAFT
    assert _has_cite_markers(result.output_markdown)


def test_no_recovery_when_no_cited_scratchpad_then_gate_fails(monkeypatch) -> None:
    """hit_max_steps + no cited scratchpad -> no recovery; m01 gate fails it."""

    def behavior(tools):
        _scratchpad_writer(tools)("notes", "uncited working notes, no markers")
        return _MockResult(output="")

    result = _run(monkeypatch, run_behavior=behavior, last_is_final=False)

    assert result.hit_max_steps is True
    assert result.recovered_from_scratchpad is False
    assert not _has_cite_markers(result.output_markdown)

    # The m01 gate (landed) resolves this max-steps-without-final_answer run to
    # VALIDATION_FAILED regardless of recovery — the safety net m03 relies on.
    telemetry = {
        "hit_max_steps": result.hit_max_steps,
        "self_terminated": result.self_terminated,
        "citation_telemetry": {"citations_extracted": 0},
        "deliverable_coverage": {"shortfall_score": 0.0, "satisfied": [], "missing": []},
        "liveness_telemetry": {"checked": 0, "dead": 0, "orphan": 0},
    }
    gate = evaluate_outcome_gate(telemetry, report_word_count=0)
    assert gate.outcome is MissionOutcome.VALIDATION_FAILED


def test_no_recovery_when_output_already_cited(monkeypatch) -> None:
    """If the agent DID produce a cited output, recovery must not fire."""

    def behavior(tools):
        _scratchpad_writer(tools)("report_draft", _CITED_DRAFT)
        return _MockResult(output="# Final\n\nClaim [cite:https://docs.example.org/x].")

    result = _run(monkeypatch, run_behavior=behavior, last_is_final=False)
    assert result.hit_max_steps is True
    assert result.recovered_from_scratchpad is False
    assert "cite:https://docs.example.org/x" in result.output_markdown


def test_no_recovery_when_not_max_steps(monkeypatch) -> None:
    """A clean self-terminated run (not hit_max_steps) never recovers."""

    def behavior(tools):
        _scratchpad_writer(tools)("report_draft", _CITED_DRAFT)
        return _MockResult(output="")  # empty but self-terminated

    result = _run(monkeypatch, run_behavior=behavior, last_is_final=True, n_steps=5)
    assert result.hit_max_steps is False
    assert result.self_terminated is True
    assert result.recovered_from_scratchpad is False


# ---------------------------------------------------------------------------
# Stack lock — model / api / temperature / max_steps UNCHANGED (m03 may touch
# only loop-stop logic + the recovery convention).
# ---------------------------------------------------------------------------


def test_smolagents_stack_lock_unchanged() -> None:
    assert DEFAULT_MODEL_ID == "deepseek-flash"
    assert DEFAULT_API_BASE == "https://api.deepseek.com/v1"
    assert DEFAULT_TEMPERATURE == 0.0
    assert DEFAULT_MAX_STEPS == 40
    assert SCRATCHPAD_RECOVERY_KEY == "report_draft"


# ---------------------------------------------------------------------------
# Prompt-side finalize discipline — the recovery-key convention is pinned.
# ---------------------------------------------------------------------------


def test_prompt_pins_recovery_key_and_finalize_discipline() -> None:
    system, _ = build_codeagent_prompts(
        {
            "mission_id": "M",
            "mission_context": {"title": "T", "objective": "O", "deliverables": ["D"]},
        }
    )
    assert "report_draft" in system, "the recovery-key convention must be pinned"
    assert "STEP BUDGET" in system or "stranded" in system.lower()
    # Must instruct finalize-before-budget-exhaustion.
    assert "final_answer" in system


# ---------------------------------------------------------------------------
# Telemetry — recovered_from_scratchpad surfaces on synthesis_telemetry.
# ---------------------------------------------------------------------------


def test_recovered_flag_surfaces_in_synthesis_telemetry() -> None:
    result = SynthesisResult(
        output_markdown=_CITED_DRAFT,
        step_count=40,
        self_terminated=False,
        hit_max_steps=True,
        runtime_seconds=1.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        tool_call_summary={},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        recovered_from_scratchpad=True,
    )
    s_tel = fnode._build_synthesis_telemetry(result)
    assert s_tel["recovered_from_scratchpad"] is True
