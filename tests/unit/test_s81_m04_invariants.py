"""Static-AST + runtime regression guards for S81 m04 — Context-Pressure Observability.

17th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD125:

* ``deepsearch/agent/codeagent_finalize_node.py`` declares
  ``_DEEPSEEK_V4_PRO_MAX_CONTEXT`` and ``_CONTEXT_PRESSURE_THRESHOLD`` at
  module scope (load-bearing constants that future analysis can re-bucket
  without changing the runtime contract).
* ``_build_synthesis_telemetry`` emits a ``context_pressure`` field on
  every synthesis-telemetry envelope, populated by ``_build_context_pressure``.
* The envelope has the documented field shape: ``model_max_context``,
  compatibility field ``last_step_input_tokens`` (the latest measured input),
  ``measurement_step``, ``final_step_usage_available``,
  ``failed_step_input_unknown``, ``pressure_pct``,
  ``self_term_under_pressure``, ``cumulative_input_tokens``, and ``step_count``.
  It remains flat (no nested arrays) per PD129 risk #7.
* ``self_term_under_pressure`` fires only when self_terminated=True,
  final-step usage is available, AND pressure_pct > 0.7. A generation failure
  with no final-step usage explicitly records that its failed input is unknown.

These guards fail loudly if a future sprint:

* Renames or removes the ``context_pressure`` field from the synthesis
  telemetry envelope.
* Changes the field shape (e.g. nesting the values, renaming the keys,
  adding nested arrays that bloat the protocol JSON size).
* Attributes a prior step's usage to an unmeasured failed or final step.
* Inverts the self_term_under_pressure logic (e.g. firing on hit_max_steps
  instead of a self-termination with measured final-step usage).
* Removes the defensive 0-fallback for absent / malformed step traces.

The guards exercise pure-function logic with synthetic SynthesisResult
instances; no LLM calls, no network, no Postgres.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FINALIZE_NODE_PATH = REPO_ROOT / "deepsearch" / "agent" / "codeagent_finalize_node.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _module_constant_value(path: Path, name: str) -> Any:
    """Return the constant value of a top-level ``name = <constant>`` assignment."""
    tree = _module_ast(path)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == name):
            continue
        value = node.value
        if isinstance(value, ast.Constant):
            return value.value
    return None


# ---------------------------------------------------------------------------
# m04 — module-level constants.
# ---------------------------------------------------------------------------


def test_deepseek_v4_pro_max_context_constant() -> None:
    """``_DEEPSEEK_V4_PRO_MAX_CONTEXT == 1_000_000`` (1M token context window).

    S81 m04 hot-fix 2026-05-14: the original 65536 value was an outdated
    DeepSeek V3 guess; V4 Pro carries 1M context. The post-S81 Alexander
    validation dispatch surfaced the error (pressure_pct > 1.0 reported
    under the wrong denominator). Constant updated in lockstep with the
    runtime; this guard pins the corrected value.
    """
    value = _module_constant_value(FINALIZE_NODE_PATH, "_DEEPSEEK_V4_PRO_MAX_CONTEXT")
    assert value == 1_000_000, (
        "S81 m04 (PD125) hot-fix — _DEEPSEEK_V4_PRO_MAX_CONTEXT must "
        "equal 1_000_000 (1M tokens). DeepSeek V4 Pro's context window "
        "is 1M; the original 65536 value was wrong by ~15x and caused "
        "false self_term_under_pressure flags on healthy missions."
    )


def test_context_pressure_threshold_constant() -> None:
    """``_CONTEXT_PRESSURE_THRESHOLD == 0.7`` (initial observational calibration)."""
    value = _module_constant_value(FINALIZE_NODE_PATH, "_CONTEXT_PRESSURE_THRESHOLD")
    assert value == 0.7, (
        "S81 m04 (PD125) — _CONTEXT_PRESSURE_THRESHOLD must equal 0.7. "
        "The threshold is observational, not load-bearing; future "
        "analysis can re-bucket it without changing the protocol contract "
        "(raw pressure_pct stays available for re-bucketing)."
    )


# ---------------------------------------------------------------------------
# m04 — synthesis telemetry envelope emits context_pressure.
# ---------------------------------------------------------------------------


def test_build_synthesis_telemetry_emits_context_pressure() -> None:
    """Runtime: the envelope must carry ``context_pressure`` as a top-level key."""
    from deepsearch.agent.codeagent_finalize_node import _build_synthesis_telemetry
    from deepsearch.agent.codeagent_runner import SynthesisResult

    result = SynthesisResult(
        output_markdown="Report", step_count=5, self_terminated=True, hit_max_steps=False,
        runtime_seconds=12.5, token_usage={"input": 100000, "output": 5000, "total": 105000},
        tool_call_summary={}, authority_tier_distribution={}, pool_size=0,
        step_trace=[
            {"step_index": 0, "token_usage": {"input_tokens": 5000}},
            {"step_index": 4, "token_usage": {"input_tokens": 55000}},
        ], effective_config={"context_window": 1_000_000},
    )
    envelope = _build_synthesis_telemetry(result)
    assert "context_pressure" in envelope, (
        "S81 m04 (PD125) — every synthesis telemetry envelope must "
        "include `context_pressure` so post-mission forensic analysis "
        "can distinguish honest finish from pressure self-termination."
    )


def test_context_pressure_field_shape_flat() -> None:
    """The envelope carries flat pressure values plus measurement provenance."""
    from dataclasses import dataclass, field
    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    @dataclass
    class _StubResult:
        step_count: int = 3
        self_terminated: bool = False
        hit_max_steps: bool = False
        runtime_seconds: float = 1.0
        token_usage: dict = field(default_factory=lambda: {"input": 12345, "output": 100, "total": 12445})
        tool_call_summary: dict = field(default_factory=dict)
        authority_tier_distribution: dict = field(default_factory=dict)
        pool_size: int = 0
        step_trace: list = field(default_factory=lambda: [
            {"step_index": 0, "token_usage": {"input_tokens": 1000}},
            {"step_index": 2, "token_usage": {"input_tokens": 8000}},
        ])
        error: str | None = None
        sources_collected: list = field(default_factory=list)
        effective_config: dict = field(default_factory=lambda: {"context_window": 1_000_000})

    cp = _build_context_pressure(_StubResult())
    expected_keys = {
        "model_max_context",
        "last_step_input_tokens",
        "measurement_step",
        "final_step_usage_available",
        "failed_step_input_unknown",
        "pressure_pct",
        "self_term_under_pressure",
        "cumulative_input_tokens",
        "step_count",
    }
    assert set(cp.keys()) == expected_keys, (
        f"S81 m04 (PD125, PD129 risk #7) — context_pressure must carry "
        f"exactly the documented keys. Got {set(cp.keys())}; expected "
        f"{expected_keys}."
    )
    assert cp["measurement_step"] == 2
    assert cp["final_step_usage_available"] is True
    assert cp["failed_step_input_unknown"] is False
    # All values must be JSON scalars or null (no nested arrays/dicts).
    for key, value in cp.items():
        assert value is None or isinstance(value, (int, float, bool)), (
            f"context_pressure.{key} must be a scalar or null; "
            f"got {type(value).__name__}. Per PD129 risk #7 the protocol "
            "JSON size envelope is kept flat."
        )


# ---------------------------------------------------------------------------
# m04 — self_term_under_pressure logic.
# ---------------------------------------------------------------------------


def _make_result(*, self_terminated: bool, last_input: int, step_count: int = 5, cumulative: int = 100000) -> Any:
    """Build a stub SynthesisResult-shaped object for the threshold tests.

    Uses ``SimpleNamespace`` rather than a dataclass so the parameters from
    the enclosing function can populate the attributes at runtime without
    the dataclass-default-vs-closure issue.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        step_count=step_count,
        self_terminated=self_terminated,
        hit_max_steps=False,
        runtime_seconds=1.0,
        token_usage={"input": cumulative, "output": 100, "total": cumulative + 100},
        tool_call_summary={},
        authority_tier_distribution={},
        pool_size=0,
        step_trace=[{"step_index": 0, "token_usage": {"input_tokens": last_input}}],
        error=None,
        sources_collected=[],
        effective_config={"context_window": 1_000_000},
    )


def test_self_term_under_pressure_fires_above_threshold_and_self_terminated() -> None:
    """Self-termination, measured final usage, and high pressure are required."""
    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    # S81 m04 hot-fix: 750_000 / 1_000_000 = 0.75 (above 0.7).
    result = _make_result(self_terminated=True, last_input=750_000)
    cp = _build_context_pressure(result)
    assert cp["self_term_under_pressure"] is True, (
        f"S81 m04 — self_term_under_pressure must fire when self_terminated "
        f"AND final-step usage is measured AND pressure_pct > 0.7. "
        f"Got pressure_pct={cp['pressure_pct']}, "
        f"self_term_under_pressure={cp['self_term_under_pressure']}."
    )


def test_self_term_under_pressure_does_not_fire_when_low_pressure() -> None:
    """When pressure_pct <= 0.7, the flag must NOT fire even on self_terminated."""
    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    # S81 m04 hot-fix: 100_000 / 1_000_000 = 0.10 (well below 0.7).
    result = _make_result(self_terminated=True, last_input=100_000)
    cp = _build_context_pressure(result)
    assert cp["self_term_under_pressure"] is False, (
        f"S81 m04 — self_term_under_pressure must NOT fire below 0.7 "
        f"threshold. Got pressure_pct={cp['pressure_pct']}."
    )


def test_self_term_under_pressure_does_not_fire_on_hit_max() -> None:
    """Measured high pressure does not fire the flag on hit_max_steps."""
    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    # S81 m04 hot-fix: 800_000 / 1_000_000 = 0.80 (above 0.7) — but agent
    # hit the step cap instead of self-terminating, so the flag must NOT fire.
    result = _make_result(self_terminated=False, last_input=800_000)
    cp = _build_context_pressure(result)
    assert cp["self_term_under_pressure"] is False, (
        "S81 m04 — self_term_under_pressure requires self_terminated. "
        "Inverting the logic to fire on hit_max_steps regresses the "
        "Alexander-mission failure-mode discriminator."
    )


def test_context_pressure_handles_empty_step_trace() -> None:
    """Defensive: empty step_trace produces last_step_input_tokens=0."""
    from dataclasses import dataclass, field
    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    @dataclass
    class _StubResult:
        step_count: int = 0
        self_terminated: bool = False
        hit_max_steps: bool = False
        runtime_seconds: float = 0.0
        token_usage: dict = field(default_factory=lambda: {"input": 0, "output": 0, "total": 0})
        tool_call_summary: dict = field(default_factory=dict)
        authority_tier_distribution: dict = field(default_factory=dict)
        pool_size: int = 0
        step_trace: list = field(default_factory=list)
        error: str | None = None
        sources_collected: list = field(default_factory=list)
        effective_config: dict = field(default_factory=lambda: {"context_window": 1_000_000})

    cp = _build_context_pressure(_StubResult())
    assert cp["last_step_input_tokens"] == 0
    assert cp["measurement_step"] is None
    assert cp["final_step_usage_available"] is False
    assert cp["failed_step_input_unknown"] is False
    assert cp["pressure_pct"] == 0.0
    assert cp["self_term_under_pressure"] is False


def test_context_pressure_handles_malformed_step_token_usage() -> None:
    """Defensive: missing / non-int token_usage entries fall back to 0."""
    from dataclasses import dataclass, field
    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    @dataclass
    class _StubResult:
        step_count: int = 1
        self_terminated: bool = True
        hit_max_steps: bool = False
        runtime_seconds: float = 1.0
        token_usage: dict = field(default_factory=lambda: {"input": 100, "output": 50, "total": 150})
        tool_call_summary: dict = field(default_factory=dict)
        authority_tier_distribution: dict = field(default_factory=dict)
        pool_size: int = 0
        step_trace: list = field(
            default_factory=lambda: [{"step_index": 0}]  # no token_usage at all
        )
        error: str | None = None
        sources_collected: list = field(default_factory=list)
        effective_config: dict = field(default_factory=lambda: {"context_window": 1_000_000})

    cp = _build_context_pressure(_StubResult())
    assert cp["last_step_input_tokens"] == 0
    assert cp["measurement_step"] is None
    assert cp["final_step_usage_available"] is False
    assert cp["failed_step_input_unknown"] is False
    assert cp["pressure_pct"] == 0.0


def test_context_pressure_uses_latest_recorded_usage_after_failed_step() -> None:
    """A usage-less failed ActionStep must not masquerade as zero pressure."""

    from types import SimpleNamespace

    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    result = SimpleNamespace(
        effective_config={"context_window": 1_000_000},        step_count=3,
        self_terminated=False,
        token_usage={"input": 246_000, "output": 3_000, "total": 249_000},
        step_trace=[
            {"step_index": 1, "token_usage": {"input_tokens": 123_456}},
            {"step_index": 2, "type": "ActionStep"},
        ],
        generation_failure_error=(
            '{"category":"execution","exception_class":"AgentGenerationError"}'
        ),
    )

    cp = _build_context_pressure(result)

    assert cp["last_step_input_tokens"] == 123_456
    assert cp["measurement_step"] == 1
    assert cp["final_step_usage_available"] is False
    assert cp["failed_step_input_unknown"] is True
    assert cp["pressure_pct"] == 0.1235


def test_self_termination_pressure_requires_final_step_measurement() -> None:
    """Prior high pressure cannot be attributed to an unmeasured final step."""

    from types import SimpleNamespace

    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    result = SimpleNamespace(
        effective_config={"context_window": 1_000_000},        step_count=2,
        self_terminated=True,
        token_usage={"input": 1_600_000, "output": 1_000, "total": 1_601_000},
        step_trace=[
            {"step_index": 0, "token_usage": {"input_tokens": 800_000}},
            {"step_index": 1, "type": "ActionStep"},
        ],
    )

    cp = _build_context_pressure(result)

    assert cp["pressure_pct"] == 0.8
    assert cp["final_step_usage_available"] is False
    assert cp["self_term_under_pressure"] is False


def test_alexander_mission_replay_shape() -> None:
    """Replay a high-pressure synthetic Alexander-shape mission.

    The Alexander mission protocol shows synthesis_telemetry.token_usage.input
    = 1,040,740 cumulative across 21 steps with self_terminated=True at 21/40.
    Pre-S81 there was no field to surface whether the agent finished honestly
    or self-terminated under pressure. Post-S81 the context_pressure envelope
    surfaces both signals.

    S81 m04 hot-fix note: the post-S81 Alexander re-dispatch (UUID 7befc41f)
    showed last_step_input_tokens=100,418 = 10% of the corrected 1M context
    window — i.e. the agent was NOT actually under context pressure on that
    real mission. This test uses a synthetic high-pressure value to exercise
    the flag's wiring; the empirical Alexander datapoint sits in the negative
    direction (low pressure) per the post-S81 protocol.
    """
    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    # S81 m04 hot-fix: 900_000 / 1_000_000 = 0.90 (above 0.7) — synthetic
    # high-pressure scenario for flag-wiring verification.
    result = _make_result(
        self_terminated=True,
        last_input=900_000,
        step_count=21,
        cumulative=1_040_740,
    )
    cp = _build_context_pressure(result)
    assert cp["step_count"] == 21
    assert cp["cumulative_input_tokens"] == 1_040_740
    assert cp["pressure_pct"] > 0.7
    assert cp["self_term_under_pressure"] is True


# ---------------------------------------------------------------------------
# m04 — AST guards on the helper function declarations.
# ---------------------------------------------------------------------------


def test_build_context_pressure_function_declared() -> None:
    """The helper function must exist as a top-level function in the module."""
    tree = _module_ast(FINALIZE_NODE_PATH)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_context_pressure":
            return
    raise AssertionError(
        "S81 m04 (PD125) — _build_context_pressure must exist as a "
        "top-level function in deepsearch/agent/codeagent_finalize_node.py."
    )


def test_last_step_input_tokens_helper_declared() -> None:
    """The defensive last-step-input helper must exist at module scope."""
    tree = _module_ast(FINALIZE_NODE_PATH)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_last_step_input_tokens":
            return
    raise AssertionError(
        "S81 m04 (PD125) — _last_step_input_tokens helper must exist at "
        "module scope. The defensive 0-fallback path lives there; removing "
        "the helper inlines a bug surface."
    )
