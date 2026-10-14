"""Static-AST regression guards for the S80 m03 single-H1 strip.

16th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD108:

* ``deepsearch/agent/codeagent_finalize_node.py`` defines
  ``_strip_leading_h1`` as a top-level function that takes a single ``str``
  argument and returns a ``str``.
* ``codeagent_finalize_node`` calls ``_strip_leading_h1`` on the agent
  ``final_markdown`` BEFORE the preamble concatenation site, so the wrapper
  H1 emitted by ``build_report_header`` is the only top-level `#` line in
  the rendered markdown.

These guards fail loudly if a future sprint:

* Renames or removes ``_strip_leading_h1``.
* Re-removes the call in ``codeagent_finalize_node`` (regressing the
  duplicate-H1 defect closed at S80 m03).
* Inverts the strip target (e.g., strips the wrapper instead of the body
  — `feedback_no_speculative_feature_flags.md` locks the wrapper-wins
  rule).

Tests do not call any LLM, do not hit the network, and exercise only AST
inspection plus a runtime call on the helper.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINALIZE_NODE_PATH = (
    REPO_ROOT / "deepsearch" / "agent" / "codeagent_finalize_node.py"
)


def _module_ast() -> ast.Module:
    return ast.parse(FINALIZE_NODE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Definition-surface invariants.
# ---------------------------------------------------------------------------


def test_strip_leading_h1_is_top_level_function() -> None:
    """Helper must be a top-level synchronous function (not nested, not async)."""
    tree = _module_ast()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_strip_leading_h1":
            return
    raise AssertionError(
        "_strip_leading_h1 must be a top-level function in "
        "deepsearch/agent/codeagent_finalize_node.py (S80 m03 PD108)."
    )


def test_strip_leading_h1_signature_takes_one_str_returns_str() -> None:
    """Pin the helper's contract — `(markdown: str) -> str`.

    Drift here would let a caller silently pass an iterable / Mapping,
    masking the stripping behavior.
    """
    tree = _module_ast()
    for node in tree.body:
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_strip_leading_h1"
        ):
            continue
        args = node.args
        assert len(args.args) == 1, (
            "_strip_leading_h1 must take exactly one positional argument "
            "(`markdown: str`) — S80 m03 PD108 contract."
        )
        arg = args.args[0]
        assert arg.arg == "markdown", (
            f"first arg must be named `markdown`; got {arg.arg!r}"
        )
        # Both annotation slots (arg + return) must read `str`.
        annotation = arg.annotation
        assert isinstance(annotation, ast.Name) and annotation.id == "str", (
            "_strip_leading_h1 markdown arg must be annotated `str`."
        )
        ret = node.returns
        assert isinstance(ret, ast.Name) and ret.id == "str", (
            "_strip_leading_h1 must be annotated to return `str`."
        )
        return
    raise AssertionError("_strip_leading_h1 not found in finalize node module.")


# ---------------------------------------------------------------------------
# Call-site invariant — strip helper is wired into the preamble concatenation.
# ---------------------------------------------------------------------------


def test_codeagent_finalize_node_calls_strip_helper_on_final_markdown() -> None:
    """The strip helper must be invoked inside ``codeagent_finalize_node``.

    Without the call, the agent body H1 reaches the preamble concatenation
    and the rendered markdown carries two top-level H1 lines — the exact
    duplicate-H1 defect closed at S80 m03.
    """
    tree = _module_ast()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "codeagent_finalize_node"
        ):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "_strip_leading_h1"
            ):
                # Argument must reference the local `final_markdown` variable.
                assert len(sub.args) == 1, (
                    "_strip_leading_h1 must be called with exactly one "
                    "positional argument (`final_markdown`)."
                )
                arg = sub.args[0]
                assert isinstance(arg, ast.Name) and arg.id == "final_markdown", (
                    "_strip_leading_h1 must be called on `final_markdown` "
                    "BEFORE preamble concatenation (S80 m03 PD108). The "
                    "preamble is metadata, not a heading — wrapping or "
                    "reordering the call regresses the duplicate-H1 defect."
                )
                return
        raise AssertionError(
            "codeagent_finalize_node body must call _strip_leading_h1 — "
            "S80 m03 PD108 wiring contract."
        )
    raise AssertionError(
        "Async function `codeagent_finalize_node` not found in "
        "deepsearch/agent/codeagent_finalize_node.py."
    )


# ---------------------------------------------------------------------------
# Runtime smoke — the helper actually behaves on the canonical fixtures.
# ---------------------------------------------------------------------------


def test_runtime_strip_removes_leading_h1() -> None:
    from deepsearch.agent.codeagent_finalize_node import _strip_leading_h1

    assert _strip_leading_h1("# Topic\n\nBody.") == "Body."


def test_runtime_strip_preserves_h2_only_input() -> None:
    from deepsearch.agent.codeagent_finalize_node import _strip_leading_h1

    src = "## Section\n\nBody."
    assert _strip_leading_h1(src) == src


def test_runtime_strip_only_first_of_consecutive_h1s() -> None:
    from deepsearch.agent.codeagent_finalize_node import _strip_leading_h1

    # Defined edge case — only the first leading H1 is stripped; subsequent
    # H1s survive (the second becomes the new leading H1 of the residual).
    assert _strip_leading_h1("# A\n# B\n\nBody.") == "# B\n\nBody."
