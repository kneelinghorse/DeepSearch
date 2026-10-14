"""Bounded-call + no-retrieval regression guards for the critique surface.

Per ``cmos/foundational-docs/rebuild_roadmap_s63_to_s73.md`` §3.7 (critique
must NOT trigger re-retrieval), these guards lock the structural contract:

1. ``run_critique`` is invoked exactly once when no S87 recovery is triggered.
   A triggered bounded recovery may run one second critique, but never a loop.
2. The critique surface does not call any retrieval tool — ``web_search``,
   ``source_fetch``, ``url_liveness``. Sources are read from the
   ``sources_collected`` accumulator on ``SynthesisResult`` only.

If a future sprint accidentally introduces critique-driven retrieval or a
multi-call critique loop, these tests fail loudly. They are structural tests
— synthesis is mocked, persistence is sandboxed, no LLMs are called.

Test layout follows the S66.1 convention (decision #501): flat in tests/unit/,
not tests/unit/agent/.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent import critique as critique_mod
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _state(mission_id: str = "OODS-FIGMA-HOST-01") -> dict:
    return {
        "mission_id": mission_id,
        "mission_objectives": ["criterion 1"],
        "mission_context": {
            "title": "Test Mission",
            "objective": "verify critique invariants",
            "background": "context blurb",
            "required_entities": ["X"],
            "excluded_entities": [],
            "deliverables": [],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _synth_result_with_sources() -> SynthesisResult:
    return SynthesisResult(
        output_markdown="# Draft\n\nClaim with citation [1].",
        step_count=4,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=42.0,
        token_usage={"input": 1000, "output": 2000, "total": 3000},
        tool_call_summary={"total_calls": 7, "by_tool": {"web_search": 3}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        error=None,
        sources_collected=[
            {
                "url": "https://docs.aws.amazon.com/lambda/x",
                "title": "Lambda doc",
                "snippet": "ok",
                "score": 0.9,
                "authority_tier": "high",
            }
        ],
    )


def _critique_result() -> CritiqueResult:
    return CritiqueResult(
        output_markdown="# Critiqued\n\nClaim with citation [1].",
        claim_count=1,
        gap_notes_inserted=0,
        runtime_seconds=2.5,
        token_usage={"input": 500, "output": 200, "total": 700},
        error=None,
    )


@pytest.fixture()
def report_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Healthy-path invariant — one critique when recovery is not triggered.
# ---------------------------------------------------------------------------


def test_critique_invoked_exactly_once_per_mission(report_dir):
    """A healthy completed draft receives one critique and no recovery pass."""

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            return_value=_synth_result_with_sources(),
        ), patch.object(
            fnode,
            "run_critique",
            return_value=_critique_result(),
        ) as mock_critique:
            await fnode.codeagent_finalize_node(_state())
            return mock_critique

    mock_critique = asyncio.run(run())
    assert mock_critique.call_count == 1, (
        f"Expected run_critique to be called exactly once, got {mock_critique.call_count}"
    )


def test_critique_skipped_on_blocked_path(report_dir):
    """Critique must NOT run when synthesis is blocked (BLOCKED short-circuits)."""

    async def run():
        with patch.object(
            fnode,
            "run_codeagent_synthesis",
            side_effect=fnode.CodeAgentRunnerError("LINKUP_API_KEY not set"),
        ), patch.object(
            fnode,
            "run_critique",
        ) as mock_critique:
            return await fnode.codeagent_finalize_node(_state()), mock_critique

    result, mock_critique = asyncio.run(run())
    assert result["mission_outcome"] == "blocked"
    assert mock_critique.call_count == 0


# ---------------------------------------------------------------------------
# 2. No-retrieval invariant — critique never calls a retrieval tool.
# ---------------------------------------------------------------------------


def test_critique_does_not_trigger_retrieval(monkeypatch: pytest.MonkeyPatch):
    """``run_critique`` must not call web_search, source_fetch, or url_liveness.

    Patches the three live-tool implementations; asserts ``call_count == 0``
    after a critique pass with a mocked LLM. Re-retrieval triggered by the
    critique stage is an explicit roadmap §3.7 anti-pattern.
    """

    import deepsearch.agent.tools.source_fetch as sf_mod
    import deepsearch.agent.tools.url_liveness as ul_mod
    import deepsearch.agent.tools.web_search as ws_mod

    web_search_calls = {"n": 0}
    source_fetch_calls = {"n": 0}
    url_liveness_calls = {"n": 0}

    def _ws(*args, **kwargs):
        web_search_calls["n"] += 1
        return []

    def _sf(*args, **kwargs):
        source_fetch_calls["n"] += 1
        return {
            "text": "",
            "status": "ok",
            "status_code": 200,
            "fetched_at": "now",
            "final_url": "",
        }

    def _ul(*args, **kwargs):
        url_liveness_calls["n"] += 1
        return {"alive": True, "status_code": 200, "final_url": ""}

    monkeypatch.setattr(ws_mod, "web_search", _ws)
    monkeypatch.setattr(sf_mod, "source_fetch", _sf)
    monkeypatch.setattr(ul_mod, "url_liveness", _ul)

    # S85 m01 — annotation-only output format (empty annotations = no-op).
    fake_model_response = (
        f"{critique_mod.CRITIQUE_OUTPUT_DELIMITER}\n"
        '{"claim_count": 1, "annotations": []}\n'
    )

    class _Resp:
        content = fake_model_response
        token_usage = None

    fake_model = lambda **kw: _Resp()  # noqa: E731 — deterministic stub

    result = critique_mod.run_critique(
        draft_markdown="# Draft\n\nClaim [1].",
        sources_collected=[
            {
                "url": "https://docs.aws.amazon.com/lambda/x",
                "title": "Lambda doc",
                "snippet": "ok",
                "score": 0.9,
                "authority_tier": "high",
            }
        ],
        mission_context={"title": "T", "objective": "O"},
        model=fake_model,
    )

    assert result.error is None
    assert web_search_calls["n"] == 0, "critique must not call web_search"
    assert source_fetch_calls["n"] == 0, "critique must not call source_fetch"
    assert url_liveness_calls["n"] == 0, "critique must not call url_liveness"


def test_critique_module_does_not_import_retrieval_tools():
    """Static guard: the critique module must not import any retrieval tool.

    AST-walks ``deepsearch.agent.critique`` and asserts no
    ``from deepsearch.agent.tools.{web_search,source_fetch,url_liveness} import``
    statements. A future drift that adds re-retrieval would have to bring in
    an import first; this test fails loudly when it does.
    """

    import ast
    from pathlib import Path

    critique_path = Path(critique_mod.__file__)
    tree = ast.parse(critique_path.read_text(encoding="utf-8"))
    forbidden_modules = {
        "deepsearch.agent.tools.web_search",
        "deepsearch.agent.tools.source_fetch",
        "deepsearch.agent.tools.url_liveness",
        "deepsearch.tools.linkup_search",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module not in forbidden_modules, (
                f"critique.py imports forbidden retrieval module: {node.module}"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, (
                    f"critique.py imports forbidden retrieval module: {alias.name}"
                )


# ---------------------------------------------------------------------------
# 3. Call-order invariant — critique invoked between synthesis and report gen.
# ---------------------------------------------------------------------------


def test_critique_runs_between_synthesis_and_report(report_dir):
    """Order: run_codeagent_synthesis → run_critique → generate_markdown_report."""

    call_order: list[str] = []

    def _record_synth(*args, **kwargs):
        call_order.append("synthesis")
        return _synth_result_with_sources()

    def _record_critique(*args, **kwargs):
        call_order.append("critique")
        return _critique_result()

    def _record_report(state, *, synthesized_content, **kwargs):
        call_order.append("report")
        # The report must receive the critique's output, not the synthesis output.
        assert synthesized_content == _critique_result().output_markdown
        return synthesized_content

    async def run():
        with patch.object(fnode, "run_codeagent_synthesis", side_effect=_record_synth), \
             patch.object(fnode, "run_critique", side_effect=_record_critique), \
             patch.object(fnode, "generate_markdown_report", side_effect=_record_report), \
             patch.object(fnode, "save_report", return_value="/tmp/report.md"), \
             patch.object(fnode, "save_protocol", return_value="/tmp/protocol.md"):
            return await fnode.codeagent_finalize_node(_state())

    asyncio.run(run())
    assert call_order == ["synthesis", "critique", "report"], (
        f"Expected synthesis → critique → report; got {call_order}"
    )
