"""S87 m03 bounded recovery persists one canonical artifact pair.

The paid synthesis and critique actors are deterministic stubs. Everything
from citation extraction through the real report/protocol filesystem writers
runs on the production finalize path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueAnnotation, CritiqueResult
from deepsearch.report.generator import REPORT_DIRECTORY_ENV


def _state() -> dict:
    return {
        "mission_id": "S87-M03-INTEGRATION",
        "mission_uuid": "87000003-0000-4000-8000-000000000003",
        "mission_title": "Bounded Recovery Persistence",
        "mission_objectives": ["persist the selected recovered artifact once"],
        "mission_context": {
            "title": "Bounded Recovery Persistence",
            "objective": "persist the selected recovered artifact once",
            "deliverables": [],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _synthesis(markdown: str, *, steps: int) -> SynthesisResult:
    return SynthesisResult(
        output_markdown=markdown,
        step_count=steps,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=1.0,
        token_usage={"input": 10, "output": 5, "total": 15},
        tool_call_summary={"total_calls": 0, "by_tool": {}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        error=None,
        sources_collected=[],
    )


def _damaged() -> CritiqueResult:
    annotations = [
        CritiqueAnnotation(
            anchor=f"Unsupported claim {index}.",
            verdict="unsupported",
            note="No collected source supports the claim.",
            applied=True,
            reason="applied_gap_note",
        )
        for index in range(10)
    ]
    return CritiqueResult(
        output_markdown="# Draft\n\n" + "\n".join(
            "[Data unavailable in primary sources]" for _ in annotations
        ),
        claim_count=20,
        gap_notes_inserted=10,
        runtime_seconds=0.1,
        annotations=annotations,
    )


def _healthy(markdown: str) -> CritiqueResult:
    return CritiqueResult(
        output_markdown=markdown,
        claim_count=1,
        gap_notes_inserted=0,
        runtime_seconds=0.1,
    )


def test_recovery_reruns_validation_and_writes_one_report_protocol_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REPORT_DIRECTORY_ENV, str(tmp_path))
    first = _synthesis("# Draft\n\nUnsupported claims.", steps=35)
    recovered_markdown = "# Draft\n\nRecovered report supported by the available record."
    recovered = _synthesis(recovered_markdown, steps=2)

    with patch.object(
        fnode,
        "run_codeagent_synthesis",
        side_effect=[first, recovered],
    ) as synthesis_mock, patch.object(
        fnode,
        "run_critique",
        side_effect=[_damaged(), _healthy(recovered_markdown)],
    ) as critique_mock, patch.object(
        fnode,
        "extract_citations",
        wraps=fnode.extract_citations,
    ) as extraction_mock, patch.object(
        fnode,
        "_run_liveness_check",
        wraps=fnode._run_liveness_check,
    ) as liveness_mock, patch.object(
        fnode,
        "save_report",
        wraps=fnode.save_report,
    ) as report_writer, patch.object(
        fnode,
        "save_protocol",
        wraps=fnode.save_protocol,
    ) as protocol_writer:
        result = asyncio.run(fnode.codeagent_finalize_node(_state()))

    assert synthesis_mock.call_count == 2
    assert critique_mock.call_count == 2
    assert extraction_mock.call_count == 2
    assert liveness_mock.call_count == 2
    assert report_writer.call_count == 1
    assert protocol_writer.call_count == 1

    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["status"] == "completed"
    assert recovery["passes"] == 1
    assert recovery["selected_artifact"] == "recovery"
    assert result["synthesis_telemetry"]["recovery_passes"] == 1

    report_path = Path(result["report_path"])
    protocol_path = Path(result["protocol_path"])
    assert report_path.exists()
    assert protocol_path.exists()
    assert "Recovered report supported" in report_path.read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.md")) == [report_path]
    assert list(tmp_path.glob("*.json")) == [protocol_path]
