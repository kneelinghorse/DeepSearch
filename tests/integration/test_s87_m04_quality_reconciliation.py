"""S87 m04 protocol, worker quality record, and CAT scoresheet reconciliation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueAnnotation, CritiqueResult
from deepsearch.eval.benchmark import benchmark
from deepsearch.report.generator import REPORT_DIRECTORY_ENV
from deepsearch.worker.writer import build_execution_metadata


def _state() -> dict:
    return {
        "mission_id": "S87-M04-RECONCILE",
        "mission_uuid": "87000004-0000-4000-8000-000000000004",
        "mission_title": "Quality Reconciliation",
        "mission_objectives": ["reconcile quality facts"],
        "mission_context": {
            "title": "Quality Reconciliation",
            "objective": "reconcile quality facts",
            "deliverables": [],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _synthesis(markdown: str, *, steps: int) -> SynthesisResult:
    source = {
        "url": "https://primary.example/evidence",
        "title": "Primary evidence",
        "body": "The primary record supports the recovered result.",
        "snippet": "The primary record supports the recovered result.",
        "content_quality": "readable_body",
        "authority_tier": "high",
    }
    return SynthesisResult(
        output_markdown=markdown,
        step_count=steps,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=1.0,
        token_usage={"input": 10, "output": 5, "total": 15},
        tool_call_summary={"total_calls": 1, "by_tool": {"web_search": 1}},
        authority_tier_distribution={"high": 1, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        sources_collected=[source],
        sources_discovered=[source],
        sources_fetched=[source],
        fetch_attempts=[{"url": source["url"], "status": "ok"}],
        probes=[],
        retrieval_state={
            "discovered": 1,
            "fetched_readable": 1,
            "fetch_attempts": 1,
            "probes": 0,
            "grounding_sources": 1,
        },
        tool_outcomes={
            "by_tool": {
                "web_search": {
                    "calls": 1,
                    "ok": 1,
                    "error": 0,
                    "results_total": 1,
                    "circuit_open": 0,
                    "by_category": {},
                }
            },
            "records": [],
            "records_truncated": 0,
        },
    )


def _damaged() -> CritiqueResult:
    annotations = [
        CritiqueAnnotation(
            anchor=f"Unsupported claim {index}.",
            verdict="unsupported",
            applied=True,
            reason="applied_gap_note",
        )
        for index in range(10)
    ]
    return CritiqueResult(
        output_markdown="# Report\n\n" + "\n".join(
            "[Data unavailable in primary sources]" for _ in annotations
        ),
        claim_count=20,
        gap_notes_inserted=10,
        runtime_seconds=0.1,
        annotations=annotations,
        grounding_telemetry={"draft_chars_dropped": 0},
    )


def _healthy(markdown: str) -> CritiqueResult:
    return CritiqueResult(
        output_markdown=markdown,
        claim_count=2,
        gap_notes_inserted=0,
        runtime_seconds=0.1,
        grounding_telemetry={"draft_chars_dropped": 0},
    )


async def _live(citations, sources_collected, budget_seconds=45.0):
    del sources_collected, budget_seconds
    for citation in citations:
        citation.live = True
    return list(citations), LivenessTelemetry(
        checked=len(citations),
        live=len(citations),
        distinct_urls_checked=len(citations),
    )


def test_recovered_quality_facts_reconcile_across_all_persisted_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REPORT_DIRECTORY_ENV, str(tmp_path))
    first = _synthesis("# Report\n\nUnsupported report.", steps=35)
    recovered_body = (
        "# Report\n\nRecovered result [cite:https://primary.example/evidence].\n\n"
        "A separate field remains [Data unavailable in primary sources].\n\n"
        "## References\n\n"
        "[1] Primary evidence — https://primary.example/evidence\n"
    )
    recovered = _synthesis(recovered_body, steps=2)

    with patch.object(
        fnode,
        "run_codeagent_synthesis",
        side_effect=[first, recovered],
    ), patch.object(
        fnode,
        "run_critique",
        side_effect=[_damaged(), _healthy(recovered_body.replace(
            "[cite:https://primary.example/evidence]", ""
        ))],
    ), patch.object(
        fnode,
        "check_citations_liveness",
        new=_live,
    ):
        result = asyncio.run(fnode.codeagent_finalize_node(_state()))

    protocol = json.loads(Path(result["protocol_path"]).read_text(encoding="utf-8"))
    report_body = Path(result["report_path"]).read_text(encoding="utf-8")
    protocol_record = protocol["report_metadata"]["forensic"]["quality_record"]
    synthesis_record = result["synthesis_telemetry"]["quality_record"]

    final_state = {**_state(), **result}
    final_state["telemetry"] = {
        **result["telemetry"],
        "quality_record": synthesis_record,
    }
    metadata_record = build_execution_metadata(final_state, 2.0).to_dict()[
        "quality_record"
    ]
    scoresheet = benchmark("S87-M04-RECONCILE", protocol, report_body)

    assert protocol_record == synthesis_record == metadata_record
    assert protocol_record["critique"]["applied"] == 0
    assert protocol_record["recovery"]["passes"] == 1
    assert protocol_record["recovery"]["outcome"] == "completed"
    assert scoresheet.critique_gap_count == protocol_record["critique"]["applied"]
    assert scoresheet.recovery == {
        "passes": 1,
        "outcome": "completed",
        "selected_artifact": "recovery",
        "triggered": True,
    }
    assert scoresheet.declared_unavailable_count == 1
    assert scoresheet.per_mission_thresholds["structured_critique_gaps_lt_10"][
        "pass"
    ] is True
