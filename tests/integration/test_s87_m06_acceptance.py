"""S87 M06 CAT-R002-shaped no-paid acceptance and benchmark truth."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import (
    GAP_NOTE_DEFAULT,
    CritiqueAnnotation,
    CritiqueResult,
)
from deepsearch.eval.benchmark import benchmark
from deepsearch.report.generator import REPORT_DIRECTORY_ENV

_SOURCE_URL = "https://primary.example/acceptance-evidence"
_OBJECTIVES = [
    "Explain the acceptance evidence",
    "Separate grounding sources from probes",
    "Preserve contract identity",
    "Reconcile runtime and persisted counts",
    "Report recovery truthfully",
]
_DELIVERABLES = [f"Contract slot {index:02d} evidence" for index in range(1, 13)]


def _state(case: str) -> dict[str, Any]:
    title = f"Synthetic {case.title()} Acceptance"
    mission_uuid = (
        "87000006-0000-4000-8000-000000000001"
        if case == "healthy"
        else "87000006-0000-4000-8000-000000000002"
    )
    return {
        "mission_id": (
            "CAT-R002-SYNTHETIC-DAMAGE"
            if case == "damaged"
            else "S87-M06-HEALTHY"
        ),
        "mission_uuid": mission_uuid,
        "mission_title": title,
        "mission_objectives": list(_OBJECTIVES),
        "mission_context": {
            "title": title,
            "objective": "Prove one canonical, reconciled acceptance artifact.",
            "deliverables": list(_DELIVERABLES),
            "required_entities": ["primary.example"],
            "authority_seed_domains": ["primary.example"],
        },
        "sources_found": [],
        "telemetry": {
            "runtime_identity": {
                "build_hash": "synthetic-build-hash",
                "build_source": "source_tree_hash",
            }
        },
    }


def _source() -> dict[str, Any]:
    body = " ".join(
        [
            "The primary acceptance source supports the contract, retrieval, "
            "authority, persistence, and recovery facts."
        ]
        * 30
    )
    return {
        "url": _SOURCE_URL,
        "title": "Primary acceptance evidence",
        "snippet": body[:300],
        "body": body,
        "body_source": "full",
        "score": 0.98,
        "authority_tier": "high",
    }


def _draft(*, damaged: bool) -> str:
    lines = ["# Synthetic acceptance report"]
    for index, deliverable in enumerate(_DELIVERABLES, start=1):
        lines.extend(
            [
                f"## {deliverable}",
                (
                    f"{deliverable} is satisfied by primary evidence and the "
                    "persisted protocol keeps this contract field aligned with "
                    f"runtime telemetry. [cite:{_SOURCE_URL}]"
                ),
                (
                    "The acceptance record distinguishes a grounding source from "
                    "fetch attempts and liveness probes while retaining the source's "
                    "high authority tier."
                ),
            ]
        )
        if damaged and index <= 10:
            for claim_index in (index * 2 - 1, index * 2):
                lines.append(
                    f"Unsupported damage claim {claim_index:02d} has no synthetic evidence."
                )
    lines.extend(
        [
            "## References",
            f"- [Primary acceptance evidence]({_SOURCE_URL})",
        ]
    )
    return "\n\n".join(lines)


def _tool_outcomes() -> dict[str, Any]:
    return {
        "by_tool": {
            "web_search": {
                "calls": 1,
                "ok": 1,
                "error": 0,
                "results_total": 1,
                "circuit_open": 0,
                "by_category": {},
            },
            "source_fetch": {
                "calls": 1,
                "ok": 1,
                "error": 0,
                "results_total": 1,
                "circuit_open": 0,
                "by_category": {},
            },
        },
        "records": [],
        "records_truncated": 0,
    }


def _synthesis(markdown: str, *, steps: int, carry_retrieval: bool) -> SynthesisResult:
    source = _source()
    sources = [source] if carry_retrieval else []
    return SynthesisResult(
        output_markdown=markdown,
        step_count=steps,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=2.0,
        token_usage={"input": 100, "output": 50, "total": 150},
        tool_call_summary={
            "total_calls": 2 if carry_retrieval else 0,
            "by_tool": {"web_search": 1, "source_fetch": 1}
            if carry_retrieval
            else {},
            "scratchpad_keys": [],
        },
        authority_tier_distribution={
            "high": len(sources),
            "medium": 0,
            "low": 0,
        },
        pool_size=len(sources),
        step_trace=[{"step_index": index} for index in range(steps)],
        sources_collected=sources,
        sources_discovered=[source] if carry_retrieval else [],
        sources_fetched=[source] if carry_retrieval else [],
        fetch_attempts=[{"url": _SOURCE_URL, "status": "ok"}]
        if carry_retrieval
        else [],
        probes=[
            {"url": _SOURCE_URL, "alive": True},
            {"url": _SOURCE_URL, "alive": True, "cache_hit": True},
        ]
        if carry_retrieval
        else [],
        retrieval_state=(
            {
                "discovered": 1,
                "fetched_readable": 1,
                "fetch_attempts": 1,
                "probes": 2,
                "grounding_sources": 1,
            }
            if carry_retrieval
            else {}
        ),
        tool_outcomes=_tool_outcomes() if carry_retrieval else {},
        model_id="deepseek-v4-pro",
    )


def _critique_side_effect(damaged: bool) -> Callable[..., CritiqueResult]:
    calls = 0

    def critique(draft_markdown: str, *_args: Any, **_kwargs: Any) -> CritiqueResult:
        nonlocal calls
        calls += 1
        if damaged and calls == 1:
            annotations: list[CritiqueAnnotation] = []
            output = draft_markdown
            for index in range(1, 21):
                anchor = (
                    f"Unsupported damage claim {index:02d} has no synthetic evidence."
                )
                annotations.append(
                    CritiqueAnnotation(
                        anchor=anchor,
                        verdict="unsupported",
                        note="No collected source supports this synthetic claim.",
                        citation_urls=(),
                        applied=True,
                        reason="applied_gap_note",
                    )
                )
                output = output.replace(anchor, GAP_NOTE_DEFAULT)
            return CritiqueResult(
                output_markdown=output,
                claim_count=150,
                gap_notes_inserted=20,
                runtime_seconds=0.1,
                grounding_telemetry={"sources_included": 1, "sources_dropped": 0},
                annotations=annotations,
            )
        return CritiqueResult(
            output_markdown=draft_markdown,
            claim_count=30,
            gap_notes_inserted=0,
            runtime_seconds=0.1,
            grounding_telemetry={"sources_included": 1, "sources_dropped": 0},
            annotations=[],
        )

    return critique


async def _all_live(**kwargs: Any) -> tuple[Any, LivenessTelemetry]:
    annotations = kwargs["annotations"]
    for annotation in annotations:
        annotation.live = True
    distinct = len({annotation.url for annotation in annotations})
    return annotations, LivenessTelemetry(
        checked=len(annotations),
        live=distinct,
        distinct_urls_checked=distinct,
        cache_hits=max(0, len(annotations) - distinct),
    )


@pytest.mark.parametrize("case", ["damaged", "healthy"])  # type: ignore[misc]
def test_no_paid_finalize_paths_reconcile_every_persisted_surface(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Damaged recovers once; healthy never recovers; both persist once."""
    report_dir = tmp_path / case
    report_dir.mkdir()
    monkeypatch.setenv(REPORT_DIRECTORY_ENV, str(report_dir))
    damaged = case == "damaged"
    first = _synthesis(_draft(damaged=damaged), steps=12, carry_retrieval=True)
    recovered = _synthesis(_draft(damaged=False), steps=3, carry_retrieval=False)
    synthesis_results = [first, recovered] if damaged else [first]

    with (
        patch.object(
            fnode,
            "run_codeagent_synthesis",
            side_effect=synthesis_results,
        ) as synthesis_mock,
        patch.object(
            fnode,
            "run_critique",
            side_effect=_critique_side_effect(damaged),
        ) as critique_mock,
        patch.object(fnode, "_run_liveness_check", side_effect=_all_live),
        patch.object(fnode, "save_report", wraps=fnode.save_report) as report_writer,
        patch.object(
            fnode,
            "save_protocol",
            wraps=fnode.save_protocol,
        ) as protocol_writer,
    ):
        result = asyncio.run(fnode.codeagent_finalize_node(_state(case)))

    expected_passes = 1 if damaged else 0
    assert synthesis_mock.call_count == expected_passes + 1
    assert critique_mock.call_count == expected_passes + 1
    assert report_writer.call_count == 1
    assert protocol_writer.call_count == 1
    assert len(list(report_dir.glob("*.md"))) == 1
    assert len(list(report_dir.glob("*.json"))) == 1

    telemetry = result["synthesis_telemetry"]
    recovery = telemetry["recovery"]
    assert recovery["passes"] == expected_passes
    assert recovery["status"] == ("completed" if damaged else "not_triggered")
    assert recovery["selected_artifact"] == (
        "recovery" if damaged else "first_pass"
    )
    assert recovery["decision"]["triggered"] is damaged
    if damaged:
        first_critique = recovery["first_pass"]["critique_telemetry"]
        assert first_critique["annotation_summary"]["applied"] == 20
        assert len(recovery["first_pass"]["deliverable_coverage"]["satisfied"]) == 12
        assert recovery["first_pass"]["deliverable_coverage"]["missing"] == []

    protocol_path = Path(result["protocol_path"])
    report_path = Path(result["report_path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    quality = telemetry["quality_record"]
    forensic = protocol["report_metadata"]["forensic"]

    assert protocol["mission_id"] == _state(case)["mission_id"]
    assert protocol["title"] == _state(case)["mission_title"]
    assert protocol["research_statement"]["objective"] == (
        "Prove one canonical, reconciled acceptance artifact."
    )
    assert protocol["research_statement"]["success_metrics"] == _OBJECTIVES
    assert protocol["status"] == "complete"
    assert result["mission_outcome"] == "complete"
    assert forensic["quality_record"] == quality
    assert telemetry["citation_consistency"]["consistent"] is True

    citations = protocol["citations"]
    assert citations
    assert all(citation["live"] is True for citation in citations)
    assert telemetry["citation_telemetry"]["citations_extracted"] == len(citations)
    assert telemetry["liveness_telemetry"] == {
        "checked": len(citations),
        "live": 1,
        "dead": 0,
        "orphan": 0,
        "unknown": 0,
        "hallucinated_url_count": 0,
        "distinct_urls_checked": 1,
        "wall_clock_ms": 0,
        "cache_hits": len(citations) - 1,
        "budget_exhausted": False,
    }

    sources = protocol["sources_collected"]
    tier_counts = Counter(source["authority_tier"] for source in sources)
    assert tier_counts == Counter({"high": 1})
    retrieval = quality["retrieval_state"]
    assert retrieval == {
        "available": True,
        "discovered": 1,
        "fetched_readable": 1,
        "grounding_sources": len(sources),
        "fetch_attempts": 1,
        "probes": 2,
    }
    assert len(protocol["evidence"]) >= 1
    assert retrieval["grounding_sources"] != retrieval["probes"]
    assert quality["retrieval_failures"] == {
        "available": True,
        "total": 0,
        "by_tool": {},
    }
    assert quality["recovery"]["passes"] == expected_passes
    assert quality["recovery"]["selected_artifact"] == recovery["selected_artifact"]
    assert quality["critique"]["status"] == "ok"
    assert quality["critique"]["applied"] == 0

    scoresheet = benchmark(protocol["mission_id"], protocol, report)
    assert scoresheet.critique_status == "ok"
    assert scoresheet.critique_gap_count == quality["critique"]["applied"]
    assert scoresheet.recovery == quality["recovery"]
    assert scoresheet.telemetry_mismatches == []
    assert scoresheet.per_mission_thresholds[
        "structured_critique_gaps_lt_10"
    ]["pass"] is True
    assert scoresheet.per_mission_thresholds["telemetry_matches_truth"]["pass"] is True
