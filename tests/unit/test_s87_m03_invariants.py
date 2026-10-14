"""Sprint 87 m03 structured critique and one-pass recovery invariants."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import (
    CRITIQUE_ANNOTATION_MAX_COUNT,
    CRITIQUE_OUTPUT_DELIMITER,
    CritiqueAnnotation,
    CritiqueResult,
    run_critique,
)
from deepsearch.agent.execution_control import (
    ExecutionCancelledError,
    bind_execution_cancel_event,
)
from deepsearch.agent.recovery import (
    RECOVERY_PROMPT_MAX_CHARS,
    build_recovery_prompt,
    decide_recovery,
    evaluate_recovery_regression,
    remaining_recovery_steps,
)
from deepsearch.error_safety import safe_error_from_parts
from deepsearch.report.citation import CitationAnnotation


def _state(*, deliverables: list[str] | None = None) -> dict:
    return {
        "mission_id": "S87-RECOVERY",
        "mission_objectives": ["repair material critique damage"],
        "mission_context": {
            "title": "Recovery test",
            "objective": "repair the report once",
            "deliverables": deliverables or [],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _synthesis(
    markdown: str,
    *,
    step_count: int,
    sources: list[dict] | None = None,
    error: str | None = None,
    self_terminated: bool | None = None,
    hit_max_steps: bool = False,
) -> SynthesisResult:
    return SynthesisResult(
        output_markdown=markdown,
        step_count=step_count,
        self_terminated=(error is None if self_terminated is None else self_terminated),
        hit_max_steps=hit_max_steps,
        runtime_seconds=5.0,
        token_usage={"input": 100, "output": 50, "total": 150},
        token_usage_complete=True,
        tool_call_summary={"total_calls": 1, "by_tool": {"web_search": 1}},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        error=error,
        sources_collected=sources or [],
    )


def _damage_result(count: int = 10, *, token_total: int = 0) -> CritiqueResult:
    annotations = [
        CritiqueAnnotation(
            anchor=f"Original damaged claim {index} with unsupported detail.",
            verdict="unsupported",
            note="No collected source supports this claim.",
            applied=True,
            reason="applied_gap_note",
        )
        for index in range(count)
    ]
    return CritiqueResult(
        output_markdown="# Report\n\n" + "\n".join(
            "[Data unavailable in primary sources]" for _ in range(count)
        ),
        claim_count=100,
        gap_notes_inserted=count,
        runtime_seconds=1.0,
        token_usage={"input": token_total, "output": 0, "total": token_total},
        token_usage_complete=True,
        annotations=annotations,
    )


def _healthy_result(markdown: str, *, token_total: int = 0) -> CritiqueResult:
    return CritiqueResult(
        output_markdown=markdown,
        claim_count=20,
        gap_notes_inserted=0,
        runtime_seconds=1.0,
        token_usage={"input": token_total, "output": 0, "total": token_total},
        token_usage_complete=True,
    )


def test_attempt_accounting_retains_lower_bound_when_critique_usage_is_missing() -> None:
    """A paid pass without provider usage cannot be labeled complete."""

    synthesis = _synthesis("# Report\n", step_count=3)
    critique = _healthy_result("# Report\n", token_total=0)
    critique.token_usage_complete = False

    accounting = fnode._build_attempt_accounting(
        [synthesis],
        [critique],
        token_usage_complete=True,
    )

    assert accounting["synthesis_token_usage_complete"] is True
    assert accounting["critique_token_usage_complete"] is False
    assert accounting["token_usage"]["total"] == 150
    assert accounting["token_usage_complete"] is False


@pytest.fixture()
def report_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    return tmp_path


def test_typed_critique_annotation_carries_disposition_and_known_citation_url() -> None:
    draft = "Supported intro.\nDamaged claim needs primary evidence."
    anchor = "Damaged claim needs primary evidence."
    start = draft.index(anchor)
    known_url = "https://primary.example/evidence"
    raw = (
        f"{CRITIQUE_OUTPUT_DELIMITER}\n"
        + json.dumps(
            {
                "claim_count": 2,
                "annotations": [
                    {
                        "anchor": anchor,
                        "verdict": "unsupported",
                        "note": "No support found.",
                        "citation_urls": [known_url, "https://invented.example/x"],
                    }
                ],
            }
        )
    )
    citation = CitationAnnotation(
        start_index=start,
        end_index=start + len(anchor),
        url=known_url,
        live=True,
    )

    result = run_critique(
        draft,
        [{"url": known_url, "body": "Primary evidence body."}],
        {"title": "T", "objective": "O"},
        model=lambda **kwargs: raw,
        citations=[citation],
    )

    assert result.error is None
    assert result.gap_notes_inserted == 1
    assert len(result.annotations) == 1
    annotation = result.annotations[0]
    assert annotation.applied is True
    assert annotation.reason == "applied_gap_note"
    assert annotation.citation_urls == (known_url,)
    assert result.annotation_summary() == {
        "total": 1,
        "applied": 1,
        "skipped": 0,
        "dropped": 0,
        "by_reason": {"applied_gap_note": 1},
        "reconciled": True,
    }


def test_critique_annotation_ledger_is_bounded_and_reconciles() -> None:
    raw = (
        f"{CRITIQUE_OUTPUT_DELIMITER}\n"
        + json.dumps(
            {
                "claim_count": CRITIQUE_ANNOTATION_MAX_COUNT + 5,
                "annotations": [
                    {
                        "anchor": f"Missing unique claim number {index}.",
                        "verdict": "unsupported",
                    }
                    for index in range(CRITIQUE_ANNOTATION_MAX_COUNT + 5)
                ],
            }
        )
    )

    result = run_critique(
        "# Draft\n\nNo matching claims are present.",
        [],
        {"title": "T", "objective": "O"},
        model=lambda **kwargs: raw,
    )

    summary = result.annotation_summary()
    assert summary["total"] == CRITIQUE_ANNOTATION_MAX_COUNT
    assert summary["applied"] == 0
    assert summary["skipped"] == CRITIQUE_ANNOTATION_MAX_COUNT
    assert summary["dropped"] == 5
    assert summary["by_reason"] == {
        "anchor_not_found": CRITIQUE_ANNOTATION_MAX_COUNT
    }
    assert summary["reconciled"] is True


def test_cat_r002_damage_triggers_even_with_full_coverage_and_zero_dead() -> None:
    coverage = {
        "satisfied": [f"slot-{index}" for index in range(12)],
        "missing": [],
        "shortfall_score": 0.0,
    }
    decision = decide_recovery(_damage_result(20), coverage)

    assert decision.triggered is True
    assert "applied_count_threshold" in decision.reasons
    assert decision.applied_count == 20
    assert decision.coverage_shortfall == 0.0


def test_healthy_draft_has_no_recovery_trigger() -> None:
    decision = decide_recovery(
        _healthy_result("# Report\n\nHealthy evidence."),
        {"satisfied": ["slot"], "missing": [], "shortfall_score": 0.0},
    )

    assert decision.triggered is False
    assert decision.reasons == ()


def test_coverage_shortfall_is_an_independent_recovery_trigger() -> None:
    decision = decide_recovery(
        _healthy_result("# Report\n\nHealthy but incomplete."),
        {"satisfied": [], "missing": ["Required matrix"], "shortfall_score": 1.0},
    )

    assert decision.triggered is True
    assert decision.reasons == ("coverage_shortfall",)


def test_recovery_prompt_is_bounded_and_uses_only_live_sources() -> None:
    live_url = "https://live.example/source"
    dead_url = "https://dead.example/source"
    prompt = build_recovery_prompt(
        original_markdown="draft " * 100_000,
        annotations=_damage_result(50).annotations,
        coverage={"missing": [f"missing slot {index}" for index in range(50)]},
        sources=[
            {"url": live_url, "title": "Live", "body": "L" * 10_000},
            {"url": dead_url, "title": "Dead", "body": "D" * 10_000},
        ],
        live_citation_urls=[live_url],
        remaining_steps=5,
    )

    assert len(prompt.directive) <= RECOVERY_PROMPT_MAX_CHARS
    assert live_url in prompt.directive
    assert dead_url not in prompt.directive
    assert "at most 5" in prompt.directive
    assert prompt.telemetry["applied_claims_included"] == 30
    assert prompt.telemetry["applied_claims_dropped"] == 20
    assert prompt.telemetry["missing_slots_included"] == 20
    assert prompt.telemetry["missing_slots_dropped"] == 30
    assert prompt.telemetry["prompt_chars"] == len(prompt.directive)
    assert remaining_recovery_steps(total_steps=40, used_steps=35) == 5


def test_one_recovery_pass_reruns_validation_then_persists_once(
    report_dir: Path,
) -> None:
    del report_dir
    first_draft = "# Report\n\nOriginal damaged report."
    recovered_draft = "# Report\n\nRecovered supported report with complete coverage."
    initial = _synthesis(first_draft, step_count=35)
    recovered = _synthesis(recovered_draft, step_count=3)
    critiques = [_damage_result(10), _healthy_result(recovered_draft)]
    order: list[str] = []
    synth_calls = 0

    def traced_synthesis(*args, **kwargs):
        nonlocal synth_calls
        synth_calls += 1
        order.append("synthesis")
        assert kwargs["verbosity"] == fnode.SMOLAGENTS_VERBOSITY_OFF == -1
        if synth_calls == 2:
            assert kwargs["max_steps"] == 5
            assert "RECOVERY PASS 1 OF 1" in kwargs["user_directive"]
        return initial if synth_calls == 1 else recovered

    original_extract = fnode.extract_citations

    def traced_extract(*args, **kwargs):
        order.append("extract")
        return original_extract(*args, **kwargs)

    async def traced_liveness(**kwargs):
        order.append("liveness")
        return kwargs["annotations"], LivenessTelemetry()

    def traced_critique(*args, **kwargs):
        order.append("critique")
        return critiques.pop(0)

    original_outcome = fnode.evaluate_outcome_gate
    original_report = fnode.generate_markdown_report
    original_save_report = fnode.save_report
    original_save_protocol = fnode.save_protocol

    def traced_outcome(*args, **kwargs):
        order.append("outcome")
        return original_outcome(*args, **kwargs)

    def traced_report(*args, **kwargs):
        order.append("report")
        return original_report(*args, **kwargs)

    def traced_save_report(*args, **kwargs):
        order.append("save_report")
        return original_save_report(*args, **kwargs)

    def traced_save_protocol(*args, **kwargs):
        order.append("save_protocol")
        return original_save_protocol(*args, **kwargs)

    with patch.object(
        fnode, "run_codeagent_synthesis", side_effect=traced_synthesis
    ) as synth_mock, patch.object(
        fnode, "extract_citations", side_effect=traced_extract
    ) as extract_mock, patch.object(
        fnode, "_run_liveness_check", side_effect=traced_liveness
    ) as liveness_mock, patch.object(
        fnode, "run_critique", side_effect=traced_critique
    ) as critique_mock, patch.object(
        fnode, "evaluate_outcome_gate", side_effect=traced_outcome
    ), patch.object(
        fnode, "generate_markdown_report", side_effect=traced_report
    ), patch.object(
        fnode, "save_report", side_effect=traced_save_report
    ) as save_report_mock, patch.object(
        fnode, "save_protocol", side_effect=traced_save_protocol
    ) as save_protocol_mock:
        result = asyncio.run(fnode.codeagent_finalize_node(_state()))

    assert synth_mock.call_count == 2
    assert extract_mock.call_count == 2
    assert liveness_mock.call_count == 2
    assert critique_mock.call_count == 2
    assert save_report_mock.call_count == 1
    assert save_protocol_mock.call_count == 1
    assert order == [
        "synthesis",
        "extract",
        "liveness",
        "critique",
        "synthesis",
        "extract",
        "liveness",
            "critique",
            "outcome",
            "report",
            # Regrade after citation coordinates are rebased onto the exact
            # persisted document; pre-rebase marker counts are not terminal.
            "outcome",
            "save_report",
            "save_protocol",
        ]
    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["status"] == "completed"
    assert recovery["passes"] == 1
    assert recovery["steps_used"] == 3
    assert recovery["steps_remaining_after"] == 2
    assert result["synthesis_telemetry"]["recovery_passes"] == 1
    assert result["synthesis_telemetry"]["step_count"] == 38
    assert "Recovered supported report" in Path(result["report_path"]).read_text()


def test_lease_loss_after_first_critique_prevents_recovery_and_persistence(
    report_dir: Path,
) -> None:
    """A critique completing after fence loss cannot launch recovery or writes."""

    del report_dir
    cancel_event = threading.Event()
    initial = _synthesis("# Report\n\nOriginal damaged report.", step_count=35)

    def lose_during_critique(*args, **kwargs):
        del args, kwargs
        cancel_event.set()
        return _damage_result(10)

    with patch.object(
        fnode, "run_codeagent_synthesis", return_value=initial
    ) as synthesis, patch.object(
        fnode, "run_critique", side_effect=lose_during_critique
    ) as critique, patch.object(
        fnode, "save_report"
    ) as save_report, patch.object(
        fnode, "save_protocol"
    ) as save_protocol, bind_execution_cancel_event(cancel_event):
        with pytest.raises(ExecutionCancelledError):
            asyncio.run(fnode.codeagent_finalize_node(_state()))

    assert synthesis.call_count == 1
    assert critique.call_count == 1
    save_report.assert_not_called()
    save_protocol.assert_not_called()


def test_lease_loss_during_recovery_prevents_second_validation_and_persistence(
    report_dir: Path,
) -> None:
    """Recovery output from a stale attempt is rejected before re-validation."""

    del report_dir
    cancel_event = threading.Event()
    initial = _synthesis("# Report\n\nOriginal damaged report.", step_count=35)
    recovered = _synthesis("# Report\n\nRecovered report.", step_count=2)
    synthesis_calls = 0

    def synthesis_side_effect(*args, **kwargs):
        nonlocal synthesis_calls
        del args, kwargs
        synthesis_calls += 1
        if synthesis_calls == 2:
            cancel_event.set()
            return recovered
        return initial

    with patch.object(
        fnode, "run_codeagent_synthesis", side_effect=synthesis_side_effect
    ) as synthesis, patch.object(
        fnode, "run_critique", return_value=_damage_result(10)
    ) as critique, patch.object(
        fnode, "save_report"
    ) as save_report, patch.object(
        fnode, "save_protocol"
    ) as save_protocol, bind_execution_cancel_event(cancel_event):
        with pytest.raises(ExecutionCancelledError):
            asyncio.run(fnode.codeagent_finalize_node(_state()))

    assert synthesis.call_count == 2
    assert critique.call_count == 1
    save_report.assert_not_called()
    save_protocol.assert_not_called()


def test_recovery_failure_persists_first_pass_once(report_dir: Path) -> None:
    del report_dir
    initial = _synthesis("# Report\n\nOriginal damaged report.", step_count=35)
    calls = 0

    def synthesis_side_effect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return initial
        raise RuntimeError("recovery backend failed")

    with patch.object(
        fnode, "run_codeagent_synthesis", side_effect=synthesis_side_effect
    ) as synth_mock, patch.object(
        fnode, "run_critique", return_value=_damage_result(10)
    ), patch.object(
        fnode, "save_report", wraps=fnode.save_report
    ) as save_report_mock, patch.object(
        fnode, "save_protocol", wraps=fnode.save_protocol
    ) as save_protocol_mock:
        result = asyncio.run(fnode.codeagent_finalize_node(_state()))

    assert synth_mock.call_count == 2
    assert save_report_mock.call_count == 1
    assert save_protocol_mock.call_count == 1
    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["status"] == "failed"
    assert recovery["passes"] == 1
    assert recovery["selected_artifact"] == "first_pass"
    assert json.loads(recovery["error"]) == {
        "category": "execution",
        "exception_class": "RuntimeError",
    }
    assert result["synthesis_telemetry"]["recovery_passes"] == 1
    persisted = Path(result["report_path"]).read_text()
    assert "[Data unavailable in primary sources]" in persisted


def test_healthy_finalize_performs_zero_additional_calls(report_dir: Path) -> None:
    del report_dir
    healthy_markdown = "# Report\n\nHealthy supported report."
    initial = _synthesis(healthy_markdown, step_count=4)

    with patch.object(
        fnode, "run_codeagent_synthesis", return_value=initial
    ) as synth_mock, patch.object(
        fnode, "run_critique", return_value=_healthy_result(healthy_markdown)
    ) as critique_mock:
        result = asyncio.run(fnode.codeagent_finalize_node(_state()))

    assert synth_mock.call_count == 1
    assert critique_mock.call_count == 1
    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["status"] == "not_triggered"
    assert recovery["passes"] == 0


def test_second_damaged_pass_cannot_trigger_a_third_pass(report_dir: Path) -> None:
    del report_dir
    initial = _synthesis("# Report\n\nFirst damaged draft.", step_count=35)
    recovered = _synthesis("# Report\n\nSecond still-damaged draft.", step_count=2)
    recovered.generation_failure_error = safe_error_from_parts(
        "execution", "AgentGenerationError"
    ).to_text()
    recovered.generation_failure_reason = "timeout"
    recovered.generation_failure_reason_source = "class"
    recovered.generation_failure_root_class = "TimeoutError"
    recovered.generation_failure_transient = True
    recovered.generation_failure_step = 2
    recovered.generation_failure_pass_count = 1
    recovered.generation_failure_consumed_tokens = 150
    recovered.generation_failure_retry_suppressed_reason = "paid_progress"
    recovered.generation_recovery_attempted = True
    recovered.generation_recovery_result = "success"
    recovered.generation_recovery_finish_reason = "stop"
    recovered.generation_error_recovered = True

    with patch.object(
        fnode, "run_codeagent_synthesis", side_effect=[initial, recovered]
    ) as synth_mock, patch.object(
        fnode, "run_critique", side_effect=[_damage_result(10), _damage_result(10)]
    ) as critique_mock:
        result = asyncio.run(fnode.codeagent_finalize_node(_state()))

    assert synth_mock.call_count == 2
    assert critique_mock.call_count == 2
    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["passes"] == 1
    assert recovery["status"] == "completed"
    assert recovery["selected_artifact"] == "recovery"
    second_failure = recovery["second_pass"]["generation_failure"]
    assert second_failure["generation_failure_reason"] == "timeout"
    assert second_failure["generation_failure_reason_source"] == "class"
    assert second_failure["generation_failure_root_class"] == "TimeoutError"
    assert second_failure["generation_failure_pass_count"] == 1
    assert result["synthesis_telemetry"]["generation_failure_pass"] == 1


def test_recovery_that_regresses_coverage_keeps_first_pass(report_dir: Path) -> None:
    """S87 #823 hardening: a recovered rewrite that INCREASES the completeness
    shortfall must be rejected — the reviewable first pass is persisted, not the
    gutted recovery. Without the non-regression guard the recovered artifact is
    selected on the sole condition that its critique did not hard-error, which
    relocated the un-diffed whole-document rewrite SPOF to the recovery stage."""
    del report_dir
    first_draft = "# Report\n\nFirst pass with full supported coverage."
    recovered_draft = "# Report\n\nGUTTED recovery that dropped supported sections."
    initial = _synthesis(
        first_draft,
        step_count=35,
        self_terminated=False,
        hit_max_steps=True,
    )
    recovered = _synthesis(
        recovered_draft,
        step_count=3,
        self_terminated=True,
        hit_max_steps=False,
    )
    liveness_pass = 0

    async def liveness_with_ledger(**kwargs):
        nonlocal liveness_pass
        liveness_pass += 1
        url = f"https://evidence.example.test/validation-{liveness_pass}"
        return kwargs["annotations"], LivenessTelemetry(
            ledger_records=[
                {
                    "tool": "url_liveness",
                    "url": url,
                    "status": "ok",
                    "status_code": 200,
                    "error_category": None,
                    "alive": True,
                }
            ],
            ledger_records_truncated=0,
        )

    def coverage_side_effect(markdown: str, spec: object) -> dict:
        if "GUTTED" in markdown:  # the recovered pass regressed coverage
            return {
                "satisfied": ["Findings"],
                "missing": ["Analysis", "Recommendations"],
                "shortfall_score": 2 / 3,
            }
        return {
            "satisfied": ["Findings", "Analysis"],
            "missing": ["Recommendations"],
            "shortfall_score": 1 / 3,
        }

    with patch.object(
        fnode, "run_codeagent_synthesis", side_effect=[initial, recovered]
    ) as synth_mock, patch.object(
        fnode,
        "run_critique",
        side_effect=[_healthy_result(first_draft), _healthy_result(recovered_draft)],
    ), patch.object(
        fnode, "completeness_check", side_effect=coverage_side_effect
    ), patch.object(
        fnode, "_run_liveness_check", side_effect=liveness_with_ledger
    ):
        result = asyncio.run(
            fnode.codeagent_finalize_node(
                _state(deliverables=["Findings", "Analysis", "Recommendations"])
            )
        )

    # Recovery WAS attempted (first-pass coverage shortfall triggered it)...
    assert synth_mock.call_count == 2
    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["passes"] == 1
    # ...but the regressive recovery is rejected and the first pass is kept.
    assert recovery["status"] == "rejected_regression"
    assert recovery["selected_artifact"] == "first_pass"
    assert result["synthesis_telemetry"]["self_terminated"] is False
    assert result["synthesis_telemetry"]["hit_max_steps"] is True
    assert recovery["first_pass"]["termination"] == {
        "step_count": 35,
        "self_terminated": False,
        "hit_max_steps": True,
        "output_truncated": False,
    }
    assert recovery["second_pass"]["termination"] == {
        "step_count": 3,
        "self_terminated": True,
        "hit_max_steps": False,
        "output_truncated": False,
    }
    guard = recovery["regression_guard"]
    assert guard["non_regressive"] is False
    assert "coverage" in guard["regressed_dimensions"]
    assert guard["recovered_shortfall"] > guard["first_shortfall"]
    assert [
        record["url"]
        for record in result["synthesis_telemetry"]["tool_outcomes"][
            "ledger_records"
        ]
    ] == [
        "https://evidence.example.test/validation-1",
        "https://evidence.example.test/validation-2",
    ]
    persisted = Path(result["report_path"]).read_text()
    assert "full supported coverage" in persisted
    assert "GUTTED" not in persisted


def test_recovery_that_holds_coverage_and_citations_is_selected(
    report_dir: Path,
) -> None:
    """The guard is do-no-harm, not disable-recovery: a non-regressive recovery
    (coverage held/improved, no live citations dropped) is still selected."""
    del report_dir
    first_draft = "# Report\n\nOriginal thin draft."
    recovered_draft = "# Report\n\nImproved recovery with restored coverage."
    initial = _synthesis(
        first_draft,
        step_count=35,
        self_terminated=False,
        hit_max_steps=True,
    )
    recovered = _synthesis(
        recovered_draft,
        step_count=3,
        # A usable recovery can return before its local ceiling without the
        # runner observing final_answer. That is not max-step exhaustion.
        self_terminated=False,
        hit_max_steps=False,
    )

    def coverage_side_effect(markdown: str, spec: object) -> dict:
        if "Improved" in markdown:  # recovery improved coverage
            return {"satisfied": ["A", "B"], "missing": [], "shortfall_score": 0.0}
        return {"satisfied": ["A"], "missing": ["B"], "shortfall_score": 1 / 2}

    with patch.object(
        fnode, "run_codeagent_synthesis", side_effect=[initial, recovered]
    ), patch.object(
        fnode,
        "run_critique",
        side_effect=[
            _healthy_result(first_draft, token_total=20),
            _healthy_result(recovered_draft, token_total=25),
        ],
    ), patch.object(
        fnode, "completeness_check", side_effect=coverage_side_effect
    ):
        result = asyncio.run(
            fnode.codeagent_finalize_node(_state(deliverables=["A", "B"]))
        )

    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["status"] == "completed"
    assert recovery["selected_artifact"] == "recovery"
    assert recovery["regression_guard"]["non_regressive"] is True
    additive = {
        "provider_attempts", "model_accounting", "retrieval_accounting", "human_repair_minutes"
    }
    assert {k: v for k, v in recovery["attempt_accounting"].items() if k not in additive} == {
        "schema_version": 1,
        "synthesis_passes": 2,
        "critique_passes": 2,
        "steps_used": 38,
        "synthesis_runtime_seconds": 10.0,
        "critique_runtime_seconds": 2.0,
        "synthesis_token_usage": {"input": 200, "output": 100, "total": 300},
        "critique_token_usage": {"input": 45, "output": 0, "total": 45},
        "synthesis_token_usage_complete": True,
        "critique_token_usage_complete": True,
        "token_usage": {"input": 245, "output": 100, "total": 345},
        "token_usage_complete": True,
    }
    assert result["synthesis_telemetry"]["self_terminated"] is False
    assert result["synthesis_telemetry"]["hit_max_steps"] is False
    assert result["mission_outcome"] == "complete"
    assert recovery["first_pass"]["termination"] == {
        "step_count": 35,
        "self_terminated": False,
        "hit_max_steps": True,
        "output_truncated": False,
    }
    assert recovery["second_pass"]["termination"] == {
        "step_count": 3,
        "self_terminated": False,
        "hit_max_steps": False,
        "output_truncated": False,
    }
    assert "Improved recovery" in Path(result["report_path"]).read_text()


# ---------------------------------------------------------------------------
# S88 lever 4 — the recovery regression guard also rejects a recovered pass
# that RE-WORSENS structured critique damage (`applied`). The #823 un-diffed
# whole-document rewrite pushed R001 from applied=10 to applied=13 during
# recovery; T2 (applied<10/report) is graded on the SELECTED pass, so a
# recovery that adds damage must not be selected. The `applied` term is
# additive and evaluated only when coverage did not already regress, keeping
# `non_regressive = not (coverage OR live_citations OR applied)` an
# AND-of-negations (monotonic: the guard can only ever reject MORE).
# ---------------------------------------------------------------------------


def _cov(shortfall: float) -> dict:
    return {"satisfied": [], "missing": [], "shortfall_score": shortfall}


def test_guard_rejects_recovery_that_worsens_applied_with_coverage_held() -> None:
    guard = evaluate_recovery_regression(
        first_coverage=_cov(0.0),
        recovered_coverage=_cov(0.0),
        first_live_citations=5,
        recovered_live_citations=5,
        first_applied=10,
        recovered_applied=13,
    )
    assert guard.non_regressive is False
    assert "applied" in guard.regressed_dimensions
    assert "coverage" not in guard.regressed_dimensions
    assert "live_citations" not in guard.regressed_dimensions
    assert guard.first_applied == 10
    assert guard.recovered_applied == 13


def test_guard_rejects_worsened_applied_even_when_coverage_improves() -> None:
    """The key adversarial case: a recovery that BUYS coverage with more critique
    damage is still refused — trading damage for coverage is not a genuine
    repair, and the `applied` term fires because coverage is non-regressive.
    """
    guard = evaluate_recovery_regression(
        first_coverage=_cov(0.5),
        recovered_coverage=_cov(0.1),  # coverage improved
        first_live_citations=5,
        recovered_live_citations=6,  # citations improved
        first_applied=10,
        recovered_applied=13,  # ...but damage worsened
    )
    assert guard.non_regressive is False
    assert "applied" in guard.regressed_dimensions


def test_guard_coverage_regression_rejects_and_omits_applied_term() -> None:
    """When coverage already regressed the pass is rejected on `coverage`; the
    `applied` term is intentionally absent (it is evaluated only in the
    coverage-non-regressive branch) — but the verdict is rejection either way.
    """
    guard = evaluate_recovery_regression(
        first_coverage=_cov(0.1),
        recovered_coverage=_cov(0.6),  # coverage regressed
        first_live_citations=5,
        recovered_live_citations=5,
        first_applied=13,
        recovered_applied=10,  # applied improved, but coverage sinks it
    )
    assert guard.non_regressive is False
    assert "coverage" in guard.regressed_dimensions
    assert "applied" not in guard.regressed_dimensions


@pytest.mark.parametrize("applied", [0, 10])
def test_guard_equal_applied_is_not_a_regression(applied: int) -> None:
    """The proven 0/0 and 10/10 cases stay green — equal applied is not worse."""
    guard = evaluate_recovery_regression(
        first_coverage=_cov(0.0),
        recovered_coverage=_cov(0.0),
        first_live_citations=3,
        recovered_live_citations=3,
        first_applied=applied,
        recovered_applied=applied,
    )
    assert guard.non_regressive is True
    assert guard.regressed_dimensions == ()


def test_guard_selects_recovery_that_reduces_applied() -> None:
    """Do-no-harm, not disable-recovery: a genuine repair that LOWERS damage and
    holds coverage/citations is still selected."""
    guard = evaluate_recovery_regression(
        first_coverage=_cov(0.2),
        recovered_coverage=_cov(0.2),
        first_live_citations=4,
        recovered_live_citations=4,
        first_applied=13,
        recovered_applied=6,
    )
    assert guard.non_regressive is True
    assert guard.regressed_dimensions == ()
    telemetry = guard.to_dict()
    assert telemetry["first_applied"] == 13
    assert telemetry["recovered_applied"] == 6


def test_node_callsite_wires_applied_from_annotation_summary(
    report_dir: Path,
) -> None:
    """The finalize callsite must feed the guard each pass's real
    `annotation_summary()['applied']`, so a recovery that adds gap-note damage
    while holding coverage is rejected and the reviewable first pass is kept."""
    del report_dir
    first_draft = "# Report\n\nFirst pass with supported coverage."
    recovered_draft = "# Report\n\nRecovery that re-worsened the critique damage."
    initial = _synthesis(first_draft, step_count=35)
    recovered = _synthesis(recovered_draft, step_count=3)

    with patch.object(
        fnode, "run_codeagent_synthesis", side_effect=[initial, recovered]
    ), patch.object(
        fnode,
        "run_critique",
        # first pass applied=10 (triggers recovery); recovery applied=13 (worse)
        side_effect=[
            _damage_result(10, token_total=30),
            _damage_result(13, token_total=40),
        ],
    ), patch.object(
        fnode,
        "completeness_check",
        return_value={"satisfied": ["A"], "missing": [], "shortfall_score": 0.0},
    ):
        result = asyncio.run(fnode.codeagent_finalize_node(_state(deliverables=["A"])))

    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["status"] == "rejected_regression"
    assert recovery["selected_artifact"] == "first_pass"
    guard = recovery["regression_guard"]
    assert "applied" in guard["regressed_dimensions"]
    assert guard["first_applied"] == 10
    assert guard["recovered_applied"] == 13
    accounting = recovery["attempt_accounting"]
    assert accounting["synthesis_passes"] == 2
    assert accounting["critique_passes"] == 2
    assert accounting["steps_used"] == 38
    assert accounting["synthesis_token_usage"]["total"] == 300
    assert accounting["critique_token_usage"]["total"] == 70
    assert accounting["token_usage"]["total"] == 370
    assert accounting["token_usage_complete"] is True


def test_exhausted_budget_makes_no_recovery_model_call(report_dir: Path) -> None:
    del report_dir
    initial = _synthesis("# Report\n\nOriginal damaged report.", step_count=40)

    with patch.object(
        fnode, "run_codeagent_synthesis", return_value=initial
    ) as synth_mock, patch.object(
        fnode, "run_critique", return_value=_damage_result(10)
    ):
        result = asyncio.run(fnode.codeagent_finalize_node(_state()))

    assert synth_mock.call_count == 1
    recovery = result["synthesis_telemetry"]["recovery"]
    assert recovery["status"] == "budget_exhausted"
    assert recovery["passes"] == 0
    assert recovery["selected_artifact"] == "first_pass"


def test_min_loops_is_not_a_runtime_prompt_instruction() -> None:
    prompt_source = Path(fnode.__file__).with_name("contract_to_prompt.py").read_text()

    assert "min_loops" not in prompt_source
    assert "exactly ONE separate recovery actor" in prompt_source


@pytest.mark.parametrize("missing_content", [False, True])
def test_s93_real_finalize_recovery_ignores_format_only_shortfall(
    report_dir: Path, missing_content: bool,
) -> None:
    """Exercise the real coverage/decision/persistence path, not a mocked score."""
    del report_dir
    draft = (
        "## Findings\nA supported finding [1] [cite:https://example.org/evidence].\n\n"
        "## References\n[1] Evidence — https://example.org/evidence"
    )
    spec = ["Markdown report"]
    if missing_content:
        spec.append("Options comparison table")
    initial = _synthesis(draft, step_count=30)
    recovered = _synthesis(draft, step_count=2)
    initial.completion_control = {"observed_completed_steps": 30}
    recovered.completion_control = {"observed_completed_steps": 2}

    async def live(**kwargs):
        return kwargs["annotations"], LivenessTelemetry()

    with patch.object(
        fnode, "run_codeagent_synthesis", side_effect=[initial, recovered]
    ) as synthesis, patch.object(
        fnode, "_run_liveness_check", side_effect=live
    ), patch.object(fnode, "run_critique", side_effect=lambda text, *a, **k: _healthy_result(text)):

        result = asyncio.run(fnode.codeagent_finalize_node(_state(deliverables=spec)))
    assert synthesis.call_count == (2 if missing_content else 1)
    assert result["report_path"]
    coverage = result["synthesis_telemetry"]["deliverable_coverage"]
    assert coverage["artifact_requirements"][0]["satisfied"] is True
    assert bool(coverage["missing"]) is missing_content
    lifecycle = result["synthesis_telemetry"]["recovery"]
    assert lifecycle["first_pass"]["completion_control"] == initial.completion_control
    if missing_content:
        assert lifecycle["second_pass"]["completion_control"] == recovered.completion_control
