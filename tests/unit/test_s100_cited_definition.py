"""S100-m03: one definition of "cited" at runtime, and refusals that say so.

Finalize has accepted numbered citations since S96, but five runtime sites still
tested for the literal ``[cite:``. A numbered-only draft was never retained, so
the controller ran no coverage check, sent no unread-source nudge and refused
late retrieval as if no draft existed. A numbered-only recovery answer was
rejected as uncited, and a numbered-only final output was replaced by an older
``[cite:`` draft, which made a max-steps run catastrophic. ``is_cited`` is the
one rule: the literal marker, or any annotation finalize would extract. Refused
fetches and probes also said nothing, and STIG-003 reported its refused
Gloaguen fetches as "FETCH FAILED"; the refusal now says no request was made.
"""

from __future__ import annotations

from typing import Any

from deepsearch.agent import citation_extraction, completion_control
from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.citation_extraction import extract_citations, is_cited
from deepsearch.agent.completion_control import CompletionController
from deepsearch.agent.outcome import MissionOutcome, evaluate_outcome_gate
from deepsearch.agent.tools import source_fetch as fetch_module
from deepsearch.agent.tools import url_liveness as liveness_module
from tests.unit.test_s93_completion_control import _code, _LifecycleModel, _step

CITED = "## Findings\nA sourced finding [cite:https://example.org/evidence]."
NUMBERED = ("## Findings\nA sourced finding [1]. A second finding [2].\n\n## References\n"
            "[1] Evidence — https://example.org/evidence\n[2] Other — https://example.org/other\n")
UNRESOLVED = "## Findings\nA sourced finding [1]. A second finding [2].\n"


def test_cited_means_a_marker_or_a_citation_finalize_would_extract(monkeypatch) -> None:
    assert is_cited(CITED) and is_cited(NUMBERED)
    assert not is_cited(UNRESOLVED) and not is_cited("") and not is_cited(None)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("extractor bug")

    # A paid step never fails on the check: the literal test's answer stands.
    monkeypatch.setattr(citation_extraction, "extract_citations", broken)
    assert not is_cited(NUMBERED)
    assert is_cited(CITED)


def test_a_numbered_draft_is_retained_once_and_checked(monkeypatch) -> None:
    calls: list[str] = []
    real = citation_extraction.extract_citations
    monkeypatch.setattr(citation_extraction, "extract_citations",
                        lambda text, **kw: calls.append(text) or real(text, **kw))
    control = CompletionController(40, ["Markdown report"])
    for number in (1, 2, 3):
        control.after_step(_step(number), {"report_draft": NUMBERED})
    assert control.retained_draft == NUMBERED and control.checks == 1
    assert control.coverage["missing"] == []
    # Extraction runs only when the draft's digest changes.
    assert calls == [NUMBERED]
    control.after_step(_step(4), {"report_draft": UNRESOLVED})
    control.after_step(_step(5), {"report_draft": UNRESOLVED})
    assert calls == [NUMBERED, UNRESOLVED] and control.retained_draft == NUMBERED
    telemetry = control.telemetry("pending")
    assert telemetry["numbered_cited_events"] == {
        "retained": 1, "salvage_skipped": 0, "recovery_accepted": 0}
    assert telemetry["cited_draft_available"] and telemetry["coverage_check_status"] == "checked"


def test_a_marker_draft_counts_no_numbered_event() -> None:
    control = CompletionController(40)
    control.observe_draft(CITED)
    assert control.retained_draft == CITED
    assert control.telemetry("pending")["numbered_cited_events"]["retained"] == 0


def test_a_saved_numbered_draft_unlocks_a_late_retrieval() -> None:
    control = CompletionController(40)
    control.completed_steps = 34
    assert control.phase == "assembly"
    assert control.allow_retrieval("missing evidence", {"report_draft": UNRESOLVED}) == (
        "assembly_draft_required")
    assert control.allow_retrieval("missing evidence", {"report_draft": NUMBERED}) is None


def test_the_nudge_counts_urls_cited_only_by_number() -> None:
    mixed = NUMBERED.replace("A second finding [2].", "A second finding [2]. A third "
                             "[cite:https://example.org/inline].")
    control = CompletionController(40, readable_sources=lambda: {"https://example.org/evidence"})
    step = _step(5)
    step.observations = ""
    control.after_step(step, {"report_draft": mixed})
    feedback = control.telemetry("pending")["unread_cited_feedback"]
    assert (feedback["cited_urls"], feedback["unread_cited_urls"], feedback["named"]) == (3, 2, 2)
    assert "https://example.org/other" in step.observations
    assert "https://example.org/inline" in step.observations
    # Named, and found only through a numbered marker: the inline URL is not.
    assert control.telemetry("pending")["nudge_numbered_only_urls"] == 1


def test_a_claim_cited_both_ways_is_counted_once() -> None:
    # [1] resolves to the inline marker's URL at the same claim, so reconciliation
    # keeps one citation: A counts once, and B, cited twice by number, leads.
    draft = ("## Findings\nClaim one [1] [cite:https://example.org/a]. Claim two [2]. "
             "Claim three [2].\n\n## References\n[1] A — https://example.org/a\n"
             "[2] B — https://example.org/b\n")
    control = CompletionController(40, readable_sources=lambda: set())
    step = _step(5)
    control.after_step(step, {"report_draft": draft})
    named = step.observations.rsplit(": ", 1)[1].rstrip(".").split(", ")
    assert named == ["https://example.org/b", "https://example.org/a"]
    assert control.telemetry("pending")["nudge_numbered_only_urls"] == 1


def test_the_nudge_falls_back_to_literal_markers_if_extraction_fails(monkeypatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("extractor bug")

    monkeypatch.setattr(completion_control, "extract_citations", broken)
    mixed = NUMBERED.replace("A second finding [2].", "A second finding [2]. A third "
                             "[cite:https://example.org/inline].")
    control = CompletionController(40, readable_sources=lambda: set())
    step = _step(5)
    control.after_step(step, {"report_draft": mixed})
    # The step is not broken; only the literal marker counts, as before S100.
    feedback = control.telemetry("pending")["unread_cited_feedback"]
    assert (feedback["cited_urls"], feedback["named"]) == (1, 1)
    assert step.observations.endswith("https://example.org/inline.")
    assert control.telemetry("pending")["nudge_numbered_only_urls"] == 0


def test_refused_fetch_and_probe_say_no_request_was_made(monkeypatch) -> None:
    def network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a refused call must not reach the network")

    monkeypatch.setattr(fetch_module, "source_fetch", network)
    monkeypatch.setattr(liveness_module, "url_liveness", network)
    control = CompletionController(40)
    control.after_step(_step(29), {})
    tools = runner.build_codeagent_tools(completion_controller=control)
    by_name = {getattr(tool, "name", None) or tool.__name__: tool for tool in tools.tools}
    fetched = by_name["source_fetch"]("https://example.org/new", gap="missing evidence")
    probed = by_name["url_liveness"]("https://example.org/new", gap="missing evidence")
    for refusal in (fetched, probed):
        assert refusal["error"] == "assembly_draft_required"
        message = refusal["message"]
        assert message.startswith("No request was made")
        assert "not a fetch failure" in message
        assert "scratchpad_write('report_draft'" in message and "gap=" in message
    records = tools.tool_outcomes()["records"]
    assert [(r["tool"], r["cited_draft_available"], r["network_requested"])
            for r in records] == [("source_fetch", False, False), ("url_liveness", False, False)]


def test_search_refusals_record_whether_a_cited_draft_existed(monkeypatch) -> None:
    control = CompletionController(40)
    control.after_step(_step(29), {"report_draft": NUMBERED})
    tools = runner.build_codeagent_tools(completion_controller=control)
    by_name = {getattr(tool, "name", None) or tool.__name__: tool for tool in tools.tools}
    for name in ("web_search", "scholar_search"):
        refusal = by_name[name]("more")  # no gap: refused before any request
        assert refusal[0]["error"] == "assembly_gap_required"
    records = tools.tool_outcomes()["records"]
    assert [(r["tool"], r["cited_draft_available"]) for r in records] == [
        ("web_search", True), ("scholar_search", True)]


def test_generation_recovery_accepts_a_numbered_provider_stop_answer(monkeypatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    for answer, accepted in ((NUMBERED, True), (UNRESOLVED, False)):
        model = _LifecycleModel([
            _code(f"scratchpad_write('report_draft', {CITED!r})"),
            RuntimeError("provider unavailable"), answer,
        ])
        result = runner.run_codeagent_synthesis(
            system_instructions="Research", user_directive="Task", model=model, max_steps=40,
        )
        # The controller retained the earlier [cite: draft either way.
        assert result.completion_control["cited_draft_available"]
        events = result.completion_control["numbered_cited_events"]
        if accepted:
            assert result.generation_recovery_result == "success"
            assert result.generation_error_recovered is True
            assert result.recovered_from_scratchpad is False
            assert result.output_markdown == NUMBERED
            # Accepted at recovery, kept at salvage, retained as the final output.
            assert events == {"retained": 1, "salvage_skipped": 1, "recovery_accepted": 1}
        else:
            # No References section: nothing resolves, so it stays uncited and
            # the retained cited draft is salvaged, as before S100.
            assert result.generation_recovery_result == "uncited"
            assert result.generation_error_recovered is False
            assert result.recovered_from_scratchpad is True
            assert result.output_markdown == CITED
            assert events == {"retained": 0, "salvage_skipped": 0, "recovery_accepted": 0}


def gate(result: Any, *, recovered_from_scratchpad: bool) -> Any:
    _, citations = extract_citations(result.output_markdown)
    telemetry = {
        "hit_max_steps": result.hit_max_steps,
        "self_terminated": result.self_terminated,
        "recovered_from_scratchpad": recovered_from_scratchpad,
        "max_steps_fallback_complete": result.max_steps_fallback_complete,
        "citation_telemetry": {"citations_extracted": len(citations)},
        "deliverable_coverage": {"shortfall_score": 0.0, "satisfied": ["Findings"],
                                 "missing": []},
        "liveness_telemetry": {"checked": 2, "dead": 0, "orphan": 0, "live": 2},
    }
    return evaluate_outcome_gate(telemetry, report_word_count=600)


def max_steps_run(fallback: str, *, saved: str | None = CITED) -> Any:
    first = (_code(f"scratchpad_write('report_draft', {saved!r})") if saved
             else _code("print('first action')"))
    model = _LifecycleModel([first, _code("print('second action')"), fallback])
    return runner.run_codeagent_synthesis(
        system_instructions="Research", user_directive="Task", model=model, max_steps=2,
        deliverable_spec=["Markdown report"],
    )


def test_a_numbered_max_steps_fallback_is_kept_not_salvaged(monkeypatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    result = max_steps_run(NUMBERED)
    assert result.hit_max_steps and result.max_steps_fallback_complete
    assert result.output_markdown == NUMBERED
    assert result.recovered_from_scratchpad is False
    assert result.completion_control["numbered_cited_events"]["salvage_skipped"] == 1
    kept = gate(result, recovered_from_scratchpad=result.recovered_from_scratchpad)
    assert (kept.outcome, kept.routing_reason) == (MissionOutcome.COMPLETE, "")
    assert kept.delivery_quality != "failed"
    assert "hit_max_steps_without_final_answer" in kept.quality_gate_failures
    # Before S100 (the literal rule at every site) the same inputs salvaged the
    # older draft, and the gate failed the run.
    monkeypatch.setattr(runner, "is_cited", runner._has_cite_markers)
    monkeypatch.setattr(completion_control, "is_cited", runner._has_cite_markers)
    before = max_steps_run(NUMBERED)
    assert (before.output_markdown, before.recovered_from_scratchpad) == (CITED, True)
    salvaged = gate(before, recovered_from_scratchpad=before.recovered_from_scratchpad)
    assert salvaged.outcome is MissionOutcome.VALIDATION_FAILED


def test_a_code_wrapped_numbered_fallback_is_still_salvaged(monkeypatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    wrapped = f"<code>\nreport = '''{NUMBERED}'''\nfinal_answer(report)\n</code>"
    assert is_cited(wrapped)
    result = max_steps_run(wrapped)
    # Code is not a report: the clean saved draft replaces it, as before S100.
    assert (result.output_markdown, result.recovered_from_scratchpad) == (CITED, True)
    assert result.completion_control["numbered_cited_events"]["salvage_skipped"] == 0


def test_a_numbered_fallback_with_nothing_to_salvage_counts_no_skip(monkeypatch) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "offline")
    result = max_steps_run(NUMBERED, saved=None)
    assert (result.output_markdown, result.recovered_from_scratchpad) == (NUMBERED, False)
    assert result.completion_control["numbered_cited_events"]["salvage_skipped"] == 0


def test_salvage_chooses_the_longest_cited_draft_numbered_ones_included() -> None:
    longer = NUMBERED + "\nA longer numbered draft with more findings.\n"
    scratchpad = {"notes": CITED, "assembled": longer, "uncited": "x " * 500}
    assert runner._recover_stranded_draft(scratchpad) == longer
    assert runner._recover_stranded_draft({"report_draft": NUMBERED, "notes": longer}) == NUMBERED
    assert runner._recover_stranded_draft({"notes": UNRESOLVED}) is None
