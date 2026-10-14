"""Pre-finalization length feedback protects the paid draft and exact final gate."""
from __future__ import annotations



from deepsearch.agent.completion_control import CompletionController

















def test_preview_failure_is_safe_advisory_and_preserves_paid_checkpoint() -> None:
    def broken(draft: str) -> dict:
        raise RuntimeError("private draft content must not escape")
    controller = CompletionController(40, report_word_budget_check=broken)
    draft = "A finding [cite:https://example.org/]."
    controller.observe_draft(draft)
    assert controller.retained_draft == draft
    assert controller.report_word_budget == {
        "version": 1, "available": False, "reason": "render_preview_failed",
    }
    assert "private" not in repr(controller.telemetry("normal_final_answer"))
