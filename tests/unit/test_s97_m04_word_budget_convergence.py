"""S97 m04: length feedback must describe the draft the agent just checked.

S96-MEMORY-ACCEPT-02 called completeness_check 12 times in a row without
saving a new report_draft, then needed 40/40 steps and 2.8M tokens. The
controller re-estimated the unchanged saved draft after every step, so its
appended notice could contradict the tool result for the revision in hand,
and it told the agent to revise and recheck even when the estimate was in
bounds. These offline tests pin the feedback contract, not model compliance.
"""
from __future__ import annotations

from typing import Any

from smolagents.memory import ActionStep, Timing

from deepsearch.agent.completion_control import (
    CompletionController,
)

# The paid S96-MEMORY-ACCEPT-01 actor draft: 958 rendered words against 900.
# A real-shaped revision: the same draft without its Reasoning Summary section.




def _fixed(projected: int) -> Any:
    """A bounded estimate independent of the renderer, for wording checks."""
    def check(draft: str) -> dict[str, Any]:
        return {
            "version": 1, "available": True, "projected_words": projected,
            "draft_words": projected - 100, "min_words": 600, "max_words": 900,
            "words_over_maximum": max(0, projected - 900),
            "words_below_minimum": max(0, 600 - projected),
        }
    return check


def _notice(step: ActionStep) -> str:
    return (step.observations or "").split("[Completion controller]")[-1]










def test_in_bounds_estimate_gives_a_stop_condition_not_a_revision_order() -> None:
    within = CompletionController(40, report_word_budget_check=_fixed(900))
    within.observe_draft("draft")
    step = ActionStep(step_number=5, timing=Timing(start_time=0), model_output="x")
    within.after_step(step, {})
    notice = _notice(step)
    assert "over maximum=0, below minimum=0" in notice
    assert "within the authored bounds" in notice
    assert "no repeat check is needed unless the draft changes" in notice
    assert "Revise" not in notice and "Shorten" not in notice and "Expand" not in notice

    for projected, action in ((958, "Shorten the draft by at least 58 words"),
                              (550, "Expand the draft by at least 50 words")):
        control = CompletionController(40, report_word_budget_check=_fixed(projected))
        control.observe_draft("draft")
        step = ActionStep(step_number=5, timing=Timing(start_time=0), model_output="x")
        control.after_step(step, {})
        assert action in _notice(step)
        assert "within the authored bounds" not in _notice(step)
