"""Deterministic completion phases within the existing CodeAgent step budget."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .citation_extraction import _CITE_MARKER_RE, extract_citations, is_cited

ASSEMBLY_START_STEP = 30
ASSEMBLY_RETRIEVAL_CALLS = 3
FINALIZATION_RESERVED_STEPS = 2
# S100 m03 — per-site counts of texts with no literal [cite: marker that are
# cited through their numbered references (is_cited), so a run shows the change.
NUMBERED_CITED_EVENTS = ("retained", "salvage_skipped", "recovery_accepted")
# S100 m03 — a refused late retrieval makes no request. STIG-003 reported two
# such refusals as "FETCH FAILED", so the refusal says why and what unblocks it.
RETRIEVAL_REFUSAL_REASONS = {
    "assembly_draft_required": (
        "new retrieval after research requires a saved cited draft. Save the cited report "
        "with scratchpad_write('report_draft', draft), then retry with "
        "gap='<the specific missing evidence>'"
    ),
    "assembly_gap_required": (
        "new retrieval after research requires gap='<the specific missing evidence>' "
        "of at most 200 characters. Retry with that gap"
    ),
    "assembly_retrieval_budget_exhausted": (
        f"all {ASSEMBLY_RETRIEVAL_CALLS} late retrievals are used. Work from the stored "
        "sources and state the remaining gap in the report"
    ),
    "finalization_phase": (
        "retrieval is closed in the last two actions. Finish the report with final_answer"
    ),
}
# S97 m04 — per-check length trajectory kept in telemetry. S96-MEMORY-ACCEPT-02
# made 27 completeness checks and 19 saves in 40 steps; extra checks are
# counted as dropped, never silently lost.
REPORT_WORD_BUDGET_TRAJECTORY_LIMIT = 64
# S98 m05 — at most this many unread cited URLs are named, most-cited first.
UNREAD_CITED_FEEDBACK_LIMIT = 10
# If coverage never completes during research, the feedback still fires with a
# saved cited draft once this many steps remain before assembly (S98-STIG-ACCEPT-01
# kept one deliverable open through step 29, so the coverage trigger never fired).
UNREAD_CITED_FEEDBACK_LEAD_STEPS = 4
# S99 m02 — per-call trajectory of the agent's completeness_check results, as
# requirement indexes only, so a run shows whether a heading remedy was acted on.
COVERAGE_TOOL_TRAJECTORY_LIMIT = 64


@dataclass
class CompletionController:
    """Observe checkpoints and limit new retrieval without discarding paid memory.

    This controller cannot make a model comply or invent a missing draft. It
    records that absence, reserves the remaining actions for assembly, and keeps
    the latest cited checkpoint available to the existing artifact salvage path.
    """

    max_steps: int
    deliverables: Sequence[str] | None = None
    completed_steps: int = 0
    retained_draft: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)
    draft_sha256: str = ""
    checks: int = 0
    retrieval_gaps: list[str] = field(default_factory=list)
    blocked_retrievals: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)
    draft_present_at_assembly: bool | None = None
    report_word_budget_check: Callable[[str], dict[str, Any]] | None = None
    report_word_budget: dict[str, Any] = field(default_factory=dict)
    report_word_budget_checks: list[dict[str, Any]] = field(default_factory=list)
    report_word_budget_check_count: int = 0
    # S98 m05 — exact URLs with a readable fetched body, read when the
    # unread-source feedback fires; the runner binds it to its live sources.
    readable_sources: Callable[[], set[str]] | None = None
    unread_cited_feedback: dict[str, Any] = field(default_factory=dict)
    coverage_tool_checks: list[dict[str, Any]] = field(default_factory=list)
    coverage_tool_check_count: int = 0
    numbered_cited_events: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(NUMBERED_CITED_EVENTS, 0)
    )
    nudge_numbered_only_urls: int = 0
    _budget_observed_sha256: str = field(default="", repr=False)
    _budget_estimate_sha256: str = field(default="", repr=False)
    _cited_sha256: str = field(default="", repr=False)
    _cited: bool = field(default=False, repr=False)

    @property
    def assembly_start(self) -> int:
        return max(1, min(ASSEMBLY_START_STEP, (self.max_steps * 3 + 3) // 4))

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.completed_steps)

    @property
    def phase(self) -> str:
        if self.remaining_steps <= FINALIZATION_RESERVED_STEPS:
            return "finalization"
        if self.completed_steps + 1 >= self.assembly_start:
            return "assembly"
        return "research"

    def observe_draft(self, draft: str | None, *, final: bool = False) -> None:
        """Check each distinct cited checkpoint once, retaining it in memory.

        Re-reading an unchanged saved draft after a step must not replace a
        newer completeness_check estimate for the revision the agent is
        weighing (S97 m04). A final output is re-estimated whenever it differs
        from the draft the current estimate describes.
        """
        if not isinstance(draft, str):
            return
        digest = hashlib.sha256(draft.encode()).hexdigest()
        if digest != (self._budget_estimate_sha256 if final else self._budget_observed_sha256):
            self._budget_observed_sha256 = digest
            self.check_report_word_budget(
                draft, source="final_output" if final else "report_draft",
            )
        if digest == self.draft_sha256:
            return
        # S100 m03: cited as finalize would cite it, decided once per draft.
        if digest != self._cited_sha256:
            self._cited_sha256, self._cited = digest, is_cited(draft)
        if not self._cited:
            return
        # tools.__init__ imports reranking, which reads the runner's authority
        # rules; defer this leaf tool import until module initialization is done.
        from .tools.completeness_check import completeness_check

        self.retained_draft = draft
        self.draft_sha256 = digest
        self.coverage = completeness_check(draft, self.deliverables)
        self.checks += 1
        if "[cite:" not in draft:
            self.numbered_cited_events["retained"] += 1

    def check_report_word_budget(
        self, draft: str | None, *, source: str = "completeness_check",
    ) -> dict[str, Any]:
        """Preview rendered length without gating completion or losing a draft."""
        if self.report_word_budget_check is None or not isinstance(draft, str):
            return {}
        try:
            self.report_word_budget = self.report_word_budget_check(draft)
        except Exception:
            # A preview is advisory. Never turn renderer trouble into lost paid
            # work or leak draft/error content through a controller observation.
            self.report_word_budget = {
                "version": 1, "available": False, "reason": "render_preview_failed",
            }
        digest = hashlib.sha256(draft.encode()).hexdigest()
        self.report_word_budget_check_count += 1
        if len(self.report_word_budget_checks) < REPORT_WORD_BUDGET_TRAJECTORY_LIMIT:
            # Counts only: no draft text, URLs or digests leave the controller.
            # ``step`` is the action in progress; a final output belongs to
            # the last completed action.
            budget = self.report_word_budget
            self.report_word_budget_checks.append({
                "seq": self.report_word_budget_check_count,
                "step": self.completed_steps + (0 if source == "final_output" else 1),
                "source": source,
                "unchanged": digest == self._budget_estimate_sha256,
                "available": bool(budget.get("available")),
                **{key: budget.get(key) for key in (
                    "projected_words", "draft_words",
                    "words_over_maximum", "words_below_minimum",
                )},
            })
        self._budget_estimate_sha256 = digest
        return dict(self.report_word_budget)

    def record_coverage_check(self, draft: str | None, coverage: Mapping[str, Any]) -> None:
        """Keep one agent completeness_check result as indexes and counts.

        ``missing`` and ``heading_remedies`` index the coverage's fixed requirement
        order, artifacts first, then content. No deliverable text, draft text or
        URL leaves the controller; extra checks are counted as dropped.
        """
        self.coverage_tool_check_count += 1
        if len(self.coverage_tool_checks) >= COVERAGE_TOOL_TRAJECTORY_LIMIT:
            return
        requirements = [
            *(coverage.get("artifact_requirements") or []),
            *(coverage.get("content_requirements") or []),
        ]
        missing = [index for index, item in enumerate(requirements) if not item.get("satisfied")]
        # The core gives exactly one remedy per missing requirement, in that order.
        kinds = [remedy.get("kind") for remedy in coverage.get("remedies") or []]
        self.coverage_tool_checks.append({
            "seq": self.coverage_tool_check_count,
            "step": self.completed_steps + 1,
            "draft_chars": len(draft) if isinstance(draft, str) else 0,
            "missing": missing,
            "heading_remedies": [
                index for index, kind in zip(missing, kinds) if kind == "heading"
            ],
        })

    def allow_retrieval(self, gap: str, scratchpad: Mapping[str, str]) -> str | None:
        """Return a local refusal reason or reserve one bounded late fetch/search."""
        self.observe_draft(scratchpad.get("report_draft"))
        if self.phase == "research":
            return None
        reason: str | None = None
        if self.phase == "finalization":
            reason = "finalization_phase"
        elif not self.retained_draft:
            reason = "assembly_draft_required"
        elif not isinstance(gap, str) or not gap.strip() or len(gap) > 200:
            reason = "assembly_gap_required"
        elif len(self.retrieval_gaps) >= ASSEMBLY_RETRIEVAL_CALLS:
            reason = "assembly_retrieval_budget_exhausted"
        if reason:
            self.blocked_retrievals += 1
            return reason
        # An explicit model-declared evidence/content gap, not a support verdict.
        self.retrieval_gaps.append(gap.strip())
        return None

    def after_step(self, step: Any, scratchpad: Mapping[str, str]) -> None:
        """Use the supported ActionStep callback before the next model request."""
        number = getattr(step, "step_number", 0)
        if not isinstance(number, int) or not 1 <= number <= self.max_steps:
            return  # Synthetic max-step fallback is not another research action.
        # Never turn an empty failed generation into apparent paid action memory
        # by adding controller-only observations; the S92 retry guard reads it.
        if not any(getattr(step, name, None) for name in (
            "model_output", "model_output_message", "tool_calls", "observations", "action_output"
        )):
            return
        self.completed_steps = number
        self.observe_draft(scratchpad.get("report_draft"))
        phase = self.phase
        if not self.transitions or self.transitions[-1]["phase"] != phase:
            self.transitions.append({"phase": phase, "next_step": number + 1})
            if phase != "research" and self.draft_present_at_assembly is None:
                self.draft_present_at_assembly = bool(self.retained_draft)
        if getattr(step, "is_final_answer", False):
            self.observe_draft(getattr(step, "action_output", None), final=True)
            return
        step.observations = (
            (getattr(step, "observations", None) or "") + "\n" + self.notice()
            + self._unread_cited_feedback()
        )

    def _unread_cited_feedback(self) -> str:
        """Once, while retrieval is open, name cited URLs with no readable body.

        In research it fires the first time the retained cited draft covers
        every deliverable, or is first saved when the mission has none, and
        otherwise with a saved cited draft once research is about to end. An
        agent that drafts only in assembly hears at its first assembly step
        with a cited draft while late retrievals remain (decision 1010). It is
        feedback, not a gate: no provider call, no refusal. The message names
        URLs that are already the draft's own citation keys; telemetry keeps
        counts only.
        """
        if (self.unread_cited_feedback or self.phase == "finalization"
                or self.readable_sources is None or not self.retained_draft):
            return ""
        late_retrievals = max(0, ASSEMBLY_RETRIEVAL_CALLS - len(self.retrieval_gaps))
        if self.phase == "assembly":
            if not late_retrievals:
                return ""
            trigger = "assembly"
        else:
            trigger = "coverage_complete" if self.deliverables else "no_deliverables"
            if self.deliverables and (
                self.coverage.get("assessment_status") != "assessed"
                or self.coverage.get("missing")
            ):
                if self.completed_steps + UNREAD_CITED_FEEDBACK_LEAD_STEPS < self.assembly_start:
                    return ""
                trigger = "research_ending"
        cited = Counter(m.group(1) for m in _CITE_MARKER_RE.finditer(self.retained_draft))
        literal = set(cited)
        try:
            _, annotations = extract_citations(self.retained_draft)
        except Exception:
            # Advisory only: the literal markers still name their URLs.
            annotations = []
        # S100 m03: numbered citations count too; every literal marker already did.
        cited.update(a.url for a in annotations if a._extraction_mode == "numbered")
        try:
            readable = self.readable_sources()
        except Exception:
            # Advisory only: a failed lookup must never break a paid step.
            self.unread_cited_feedback = {"version": 1, "fired": False, "available": False}
            return ""
        unread = [url for url, _ in cited.most_common() if url not in readable]
        named = unread[:UNREAD_CITED_FEEDBACK_LIMIT]
        self.nudge_numbered_only_urls = sum(url not in literal for url in named)
        self.unread_cited_feedback = {
            "version": 1, "fired": True, "step": self.completed_steps, "trigger": trigger,
            "cited_urls": len(cited), "unread_cited_urls": len(unread), "named": len(named),
        }
        if not named:
            return ""
        more = len(unread) - len(named)
        if trigger == "assembly":
            action = (
                f"{late_retrievals} late retrievals remain: spend them on the most "
                "important of these with source_fetch(url, gap='unread cited source'), "
                "and state where you cite the rest that only a snippet or abstract was "
                "available: "
            )
        else:
            action = (
                "While research is still open, read them with source_fetch, or state "
                "where you cite them that only a snippet or abstract was available: "
            )
        return (
            f"\n[Completion controller] {len(unread)} of {len(cited)} cited URLs have no "
            "readable fetched body. " + action + ", ".join(named)
            + (f" (+{more} more)." if more else ".")
        )

    def refusal_message(self, reason: str) -> str:
        """Say that a refused retrieval made no request, why, and what unblocks it."""
        return (
            "No request was made, so the URL was not tried and this is not a fetch "
            f"failure: {RETRIEVAL_REFUSAL_REASONS.get(reason, reason)}."
        )

    def notice(self) -> str:
        """Return bounded controller guidance, never report/source content."""
        base = (
            f"[Completion controller] Phase={self.phase}; completed={self.completed_steps}/"
            f"{self.max_steps}; remaining={self.remaining_steps}. "
            f"Cited checkpoint={'available' if self.retained_draft else 'absent'}; "
            f"coverage={self.coverage.get('assessment_status', 'not_checked')}. "
        )
        if self.report_word_budget.get("available"):
            budget = self.report_word_budget
            over = budget["words_over_maximum"]
            below = budget["words_below_minimum"]
            # S97 m04 — direct a revision only when a bound is missed. An
            # unconditional "revise and check again" gave an in-bounds draft
            # no stop condition on every step.
            if over or below:
                action = (
                    f"{'Shorten' if over else 'Expand'} the draft by at least "
                    f"{over or below} words, save it, and check the revision "
                    "before final_answer. "
                )
            else:
                action = (
                    "Length is within the authored bounds; no repeat check is "
                    "needed unless the draft changes. "
                )
            base += (
                f"Rendered report estimate={budget['projected_words']} words "
                "(latest checked draft); "
                f"authored minimum={budget['min_words']}, maximum={budget['max_words']}; "
                f"over maximum={over}, below minimum={below}. "
                + action
                + "The estimate includes wrapper text, excludes later critique "
                "edits, and does not replace final persisted validation. "
            )
        if self.phase == "research":
            return base + (
                f"Begin assembly by step {self.assembly_start}. Save report_draft as you work."
            )
        return base + (
            "Assemble a complete cited report now, save it with scratchpad_write('report_draft', "
            "draft), check completeness, and call final_answer within this budget. Use honest gap "
            "notes for missing evidence. Stored source passages remain readable. New retrieval "
            "requires a saved cited draft and gap='specific missing evidence/content'; at most "
            f"{max(0, ASSEMBLY_RETRIEVAL_CALLS - len(self.retrieval_gaps))} such calls remain, "
            "and none are allowed in the last two actions."
        )

    def telemetry(self, completion_kind: str) -> dict[str, Any]:
        """Return bounded, content-free lifecycle facts; absence stays explicit."""
        return {
            "policy_version": 1,
            "max_steps": self.max_steps,
            "assembly_start_step": self.assembly_start,
            "observed_completed_steps": self.completed_steps,
            "remaining_steps": self.remaining_steps,
            "phase": self.phase,
            "transitions": list(self.transitions),
            "draft_present_at_assembly": self.draft_present_at_assembly,
            "cited_draft_available": bool(self.retained_draft),
            "draft_chars": len(self.retained_draft),
            "draft_sha256": self.draft_sha256 or None,
            "coverage_check_status": "checked" if self.checks else "not_checked",
            "coverage_checks": self.checks,
            "coverage": dict(self.coverage),
            "late_retrieval_calls": len(self.retrieval_gaps),
            "declared_gaps": list(self.retrieval_gaps),
            "blocked_retrievals": self.blocked_retrievals,
            "completion_kind": completion_kind,
            "unread_cited_feedback": (
                dict(self.unread_cited_feedback) or {"version": 1, "fired": False}
            ),
            "numbered_cited_events": dict(self.numbered_cited_events),
            "nudge_numbered_only_urls": self.nudge_numbered_only_urls,
            "report_word_budget": dict(self.report_word_budget),
            "report_word_budget_trajectory": {
                "version": 1,
                "limit": REPORT_WORD_BUDGET_TRAJECTORY_LIMIT,
                "total": self.report_word_budget_check_count,
                "dropped": self.report_word_budget_check_count
                - len(self.report_word_budget_checks),
                "capped": self.report_word_budget_check_count
                > len(self.report_word_budget_checks),
                "checks": [dict(check) for check in self.report_word_budget_checks],
            },
            "coverage_tool_trajectory": {
                "version": 1,
                "limit": COVERAGE_TOOL_TRAJECTORY_LIMIT,
                "total": self.coverage_tool_check_count,
                "dropped": self.coverage_tool_check_count - len(self.coverage_tool_checks),
                "checks": [
                    {**check, "missing": list(check["missing"]),
                     "heading_remedies": list(check["heading_remedies"])}
                    for check in self.coverage_tool_checks
                ],
            },
        }
