"""TraceLab Synthesis and ResearchStatement generation from AgentState.

This module aggregates findings, reflection results, and objectives into
TraceLab-compatible objects:
- Synthesis: key_insights, recommendations, next_steps
- ResearchStatement: topic, objective, scope

Quality gate requirements:
Synthesis:
- key_insights has at least 1 item with >= 40 characters
- recommendations has at least 1 item
- next_steps has at least 1 item

ResearchStatement:
- topic is non-empty
- objective is non-empty
- scope is non-empty
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set

from pydantic import BaseModel, Field

from deepsearch.agent.concept_clustering import summarize_state_findings
from deepsearch.agent.state import AgentState
from deepsearch.mission.contract import resolve_research_objectives

# Minimum characters for a valid key insight (TraceLab gate)
MIN_INSIGHT_LENGTH = 40


class ResearchStatement(BaseModel):
    """TraceLab-compatible ResearchStatement object.

    Defines the research mission context: what is being researched (topic),
    why (objective), and what boundaries apply (scope). All three required
    fields must be non-empty for the research_statement quality gate.
    """

    topic: str = Field(
        ...,
        min_length=1,
        description="Research topic (what is being investigated)",
    )
    objective: str = Field(
        ...,
        min_length=1,
        description="Research objective (why this research matters)",
    )
    scope: str = Field(
        ...,
        min_length=1,
        description="Research scope (boundaries and limitations)",
    )
    audience: Optional[str] = Field(
        default=None,
        description="Intended audience for the research",
    )
    methodology: Optional[str] = Field(
        default=None,
        description="Research methodology description",
    )
    success_metrics: List[str] = Field(
        default_factory=list,
        description="Metrics for evaluating research success",
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Identified risks or limitations",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()

    def passes_quality_gate(self) -> bool:
        """Check if this ResearchStatement passes TraceLab's research_statement gate.

        Returns:
            True if topic, objective, and scope are all non-empty.
        """
        return bool(self.topic.strip() and self.objective.strip() and self.scope.strip())


class Synthesis(BaseModel):
    """TraceLab-compatible Synthesis object.

    This model matches the TraceLab Synthesis schema for mission protocol
    ingestion. All list fields are validated for non-empty requirements.
    """

    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights from research (required: >=1 item, >=40 chars each)",
    )
    surprising_findings: List[str] = Field(
        default_factory=list,
        description="Unexpected or surprising discoveries",
    )
    contradictory_information: List[str] = Field(
        default_factory=list,
        description="Conflicting information found during research",
    )
    contradiction_resolutions: List[str] = Field(
        default_factory=list,
        description="Resolutions for contradictions (required if contradictions exist)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations (required: >=1 item)",
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Suggested next actions (required: >=1 item)",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()

    def passes_quality_gate(self) -> bool:
        """Check if this Synthesis passes TraceLab's synthesis_quality gate.

        Returns:
            True if all quality requirements are met.
        """
        if not self.key_insights:
            return False
        if len(self.key_insights[0]) < MIN_INSIGHT_LENGTH:
            return False
        if not self.recommendations:
            return False
        if not self.next_steps:
            return False
        return True


def generate_synthesis(state: AgentState) -> Synthesis:
    """Generate TraceLab Synthesis from AgentState.

    Transforms the research findings, reflection results, and objectives into
    a structured Synthesis that meets TraceLab quality gate requirements.

    Args:
        state: AgentState with findings, reflection_result, reflection_scores,
               mission_objectives.

    Returns:
        Synthesis object meeting TraceLab quality gate requirements.

    Example:
        >>> state = {
        ...     "mission_id": "TEST-001",
        ...     "findings": ["Finding one with enough characters to pass the gate"],
        ...     "reflection_result": {"decision": "finalize", "reasoning": "Complete"},
        ...     "reflection_scores": {"overall_coverage_score": 0.85},
        ...     "mission_objectives": ["Document API patterns"],
        ... }
        >>> synthesis = generate_synthesis(state)
        >>> synthesis.passes_quality_gate()
        True
    """
    findings = list(summarize_state_findings(state).get("display_findings") or [])
    reflection_result = state.get("reflection_result") or {}
    reflection_scores = state.get("reflection_scores") or {}
    objectives = resolve_research_objectives(
        state.get("mission_contract"),
        fallback=state.get("mission_objectives") or [],
    )

    synthesis_telemetry = state.get("synthesis_telemetry") or {}

    # Select key insights from findings (S75 m01: 5-cap removed; emit all
    # unique findings that pass the length gate so the legacy hard-cap stops
    # truncating CodeAgent-produced synthesis output).
    key_insights = _select_key_insights(findings)

    # Ensure at least 1 insight (gate requirement). S84 m04 truth repair:
    # the fallback now reads LIVE gate output — the pre-m04 boilerplate
    # interpolated dead reflection_scores and shipped 'Coverage score: 0%'
    # on every production mission.
    if not key_insights:
        key_insights = [_gate_derived_insight(state, synthesis_telemetry, objectives)]

    # Derive recommendations
    recommendations = _derive_recommendations(
        reflection_result,
        reflection_scores,
        synthesis_telemetry=synthesis_telemetry,
    )

    # Derive next steps from coverage gaps
    criteria_coverage = reflection_result.get("criteria_coverage") or {}
    next_steps = _derive_next_steps(
        objectives,
        criteria_coverage,
        deliverable_coverage=synthesis_telemetry.get("deliverable_coverage"),
    )

    return Synthesis(
        key_insights=key_insights,
        surprising_findings=[],  # Future: detect surprising patterns
        contradictory_information=[],  # Future: detect conflicts
        contradiction_resolutions=[],
        recommendations=recommendations,
        next_steps=next_steps,
    )


def _select_key_insights(
    findings: List[str],
) -> List[str]:
    """Select key insights from findings.

    Filters for length >= MIN_INSIGHT_LENGTH and deduplicates by prefix.

    S75 m01: the legacy ``max_count`` 5-cap was removed; all unique findings
    that pass the length gate are emitted. The cap pre-dated the CodeAgent
    rebuild and truncated synthesis output to 5 items regardless of mission
    breadth.

    Args:
        findings: List of finding strings from research.

    Returns:
        List of selected insights meeting length requirements.
    """
    if not findings:
        return []

    # Deduplicate and filter
    seen: Set[str] = set()
    unique: List[str] = []

    for finding in findings:
        text = finding.strip()

        # Skip too short
        if len(text) < MIN_INSIGHT_LENGTH:
            continue

        # Skip duplicates (by first 50 chars, case-insensitive)
        key = text[:50].lower()
        if key in seen:
            continue
        seen.add(key)

        unique.append(text)

    return unique


def _gate_derived_insight(
    state: Mapping[str, Any],
    synthesis_telemetry: Dict[str, Any],
    objectives: List[str],
) -> str:
    """S84 m04 — honest fallback insight from LIVE gate output.

    The pre-m04 fallback interpolated dead ``reflection_scores`` (no writer
    on the post-S64 codeagent path) and shipped the constant lie
    'Coverage score: 0%' on every production mission.
    """
    mission_id = state.get("mission_id", "research")
    coverage = synthesis_telemetry.get("deliverable_coverage") or {}
    satisfied = list(coverage.get("satisfied") or [])
    missing = list(coverage.get("missing") or [])
    citations = int(
        (synthesis_telemetry.get("citation_telemetry") or {}).get(
            "citations_extracted"
        )
        or 0
    )
    words = int(synthesis_telemetry.get("report_word_count") or 0)
    if satisfied or missing:
        return (
            f"Report for mission {mission_id} covers {len(satisfied)} of "
            f"{len(satisfied) + len(missing)} deliverables, backed by "
            f"{citations} citations across {words} words."
        )
    return (
        f"Report for mission {mission_id} shipped with {citations} citations "
        f"across {words} words; deliverable coverage was not evaluated "
        f"({len(objectives)} objectives declared)."
    )


def _derive_recommendations(
    reflection_result: Dict[str, Any],
    reflection_scores: Dict[str, Any],
    synthesis_telemetry: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate recommendations from live gate output (S84 m04).

    Sources recommendations from:
    1. Reflection reasoning text (live only on legacy reflection fixtures)
    2. LIVE deliverable-coverage misses + delivery-quality tier
    3. Honest defaults

    The pre-m04 coverage branch read dead ``reflection_scores`` and emitted
    'Consider additional research to improve coverage (current: 0%, ...)'
    on every production mission regardless of outcome — deleted.

    Args:
        reflection_result: Dictionary with decision, reasoning, criteria_coverage.
        reflection_scores: Dictionary with overall_coverage_score, is_mission_complete.
        synthesis_telemetry: Live pipeline telemetry (deliverable_coverage,
            delivery_quality); None/empty on legacy states.

    Returns:
        List of recommendation strings (at least 1 item guaranteed).
    """
    telemetry = synthesis_telemetry or {}
    recommendations: List[str] = []

    # From reflection reasoning
    reasoning = reflection_result.get("reasoning", "")
    if reasoning and len(reasoning) > 20:
        # Truncate to reasonable length for recommendation
        truncated = reasoning[:200]
        if len(reasoning) > 200:
            truncated = truncated.rsplit(" ", 1)[0] + "..."
        recommendations.append(f"Based on analysis: {truncated}")

    # From LIVE deliverable coverage — only when the signal exists.
    coverage = telemetry.get("deliverable_coverage") or {}
    missing = [str(m) for m in (coverage.get("missing") or []) if str(m).strip()]
    if missing:
        named = ", ".join(missing[:3])
        suffix = ", ..." if len(missing) > 3 else ""
        recommendations.append(
            f"Address the unsatisfied deliverable(s): {named}{suffix}"
        )

    # Default if empty — keyed on the live delivery tier when present,
    # else on explicit legacy reflection data; never on dead defaults.
    if not recommendations:
        delivery_quality = telemetry.get("delivery_quality")
        if delivery_quality == "clean":
            recommendations.append(
                "Research objectives achieved. Proceed with implementation."
            )
        elif isinstance(delivery_quality, str) and delivery_quality:
            recommendations.append(
                "Review the flagged quality-gate findings before adopting the report."
            )
        elif reflection_scores.get("is_mission_complete", False):
            recommendations.append(
                "Research objectives achieved. Proceed with implementation."
            )
        elif reflection_scores:
            recommendations.append(
                "Continue research to address remaining gaps."
            )
        else:
            recommendations.append(
                "Review findings and prepare final deliverables."
            )

    return recommendations


def _derive_next_steps(
    objectives: List[str],
    criteria_coverage: Dict[str, str],
    deliverable_coverage: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate next steps from LIVE uncovered deliverables (S84 m04).

    Priority: (1) live deliverable_coverage.missing — the completeness check
    over the shipped report; (2) explicit legacy criteria_coverage entries;
    (3) the honest default. The pre-m04 behavior defaulted every objective
    to "not_addressed" when criteria_coverage was empty (its only writer is
    retired), emitting 'Complete research: <objective>' for up to 5 fully
    satisfied objectives on every production mission — deleted.

    Args:
        objectives: List of mission objective strings.
        criteria_coverage: Dictionary mapping OBJ-N to coverage status
                          (complete, partial, not_addressed).
        deliverable_coverage: Live completeness_check output
                          ({satisfied, missing, shortfall_score}) or None.

    Returns:
        List of next step strings (at least 1 item guaranteed, max 5).
    """
    next_steps: List[str] = []

    live = deliverable_coverage or {}
    live_missing = [str(m) for m in (live.get("missing") or []) if str(m).strip()]
    if live.get("satisfied") or live_missing:
        # Live signal present — derive from what the shipped report misses.
        for name in live_missing:
            next_steps.append(f"Address unsatisfied deliverable: {name}")
    elif criteria_coverage:
        # Explicit legacy reflection data — honor it verbatim.
        for idx, objective in enumerate(objectives):
            obj_id = f"OBJ-{idx + 1}"
            coverage = criteria_coverage.get(obj_id, "not_addressed")
            if coverage in ("partial", "not_addressed"):
                next_steps.append(f"Complete research: {objective}")

    # Default if all complete or no coverage signal exists.
    if not next_steps:
        next_steps.append("Review findings and prepare final deliverables")

    return next_steps[:5]  # Limit to 5


def generate_research_statement(state: AgentState) -> ResearchStatement:
    """Generate TraceLab ResearchStatement from mission context.

    Transforms the AgentState mission_context, mission_objectives, and mission_id
    into a ResearchStatement that meets TraceLab quality gate requirements.

    The function derives required fields using fallback chains:
    - topic: context.title → objectives[0] → mission_id
    - objective: context.objective → context.summary → objectives
    - scope: context.focus → context.background → default

    Args:
        state: AgentState with mission_context, mission_objectives, mission_id.

    Returns:
        ResearchStatement object meeting TraceLab quality gate requirements.

    Example:
        >>> state = {
        ...     "mission_id": "DRM.0.1",
        ...     "mission_context": {
        ...         "title": "Gemini Deep Search Research",
        ...         "objective": "Investigate CLI capabilities",
        ...         "focus": "macOS environment",
        ...     },
        ...     "mission_objectives": ["Catalog CLI commands"],
        ... }
        >>> rs = generate_research_statement(state)
        >>> rs.passes_quality_gate()
        True
    """
    context = state.get("mission_context") or {}
    objectives = resolve_research_objectives(
        state.get("mission_contract"),
        fallback=state.get("mission_objectives") or [],
    )
    mission_id = state.get("mission_id", "research")

    # Derive required fields
    topic = _derive_research_topic(context, objectives, mission_id)
    objective = _derive_research_objective(context, objectives)
    scope = _derive_research_scope(context)

    return ResearchStatement(
        topic=topic,
        objective=objective,
        scope=scope,
        audience=context.get("audience"),
        methodology="Autonomous web research with LLM synthesis",
        success_metrics=objectives[:5],  # Use objectives as metrics
        risks=[],
    )


def _derive_research_topic(
    context: Dict[str, Any],
    objectives: List[str],
    mission_id: str,
) -> str:
    """Derive research topic from available data.

    Fallback chain: title → first objective → mission_id.

    Args:
        context: Mission context dictionary.
        objectives: List of mission objectives.
        mission_id: Mission identifier.

    Returns:
        Non-empty topic string (guaranteed).
    """
    # Try title first
    title = context.get("title", "")
    if title and len(title.strip()) >= 3:
        return title.strip()

    # Try first objective
    if objectives:
        first_obj = objectives[0]
        # Return first 80 chars, truncating at word boundary if needed
        if len(first_obj) > 80:
            truncated = first_obj[:80].rsplit(" ", 1)[0]
            return truncated.strip()
        return first_obj.strip()

    # Fallback to mission_id
    return f"Research mission {mission_id}"


def _derive_research_objective(
    context: Dict[str, Any],
    objectives: List[str],
) -> str:
    """Derive research objective from context.

    Fallback chain: objective → summary → synthesized from objectives list.

    Args:
        context: Mission context dictionary.
        objectives: List of mission objectives.

    Returns:
        Non-empty objective string (guaranteed).
    """
    # Try explicit objective
    obj = context.get("objective", "")
    if obj and len(obj.strip()) >= 10:
        return obj.strip()

    # Try summary
    summary = context.get("summary", "")
    if summary and len(summary.strip()) >= 10:
        # Truncate if too long
        if len(summary) > 200:
            return summary[:200].rsplit(" ", 1)[0].strip() + "..."
        return summary.strip()

    # Synthesize from objectives list
    if objectives:
        return "Investigate: " + "; ".join(objectives[:3])

    return "Conduct autonomous research and synthesize findings"


def _derive_research_scope(context: Dict[str, Any]) -> str:
    """Derive research scope from context.

    Fallback chain: focus → background first sentence → default.

    Args:
        context: Mission context dictionary.

    Returns:
        Non-empty scope string (guaranteed).
    """
    # Try focus field
    focus = context.get("focus", "")
    if focus and len(focus.strip()) >= 5:
        return focus.strip()

    # Try to extract from background
    background = context.get("background", "")
    if background and len(background) >= 20:
        # Extract first sentence as scope hint
        first_sentence = background.split(".")[0]
        if len(first_sentence) >= 10:
            return f"Scope: {first_sentence.strip()}"

    return "General research scope - web sources and documentation"


__all__ = [
    "ResearchStatement",
    "Synthesis",
    "generate_research_statement",
    "generate_synthesis",
]
