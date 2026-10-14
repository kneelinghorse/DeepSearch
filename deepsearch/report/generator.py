"""Markdown report generation helpers."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from deepsearch.agent.concept_clustering import summarize_state_findings
from deepsearch.agent.coverage import coerce_source_coverage_gate_result
from deepsearch.agent.evaluation import check_quality_gates
from deepsearch.agent.execution_control import raise_if_execution_cancelled
from deepsearch.agent.slot_coverage import extract_failing_slot_details
from deepsearch.agent.state import AgentState
from deepsearch.data_paths import data_path
from deepsearch.mission.contract import resolve_research_objectives
from deepsearch.report.templates import (
    _inject_citation,
    build_findings_section,
    build_mission_context_section,
    build_related_research_section,
    build_report_header,
    build_sources_section,
)
from deepsearch.report.text import require_canonical_report_text

REPORT_DIRECTORY_ENV = "DEEPSEARCH_REPORT_DIR"


def generate_markdown_report(
    state: AgentState,
    *,
    synthesized_content: str | None = None,
) -> str:
    """Render the final markdown report for the provided ``AgentState``.

    Args:
        state: Agent state containing findings, sources, mission context.
        synthesized_content: Optional LLM-synthesized report content. When provided,
            this replaces the template-based Executive Summary and Findings sections
            with the professionally synthesized content.

    Returns:
        Rendered markdown report string.
    """
    mission_id = _coerce_identifier(state.get("mission_id"), fallback="MISSION")
    context = _coerce_mapping(state.get("mission_context"))
    # S80 m02 (PD107): wrapper H1 reads the canonical state["mission_title"]
    # pinned once in empty_state(). The legacy title-or-objective-or-mission_id
    # resolution that lived here was the third independent derivation site and
    # silently diverged from protocol._derive_title when context["title"] was
    # empty (the wrapper would render the objective; the protocol would render
    # `"Research Mission: <id>"`). The single-key read collapses the drift.
    mission_title = state.get("mission_title") or context.get("title") or mission_id
    loop_count = int(state.get("research_loop_count", 0) or 0)
    finding_summary = summarize_state_findings(state)
    findings = _normalise_list(finding_summary.get("display_findings"))
    # S83 m06 (PD83-b) CARVE-OUT: the human-readable report's source section is a
    # tested read of the (dead-in-production) sources_found; kept rather than
    # stripped. Excluded from the never-reappears guards.
    sources = _normalise_source_payloads(state.get("sources_found"))
    objectives = resolve_research_objectives(
        state.get("mission_contract"),
        fallback=state.get("mission_objectives") or [],
    )
    graph_context = _pedr_graph_context(state)
    research_phases = _resolve_research_phases(context)
    research_depth = str(state.get("research_depth", "") or "").lower()

    header = build_report_header(
        mission_id=mission_id,
        title=str(mission_title or mission_id),
        loop_count=loop_count,
        source_count=len(sources),
        research_depth=research_depth or None,
    )
    degradation_notice = _build_degradation_notice(state)
    quality_floor_notice = _build_quality_floor_notice(state)
    contradiction_notice = _build_contradiction_notice(state)
    pedr_notice = _build_pedr_notice(state)
    quality_gates_notice = _build_quality_gates_notice(state)
    quality_notes_section = _build_research_quality_notes_section(state)
    context_section = build_mission_context_section(
        objectives,
        context=context,
    )
    graph_section = build_related_research_section(graph_context)
    sources_section = build_sources_section(sources)

    sections: List[str] = [header.rstrip(), ""]

    # Add notices
    if degradation_notice:
        sections.extend([degradation_notice.strip(), "", "---", ""])
    if pedr_notice:
        sections.extend([pedr_notice.strip(), "", "---", ""])
    if quality_floor_notice:
        sections.extend([quality_floor_notice.strip(), "", "---", ""])
    if contradiction_notice:
        sections.extend([contradiction_notice.strip(), "", "---", ""])
    if quality_gates_notice:
        sections.extend([quality_gates_notice.strip(), "", "---", ""])

    # Use synthesized content if available, otherwise use template-based generation
    if synthesized_content:
        # LLM-synthesized content includes Executive Summary, Key Findings,
        # Analysis & Implications, Recommendations, and Conclusion sections.
        cleaned = prepare_synthesized_content(synthesized_content)
        # Check if synthesis already includes its own References section
        has_inline_refs = bool(re.search(r'^#{1,3}\s+References', cleaned, re.MULTILINE))
        sections.extend([
            cleaned,
            "",
            "---",
            "",
        ])
        # Only append a numbered References section if the synthesis didn't
        # already produce one.  Build from cited sources only — not all raw
        # sources — to avoid irrelevant URLs leaking into references.
        if not has_inline_refs:
            cited_indices = _extract_cited_indices(cleaned)
            if cited_indices:
                cited_refs = _build_numbered_references_cited(sources, cited_indices)
            else:
                cited_refs = _build_numbered_references(sources)
            if cited_refs:
                sections.extend([
                    cited_refs.strip(),
                    "",
                    "---",
                    "",
                ])
    else:
        # Fallback to template-based generation
        executive_summary = _build_executive_summary(findings, sources=sources)
        findings_section = build_findings_section(
            findings, research_phases=research_phases, sources=sources
        )
        sections.extend([
            "## Executive Summary",
            "",
            executive_summary.strip() or "No significant findings were generated during research.",
            "",
            "---",
            "",
            findings_section.strip(),
            "",
            "---",
            "",
        ])

    if quality_notes_section:
        sections.extend([
            quality_notes_section.strip(),
            "",
            "---",
            "",
        ])

    if graph_section:
        sections.extend([
            graph_section.strip(),
            "",
            "---",
            "",
        ])

    # Append common sections
    # When synthesis produced the report, skip the raw "Sources Consulted" list —
    # it includes ALL gathered sources (including irrelevant ones), and the
    # References section already covers cited sources.
    if not synthesized_content:
        sections.extend([
            sources_section.strip(),
            "",
            "---",
            "",
        ])
    sections.extend([
        context_section.strip(),
    ])

    report = "\n".join(sections)
    return report.strip() + "\n"


def _clean_synthesized_content(content: str) -> str:
    """Clean and normalize synthesized content from LLM.

    Removes any meta-commentary or instructions that may have leaked through,
    and ensures proper markdown formatting.
    """
    lines = content.split("\n")
    cleaned_lines: List[str] = []
    skip_patterns = (
        "here is the",
        "here's the",
        "i will now",
        "let me",
        "below is",
        "as requested",
        "output:",
    )

    for line in lines:
        lower_line = line.lower().strip()
        # Skip meta-commentary lines
        if any(lower_line.startswith(pattern) for pattern in skip_patterns):
            continue
        # Skip lines that are just "---" at the very beginning or end
        if lower_line == "---" and (not cleaned_lines or not line.strip()):
            continue
        cleaned_lines.append(line)

    # Remove leading/trailing empty lines
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    result = "\n".join(cleaned_lines)
    # Fix bold formatting artifact: LLM emits '* *Heading:**' instead of '**Heading:**'
    result = re.sub(r"\* \*([^*:]+):\*\*", r"**\1:**", result)
    return result


def prepare_synthesized_content(content: str) -> str:
    """Return the exact synthesized segment inserted into the report.

    Centralizing the cleaner plus final trim gives citation rebasing the same
    segment bytes used by :func:`generate_markdown_report`; callers never need
    to infer a wrapper prefix or duplicate the transform sequence.
    """

    return _clean_synthesized_content(content).strip()


def _build_numbered_references(
    sources: Sequence[Mapping[str, object]],
) -> str:
    """Build a numbered References section mapping [n] citations to sources.

    This section is appended to LLM-synthesized reports so that [n] citations
    in the synthesis body can be resolved by readers and the citation validator.
    """
    if not sources:
        return ""

    lines = ["## References", ""]
    for idx, source in enumerate(sources, 1):
        title = str(source.get("title") or f"Source {idx}").strip()
        url = str(source.get("url") or "").strip()
        if url:
            lines.append(f"[{idx}] {title} — {url}")
        else:
            lines.append(f"[{idx}] {title}")
    return "\n".join(lines)


def _extract_cited_indices(content: str) -> set[int]:
    """Extract [n] citation indices used in the synthesized content."""
    return {int(m) for m in re.findall(r'\[(\d+)\]', content)}


def _build_numbered_references_cited(
    sources: Sequence[Mapping[str, object]],
    cited_indices: set[int],
) -> str:
    """Build a References section containing only sources that were actually cited."""
    if not sources or not cited_indices:
        return ""

    lines = ["## References", ""]
    for idx in sorted(cited_indices):
        if idx < 1 or idx > len(sources):
            continue
        source = sources[idx - 1]
        title = str(source.get("title") or f"Source {idx}").strip()
        url = str(source.get("url") or "").strip()
        if url:
            lines.append(f"[{idx}] {title} — {url}")
        else:
            lines.append(f"[{idx}] {title}")
    # Only return if we actually matched some sources
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


def _build_executive_summary(
    findings: Sequence[str],
    sources: Sequence[Mapping[str, object]] | None = None,
) -> str:
    """Return a short narrative summary of the most recent findings with citations.

    Args:
        findings: List of finding strings
        sources: Optional list of sources for inline citation injection
    """
    clean_findings = [item.strip() for item in findings if item.strip()]
    if not clean_findings:
        return "No significant findings were generated during research."

    sources_list = list(sources) if sources else []
    cited_urls: set[str] = set()

    sentences: List[str] = []
    for entry in clean_findings[:6]:
        text = entry.strip()
        if not text:
            continue

        if sources_list:
            text = _inject_citation(text, sources_list, cited_urls)

        if not text.rstrip().endswith((".", "!", "?", ")")):
            text = f"{text}."
        sentences.append(text)

    if not sentences:
        return "No significant findings were generated during research."

    if len(sentences) <= 2:
        return " ".join(sentences)

    midpoint = max(1, len(sentences) // 2)
    paragraphs = [" ".join(sentences[:midpoint]), " ".join(sentences[midpoint:])]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph).strip()


def _build_degradation_notice(state: AgentState) -> str:
    """Return a markdown warning banner when research ran in degraded mode."""

    if not state.get("degraded_mode"):
        return ""
    entries = state.get("error_history") or []
    lines = [
        "⚠️ **Research Quality Notice**",
        "",
        "This mission completed with degraded capabilities due to:",
    ]
    added = 0
    for entry in entries[-5:]:
        if not isinstance(entry, Mapping):
            continue
        component = str(entry.get("component", "unknown"))
        message = str(entry.get("message", "Unknown failure"))
        lines.append(f"- {component}: {message}")
        added += 1
    if added == 0:
        lines.append("- Service interruptions (details unavailable)")
    policy = str(state.get("degradation_policy") or "").strip()
    if policy and policy != "normal":
        lines.append("")
        lines.append(f"_Active degradation policy_: {policy}")
    return "\n".join(lines).strip()


def _build_pedr_notice(state: AgentState) -> str:
    """Return a markdown notice when PEDR existing research was utilized.

    PEDR (Protocol-Enhanced Deep Research) pre-flight queries can find
    high-quality existing research before launching new web searches.
    """
    pedr_reused = state.get("pedr_reused")
    pedr_reviewed = state.get("pedr_reviewed")

    if not pedr_reused and not pedr_reviewed:
        return ""

    pedr_result = _pedr_result(state)
    source_mission = str(pedr_result.get("source_mission") or "unknown")
    top_score = pedr_result.get("top_score") or 0.0

    if pedr_reused:
        lines = [
            "**Knowledge Reuse Notice**",
            "",
            f"This research leveraged existing validated findings from **{source_mission}** "
            f"(similarity: {top_score:.0%}). Web search was skipped because high-quality "
            "prior research already covers this topic.",
            "",
            "_Source_: TraceLab PEDR pre-flight query",
        ]
    else:  # pedr_reviewed
        lines = [
            "**Related Research Found**",
            "",
            (
                f"Related research was found in **{source_mission}** "
                f"(similarity: {top_score:.0%}) and included as supplementary "
                "context. New web search was performed to expand coverage."
            ),
            "",
            "_Source_: TraceLab PEDR pre-flight query",
        ]

    return "\n".join(lines).strip()


def _build_quality_floor_notice(state: AgentState) -> str:
    """Return a markdown notice when Alpha-tier quality floor filtering was applied.

    Quality floor filtering removes low-authority sources to ensure research
    reliability in Alpha-tier missions.
    """
    depth_config = state.get("depth_config") or {}
    research_depth = str(state.get("research_depth", "") or "").lower()

    # Only show for Alpha tier (or when quality floor is high)
    quality_floor = float(depth_config.get("source_quality_floor", 0) or 0)
    if research_depth != "alpha" and quality_floor < 0.7:
        return ""

    # Get filtering stats from telemetry if available
    telemetry = state.get("telemetry") or {}
    quality_stats = telemetry.get("quality_filtering") or {}
    filtered_count = int(quality_stats.get("filtered_count", 0) or 0)
    total_evaluated = int(quality_stats.get("total_evaluated", 0) or 0)

    lines = [
        "🔒 **Quality Floor Applied**",
        "",
        f"This Alpha-tier research applied a quality floor of **{quality_floor:.0%}** "
        "to filter low-authority sources.",
    ]

    if total_evaluated > 0:
        passed_count = total_evaluated - filtered_count
        lines.append(
            f"**{passed_count}/{total_evaluated}** sources met the quality threshold."
        )

    # Check diversity requirements
    require_diversity = depth_config.get("require_source_diversity", False)
    if require_diversity:
        min_domains = int(depth_config.get("min_unique_domains", 4) or 4)
        lines.append("")
        lines.append(
            f"Source diversity enforced: minimum **{min_domains}** unique domains required."
        )

    return "\n".join(lines).strip()


def _build_contradiction_notice(state: AgentState) -> str:
    """Return a markdown warning section when contradictions were detected.

    Contradiction detection is an Alpha-tier feature that identifies conflicting
    claims across research sources. High-confidence contradictions reduce overall
    research confidence.
    """
    contradiction_report = state.get("contradiction_report")
    if not contradiction_report:
        return ""

    # Handle both dict and object forms
    if hasattr(contradiction_report, "to_dict"):
        report_dict = contradiction_report.to_dict()
    elif isinstance(contradiction_report, Mapping):
        report_dict = dict(contradiction_report)
    else:
        return ""

    has_contradictions = report_dict.get("has_contradictions", False)
    if not has_contradictions:
        return ""

    contradictions = report_dict.get("contradictions", [])
    high_confidence_count = int(report_dict.get("high_confidence_count", 0) or 0)
    confidence_penalty = float(report_dict.get("confidence_penalty", 0) or 0)

    lines = [
        "⚠️ **Contradiction Warning**",
        "",
        f"**{len(contradictions)}** potential contradiction(s) detected in research findings.",
    ]

    if high_confidence_count > 0:
        lines.append(
            f"**{high_confidence_count}** high-confidence contradiction(s) "
            f"reduced overall confidence by **{confidence_penalty:.0%}**."
        )

    lines.extend(["", "### Detected Contradictions", ""])

    for i, contradiction in enumerate(contradictions[:5], 1):  # Limit to 5
        topic = contradiction.get("topic", "Unknown topic")
        claim_a = contradiction.get("claim_a", "")
        claim_b = contradiction.get("claim_b", "")
        confidence = float(contradiction.get("confidence", 0) or 0)
        explanation = contradiction.get("explanation", "")

        conf_label = "🔴 High" if confidence >= 0.8 else "🟡 Medium"

        lines.extend([
            f"**{i}. {topic}** ({conf_label} confidence: {confidence:.0%})",
            "",
            f"> Claim A: _{claim_a[:200]}{'...' if len(claim_a) > 200 else ''}_",
            "",
            f"> Claim B: _{claim_b[:200]}{'...' if len(claim_b) > 200 else ''}_",
        ])

        if explanation:
            lines.extend(["", f"_{explanation}_"])

        lines.append("")

    if len(contradictions) > 5:
        lines.append(f"_...and {len(contradictions) - 5} more contradiction(s)._")

    return "\n".join(lines).strip()


def _build_quality_gates_notice(state: AgentState) -> str:
    """Return a markdown warning banner when quality gates failed at max_loops.

    B11.3: Quality gates prevent finalization when output quality is below
    tier-specific thresholds. If gates fail but max_loops is reached, we
    finalize anyway but show a warning banner.
    """
    loop_count = int(state.get("research_loop_count", 0) or 0)
    max_loops = int(state.get("max_loops", 0) or 0)

    # Only show warning if at max_loops (forced finalization)
    if max_loops <= 0 or loop_count < max_loops:
        return ""

    # Get depth config for tier-specific thresholds
    depth_config = state.get("depth_config") or {}
    min_unique_concepts = int(
        depth_config.get("min_unique_concepts", depth_config.get("min_total_findings", 5)) or 5
    )
    min_total_sources = int(depth_config.get("min_total_sources", 3) or 3)
    min_unique_domains = int(depth_config.get("min_unique_domains_gate", 2) or 2)

    # Get current research output
    concept_clusters = state.get("concept_clusters") or state.get("findings") or []
    # S83 m06 (PD83-b) CARVE-OUT: tested report source section; kept (see above).
    sources = state.get("sources_found") or []

    # Check quality gates
    gate_result = check_quality_gates(
        concept_clusters,
        sources,
        min_unique_concepts=min_unique_concepts,
        min_total_sources=min_total_sources,
        min_unique_domains=min_unique_domains,
    )

    if gate_result.passed:
        return ""

    # Build warning banner
    research_depth = str(state.get("research_depth", "") or "").lower()
    tier_label = research_depth.capitalize() if research_depth else "Research"

    lines = [
        "⚠️ **Quality Gates Not Met**",
        "",
        f"This {tier_label}-tier research completed at maximum loops ({loop_count}/{max_loops}) "
        "without meeting all quality thresholds:",
        "",
    ]

    for failed in gate_result.failed_gates:
        lines.append(f"- {failed}")

    lines.extend([
        "",
        (
            "_Consider running additional research or using a deeper tier "
            "for more comprehensive results._"
        ),
    ])

    return "\n".join(lines).strip()


def _build_research_quality_notes_section(state: AgentState) -> str:
    """Return user-facing notes for non-terminal quality warnings."""

    coverage_lines = _build_coverage_warning_lines(state)
    structural_lines = _build_structural_warning_lines(state)
    if not coverage_lines and not structural_lines:
        return ""

    lines = [
        "## Research Quality Notes",
        "",
        "This research completed and the findings are above. The quality layer flagged",
        "the following during the run. These are informational - the research body is",
        "returned regardless.",
        "",
    ]

    if coverage_lines:
        lines.extend(["**Coverage gate warnings:**", ""])
        lines.extend(coverage_lines)
        lines.append("")

    if structural_lines:
        lines.extend(["**Structural warnings:**", ""])
        lines.extend(structural_lines)
        lines.append("")

    lines.append(
        "Full machine-readable signal at "
        "`execution_metadata.final_outcome.coverage_gate_failures` and "
        "`execution_metadata.final_outcome.structural_validation_failures`."
    )
    return "\n".join(lines).strip()


def _build_coverage_warning_lines(state: AgentState) -> list[str]:
    raw_coverage = state.get("coverage_gates")
    if raw_coverage is None:
        return _build_final_outcome_coverage_warning_lines(state)

    coverage = coerce_source_coverage_gate_result(raw_coverage)
    if coverage.passed and not coverage.failed_gates:
        return []

    lines: list[str] = []
    remaining = set(coverage.failed_gates)
    if "entity" in remaining:
        remaining.remove("entity")
        uncovered = coverage.entity.uncovered
        if uncovered:
            lines.append(
                "- Entity coverage fell below threshold for: "
                f"{', '.join(uncovered)}"
            )
        else:
            lines.append("- Entity coverage fell below threshold.")
        lines.append(
            "- Interpretation: the compiler inferred these entities as required "
            "from the objective and retrieval did not find enough sources bound "
            "to them. Findings on these topics may be thin."
        )

    if "objective" in remaining:
        remaining.remove("objective")
        objective_labels = _objective_label_map(state)
        uncovered_objectives = [
            objective_labels.get(objective_id, objective_id)
            for objective_id in coverage.objective.uncovered_objective_ids
        ]
        if uncovered_objectives:
            lines.append(
                "- Objective coverage fell below threshold for: "
                f"{'; '.join(uncovered_objectives)}"
            )
        else:
            lines.append("- Objective coverage fell below threshold.")

    if "slot" in remaining:
        remaining.remove("slot")
        slot_ids = coverage.slot.thin_slot_ids + coverage.slot.unaddressed_slot_ids
        if slot_ids:
            lines.append(
                "- Evidence-slot coverage fell below threshold for: "
                f"{', '.join(slot_ids)}"
            )
        else:
            lines.append("- Evidence-slot coverage fell below threshold.")

    for failed_gate in sorted(remaining):
        lines.append(f"- Coverage gate failed: {failed_gate}")
    return lines


def _build_final_outcome_coverage_warning_lines(state: AgentState) -> list[str]:
    final_outcome = _resolve_final_outcome(state)
    failed_gates = _normalise_list(final_outcome.get("coverage_gate_failures"))
    return [f"- Coverage gate failed: {gate}" for gate in failed_gates]


def _build_structural_warning_lines(state: AgentState) -> list[str]:
    warnings = state.get("validation_warnings")
    if not isinstance(warnings, list):
        return []

    lines: list[str] = []
    for warning in warnings:
        if not isinstance(warning, Mapping):
            continue
        validator_id = str(warning.get("validator_id") or "validator").strip()
        detail = str(warning.get("detail") or "warning raised").strip()
        lines.append(f"- `{validator_id}` - {detail}")
    return lines




def _filename_unique_suffix(mission_uuid: Optional[str]) -> str:
    """Return an 8-hex-char filename disambiguator (S80 m04 PD109).

    When ``mission_uuid`` is non-empty, take the first 8 hex chars of the
    UUID with hyphens stripped — this is deterministic per mission UUID,
    so the same TraceLab mission row always resolves to the same suffix.
    Two re-runs of the same mission_id slug have *different* TraceLab
    UUIDs, which is the entire point of the disambiguation.

    When ``mission_uuid`` is None / empty (locally-loaded YAML mission or
    the local API path with no TraceLab context), fall back to 4 bytes of
    OS-randomness hex-encoded so re-runs still don't collide on disk.
    """
    if mission_uuid:
        clean = str(mission_uuid).replace("-", "")
        if len(clean) >= 8:
            return clean[:8].lower()
    return os.urandom(4).hex()


def save_report(
    content: str,
    mission_id: str,
    title: str = "",
    *,
    mission_uuid: Optional[str] = None,
) -> str:
    """Persist the rendered report to disk and return the file path.

    S80 m04 (PD109) — filename pattern is
    ``{mission_id}_{title_slug}_{uuid8}.md`` where ``uuid8`` is the first
    8 hex chars of ``mission_uuid`` (TraceLab system-assigned UUID) or a
    random 4-byte hex token when no UUID is available. Pre-S80 the
    filename was ``{mission_id}_{title_slug}.md`` and same-slug re-runs
    silently overwrote the prior run.
    """

    # Spans and identity hashes already address this exact string. Refuse any
    # boundary-time newline/NUL mutation instead of moving those coordinates.
    content = require_canonical_report_text(content)
    directory = _resolve_report_directory()
    directory.mkdir(parents=True, exist_ok=True)
    mission_component = _sanitize_mission_id(mission_id)
    title_slug = _slugify(title) if title else "report"
    suffix = _filename_unique_suffix(mission_uuid)
    filename = f"{mission_component}_{title_slug}_{suffix}.md"
    report_path = directory / filename
    raise_if_execution_cancelled()
    report_path.write_text(content, encoding="utf-8", newline="")
    raise_if_execution_cancelled()
    return str(report_path)


def _resolve_report_directory() -> Path:
    override = os.getenv(REPORT_DIRECTORY_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return data_path("reports")


def _pedr_result(state: AgentState) -> MutableMapping[str, Any]:
    return _coerce_mapping(state.get("pedr_result"))


def _pedr_graph_context(state: AgentState) -> MutableMapping[str, Any]:
    return _coerce_mapping(_pedr_result(state).get("graph_context"))


def _resolve_final_outcome(state: AgentState) -> MutableMapping[str, Any]:
    telemetry = _coerce_mapping(state.get("telemetry"))
    final_outcome = _coerce_mapping(telemetry.get("final_outcome"))
    if final_outcome:
        if "outcome" not in final_outcome and "status" in final_outcome:
            final_outcome["outcome"] = final_outcome.get("status")
        if "routing_reason" not in final_outcome and "reason" in final_outcome:
            final_outcome["routing_reason"] = final_outcome.get("reason")
        return final_outcome

    outcome = str(state.get("mission_outcome") or "").strip()
    routing_reason = str(state.get("routing_reason") or "").strip()
    if not outcome and not routing_reason:
        return {}

    fallback: MutableMapping[str, Any] = {}
    if outcome:
        fallback["outcome"] = outcome
    if routing_reason:
        fallback["routing_reason"] = routing_reason
    return fallback


def _build_objective_gap_row(state: AgentState) -> str:
    coverage_gates = _coerce_mapping(state.get("coverage_gates"))
    objective_gate = _coerce_mapping(coverage_gates.get("objective"))
    if not objective_gate:
        return ""

    threshold = float(objective_gate.get("threshold") or 0.70)
    score_map = _coerce_mapping(objective_gate.get("per_objective_scores"))
    uncovered = [
        str(item).strip()
        for item in objective_gate.get("uncovered_objective_ids") or []
        if str(item).strip()
    ]
    if not score_map or not uncovered:
        return ""

    objective_labels = _objective_label_map(state)
    details: List[str] = []
    for objective_id in uncovered:
        raw_score = score_map.get(objective_id)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        label = objective_labels.get(objective_id, objective_id)
        details.append(f"{objective_id} {score:.2f}<{threshold:.2f} ({label})")
    return f"| Objective gaps | {'; '.join(details)} |"


def _build_research_context_runs_table(state: AgentState) -> str:
    telemetry = _coerce_mapping(state.get("telemetry"))
    rows = telemetry.get("research_context_runs") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        return ""

    lines = [
        "### Research Context Runs",
        "",
        "| Loop | Query | Source origin | Retained | Selected | Displayed | Skip reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        loop_index = int(row.get("loop_index", 0) or 0) + 1
        query = _table_cell(str(row.get("query") or ""))
        source_origin = _table_cell(str(row.get("source_origin") or "unknown"))
        retained_count = int(row.get("retained_source_count", 0) or 0)
        selected_count = int(
            row.get(
                "extraction_selected_source_count",
                row.get("selected_source_count", 0),
            )
            or 0
        )
        displayed_count = int(
            row.get("extraction_displayed_source_count", selected_count) or 0
        )
        skip_reason = _table_cell(str(row.get("skip_reason") or "—"))
        lines.append(
            f"| {loop_index} | {query or '—'} | {source_origin or '—'} | "
            f"{retained_count} | {selected_count} | {displayed_count} | "
            f"{skip_reason or '—'} |"
        )
    return "\n".join(lines) if len(lines) > 4 else ""


def _build_query_telemetry_table(state: AgentState) -> str:
    telemetry = _coerce_mapping(state.get("telemetry"))
    rows = telemetry.get("query_telemetry") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        return ""

    lines = [
        "### Query Telemetry",
        "",
        "| Query | Class | Target slots | Resolved slots | Lane | Requested | Returned | Accepted | Drop reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        query = _table_cell(str(row.get("query") or ""))
        query_class = _table_cell(str(row.get("query_class") or ""))
        target_slots = _table_cell(
            ", ".join(
                str(slot).strip()
                for slot in row.get("target_slots") or []
                if str(slot).strip()
            )
        )
        resolved_slots = _table_cell(
            ", ".join(
                str(slot).strip()
                for slot in row.get("resolved_target_slots") or []
                if str(slot).strip()
            )
        )
        lane = _table_cell(str(row.get("lane") or ""))
        requested = int(row.get("requested", 0) or 0)
        returned = int(row.get("returned", 0) or 0)
        accepted = int(row.get("accepted", 0) or 0)
        drop_reason = _table_cell(str(row.get("drop_reason") or "—"))
        lines.append(
            f"| {query or '—'} | {query_class or '—'} | {target_slots or '—'} | "
            f"{resolved_slots or '—'} | {lane or '—'} | {requested} | {returned} | "
            f"{accepted} | {drop_reason or '—'} |"
        )
    return "\n".join(lines) if len(lines) > 4 else ""


def _build_report_render_row(state: AgentState) -> str:
    synthesis_telemetry = _coerce_mapping(state.get("synthesis_telemetry"))
    render_path = str(synthesis_telemetry.get("report_render_path") or "").strip()
    if not render_path:
        return ""
    render_reason = str(synthesis_telemetry.get("report_render_reason") or "").strip()
    render_detail = render_path
    if render_reason:
        render_detail += f" ({render_reason})"
    return f"| Report render | {render_detail} |"


def _build_finalize_schema_row(state: AgentState) -> str:
    synthesis_telemetry = _coerce_mapping(state.get("synthesis_telemetry"))
    has_schemas = synthesis_telemetry.get("finalize_has_deliverable_schemas")
    schema_ids = [
        str(item).strip()
        for item in synthesis_telemetry.get("finalize_deliverable_schema_ids") or []
        if str(item).strip()
    ]
    if has_schemas is None and not schema_ids:
        return ""
    status = "yes" if has_schemas else "no"
    if schema_ids:
        return f"| Finalize schemas | {status} ({', '.join(schema_ids)}) |"
    return f"| Finalize schemas | {status} |"


def _extract_entity_extraction_path(state: AgentState) -> str:
    """Read entity_extraction_path off the compiled mission contract.

    The compiler appends a row of the shape ``{"entity": "", "decision":
    "extraction_path", "reason": "<path>"}`` to
    ``mission_contract.entity_extraction_decisions``. Surface it as a
    dedicated Telemetry row so paid-smoke audits can confirm the
    declared-authoritative short-circuit (S59.2) without depending on the
    MCP envelope's execution_metadata.
    """

    mission_contract = _coerce_mapping(state.get("mission_contract"))
    if not mission_contract:
        return ""
    decisions = mission_contract.get("entity_extraction_decisions") or []
    if not isinstance(decisions, list):
        return ""
    for entry in decisions:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("decision") or "").strip() != "extraction_path":
            continue
        reason = str(entry.get("reason") or "").strip()
        if reason:
            return reason
    return ""


def _build_slot_gap_row(state: AgentState) -> str:
    failing_slots = extract_failing_slot_details(_coerce_mapping(state.get("slot_coverage")))
    if not failing_slots:
        return ""
    details = []
    for slot in failing_slots[:4]:
        slot_id = str(slot.get("slot_id") or "unknown-slot").strip()
        kind = str(slot.get("kind") or "unknown").strip()
        status = str(slot.get("status") or "unknown").strip()
        details.append(f"{slot_id} [{kind}] {status}")
    if len(failing_slots) > 4:
        details.append(f"+{len(failing_slots) - 4} more")
    return f"| Slot gaps | {'; '.join(details)} |"


def _build_required_entity_coverage_table(state: AgentState) -> str:
    slot_coverage = _coerce_mapping(state.get("slot_coverage"))
    slot_rows = [
        _coerce_mapping(item)
        for item in slot_coverage.get("slots") or []
        if isinstance(item, Mapping)
    ]
    entity_rows = [
        row
        for row in slot_rows
        if str(_coerce_mapping(row.get("target")).get("entity") or "").strip()
    ]
    if not entity_rows:
        return ""

    lines = [
        "### Required Entity Coverage",
        "",
        "| Entity | Status | Sources | Domains | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in entity_rows:
        target = _coerce_mapping(row.get("target"))
        entity = _table_cell(str(target.get("entity") or ""))
        status = _table_cell(str(row.get("status") or "unknown"))
        source_count = int(row.get("source_count", 0) or 0)
        domains = ", ".join(
            str(domain).strip()
            for domain in row.get("domains") or []
            if str(domain).strip()
        )
        reasons = ", ".join(
            str(reason).strip()
            for reason in row.get("reasons") or []
            if str(reason).strip()
        )
        notes = reasons or ("covered" if status == "filled" else "—")
        lines.append(
            f"| {entity or '—'} | {status or '—'} | {source_count} | "
            f"{_table_cell(domains) or '—'} | {_table_cell(notes) or '—'} |"
        )
    return "\n".join(lines) if len(lines) > 4 else ""


def _build_slot_coverage_gaps_table(state: AgentState) -> str:
    failing_slots = extract_failing_slot_details(_coerce_mapping(state.get("slot_coverage")))
    if not failing_slots:
        return ""

    lines = [
        "### Slot Coverage Gaps",
        "",
        "| Slot ID | Kind | Status | Target | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in failing_slots:
        slot_id = _table_cell(str(row.get("slot_id") or ""))
        kind = _table_cell(str(row.get("kind") or ""))
        status = _table_cell(str(row.get("status") or ""))
        target = _table_cell(_slot_target_label(_coerce_mapping(row.get("target"))))
        notes_parts: List[str] = []
        source_count = int(row.get("source_count", 0) or 0)
        notes_parts.append(f"{source_count} sources")
        domains = ", ".join(
            str(domain).strip()
            for domain in row.get("domains") or []
            if str(domain).strip()
        )
        if domains:
            notes_parts.append(domains)
        reasons = ", ".join(
            str(reason).strip()
            for reason in row.get("reasons") or []
            if str(reason).strip()
        )
        if reasons:
            notes_parts.append(reasons)
        description = str(row.get("description") or "").strip()
        if description:
            notes_parts.append(description)
        notes = _table_cell("; ".join(part for part in notes_parts if part))
        lines.append(
            f"| {slot_id or '—'} | {kind or '—'} | {status or '—'} | "
            f"{target or '—'} | {notes or '—'} |"
        )
    return "\n".join(lines)


def _slot_target_label(target: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in ("entity", "objective_id", "schema_id", "format"):
        value = str(target.get(key) or "").strip()
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _table_cell(value: str) -> str:
    compact = " ".join(str(value or "").split())
    return compact.replace("|", "\\|")


def _objective_label_map(state: AgentState) -> dict[str, str]:
    contract = _coerce_mapping(state.get("mission_contract"))
    objectives = contract.get("objectives") or []
    labels: dict[str, str] = {}
    for item in objectives:
        if not isinstance(item, Mapping):
            continue
        objective_id = str(item.get("id") or "").strip()
        if not objective_id:
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        labels[objective_id] = _truncate_objective_label(text)
    return labels


def _truncate_objective_label(text: str, *, limit: int = 72) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _coerce_mapping(value: object | None) -> MutableMapping[str, Any]:
    if isinstance(value, MutableMapping):
        return dict(value)
    return {}


def _normalise_list(value: object | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalise_source_payloads(value: object | None) -> List[Mapping[str, object]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, str):
        return [{"title": value}]
    if isinstance(value, Iterable):
        normalised: List[Mapping[str, object]] = []
        for entry in value:
            if isinstance(entry, Mapping):
                normalised.append(dict(entry))
        return normalised
    return []


def _resolve_research_phases(context: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw = context.get("research_phases") or context.get("researchPhases")
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [dict(value) for value in raw.values() if isinstance(value, Mapping)]
    if isinstance(raw, Iterable):
        items: List[Mapping[str, Any]] = []
        for entry in raw:
            if isinstance(entry, Mapping):
                items.append(dict(entry))
        return items
    return []


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "report"


def _sanitize_mission_id(value: str) -> str:
    text = value.strip() or "MISSION"
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def _coerce_identifier(value: object | None, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def save_protocol(
    protocol: Any,
    mission_id: str,
    title: str = "",
    *,
    mission_uuid: Optional[str] = None,
) -> str:
    """Persist the protocol to a JSON file for TraceLab ingestion.

    S80 m04 (PD109) — filename pattern is
    ``{mission_id}_{title_slug}_{uuid8}_protocol.json``; ``uuid8`` is the
    same disambiguator computed by ``save_report`` so the markdown report
    and the protocol JSON share an identical run-tag suffix on disk.

    Args:
        protocol: MissionProtocolComplete instance to serialize.
        mission_id: Mission identifier for filename.
        title: Optional title for filename slug.
        mission_uuid: TraceLab system-assigned UUID. None for local missions
            (random fallback suffix used).

    Returns:
        Path to saved JSON file.
    """

    directory = _resolve_report_directory()
    directory.mkdir(parents=True, exist_ok=True)

    mission_component = _sanitize_mission_id(mission_id)
    title_slug = _slugify(title) if title else "report"
    suffix = _filename_unique_suffix(mission_uuid)
    filename = f"{mission_component}_{title_slug}_{suffix}_protocol.json"
    protocol_path = directory / filename

    # Serialize with Pydantic
    if hasattr(protocol, "model_dump_json"):
        json_content = protocol.model_dump_json(indent=2)
    else:
        # Fallback for dict input
        import json

        json_content = json.dumps(
            protocol if isinstance(protocol, dict) else protocol.to_dict(),
            indent=2,
        )

    raise_if_execution_cancelled()
    protocol_path.write_text(json_content, encoding="utf-8")
    raise_if_execution_cancelled()
    return str(protocol_path)


__all__ = [
    "generate_markdown_report",
    "save_report",
    "save_protocol",
    "_build_executive_summary",
    "_build_pedr_notice",
    "_build_quality_floor_notice",
    "_build_contradiction_notice",
    "_build_quality_gates_notice",
    "_clean_synthesized_content",
    "prepare_synthesized_content",
]
