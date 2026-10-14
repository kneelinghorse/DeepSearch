"""High-level research mission execution helpers."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    cast,
)

from deepsearch.agent import graph as agent_graph
from deepsearch.agent.checkpointer import (
    CheckpointerUnavailableError,
    calculate_recursion_limit,
    clear_thread_checkpoints,
    create_checkpointer,
    get_checkpoint_history,
    thread_config,
)
from deepsearch.agent.evidence import (
    TelemetryCollector,
    TelemetryWriter,
    extract_sources_from_state,
    record_finding_records_from_state,
)
from deepsearch.agent.outcome import resolve_final_outcome_payload
from deepsearch.agent.state import AgentState, empty_state
from deepsearch.agent.telemetry import normalize_final_telemetry
from deepsearch.config import DeepSearchSettings
from deepsearch.data_paths import data_path
from deepsearch.mission import load_mission_from_yaml, prepare_execution_contract
from deepsearch.observability import apply_runtime_context, build_runtime_context
from deepsearch.report import generate_markdown_report, save_report
from deepsearch.validation import (
    HybridEvaluator,
    PerformanceMetrics,
    QualityThresholds,
    ValidationResult,
)

if TYPE_CHECKING:
    from deepsearch.api.models.requests import ExecuteMissionRequest

logger = logging.getLogger(__name__)

_SUPPORTED_LLM_BACKENDS = {"ollama", "gemini", "deepseek", "minimax", "openai"}
_MODEL_ENV_BY_BACKEND = {
    "ollama": "OLLAMA_MODEL",
    "gemini": "GEMINI_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
    "minimax": "MINIMAX_MODEL",
    "openai": "OPENAI_MODEL",
}


def _default_model_for_backend(settings: DeepSearchSettings, backend: str) -> str:
    if backend == "openai":
        return settings.openai_model
    if backend == "gemini":
        return settings.gemini_model
    if backend == "deepseek":
        return settings.deepseek_model
    if backend == "minimax":
        return settings.minimax_model
    return settings.ollama_model


def _resolve_backend_and_model(
    settings: DeepSearchSettings,
    *,
    llm_backend: str | None,
    model: str | None,
) -> tuple[str, str]:
    resolved_backend = (llm_backend or settings.llm_backend).strip().lower() or "deepseek"
    if resolved_backend not in _SUPPORTED_LLM_BACKENDS:
        raise ValueError("Unsupported LLM_BACKEND")
    resolved_model = model or _default_model_for_backend(settings, resolved_backend)
    return resolved_backend, resolved_model


def _ensure_langchain_debug_flag() -> None:
    """Ensure langchain debug/verbose flags exist for compatibility with langchain_core."""
    try:  # pragma: no cover - runtime guard
        import langchain
    except Exception:
        return
    if not hasattr(langchain, "debug"):
        setattr(langchain, "debug", False)
    if not hasattr(langchain, "verbose"):
        setattr(langchain, "verbose", False)
    if not hasattr(langchain, "llm_cache"):
        setattr(langchain, "llm_cache", None)


async def execute_research_mission(
    mission_path: str | Path,
    *,
    llm_backend: str | None = None,
    model: str | None = None,
    dry_run: bool = False,
    resume: bool = True,
    clear_checkpoints: bool = False,
    show_checkpoints: bool = False,
    thread_id: str | None = None,
    validate: bool = False,
    validation_output: str | Path | None = None,
    quality_thresholds: QualityThresholds | None = None,
    auto_upload: bool = False,
    force_loops: int | None = None,
    research_depth: str | None = None,
) -> Dict[str, Any]:
    """Execute a Mission Protocol YAML file end-to-end and return result metadata.

    Args:
        mission_path: Path to the mission YAML file.
        llm_backend: Optional backend override ("ollama" or "gemini"). Defaults to env/config.
        model: Optional model override. Defaults to env/config for the chosen backend.
        dry_run: When ``True`` validate the mission without executing the graph.
        resume: When ``True`` reuse persisted checkpoints for the mission.
        clear_checkpoints: Delete checkpoints after the run completes.
        show_checkpoints: Include checkpoint history in the returned metadata.
        thread_id: Optional explicit thread identifier override.
        validate: When ``True`` run the hybrid validation framework after the mission.
        validation_output: Optional directory to persist validation metrics.
        quality_thresholds: Optional overrides for research quality thresholds.
        auto_upload: When ``True`` auto-upload to TraceLab if all quality gates pass.
        force_loops: B7.5.1 - Force minimum loops before convergence can trigger.
        research_depth: Optional depth tier override ("baseline", "deep", "alpha").

    Returns:
        Dictionary describing the execution status, loops, and deliverables.

    Raises:
        MissionLoadError: If the mission file is missing or invalid.
    """

    path = Path(mission_path)
    logger.info("Loading mission from %s", path)
    state = load_mission_from_yaml(path)
    if research_depth:
        _apply_depth_override(state, research_depth)
    # B7.5.1: Apply force_loops override to prevent convergence early exit
    if force_loops is not None:
        state["min_loops"] = force_loops
        logger.info("Forcing minimum %d loops before convergence", force_loops)
    settings = DeepSearchSettings.load()
    contract_artifact = prepare_execution_contract(state, origin="yaml")
    mission_id = state["mission_id"]
    project_id = state.get("project_id")
    logger.info("Loaded mission %s", mission_id)
    logger.info(
        "Compiled mission contract %s -> %s",
        state.get("mission_contract_id"),
        state.get("mission_contract_path"),
    )
    logger.debug("Mission objectives: %s", state.get("mission_objectives"))
    run_started = time.perf_counter()

    resolved_backend, resolved_model = _resolve_backend_and_model(
        settings,
        llm_backend=llm_backend,
        model=model,
    )
    _configure_llm_environment(resolved_backend, resolved_model)
    settings = DeepSearchSettings.load()
    thread_identifier = thread_id or settings.generate_thread_id(mission_id)
    apply_runtime_context(
        state,
        build_runtime_context(
            settings=settings,
            state=state,
            llm_backend=resolved_backend,
            model=resolved_model,
            execution_path="yaml",
            thread_id=thread_identifier,
            validate=validate,
            auto_upload=auto_upload,
        ),
    )

    # Initialize telemetry collector for evidence capture
    telemetry_collector = TelemetryCollector(
        mission_id=mission_id,
        project_id=project_id,
    )
    telemetry_collector.set_contract(
        contract_id=contract_artifact.contract.contract_id,
        contract_path=str(contract_artifact.path),
        contract_version=contract_artifact.contract.contract_version,
        contract_origin=contract_artifact.contract.origin,
    )

    if dry_run:
        logger.info("Dry run requested – mission validated successfully")
        return {
            "status": "validated",
            "mission_id": mission_id,
            "contract_id": state.get("mission_contract_id"),
            "contract_path": state.get("mission_contract_path"),
            "loops_executed": 0,
            "sources_count": 0,
            "report_path": "",
            "thread_id": thread_id or settings.generate_thread_id(mission_id),
            "checkpoint_db_path": str(settings.checkpoint_db_path),
            "persistence_enabled": False,
            "checkpoint_history": None,
            "persistence_error": "dry_run",
            "performance_metrics": {
                "total_duration_seconds": 0.0,
                "loops_executed": 0,
                "api_calls": {},
                "token_usage": {},
                "estimated_cost_usd": 0.0,
            },
            "telemetry": _safe_telemetry(state),
            "validation": None,
            "validation_error": None,
        }
    persistence_enabled = True
    persistence_error: str | None = None
    checkpointer = None
    checkpointer_ctx = None
    try:
        checkpointer_ctx = create_checkpointer(settings.checkpoint_db_path)
        checkpointer = await checkpointer_ctx.__aenter__()
    except CheckpointerUnavailableError as exc:  # pragma: no cover - runtime guard
        logger.warning("Checkpoint persistence unavailable: %s", exc)
        persistence_enabled = False
        persistence_error = str(exc)
        checkpointer_ctx = None
    except Exception as exc:
        logger.warning("Checkpoint persistence failed, continuing without: %s", exc)
        persistence_enabled = False
        persistence_error = str(exc)
        checkpointer_ctx = None
        checkpointer = None

    async def _close_checkpointer() -> None:
        if checkpointer_ctx is not None:
            await checkpointer_ctx.__aexit__(None, None, None)

    depth_cfg = state.get("depth_config") or {}
    recursion_limit = _resolve_recursion_limit(depth_cfg, state.get("max_loops", 3))
    graph_config = thread_config(thread_identifier, recursion_limit=recursion_limit)

    try:
        if checkpointer is not None and not resume:
            logger.info("Clearing checkpoints before run for thread %s", thread_identifier)
            await clear_thread_checkpoints(thread_identifier, checkpointer=checkpointer)

        _ensure_langchain_debug_flag()
        graph = agent_graph.build_workflow(checkpointer=checkpointer)
        logger.info("LangGraph workflow compiled successfully")
        # Once execution starts, a checkpoint failure can follow paid work.
        # Propagate it; never build a fresh graph and replay from step one.
        final_state = cast(AgentState, await graph.ainvoke(state, config=graph_config))
        status = "complete"
    except RuntimeError as exc:
        logger.error(
            "LangGraph execution failed for mission %s: %s", mission_id, exc
        )
        raise
    finally:
        if clear_checkpoints and checkpointer is not None:
            logger.info("Clearing checkpoints after run for thread %s", thread_identifier)
            await clear_thread_checkpoints(thread_identifier, checkpointer=checkpointer)
        if checkpointer_ctx is not None:
            await _close_checkpointer()

    report_path = _ensure_report(final_state, mission_id)
    # S83 m06 (PD83-b) CARVE-OUT: the local-runner result's sources_count is a
    # tested read of the (dead-in-production) sources_found; kept rather than
    # stripped. Excluded from the never-reappears guards.
    sources = final_state.get("sources_found") or []
    loops = int(final_state.get("research_loop_count", 0) or 0)
    checkpoint_history: list[Dict[str, Any]] | None = None

    if show_checkpoints and checkpointer is not None:
        checkpoint_history = await get_checkpoint_history(
            thread_identifier,
            checkpointer=checkpointer,
        )

    duration_seconds = max(time.perf_counter() - run_started, 0.0)
    telemetry = _safe_telemetry(final_state)
    performance_metrics = _build_performance_metrics(
        duration=duration_seconds,
        loops=loops,
        telemetry=telemetry,
        backend=settings.llm_backend,
    )

    # Capture structured evidence from final state
    record_finding_records_from_state(final_state, telemetry_collector)
    extract_sources_from_state(final_state, telemetry_collector)

    # Set telemetry metadata from final state
    if final_state.get("degraded_mode"):
        policy = str(final_state.get("degradation_policy") or "")
        telemetry_collector.set_degradation([policy] if policy else ["degraded"])
    termination_reason = str(final_state.get("termination_reason") or "").strip()
    if termination_reason:
        telemetry_collector.set_termination(early=True, reason=termination_reason)
    telemetry_collector.set_cost(performance_metrics.estimated_cost_usd)
    validation_result: ValidationResult | None = None
    validation_error: str | None = None
    if validate:
        evaluator = HybridEvaluator(
            thresholds=quality_thresholds or QualityThresholds(),
            output_dir=_normalise_output_dir(validation_output),
        )
        try:
            validation_result = await evaluator.evaluate(
                state=final_state,
                report_path=report_path,
                performance_metrics=performance_metrics,
            )
            if validation_result.output_path:
                logger.info("Validation metrics stored at %s", validation_result.output_path)
        except Exception as exc:  # pragma: no cover - validation guard
            validation_error = str(exc)
            logger.error("Hybrid validation failed: %s", exc)

    # Auto-upload to TraceLab if enabled and quality gates pass
    upload_result: Dict[str, Any] | None = None
    upload_error: str | None = None
    if auto_upload and not dry_run:
        upload_result, upload_error = await _maybe_upload_to_tracelab(
            final_state, mission_id, validation_result
        )

    failed_checks: list[str] = []
    if validation_result is not None:
        failed_checks = [
            name
            for name, passed in validation_result.passes_standards.items()
            if not passed
        ]
        telemetry["validation_summary"] = {
            "requested": True,
            "status": "passed" if validation_result.overall_pass else "failed",
            "failed_checks": failed_checks,
            "output_path": validation_result.output_path,
        }
    elif validation_error:
        telemetry["validation_summary"] = {
            "requested": True,
            "status": "error",
            "error": validation_error,
        }
    else:
        telemetry["validation_summary"] = {
            "requested": bool(validate),
            "status": "not_requested" if not validate else "not_run",
        }

    completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    final_outcome = resolve_final_outcome_payload(final_state, completed_at=completed_at)
    telemetry["final_outcome"] = final_outcome
    final_state["mission_outcome"] = final_outcome["outcome"]
    final_state["routing_reason"] = final_outcome["routing_reason"]
    telemetry_collector.merge_state_telemetry(telemetry)

    # Finalize and persist telemetry to JSONL
    mission_telemetry = telemetry_collector.finalize()
    telemetry_writer = TelemetryWriter()
    telemetry_path = telemetry_writer.write(mission_telemetry)
    if telemetry_path:
        logger.info("Telemetry persisted to %s", telemetry_path)
        telemetry["telemetry_artifact_path"] = telemetry_path
    final_state["telemetry"] = telemetry

    logger.info(
        "Mission %s finished with status %s after %d loops (%d sources)",
        mission_id,
        status,
        loops,
        len(sources),
    )

    return {
        "status": status,
        "mission_id": mission_id,
        "contract_id": state.get("mission_contract_id"),
        "contract_path": state.get("mission_contract_path"),
        "loops_executed": loops,
        "sources_count": len(sources),
        "report_path": report_path,
        "total_duration_seconds": duration_seconds,
        "llm_backend": settings.llm_backend,
        "model": resolved_model,
        "thread_id": thread_identifier,
        "checkpoint_db_path": str(settings.checkpoint_db_path),
        "persistence_enabled": persistence_enabled,
        "checkpoint_history": checkpoint_history,
        "persistence_error": persistence_error,
        "telemetry": telemetry,
        "telemetry_path": telemetry_path,
        "performance_metrics": performance_metrics.to_dict(),
        "validation": validation_result.to_dict() if validation_result else None,
        "validation_error": validation_error,
        "final_outcome": final_outcome,
        "upload": upload_result,
        "upload_error": upload_error,
    }


def _configure_llm_environment(llm_backend: str, model: str) -> None:
    """Persist CLI overrides into environment variables for downstream loaders."""

    backend = (llm_backend or "deepseek").strip().lower() or "deepseek"
    os.environ["LLM_BACKEND"] = backend
    os.environ[_MODEL_ENV_BY_BACKEND.get(backend, "OLLAMA_MODEL")] = model


def _apply_depth_override(state: AgentState, depth_override: str) -> None:
    """Apply a depth label override to an already-initialised AgentState.

    Boundary string carrier post-S76 — sets state["research_depth"] only;
    no in-process dispatch on the value.
    """
    if not depth_override:
        return
    normalized = depth_override.strip().lower()
    if normalized not in {"baseline", "deep", "alpha"}:
        normalized = "baseline"
    state["research_depth"] = normalized
    logger.info("Research depth label applied: %s", normalized)


def _ensure_report(state: AgentState, mission_id: str) -> str:
    """Guarantee that ``report_path`` exists on the final state."""

    existing = state.get("report_path")
    if isinstance(existing, (str, Path)) and str(existing).strip():
        return str(existing)

    report_content = generate_markdown_report(state)
    title = str((state.get("mission_context") or {}).get("title", ""))
    # S80 m04 (PD109): mission_uuid is None on the local API path (the
    # ExecuteMissionRequest doesn't carry a TraceLab UUID); save_report
    # falls back to a random suffix so re-runs still don't collide.
    path = save_report(
        report_content,
        mission_id=mission_id,
        title=title,
        mission_uuid=state.get("mission_uuid") or None,
    )
    state["report_path"] = path
    return path


def _merge_updates(state: AgentState, update: Mapping[str, Any]) -> AgentState:
    """Merge LangGraph-style state updates into a mutable copy."""

    merged: MutableMapping[str, Any] = dict(state)
    for key, value in update.items():
        if key == "messages" and isinstance(value, Sequence):
            existing = list(merged.get("messages") or [])
            existing.extend(value)
            merged["messages"] = existing
            continue
        merged[key] = value
    return cast(AgentState, merged)


def _safe_telemetry(state: AgentState) -> Dict[str, Any]:
    return normalize_final_telemetry(state)


def _build_performance_metrics(
    *,
    duration: float,
    loops: int,
    telemetry: Mapping[str, Any],
    backend: str,
) -> PerformanceMetrics:
    api_calls = dict(telemetry.get("api_calls") or {})
    token_usage = dict(telemetry.get("token_usage") or {})
    estimated_cost = float(telemetry.get("estimated_cost_usd") or 0.0)
    if not estimated_cost:
        estimated_cost = _estimate_llm_cost(token_usage, backend)
    return PerformanceMetrics(
        total_duration_seconds=max(duration, 0.0),
        loops_executed=max(loops, 0),
        api_calls=api_calls,
        token_usage=token_usage,
        estimated_cost_usd=max(estimated_cost, 0.0),
    )


def _estimate_llm_cost(token_usage: Mapping[str, Any], backend: str) -> float:
    prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
    completion_tokens = int(token_usage.get("completion_tokens") or 0)
    backend_lower = backend.lower()

    if backend_lower == "gemini":
        # Gemini 2.5 Pro pricing (per token)
        input_rate = 1.25 / 1_000_000   # $1.25/1M input tokens
        output_rate = 10.00 / 1_000_000  # $10.00/1M output tokens
    elif backend_lower == "deepseek":
        # DeepSeek R1 pricing (per token) — conservative estimate
        # R1: $0.55/1M input, $2.19/1M output
        # V3: $0.27/1M input, $1.10/1M output
        # Use R1 rates as upper bound since primary model is R1
        input_rate = 0.55 / 1_000_000
        output_rate = 2.19 / 1_000_000
    else:
        return 0.0

    cost = (prompt_tokens * input_rate) + (completion_tokens * output_rate)
    return round(cost, 6)


def _normalise_output_dir(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.suffix:
        parent = candidate.parent
        return parent if str(parent) else Path(".")
    return candidate


async def _maybe_upload_to_tracelab(
    state: AgentState,
    mission_id: str,
    validation_result: ValidationResult | None,
) -> tuple[Dict[str, Any] | None, str | None]:
    """Upload to TraceLab if all quality gates pass.

    Args:
        state: Final agent state containing research results.
        mission_id: The mission identifier.
        validation_result: Optional validation result to check gates.

    Returns:
        Tuple of (upload_result_dict, error_string).
        If successful, error_string is None.
        If skipped/failed, upload_result_dict may still contain info.
    """
    from deepsearch.report.protocol import generate_mission_protocol
    from deepsearch.tracelab.uploader import upload_protocol

    # Check quality gates from validation result
    if validation_result is not None:
        quality_scores = validation_result.quality_scores
        thresholds = validation_result.thresholds

        # Get scores with defaults
        faithfulness = quality_scores.get("faithfulness", 0.0)
        relevance = quality_scores.get("relevance", 0.0)
        citation_rate = quality_scores.get("citation_rate", 0.0)

        # Check against thresholds (defaults: F>=0.9, R>=0.8, C>=0.8)
        failing_gates = []
        if faithfulness < (thresholds.faithfulness if thresholds else 0.9):
            failing_gates.append(f"faithfulness ({faithfulness:.2f} < 0.9)")
        if relevance < (thresholds.relevance if thresholds else 0.8):
            failing_gates.append(f"relevance ({relevance:.2f} < 0.8)")
        if citation_rate < (thresholds.citation_rate if thresholds else 0.8):
            failing_gates.append(f"citation_rate ({citation_rate:.2f} < 0.8)")

        if failing_gates:
            logger.warning(
                "Skipping upload for %s - failing gates: %s",
                mission_id,
                ", ".join(failing_gates),
            )
            return {
                "skipped": True,
                "reason": "quality_gates_failed",
                "failing_gates": failing_gates,
            }, None

    try:
        # Generate protocol from state
        protocol = generate_mission_protocol(state)

        # Upload protocol to TraceLab
        result = upload_protocol(protocol)

        if result.success:
            logger.info(
                "Uploaded to TraceLab: mission_uuid=%s",
                result.mission_uuid,
            )
            return result.to_dict(), None
        else:
            logger.error(
                "Upload failed for %s: %s",
                mission_id,
                result.errors,
            )
            # Save failed submission for manual retry
            _save_failed_submission(mission_id, protocol, result.errors)
            return result.to_dict(), None

    except Exception as exc:
        error_msg = f"Upload exception: {exc}"
        logger.error("Upload failed for %s: %s", mission_id, error_msg)
        return None, error_msg


def _save_failed_submission(
    mission_id: str,
    protocol: Any,
    errors: list[str],
) -> None:
    """Save failed submission for later manual retry."""
    failed_dir = data_path("failed_submissions")
    failed_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{mission_id}_{timestamp}.json"
    filepath = failed_dir / filename

    try:
        import json

        if hasattr(protocol, "model_dump_json"):
            protocol_data = json.loads(protocol.model_dump_json())
        elif hasattr(protocol, "to_dict"):
            protocol_data = protocol.to_dict()
        else:
            protocol_data = dict(protocol)

        failed_data = {
            "mission_id": mission_id,
            "timestamp": timestamp,
            "errors": errors,
            "protocol": protocol_data,
        }

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)

        logger.info("Failed submission saved to %s", filepath)
    except Exception as exc:
        logger.warning("Failed to save failed submission: %s", exc)


async def run_research_from_request(
    request: ExecuteMissionRequest,
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    llm_backend: str | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """Run research from an API ExecuteMissionRequest.

    This function creates an AgentState from the API request and executes
    the research pipeline without needing a YAML file.

    Args:
        request: The ExecuteMissionRequest from the API.
        progress_callback: Optional callback(loop_num, findings_count) for progress.
        llm_backend: Optional backend override. Defaults to env/config.
        model: Optional model override for the chosen backend. Defaults to env/config.

    Returns:
        Dictionary with execution results including report_path.
    """
    # Build mission context from request. S39.2: route the full ExecuteMissionRequest
    # through build_mission_context_from_execute_request so the 8 extended fields
    # (required_entities, excluded_entities, expected_output_schema,
    # coverage_thresholds, validation_thresholds, deliverable_format, references,
    # plus success_criteria) land in mission_context where compile_contract_from_state
    # reads them, identically to the /missions/preview path.
    from deepsearch.api.models.requests import build_mission_context_from_execute_request

    context_payload: Dict[str, Any] = build_mission_context_from_execute_request(request)

    # Use objectives as success criteria if none provided
    objectives = request.success_criteria if request.success_criteria else request.objectives

    raw_depth = getattr(request, "research_depth", None)
    depth_label = raw_depth.strip().lower() if isinstance(raw_depth, str) and raw_depth.strip() else "baseline"
    if depth_label not in {"baseline", "deep", "alpha"}:
        depth_label = "baseline"
    max_loops = request.max_loops

    # Create initial state. deliverable_format + min_loops come from the request
    # when present so the execute path mirrors the preview payload exactly.
    state = empty_state(
        mission_id=request.mission_id,
        mission_objectives=objectives,
        project_id=_extract_project_id_from_mission(request.mission_id),
        mission_context=context_payload,
        deliverable_format=request.deliverable_format or "markdown",
        research_depth=depth_label,
        max_loops=max_loops,
        min_loops=request.min_loops,
    )
    contract_artifact = prepare_execution_contract(state, origin="api_request")

    logger.info("Running research for API mission %s", request.mission_id)
    run_started = time.perf_counter()

    # Initialize telemetry
    telemetry_collector = TelemetryCollector(
        mission_id=request.mission_id,
        project_id=state.get("project_id"),
    )
    telemetry_collector.set_contract(
        contract_id=contract_artifact.contract.contract_id,
        contract_path=str(contract_artifact.path),
        contract_version=contract_artifact.contract.contract_version,
        contract_origin=contract_artifact.contract.origin,
    )

    settings = DeepSearchSettings.load()
    resolved_backend, resolved_model = _resolve_backend_and_model(
        settings,
        llm_backend=llm_backend,
        model=model,
    )
    _configure_llm_environment(resolved_backend, resolved_model)
    settings = DeepSearchSettings.load()
    thread_identifier = settings.generate_thread_id(request.mission_id)
    apply_runtime_context(
        state,
        build_runtime_context(
            settings=settings,
            state=state,
            llm_backend=resolved_backend,
            model=resolved_model,
            execution_path="api_request",
            thread_id=thread_identifier,
        ),
    )

    # Set up checkpointer
    persistence_enabled = True
    persistence_error: str | None = None
    checkpointer = None
    checkpointer_ctx = None
    try:
        checkpointer_ctx = create_checkpointer(settings.checkpoint_db_path)
        checkpointer = await checkpointer_ctx.__aenter__()
    except CheckpointerUnavailableError as exc:
        logger.warning("Checkpoint persistence unavailable: %s", exc)
        persistence_enabled = False
        persistence_error = str(exc)
        checkpointer_ctx = None
    except Exception as exc:
        logger.warning("Checkpoint persistence failed, continuing without: %s", exc)
        persistence_enabled = False
        persistence_error = str(exc)
        checkpointer_ctx = None
        checkpointer = None

    async def _close_checkpointer() -> None:
        if checkpointer_ctx is not None:
            await checkpointer_ctx.__aexit__(None, None, None)

    depth_cfg = state.get("depth_config") or {}
    recursion_limit = _resolve_recursion_limit(depth_cfg, state.get("max_loops", 3))
    graph_config = thread_config(thread_identifier, recursion_limit=recursion_limit)

    try:
        graph = agent_graph.build_workflow(checkpointer=checkpointer)
        logger.info("LangGraph workflow compiled successfully")

        # Note: progress_callback not stored in state as functions aren't serializable
        # Progress updates happen via the worker's queue.update_progress() calls
        # Once execution starts, a checkpoint failure can follow paid work.
        # Propagate it; never build a fresh graph and replay from step one.
        final_state = cast(AgentState, await graph.ainvoke(state, config=graph_config))
        status = "complete"
    except RuntimeError as exc:
        logger.error(
            "LangGraph execution failed for mission %s: %s",
            request.mission_id,
            exc,
        )
        raise
    finally:
        if checkpointer_ctx is not None:
            await _close_checkpointer()

    # Generate report
    report_path = _ensure_report(final_state, request.mission_id)
    # S83 m06 (PD83-b) CARVE-OUT: see the sibling read above — tested local-runner
    # sources_count, kept rather than stripped.
    sources = final_state.get("sources_found") or []
    loops = int(final_state.get("research_loop_count", 0) or 0)

    # Final progress callback
    if progress_callback:
        progress_callback(loops, len(sources))

    duration_seconds = max(time.perf_counter() - run_started, 0.0)
    telemetry = _safe_telemetry(final_state)
    performance_metrics = _build_performance_metrics(
        duration=duration_seconds,
        loops=loops,
        telemetry=telemetry,
        backend=settings.llm_backend,
    )

    # Capture telemetry
    record_finding_records_from_state(final_state, telemetry_collector)
    extract_sources_from_state(final_state, telemetry_collector)
    telemetry_collector.set_cost(performance_metrics.estimated_cost_usd)
    telemetry["validation_summary"] = {
        "requested": False,
        "status": "not_requested",
    }
    completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    final_outcome = resolve_final_outcome_payload(final_state, completed_at=completed_at)
    telemetry["final_outcome"] = final_outcome
    final_state["mission_outcome"] = final_outcome["outcome"]
    final_state["routing_reason"] = final_outcome["routing_reason"]
    telemetry_collector.merge_state_telemetry(telemetry)
    mission_telemetry = telemetry_collector.finalize()
    telemetry_writer = TelemetryWriter()
    telemetry_path = telemetry_writer.write(mission_telemetry)
    if telemetry_path:
        telemetry["telemetry_artifact_path"] = telemetry_path
    final_state["telemetry"] = telemetry

    # Get protocol_path from final_state (finalize_node generates it)
    protocol_path = final_state.get("protocol_path")

    logger.info(
        "API mission %s finished with status %s after %d loops (%d sources)",
        request.mission_id,
        status,
        loops,
        len(sources),
    )

    return {
        "status": status,
        "mission_id": request.mission_id,
        "contract_id": state.get("mission_contract_id"),
        "contract_path": state.get("mission_contract_path"),
        "loops_executed": loops,
        "sources_count": len(sources),
        "report_path": report_path,
        "protocol_path": protocol_path,
        "total_duration_seconds": duration_seconds,
        "llm_backend": settings.llm_backend,
        "model": resolved_model,
        "thread_id": thread_identifier,
        "persistence_enabled": persistence_enabled,
        "persistence_error": persistence_error,
        "telemetry": telemetry,
        "telemetry_path": telemetry_path,
        "performance_metrics": performance_metrics.to_dict(),
        "final_outcome": final_outcome,
    }


def _extract_project_id_from_mission(mission_id: str) -> Optional[str]:
    """Extract project ID from mission ID prefix."""
    if "." in mission_id:
        prefix = mission_id.split(".", 1)[0].strip()
        return prefix or None
    if "-" in mission_id:
        prefix = mission_id.split("-", 1)[0].strip()
        return prefix or None
    return None


def _resolve_recursion_limit(depth_cfg: Mapping[str, Any], max_loops: Any) -> int:
    configured = int(depth_cfg.get("recursion_limit", 0) or 0)
    if configured > 0:
        return configured
    return calculate_recursion_limit(max_loops or 3)


__all__ = ["execute_research_mission", "run_research_from_request"]
