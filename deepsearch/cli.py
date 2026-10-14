"""Command-line interface for executing DeepSearch missions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from dotenv import load_dotenv

# Load .env EARLY before any other imports that might need env vars
load_dotenv(verbose=True)  # verbose=True helps debug loading issues

LOG_FILE = Path("deepsearch.log")
LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser used to run missions.

    The ``deepsearch-run`` console script and ``python -m deepsearch.run_mission`` both
    expect the mission path as the first positional argument:

        deepsearch-run path/to/mission.yaml
    """

    parser = argparse.ArgumentParser(
        description="DeepSearch - Autonomous research agent.",
        prog="deepsearch-run",
    )
    parser.add_argument(
        "mission_file",
        type=Path,
        help="Path to a Mission Protocol YAML file",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "gemini", "deepseek", "minimax", "openai"],
        default=None,
        help="LLM backend to use for research (defaults to env/config)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the selected backend (defaults to env/config)",
    )
    parser.add_argument(
        "--depth",
        choices=["baseline", "deep", "alpha"],
        default=None,
        help="Override research depth tier (baseline/deep/alpha)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the mission without executing LangGraph",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing checkpoints (disable with --no-resume)",
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Delete checkpoints for this mission after execution",
    )
    parser.add_argument(
        "--show-checkpoints",
        action="store_true",
        help="Print checkpoint history after the run completes",
    )
    parser.add_argument(
        "--thread-id",
        help="Override the deterministic thread ID when running concurrent sessions",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run hybrid quality validation after the mission completes",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        help="Directory to store validation metrics (default: <data root>/validation)",
    )
    parser.add_argument(
        "--auto-upload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-upload to TraceLab if quality gates pass (disable with --no-auto-upload)",
    )
    parser.add_argument(
        "--force-loops",
        type=int,
        default=None,
        metavar="N",
        help="Force minimum N loops before convergence can trigger (B7.5.1)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    return parser


def _build_health_parser() -> argparse.ArgumentParser:
    """Return the argument parser used by ``deepsearch-run health``."""

    parser = argparse.ArgumentParser(
        description="Verify all required services are available.",
        prog="deepsearch-run health",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output health status as JSON",
    )
    return parser


def _build_compare_contract_parser() -> argparse.ArgumentParser:
    """Return the parser used by ``deepsearch-run compare-contract``."""

    parser = argparse.ArgumentParser(
        description="Run the local-vs-deployed contract validation harness.",
        prog="deepsearch-run compare-contract",
    )
    parser.add_argument(
        "--mission-file",
        type=Path,
        help="Optional mission YAML to execute locally before comparison.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "gemini", "deepseek", "minimax", "openai"],
        default=None,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--local-contract", type=Path, help="Existing local contract artifact.")
    parser.add_argument("--local-report", type=Path, help="Existing local report artifact.")
    parser.add_argument("--local-telemetry", type=Path, help="Existing local telemetry artifact.")
    parser.add_argument("--local-validation", type=Path, help="Existing local validation metrics.")
    parser.add_argument("--deployed-contract", type=Path, required=True)
    parser.add_argument("--deployed-report", type=Path, required=True)
    parser.add_argument("--deployed-telemetry", type=Path, required=True)
    parser.add_argument("--deployed-validation", type=Path)
    parser.add_argument("--local-reviewer", type=Path, help="Optional local reviewer attachment.")
    parser.add_argument(
        "--deployed-reviewer",
        type=Path,
        help="Optional deployed reviewer attachment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the comparison JSON artifact.",
    )
    return parser


def _build_review_parser() -> argparse.ArgumentParser:
    """Return the parser used by ``deepsearch review``."""

    parser = argparse.ArgumentParser(
        description="Capture reviewer adjudication for an acceptance artifact bundle.",
        prog="deepsearch review",
    )
    parser.add_argument("--contract-id", required=True, help="Acceptance contract ID.")
    parser.add_argument("--run-id", required=True, help="Run identifier used in the reviewer JSON.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help=(
            "Artifact bundle directory containing contract, report, protocol, "
            "and telemetry files."
        ),
    )
    parser.add_argument(
        "--environment",
        choices=["local", "deployed"],
        help="Execution environment for the artifact bundle.",
    )
    parser.add_argument("--reviewer", help="Reviewer identifier, for example 'derek'.")
    parser.add_argument(
        "--verdict",
        choices=["accept", "accept_with_minor_notes", "revise", "reject"],
        help="Final reviewer verdict.",
    )
    parser.add_argument(
        "--scores",
        help="JSON object containing rubric scores, e.g. '{\"contract_fit\": 3, ...}'.",
    )
    parser.add_argument(
        "--rationales",
        help="JSON object containing one rationale per rubric category.",
    )
    return parser


def _run_health(args: argparse.Namespace) -> int:
    """Execute the health check command."""
    from deepsearch.health import overall_health, print_health_report

    if args.json:
        result = overall_health()
        print(json.dumps(result, indent=2))
        return 0 if result["healthy"] else 1
    else:
        healthy = print_health_report()
        return 0 if healthy else 1


def _run_compare_contract(args: argparse.Namespace) -> int:
    """Execute the contract validation harness."""

    from deepsearch.validation import run_validation_harness

    result = asyncio.run(
        run_validation_harness(
            mission_path=args.mission_file,
            local_contract_path=args.local_contract,
            local_report_path=args.local_report,
            local_telemetry_path=args.local_telemetry,
            local_validation_path=args.local_validation,
            deployed_contract_path=args.deployed_contract,
            deployed_report_path=args.deployed_report,
            deployed_telemetry_path=args.deployed_telemetry,
            deployed_validation_path=args.deployed_validation,
            local_reviewer_path=args.local_reviewer,
            deployed_reviewer_path=args.deployed_reviewer,
            llm_backend=args.llm_backend,
            model=args.model,
            output_path=args.output,
        )
    )

    print(json.dumps(result, indent=2))
    return 0 if result.get("parity_passed") else 1


def _run_review(args: argparse.Namespace) -> int:
    """Execute the reviewer adjudication command."""

    from deepsearch.validation.reviewer import (
        REQUIRED_REVIEWER_CATEGORIES,
        ReviewerAdjudication,
    )

    try:
        bundle = _load_review_bundle(
            artifact_dir=args.artifact_dir.expanduser(),
            contract_id=args.contract_id,
        )
        scores = (
            _parse_review_mapping(args.scores, cast=int)
            if args.scores
            else _prompt_review_scores(REQUIRED_REVIEWER_CATEGORIES)
        )
        rationales = (
            _parse_review_mapping(args.rationales, cast=str)
            if args.rationales
            else _prompt_review_rationales(REQUIRED_REVIEWER_CATEGORIES)
        )
        reviewer = args.reviewer or _prompt_text("Reviewer")
        environment = (
            args.environment
            or _infer_environment(args.artifact_dir.expanduser())
            or _prompt_choice("Environment", ("local", "deployed"))
        )
        verdict = args.verdict or _prompt_choice(
            "Verdict",
            ("accept", "accept_with_minor_notes", "revise", "reject"),
        )
        adjudication = ReviewerAdjudication(
            contract_id=args.contract_id,
            run_id=args.run_id,
            environment=environment,
            reviewer=reviewer,
            rubric_scores=scores,
            per_category_rationale=rationales,
            verdict=verdict,
            adjudicated_at=ReviewerAdjudication.new_timestamp(),
        )
        reviewer_path = adjudication.write_to(args.artifact_dir.expanduser())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "reviewer_path": str(reviewer_path),
                "artifact_bundle": {key: str(value) for key, value in bundle.items()},
                "reviewer_adjudication": adjudication.to_dict(),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute the mission, and emit a summary."""

    argv_list = list(argv) if argv is not None else sys.argv[1:]

    # Support: deepsearch-run health [--json]
    if argv_list and argv_list[0] == "health":
        health_args = _build_health_parser().parse_args(argv_list[1:])
        return _run_health(health_args)
    if argv_list and argv_list[0] == "compare-contract":
        compare_args = _build_compare_contract_parser().parse_args(argv_list[1:])
        return _run_compare_contract(compare_args)
    if argv_list and argv_list[0] == "review":
        review_args = _build_review_parser().parse_args(argv_list[1:])
        return _run_review(review_args)

    # Default: deepsearch-run <mission-yaml> [options]
    parser = build_parser()
    if not argv_list:
        parser.print_help()
        print(
            "\nAdditional commands:\n"
            "  deepsearch-run health [--json]\n"
            "  deepsearch-run compare-contract [options]\n"
            "  deepsearch review [options]",
            file=sys.stderr,
        )
        return 0

    args = parser.parse_args(argv_list)
    mission_file = args.mission_file.expanduser()
    if not mission_file.exists():
        parser.error(f"Mission file not found: {mission_file}")

    # Import heavy modules only for run command
    from deepsearch.mission import MissionLoadError
    from deepsearch.runner import execute_research_mission

    _configure_logging(verbose=args.verbose)
    logger = logging.getLogger("deepsearch.cli")
    logger.info("DeepSearch mission runner initialised")
    logger.info("Mission file: %s", mission_file)
    if args.llm_backend or args.model:
        logger.info("LLM overrides: backend=%s model=%s", args.llm_backend, args.model)

    try:
        result = asyncio.run(
            execute_research_mission(
                mission_path=mission_file,
                llm_backend=args.llm_backend,
                model=args.model,
                dry_run=args.dry_run,
                resume=args.resume,
                clear_checkpoints=args.clear_checkpoints,
                show_checkpoints=args.show_checkpoints,
                thread_id=args.thread_id,
                validate=args.validate,
                validation_output=args.validation_output,
                auto_upload=args.auto_upload,
                force_loops=args.force_loops,
                research_depth=args.depth,
            )
        )
    except MissionLoadError as exc:
        logger.error("Mission validation failed: %s", exc)
        parser.error(str(exc))
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        logger.exception("Mission execution failed")
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    _render_summary(result, show_checkpoints=args.show_checkpoints)
    return 0


def _configure_logging(*, verbose: bool) -> None:
    """Configure console and file logging according to CLI flags."""

    level = logging.DEBUG if verbose else logging.INFO
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
        force=True,
    )


def _render_summary(result: Dict[str, Any], *, show_checkpoints: bool) -> None:
    """Print a concise execution summary after the mission completes."""

    heading = "Dry Run Complete" if result.get("status") == "validated" else "Research Complete"
    report_path = result.get("report_path") or "not generated"
    persistence_status = "enabled" if result.get("persistence_enabled") else "disabled"
    if result.get("persistence_error"):
        persistence_status += f" ({result['persistence_error']})"
    lines = [
        "",
        "=" * 60,
        heading,
        "=" * 60,
        f"Mission ID    : {result.get('mission_id', 'unknown')}",
        f"Status        : {result.get('status', 'unknown')}",
        f"Contract ID   : {result.get('contract_id', 'n/a')}",
        f"Loops Executed: {result.get('loops_executed', 0)}",
        f"Sources Found : {result.get('sources_count', 0)}",
        f"Report Path   : {report_path}",
        f"Duration      : {result.get('total_duration_seconds', 0.0):.2f}s",
    ]
    if result.get("contract_path"):
        lines.append(f"Contract Path : {result['contract_path']}")
    lines.append(f"Thread ID     : {result.get('thread_id', 'n/a')}")
    lines.append(f"Checkpoints   : {persistence_status}")
    if result.get("checkpoint_db_path"):
        lines.append(f"Checkpoint DB : {result['checkpoint_db_path']}")
    performance = result.get("performance_metrics") or {}
    if performance:
        lines.append(f"API Calls     : {performance.get('api_calls', {})}")
        lines.append(f"Tokens Used   : {performance.get('token_usage', {})}")
        lines.append(f"Est. Cost USD : {performance.get('estimated_cost_usd', 0.0):.4f}")
    validation = result.get("validation")
    validation_error = result.get("validation_error")
    if isinstance(validation, dict):
        scores = validation.get("quality_scores") or {}
        faith = _coerce_float(scores, "faithfulness")
        relevance = _coerce_float(scores, "relevance")
        citation = _coerce_float(scores, "citation_rate")
        lines.append(
            "Validation    : PASS "
            f"(faith={faith:.2f}, relevance={relevance:.2f}, citation={citation:.2f})"
        )
        if validation.get("output_path"):
            lines.append(f"Validation Log: {validation['output_path']}")
    elif validation_error:
        lines.append(f"Validation    : FAILED ({validation_error})")
    upload = result.get("upload")
    upload_error = result.get("upload_error")
    if isinstance(upload, dict):
        if upload.get("skipped"):
            failing = upload.get("failing_gates", [])
            lines.append(f"Upload        : SKIPPED (gates: {', '.join(failing[:2])})")
        elif upload.get("success"):
            lines.append(f"Upload        : SUCCESS (uuid={upload.get('mission_uuid')})")
        else:
            errors = upload.get("errors", ["unknown error"])
            lines.append(f"Upload        : FAILED ({errors[0][:50]})")
    elif upload_error:
        lines.append(f"Upload        : FAILED ({upload_error[:50]})")
    lines.extend(
        [
            "=" * 60,
            "",
        ]
    )
    print("\n".join(lines))
    if show_checkpoints:
        history = result.get("checkpoint_history")
        if not history:
            print("No checkpoint history available.")
            return
        print("Checkpoint History:")
        for entry in history:
            print(
                f"- id={entry.get('checkpoint_id')} parent={entry.get('parent_checkpoint_id')} "
                f"metadata={entry.get('metadata')}"
            )


def _coerce_float(container: Dict[str, Any], key: str) -> float:
    value = container.get(key, 0.0) if isinstance(container, dict) else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_review_bundle(*, artifact_dir: Path, contract_id: str) -> Dict[str, Path]:
    """Resolve the required artifact bundle files for reviewer adjudication."""

    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
    if not artifact_dir.is_dir():
        raise ValueError(f"Artifact path is not a directory: {artifact_dir}")

    bundle = {
        "contract": _resolve_bundle_artifact(
            artifact_dir,
            label="contract",
            patterns=("contract.json", "mission-*.json", "*contract*.json"),
        ),
        "report": _resolve_bundle_artifact(
            artifact_dir,
            label="report",
            patterns=("report.md", "*_report.md", "*.md"),
            excluded_names={"failure_modes.md"},
        ),
        "protocol": _resolve_bundle_artifact(
            artifact_dir,
            label="protocol",
            patterns=("protocol.json", "*_protocol.json"),
        ),
        "telemetry": _resolve_bundle_artifact(
            artifact_dir,
            label="telemetry",
            patterns=("telemetry.jsonl", "*.jsonl"),
        ),
    }

    contract_payload = json.loads(bundle["contract"].read_text(encoding="utf-8"))
    if not isinstance(contract_payload, Mapping):
        raise ValueError("Contract artifact must contain a JSON object")
    mission_id = str(contract_payload.get("mission_id") or "").strip()
    if mission_id and mission_id != contract_id:
        raise ValueError(
            "Artifact contract mission_id "
            f"{mission_id!r} does not match --contract-id {contract_id!r}"
        )
    return bundle


def _resolve_bundle_artifact(
    artifact_dir: Path,
    *,
    label: str,
    patterns: Sequence[str],
    excluded_names: Iterable[str] = (),
) -> Path:
    """Return the first matching bundle artifact for ``label``."""

    excluded = set(excluded_names)
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in patterns:
        for candidate in sorted(artifact_dir.glob(pattern)):
            if candidate.name in excluded or candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"Required {label} artifact not found in {artifact_dir}")
    return candidates[0]


def _parse_review_mapping(
    payload: str,
    *,
    cast: type[int] | type[str],
) -> Dict[str, Any]:
    """Parse a CLI JSON mapping into primitive reviewer payload values."""

    parsed = json.loads(payload)
    if not isinstance(parsed, Mapping):
        raise ValueError("Expected a JSON object")
    normalized: Dict[str, Any] = {}
    for key, value in parsed.items():
        text_key = str(key)
        normalized[text_key] = cast(value)
    return normalized


def _prompt_review_scores(categories: Sequence[str]) -> Dict[str, int]:
    """Interactively collect rubric scores."""

    scores: Dict[str, int] = {}
    for category in categories:
        while True:
            raw_value = input(f"{category} score (0-3): ").strip()
            try:
                score = int(raw_value)
            except ValueError:
                print("Enter an integer between 0 and 3.", file=sys.stderr)
                continue
            scores[category] = score
            break
    return scores


def _prompt_review_rationales(categories: Sequence[str]) -> Dict[str, str]:
    """Interactively collect one rationale per rubric category."""

    rationales: Dict[str, str] = {}
    for category in categories:
        rationales[category] = _prompt_text(f"{category} rationale")
    return rationales


def _prompt_text(label: str) -> str:
    """Prompt until a non-empty text value is provided."""

    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print(f"{label} is required.", file=sys.stderr)


def _prompt_choice(label: str, choices: Sequence[str]) -> str:
    """Prompt until one of ``choices`` is selected."""

    choice_list = ", ".join(choices)
    while True:
        value = input(f"{label} ({choice_list}): ").strip()
        if value in choices:
            return value
        print(f"{label} must be one of {choice_list}.", file=sys.stderr)


def _infer_environment(artifact_dir: Path) -> str | None:
    """Infer the environment from common artifact directory naming patterns."""

    parts = [artifact_dir.name.lower(), str(artifact_dir).lower()]
    if any(part.endswith("_local") or "/local" in part for part in parts):
        return "local"
    if any(part.endswith("_deployed") or "/deployed" in part for part in parts):
        return "deployed"
    return None


def console_main() -> None:
    """Entry point used by the ``deepsearch-run`` console script."""

    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover - manual execution
    console_main()
