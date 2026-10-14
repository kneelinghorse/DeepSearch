"""S69 m02 + S70 m01 — CLI wrapper for the FActScore local-eval harness.

Usage::

    python -m deepsearch.eval.factscore <mission_id> [--out PATH] [--budget-seconds N]
    python -m deepsearch.eval.factscore --protocol PATH [--sources PATH] ...
    python -m deepsearch.eval.factscore --calibrate <mission_id> <mission_id> ...
    python -m deepsearch.eval extract <mission_id> [--out PATH] [--database-url URL]

Exit codes:

* ``0`` — operation completed (the FActScore report itself may flag
  ``budget_exhausted=True`` or count entries as ``unknown`` — the harness
  reports those, it doesn't raise).
* ``2`` — argparse / cache-discovery failure (FileNotFoundError on
  protocol resolution, missing required arg, ``extract`` Postgres failure).

Pure stdlib (argparse + json + asyncio) plus the local-only
:mod:`deepsearch.eval.factscore` and :mod:`deepsearch.eval.protocol_extract`
modules. PD10 — no production-runtime imports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from deepsearch.eval.benchmark import (
    BenchmarkReport,
    benchmark_from_paths,
    compare_to_baseline,
    evaluate_go_no_go,
)
from deepsearch.eval.factscore import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_DECOMPOSITION_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_VERIFICATION_MODEL,
    FActScoreReport,
    resolve_protocol_path,
    score_mission,
)
from deepsearch.eval.protocol_extract import (
    ProtocolExtractError,
    extract_protocol,
)

DEFAULT_OUTPUT_DIR = Path("checkpoints/factscore")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepsearch.eval",
        description=(
            "Score a cached mission's claim-level faithfulness via DeepEval "
            "FaithfulnessMetric. Local-only; consumes cached protocol + "
            "stripped markdown + optional sources sidecar."
        ),
    )
    parser.add_argument(
        "mission_id",
        nargs="?",
        help=(
            "Mission identifier. Resolved against the protocol search dirs "
            "(<data root>/protocols, <data root>/reports). Mutually exclusive with "
            "--protocol and --calibrate."
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Direct path to the cached protocol JSON (bypasses discovery).",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=None,
        help="Optional sidecar JSON containing sources_collected (PD4).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            f"Output JSON path. Default: {DEFAULT_OUTPUT_DIR}/<mission_id>.json"
        ),
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=DEFAULT_BUDGET_SECONDS,
        help=f"Wall-clock budget per mission. Default: {DEFAULT_BUDGET_SECONDS}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"FaithfulnessMetric threshold. Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--decomposition-model",
        default=DEFAULT_DECOMPOSITION_MODEL,
        help=f"Decomposition model id. Default: {DEFAULT_DECOMPOSITION_MODEL}",
    )
    parser.add_argument(
        "--verification-model",
        default=DEFAULT_VERIFICATION_MODEL,
        help=f"Verification model id. Default: {DEFAULT_VERIFICATION_MODEL}",
    )
    parser.add_argument(
        "--calibrate",
        nargs="+",
        default=None,
        metavar="MISSION_ID",
        help=(
            "Calibration mode: score each listed mission_id sequentially "
            "and print a comparison table to stdout. m03 prep helper."
        ),
    )
    return parser


def _output_path_for(mission_id: str, out: Optional[Path]) -> Path:
    if out is not None:
        return out
    return DEFAULT_OUTPUT_DIR / f"{mission_id}.json"


def _write_report(report: FActScoreReport, out_path: Path) -> Path:
    """Persist ``report.to_dict()`` to ``out_path`` (parent dirs created)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def _resolve_protocol_for(
    mission_id: Optional[str],
    explicit: Optional[Path],
) -> Path:
    """Pick the protocol path: explicit ``--protocol`` wins; else discover."""
    if explicit is not None:
        return explicit
    if not mission_id:
        raise FileNotFoundError("Either mission_id or --protocol is required")
    return resolve_protocol_path(mission_id)


async def _score_one(
    mission_id: str,
    args: argparse.Namespace,
) -> FActScoreReport:
    protocol = _resolve_protocol_for(mission_id, args.protocol)
    return await score_mission(
        mission_id=mission_id,
        protocol_path=protocol,
        sources_path=args.sources,
        budget_seconds=args.budget_seconds,
        decomposition_model=args.decomposition_model,
        verification_model=args.verification_model,
        threshold=args.threshold,
    )


def _format_calibration_row(report: FActScoreReport) -> str:
    return (
        f"{report.mission_id:<48} "
        f"factscore={report.factscore:>5.3f} "
        f"decomposed={report.decomposed_claim_count:>4d} "
        f"supported={report.supported_count:>4d} "
        f"unsupported={report.unsupported_count:>4d} "
        f"unknown={report.unknown_count:>4d} "
        f"budget_exhausted={str(report.budget_exhausted):<5}"
    )


async def _run(args: argparse.Namespace) -> int:
    if args.calibrate:
        if args.protocol is not None or args.mission_id is not None:
            print(
                "error: --calibrate is mutually exclusive with mission_id / --protocol",
                file=sys.stderr,
            )
            return 2
        rows: List[FActScoreReport] = []
        for mission_id in args.calibrate:
            try:
                report = await _score_one(mission_id, args)
            except FileNotFoundError as exc:
                print(f"error: {mission_id}: {exc}", file=sys.stderr)
                return 2
            rows.append(report)
            out_path = _output_path_for(mission_id, args.out)
            _write_report(report, out_path)
            print(f"wrote {out_path}", file=sys.stderr)
        print("\nCalibration summary:")
        for row in rows:
            print(_format_calibration_row(row))
        return 0

    if args.mission_id is None and args.protocol is None:
        print(
            "error: either mission_id or --protocol is required",
            file=sys.stderr,
        )
        return 2

    mission_id = args.mission_id
    if mission_id is None:
        # ``--protocol`` without mission id: derive a placeholder from the
        # protocol stem so the report carries a stable identifier.
        stem = args.protocol.stem if args.protocol else "unknown"
        if stem.endswith("_protocol"):
            stem = stem[: -len("_protocol")]
        mission_id = stem.split("_", 1)[0] or stem or "unknown"

    try:
        report = await _score_one(mission_id, args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_path = _output_path_for(mission_id, args.out)
    _write_report(report, out_path)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


def _build_extract_parser() -> argparse.ArgumentParser:
    """Argparse for the ``extract`` subcommand (S70 m01).

    Kept distinct from the FActScore parser so the existing S69 invocation
    forms (``python -m deepsearch.eval <mission_id>``,
    ``--calibrate``, ``--protocol``) continue to work unchanged.
    """
    parser = argparse.ArgumentParser(
        prog="python -m deepsearch.eval extract",
        description=(
            "Read-only Postgres extractor: pulls a mission's "
            "result_protocol JSONB from TraceLab/Railway Postgres and "
            "writes it to local disk where the FActScore harness can "
            "score it. NO DB writes; NO TraceLab worker dispatch."
        ),
    )
    parser.add_argument(
        "mission_id",
        help=(
            "Human-readable mission_id (the missions.mission_id TEXT "
            "column; e.g. OODS-FIGMA-HOST-01-S68-WIRING). NOT the UUID."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: <data root>/protocols/<mission_id>_protocol.json "
            "(matches the path resolve_protocol_path walks via prefix match)."
        ),
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "Override the DATABASE_URL env var for this invocation. "
            "Defaults to os.environ['DATABASE_URL']."
        ),
    )
    return parser


def _run_extract(argv: Sequence[str]) -> int:
    parser = _build_extract_parser()
    args = parser.parse_args(list(argv))
    try:
        out = asyncio.run(
            extract_protocol(
                args.mission_id,
                conn_str=args.database_url,
                out_path=args.out,
            )
        )
    except ProtocolExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(str(out))
    return 0


def _build_benchmark_parser() -> argparse.ArgumentParser:
    """Argparse for the ``benchmark`` subcommand (S85 m02).

    Kept distinct from the FActScore parser so the existing invocation forms
    keep working. Scores one or more missions against the kill-switch rubric
    and prints the go/no-go rollup (decision #823).
    """
    parser = argparse.ArgumentParser(
        prog="python -m deepsearch.eval benchmark",
        description=(
            "Forensic kill-switch benchmark: gap-marker count, citation "
            "liveness sample sheet, telemetry-vs-truth diff, and contract-slot "
            "table per mission, rolled into the 4-threshold go/no-go verdict. "
            "Read-only; NO paid calls."
        ),
    )
    parser.add_argument(
        "mission_id",
        nargs="+",
        help=(
            "One or more mission ids (resolved against <data root>/protocols + "
            "<data root>/reports)."
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Direct protocol JSON path (valid only with a single mission_id).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/benchmark"),
        help="Directory for per-mission scoresheets. Default: checkpoints/benchmark",
    )
    parser.add_argument(
        "--scoresheet",
        type=Path,
        default=None,
        help="Path to write the combined go/no-go rollup + per-mission reports.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "When a protocol is not cached locally, extract it (+ report body) "
            "from Postgres first. Defaults to $DATABASE_URL. Read-only SELECT."
        ),
    )
    return parser


def _resolve_or_extract_protocol(
    mission_id: str,
    explicit: Optional[Path],
    database_url: Optional[str],
) -> Path:
    """Pick a protocol path: ``--protocol`` wins, else cached discovery, else
    (when a DB URL is available) a read-only Postgres extraction."""
    if explicit is not None:
        return explicit
    try:
        return resolve_protocol_path(mission_id)
    except FileNotFoundError:
        if database_url or os.environ.get("DATABASE_URL"):
            return asyncio.run(extract_protocol(mission_id, conn_str=database_url))
        raise


def _run_benchmark(argv: Sequence[str]) -> int:
    parser = _build_benchmark_parser()
    args = parser.parse_args(list(argv))
    if args.protocol is not None and len(args.mission_id) != 1:
        print("error: --protocol requires exactly one mission_id", file=sys.stderr)
        return 2

    reports: List[BenchmarkReport] = []
    for idx, mission_id in enumerate(args.mission_id):
        explicit = args.protocol if (args.protocol is not None and idx == 0) else None
        try:
            protocol_path = _resolve_or_extract_protocol(
                mission_id, explicit, args.database_url
            )
        except (FileNotFoundError, ProtocolExtractError, ValueError) as exc:
            print(f"error: {mission_id}: {exc}", file=sys.stderr)
            return 2
        report = benchmark_from_paths(mission_id, protocol_path)
        reports.append(report)
        out_path = args.out / f"{mission_id}_benchmark.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_path}", file=sys.stderr)

    rollup = evaluate_go_no_go(reports)
    baseline_comparison = compare_to_baseline(reports)
    output = {"rollup": rollup, "baseline_comparison": baseline_comparison}
    print(json.dumps(output, indent=2))
    if args.scoresheet is not None:
        args.scoresheet.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rollup": rollup,
            "baseline_comparison": baseline_comparison,
            "reports": [r.to_dict() for r in reports],
        }
        args.scoresheet.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote scoresheet {args.scoresheet}", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if args_list and args_list[0] == "comparison":
        if not Path(__file__).with_name("comparison.py").is_file():
            print("The archived comparison command is available only in the private repository.",
                  file=sys.stderr)
            return 2
        from deepsearch.eval.comparison import verify_manifest

        print(json.dumps(verify_manifest(), indent=2))
        return 0
    if args_list and args_list[0] == "extract":
        return _run_extract(args_list[1:])
    if args_list and args_list[0] == "benchmark":
        return _run_benchmark(args_list[1:])
    parser = _build_parser()
    args = parser.parse_args(args_list)
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover — invoked via python -m
    raise SystemExit(main())
