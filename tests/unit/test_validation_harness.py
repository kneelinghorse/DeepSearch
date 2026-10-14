"""Unit tests for the contract validation harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch import cli  # noqa: E402
from deepsearch.validation import (  # noqa: E402
    EnvironmentBundle,
    StructuralCheck,
    StructuralValidationSummary,
    compare_bundles,
    evaluate_structural_validation,
    save_comparison_result,
)


def _bundle(
    *,
    environment: str,
    runtime_identity: dict[str, object] | None = None,
    validation_summary: dict[str, object] | None = None,
    structural_passed: bool = True,
    reviewer_attached: bool = False,
) -> EnvironmentBundle:
    return EnvironmentBundle(
        environment=environment,
        mission_id="ACC.1",
        contract_id="contract-123",
        contract_path=f"{environment}-contract.json",
        report_path=f"{environment}-report.md",
        telemetry_path=f"{environment}-telemetry.jsonl",
        validation_path=f"{environment}-metrics.json",
        runtime_identity={
            "build_hash": "abc123",
            "build_source": "git",
            "git_dirty": False,
            "llm_backend": "ollama",
            "model": "llama3",
            "execution_path": "yaml",
            "python_version": "3.12.0",
            "platform": "darwin",
            **(runtime_identity or {}),
        },
        config_fingerprint="fingerprint-1",
        route_summary={"next_action": "finalize", "reason": "quality_gates_passed"},
        final_outcome={"outcome": "complete", "routing_reason": "converged_with_coverage"},
        validation_summary={"status": "passed", "failed_checks": [], **(validation_summary or {})},
        structural_validation=StructuralValidationSummary(
            passed=structural_passed,
            checks=[
                StructuralCheck(
                    schema_id="deliverable-schema-1",
                    title="Comparison Table",
                    kind="table",
                    passed=structural_passed,
                    detail="ok" if structural_passed else "failed",
                )
            ],
        ),
        performance_metrics={"loops_executed": 2},
        reviewer_adjudication={"score": "accept"} if reviewer_attached else None,
    )


def test_compare_bundles_allows_missing_reviewer_adjudication() -> None:
    """Reviewer attachment is optional and should not fail parity by itself."""

    comparison = compare_bundles(
        _bundle(environment="local"),
        _bundle(environment="deployed"),
    )

    payload = comparison.to_dict()
    assert comparison.parity_passed is True
    assert comparison.differences == []
    assert payload["checks"]["local_reviewer_attached"] is False
    assert payload["checks"]["deployed_reviewer_attached"] is False


def test_compare_bundles_flags_runtime_identity_mismatch() -> None:
    """Stable runtime identity differences should be surfaced in the comparison."""

    comparison = compare_bundles(
        _bundle(environment="local", runtime_identity={"build_hash": "abc123"}),
        _bundle(environment="deployed", runtime_identity={"build_hash": "def456"}),
    )

    payload = comparison.to_dict()
    assert comparison.parity_passed is False
    assert "Runtime identity mismatch between local and deployed bundles." in comparison.differences
    assert payload["checks"]["runtime_identity_match"] is False
    assert payload["runtime_identity_diff"]["build_hash"] == {
        "local": "abc123",
        "deployed": "def456",
    }


def test_evaluate_structural_validation_returns_empty_passing_summary_post_s76(
    tmp_path: Path,
) -> None:
    """Post-S76 structural_validators deletion: harness emits an empty pass-through summary."""

    contract_path = tmp_path / "contract.json"
    report_path = tmp_path / "report.md"
    contract_path.write_text("{}", encoding="utf-8")
    report_path.write_text("# stub\n", encoding="utf-8")

    result = evaluate_structural_validation(
        contract_path=contract_path,
        report_path=report_path,
    )

    assert result.passed is True
    assert result.checks == []


def test_save_comparison_result_creates_parent_directories(tmp_path: Path) -> None:
    """Saving a comparison should create parent directories and markdown output."""

    comparison = compare_bundles(
        _bundle(environment="local", reviewer_attached=True),
        _bundle(environment="deployed", reviewer_attached=True),
    )

    saved = save_comparison_result(
        comparison,
        output_path=tmp_path / "nested" / "comparison-output",
    )

    assert saved["json"].exists()
    assert saved["markdown"].exists()
    assert saved["json"].suffix == ".json"
    payload = json.loads(saved["json"].read_text(encoding="utf-8"))
    assert payload["checks"]["runtime_identity_match"] is True


def test_compare_contract_cli_invokes_harness_and_emits_json(
    monkeypatch,
    capsys,
) -> None:
    """The compare-contract CLI should run the harness and print JSON."""

    import deepsearch.validation as validation

    recorded: dict[str, object] = {}

    async def _fake_harness(**kwargs: object) -> dict[str, object]:
        recorded.update(kwargs)
        return {
            "mission_id": "ACC.1",
            "parity_passed": True,
            "differences": [],
        }

    monkeypatch.setattr(validation, "run_validation_harness", _fake_harness)

    exit_code = cli.main(
        [
            "compare-contract",
            "--deployed-contract",
            "deployed-contract.json",
            "--deployed-report",
            "deployed-report.md",
            "--deployed-telemetry",
            "deployed-telemetry.jsonl",
            "--deployed-reviewer",
            "reviewer.json",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["parity_passed"] is True
    assert recorded["deployed_reviewer_path"] == Path("reviewer.json")


def test_runner_imports_without_validation_cycle() -> None:
    """Importing the runner should not trip a validation harness cycle."""

    result = subprocess.run(
        [sys.executable, "-c", "import deepsearch.runner"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
