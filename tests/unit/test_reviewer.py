"""Unit tests for reviewer adjudication schema and CLI plumbing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch import cli  # noqa: E402
from deepsearch.validation.harness import load_reviewer_adjudication  # noqa: E402
from deepsearch.validation.reviewer import (  # noqa: E402
    REQUIRED_REVIEWER_CATEGORIES,
    ReviewerAdjudication,
)


def _valid_scores() -> dict[str, int]:
    return {category: 3 for category in REQUIRED_REVIEWER_CATEGORIES}


def _valid_rationales() -> dict[str, str]:
    return {
        category: f"{category} rationale is complete and reviewable."
        for category in REQUIRED_REVIEWER_CATEGORIES
    }


def _make_adjudication() -> ReviewerAdjudication:
    return ReviewerAdjudication(
        contract_id="ACC-TABLE-COMP-01",
        run_id="run-001",
        environment="local",
        reviewer="derek",
        rubric_scores=_valid_scores(),
        per_category_rationale=_valid_rationales(),
        verdict="accept",
        adjudicated_at="2026-04-20T00:00:00+00:00",
    )


def _write_bundle(tmp_path: Path, *, contract_id: str = "ACC-TABLE-COMP-01") -> Path:
    artifact_dir = tmp_path / "run-001_local"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "contract.json").write_text(
        json.dumps(
            {
                "mission_id": contract_id,
                "contract_id": "compiled-contract-123",
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (artifact_dir / "protocol.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    return artifact_dir


def test_reviewer_adjudication_round_trips_json(tmp_path: Path) -> None:
    adjudication = _make_adjudication()

    path = adjudication.write_to(tmp_path)
    loaded = ReviewerAdjudication.read_from(path)

    assert loaded == adjudication


def test_reviewer_adjudication_rejects_missing_score_category() -> None:
    adjudication = _make_adjudication()
    adjudication.rubric_scores.pop("contract_fit")

    with pytest.raises(ValueError, match="rubric_scores categories invalid"):
        adjudication.validate()


def test_reviewer_adjudication_rejects_missing_rationale_category() -> None:
    adjudication = _make_adjudication()
    adjudication.per_category_rationale.pop("contract_fit")

    with pytest.raises(ValueError, match="per_category_rationale categories invalid"):
        adjudication.validate()


def test_reviewer_adjudication_rejects_score_out_of_range() -> None:
    adjudication = _make_adjudication()
    adjudication.rubric_scores["contract_fit"] = 4

    with pytest.raises(ValueError, match="contract_fit score must be between 0 and 3"):
        adjudication.validate()


def test_reviewer_adjudication_rejects_accept_verdict_below_average_threshold() -> None:
    adjudication = _make_adjudication()
    adjudication.rubric_scores = {
        "contract_fit": 2,
        "structural_integrity": 2,
        "bibliography_quality": 2,
        "corpus_grounding": 2,
        "practical_usefulness": 2,
    }

    with pytest.raises(ValueError, match="average >= 2.5"):
        adjudication.validate()


def test_reviewer_adjudication_rejects_invalid_environment() -> None:
    adjudication = _make_adjudication()
    adjudication.environment = "staging"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="environment must be one of"):
        adjudication.validate()


def test_load_reviewer_adjudication_uses_typed_schema(tmp_path: Path) -> None:
    path = _make_adjudication().write_to(tmp_path)

    payload = load_reviewer_adjudication(path)

    assert payload is not None
    assert payload["contract_id"] == "ACC-TABLE-COMP-01"
    assert payload["reviewer"] == "derek"


def test_review_cli_help_shows_review_command(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["review", "--help"])

    assert excinfo.value.code == 0
    assert "deepsearch review" in capsys.readouterr().out


def test_review_cli_exits_non_zero_for_invalid_scores(tmp_path: Path, capsys) -> None:
    artifact_dir = _write_bundle(tmp_path)

    exit_code = cli.main(
        [
            "review",
            "--contract-id",
            "ACC-TABLE-COMP-01",
            "--run-id",
            "run-001",
            "--artifact-dir",
            str(artifact_dir),
            "--environment",
            "local",
            "--reviewer",
            "derek",
            "--verdict",
            "accept",
            "--scores",
            json.dumps({**_valid_scores(), "contract_fit": 4}),
            "--rationales",
            json.dumps(_valid_rationales()),
        ]
    )

    assert exit_code == 1
    assert "between 0 and 3" in capsys.readouterr().err


def test_review_cli_writes_reviewer_json(tmp_path: Path) -> None:
    artifact_dir = _write_bundle(tmp_path)

    exit_code = cli.main(
        [
            "review",
            "--contract-id",
            "ACC-TABLE-COMP-01",
            "--run-id",
            "run-001",
            "--artifact-dir",
            str(artifact_dir),
            "--environment",
            "local",
            "--reviewer",
            "derek",
            "--verdict",
            "accept_with_minor_notes",
            "--scores",
            json.dumps(_valid_scores()),
            "--rationales",
            json.dumps(_valid_rationales()),
        ]
    )

    reviewer_path = artifact_dir / "run-001_reviewer.json"

    assert exit_code == 0
    assert reviewer_path.exists()
    payload = json.loads(reviewer_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "accept_with_minor_notes"
