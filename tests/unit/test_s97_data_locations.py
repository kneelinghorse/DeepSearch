"""Data-root routing must stay late-bound and subordinate to explicit paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepsearch.agent.evidence import TelemetryWriter
from deepsearch.api.webhook import WebhookClient
from deepsearch.config import DeepSearchSettings
from deepsearch.data_paths import data_path
from deepsearch.eval.protocol_extract import _output_path_for
from deepsearch.mission.contract import _resolve_contract_dir
from deepsearch.report.generator import _resolve_report_directory
from deepsearch.tracelab.uploader import TraceLabUploader
from deepsearch.validation.hybrid_evaluator import HybridEvaluator

_PATH_ENV = (
    "DEEPSEARCH_DATA_ROOT", "DEEPSEARCH_REPORT_DIR", "DEEPSEARCH_TELEMETRY_DIR",
    "DEEPSEARCH_CONTRACT_DIR", "CHECKPOINT_DB_PATH",
)


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PATH_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("deepsearch.config.load_dotenv", lambda: None)
    monkeypatch.chdir(tmp_path)




@pytest.mark.parametrize("value", ["", "   "])
def test_blank_root_uses_gitignored_default(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DEEPSEARCH_DATA_ROOT", value)
    assert data_path("reports") == Path("checkpoints/reports")


def test_home_relative_root_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEARCH_DATA_ROOT", "~/deepsearch-runtime")
    assert data_path("reports") == Path.home() / "deepsearch-runtime/reports"


def test_explicit_paths_win_over_per_kind_environment_and_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEARCH_DATA_ROOT", str(tmp_path / "root"))
    for name in _PATH_ENV[1:]:
        monkeypatch.setenv(name, str(tmp_path / "per-kind"))
    explicit = tmp_path / "explicit"
    assert _resolve_contract_dir(explicit) == explicit
    assert TelemetryWriter(output_dir=explicit)._output_dir == explicit
    assert DeepSearchSettings(checkpoint_db_path=explicit).checkpoint_db_path == explicit
    assert WebhookClient(failed_payloads_dir=explicit).failed_payloads_dir == explicit
    assert HybridEvaluator(output_dir=explicit)._output_dir == explicit
    assert TraceLabUploader(telemetry_path=explicit).telemetry_path == explicit
    assert _output_path_for("retained", explicit) == explicit
    assert not (tmp_path / "root").exists()
    assert not (tmp_path / "per-kind").exists()


def test_per_kind_overrides_win_over_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEEPSEARCH_DATA_ROOT", str(tmp_path / "root"))
    for name in _PATH_ENV[1:]:
        monkeypatch.setenv(name, str(tmp_path / name))
    assert _resolve_report_directory() == tmp_path / "DEEPSEARCH_REPORT_DIR"
    assert _resolve_contract_dir(None) == tmp_path / "DEEPSEARCH_CONTRACT_DIR"
    assert TelemetryWriter()._output_dir == tmp_path / "DEEPSEARCH_TELEMETRY_DIR"
    assert DeepSearchSettings.load().checkpoint_db_path == tmp_path / "CHECKPOINT_DB_PATH"
    assert TraceLabUploader().telemetry_path == (
        tmp_path / "DEEPSEARCH_TELEMETRY_DIR/tracelab-uploads.jsonl"
    )
