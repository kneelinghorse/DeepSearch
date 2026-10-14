"""Runtime regression guards for the S69 FActScore harness.

Mirrors the layered-guard pattern from S65/S66/S67/S68/S68.5: this file
covers happy-path + edge cases via a fake DeepEval surface monkeypatched
in to ``deepsearch.eval.factscore._resolve_deepeval``. No real LLM call
ever runs; the queue mechanism in ``fake_deepeval`` lets each test
prescribe per-citation verdicts.

m02 extension: ``resolve_protocol_path`` discovery + CLI runtime cases
(``deepsearch.eval.__main__.main``). The CLI exit-code contract is
locked here — ``0`` on score, ``2`` on missing protocol / argparse error.

Static structural guards live in ``test_factscore_invariants.py``.
HD4 round-trip persistence in ``tests/integration/test_factscore_persistence.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import pytest

from deepsearch.eval import __main__ as factscore_cli
from deepsearch.eval.factscore import (
    DEFAULT_DECOMPOSITION_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_VERIFICATION_MODEL,
    FActScoreReport,
    HARNESS_VERSION,
    resolve_protocol_path,
    score_mission,
)


# ---------------------------------------------------------------------------
# Fakes + fixtures.
# ---------------------------------------------------------------------------


class _FakeVerdict:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict


@pytest.fixture
def fake_deepeval(monkeypatch):
    """Monkeypatch ``_resolve_deepeval`` to return fake metric/test-case classes.

    Returns the FIFO outcome queue. Each test pre-populates it with one
    ``(verdicts_list, score)`` tuple per probed citation. When the queue
    drains, the fake metric defaults to ``(["yes"], 1.0)``.
    """
    queue: List[Tuple[List[str], float]] = []

    class _FakeMetric:
        def __init__(self, threshold: float, model=None, **_kwargs) -> None:
            self.threshold = threshold
            self.model = model
            self.score = 0.0
            self.verdicts: List[_FakeVerdict] = []

        def measure(self, _test_case) -> None:
            if queue:
                verdict_strs, score = queue.pop(0)
            else:
                verdict_strs, score = ["yes"], 1.0
            self.verdicts = [_FakeVerdict(v) for v in verdict_strs]
            self.score = score

    class _FakeLLMTestCase:
        def __init__(self, *, input=None, actual_output=None, retrieval_context=None) -> None:
            self.input = input
            self.actual_output = actual_output
            self.retrieval_context = retrieval_context

    monkeypatch.setattr(
        "deepsearch.eval.factscore._resolve_deepeval",
        lambda: (_FakeMetric, _FakeLLMTestCase),
    )
    return queue


@pytest.fixture
def cached_mission(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Write a small protocol JSON + sibling markdown + sources sidecar."""
    markdown = "Lambda is serverless. RDS uses MySQL by default."
    # Indices align with the markdown above.
    protocol = {
        "mission_id": "TEST-MISSION-01",
        "citations": [
            {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 20,
                "url": "https://aws.amazon.com/lambda",
                "title": "AWS Lambda",
                "live": True,
            },
            {
                "type": "url_citation",
                "start_index": 22,
                "end_index": 47,
                "url": "https://aws.amazon.com/rds",
                "title": "AWS RDS",
                "live": True,
            },
        ],
    }
    sources = [
        {
            "url": "https://aws.amazon.com/lambda",
            "text": "AWS Lambda is a serverless compute service that runs code on demand.",
        },
        {
            "url": "https://aws.amazon.com/rds",
            "text": "Amazon RDS for MySQL: a fully managed MySQL relational database service.",
        },
    ]
    proto = tmp_path / "TEST-MISSION-01_research_protocol.json"
    md = tmp_path / "TEST-MISSION-01_research.md"
    src = tmp_path / "TEST-MISSION-01_sources.json"
    proto.write_text(json.dumps(protocol), encoding="utf-8")
    md.write_text(markdown, encoding="utf-8")
    src.write_text(json.dumps(sources), encoding="utf-8")
    return proto, md, src


# ---------------------------------------------------------------------------
# Runtime guards.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_mission_happy_path_all_supported(fake_deepeval, cached_mission):
    proto, _, src = cached_mission
    fake_deepeval.extend([(["yes", "yes"], 1.0), (["yes"], 1.0)])

    report = await score_mission("TEST-MISSION-01", proto, src)

    assert isinstance(report, FActScoreReport)
    assert report.mission_id == "TEST-MISSION-01"
    assert report.decomposed_claim_count == 3
    assert report.supported_count == 3
    assert report.unsupported_count == 0
    assert report.unknown_count == 0
    assert report.factscore == pytest.approx(1.0)
    assert report.budget_exhausted is False
    assert report.unsupported_count_per_citation == {}
    assert report.decomposition_model_id == DEFAULT_DECOMPOSITION_MODEL
    assert report.verification_model_id == DEFAULT_VERIFICATION_MODEL
    assert report.threshold == DEFAULT_THRESHOLD
    assert report.harness_version == HARNESS_VERSION


@pytest.mark.asyncio
async def test_score_mission_empty_citations_returns_zero(fake_deepeval, tmp_path):
    proto = tmp_path / "EMPTY_research_protocol.json"
    proto.write_text(
        json.dumps({"mission_id": "EMPTY", "citations": []}),
        encoding="utf-8",
    )

    report = await score_mission("EMPTY", proto)

    assert report.decomposed_claim_count == 0
    assert report.supported_count == 0
    assert report.unsupported_count == 0
    assert report.unknown_count == 0
    assert report.factscore == 0.0
    assert report.budget_exhausted is False
    assert report.unsupported_count_per_citation == {}


@pytest.mark.asyncio
async def test_score_mission_classifies_missing_bodies_as_unknown(
    fake_deepeval, tmp_path
):
    """Live citations whose URL is absent from sources_collected ⇒ unknown.

    PD4: missing body ≠ unsupported; the harness can't probe what it doesn't
    have. Each missing-body citation contributes ``(1 atom, 1 unknown)``.
    """
    md = "Claim alpha. Claim beta."
    protocol = {
        "mission_id": "GAPS",
        "citations": [
            {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 12,
                "url": "https://nowhere.example/a",
                "live": True,
            },
            {
                "type": "url_citation",
                "start_index": 14,
                "end_index": 25,
                "url": "https://nowhere.example/b",
                "live": True,
            },
        ],
    }
    proto = tmp_path / "GAPS_run_protocol.json"
    md_path = tmp_path / "GAPS_run.md"
    proto.write_text(json.dumps(protocol), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    report = await score_mission("GAPS", proto)  # no sources sidecar

    assert report.decomposed_claim_count == 2
    assert report.supported_count == 0
    assert report.unsupported_count == 0
    assert report.unknown_count == 2
    assert report.factscore == 0.0
    assert report.unsupported_count_per_citation == {}


@pytest.mark.asyncio
async def test_score_mission_all_unsupported_populates_per_citation(
    fake_deepeval, cached_mission
):
    """All ``no`` verdicts ⇒ factscore=0.0 and unsupported_count_per_citation populated."""
    proto, _, src = cached_mission
    fake_deepeval.extend([(["no", "no"], 0.0), (["no"], 0.0)])

    report = await score_mission("TEST-MISSION-01", proto, src)

    assert report.decomposed_claim_count == 3
    assert report.supported_count == 0
    assert report.unsupported_count == 3
    assert report.unknown_count == 0
    assert report.factscore == 0.0
    assert report.unsupported_count_per_citation == {
        "https://aws.amazon.com/lambda": 2,
        "https://aws.amazon.com/rds": 1,
    }


@pytest.mark.asyncio
async def test_score_mission_skips_dead_and_none_live_citations(fake_deepeval, tmp_path):
    """Only ``live==True`` citations are probed (PD4 + S68 critique rule 8)."""
    md = "Alpha here. Beta here. Gamma here."
    protocol = {
        "mission_id": "FILTER",
        "citations": [
            {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 10,
                "url": "https://a.example",
                "live": True,
            },
            {
                "type": "url_citation",
                "start_index": 12,
                "end_index": 21,
                "url": "https://b.example",
                "live": False,
            },
            {
                "type": "url_citation",
                "start_index": 23,
                "end_index": 33,
                "url": "https://c.example",
                # ``live`` field absent → None → skip
            },
        ],
    }
    sources = [
        {"url": "https://a.example", "text": "Alpha is described here in the source body."},
        {"url": "https://b.example", "text": "Beta is described here too."},
        {"url": "https://c.example", "text": "Gamma is described here too."},
    ]
    proto = tmp_path / "FILTER_run_protocol.json"
    md_path = tmp_path / "FILTER_run.md"
    src = tmp_path / "FILTER_sources.json"
    proto.write_text(json.dumps(protocol), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    src.write_text(json.dumps(sources), encoding="utf-8")

    fake_deepeval.append((["yes"], 1.0))

    report = await score_mission("FILTER", proto, src)

    # Only the live==True citation was probed.
    assert report.decomposed_claim_count == 1
    assert report.supported_count == 1
    assert report.unsupported_count == 0
    assert report.unknown_count == 0


@pytest.mark.asyncio
async def test_score_mission_budget_exhausted_returns_partial(monkeypatch, tmp_path):
    """When the wall-clock budget cuts off in-flight probes, mark partial."""
    md = "Claim alpha. Claim beta."
    protocol = {
        "mission_id": "BUDGET",
        "citations": [
            {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 12,
                "url": "https://a.example",
                "live": True,
            },
            {
                "type": "url_citation",
                "start_index": 13,
                "end_index": 24,
                "url": "https://b.example",
                "live": True,
            },
        ],
    }
    sources = [
        {"url": "https://a.example", "text": "Body for alpha citation."},
        {"url": "https://b.example", "text": "Body for beta citation."},
    ]
    proto = tmp_path / "BUDGET_run_protocol.json"
    md_path = tmp_path / "BUDGET_run.md"
    src = tmp_path / "BUDGET_sources.json"
    proto.write_text(json.dumps(protocol), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    src.write_text(json.dumps(sources), encoding="utf-8")

    class _SlowMetric:
        def __init__(self, threshold: float, model=None, **_kw) -> None:
            self.threshold = threshold
            self.score = 0.0
            self.verdicts: list = []

        def measure(self, _test_case) -> None:
            time.sleep(0.6)  # well beyond the 0.05s budget below
            self.score = 1.0
            self.verdicts = [_FakeVerdict("yes")]

    class _Ltc:
        def __init__(self, **_kw) -> None:
            pass

    monkeypatch.setattr(
        "deepsearch.eval.factscore._resolve_deepeval",
        lambda: (_SlowMetric, _Ltc),
    )

    report = await score_mission("BUDGET", proto, src, budget_seconds=0.05)

    assert report.budget_exhausted is True
    # Cancelled probes are tallied as ``unknown``.
    assert report.unknown_count >= 1
    assert report.decomposed_claim_count >= report.unknown_count


@pytest.mark.asyncio
async def test_score_mission_to_dict_round_trip_preserves_pd7(
    fake_deepeval, cached_mission
):
    proto, _, src = cached_mission
    fake_deepeval.extend([(["yes", "no"], 0.5), (["yes"], 1.0)])

    report = await score_mission("TEST-MISSION-01", proto, src)
    encoded = json.dumps(report.to_dict())
    restored = json.loads(encoded)

    for field_name in (
        "mission_id",
        "factscore",
        "decomposed_claim_count",
        "supported_count",
        "unsupported_count",
        "unknown_count",
        "unsupported_count_per_citation",
        "wall_clock_seconds",
        "budget_exhausted",
        "decomposition_model_id",
        "verification_model_id",
        "threshold",
        "harness_version",
    ):
        assert field_name in restored
    assert restored["mission_id"] == report.mission_id
    assert restored["factscore"] == pytest.approx(report.factscore)
    assert restored["unsupported_count_per_citation"] == report.unsupported_count_per_citation


@pytest.mark.asyncio
async def test_score_mission_no_deepeval_returns_all_unknown(monkeypatch, cached_mission):
    """When DeepEval is unavailable, every live citation lands in unknown."""
    proto, _, src = cached_mission

    def _raise_runtime():
        raise RuntimeError("deepeval is not installed")

    monkeypatch.setattr(
        "deepsearch.eval.factscore._resolve_deepeval",
        _raise_runtime,
    )

    report = await score_mission("TEST-MISSION-01", proto, src)

    assert report.unknown_count == 2
    assert report.supported_count == 0
    assert report.unsupported_count == 0
    assert report.decomposed_claim_count == 2
    assert report.factscore == 0.0


# ---------------------------------------------------------------------------
# m02 — resolve_protocol_path discovery.
# ---------------------------------------------------------------------------


def test_resolve_protocol_path_returns_first_prefix_match(tmp_path):
    """Walks search dirs in order; returns the first ``<id>_*_protocol.json``."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    # Decoy file under the first dir whose prefix doesn't match.
    (dir_a / "OTHER-MISSION_run_protocol.json").write_text("{}", encoding="utf-8")
    # The match lives in the second search dir.
    target = dir_b / "OODS-FIGMA-HOST-01-S68-WIRING_run_protocol.json"
    target.write_text("{}", encoding="utf-8")

    found = resolve_protocol_path(
        "OODS-FIGMA-HOST-01-S68-WIRING",
        search_dirs=(dir_a, dir_b),
    )
    assert found == target


def test_resolve_protocol_path_raises_when_missing(tmp_path):
    """Descriptive ``FileNotFoundError`` lists every directory walked."""
    dir_a = tmp_path / "empty_a"
    dir_b = tmp_path / "empty_b"
    dir_a.mkdir()
    dir_b.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_protocol_path("NO-SUCH-MISSION", search_dirs=(dir_a, dir_b))
    msg = str(exc_info.value)
    assert "NO-SUCH-MISSION" in msg
    assert str(dir_a) in msg
    assert str(dir_b) in msg


def test_resolve_protocol_path_skips_non_protocol_files(tmp_path):
    """Files lacking the ``_protocol.json`` suffix are not eligible matches."""
    d = tmp_path / "research"
    d.mkdir()
    # Sibling markdown / notes that begin with the prefix but are not protocol JSONs.
    (d / "MISSION-X_run.md").write_text("body", encoding="utf-8")
    (d / "MISSION-X_run.json").write_text("{}", encoding="utf-8")
    target = d / "MISSION-X_run_protocol.json"
    target.write_text("{}", encoding="utf-8")

    assert resolve_protocol_path("MISSION-X", search_dirs=(d,)) == target


def test_resolve_protocol_path_rejects_empty_mission_id(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_protocol_path("   ", search_dirs=(tmp_path,))


# ---------------------------------------------------------------------------
# m02 — CLI runtime cases (``deepsearch.eval.__main__.main``).
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_stub_score(monkeypatch):
    """Stub ``score_mission`` so the CLI can be exercised without DeepEval."""
    captured: dict = {}

    async def _fake_score(
        mission_id,
        protocol_path,
        sources_path=None,
        *,
        budget_seconds,
        decomposition_model,
        verification_model,
        threshold,
    ):
        captured["mission_id"] = mission_id
        captured["protocol_path"] = Path(protocol_path)
        captured["sources_path"] = (
            Path(sources_path) if sources_path is not None else None
        )
        captured["budget_seconds"] = budget_seconds
        captured["decomposition_model"] = decomposition_model
        captured["verification_model"] = verification_model
        captured["threshold"] = threshold
        return FActScoreReport(
            mission_id=mission_id,
            factscore=0.75,
            decomposed_claim_count=4,
            supported_count=3,
            unsupported_count=1,
            unknown_count=0,
            unsupported_count_per_citation={"https://x.example": 1},
            wall_clock_seconds=1.5,
            budget_exhausted=False,
            decomposition_model_id=decomposition_model,
            verification_model_id=verification_model,
            threshold=threshold,
        )

    monkeypatch.setattr(factscore_cli, "score_mission", _fake_score)
    return captured


def test_cli_writes_output_file_and_returns_zero(cli_stub_score, tmp_path):
    proto = tmp_path / "M_research_protocol.json"
    proto.write_text("{}", encoding="utf-8")
    out = tmp_path / "out" / "M.json"

    rc = factscore_cli.main(
        [
            "M",
            "--protocol",
            str(proto),
            "--out",
            str(out),
            "--budget-seconds",
            "60",
            "--threshold",
            "0.5",
        ]
    )

    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mission_id"] == "M"
    assert payload["factscore"] == 0.75
    assert payload["unsupported_count_per_citation"] == {"https://x.example": 1}
    assert cli_stub_score["mission_id"] == "M"
    assert cli_stub_score["budget_seconds"] == 60.0


def test_cli_emits_json_on_stdout(cli_stub_score, tmp_path, capsys):
    proto = tmp_path / "M_research_protocol.json"
    proto.write_text("{}", encoding="utf-8")
    out = tmp_path / "M.json"

    rc = factscore_cli.main(
        ["M", "--protocol", str(proto), "--out", str(out)]
    )

    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.split("\n\nwrote")[0])
    assert parsed["mission_id"] == "M"
    assert parsed["factscore"] == 0.75


def test_cli_returns_two_when_protocol_not_resolvable(monkeypatch, tmp_path, capsys):
    """No ``--protocol`` and the mission id can't be discovered ⇒ exit 2."""

    def _missing(mission_id, search_dirs=None):
        raise FileNotFoundError(f"no cached protocol for {mission_id}")

    monkeypatch.setattr(factscore_cli, "resolve_protocol_path", _missing)

    rc = factscore_cli.main(["NO-SUCH-MISSION"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "NO-SUCH-MISSION" in err


def test_cli_returns_two_when_no_target_provided(capsys):
    rc = factscore_cli.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "mission_id" in err or "protocol" in err


def test_cli_calibrate_runs_sequentially_and_writes_each(cli_stub_score, tmp_path, capsys):
    proto_a = tmp_path / "A_run_protocol.json"
    proto_b = tmp_path / "B_run_protocol.json"
    proto_a.write_text("{}", encoding="utf-8")
    proto_b.write_text("{}", encoding="utf-8")

    def _resolve(mission_id, search_dirs=None):
        if mission_id == "A":
            return proto_a
        if mission_id == "B":
            return proto_b
        raise FileNotFoundError(mission_id)

    import deepsearch.eval.__main__ as cli
    cli.resolve_protocol_path = _resolve  # type: ignore[assignment]

    rc = factscore_cli.main(
        [
            "--calibrate",
            "A",
            "B",
            "--out",
            str(tmp_path / "shared_out.json"),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "factscore=0.750" in captured.out
    assert "Calibration summary" in captured.out


def test_cli_calibrate_rejects_simultaneous_mission_id(cli_stub_score, capsys):
    """``--calibrate`` is mutually exclusive with the positional mission_id."""
    rc = factscore_cli.main(["X", "--calibrate", "A", "B"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err
