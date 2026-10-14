"""S69 HD4 carry-forward — JSON round-trip persistence test for FActScoreReport.

Mirrors the S68.5 HD4 pattern (``tests/integration/test_writer_telemetry_persistence.py``):
build a realistic dataclass instance, run it through ``json.dumps`` +
``json.loads``, and assert every PD7 field survives. Any future schema
change to ``FActScoreReport`` that drops a field, mistypes a value, or
introduces a non-JSON-serialisable default fails this test loudly — the
m02 CLI persists ``FActScoreReport.to_dict()`` to disk verbatim, so a
silent drop here would silently drop the daily-eval signal.

The cold-process smoke at the bottom catches lazy-import regressions that
unit tests miss when the module is already cached in ``sys.modules``.
"""

from __future__ import annotations

import json
import subprocess
import sys

from deepsearch.eval.factscore import HARNESS_VERSION, FActScoreReport


def _build_realistic_report() -> FActScoreReport:
    return FActScoreReport(
        mission_id="OODS-FIGMA-HOST-01-S68-WIRING",
        factscore=0.84,
        decomposed_claim_count=125,
        supported_count=105,
        unsupported_count=18,
        unknown_count=2,
        unsupported_count_per_citation={
            "https://www.figma.com/community/file/abc": 4,
            "https://help.figma.com/hc/en-us/articles/xyz": 7,
            "https://www.figma.com/blog/post-x": 7,
        },
        wall_clock_seconds=287.341,
        budget_exhausted=False,
        decomposition_model_id="deepseek-v4-pro",
        verification_model_id="deepseek-v4-pro",
        threshold=0.5,
        harness_version=HARNESS_VERSION,
    )


def test_factscore_report_round_trips_through_json_dumps_loads():
    report = _build_realistic_report()
    payload = report.to_dict()
    encoded = json.dumps(payload)
    restored = json.loads(encoded)

    pd7_fields = (
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
    )
    for field_name in pd7_fields:
        assert field_name in restored, (
            f"PD7 field {field_name!r} disappeared after JSON round-trip"
        )

    assert restored["mission_id"] == report.mission_id
    assert restored["factscore"] == report.factscore
    assert restored["decomposed_claim_count"] == report.decomposed_claim_count
    assert restored["supported_count"] == report.supported_count
    assert restored["unsupported_count"] == report.unsupported_count
    assert restored["unknown_count"] == report.unknown_count
    assert (
        restored["unsupported_count_per_citation"]
        == report.unsupported_count_per_citation
    )
    assert restored["wall_clock_seconds"] == report.wall_clock_seconds
    assert restored["budget_exhausted"] == report.budget_exhausted
    assert restored["decomposition_model_id"] == report.decomposition_model_id
    assert restored["verification_model_id"] == report.verification_model_id
    assert restored["threshold"] == report.threshold
    assert restored["harness_version"] == report.harness_version


def test_unsupported_count_per_citation_round_trips_as_dict_url_to_int():
    """The load-bearing PD7 field must survive as ``Dict[str, int]``.

    JSON keys are always strings, so the key-type guard is a typing
    discipline guard rather than a wire-format guard; the integer value
    must remain ``int`` (not float, not string).
    """
    report = _build_realistic_report()
    encoded = json.dumps(report.to_dict())
    restored = json.loads(encoded)

    per_citation = restored["unsupported_count_per_citation"]
    assert isinstance(per_citation, dict)
    assert per_citation, "fixture must populate at least one URL"
    for url, count in per_citation.items():
        assert isinstance(url, str) and url.startswith("http")
        assert isinstance(count, int) and count >= 0


def test_factscore_report_json_serialises_with_default_str():
    """``json.dumps(default=str)`` is the persistence-CLI fallback path.

    The m02 CLI emits the report dict to disk via ``json.dumps``. This
    guard confirms the dataclass output is wire-clean today — a future
    schema addition that introduces a ``datetime``/``set``/``Path`` field
    without overriding ``to_dict`` will fail this test loudly before
    breaking the CLI silently.
    """
    report = _build_realistic_report()
    payload = report.to_dict()
    encoded = json.dumps(payload, default=str)

    assert isinstance(encoded, str)
    assert HARNESS_VERSION in encoded
    assert "OODS-FIGMA-HOST-01-S68-WIRING" in encoded


def test_cold_process_smoke_imports_and_round_trips():
    """Run the round-trip in a fresh interpreter to catch lazy-import bugs.

    The harness lazy-imports DeepEval inside ``score_mission``. The
    dataclass + ``to_dict`` path must remain importable + serialisable
    even when DeepEval is not installed — which is the daily-runner
    scenario for any sprint that hasn't installed ``pip install deepeval``
    locally yet.
    """
    code = (
        "import json; "
        "from deepsearch.eval.factscore import FActScoreReport, HARNESS_VERSION; "
        "report = FActScoreReport("
        "mission_id='M', factscore=0.5, decomposed_claim_count=4, "
        "supported_count=2, unsupported_count=2, unknown_count=0, "
        "unsupported_count_per_citation={'https://a.example': 1, 'https://b.example': 1}, "
        "wall_clock_seconds=1.25, budget_exhausted=False, "
        "decomposition_model_id='deepseek-v4-pro', "
        "verification_model_id='deepseek-v4-pro', "
        "threshold=0.5, harness_version=HARNESS_VERSION); "
        "data = report.to_dict(); "
        "restored = json.loads(json.dumps(data)); "
        "assert restored['mission_id'] == 'M'; "
        "assert restored['unsupported_count_per_citation'] == {'https://a.example': 1, 'https://b.example': 1}; "
        "assert restored['harness_version'] == HARNESS_VERSION; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"cold-process smoke failed: stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout
