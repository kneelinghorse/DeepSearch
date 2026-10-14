"""Quality gate for TraceLab upload readiness.

This module provides a simple interface to evaluate whether a report's
quality metrics meet the thresholds required for TraceLab submission.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from deepsearch.data_paths import data_path

from .hybrid_evaluator import QualityScores, QualityThresholds

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GateResult:
    """Result of a quality gate evaluation."""

    passed: bool
    score_checks: Dict[str, bool]
    scores: QualityScores
    thresholds: QualityThresholds
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score_checks": self.score_checks,
            "scores": self.scores.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "failure_reasons": self.failure_reasons,
            "warnings": self.warnings,
        }


def evaluate(
    metrics: Mapping[str, Any],
    telemetry: Optional[Mapping[str, Any]] = None,
    *,
    thresholds: Optional[QualityThresholds] = None,
    require_telemetry: bool = False,
) -> GateResult:
    """Evaluate whether quality metrics pass the upload gate.

    Args:
        metrics: Validation metrics dictionary (from hybrid evaluator output).
                 Expected keys: quality_scores.faithfulness, quality_scores.relevance,
                 quality_scores.citation_rate (or flat keys at root).
        telemetry: Optional telemetry blob with execution metadata.
        thresholds: Quality thresholds to use (defaults to standard thresholds).
        require_telemetry: If True, fail when telemetry is missing or incomplete.

    Returns:
        GateResult with pass/fail status and detailed breakdown.
    """
    thresholds = thresholds or QualityThresholds()
    failure_reasons: List[str] = []
    warnings: List[str] = []

    # Extract scores from metrics (support nested or flat structure)
    quality_scores = metrics.get("quality_scores", {})
    if not quality_scores:
        quality_scores = metrics

    faithfulness = _extract_float(quality_scores, "faithfulness", 0.0)
    relevance = _extract_float(quality_scores, "relevance", 0.0)
    citation_rate = _extract_float(quality_scores, "citation_rate", 0.0)

    scores = QualityScores(
        faithfulness=faithfulness,
        relevance=relevance,
        citation_rate=citation_rate,
    )

    # Check each threshold
    score_checks = {
        "faithfulness": scores.faithfulness >= thresholds.faithfulness,
        "relevance": scores.relevance >= thresholds.relevance,
        "citation_rate": scores.citation_rate >= thresholds.citation_rate,
    }

    if not score_checks["faithfulness"]:
        failure_reasons.append(
            f"Faithfulness {scores.faithfulness:.2f} < {thresholds.faithfulness:.2f} threshold"
        )
    if not score_checks["relevance"]:
        failure_reasons.append(
            f"Relevance {scores.relevance:.2f} < {thresholds.relevance:.2f} threshold"
        )
    if not score_checks["citation_rate"]:
        failure_reasons.append(
            f"Citation rate {scores.citation_rate:.2f} < {thresholds.citation_rate:.2f} threshold"
        )

    # Check telemetry requirements
    telemetry_ok = True
    if require_telemetry:
        if not telemetry:
            failure_reasons.append("Telemetry data is required but missing")
            telemetry_ok = False
        else:
            required_fields = ["mission_id", "loops_executed", "sources_count"]
            missing = [f for f in required_fields if f not in telemetry]
            if missing:
                failure_reasons.append(f"Telemetry missing required fields: {missing}")
                telemetry_ok = False
    elif telemetry is None:
        warnings.append("No telemetry provided; upload will proceed without execution metadata")

    # Check for zero scores (likely missing data)
    if scores.faithfulness == 0.0:
        warnings.append("Faithfulness score is 0.0 - validation may not have run")
    if scores.relevance == 0.0:
        warnings.append("Relevance score is 0.0 - validation may not have run")
    if scores.citation_rate == 0.0:
        warnings.append("Citation rate is 0.0 - report may lack citations")

    passed = all(score_checks.values()) and (not require_telemetry or telemetry_ok)

    return GateResult(
        passed=passed,
        score_checks=score_checks,
        scores=scores,
        thresholds=thresholds,
        failure_reasons=failure_reasons,
        warnings=warnings,
    )


def load_metrics_file(path: Path | str) -> Dict[str, Any]:
    """Load validation metrics from a JSON file.

    Args:
        path: Path to the metrics JSON file.

    Returns:
        Parsed metrics dictionary.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_telemetry_file(path: Path | str) -> Dict[str, Any]:
    """Load telemetry data from a JSONL or JSON file.

    Args:
        path: Path to the telemetry file.

    Returns:
        Parsed telemetry dictionary (last entry if JSONL).

    Raises:
        FileNotFoundError: If the telemetry file does not exist.
    """
    telemetry_path = Path(path)
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

    content = telemetry_path.read_text(encoding="utf-8").strip()

    # Handle JSONL format (return last non-empty line)
    if telemetry_path.suffix == ".jsonl":
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if not lines:
            return {}
        return json.loads(lines[-1])

    return json.loads(content)


def find_metrics_for_mission(
    mission_id: str,
    validation_dir: Path | str | None = None,
) -> Optional[Path]:
    """Find the most recent validation metrics file for a mission.

    Args:
        mission_id: The mission identifier (e.g., "DRM.0.1").
        validation_dir: Directory containing validation output files.

    Returns:
        Path to the most recent metrics file, or None if not found.
    """
    validation_path = (
        Path(validation_dir) if validation_dir is not None else data_path("validation")
    )
    if not validation_path.exists():
        return None

    # Extract the core mission identifier parts (e.g., "DRM.0.1" -> ["drm", "0", "1"])
    mission_lower = mission_id.lower()
    # Split on common separators to get parts
    parts = re.split(r"[.\-_]", mission_lower)
    # Filter out empty parts
    parts = [p for p in parts if p]

    # Build a pattern that matches the parts in sequence with any separator
    # e.g., ["drm", "0", "1"] matches "drm-0.1", "drm.0.1", "drm_0_1", etc.
    if len(parts) >= 2:
        # Create regex pattern: drm[.\-_]0[.\-_]1 or similar
        pattern_str = r"[.\-_]?".join(re.escape(p) for p in parts)
        pattern = re.compile(pattern_str, re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(mission_lower), re.IGNORECASE)

    matches: list[Path] = []
    for path in validation_path.glob("*_metrics.json"):
        # Check if the pattern matches anywhere in the filename
        if pattern.search(path.name):
            matches.append(path)

    # Remove duplicates while preserving order
    seen: set[Path] = set()
    unique_matches: list[Path] = []
    for path in matches:
        if path not in seen:
            seen.add(path)
            unique_matches.append(path)

    if not unique_matches:
        return None

    # Return most recent by modification time
    return max(unique_matches, key=lambda p: p.stat().st_mtime)


def _extract_float(data: Mapping[str, Any], key: str, default: float) -> float:
    """Safely extract a float value from a mapping."""
    value = data.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "GateResult",
    "evaluate",
    "find_metrics_for_mission",
    "load_metrics_file",
    "load_telemetry_file",
]
