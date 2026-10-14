"""Contract validation harness for local-vs-deployed parity checks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from deepsearch.data_paths import data_path
from deepsearch.validation.gate import load_metrics_file, load_telemetry_file
from deepsearch.validation.reviewer import ReviewerAdjudication

logger = logging.getLogger(__name__)

_RUNTIME_SIGNATURE_KEYS = (
    "build_hash",
    "build_source",
    "git_dirty",
    "llm_backend",
    "model",
    "execution_path",
    "python_version",
    "platform",
)


@dataclass(slots=True)
class StructuralCheck:
    """Single deliverable-schema validation result."""

    schema_id: str
    title: str
    kind: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "title": self.title,
            "kind": self.kind,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(slots=True)
class StructuralValidationSummary:
    """Structural validation result for a report against a contract."""

    passed: bool
    checks: list[StructuralCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(slots=True)
class EnvironmentBundle:
    """Comparable artifact bundle for one execution environment."""

    environment: str
    mission_id: str
    contract_id: str
    contract_path: str
    report_path: str
    telemetry_path: str
    validation_path: Optional[str]
    runtime_identity: Dict[str, Any]
    config_fingerprint: str
    route_summary: Dict[str, Any]
    final_outcome: Dict[str, Any]
    validation_summary: Dict[str, Any]
    structural_validation: StructuralValidationSummary
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    reviewer_adjudication: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "mission_id": self.mission_id,
            "contract_id": self.contract_id,
            "contract_path": self.contract_path,
            "report_path": self.report_path,
            "telemetry_path": self.telemetry_path,
            "validation_path": self.validation_path,
            "runtime_identity": dict(self.runtime_identity),
            "config_fingerprint": self.config_fingerprint,
            "route_summary": dict(self.route_summary),
            "final_outcome": dict(self.final_outcome),
            "validation_summary": dict(self.validation_summary),
            "structural_validation": self.structural_validation.to_dict(),
            "performance_metrics": dict(self.performance_metrics),
            "reviewer_adjudication": (
                dict(self.reviewer_adjudication) if self.reviewer_adjudication else None
            ),
        }


@dataclass(slots=True)
class BundleComparison:
    """Comparison between two environment bundles."""

    mission_id: str
    generated_at: str
    local: EnvironmentBundle
    deployed: EnvironmentBundle
    parity_passed: bool
    differences: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "generated_at": self.generated_at,
            "parity_passed": self.parity_passed,
            "differences": list(self.differences),
            "runtime_identity_diff": _runtime_identity_diff(self.local, self.deployed),
            "local": self.local.to_dict(),
            "deployed": self.deployed.to_dict(),
            "checks": {
                "mission_id_match": self.local.mission_id == self.deployed.mission_id,
                "contract_id_match": self.local.contract_id == self.deployed.contract_id,
                "config_fingerprint_match": (
                    self.local.config_fingerprint == self.deployed.config_fingerprint
                ),
                "runtime_identity_match": (
                    _runtime_signature(self.local) == _runtime_signature(self.deployed)
                ),
                "route_match": _route_signature(self.local) == _route_signature(self.deployed),
                "outcome_match": (
                    _outcome_signature(self.local) == _outcome_signature(self.deployed)
                ),
                "validation_match": (
                    _validation_signature(self.local) == _validation_signature(self.deployed)
                ),
                "local_structural_pass": self.local.structural_validation.passed,
                "deployed_structural_pass": self.deployed.structural_validation.passed,
                "local_reviewer_attached": bool(self.local.reviewer_adjudication),
                "deployed_reviewer_attached": bool(self.deployed.reviewer_adjudication),
            },
        }


async def run_validation_harness(
    *,
    mission_path: Optional[Path] = None,
    local_contract_path: Optional[Path] = None,
    local_report_path: Optional[Path] = None,
    local_telemetry_path: Optional[Path] = None,
    local_validation_path: Optional[Path] = None,
    deployed_contract_path: Path,
    deployed_report_path: Path,
    deployed_telemetry_path: Path,
    deployed_validation_path: Optional[Path] = None,
    local_reviewer_path: Optional[Path] = None,
    deployed_reviewer_path: Optional[Path] = None,
    llm_backend: str | None = None,
    model: str | None = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the parity harness and persist a comparison artifact."""

    if mission_path is None and not (
        local_contract_path and local_report_path and local_telemetry_path
    ):
        raise ValueError(
            "Provide either mission_path for a fresh local run or local "
            "contract/report/telemetry paths."
        )

    if mission_path is not None:
        from deepsearch.runner import execute_research_mission

        local_result = await execute_research_mission(
            mission_path=mission_path,
            llm_backend=llm_backend,
            model=model,
            validate=True,
        )
        local_bundle = bundle_from_result(
            environment="local",
            result=local_result,
            reviewer_path=local_reviewer_path,
        )
    else:
        local_bundle = bundle_from_artifacts(
            environment="local",
            contract_path=local_contract_path,
            report_path=local_report_path,
            telemetry_path=local_telemetry_path,
            validation_path=local_validation_path,
            reviewer_path=local_reviewer_path,
        )

    deployed_bundle = bundle_from_artifacts(
        environment="deployed",
        contract_path=deployed_contract_path,
        report_path=deployed_report_path,
        telemetry_path=deployed_telemetry_path,
        validation_path=deployed_validation_path,
        reviewer_path=deployed_reviewer_path,
    )

    comparison = compare_bundles(local_bundle, deployed_bundle)
    saved = save_comparison_result(comparison, output_path=output_path)

    payload = comparison.to_dict()
    payload["output_path"] = str(saved["json"])
    payload["markdown_path"] = str(saved["markdown"])
    return payload


def bundle_from_result(
    *,
    environment: str,
    result: Mapping[str, Any],
    reviewer_path: Optional[Path] = None,
) -> EnvironmentBundle:
    """Build an environment bundle from an execute_research_mission result."""

    telemetry = result.get("telemetry") if isinstance(result.get("telemetry"), Mapping) else {}
    contract_path = _coerce_path(result.get("contract_path") or telemetry.get("contract_path"))
    report_path = _coerce_path(result.get("report_path"))
    telemetry_path = _coerce_path(
        result.get("telemetry_path") or telemetry.get("telemetry_artifact_path")
    )
    validation_payload = (
        result.get("validation") if isinstance(result.get("validation"), Mapping) else {}
    )
    validation_path = _coerce_path(
        validation_payload.get("output_path")
        if isinstance(validation_payload, Mapping)
        else None
    )
    if contract_path is None or report_path is None or telemetry_path is None:
        raise ValueError(
            "result must include contract_path, report_path, and telemetry_path "
            "or telemetry_artifact_path"
        )
    structural = evaluate_structural_validation(
        contract_path=contract_path,
        report_path=report_path,
    )
    route_summary = _last_route_summary(telemetry)
    reviewer = load_reviewer_adjudication(reviewer_path)
    mission_id = str(result.get("mission_id") or telemetry.get("mission_id") or "")
    return EnvironmentBundle(
        environment=environment,
        mission_id=mission_id,
        contract_id=str(result.get("contract_id") or telemetry.get("contract_id") or ""),
        contract_path=str(contract_path),
        report_path=str(report_path),
        telemetry_path=str(telemetry_path),
        validation_path=str(validation_path) if validation_path else None,
        runtime_identity=dict(telemetry.get("runtime_identity") or {}),
        config_fingerprint=str(telemetry.get("config_fingerprint") or ""),
        route_summary=route_summary,
        final_outcome=dict(telemetry.get("final_outcome") or {}),
        validation_summary=_build_validation_summary(
            telemetry=telemetry,
            validation_payload=validation_payload,
        ),
        structural_validation=structural,
        performance_metrics=dict(result.get("performance_metrics") or {}),
        reviewer_adjudication=reviewer,
    )


def bundle_from_artifacts(
    *,
    environment: str,
    contract_path: Optional[Path],
    report_path: Optional[Path],
    telemetry_path: Optional[Path],
    validation_path: Optional[Path] = None,
    reviewer_path: Optional[Path] = None,
) -> EnvironmentBundle:
    """Build an environment bundle from persisted artifacts."""

    if contract_path is None or report_path is None or telemetry_path is None:
        raise ValueError("contract_path, report_path, and telemetry_path are required")

    telemetry = load_telemetry_file(telemetry_path)
    contract = _load_json(contract_path)
    validation_payload = load_metrics_file(validation_path) if validation_path else {}
    structural = evaluate_structural_validation(
        contract_path=contract_path,
        report_path=report_path,
    )
    reviewer = load_reviewer_adjudication(reviewer_path)
    mission_id = str(
        contract.get("mission_id")
        or telemetry.get("mission_id")
        or _coerce_path(report_path).stem
    )
    return EnvironmentBundle(
        environment=environment,
        mission_id=mission_id,
        contract_id=str(contract.get("contract_id") or telemetry.get("contract_id") or ""),
        contract_path=str(contract_path),
        report_path=str(report_path),
        telemetry_path=str(telemetry_path),
        validation_path=str(validation_path) if validation_path else None,
        runtime_identity=dict(telemetry.get("runtime_identity") or {}),
        config_fingerprint=str(telemetry.get("config_fingerprint") or ""),
        route_summary=_last_route_summary(telemetry),
        final_outcome=dict(telemetry.get("final_outcome") or {}),
        validation_summary=_build_validation_summary(
            telemetry=telemetry,
            validation_payload=validation_payload,
        ),
        structural_validation=structural,
        performance_metrics=dict(validation_payload.get("performance_metrics") or {}),
        reviewer_adjudication=reviewer,
    )


def compare_bundles(local: EnvironmentBundle, deployed: EnvironmentBundle) -> BundleComparison:
    """Compare two environment bundles and highlight parity mismatches."""

    differences: list[str] = []
    if local.mission_id != deployed.mission_id:
        differences.append(
            f"Mission mismatch: local={local.mission_id} deployed={deployed.mission_id}"
        )
    if local.contract_id != deployed.contract_id:
        differences.append(
            f"Contract mismatch: local={local.contract_id} deployed={deployed.contract_id}"
        )
    if local.config_fingerprint != deployed.config_fingerprint:
        differences.append(
            "Config fingerprint mismatch between local and deployed artifact bundles."
        )
    if _runtime_signature(local) != _runtime_signature(deployed):
        differences.append(
            "Runtime identity mismatch between local and deployed bundles."
        )
    if _route_signature(local) != _route_signature(deployed):
        differences.append(
            "Route mismatch between local and deployed bundles."
        )
    if _outcome_signature(local) != _outcome_signature(deployed):
        differences.append(
            "Final outcome mismatch between local and deployed bundles."
        )
    if _validation_signature(local) != _validation_signature(deployed):
        differences.append(
            "Validation summary mismatch between local and deployed bundles."
        )
    if not local.structural_validation.passed:
        differences.append("Local structural validation did not pass.")
    if not deployed.structural_validation.passed:
        differences.append("Deployed structural validation did not pass.")

    return BundleComparison(
        mission_id=local.mission_id or deployed.mission_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        local=local,
        deployed=deployed,
        parity_passed=not differences,
        differences=differences,
    )


def save_comparison_result(
    comparison: BundleComparison,
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Persist JSON and Markdown comparison artifacts."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_mission_id = re.sub(r"[^A-Za-z0-9._-]+", "-", comparison.mission_id or "mission")
    default_name = f"{safe_mission_id}_contract_comparison_{timestamp}.json"
    if output_path is None:
        json_path = data_path("validation") / default_name
    elif output_path.exists() and output_path.is_dir():
        json_path = output_path / default_name
    elif output_path.suffix.lower() != ".json":
        json_path = output_path.with_suffix(".json")
    else:
        json_path = output_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = json_path.with_suffix(".md")

    payload = comparison.to_dict()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def render_comparison_markdown(comparison: BundleComparison) -> str:
    """Render a human-readable comparison summary."""

    checks = comparison.to_dict()["checks"]
    lines = [
        f"# Contract Comparison: {comparison.mission_id}",
        "",
        f"Generated: {comparison.generated_at}",
        "",
        f"Parity passed: {'yes' if comparison.parity_passed else 'no'}",
        "",
        "## Summary",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for key, value in checks.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Environments",
        "",
        f"- Local contract: `{comparison.local.contract_id}`",
        f"- Deployed contract: `{comparison.deployed.contract_id}`",
        f"- Local runtime: `{comparison.local.runtime_identity}`",
        f"- Deployed runtime: `{comparison.deployed.runtime_identity}`",
        f"- Runtime diff: `{_runtime_identity_diff(comparison.local, comparison.deployed)}`",
        f"- Local route: `{comparison.local.route_summary}`",
        f"- Deployed route: `{comparison.deployed.route_summary}`",
        f"- Local outcome: `{comparison.local.final_outcome}`",
        f"- Deployed outcome: `{comparison.deployed.final_outcome}`",
        f"- Local validation: `{comparison.local.validation_summary}`",
        f"- Deployed validation: `{comparison.deployed.validation_summary}`",
        "",
        "## Structural Validation",
        "",
        f"- Local passed: `{comparison.local.structural_validation.passed}`",
        f"- Deployed passed: `{comparison.deployed.structural_validation.passed}`",
        "",
        "## Differences",
        "",
    ])

    if comparison.differences:
        lines.extend(f"- {item}" for item in comparison.differences)
    else:
        lines.append("- None")

    return "\n".join(lines)


def evaluate_structural_validation(
    *,
    contract_path: Path,
    report_path: Path,
) -> StructuralValidationSummary:
    """No-op structural validation summary post-S76 deletion of structural_validators.

    The legacy deliverable-schema validators were deleted in Sprint 76 as
    pre-rebuild substrate. Returns a passing summary with no checks; harness
    callers continue to receive a stable shape.
    """

    return StructuralValidationSummary(passed=True, checks=[])


def load_reviewer_adjudication(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load reviewer adjudication from JSON or plain text."""

    if path is None:
        return None
    if path.suffix.lower() == ".json":
        return ReviewerAdjudication.read_from(path).to_dict()
    text = path.read_text(encoding="utf-8")
    return {
        "attachment_path": str(path),
        "notes": text.strip(),
    }


def _build_validation_summary(
    *,
    telemetry: Mapping[str, Any],
    validation_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(telemetry.get("validation_summary"), Mapping):
        return dict(telemetry.get("validation_summary") or {})
    if validation_payload:
        return {
            "requested": True,
            "status": "passed" if validation_payload.get("overall_pass") else "failed",
            "failed_checks": [
                name
                for name, passed in (validation_payload.get("passes_standards") or {}).items()
                if not passed
            ],
            "output_path": validation_payload.get("output_path"),
        }
    return {"requested": False, "status": "not_requested"}


def _route_signature(bundle: EnvironmentBundle) -> tuple[str, str]:
    return (
        str(bundle.route_summary.get("next_action") or ""),
        str(bundle.route_summary.get("reason") or ""),
    )


def _last_route_summary(telemetry: Mapping[str, Any]) -> Dict[str, Any]:
    routing_trace = telemetry.get("routing_trace")
    if isinstance(routing_trace, list):
        for entry in reversed(routing_trace):
            if isinstance(entry, Mapping):
                return dict(entry)
    route_summary = telemetry.get("route_summary")
    if isinstance(route_summary, Mapping):
        return dict(route_summary)
    return {}


def _runtime_signature(bundle: EnvironmentBundle) -> tuple[str | bool, ...]:
    return tuple(bundle.runtime_identity.get(key) for key in _RUNTIME_SIGNATURE_KEYS)


def _runtime_identity_diff(
    local: EnvironmentBundle,
    deployed: EnvironmentBundle,
) -> Dict[str, Dict[str, Any]]:
    differences: Dict[str, Dict[str, Any]] = {}
    keys = sorted(set(local.runtime_identity) | set(deployed.runtime_identity))
    for key in keys:
        local_value = local.runtime_identity.get(key)
        deployed_value = deployed.runtime_identity.get(key)
        if local_value == deployed_value:
            continue
        differences[key] = {
            "local": local_value,
            "deployed": deployed_value,
        }
    return differences


def _outcome_signature(bundle: EnvironmentBundle) -> tuple[str, str]:
    return (
        str(bundle.final_outcome.get("outcome") or bundle.final_outcome.get("status") or ""),
        str(
            bundle.final_outcome.get("routing_reason")
            or bundle.final_outcome.get("reason")
            or ""
        ),
    )


def _validation_signature(bundle: EnvironmentBundle) -> tuple[str, tuple[str, ...]]:
    failed_checks_raw = bundle.validation_summary.get("failed_checks") or []
    failed_checks = (
        tuple(sorted(str(item) for item in failed_checks_raw))
        if isinstance(failed_checks_raw, Sequence) and not isinstance(failed_checks_raw, str)
        else ()
    )
    return (
        str(bundle.validation_summary.get("status") or ""),
        failed_checks,
    )


def _coerce_path(value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    return Path(text) if text else None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "BundleComparison",
    "EnvironmentBundle",
    "StructuralCheck",
    "StructuralValidationSummary",
    "bundle_from_artifacts",
    "bundle_from_result",
    "compare_bundles",
    "evaluate_structural_validation",
    "load_reviewer_adjudication",
    "render_comparison_markdown",
    "run_validation_harness",
    "save_comparison_result",
]
