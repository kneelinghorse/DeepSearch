"""S92 regression tests for the retired structural-validation checkpoint.

Structural validation was intentionally deleted in S76.  A mission contract
may still carry deliverable schemas, but their presence alone is not evidence
that validation ran or failed.  Historical/external execution signals remain
authoritative when they are present.
"""

from __future__ import annotations

import pytest

from deepsearch.report.protocol import (
    QualityCheckpoint,
    _build_quality_checkpoints,
)
from deepsearch.report.synthesis import Synthesis

_VALID_SYNTHESIS = Synthesis(
    key_insights=[
        "The completed synthesis contains enough substantive detail for the baseline gate."
    ],
    recommendations=["Preserve truthful checkpoint semantics."],
    next_steps=["Persist the resulting protocol."],
)


def _synthesis_checkpoint(state: dict[str, object]) -> QualityCheckpoint:
    checkpoints = _build_quality_checkpoints(state, synthesis=_VALID_SYNTHESIS)
    return next(checkpoint for checkpoint in checkpoints if checkpoint.gate == "synthesis_quality")


def _state_with_schema(**updates: object) -> dict[str, object]:
    state: dict[str, object] = {
        "mission_contract": {
            "deliverable_schemas": [{"id": "classification_matrix"}],
        }
    }
    state.update(updates)
    return state


def test_schema_without_result_skips_retired_structural_validation() -> None:
    """Schema presence cannot manufacture failure after validator retirement."""
    checkpoint = _synthesis_checkpoint(_state_with_schema())

    assert checkpoint.status == "skip"
    assert checkpoint.notes is not None
    assert "retired" in checkpoint.notes
    assert "not evaluated" in checkpoint.notes


@pytest.mark.parametrize(
    "execution_signal",
    [
        {"synthesis_telemetry": {"structural_validation_passed": False}},
        {"structural_validation_failures": ["classification_matrix"]},
        {"routing_reason": "structural_validation_failed"},
    ],
)
def test_explicit_structural_failure_signals_still_fail(
    execution_signal: dict[str, object],
) -> None:
    """Distinguishable executed failures remain failures, not retirement skips."""
    checkpoint = _synthesis_checkpoint(_state_with_schema(**execution_signal))

    assert checkpoint.status == "fail"


def test_explicit_structural_success_still_passes() -> None:
    """A recorded historical/external validation success remains authoritative."""
    checkpoint = _synthesis_checkpoint(
        _state_with_schema(
            synthesis_telemetry={"structural_validation_passed": True},
        )
    )

    assert checkpoint.status == "pass"


def test_baseline_synthesis_without_schema_still_passes() -> None:
    """Retirement skip is scoped to schema contracts lacking an executed result."""
    checkpoint = _synthesis_checkpoint({})

    assert checkpoint.status == "pass"
