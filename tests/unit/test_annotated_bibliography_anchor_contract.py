"""S43.2 — regression that the annotated_bibliography output contract specifies `[n]`.

Prior defect (surfaced by S42.4 paid DEEP smoke ACC-BIB-EVID-01-S42-RUN-01):
the annotated_bibliography schema block in render_output_schema_contracts
prescribed row count + annotation depth + thin-label prohibition but said
nothing about the `[n]` citation anchor. Three consecutive sprints saw the
LLM render three different shapes (S40 prefix `**[n] Title**`, S41 suffix
`**Title** [n][n][n]`, S42 no markers at all) because the contract left
the shape to inference.

This test exercises the REAL rendering entry point
(``render_output_schema_contracts``) with an annotated_bibliography schema
extracted via ``extract_deliverable_schemas`` and asserts the rendered block
explicitly names the `[n]` anchor at both BASELINE and DEEP tiers — the
shape is a contract property, not a quality floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

from deepsearch.agent.deliverable_schemas import (  # noqa: E402
    extract_deliverable_schemas,
    render_output_schema_contracts,
)


_DELIVERABLE_LINE = (
    "Annotated bibliography of at least 10 papers with per-row annotations"
)


def _rendered_contract(research_depth: str) -> str:
    schemas = extract_deliverable_schemas(
        mission_deliverables=[_DELIVERABLE_LINE],
        success_criteria=[],
    )
    assert any(
        schema.kind == "annotated_bibliography" for schema in schemas
    ), "fixture must produce an annotated_bibliography schema via the real extractor"
    return render_output_schema_contracts(schemas, research_depth=research_depth)


@pytest.mark.parametrize("research_depth", ["baseline", "deep", "alpha"])
def test_annotated_bibliography_contract_requires_n_anchor(
    research_depth: str,
) -> None:
    """Every tier must surface the `[n]` anchor requirement — shape is a contract property."""

    rendered = _rendered_contract(research_depth)

    assert "`[n]`" in rendered, (
        "annotated_bibliography output contract must explicitly name the `[n]` "
        f"citation anchor at tier '{research_depth}'. Rendered contract:\n{rendered}"
    )
    # Load-bearing phrasing — the test pins the specific instruction wording so
    # future refactors of the block-builder preserve the intent.
    assert "MUST begin with the citation index" in rendered, (
        "annotated_bibliography output contract must carry the load-bearing "
        "'MUST begin with the citation index' instruction at tier "
        f"'{research_depth}'. Rendered contract:\n{rendered}"
    )
    assert "mandatory" in rendered.lower(), (
        "the `[n]` anchor instruction must be described as mandatory so the LLM "
        "does not treat it as optional emphasis. Rendered contract:\n{rendered}"
    )


def test_n_anchor_instruction_is_tier_agnostic() -> None:
    """The instruction wording must be identical across tiers (shape ≠ quality floor)."""

    baseline = _rendered_contract("baseline")
    deep = _rendered_contract("deep")
    alpha = _rendered_contract("alpha")

    # Extract the anchor-instruction line from each rendering; assert byte-identical.
    def _anchor_line(text: str) -> str:
        for line in text.splitlines():
            if "`[n]`" in line and "MUST begin with the citation index" in line:
                return line.strip()
        raise AssertionError(
            f"no [n]-anchor instruction line found in rendered contract:\n{text}"
        )

    assert _anchor_line(baseline) == _anchor_line(deep) == _anchor_line(alpha), (
        "The `[n]` anchor instruction must be tier-agnostic — identical wording at "
        "BASELINE, DEEP, and ALPHA. Baseline: {0!r} | Deep: {1!r} | Alpha: {2!r}".format(
            _anchor_line(baseline), _anchor_line(deep), _anchor_line(alpha)
        )
    )


def test_n_anchor_precedes_depth_floor_instruction() -> None:
    """Shape contract must appear before quality floor — LLM reads top-down.

    Ordering matters: the `[n]` anchor is a shape contract (how to render),
    the annotation-depth floor is a quality floor (how much to write).
    Emitting shape before floor avoids the LLM writing a long annotation
    without the anchor and then having to retro-fit citation indices.
    """

    rendered = _rendered_contract("deep")
    anchor_index = rendered.find("MUST begin with the citation index")
    depth_index = rendered.find("Each annotation row MUST include at least")

    assert anchor_index >= 0, "anchor-instruction line missing"
    assert depth_index >= 0, "depth-floor-instruction line missing"
    assert anchor_index < depth_index, (
        "anchor instruction must precede depth-floor instruction in the rendered "
        "contract so shape is established before quality floor."
    )
