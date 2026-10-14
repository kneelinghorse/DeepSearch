"""Verify the frozen S94 cross-service contract and scope fixture without providers."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(fixture: Path) -> dict[str, Any]:
    """Check byte hashes, canonical compilation, admission and exact output verdicts."""
    from deepsearch.agent.contract_to_prompt import build_codeagent_prompts
    from deepsearch.mission.compiler_provenance import canonical_json
    from deepsearch.mission.contract import compile_canonical_contract_from_state
    from deepsearch.mission.scope import ScopeAdmission
    from deepsearch.report.scope_validation import validate_persisted_scope

    manifest = json.loads((fixture / "manifest.json").read_text())
    assert manifest["fixture_version"] == "authored-scope-parity-v1"
    for name, expected in manifest["file_sha256"].items():
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == expected, name
    state = json.loads((fixture / "canonical-input.json").read_text())
    contract = compile_canonical_contract_from_state(state)
    assert canonical_json(contract) == (fixture / "canonical-contract.json").read_text()
    state["mission_contract"] = contract.model_dump(mode="json")
    system, directive = build_codeagent_prompts(state)
    assert system == (fixture / "actor-system.txt").read_text()
    assert directive == (fixture / "actor-user.txt").read_text()
    for case in manifest["actions"]:
        admission = ScopeAdmission(contract.authored_scope)
        for url in case["already_consulted"]:
            assert admission.admit(url, "fixture")
        if case["tool"] in {"web_search", "scholar_search"}:
            actual = admission.search_refusal(case["tool"], case["input"])
        else:
            actual = admission.refusal(case["input"])
        assert actual == case["refusal"], case["name"]
    for case in manifest["outputs"]:
        admission = ScopeAdmission(contract.authored_scope)
        for url in case["admitted_urls"]:
            assert admission.admit(url, "fixture")
        admission.unavailable.update(case["unavailable"])
        result = validate_persisted_scope(
            admission=admission,
            pre_render=(fixture / case["pre_render"]).read_text(),
            persisted=(fixture / case["persisted"]).read_text(),
            observed_sources=[{"url": u} for u in case["observed_urls"]],
            cited_urls=case["cited_urls"],
        )
        assert result == json.loads((fixture / case["expected"]).read_text()), case["name"]
    return {
        "fixture_version": manifest["fixture_version"],
        "contract_id": contract.contract_id,
        "schema": contract.contract_schema_version,
        "compiler_revision": contract.compiler_revision,
        "actions_checked": len(manifest["actions"]),
        "outputs_checked": len(manifest["outputs"]),
    }


def main() -> None:
    """Run the check with all socket connections and DNS explicitly disabled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path,
                        default=ROOT / "tests/fixtures/authored_scope_v1")
    args = parser.parse_args()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Parity checks must not connect to providers or DNS")

    socket.socket.connect = forbidden
    socket.getaddrinfo = forbidden
    print(json.dumps(verify(args.fixture_dir), sort_keys=True))


if __name__ == "__main__":
    main()
