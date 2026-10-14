"""S91 m02 deterministic compiler provenance and golden parity invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

import deepsearch.mission.contract as contract_module
from deepsearch.api.app import create_app
from deepsearch.api.models.requests import MissionContractPreviewRequest
from deepsearch.api.queue import JobQueue
from deepsearch.api.routes.missions import _build_preview_state
from deepsearch.config import TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV
from deepsearch.mission import (
    CONTRACT_COMPILER_REVISION,
    CONTRACT_SCHEMA_VERSION,
    NO_ENRICHMENT_MODE,
    MissionContract,
    build_contract_compiler_manifest,
    canonical_json,
    compile_canonical_contract_from_state,
    compile_contract_from_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "contract_compiler"
_INPUT_PATH = _FIXTURE_DIR / "s91_m02_preview_input.json"
_OUTPUT_PATH = _FIXTURE_DIR / "s91_m02_canonical_contract.json"
_OUTPUT_SHA256_PATH = _FIXTURE_DIR / "s91_m02_canonical_contract.sha256"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _golden_state() -> dict[str, object]:
    request = MissionContractPreviewRequest.model_validate(_load_json(_INPUT_PATH))
    return _build_preview_state(request)


def test_canonical_compiler_matches_independent_golden_output(monkeypatch) -> None:
    """Parity is checked against frozen output, not another compiler invocation."""

    def _model_path_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("canonical no-enrichment mode must not touch a model path")

    monkeypatch.setattr(
        contract_module,
        "_extract_entities_llm_first",
        _model_path_called,
    )
    monkeypatch.setattr(
        contract_module,
        "_disambiguate_declared_entities",
        _model_path_called,
    )

    contract = compile_canonical_contract_from_state(_golden_state())

    assert contract.model_dump(mode="json") == _load_json(_OUTPUT_PATH)
    assert hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest() == (
        _OUTPUT_SHA256_PATH.read_text(encoding="utf-8").strip()
    )


def test_canonical_compiler_normalizes_metadata_and_serialization() -> None:
    """Injected time/origin and canonical JSON keep parity output byte-stable."""

    contract = compile_contract_from_state(
        _golden_state(),
        origin="fixture",
        enrichment_mode=NO_ENRICHMENT_MODE,
        compiled_at="2026-08-14T05:06:07Z",
    )

    assert contract.origin == "fixture"
    assert contract.compiled_at == "2026-08-14T05:06:07+00:00"
    assert canonical_json(contract) == canonical_json(contract.model_dump(mode="json"))


def test_no_enrichment_uses_deterministic_regex_when_entities_are_not_declared(
    monkeypatch,
) -> None:
    """Disabling providers retains structural entity extraction via deterministic rules."""

    def _model_path_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("no-enrichment mode must not touch a model path")

    monkeypatch.setattr(
        contract_module,
        "_extract_entities_llm_first",
        _model_path_called,
    )
    contract = compile_contract_from_state(
        {
            "mission_id": "S91-REGEX-01",
            "mission_context": {
                "title": "Bundler comparison",
                "objective": "Compare Vite, Webpack, Rollup, and Turbopack.",
            },
        },
        origin="test",
        enrichment_mode=NO_ENRICHMENT_MODE,
        compiled_at="2026-08-14T00:00:00Z",
    )

    assert {"Vite", "Webpack", "Rollup", "Turbopack"}.issubset(
        set(contract.named_entities)
    )
    assert contract.entity_extraction_decisions[-1] == {
        "entity": "",
        "decision": "extraction_path",
        "reason": "regex_no_enrichment",
    }


def test_default_mode_preserves_configured_enrichment(monkeypatch) -> None:
    """Omitting the new option keeps the existing provider-backed compile behavior."""

    monkeypatch.setattr(
        contract_module,
        "_extract_entities_llm_first",
        lambda **kwargs: (["ProviderEntity"], {}, {}, "llm_primary", []),
    )
    contract = compile_contract_from_state(
        {
            "mission_id": "S91-DEFAULT-01",
            "mission_context": {
                "title": "Default compiler mode",
                "objective": "Compile the configured entity path.",
            },
        },
        origin="test",
        compiled_at="2026-08-14T00:00:00Z",
    )

    assert "ProviderEntity" in contract.named_entities
    assert contract.entity_extraction_decisions[-1]["reason"] == "llm_primary"


def test_legacy_contract_does_not_claim_current_compiler_revision() -> None:
    """Loading pre-provenance output preserves its schema and marks revision unknown."""

    payload = _load_json(_OUTPUT_PATH)
    payload["contract_version"] = "1.0"
    payload.pop("contract_schema_version")
    payload.pop("compiler_revision")

    contract = MissionContract.model_validate(payload)

    assert contract.contract_version == "1.0"
    assert contract.contract_schema_version == "1.0"
    assert contract.compiler_revision == 0


def test_manifest_hashes_every_declared_semantic_source() -> None:
    """Each manifest digest independently verifies the source used for parity."""

    manifest = build_contract_compiler_manifest(enrichment_mode=NO_ENRICHMENT_MODE)

    assert manifest["contract_schema_version"] == CONTRACT_SCHEMA_VERSION
    assert manifest["compiler_revision"] == CONTRACT_COMPILER_REVISION
    assert isinstance(manifest["compiler_revision"], int)
    assert manifest["compiler_revision"] >= 1
    assert manifest["source_revision"]
    assert manifest["source_revision_source"]
    assert isinstance(manifest["source_dirty"], bool)
    assert manifest["enrichment"] == {
        "mode": "none",
        "backend": None,
        "model": None,
    }
    assert manifest["canonical_metadata"] == {
        "origin": "canonical",
        "compiled_at": "1970-01-01T00:00:00+00:00",
    }
    assert {
        "mission_id",
        "project_id",
        "mission_context",
        "research_depth",
        "max_loops",
        "min_loops",
        "depth_config",
    }.issubset(set(manifest["semantic_inputs"]["state"]))
    assert {
        "required_entities",
        "excluded_entities",
        "expected_output_schema",
        "coverage_thresholds",
        "validation_thresholds",
        "constraints",
        "references",
    }.issubset(set(manifest["semantic_inputs"]["mission_context"]))

    source_paths = set()
    for source in manifest["required_source_files"]:
        source_path = _REPO_ROOT / source["path"]
        source_paths.add(source["path"])
        assert source_path.is_file()
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]
    assert "deepsearch/mission/contract.py" in source_paths
    assert "deepsearch/mission/entity_extractor_llm.py" not in source_paths

    manifest_without_identity = dict(manifest)
    identity = manifest_without_identity.pop("compiler_manifest_sha256")
    assert hashlib.sha256(
        canonical_json(manifest_without_identity).encode("utf-8")
    ).hexdigest() == identity


def test_configured_manifest_names_provider_dependent_sources() -> None:
    """Default-mode provenance records the selected provider instead of hiding it."""

    manifest = build_contract_compiler_manifest(
        enrichment_mode="configured",
        enrichment_backend="deepseek",
        enrichment_model="deepseek-reasoner",
    )

    assert manifest["enrichment"] == {
        "mode": "configured",
        "backend": "deepseek",
        "model": "deepseek-reasoner",
    }
    source_paths = {source["path"] for source in manifest["required_source_files"]}
    assert "deepsearch/mission/entity_extractor_llm.py" in source_paths
    assert "deepsearch/llm/structured_extractor.py" in source_paths


def test_preview_no_enrichment_returns_golden_and_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    """TraceLab can request canonical parity output through the signed preview surface."""

    monkeypatch.delenv(TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV, raising=False)
    app = create_app()
    app.state.queue = JobQueue(db_path=str(tmp_path / "jobs.db"))
    client = TestClient(app)

    response = client.post("/api/v1/missions/preview", json=_load_json(_INPUT_PATH))

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract"] == _load_json(_OUTPUT_PATH)
    assert payload["contract_schema_version"] == CONTRACT_SCHEMA_VERSION
    assert payload["compiler_revision"] == CONTRACT_COMPILER_REVISION
    assert payload["compiler_manifest"]["enrichment"] == {
        "mode": "none",
        "backend": None,
        "model": None,
    }
    assert (
        payload["compiler_manifest_sha256"]
        == payload["compiler_manifest"]["compiler_manifest_sha256"]
    )
    assert payload["named_entities"] == payload["contract"]["named_entities"]
