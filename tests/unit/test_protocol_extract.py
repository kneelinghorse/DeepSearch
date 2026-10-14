"""S70 m01 runtime guards for the read-only Postgres protocol extractor.

Layered-guard pattern (RD2, seventh consecutive sprint). These tests
exercise :mod:`deepsearch.eval.protocol_extract` and the CLI dispatch
into :mod:`deepsearch.eval.__main__` without touching real Postgres —
``DatabaseClient`` is monkeypatched at the module-level import alias so
no asyncpg connection is opened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from deepsearch.eval import __main__ as cli
from deepsearch.eval import protocol_extract as extractor
from deepsearch.eval.protocol_extract import (
    EXTRACT_PROTOCOL_SQL,
    ProtocolExtractEnvMissingError,
    ProtocolExtractError,
    ProtocolExtractMissingProtocolError,
    ProtocolExtractMissionNotFoundError,
    extract_protocol,
)


# ---------------------------------------------------------------------------
# Fake DatabaseClient — captures the SQL + arg, returns a configurable record.
# ---------------------------------------------------------------------------


class _FakeRecord(dict):
    """asyncpg.Record stand-in. The extractor only does dict-style lookup."""


class _FakeClient:
    """Minimal DatabaseClient stand-in.

    Records the (query, args) tuple it was called with; returns a
    configurable record (or None for not-found) on ``fetchone``.
    """

    last_sql: Optional[str] = None
    last_args: tuple = ()
    closed: bool = False
    connected: bool = False

    def __init__(self, record: Optional[Dict[str, Any]]) -> None:
        self._record = record

    async def connect(self) -> None:
        type(self).connected = True

    async def close(self) -> None:
        type(self).closed = True

    async def fetchone(self, query: str, *args: Any) -> Optional[_FakeRecord]:
        type(self).last_sql = query
        type(self).last_args = args
        if self._record is None:
            return None
        return _FakeRecord(self._record)


def _patch_client(monkeypatch, record: Optional[Dict[str, Any]]) -> type[_FakeClient]:
    """Patch the ``DatabaseClient`` symbol referenced inside the extractor."""

    instances: list[_FakeClient] = []

    def _factory(_config) -> _FakeClient:
        instance = _FakeClient(record)
        instances.append(instance)
        return instance

    monkeypatch.setattr(extractor, "DatabaseClient", _factory)
    # Reset class state from any prior test
    _FakeClient.last_sql = None
    _FakeClient.last_args = ()
    _FakeClient.closed = False
    _FakeClient.connected = False
    return _FakeClient


# ---------------------------------------------------------------------------
# 1. Happy path — JSONB returned as a Python dict (typical asyncpg behaviour).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_writes_dict_payload(monkeypatch, tmp_path):
    payload = {
        "mission_id": "OODS-FIGMA-HOST-01-S68-WIRING",
        "citations": [{"url": "https://x.example", "live": True}],
        "evidence": [],
    }
    _patch_client(monkeypatch, {"result_protocol": payload})

    out_path = tmp_path / "extracted_protocol.json"
    result = await extract_protocol(
        "OODS-FIGMA-HOST-01-S68-WIRING",
        conn_str="postgresql://user:pw@example/db",
        out_path=out_path,
    )

    assert result == out_path
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == payload
    assert _FakeClient.connected is True
    assert _FakeClient.closed is True
    assert _FakeClient.last_sql.strip().startswith("SELECT result_protocol")
    assert _FakeClient.last_args == ("OODS-FIGMA-HOST-01-S68-WIRING",)


# ---------------------------------------------------------------------------
# 2. JSONB returned as a string (alt asyncpg config) — must still decode.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_decodes_jsonb_string(monkeypatch, tmp_path):
    payload = {"citations": [{"url": "https://y.example", "live": True}]}
    _patch_client(monkeypatch, {"result_protocol": json.dumps(payload)})

    out_path = tmp_path / "M_protocol.json"
    result = await extract_protocol(
        "M",
        conn_str="postgresql://user:pw@example/db",
        out_path=out_path,
    )

    assert json.loads(result.read_text(encoding="utf-8")) == payload


# ---------------------------------------------------------------------------
# 3. Default out_path lands in checkpoints/protocols/<mission_id>_protocol.json,
#    never in the project-record folder.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_default_out_path_under_checkpoints_protocols(
    monkeypatch, tmp_path
):
    payload = {"citations": []}
    _patch_client(monkeypatch, {"result_protocol": payload})

    monkeypatch.chdir(tmp_path)
    result = await extract_protocol(
        "MISSION-A",
        conn_str="postgresql://user:pw@example/db",
    )

    expected = tmp_path / "checkpoints" / "protocols" / "MISSION-A_protocol.json"
    assert result == Path("checkpoints/protocols/MISSION-A_protocol.json")
    assert result.resolve() == expected.resolve()
    assert expected.exists()


# ---------------------------------------------------------------------------
# 4. resolve_protocol_path() round-trip — extractor lands a discoverable file.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_lands_in_resolve_protocol_path_search_dir(
    monkeypatch, tmp_path,
):
    """The default landing path must satisfy the S69 prefix-match discovery."""
    from deepsearch.eval.factscore import resolve_protocol_path

    payload = {"citations": []}
    _patch_client(monkeypatch, {"result_protocol": payload})
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEARCH_REPORT_DIR", raising=False)

    written = await extract_protocol(
        "MISSION-Q",
        conn_str="postgresql://user:pw@example/db",
    )

    # Default search dirs, no override: the extractor default must be one of them.
    discovered = resolve_protocol_path("MISSION-Q")
    # Both resolve to the same absolute path; written may be relative.
    assert (tmp_path / discovered).resolve() == (tmp_path / written).resolve()


# ---------------------------------------------------------------------------
# 5. Empty mission_id raises ValueError (input-validation contract).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_rejects_empty_mission_id():
    with pytest.raises(ValueError):
        await extract_protocol("   ", conn_str="postgresql://x/y")


# ---------------------------------------------------------------------------
# 6. Missing DATABASE_URL (no override) raises EnvMissingError before any DB I/O.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_raises_when_database_url_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def _no_db(_config):
        raise AssertionError("DatabaseClient must NOT be constructed when DATABASE_URL is unset")

    monkeypatch.setattr(extractor, "DatabaseClient", _no_db)
    with pytest.raises(ProtocolExtractEnvMissingError):
        await extract_protocol("M")


# ---------------------------------------------------------------------------
# 7. Mission not found in `missions` raises MissionNotFoundError.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_raises_when_mission_id_not_found(monkeypatch):
    _patch_client(monkeypatch, None)

    with pytest.raises(ProtocolExtractMissionNotFoundError):
        await extract_protocol(
            "NO-SUCH-MISSION",
            conn_str="postgresql://user:pw@example/db",
        )


# ---------------------------------------------------------------------------
# 8. NULL result_protocol raises MissingProtocolError.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_raises_when_result_protocol_is_null(monkeypatch):
    _patch_client(monkeypatch, {"result_protocol": None})

    with pytest.raises(ProtocolExtractMissingProtocolError):
        await extract_protocol(
            "MISSION-NO-RESULT",
            conn_str="postgresql://user:pw@example/db",
        )


# ---------------------------------------------------------------------------
# 9. Malformed JSON in the JSONB column raises base ProtocolExtractError.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_raises_on_malformed_json(monkeypatch):
    _patch_client(monkeypatch, {"result_protocol": "{not valid json"})

    with pytest.raises(ProtocolExtractError) as info:
        await extract_protocol(
            "MISSION-BAD",
            conn_str="postgresql://user:pw@example/db",
        )
    assert "JSON decode" in str(info.value)


# ---------------------------------------------------------------------------
# 10. result_protocol decoded as a non-object (e.g., list) raises base error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_raises_when_decoded_shape_is_not_object(monkeypatch):
    _patch_client(monkeypatch, {"result_protocol": ["not", "an", "object"]})

    with pytest.raises(ProtocolExtractError) as info:
        await extract_protocol(
            "MISSION-LIST",
            conn_str="postgresql://user:pw@example/db",
        )
    assert "expected JSON object" in str(info.value)


# ---------------------------------------------------------------------------
# 11. DATABASE_URL env var fallback works when no explicit conn_str passed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_reads_database_url_from_env(monkeypatch, tmp_path):
    payload = {"citations": []}
    captured: list[Any] = []

    def _factory(config):
        captured.append(config.url)
        return _FakeClient({"result_protocol": payload})

    monkeypatch.setattr(extractor, "DatabaseClient", _factory)
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")

    out_path = tmp_path / "M_protocol.json"
    await extract_protocol("M", out_path=out_path)

    assert captured == ["postgresql://from-env/db"]


# ---------------------------------------------------------------------------
# 12. Explicit conn_str takes precedence over DATABASE_URL env var.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_protocol_conn_str_overrides_env(monkeypatch, tmp_path):
    payload = {"citations": []}
    captured: list[Any] = []

    def _factory(config):
        captured.append(config.url)
        return _FakeClient({"result_protocol": payload})

    monkeypatch.setattr(extractor, "DatabaseClient", _factory)
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")

    out_path = tmp_path / "M_protocol.json"
    await extract_protocol(
        "M",
        conn_str="postgresql://override/db",
        out_path=out_path,
    )

    assert captured == ["postgresql://override/db"]


# ---------------------------------------------------------------------------
# 13. SQL invariant — exactly one ``SELECT result_protocol FROM missions``
#     scoped by ``mission_id``. Catches an accidental rewrite that would
#     widen the SELECT (cost), drop the WHERE (full table scan), or hit the
#     wrong column.
# ---------------------------------------------------------------------------


def test_extract_protocol_sql_is_read_only_and_bounded():
    sql_lower = EXTRACT_PROTOCOL_SQL.lower().strip()
    assert sql_lower.startswith("select result_protocol"), (
        "EXTRACT_PROTOCOL_SQL must start with `SELECT result_protocol` — never widen the SELECT"
    )
    assert "from missions" in sql_lower
    assert "where mission_id = $1" in sql_lower
    for keyword in (" insert ", " update ", " delete ", " drop ", " truncate ", " alter "):
        assert keyword not in f" {sql_lower} ", (
            f"EXTRACT_PROTOCOL_SQL is read-only; {keyword.strip()} keyword leaked"
        )


# ---------------------------------------------------------------------------
# 14. CLI ``extract`` subcommand happy path — exit 0 + protocol on disk.
# ---------------------------------------------------------------------------


def test_cli_extract_subcommand_routes_to_extractor(monkeypatch, tmp_path, capsys):
    """``main(["extract", <mission_id>, ...])`` must dispatch to extract_protocol."""
    captured: dict = {}

    async def _fake_extract(mission_id, *, conn_str=None, out_path=None):
        captured["mission_id"] = mission_id
        captured["conn_str"] = conn_str
        captured["out_path"] = out_path
        target = out_path or (tmp_path / "default" / f"{mission_id}_protocol.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        return target

    monkeypatch.setattr(cli, "extract_protocol", _fake_extract)

    out_path = tmp_path / "out_protocol.json"
    rc = cli.main(
        [
            "extract",
            "MISSION-Z",
            "--out",
            str(out_path),
            "--database-url",
            "postgresql://user:pw@example/db",
        ]
    )

    assert rc == 0
    assert captured["mission_id"] == "MISSION-Z"
    assert captured["conn_str"] == "postgresql://user:pw@example/db"
    assert captured["out_path"] == out_path
    stdout = capsys.readouterr().out.strip()
    assert stdout == str(out_path)


# ---------------------------------------------------------------------------
# 15. CLI ``extract`` returns 2 with a clean stderr message on extractor error.
# ---------------------------------------------------------------------------


def test_cli_extract_subcommand_returns_two_on_extract_error(monkeypatch, capsys):
    async def _fake_extract(_mission_id, **_kwargs):
        raise ProtocolExtractMissionNotFoundError("no such mission MISSION-DNE")

    monkeypatch.setattr(cli, "extract_protocol", _fake_extract)

    rc = cli.main(["extract", "MISSION-DNE"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "MISSION-DNE" in err


# ---------------------------------------------------------------------------
# 16. Existing FActScore CLI invocation forms still work after the dispatch
#     wrapper lands. Smoke-test the no-arg branch.
# ---------------------------------------------------------------------------


def test_cli_factscore_path_unchanged_after_extract_dispatch(capsys):
    rc = cli.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "mission_id" in err or "protocol" in err
