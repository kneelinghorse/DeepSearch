"""S70 m01 — Read-only Postgres extractor for cached mission protocols.

Pulls ``result_protocol`` JSONB from TraceLab/Railway Postgres ``missions``
for a given ``mission_id`` (the human-readable text column, e.g.
``OODS-FIGMA-HOST-01-S68-WIRING`` — NOT the UUID primary key) and writes it
to local disk where the S69 FActScore harness can score it.

Local-only utility. PD11 (S70 plan-doc) names this Option A: bypass LLM
cost by reusing the already-paid S68 wiring run output instead of running
a fresh local mission. Read-only SELECT only; no DB writes; no TraceLab
worker dispatch.

PD10 carry-forward — no production-runtime imports (codeagent_finalize_node,
citation_liveness, critique, contract_to_prompt, codeagent_runner,
worker.writer, agent.tools.web_search / source_fetch). The shared async
asyncpg wrapper in :mod:`deepsearch.worker.database` is allowed: it is a
bare client, not mission-runtime processing logic. The data-path helper uses
only the standard library; neither import starts a worker or a model request.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from deepsearch.data_paths import data_path
from deepsearch.worker.database import DatabaseClient, DatabaseConfig

logger = logging.getLogger(__name__)


EXTRACT_PROTOCOL_SQL = """
    SELECT result_protocol, result_markdown
    FROM missions
    WHERE mission_id = $1
    LIMIT 1
"""


def _markdown_sibling(protocol_path: Path) -> Path:
    """``<stem>_protocol.json`` -> sibling ``<stem>.md`` (save_report convention).

    The S69 FActScore harness and the S85 benchmark harness both resolve the
    report body by this convention, so writing it here makes an extracted
    mission scoreable end-to-end from Postgres.
    """
    name = protocol_path.name
    stem = name[: -len("_protocol.json")] if name.endswith("_protocol.json") else protocol_path.stem
    return protocol_path.with_name(f"{stem}.md")


class ProtocolExtractError(Exception):
    """Base exception for :func:`extract_protocol` failures."""


class ProtocolExtractEnvMissingError(ProtocolExtractError):
    """``DATABASE_URL`` (or explicit ``conn_str``) is not set."""


class ProtocolExtractMissionNotFoundError(ProtocolExtractError):
    """No ``missions.mission_id = <mission_id>`` row matched."""


class ProtocolExtractMissingProtocolError(ProtocolExtractError):
    """Matched row has ``NULL result_protocol`` (mission not completed yet,
    or the writer never persisted it)."""


def _output_path_for(mission_id: str, out_path: Optional[Path]) -> Path:
    if out_path is not None:
        return out_path
    sanitised = mission_id.strip()
    return data_path("protocols") / f"{sanitised}_protocol.json"


def _decode_protocol(raw: Any, mission_id: str) -> dict:
    """Coerce ``result_protocol`` into a Python ``dict``.

    asyncpg may surface JSONB as a Python ``dict`` directly or as a raw
    JSON string depending on driver configuration. Both shapes round-trip
    through ``json.loads``; non-object values raise.
    """
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolExtractError(
                f"Mission {mission_id!r}: result_protocol failed JSON decode: {exc!s}"
            ) from exc
    else:
        data = raw

    if not isinstance(data, dict):
        raise ProtocolExtractError(
            f"Mission {mission_id!r}: result_protocol decoded to "
            f"{type(data).__name__}, expected JSON object."
        )
    return data


async def extract_protocol(
    mission_id: str,
    *,
    conn_str: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Extract the cached ``result_protocol`` JSONB for ``mission_id`` to disk.

    Args:
        mission_id: Human-readable mission identifier (the
            ``missions.mission_id`` TEXT column; e.g.
            ``OODS-FIGMA-HOST-01-S68-WIRING``). NOT the UUID
            ``missions.id``.
        conn_str: Optional Postgres connection URL. Defaults to
            ``os.environ["DATABASE_URL"]`` — the worker convention; same
            env var the S63 boundary contract names.
        out_path: Optional override for the output JSON path. Defaults to
            ``<data root>/protocols/<mission_id>_protocol.json``, which
            ``resolve_protocol_path()`` walks via prefix match.

    Returns:
        ``Path`` written.

    Raises:
        ValueError: ``mission_id`` is empty / whitespace.
        ProtocolExtractEnvMissingError: ``DATABASE_URL`` not set and no
            ``conn_str`` provided.
        ProtocolExtractMissionNotFoundError: No matching ``mission_id``.
        ProtocolExtractMissingProtocolError: Matching row has NULL
            ``result_protocol``.
        ProtocolExtractError: JSON decode failed or decoded shape is not
            an object.
    """
    sanitised = mission_id.strip()
    if not sanitised:
        raise ValueError("mission_id is empty; cannot extract protocol")

    url = (conn_str or os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        raise ProtocolExtractEnvMissingError(
            "DATABASE_URL is not set; export it (or pass conn_str=) before "
            "running the extractor. The S63 boundary contract names this env "
            "var; see reference_tracelab_boundary_contract.md."
        )

    target = _output_path_for(sanitised, out_path)

    config = DatabaseConfig(url=url, min_pool_size=1, max_pool_size=2)
    client = DatabaseClient(config)
    try:
        await client.connect()
        record = await client.fetchone(EXTRACT_PROTOCOL_SQL, sanitised)
    finally:
        await client.close()

    if record is None:
        raise ProtocolExtractMissionNotFoundError(
            f"No mission with mission_id={sanitised!r} found in `missions`. "
            f"Verify the mission has been dispatched + completed on the worker."
        )

    raw_protocol = record["result_protocol"]
    if raw_protocol is None:
        raise ProtocolExtractMissingProtocolError(
            f"Mission {sanitised!r} has NULL result_protocol. The writer only "
            f"persists this column on `mark_completed`; check the mission "
            f"status / error_message before re-running extraction."
        )

    data = _decode_protocol(raw_protocol, sanitised)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "extract_protocol: wrote %d bytes for mission %s to %s",
        target.stat().st_size,
        sanitised,
        target,
    )

    # S85 m02 — also persist the report body (``result_markdown``) as the
    # sibling ``<stem>.md`` so the benchmark/FActScore harnesses find it by
    # convention. A NULL body is skipped (the harness treats a missing sibling
    # as an empty body and flags it) — never write a "None" file.
    body = record["result_markdown"] if "result_markdown" in record else None
    if isinstance(body, str) and body:
        md_path = _markdown_sibling(target)
        md_path.write_text(body, encoding="utf-8")
        logger.info(
            "extract_protocol: wrote %d-byte report body for mission %s to %s",
            len(body),
            sanitised,
            md_path,
        )

    return target


__all__ = [
    "EXTRACT_PROTOCOL_SQL",
    "ProtocolExtractError",
    "ProtocolExtractEnvMissingError",
    "ProtocolExtractMissionNotFoundError",
    "ProtocolExtractMissingProtocolError",
    "extract_protocol",
]
