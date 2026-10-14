"""Runtime identity and config fingerprint helpers for execution telemetry."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple, cast

from deepsearch.config import DeepSearchSettings

logger = logging.getLogger(__name__)

_BUILD_HASH_ENV_VARS = (
    "DEEPSEARCH_BUILD_HASH",
    "BUILD_HASH",
    "RENDER_GIT_COMMIT",
    "GIT_COMMIT",
    "SOURCE_VERSION",
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "deepsearch"
_BUILD_HASH_MISSING_LOGGED = False


def build_runtime_context(
    *,
    settings: DeepSearchSettings,
    state: Mapping[str, Any],
    llm_backend: str,
    model: str,
    execution_path: str,
    thread_id: str,
    validate: bool = False,
    auto_upload: bool = False,
) -> Dict[str, Any]:
    """Return runtime identity and fingerprintable config for a mission run."""

    extraction_config = settings.build_llm_config(call_type="extraction")
    reflection_config = extraction_config
    config_snapshot = {
        "llm_backend": llm_backend,
        "model": model,
        "extraction_model": extraction_config.model,
        "reflection_model": reflection_config.model,
        "search_backend": settings.search_backend,
        "dual_search": settings.dual_search,
        "triple_search": settings.triple_search,
        "tavily_dual_search": settings.tavily_dual_search,
        "use_v2_prompts": settings.use_v2_prompts,
        "use_new_reflection_parser": settings.use_new_reflection_parser,
        "use_new_citation_engine": settings.use_new_citation_engine,
        "pedr_enable_graph": settings.pedr_enable_graph,
        "research_depth": str(state.get("research_depth") or ""),
        "max_loops": int(state.get("max_loops") or 0),
        "min_loops": int(state.get("min_loops") or 0),
        "depth_config": dict(state.get("depth_config") or {}),
        "validate": bool(validate),
        "auto_upload": bool(auto_upload),
    }
    if settings.llm_backend in {"deepseek", "openai"}:
        from deepsearch.llm.runtime import resolve_effective_model

        config_snapshot["effective_model"] = resolve_effective_model(settings).public_metadata()
    config_fingerprint = _fingerprint_payload(config_snapshot)
    runtime_identity = _build_runtime_identity(
        llm_backend=llm_backend,
        model=model,
        execution_path=execution_path,
        thread_id=thread_id,
    )
    return {
        "runtime_identity": runtime_identity,
        "config_snapshot": config_snapshot,
        "config_fingerprint": config_fingerprint,
    }


def apply_runtime_context(
    state: MutableMapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    """Attach runtime metadata onto an execution state's telemetry payload."""

    telemetry = state.get("telemetry")
    if not isinstance(telemetry, MutableMapping):
        telemetry = {}
    merged = dict(telemetry)
    merged.update(dict(context))
    state["telemetry"] = cast(Dict[str, Any], merged)


def _build_runtime_identity(
    *,
    llm_backend: str,
    model: str,
    execution_path: str,
    thread_id: str,
) -> Dict[str, Any]:
    build_hash, build_source = _resolve_build_hash()
    return {
        "build_hash": build_hash,
        "build_source": build_source,
        "git_dirty": _is_git_dirty(),
        "llm_backend": llm_backend,
        "model": model,
        "execution_path": execution_path,
        "thread_id": thread_id,
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "hostname": platform.node(),
        "pid": os.getpid(),
    }


def _resolve_build_hash() -> Tuple[str, str]:
    for env_var in _BUILD_HASH_ENV_VARS:
        value = (os.getenv(env_var) or "").strip()
        if value:
            return value, f"env:{env_var}"

    git_hash = _run_git("rev-parse", "HEAD")
    if git_hash:
        return git_hash, "git"

    # S41.2 — source-tree-hash fallback. Railway/Nixpacks runtime images
    # do not ship the `.git` directory and do not inject a BUILD_HASH env
    # var by default, so both branches above return empty in production.
    # S40.4's paid artifact came back with build_hash="" for exactly this
    # reason, which blocks deploy-fingerprint verification at the S41.5
    # LAND-AND-PAUSE gate. Hash the packaged deepsearch/ source tree to
    # preserve provenance — not a git SHA, but deterministically
    # distinguishes any two deployed code trees.
    source_hash = _compute_source_tree_hash()
    if source_hash:
        return source_hash, "source_tree_hash"

    _log_build_hash_missing()
    return "", "unknown"


def _compute_source_tree_hash() -> str:
    """Deterministic content hash of the packaged ``deepsearch/`` source tree.

    Preserves deploy-fingerprint provenance when neither env-var-injected
    build hashes nor an in-image ``.git`` directory are available at
    runtime (the standard Railway/Nixpacks runtime layout).
    """

    if not _PACKAGE_ROOT.is_dir():
        return ""
    try:
        files = sorted(
            path
            for path in _PACKAGE_ROOT.rglob("*.py")
            if path.is_file() and "__pycache__" not in path.parts
        )
    except OSError:
        return ""
    if not files:
        return ""
    hasher = hashlib.sha256()
    for path in files:
        try:
            content = path.read_bytes()
        except OSError:
            continue
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(content)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _log_build_hash_missing() -> None:
    """Loud-log once when build_hash cannot be resolved at all.

    In paid/DEEP environments, unresolved build_hash blocks deploy-
    fingerprint verification at the S41.5 LAND-AND-PAUSE authorization
    gate. Degrade gracefully (do not refuse to serve) but surface the gap
    prominently in operator logs.
    """

    global _BUILD_HASH_MISSING_LOGGED
    if _BUILD_HASH_MISSING_LOGGED:
        return
    logger.error(
        "runtime_identity.build_hash unresolved: no BUILD_HASH env var set, "
        "no in-image `.git` directory, and source-tree hash unavailable. "
        "Paid-run provenance cannot be verified — deploy-fingerprint "
        "verification at S41.5 LAND-AND-PAUSE will fail."
    )
    _BUILD_HASH_MISSING_LOGGED = True


def _is_git_dirty() -> bool:
    status = _run_git("status", "--short")
    return bool(status)


def _run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            cwd=_REPO_ROOT,
            text=True,
        )
    except OSError:
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "apply_runtime_context",
    "build_runtime_context",
]
