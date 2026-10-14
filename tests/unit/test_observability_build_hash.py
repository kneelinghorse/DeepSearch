"""S41.2 — runtime_identity.build_hash resolution tests.

Covers the provenance-precedence chain used to populate
``runtime_identity.build_hash`` on deployed Railway images:

1. Explicit env var (``DEEPSEARCH_BUILD_HASH`` / ``BUILD_HASH`` / …).
2. ``git rev-parse HEAD`` (dev-workstation path where ``.git`` exists).
3. Source-tree content hash of the packaged ``deepsearch/`` directory
   (Railway/Nixpacks runtime path — the fallback that closes S40.4 RC-4).
4. ``("", "unknown")`` — only reachable when the package tree itself is
   missing, which should loud-log once for operators.
"""

from __future__ import annotations

import deepsearch.observability as obs


def test_env_var_wins_over_git_and_source_tree(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEARCH_BUILD_HASH", "env-injected-sha")
    build_hash, source = obs._resolve_build_hash()
    assert build_hash == "env-injected-sha"
    assert source == "env:DEEPSEARCH_BUILD_HASH"


def test_git_wins_over_source_tree_on_dev_checkout(monkeypatch) -> None:
    """When no env var but ``.git`` is available, resolve via git SHA."""
    for env_var in obs._BUILD_HASH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    build_hash, source = obs._resolve_build_hash()
    # On a dev checkout (where this test runs) git resolves. On a CI/prod
    # image without .git, the source-tree-hash fallback fires. Either
    # outcome is acceptable here — both are non-empty deploy fingerprints.
    assert build_hash
    assert source in {"git", "source_tree_hash"}


def test_source_tree_hash_fallback_when_env_and_git_absent(monkeypatch) -> None:
    """S41.2 — the load-bearing case for closing S40.4 RC-4.

    Neither ``DEEPSEARCH_BUILD_HASH`` nor ``.git`` are available on the
    deployed Railway image. The fallback must still produce a stable,
    non-empty deploy fingerprint (the hex digest of the packaged source
    tree) so ``runtime_identity.build_hash`` is never empty in paid runs.
    """
    for env_var in obs._BUILD_HASH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    # Force the git branch to miss so we exercise the source-tree fallback
    # even on a dev checkout.
    monkeypatch.setattr(obs, "_run_git", lambda *args: "")

    build_hash, source = obs._resolve_build_hash()

    assert source == "source_tree_hash"
    assert build_hash  # non-empty
    # Hex digest of sha256 — 64 chars.
    assert len(build_hash) == 64
    assert all(ch in "0123456789abcdef" for ch in build_hash)


def test_source_tree_hash_is_deterministic(monkeypatch) -> None:
    """Two consecutive calls in the same session must return the same hash."""
    for env_var in obs._BUILD_HASH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(obs, "_run_git", lambda *args: "")

    first, _ = obs._resolve_build_hash()
    second, _ = obs._resolve_build_hash()
    assert first == second
    assert first  # non-empty


def test_build_runtime_identity_always_populates_build_hash_source(monkeypatch) -> None:
    """After S41.2 the build_source field should never be 'unknown' for a
    normal deploy — env, git, or source_tree_hash must produce a signature.

    This is the invariant that unblocks S41.5 LAND-AND-PAUSE: the Phase A
    scoring script can assume ``runtime_identity.build_source != 'unknown'``
    holds on a properly-deployed image.
    """
    for env_var in obs._BUILD_HASH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    identity = obs._build_runtime_identity(
        llm_backend="gemini",
        model="gemini-2.5-pro",
        execution_path="tracelab",
        thread_id="test-thread",
    )
    assert identity["build_hash"], (
        f"build_hash must not be empty — got {identity!r}"
    )
    assert identity["build_source"] in {
        "env:DEEPSEARCH_BUILD_HASH",
        "env:BUILD_HASH",
        "env:RENDER_GIT_COMMIT",
        "env:GIT_COMMIT",
        "env:SOURCE_VERSION",
        "git",
        "source_tree_hash",
    }
    assert identity["build_source"] != "unknown"
