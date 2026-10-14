"""S97 m01 static guard: runtime code never names the project-record folder.

DeepSearch's public repository ships without ``cmos/``, and the record folder
itself is being renamed (Reasonbook 4.0, ``cmos/`` -> ``rbook/``). Runtime code
that writes into or reads from either folder breaks in a public checkout, or
silently re-creates a private folder there. Runtime output belongs under the
gitignored ``checkpoints/`` tree (or the existing ``DEEPSEARCH_*_DIR``
overrides). Comments and docstrings may cite historical record documents; only
executable string literals are checked.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "deepsearch"

_RECORD_PATH = re.compile(r"(?<![\w.-])(?:cmos|rbook)/")
_RECORD_SEGMENTS = {"cmos", "rbook"}

# Literals that mention the folder without addressing it on disk.
_ALLOWED = {
    # Mission-content validator rejecting research objectives that ask
    # DeepSearch to write into the internal folder.
    ("api/validators.py", r"cmos/"),
    ("api/validators.py", "internal cmos/ path"),
}


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
        # Attribute docstrings: a bare string expression after an assignment.
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    ids.add(id(stmt.value))
    return ids


def _violations() -> list[str]:
    found: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = _docstring_ids(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in docstrings or (rel, node.value) in _ALLOWED:
                continue
            if _RECORD_PATH.search(node.value) or node.value in _RECORD_SEGMENTS:
                found.append(f"{rel}:{node.lineno}: {node.value[:80]!r}")
    return found


def test_runtime_code_never_addresses_the_record_folder() -> None:
    assert _violations() == []


def test_guard_detects_record_folder_literals(tmp_path: Path) -> None:
    """The guard must fail on the shapes S97 removed, or it guards nothing."""

    samples = {
        'DEFAULT = Path("cmos/research")': True,
        'root / "cmos" / "db"': True,
        'DEFAULT = Path("rbook/telemetry/x.jsonl")': True,
        'DEFAULT = Path("checkpoints/reports")': False,
        'label = "telemetry/cmos_like"': False,
    }
    for source, should_flag in samples.items():
        tree = ast.parse(source)
        flagged = any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (_RECORD_PATH.search(node.value) or node.value in _RECORD_SEGMENTS)
            for node in ast.walk(tree)
        )
        assert flagged is should_flag, source


def test_removed_record_loader_and_settings_have_no_runtime_consumers() -> None:
    """Definition and consumer guards prevent stripped-field false greens."""
    from dataclasses import fields

    import deepsearch.mission as mission
    from deepsearch.config import DeepSearchSettings

    assert "cmos_db_path" not in {item.name for item in fields(DeepSearchSettings)}
    assert not hasattr(mission, "load_mission_from_cmos")
    assert "load_mission_from_cmos" not in mission.__all__
    violations = []
    removed = {"cmos_db_path", "load_mission_from_cmos", "DEFAULT_CMOS_DB_PATH"}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
            if name in removed:
                violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                if any(
                    name in removed or name.split(".")[0] in _RECORD_SEGMENTS
                    for name in names
                ):
                    violations.append(f"{path.name}:{node.lineno}: record import")
    assert violations == []
