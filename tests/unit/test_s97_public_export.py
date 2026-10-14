"""Public snapshot safety is tested only against disposable repositories."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from scripts import export_public as export


def git(root: Path, *args: str) -> str:
    return export._git(root, *args).decode().strip()


def commit(source: Path) -> str:
    git(source, "add", "--all")
    git(source, "commit", "-qm", "Disposable test input")
    return git(source, "rev-parse", "HEAD")


def approve(source: Path, files: dict[str, bytes], *, inventory: list[dict] | None = None) -> str:
    """Authorize known synthetic safety-test bytes, never production fixtures."""
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {
        "format": export.FORMAT, "status": "approved", "public_fixture_gate": "approved",
        "files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                   "mode": (source / name).stat().st_mode & 0o777,
                   "publication": "approved", "basis": "Authored disposable safety fixture",
                   "license": "MIT"} for name, data in sorted(files.items())],
    }
    (source / "scripts").mkdir(exist_ok=True)
    (source / export.MANIFEST).write_text(json.dumps(manifest))
    (source / "tests/fixtures").mkdir(parents=True, exist_ok=True)
    (source / "tests/fixtures/publication_inventory.json").write_text(
        json.dumps({"fixtures": inventory or []}))
    return commit(source)


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Resolve the system temp alias before testing the explicit alias rejection.
    source = tmp_path.resolve() / "private"
    source.mkdir()
    git(source, "init", "-q", "-b", "main")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Offline Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "offline@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Offline Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "offline@example.invalid")
    approve(source, {"README.md": b"Public authored example\n", "bin/run": b"print('example')\n"})
    (source / "bin/run").chmod(0o755)
    approve(source, {"README.md": b"Public authored example\n",
                     "bin/run": b"print('example')\n"})
    return source


@pytest.fixture
def scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[set[str]]:
    seen: list[set[str]] = []

    def inspect(stage: Path) -> Path:
        seen.append({str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()})
        assert (stage / export.RECEIPT).is_file()
        return tmp_path / "redacted-scan.json"

    monkeypatch.setattr(export, "_scan_stage", inspect)
    return seen


def test_export_is_committed_deterministic_and_has_no_private_ancestry(
    source: Path, scan: list[set[str]],
) -> None:
    prepared = export.prepare(source)
    # Working-tree changes and unrelated committed files must not enter a snapshot.
    (source / "README.md").write_text("uncommitted private text")
    (source / "private-notes").write_text("untracked")
    assert export.prepare(source) == prepared
    out = source.parent / "public"
    result = export.sync(prepared, source, out)
    receipt = json.loads((out / export.RECEIPT).read_text())
    assert (out / "README.md").read_bytes() == b"Public authored example\n"
    assert receipt["source_commit"] == prepared.source_commit
    assert receipt["source_tree"] == git(source, "rev-parse", prepared.source_commit + "^{tree}")
    assert receipt["reviewed_manifest"] == json.loads(
        export._git(source, "show", f"{prepared.source_commit}:{export.MANIFEST}"))
    assert (out / "bin/run").stat().st_mode & 0o777 == 0o755
    assert git(out, "rev-list", "--count", "HEAD") == "1"
    assert prepared.source_commit not in git(out, "rev-list", "--all")
    assert git(out, "remote") == ""
    again = export.sync(prepared, source, out, result["head"])
    assert again["head"] == result["head"]
    other = source.parent / "other-public"
    export.sync(prepared, source, other)
    assert git(other, "rev-parse", "HEAD^{tree}") == git(out, "rev-parse", "HEAD^{tree}")
    assert len(scan) == 3


@pytest.mark.parametrize("target", ["root", "home", "source", "ancestor", "descendant", "alias"])
def test_unsafe_destinations_are_refused_before_writes(
    source: Path, scan: list[set[str]], target: str,
) -> None:
    destinations = {"root": Path(source.anchor), "home": Path.home().resolve(),
                    "source": source, "ancestor": source.parent, "descendant": source / "child"}
    if target == "alias":
        alias = source.parent / "alias"
        alias.symlink_to(source.parent, target_is_directory=True)
        out = alias / "new-public"
    else:
        out = destinations[target]
    before = (source / "README.md").read_bytes()
    with pytest.raises(export.ExportError):
        export.sync(export.prepare(source), source, out)
    assert (source / "README.md").read_bytes() == before
    assert not scan


def test_unmanaged_repo_and_nonempty_directory_are_preserved(source: Path, scan: list) -> None:
    for name in ("unmanaged", "user-data"):
        out = source.parent / name
        out.mkdir()
        (out / "precious").write_bytes(b"preserve me")
        if name == "unmanaged":
            git(out, "init", "-q")
            commit(out)
        with pytest.raises(export.ExportError):
            export.sync(export.prepare(source), source, out, git(out, "rev-parse", "HEAD")
                        if name == "unmanaged" else None)
        assert (out / "precious").read_bytes() == b"preserve me"


@pytest.mark.parametrize("kind", ["tracked", "staged", "untracked", "ignored", "empty_dir"])
def test_dirty_or_unknown_destination_data_is_never_deleted(
    source: Path, scan: list, kind: str,
) -> None:
    approve(source, {"README.md": b"public\n", ".gitignore": b"local.cache\n"})
    prepared = export.prepare(source)
    out = source.parent / "public"
    head = export.sync(prepared, source, out)["head"]
    if kind in {"tracked", "staged"}:
        path = out / "README.md"
        path.write_bytes(b"user edit")
        if kind == "staged":
            git(out, "add", "README.md")
    else:
        path = out / ("local.cache" if kind == "ignored" else "unknown")
        if kind == "empty_dir":
            path.mkdir()
        else:
            path.write_bytes(b"private local data")
    with pytest.raises(export.ExportError):
        export.sync(prepared, source, out, head)
    assert path.exists()
    if kind != "empty_dir":
        assert path.read_bytes() in {b"user edit", b"private local data"}
    assert git(out, "rev-parse", "HEAD") == head


def test_sync_deletes_only_previous_managed_paths(source: Path, scan: list) -> None:
    out = source.parent / "public"
    head = export.sync(export.prepare(source), source, out)["head"]
    approve(source, {"README.md": b"updated\n", "new.txt": b"new\n"})
    result = export.sync(export.prepare(source), source, out, head)
    assert not (out / "bin").exists()
    assert (out / "new.txt").read_bytes() == b"new\n"
    assert git(out, "rev-parse", "HEAD^") == head
    assert export.inspect_destination(source, out, result["head"])["head"] == result["head"]


def test_scan_failure_leaves_destination_byte_exact(
    source: Path, scan: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = source.parent / "public"
    head = export.sync(export.prepare(source), source, out)["head"]
    original = (out / "README.md").read_bytes()
    approve(source, {"README.md": b"changed\n"})

    def failed(stage: Path) -> Path:
        raise export.ExportError("Injected scan failure")

    monkeypatch.setattr(export, "_scan_stage", failed)
    with pytest.raises(export.ExportError, match="scan failure"):
        export.sync(export.prepare(source), source, out, head)
    assert (out / "README.md").read_bytes() == original
    assert git(out, "status", "--porcelain") == ""
    assert git(out, "rev-parse", "HEAD") == head


def test_destination_is_rechecked_after_scan(
    source: Path, scan: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = source.parent / "public"
    head = export.sync(export.prepare(source), source, out)["head"]

    def changed(stage: Path) -> Path:
        (out / "unknown").write_bytes(b"arrived while scanning")
        return stage.parent / "scan.json"

    monkeypatch.setattr(export, "_scan_stage", changed)
    with pytest.raises(export.ExportError, match="dirty"):
        export.sync(export.prepare(source), source, out, head)
    assert (out / "unknown").read_bytes() == b"arrived while scanning"
    assert git(out, "rev-parse", "HEAD") == head


def test_mid_write_failure_leaves_prior_commit_recoverable_and_refuses_rerun(
    source: Path, scan: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = source.parent / "public"
    head = export.sync(export.prepare(source), source, out)["head"]
    original = export._git(out, "show", "HEAD:README.md")
    approve(source, {"README.md": b"changed\n", "z.txt": b"new\n"})
    prepared = export.prepare(source)
    write = export._write_file

    def fail(out_path: Path, file: export.ExportFile, previous: Any) -> None:
        if out_path == out and file.path == "z.txt":
            (out / "unknown").write_bytes(b"user data arriving mid-write")
            raise OSError("Injected disk failure")
        write(out_path, file, previous)

    monkeypatch.setattr(export, "_write_file", fail)
    with pytest.raises(export.ExportError, match="Incomplete export"):
        export.sync(prepared, source, out, head)
    assert git(out, "rev-parse", "HEAD") == head
    assert export._git(out, "show", "HEAD:README.md") == original
    assert (out / "README.md").read_bytes() == b"changed\n"
    assert (out / "unknown").read_bytes() == b"user data arriving mid-write"
    with pytest.raises(export.ExportError, match="dirty"):
        export.sync(prepared, source, out, head)


@pytest.mark.parametrize("data,reason", [(b"x" * 9000 + b"\0", "NUL"),
                                         (b"bad\xff", "UTF-8"),
                                         (b"x" * 1_000_001, "Oversized")],
                         ids=["late-nul", "invalid-utf8", "oversized"])
def test_full_content_checks_precede_destination_creation(
    source: Path, data: bytes, reason: str,
) -> None:
    approve(source, {"payload.txt": data})
    with pytest.raises(export.ExportError, match=reason):
        export.prepare(source)
    assert not (source.parent / "public").exists()


@pytest.mark.parametrize("entry_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE,
                                        tarfile.FIFOTYPE])
def test_selected_unsupported_archive_entries_fail_loud(entry_type: bytes) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        entry = tarfile.TarInfo("selected")
        entry.type = entry_type
        entry.linkname = "outside"
        archive.addfile(entry)
    with pytest.raises(export.ExportError, match="Unsupported"):
        export._archive_files(buffer.getvalue(), {"selected"})


def test_selected_git_symlink_is_rejected(source: Path) -> None:
    approve(source, {"link.txt": b"README.md"})
    (source / "link.txt").unlink()
    (source / "link.txt").symlink_to("README.md")
    commit(source)
    with pytest.raises(export.ExportError, match="Unsupported"):
        export.prepare(source)


def test_new_content_needs_review_and_private_bytes_cannot_be_relocated(source: Path) -> None:
    (source / "README.md").write_text("new unreviewed bytes")
    commit(source)
    with pytest.raises(export.ExportError, match="Reviewed content changed"):
        export.prepare(source)
    data = b"Retained private evidence"
    approve(source, {"innocent.txt": data}, inventory=[{
        "path": "tests/fixtures/private.md", "sha256": hashlib.sha256(data).hexdigest(),
        "publication": "private",
    }])
    with pytest.raises(export.ExportError, match="Unapproved fixture bytes"):
        export.prepare(source)


def test_unresolved_public_fixture_gate_cannot_be_overridden_by_cli(source: Path) -> None:
    manifest = source / export.MANIFEST
    data = json.loads(manifest.read_text())
    data["public_fixture_gate"] = "unresolved"
    manifest.write_text(json.dumps(data))
    commit(source)
    with pytest.raises(export.ExportError, match="unresolved"):
        export.prepare(source)


@pytest.mark.parametrize("returncode", [0, 1])
def test_scanner_requires_full_redaction_and_external_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "README.md").write_text("Authored harmless exporter test.\n")
    monkeypatch.setattr(export.shutil, "which", lambda name: "/fake/gitleaks")

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        assert "--redact=100" in args and "--ignore-gitleaks-allow" in args
        report = Path(args[args.index("--report-path") + 1])
        assert not report.is_relative_to(stage)
        report.write_text("[]")
        return subprocess.CompletedProcess(args, returncode, b"", b"")

    monkeypatch.setattr(export.subprocess, "run", run)
    if returncode:
        with pytest.raises(export.ExportError, match="redacted report"):
            export._scan_stage(stage)
    else:
        assert json.loads(export._scan_stage(stage).read_text()) == []


@pytest.mark.parametrize("name", ["../escape", "/absolute", "cmos/private.txt",
                                  "rbook/db.txt", ".env.local", "nested/state.sqlite",
                                  ".git/config", "AGENTS.md"])
def test_denied_or_unsafe_manifest_paths_fail_before_archive_selection(name: str) -> None:
    with pytest.raises(export.ExportError):
        export._path(name)


def test_expected_head_is_mandatory_and_stale_heads_are_refused(source: Path, scan: list) -> None:
    out = source.parent / "public"
    prepared = export.prepare(source)
    head = export.sync(prepared, source, out)["head"]
    for expected in (None, "0" * 40):
        with pytest.raises(export.ExportError):
            export.sync(prepared, source, out, expected)
    assert git(out, "rev-parse", "HEAD") == head


def test_known_tracked_file_hardlink_cannot_modify_an_external_file(
    source: Path, scan: list,
) -> None:
    out = source.parent / "public"
    prepared = export.prepare(source)
    head = export.sync(prepared, source, out)["head"]
    external = source.parent / "precious.txt"
    external.hardlink_to(out / "README.md")
    with pytest.raises(export.ExportError, match="linked"):
        export.sync(prepared, source, out, head)
    assert external.read_bytes() == b"Public authored example\n"


def test_check_only_scans_without_creating_destination(source: Path, scan: list) -> None:
    out = source.parent / "public"
    export.sync(export.prepare(source), source, out, check=True)
    assert scan and not out.exists()


@pytest.mark.parametrize("change", ["filter", "unexpected_index"])
def test_index_must_match_scanned_bytes_and_paths_before_commit(
    source: Path, scan: list, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    out = source.parent / "public"
    head = export.sync(export.prepare(source), source, out)["head"]
    approve(source, {"README.md": b"updated public text\n"})
    original_git = export._git
    if change == "filter":
        # A local clean filter can change the committed blob without changing
        # the working file. Receipt hashes must describe the actual Git tree.
        (out / ".git/info/attributes").write_text("README.md filter=transform\n")
        git(out, "config", "filter.transform.clean", "sed s/updated/transformed/")
    else:
        def inject(root: Path, *args: str) -> bytes:
            result = original_git(root, *args)
            if root == out and args[0] == "add":
                blob = original_git(out, "rev-parse", "HEAD:README.md").decode().strip()
                original_git(out, "update-index", "--add", "--cacheinfo",
                             f"100644,{blob},unexpected.txt")
            return result
        monkeypatch.setattr(export, "_git", inject)
    with pytest.raises(export.ExportError, match="Incomplete export"):
        export.sync(export.prepare(source), source, out, head)
    assert git(out, "rev-parse", "HEAD") == head
    assert git(out, "status", "--porcelain")


@pytest.mark.parametrize("mutation", ["none", "changed_byte", "new_location", "new_rule"])
def test_intentional_scan_controls_require_exact_file_hash_rule_and_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    stage = tmp_path / "stage"
    name, rule, start, end = next(iter(export.SCAN_CONTROLS))
    file = stage / name
    file.parent.mkdir(parents=True)
    file.write_bytes((Path(__file__).resolve().parents[2] / name).read_bytes())
    if mutation == "changed_byte":
        file.write_bytes(file.read_bytes() + b"\n")
    monkeypatch.setattr(export.shutil, "which", lambda name: "/fake/gitleaks")

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        report = Path(args[args.index("--report-path") + 1])
        report.write_text(json.dumps([{
            "File": str(file), "RuleID": rule if mutation != "new_rule" else "unexpected",
            "StartLine": start + (mutation == "new_location"), "EndLine": end,
        }]))
        return subprocess.CompletedProcess(args, 1, b"", b"")

    monkeypatch.setattr(export.subprocess, "run", run)
    if mutation == "none":
        report = export._scan_stage(stage)
        disposition = json.loads((report.parent / "disposition.json").read_text())
        assert disposition["unexpected_findings"] == 0
        assert len(disposition["intentional_controls"]) == 1
    else:
        with pytest.raises(export.ExportError, match="redacted report"):
            export._scan_stage(stage)
