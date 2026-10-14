#!/usr/bin/env python3
"""Stage, scan and sync an explicitly reviewed committed public snapshot.

No network operations. The repository manifest remains blocked until publication
review is approved. A sync requires the managed destination's exact expected HEAD.
An interrupted write is left recoverable and dirty; this tool never resets it.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "scripts/public_export_manifest.json"
RECEIPT = ".deepsearch-export.json"
FORMAT = "deepsearch-public-v1"
MAX_FILE_BYTES = 1_000_000
LOCAL_PATH = re.compile(r"(?<![\w.-])/(?:Users|home)/[A-Za-z0-9._-]+/")
DENIED = ("*.sqlite*", "*.db", "*.db-*", "*.bak", "*.pem", "*.key", "*.p12", "*.zip")
# Reviewed deliberately invalid controls. A changed byte anywhere in the file,
# another rule, or another location requires a new review; no directory exemption.
SCAN_CONTROLS = {
    ("tests/integration/test_google_search_integration.py", "generic-api-key", 153, 153):
        ("4137bc8914e95b78a43d58cb07e0846ad01baea54dfb70016d130276b66d2e0a",
         "Deliberately invalid authentication input in an opt-in error test"),
    ("tests/unit/test_no_tracked_credential_literals.py", "aws-access-token", 185, 185):
        ("b6b4da6a941721b29ee5366f9a44111b83122c91baed41b51c645320fdb61942",
         "Alphabet-sequence negative control for provider-prefix detection"),
    ("tests/unit/test_no_tracked_credential_literals.py", "private-key", 201, 225):
        ("b6b4da6a941721b29ee5366f9a44111b83122c91baed41b51c645320fdb61942",
         "PEM header-only negative control; no private-key payload"),
}


class ExportError(ValueError):
    """A value-safe refusal; no source contents belong in this message."""


@dataclass(frozen=True)
class ExportFile:
    path: str
    mode: int
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class PreparedExport:
    source_commit: str
    source_tree: str
    manifest_sha256: str
    files: tuple[ExportFile, ...]
    receipt: bytes


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "--literal-pathspecs", "-c", "core.hooksPath=/dev/null",
         "-c", "commit.gpgsign=false",
         "-c", "tar.umask=0022", *args],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise ExportError(f"Git operation failed: {args[0]}")
    return result.stdout


def _path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or str(path) != name
            or any(p in {"..", "."} for p in path.parts)
            or "\\" in name or any(ord(c) < 32 for c in name)):
        raise ExportError("Unsafe manifest/archive path")
    if any(p.casefold() in {".git", "cmos", "rbook", "__pycache__"} for p in path.parts):
        raise ExportError(f"Denied path: {name}")
    if path.name.casefold() in {"agents.md", "claude.md"}:
        raise ExportError(f"Denied private rules: {name}")
    if any(p.startswith(".env") for p in path.parts) and name != ".env.example":
        raise ExportError(f"Denied environment carrier: {name}")
    if any(fnmatch.fnmatchcase(path.name.lower(), pattern) for pattern in DENIED):
        raise ExportError(f"Denied carrier: {name}")
    if name.casefold() in {MANIFEST, RECEIPT, "tests/fixtures/publication_inventory.json"}:
        raise ExportError(f"Reserved path: {name}")
    return path


def _json(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ExportError(f"Invalid UTF-8 JSON: {label}") from exc


def _validate_file(file: ExportFile) -> None:
    _path(file.path)
    if file.mode not in {0o644, 0o755}:
        raise ExportError(f"Unsupported mode: {file.path}")
    if len(file.data) > MAX_FILE_BYTES:
        raise ExportError(f"Oversized file: {file.path}")
    if b"\0" in file.data:
        raise ExportError(f"NUL carrier: {file.path}")
    try:
        text = file.data.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ExportError(f"Invalid UTF-8: {file.path}") from exc
    if LOCAL_PATH.search(text):
        raise ExportError(f"Local absolute path: {file.path}")


def _archive_files(raw: bytes, selected: set[str]) -> tuple[ExportFile, ...]:
    files: dict[str, ExportFile] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if member.name not in selected:
                continue
            _path(member.name)
            if not member.isfile() or member.islnk() or member.issym():
                raise ExportError(f"Unsupported archive entry: {member.name}")
            if member.name in files:
                raise ExportError(f"Duplicate archive entry: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ExportError(f"Missing archive bytes: {member.name}")
            file = ExportFile(member.name, member.mode, stream.read())
            _validate_file(file)
            files[file.path] = file
    if files.keys() != selected:
        raise ExportError("Manifest selects missing or unsupported committed entries")
    return tuple(files[name] for name in sorted(files))


def prepare(source: Path, rev: str = "HEAD") -> PreparedExport:
    """Read only one immutable commit, its reviewed manifest and fixture registry."""
    commit = _git(source, "rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()
    tree = _git(source, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    raw_manifest = _git(source, "show", f"{commit}:{MANIFEST}")
    manifest = _json(raw_manifest, MANIFEST)
    if (manifest.get("format") != FORMAT or manifest.get("status") != "approved"
            or manifest.get("public_fixture_gate") != "approved"):
        raise ExportError("Publication manifest/fixture review is unresolved")
    inventory_path = "tests/fixtures/publication_inventory.json"
    inventory = _json(_git(source, "show", f"{commit}:{inventory_path}"), inventory_path)
    fixtures = {row["path"]: row for row in inventory["fixtures"]}
    private_hashes = {row["sha256"] for row in fixtures.values()
                      if row["publication"] != "approved"}
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ExportError("Empty or malformed reviewed manifest")
    by_path = {}
    for row in entries:
        name = row["path"]
        _path(name)
        if (name in by_path or row.get("publication") != "approved"
                or not row.get("basis") or row.get("license") in {None, "", "unresolved"}
                or row.get("mode") not in {0o644, 0o755}
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))):
            raise ExportError(f"Unreviewed or duplicate entry: {name}")
        by_path[name] = row
    folded = {name.casefold() for name in by_path}
    if len(folded) != len(by_path) or any(
        str(parent) in folded for name in folded for parent in PurePosixPath(name).parents
    ):
        raise ExportError("Manifest paths collide on a case-insensitive filesystem")
    files = _archive_files(_git(source, "archive", "--format=tar", commit), set(by_path))
    for file in files:
        if (file.sha256 != by_path[file.path]["sha256"]
                or file.mode != by_path[file.path]["mode"]):
            raise ExportError(f"Reviewed content changed: {file.path}")
        if file.sha256 in private_hashes:
            raise ExportError(f"Unapproved fixture bytes: {file.path}")
        if file.path in fixtures and fixtures[file.path]["publication"] != "approved":
            raise ExportError(f"Unapproved fixture: {file.path}")
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    receipt = json.dumps({
        "format": FORMAT, "source_commit": commit, "source_tree": tree,
        "manifest_sha256": manifest_hash, "reviewed_manifest": manifest,
        "files": [{"path": f.path, "mode": f.mode, "sha256": f.sha256} for f in files],
    }, sort_keys=True, indent=2).encode() + b"\n"
    return PreparedExport(commit, tree, manifest_hash, files, receipt)


def _canonical_destination(source: Path, out: Path) -> Path:
    if ".." in out.parts:
        raise ExportError("Destination aliases are not accepted")
    out = Path(os.path.abspath(out.expanduser()))
    if any(p.is_symlink() for p in (out, *out.parents)) or out.resolve() != out:
        raise ExportError("Destination symlink aliases are not accepted")
    source = source.resolve()
    if (out in {Path(out.anchor), Path.home().resolve(), source}
            or out in source.parents or source in out.parents):
        raise ExportError("Destination is root, home or overlaps the source")
    if not out.parent.is_dir():
        raise ExportError("Destination parent must already exist")
    return out


def _tree_entries(out: Path) -> set[str]:
    entries = set()
    for root, dirs, files in os.walk(out, followlinks=False):
        if Path(root) == out and ".git" in dirs:
            dirs.remove(".git")
        for name in dirs + files:
            path = Path(root) / name
            if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
                raise ExportError("Destination contains a linked filesystem entry")
            if not path.is_dir() and not path.is_file():
                raise ExportError("Destination contains an unsupported filesystem entry")
            entries.add(path.relative_to(out).as_posix())
    return entries


def _with_parents(paths: set[str]) -> set[str]:
    return paths | {str(p) for name in paths for p in PurePosixPath(name).parents if str(p) != "."}


def _verify_bytes(out: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = out / row["path"]
        if (not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
                or path.stat().st_mode & 0o777 != row["mode"]):
            raise ExportError("Managed destination bytes or modes changed")


def inspect_destination(source: Path, out: Path, expected_head: str | None) -> dict[str, Any]:
    """Refuse every destination except empty new or explicitly managed and clean."""
    out = _canonical_destination(source, out)
    if not out.exists() or (out.is_dir() and not any(out.iterdir())):
        if expected_head is not None:
            raise ExportError("Expected managed HEAD, but destination is empty")
        probe = subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=out.parent,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if probe.returncode == 0:
            raise ExportError("New destination is inside another repository")
        return {"head": None, "files": [], "receipt_sha256": None}
    if not out.is_dir() or not (out / ".git").is_dir() or (out / ".git").is_symlink():
        raise ExportError("Destination is not an empty or managed repository")
    if not expected_head or not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise ExportError("Managed sync requires the exact expected HEAD")
    if Path(_git(out, "rev-parse", "--show-toplevel").decode().strip()).resolve() != out:
        raise ExportError("Destination is not a standalone repository root")
    head = _git(out, "rev-parse", "HEAD").decode().strip()
    if head != expected_head:
        raise ExportError("Managed destination HEAD changed")
    if _git(out, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"):
        raise ExportError("Destination is dirty, untracked or ignored data is present")
    raw = _git(out, "show", f"HEAD:{RECEIPT}")
    receipt = _json(raw, RECEIPT)
    if receipt.get("format") != FORMAT:
        raise ExportError("Unknown destination management receipt")
    rows = receipt.get("files", [])
    paths = set()
    for row in rows:
        _path(row["path"])
        if row["path"] in paths:
            raise ExportError("Duplicate managed path")
        paths.add(row["path"])
    tracked = set(_git(out, "ls-files", "-z").decode().rstrip("\0").split("\0"))
    if tracked != paths | {RECEIPT} or _tree_entries(out) != _with_parents(tracked):
        raise ExportError("Destination includes unknown files or directories")
    _verify_bytes(out, rows)
    if (out / RECEIPT).read_bytes() != raw:
        raise ExportError("Management receipt changed")
    return {"head": head, "files": rows, "receipt_sha256": hashlib.sha256(raw).hexdigest()}


def _scan_stage(stage: Path) -> Path:
    """Use redacted external scan output; never print possible secret values."""
    executable = shutil.which("gitleaks")
    if executable is None:
        raise ExportError("gitleaks is required before an export can be written")
    audit = Path(tempfile.mkdtemp(prefix="deepsearch-export-scan-"))
    config = audit / "config.toml"
    config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
    report = audit / "gitleaks.json"
    result = subprocess.run([
        executable, "dir", str(stage), "--config", str(config), "--redact=100",
        "--ignore-gitleaks-allow", "--gitleaks-ignore-path", str(audit / "no-exemptions"),
        "--report-format", "json", "--report-path", str(report),
        "--no-banner", "--log-level", "error",
    ], cwd=audit, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode == 1 and report.is_file():
        findings = _json(report.read_bytes(), "redacted scanner output")
        accepted = []
        for finding in findings if isinstance(findings, list) else []:
            path = Path(finding.get("File", ""))
            if not path.is_absolute():
                path = stage / path
            if not path.is_relative_to(stage):
                break
            name = path.relative_to(stage).as_posix()
            key = (name, finding.get("RuleID"), finding.get("StartLine"), finding.get("EndLine"))
            control = SCAN_CONTROLS.get(key)
            if (control is None or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != control[0]):
                break
            accepted.append({"path": name, "rule": key[1], "start_line": key[2],
                             "end_line": key[3], "sha256": control[0], "reason": control[1]})
        else:
            if accepted:
                (audit / "disposition.json").write_text(json.dumps({
                    "unexpected_findings": 0, "intentional_controls": accepted,
                }, indent=2) + "\n", encoding="utf-8")
                return report
    if result.returncode:
        raise ExportError(f"Staged secret scan failed; redacted report: {report}")
    return report


def _write_file(out: Path, file: ExportFile, previous: dict[str, Any] | None) -> None:
    path = out / file.path
    if any(parent.is_symlink() for parent in path.parents
           if parent.is_relative_to(out) and parent != out):
        raise ExportError("A destination parent became a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        _verify_bytes(out, [previous])
    # New entries are never permitted to overwrite an unknown file.
    with path.open("wb" if previous is not None else "xb") as stream:
        stream.write(file.data)
    path.chmod(file.mode)


def _verify_index(out: Path, files: tuple[ExportFile, ...]) -> str:
    """Pin an exact tree so filters or unrelated index changes cannot be published."""
    tree = _git(out, "write-tree").decode().strip()
    entries = {}
    for entry in _git(out, "ls-tree", "-rz", tree).split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, kind, blob = metadata.split()
            entries[name.decode()] = (mode, kind, blob.decode())
    if set(entries) != {file.path for file in files}:
        raise ExportError("Indexed paths differ from the scanned export")
    for file in files:
        mode, kind, blob = entries[file.path]
        if (kind != b"blob" or int(mode, 8) != 0o100000 | file.mode
                or _git(out, "cat-file", "blob", blob) != file.data):
            raise ExportError("Indexed bytes or modes differ from the scanned export")
    return tree


def sync(prepared: PreparedExport, source: Path, out: Path,
         expected_head: str | None = None, *, check: bool = False) -> dict[str, Any]:
    """Scan staging, recheck ownership, then write only named managed paths."""
    out = _canonical_destination(source, out)
    before = inspect_destination(source, out, expected_head)
    with tempfile.TemporaryDirectory(prefix="deepsearch-export-stage-") as temp:
        stage = Path(temp)
        for file in prepared.files:
            _write_file(stage, file, None)
        (stage / RECEIPT).write_bytes(prepared.receipt)
        scan_report = _scan_stage(stage)
        if check:
            return {"source_commit": prepared.source_commit, "scan_report": str(scan_report)}
        if inspect_destination(source, out, expected_head) != before:
            raise ExportError("Destination changed during staging; nothing written")
        old = {row["path"]: row for row in before["files"]}
        new = {file.path: file for file in prepared.files}
        old_receipt = None
        if before["head"] is not None:
            old_receipt = {"path": RECEIPT, "sha256": before["receipt_sha256"], "mode": 0o644}
        try:
            out.mkdir(exist_ok=True)
            for name in sorted(old.keys() - new.keys()):
                _verify_bytes(out, [old[name]])
                (out / name).unlink()
            for name in sorted(_with_parents(set(old)) - set(old), key=len, reverse=True):
                directory = out / name
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            for file in prepared.files:
                _write_file(out, file, old.get(file.path))
            _write_file(out, ExportFile(RECEIPT, 0o644, prepared.receipt), old_receipt)
            wanted = [{"path": f.path, "sha256": f.sha256, "mode": f.mode}
                      for f in prepared.files]
            wanted.append({"path": RECEIPT, "sha256": hashlib.sha256(prepared.receipt).hexdigest(),
                           "mode": 0o644})
            if _tree_entries(out) != _with_parents({row["path"] for row in wanted}):
                raise ExportError("Unknown data appeared during the write")
            _verify_bytes(out, wanted)
            if before["head"] is None:
                _git(out, "init", "-q", "-b", "main")
            elif _git(out, "rev-parse", "HEAD").decode().strip() != before["head"]:
                raise ExportError("Destination HEAD changed during the write")
            _git(out, "add", "--all", "--", *sorted(set(old) | set(new) | {RECEIPT}))
            files_with_receipt = (*prepared.files, ExportFile(RECEIPT, 0o644, prepared.receipt))
            tree = _verify_index(out, files_with_receipt)
            prior_tree = (_git(out, "rev-parse", "HEAD^{tree}").decode().strip()
                          if before["head"] else None)
            if tree != prior_tree:
                parents = ["-p", before["head"]] if before["head"] else []
                head = _git(out, "commit-tree", tree, *parents,
                     "-m", f"Sync DeepSearch {prepared.source_commit[:12]}",
                     "-m", f"Source commit: {prepared.source_commit}\nSource tree: "
                     f"{prepared.source_tree}\nManifest SHA-256: "
                     f"{prepared.manifest_sha256}").decode().strip()
                # Compare-and-swap advances HEAD only after all writes and checks.
                # The commit references the verified tree, never a mutable index.
                _git(out, "update-ref", "HEAD", head, before["head"] or "0" * 40)
            head = _git(out, "rev-parse", "HEAD").decode().strip()
            inspect_destination(source, out, head)
        except (OSError, ExportError) as exc:
            raise ExportError(
                "Incomplete export; no automatic reset. Previous commit remains recoverable; "
                "inspect and recover only known generated changes."
            ) from exc
    return {"head": head, "source_commit": prepared.source_commit, "scan_report": str(scan_report)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rev", default="HEAD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-head", help="Exact current downstream commit for a sync")
    parser.add_argument("--check", action="store_true", help="Stage and scan without writing")
    args = parser.parse_args(argv)
    try:
        prepared = prepare(REPO_ROOT, args.rev)
        result = sync(prepared, REPO_ROOT, args.out, args.expected_head, check=args.check)
    except (ExportError, KeyError, TypeError, tarfile.TarError) as exc:
        print(f"Export refused: {exc if isinstance(exc, ExportError) else 'malformed input'}",
              file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
