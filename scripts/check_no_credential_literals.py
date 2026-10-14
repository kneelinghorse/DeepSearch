#!/usr/bin/env python3
"""Reject tracked credential literals without ever displaying their values."""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_ASSIGNMENT_RE = re.compile(
    r"""
    ^[ \t]*
    (?:(?:[-*+>][ \t]+)|(?:\#[ \t]*))?
    `?
    (?:export[ \t]+)?
    ["']?
    (?P<key>[A-Z][A-Z0-9_]*)
    ["']?
    [ \t]*(?P<separator>[:=])[ \t]*
    (?P<value>.*)
    $
    """,
    re.VERBOSE,
)
_SENSITIVE_KEY_SUFFIXES = (
    "PASSWORD",
    "PASSWORD_HASH",
    "PASSCODE",
    "SECRET",
    "SECRET_KEY",
    "CLIENT_SECRET",
    "WEBHOOK_SECRET",
    "TOKEN",
    "API_KEY",
    "PRIVATE_KEY",
    "SECRET_ACCESS_KEY",
    "SIGNING_KEY",
    "ENCRYPTION_KEY",
)
_COLON_ASSIGNMENT_SUFFIXES = {".json", ".md", ".markdown", ".toml", ".yaml", ".yml"}
_ALLOW_DIRECTIVE_RE = re.compile(r"#\s*secret-scan:\s*allow\s+--\s*\S", re.IGNORECASE)
_SIMPLE_REFERENCE_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
_BRACED_REFERENCE_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_GITHUB_REFERENCE_RE = re.compile(
    r"\$\{\{\s*(?:env|secrets|vars)\.[A-Za-z_][A-Za-z0-9_]*\s*\}\}"
)
_SHELL_DEFAULT_RE = re.compile(
    r"\$\{[A-Za-z_][A-Za-z0-9_]*(?P<operator>:-|-|:\+|\+|:\?|\?)(?P<default>.*)\}"
)
_RUNTIME_LOOKUP_RE = re.compile(
    r"(?:os\.)?(?:getenv|environ(?:\.get)?)[ \t]*(?:\(|\[)", re.IGNORECASE
)
_PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"a-(?:very-)?strong-random-(?:key|password|secret)|"
    r"change-?me|"
    r"generate-(?:with-)?openssl-rand-(?:base64|hex)-[0-9]+|"
    r"replace[-_].*|"
    r"(?:dev|development|dummy|example|fake|placeholder|sample|test|testing|your)"
    r"(?:[-_.].*)?"
    r")",
    re.IGNORECASE,
)
_PROVIDER_PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"[A-Za-z0-9]+-(?:(?:dummy|example|fake|placeholder|test|your)[-_.].*|\.{3}|…)|"
    r"tvly-x+"
    r")",
    re.IGNORECASE,
)
_JSON_LITERAL_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    (?P<key_quote>["'])?
    (?P<key>password|passwd|pass|secret|api_key|apiKey|token|access_token|refresh_token)
    (?(key_quote)(?P=key_quote))
    [ \t]*:[ \t]*
    (?P<value_quote>["'])
    (?P<value>[^"'\r\n]{3,})
    (?P=value_quote)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_PROSE_CREDENTIAL_RE = re.compile(
    r"""
    \b(?:shared|use|using)\b
    [^\r\n]{0,40}?
    \b(?:credentials?|creds?|login|account)\b
    \s*\(?\s*
    (?P<username>[A-Za-z0-9._@-]{3,})
    \s*/\s*
    (?P<value>[^\s)'"`]{4,})
    """,
    re.IGNORECASE | re.VERBOSE,
)
_USERINFO_URL_RE = re.compile(
    r"""
    \b[a-z][a-z0-9+.-]*://
    [^:/\s'"@]+:
    (?P<value>[^@\s'"]{4,})@
    (?P<host>[^/\s:'"]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SAFE_URL_PASSWORDS = {"password", "pass", "postgres", "tracelab"}
_SAFE_URL_HOSTS = {"localhost", "127.0.0.1", "host", "postgres", "db", "example.com"}
_JSON_PLACEHOLDERS = {"testpass", "password123", "hunter2"}
_PROVIDER_PATTERNS = (
    (
        "PROVIDER_SK_TOKEN",
        re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),
    ),
    ("PROVIDER_TAVILY_TOKEN", re.compile(r"(?<![A-Za-z0-9_-])tvly-[A-Za-z0-9_-]{10,}")),
    (
        "PROVIDER_JWT",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
            r"\.[A-Za-z0-9_-]{8,}"
        ),
    ),
    (
        "PROVIDER_GITHUB_TOKEN",
        re.compile(
            r"\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|"
            r"github_pat_[A-Za-z0-9_]{20,})"
        ),
    ),
    ("PROVIDER_AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("PROVIDER_SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("PROVIDER_GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    (
        "BEARER_TOKEN",
        re.compile(r"\bBearer[ \t]+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
    ),
    (
        "PRIVATE_KEY_BLOCK",
        re.compile(r"-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}"),
    ),
    (
        "RAILWAY_PROXY_HOST",
        re.compile(r"\b(?:[a-z0-9-]+\.)*proxy[.]rlwy[.]net\b", re.IGNORECASE),
    ),
)
_PROHIBITED_CARRIER_PATTERNS = ("*.sqlite*", "*.db", "*.bak", "*-wal", "*-shm")
_MAX_TEXT_BYTES = 5 * 1024 * 1024
_LARGE_RESEARCH_ALLOWLIST: frozenset[Path] = frozenset()
_KNOWN_FINDING_EXEMPTIONS = {
    ("cmos/missions/B1.4_research-node-web-search.yaml", 102, "JSON_API_KEY"): (
        "historical response-schema placeholder"
    ),
    ("cmos/missions/B5.1_automated-post-research-upload.yaml", 35, "JSON_ACCESS_TOKEN"): (
        "historical response-schema ellipsis"
    ),
    ("cmos/planning/Answers-for-deepsearch.md", 320, "JSON_PASSWORD"): (
        "historical command placeholder"
    ),
    ("cmos/planning/Answers-for-deepsearch.md", 325, "JSON_ACCESS_TOKEN"): (
        "historical response-schema ellipsis"
    ),
    ("cmos/planning/Answers-from-tracelab-sprint12.md", 53, "JSON_ACCESS_TOKEN"): (
        "historical response-schema ellipsis"
    ),
    ("cmos/reports/sprint_87_cat_acceptance_contract.json", 90, "JSON_PASS"): (
        "CAT pass/fail prose, not an authentication field"
    ),
    ("cmos/research/sprint_61_spike/synthesis_prompt.md", 3106, "RAILWAY_TOKEN"): (
        "quoted third-party shell-source prose"
    ),
    (
        "cmos/reports/sprint_93_m08_production_observation.json",
        29,
        "DEEPSEEK_API_KEY",
    ): "boolean credential-presence telemetry; no credential value is stored",
    (
        "cmos/reports/sprint_93_m08_production_observation.json",
        30,
        "OPENAI_API_KEY",
    ): "boolean credential-presence telemetry; no credential value is stored",
    (
        "cmos/reports/sprint_93_m08_production_observation.json",
        31,
        "LINKUP_API_KEY",
    ): "boolean credential-presence telemetry; no credential value is stored",
    ("tests/integration/test_auto_upload.py", 205, "JSON_ACCESS_TOKEN"): (
        "synthetic upload response fixture"
    ),
    ("tests/integration/test_auto_upload.py", 212, "JSON_ACCESS_TOKEN"): (
        "synthetic refresh response fixture"
    ),
    ("tests/integration/test_auto_upload.py", 243, "JSON_ACCESS_TOKEN"): (
        "synthetic upload response fixture"
    ),
    ("tests/integration/test_auto_upload.py", 253, "JSON_ACCESS_TOKEN"): (
        "synthetic refresh response fixture"
    ),
    ("tests/integration/test_auto_upload.py", 407, "JSON_ACCESS_TOKEN"): (
        "synthetic refresh response fixture"
    ),
    ("tests/unit/test_database_client.py", 57, "URL_USERINFO_PASSWORD"): (
        "synthetic database DSN fixture"
    ),
    ("tests/unit/test_database_client.py", 64, "URL_USERINFO_PASSWORD"): (
        "synthetic database DSN fixture"
    ),
    ("tests/unit/test_s91_m05_log_transport.py", 233, "JSON_ACCESS_TOKEN"): (
        "synthetic auth response fixture"
    ),
    ("tests/unit/test_s91_m05_log_transport.py", 246, "JSON_PASSWORD"): (
        "synthetic echoed-body sanitization fixture"
    ),
}


@dataclass(frozen=True)
class Finding:
    """A value-safe location and rule identifier for one credential finding."""

    path: Path
    line_number: int
    key: str


def _is_sensitive_key(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _SENSITIVE_KEY_SUFFIXES)


def _allows_colon_assignment(path: Path) -> bool:
    return path.suffix.lower() in _COLON_ASSIGNMENT_SUFFIXES


def _scalar_value(raw_value: str) -> str:
    value = re.split(r"\s+#", raw_value.strip(), maxsplit=1)[0].strip()
    value = value.removesuffix("`").strip()

    if len(value) >= 2 and value[0] in {'"', "'"}:
        closing_index = value.rfind(value[0])
        if closing_index > 0:
            value = value[1:closing_index].strip()

    return value


def _is_safe_shell_default(value: str) -> bool:
    match = _SHELL_DEFAULT_RE.fullmatch(value)
    if match is None:
        return False

    default = match.group("default").strip()
    return not default or _is_safe_value(default)


def _is_safe_value(value: str) -> bool:
    value = value.strip()
    if not value:
        return True

    if (
        _SIMPLE_REFERENCE_RE.fullmatch(value)
        or _BRACED_REFERENCE_RE.fullmatch(value)
        or _GITHUB_REFERENCE_RE.fullmatch(value)
        or _is_safe_shell_default(value)
        or (value.startswith("$(") and value.endswith(")"))
        or _RUNTIME_LOOKUP_RE.match(value)
    ):
        return True

    normalized = value.casefold()
    if normalized in {"none", "null", "postgres", "unset", "???"}:
        return True
    if normalized.startswith("[redacted") and normalized.endswith("]"):
        return True
    if (value.startswith("<") and value.endswith(">")) or value in {"...", "…"}:
        return True
    return bool(_PLACEHOLDER_RE.fullmatch(value) or _PROVIDER_PLACEHOLDER_RE.fullmatch(value))


def _is_safe_json_value(value: str) -> bool:
    normalized = value.strip().casefold()
    return (
        _is_safe_value(value)
        or normalized in _JSON_PLACEHOLDERS
        or normalized.startswith("correct-horse")
    )


def _provider_match_is_safe(rule: str, value: str) -> bool:
    if _is_safe_value(value):
        return True
    if rule == "PROVIDER_TAVILY_TOKEN":
        payload = value.removeprefix("tvly-").casefold()
        return bool(payload) and set(payload) == {"x"}
    return False


def _provider_findings(line: str, path: Path, line_number: int) -> list[Finding]:
    findings: list[Finding] = []
    for rule, pattern in _PROVIDER_PATTERNS:
        for match in pattern.finditer(line):
            if not _provider_match_is_safe(rule, match.group(0)):
                findings.append(Finding(path=path, line_number=line_number, key=rule))
    return findings


def find_provider_literals(text: str, path: Path) -> list[Finding]:
    """Return provider-shaped literals while retaining only rule locations."""
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _ALLOW_DIRECTIVE_RE.search(line):
            continue
        findings.extend(_provider_findings(line, path, line_number))
    return findings


def find_credential_literals(text: str, path: Path) -> list[Finding]:
    """Return unsafe credential literals without retaining their values."""
    findings: list[Finding] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if _ALLOW_DIRECTIVE_RE.search(line):
            continue

        assignment = _ASSIGNMENT_RE.match(line)
        if assignment is not None:
            key = assignment.group("key")
            separator = assignment.group("separator")
            if (
                _is_sensitive_key(key)
                and (separator != ":" or _allows_colon_assignment(path))
                and not _is_safe_value(_scalar_value(assignment.group("value")))
            ):
                findings.append(Finding(path=path, line_number=line_number, key=key))

        normalized_line = line.replace(r'\"', '"').replace(r"\'", "'")
        for match in _JSON_LITERAL_RE.finditer(normalized_line):
            if not _is_safe_json_value(match.group("value")):
                findings.append(
                    Finding(
                        path=path,
                        line_number=line_number,
                        key=f"JSON_{match.group('key').upper()}",
                    )
                )

        prose = _PROSE_CREDENTIAL_RE.search(line)
        if prose is not None and not _is_safe_value(prose.group("value")):
            findings.append(
                Finding(path=path, line_number=line_number, key="PROSE_CREDENTIAL_PAIR")
            )

        for match in _USERINFO_URL_RE.finditer(line):
            password = match.group("value")
            host = match.group("host").casefold()
            if (
                password.casefold() not in _SAFE_URL_PASSWORDS
                and host not in _SAFE_URL_HOSTS
                and not _is_safe_value(password)
            ):
                findings.append(
                    Finding(path=path, line_number=line_number, key="URL_USERINFO_PASSWORD")
                )

        findings.extend(_provider_findings(line, path, line_number))

    return findings


def is_prohibited_carrier(path: Path) -> bool:
    """Return whether a tracked path is forbidden even when it is binary."""
    name = path.name.casefold()
    if name.startswith(".env"):
        return name != ".env.example"
    return any(fnmatch.fnmatch(name, pattern) for pattern in _PROHIBITED_CARRIER_PATTERNS)


def _read_text(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None

    content = path.read_bytes()
    if b"\0" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _contains_nul(path: Path) -> bool:
    with path.open("rb") as stream:
        return any(b"\0" in chunk for chunk in iter(lambda: stream.read(64 * 1024), b""))


def tracked_paths() -> list[Path]:
    """List paths currently present in Git's index."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git executable not found")
    result = subprocess.run(  # noqa: S603 - fixed executable and arguments
        [git, "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        Path(item.decode("utf-8", errors="surrogateescape"))
        for item in result.stdout.split(b"\0")
        if item
    ]


def scan_paths(
    paths: Sequence[Path],
    *,
    enforce_carrier_guard: bool = False,
) -> list[Finding]:
    """Scan paths, optionally rejecting tracked credential-bearing file classes."""
    findings: list[Finding] = []
    for path in dict.fromkeys(paths):
        if ".venv" in path.parts:
            continue
        if enforce_carrier_guard and is_prohibited_carrier(path):
            findings.append(Finding(path=path, line_number=0, key="PROHIBITED_CARRIER"))
            continue

        if not path.is_file() or path.is_symlink():
            continue
        if path.stat().st_size > _MAX_TEXT_BYTES:
            if _contains_nul(path):
                continue
            if path not in _LARGE_RESEARCH_ALLOWLIST:
                findings.append(
                    Finding(path=path, line_number=0, key="LARGE_FILE_REQUIRES_REVIEW")
                )
                continue

        text = _read_text(path)
        if text is None:
            continue
        if path.stat().st_size > _MAX_TEXT_BYTES and path in _LARGE_RESEARCH_ALLOWLIST:
            findings.extend(find_provider_literals(text, path))
        else:
            findings.extend(find_credential_literals(text, path))
    if not enforce_carrier_guard:
        return findings
    return [
        finding
        for finding in findings
        if (finding.path.as_posix(), finding.line_number, finding.key)
        not in _KNOWN_FINDING_EXEMPTIONS
    ]


def format_finding(finding: Finding) -> str:
    """Format a diagnostic containing location and rule, never credential data."""
    return f"{finding.path}:{finding.line_number}:{finding.key}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reject hard-coded credential literals without displaying their values."
    )
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="scan every file tracked by Git instead of explicit paths",
    )
    parser.add_argument("paths", nargs="*", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.tracked and args.paths:
        parser.error("--tracked cannot be combined with explicit paths")

    try:
        paths = tracked_paths() if args.tracked else args.paths
    except (OSError, subprocess.CalledProcessError):
        print("Credential scan failed: unable to list Git-tracked files.", file=sys.stderr)
        return 2

    findings = scan_paths(paths, enforce_carrier_guard=args.tracked)
    for finding in findings:
        print(format_finding(finding), file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
