"""Regression tests for value-safe tracked credential detection.

The guard is semantic because real low-entropy credentials and response-body
echoes evaded provider-prefix scanners. Tests cover each known carrier class
without preserving any exposed value or fingerprint.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import scripts.check_no_credential_literals as scanner
from scripts.check_no_credential_literals import (
    find_credential_literals,
    format_finding,
    is_prohibited_carrier,
    main,
    scan_paths,
    tracked_paths,
)


@pytest.mark.parametrize(
    ("path", "line"),
    [
        (Path("mission.md"), "- AUTH_PASSWORD=summer"),
        (Path("config.yaml"), "OPENAI_API_KEY: literal-with-spaces"),
        (Path("settings.json"), '"SECRET_KEY": "literal-value"'),
        (Path("service.env"), "AUTH_PASSWORD_HASH=short-value"),
        (Path("compose.yaml"), "AUTH_PASSWORD=${AUTH_PASSWORD:-winter}"),
    ],
)
def test_assignment_rule_rejects_literals_regardless_of_entropy(
    path: Path,
    line: str,
) -> None:
    """Credential semantics matter even when a literal has low entropy."""
    findings = find_credential_literals(line, path)

    assert len(findings) == 1
    assert findings[0].line_number == 1


@pytest.mark.parametrize(
    "line",
    [
        "AUTH_PASSWORD=",  # secret-scan: allow -- parser fixture
        "AUTH_PASSWORD=$AUTH_PASSWORD",  # secret-scan: allow -- parser fixture
        "AUTH_PASSWORD=${AUTH_PASSWORD}",  # secret-scan: allow -- parser fixture
        "AUTH_PASSWORD=${AUTH_PASSWORD:-placeholder}",  # secret-scan: allow -- fixture
        "AUTH_PASSWORD=${{ secrets.AUTH_PASSWORD }}",  # secret-scan: allow -- fixture
        "AUTH_PASSWORD=os.getenv('AUTH_PASSWORD')",
        "AUTH_PASSWORD=environ['AUTH_PASSWORD']",
        "AUTH_PASSWORD=<set-in-secret-manager>",  # secret-scan: allow -- fixture
        "AUTH_PASSWORD=[REDACTED -- credential rotated]",  # secret-scan: allow -- fixture
        "AUTH_PASSWORD=???",  # secret-scan: allow -- parser fixture
        "AUTH_PASSWORD=change-me",  # secret-scan: allow -- parser fixture
        "OPENAI_API_KEY=sk-your_openai_api_key_here",
        "TAVILY_API_KEY=tvly-xxxxxxxxxxxx",  # secret-scan: allow -- parser fixture
    ],
)
def test_assignment_rule_accepts_runtime_references_and_obvious_placeholders(line: str) -> None:
    """The guard must preserve deployable templates and explicit redactions."""
    assert find_credential_literals(line, Path("instructions.md")) == []


def test_colon_assignment_is_limited_to_data_and_document_formats() -> None:
    """Python annotations and prose labels must not become assignment findings."""
    line = "TRACELAB_TOKEN: Bearer access token for authentication"

    assert find_credential_literals(line, Path("types.py")) == []
    assert len(find_credential_literals(line, Path("deployment.md"))) == 1


@pytest.mark.parametrize(
    "line",
    [
        'payload = {"password": "opaque-value"}',  # secret-scan: allow -- synthetic rule fixture
        "payload = {'api_key': 'opaque-value'}",  # secret-scan: allow -- synthetic rule fixture
        (
            r'error = "{\"input\":{'
            r'\"token\":\"opaque-value\"}}"'  # secret-scan: allow -- fixture
        ),
    ],
)
def test_json_rule_rejects_plain_and_escaped_literals(line: str) -> None:
    """Response-body echoes and dict fixtures are credential carriers too."""
    findings = find_credential_literals(line, Path("artifact.json"))

    assert len(findings) == 1
    assert findings[0].key.startswith("JSON_")


@pytest.mark.parametrize(
    "value",
    [
        "[REDACTED]",
        "<runtime-secret>",
        "testpass",
        "password123",
        "hunter2",
        "correct-horse-placeholder",
    ],
)
def test_json_rule_accepts_deliberate_non_production_values(value: str) -> None:
    """Tests and redacted artifacts need unmistakable safe stand-ins."""
    line = f'payload = {{"password": "{value}"}}'  # secret-scan: allow -- fixture

    assert find_credential_literals(line, Path("fixture.py")) == []


def test_prose_rule_rejects_a_shared_user_password_pair() -> None:
    """Credentials written as user/password prose must not bypass assignments."""
    line = (
        "Use shared credentials operator/opaque-value"  # secret-scan: allow -- fixture
    )

    findings = find_credential_literals(line, Path("handoff.md"))

    assert [finding.key for finding in findings] == ["PROSE_CREDENTIAL_PAIR"]


@pytest.mark.parametrize(
    "line",
    [
        "Use shared credentials <username>/<password>",
        "Use credentials example-user/test-password",
        "The account is described without a credential pair",
    ],
)
def test_prose_rule_accepts_placeholders_and_non_pairs(line: str) -> None:
    """Documentation remains readable when it refers only to placeholders."""
    assert find_credential_literals(line, Path("handoff.md")) == []


def test_userinfo_url_rule_rejects_a_remote_embedded_password() -> None:
    """Remote connection URLs are secrets even without a credential key name."""
    line = (
        "postgresql://worker:opaque-value@"
        "database.invalid/app"  # secret-scan: allow -- synthetic rule fixture
    )

    findings = find_credential_literals(line, Path("notes.md"))

    assert [finding.key for finding in findings] == ["URL_USERINFO_PASSWORD"]


@pytest.mark.parametrize(
    "line",
    [
        "postgresql://user:password@host/app",
        "postgresql://user:dev-password@database.invalid/app",
        "postgresql://user:opaque-value@localhost/app",
    ],
)
def test_userinfo_url_rule_accepts_documented_development_defaults(line: str) -> None:
    """Local examples must remain usable without weakening remote URL checks."""
    assert find_credential_literals(line, Path("example.md")) == []


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        (
            "sk-proj-ABCDEFGHIJKLMNOPQRSTUV",  # secret-scan: allow -- synthetic fixture
            "PROVIDER_SK_TOKEN",
        ),
        (
            "tvly-ABCDEFGHIJKLMNO",  # secret-scan: allow -- synthetic fixture
            "PROVIDER_TAVILY_TOKEN",
        ),
        (
            "eyJabcdefgh.ijklmnopq.rstuvwxyz",  # secret-scan: allow -- synthetic fixture
            "PROVIDER_JWT",
        ),
        (
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWX",  # secret-scan: allow -- synthetic fixture
            "PROVIDER_GITHUB_TOKEN",
        ),
        (
            "AKIAABCDEFGHIJKLMNOP",  # secret-scan: allow -- synthetic fixture
            "PROVIDER_AWS_ACCESS_KEY",
        ),
        (
            "xoxb-ABCDEFGHIJKLMNO",  # secret-scan: allow -- synthetic fixture
            "PROVIDER_SLACK_TOKEN",
        ),
        (
            "AIzaABCDEFGHIJKLMNOPQRSTUVWXYZ1234",  # secret-scan: allow -- synthetic fixture
            "PROVIDER_GOOGLE_API_KEY",
        ),
        (
            "Bearer ABCDEFGHIJKLMNOPQRSTUV",  # secret-scan: allow -- synthetic fixture
            "BEARER_TOKEN",
        ),
        (
            "-----BEGIN PRIVATE KEY-----",  # secret-scan: allow -- synthetic fixture
            "PRIVATE_KEY_BLOCK",
        ),
        (
            "railcar.proxy.rlwy.net",  # secret-scan: allow -- synthetic fixture
            "RAILWAY_PROXY_HOST",
        ),
    ],
)
def test_provider_rules_reject_each_known_literal_shape(line: str, rule: str) -> None:
    """Provider-specific shapes catch literals that have no nearby key."""
    findings = find_credential_literals(line, Path("artifact.txt"))

    assert [finding.key for finding in findings] == [rule]


@pytest.mark.parametrize(
    "line",
    [
        "desktop-desk-settings",
        "sk-your_openai_api_key_here",
        "tvly-xxxxxxxxxxxx",
        "Bearer <token>",
        "Bearer $ACCESS_TOKEN",
        "-----BEGIN PUBLIC KEY-----",
        "<railway-proxy-domain>",
    ],
)
def test_provider_rules_accept_placeholders_and_near_misses(line: str) -> None:
    """Provider rules must not reject slugs, templates, or public-key blocks."""
    assert find_credential_literals(line, Path("example.md")) == []


@pytest.mark.parametrize(
    "path",
    [
        Path("cmos/db/state.sqlite"),
        Path("cmos/db/state.sqlite-wal"),
        Path("cache/state.db"),
        Path("archive/state.bak"),
        Path("runtime/state-wal"),
        Path("runtime/state-shm"),
        Path(".env"),
        Path("config/.env.production"),
    ],
)
def test_binary_carrier_guard_rejects_secret_bearing_tracked_formats(path: Path) -> None:
    """Binary and environment carriers cannot be safely reviewed line by line."""
    assert is_prohibited_carrier(path) is True


def test_binary_carrier_guard_allows_the_placeholder_environment_template() -> None:
    """The documented environment schema remains tracked and reviewable."""
    assert is_prohibited_carrier(Path(".env.example")) is False


def test_nul_binary_and_dot_venv_content_are_not_decoded(tmp_path: Path) -> None:
    """Binary parsing and vendored virtualenv noise are outside semantic scope."""
    binary = tmp_path / "payload.bin"
    binary.write_bytes(b"AUTH_PASSWORD=opaque-value\0")
    vendored = tmp_path / ".venv" / "fixture.env"
    vendored.parent.mkdir()
    vendored.write_text("AUTH_PASSWORD=opaque-value\n")

    assert scan_paths([binary, vendored]) == []


def test_large_text_requires_explicit_value_safe_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large text cannot silently evade semantic scanning by exhausting the runner."""
    candidate = tmp_path / "large-research.md"
    candidate.write_text("ordinary research text")
    monkeypatch.setattr(scanner, "_MAX_TEXT_BYTES", 4)

    findings = scan_paths([candidate])

    assert [finding.key for finding in findings] == ["LARGE_FILE_REQUIRES_REVIEW"]


def test_allowlisted_large_text_still_runs_provider_prefix_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A size exemption narrows semantic work but never disables provider detection."""
    candidate = tmp_path / "large-research.md"
    candidate.write_text(
        "sk-proj-ABCDEFGHIJKLMNOPQRSTUV"  # secret-scan: allow -- synthetic fixture
    )
    monkeypatch.setattr(scanner, "_MAX_TEXT_BYTES", 4)
    monkeypatch.setattr(scanner, "_LARGE_RESEARCH_ALLOWLIST", frozenset({candidate}))

    findings = scan_paths([candidate])

    assert [finding.key for finding in findings] == ["PROVIDER_SK_TOKEN"]


def test_fixture_exemption_requires_a_reason() -> None:
    """An exemption without an auditable explanation cannot silence a finding."""
    without_reason = "AUTH_PASSWORD=opaque-value # secret-scan: allow"
    with_reason = "AUTH_PASSWORD=opaque-value # secret-scan: allow -- synthetic CLI fixture"

    assert len(find_credential_literals(without_reason, Path("fixture.env"))) == 1
    assert find_credential_literals(with_reason, Path("fixture.env")) == []


def test_cli_diagnostic_contains_only_path_line_and_rule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A scanner failure must never amplify the credential into logs."""
    fake_value = "value-that-must-not-appear"
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(f"AUTH_PASSWORD={fake_value}\n")

    assert main([str(candidate)]) == 1

    stderr = capsys.readouterr().err
    assert stderr.strip() == f"{candidate}:1:AUTH_PASSWORD"
    assert fake_value not in stderr


def test_tracked_mode_ignores_untracked_files_and_rejects_tracked_carriers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Repository enforcement must use the index, not arbitrary local files."""
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "init", "-q", str(tmp_path)], check=True)  # noqa: S603
    tracked = tmp_path / "tracked.env"
    tracked.write_text("AUTH_PASSWORD=$AUTH_PASSWORD\n")
    subprocess.run(  # noqa: S603
        [git, "-C", str(tmp_path), "add", tracked.name],
        check=True,
    )
    (tmp_path / "untracked.env").write_text("AUTH_PASSWORD=opaque-value\n")
    monkeypatch.chdir(tmp_path)

    assert main(["--tracked"]) == 0

    carrier = tmp_path / "state.sqlite"
    carrier.write_bytes(b"\0binary")
    subprocess.run(  # noqa: S603
        [git, "-C", str(tmp_path), "add", carrier.name],
        check=True,
    )
    assert main(["--tracked"]) == 1
    assert capsys.readouterr().err.strip() == "state.sqlite:0:PROHIBITED_CARRIER"


def test_current_tracked_tree_has_no_credential_literals() -> None:
    """A clean tree prevents known credential carrier classes from returning."""
    findings = scan_paths(tracked_paths(), enforce_carrier_guard=True)

    diagnostics = "\n".join(format_finding(finding) for finding in findings)
    assert findings == [], diagnostics
