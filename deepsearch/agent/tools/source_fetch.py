"""Safe URL-to-readable-text retrieval for the CodeAgent.

The fetch boundary validates every target (including redirects) before the
connection, accepts only readable content types, streams through a hard body
limit, and extracts deterministic text from HTML.  It never raises.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Sequence
from urllib.parse import urljoin, urlparse

import httpx

from deepsearch.domain_policy import domain_matches, normalize_domains
from deepsearch.mission.scope import ScopeAdmission

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; DeepSearchBot/1.0)"
DEFAULT_TIMEOUT_S = 15.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_READABLE_CHARS = 500_000
MAX_REDIRECTS = 5

# Compatibility export for older callers. The boundary now rejects raw bodies
# over MAX_RESPONSE_BYTES instead of clipping a multi-megabyte response.
MAX_TEXT_BYTES = MAX_RESPONSE_BYTES

_BLOCKED_CODES = frozenset({401, 403, 429, 451})
_DEAD_CODES = frozenset({404, 410})
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_CLOUD_METADATA_HOSTS = frozenset(
    {
        "metadata.google.internal",
        "metadata.google.internal.",
        "metadata.aws.internal",
        "metadata.azure.internal",
    }
)
_READABLE_APPLICATION_TYPES = frozenset(
    {
        "application/json",
        "application/ld+json",
        "application/xhtml+xml",
        "application/xml",
    }
)
_SKIP_HTML_TAGS = frozenset(
    {"head", "script", "style", "noscript", "svg", "canvas", "template"}
)
_BLOCK_HTML_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)


class UnsafeTargetError(ValueError):
    """Raised when a URL could reach a non-public network target."""


class _ReadableHTMLParser(HTMLParser):
    """Small stdlib-only HTML-to-text extractor."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in _SKIP_HTML_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and lowered in _BLOCK_HTML_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in _SKIP_HTML_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._skip_depth == 0 and lowered in _BLOCK_HTML_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._parts.append(data)

    def text(self) -> str:
        lines = [re.sub(r"\s+", " ", part).strip() for part in "".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result(
    *,
    text: str,
    status: str,
    status_code: int,
    final_url: str,
    error: str | None = None,
    content_type: str = "",
    bytes_read: int = 0,
    text_truncated: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "text": text,
        "status": status,
        "status_code": status_code,
        "fetched_at": _now_iso(),
        "final_url": final_url,
        "content_type": content_type,
        "bytes_read": bytes_read,
        "text_truncated": text_truncated,
    }
    if error:
        result["error"] = error
    return result


def _blocked_result(url: str, error: str) -> dict[str, Any]:
    return _result(
        text="",
        status="blocked",
        status_code=0,
        final_url=url,
        error=error,
    )


def _resolve_host_addresses(host: str, port: int) -> set[str]:
    """Resolve every address for one connection target."""

    try:
        return {host} if ipaddress.ip_address(host) else set()
    except ValueError:
        return {
            str(item[4][0])
            for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }


def _validate_public_target(url: str) -> None:
    """Reject malformed, credentialed, metadata, and non-public targets."""

    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise UnsafeTargetError("invalid_scheme_or_host")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeTargetError("url_credentials_not_allowed")

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise UnsafeTargetError("loopback_target")
    if host in _CLOUD_METADATA_HOSTS or host.endswith(".metadata.google.internal"):
        raise UnsafeTargetError("cloud_metadata_target")

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise UnsafeTargetError("invalid_port") from exc
    addresses = _resolve_host_addresses(host, port)
    if not addresses:
        raise socket.gaierror(f"no addresses for {host}")
    for raw_address in addresses:
        address = ipaddress.ip_address(raw_address)
        if (
            not address.is_global
            or address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise UnsafeTargetError(f"non_public_address:{address}")


def _content_type(headers: Any) -> str:
    raw = str(headers.get("content-type", "") or "")
    return raw.split(";", 1)[0].strip().lower()


def _is_readable_content_type(content_type: str) -> bool:
    return content_type.startswith("text/") or content_type in _READABLE_APPLICATION_TYPES


def _decode_body(body: bytes, headers: Any) -> str:
    raw_content_type = str(headers.get("content-type", "") or "")
    match = re.search(r"charset=([^;\s]+)", raw_content_type, flags=re.IGNORECASE)
    encoding = match.group(1).strip("\"'") if match else "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _extract_readable_text(body: bytes, content_type: str, headers: Any) -> str:
    decoded = _decode_body(body, headers)
    if content_type in {"text/html", "application/xhtml+xml"}:
        parser = _ReadableHTMLParser()
        parser.feed(decoded)
        parser.close()
        return parser.text().strip()
    return decoded.strip()


def source_fetch(
    url: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    *,
    excluded_domains: Sequence[str] | None = None,
    scope_admission: ScopeAdmission | None = None,
) -> dict[str, Any]:
    """Fetch a public URL and return bounded, readable text with provenance.

    Args:
        url: Absolute HTTP(S) URL to retrieve.
        timeout_s: Per-request timeout in seconds.
        excluded_domains: Mission-local domains rejected before DNS/transport.

    Returns:
        A no-raise status mapping containing readable ``text`` on success and
        explicit provenance/error fields on every path.
    """

    if not isinstance(url, str) or not url.strip():
        return _result(text="", status="error", status_code=0, final_url="")
    current_url = url.strip()
    bound_excluded_domains = normalize_domains(excluded_domains)

    try:
        with httpx.Client(
            timeout=timeout_s,
            follow_redirects=False,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            for redirect_count in range(MAX_REDIRECTS + 1):
                if scope_admission is not None:
                    reason = scope_admission.refusal(current_url)
                    if reason:
                        scope_admission.reject(current_url, "fetch_request", reason)
                        return _blocked_result(current_url, reason)
                if domain_matches(current_url, bound_excluded_domains):
                    return _blocked_result(current_url, "excluded_domain")
                try:
                    _validate_public_target(current_url)
                except UnsafeTargetError as exc:
                    return _blocked_result(current_url, str(exc))
                except socket.gaierror:
                    return _result(
                        text="",
                        status="dead",
                        status_code=0,
                        final_url=current_url,
                        error="dns",
                    )

                with client.stream("GET", current_url) as response:
                    code = int(response.status_code)
                    final_url = str(response.url or current_url)
                    if code in _REDIRECT_CODES:
                        location = str(response.headers.get("location", "") or "").strip()
                        if not location:
                            return _result(
                                text="",
                                status="error",
                                status_code=code,
                                final_url=final_url,
                                error="redirect_missing_location",
                            )
                        if redirect_count >= MAX_REDIRECTS:
                            return _result(
                                text="",
                                status="error",
                                status_code=code,
                                final_url=final_url,
                                error="too_many_redirects",
                            )
                        current_url = urljoin(current_url, location)
                        continue

                    if code in _BLOCKED_CODES:
                        return _result(
                            text="",
                            status="blocked",
                            status_code=code,
                            final_url=final_url,
                            error="http_blocked",
                        )
                    if code in _DEAD_CODES:
                        return _result(
                            text="",
                            status="dead",
                            status_code=code,
                            final_url=final_url,
                            error="http_dead",
                        )
                    if not 200 <= code < 300:
                        return _result(
                            text="",
                            status="error",
                            status_code=code,
                            final_url=final_url,
                            error="http_error",
                        )

                    content_type = _content_type(response.headers)
                    if not _is_readable_content_type(content_type):
                        return _result(
                            text="",
                            status="error",
                            status_code=code,
                            final_url=final_url,
                            error="unsupported_content_type",
                            content_type=content_type,
                        )
                    declared_size = response.headers.get("content-length")
                    if declared_size:
                        try:
                            if int(declared_size) > MAX_RESPONSE_BYTES:
                                return _result(
                                    text="",
                                    status="error",
                                    status_code=code,
                                    final_url=final_url,
                                    error="body_too_large",
                                    content_type=content_type,
                                )
                        except (TypeError, ValueError):
                            pass

                    chunks: list[bytes] = []
                    bytes_read = 0
                    for chunk in response.iter_bytes():
                        bytes_read += len(chunk)
                        if bytes_read > MAX_RESPONSE_BYTES:
                            return _result(
                                text="",
                                status="error",
                                status_code=code,
                                final_url=final_url,
                                error="body_too_large",
                                content_type=content_type,
                                bytes_read=bytes_read,
                            )
                        chunks.append(chunk)

                    try:
                        text = _extract_readable_text(
                            b"".join(chunks),
                            content_type,
                            response.headers,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.info(
                            "source_fetch failed category=extraction_failed exception=%s",
                            type(exc).__name__,
                        )
                        return _result(
                            text="",
                            status="error",
                            status_code=code,
                            final_url=final_url,
                            error="extraction_failed",
                            content_type=content_type,
                            bytes_read=bytes_read,
                        )
                    if not text:
                        return _result(
                            text="",
                            status="error",
                            status_code=code,
                            final_url=final_url,
                            error="extraction_failed_or_spa_shell",
                            content_type=content_type,
                            bytes_read=bytes_read,
                        )
                    truncated = len(text) > MAX_READABLE_CHARS
                    if truncated:
                        text = text[:MAX_READABLE_CHARS]
                    return _result(
                        text=text,
                        status="ok",
                        status_code=code,
                        final_url=final_url,
                        content_type=content_type,
                        bytes_read=bytes_read,
                        text_truncated=truncated,
                    )
    except httpx.TimeoutException:
        return _result(
            text="", status="timeout", status_code=0, final_url=current_url, error="timeout"
        )
    except httpx.ConnectError:
        return _result(
            text="", status="dead", status_code=0, final_url=current_url, error="connect"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "source_fetch failed category=unexpected exception=%s",
            type(exc).__name__,
        )
        return _result(
            text="", status="error", status_code=0, final_url=current_url, error="unexpected"
        )

    return _result(
        text="", status="error", status_code=0, final_url=current_url, error="redirect_loop"
    )


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_USER_AGENT",
    "MAX_READABLE_CHARS",
    "MAX_REDIRECTS",
    "MAX_RESPONSE_BYTES",
    "MAX_TEXT_BYTES",
    "UnsafeTargetError",
    "source_fetch",
]
