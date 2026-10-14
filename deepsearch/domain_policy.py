"""Pure domain normalization and matching for mission-scoped retrieval policy."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse


def normalize_domain(value: object) -> str:
    """Return a lowercase hostname without ``www.``, port, or trailing dot."""

    text = str(value or "").strip().lower()
    if not text:
        return ""
    candidate = text if "://" in text else f"https://{text.lstrip('*.')}"
    try:
        host = (urlparse(candidate).hostname or "").lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host.rstrip(".")


def normalize_domains(values: Iterable[object] | object | None) -> tuple[str, ...]:
    """Normalize and case-insensitively deduplicate a domain collection."""

    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        candidates: Iterable[object] = (values,)
    elif isinstance(values, Iterable):
        candidates = values
    else:
        candidates = (values,)

    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        domain = normalize_domain(value)
        if domain and domain not in seen:
            seen.add(domain)
            normalized.append(domain)
    return tuple(normalized)


def domain_matches(host_or_url: object, domains: Iterable[object] | object | None) -> bool:
    """Return whether a host is a configured domain or one of its subdomains."""

    host = normalize_domain(host_or_url)
    if not host:
        return False
    return any(
        host == domain or host.endswith(f".{domain}")
        for domain in normalize_domains(domains)
    )


class ExcludedDomainError(RuntimeError):
    """Raised by an HTTP request hook before an excluded target is connected."""

    def __init__(self, url: object) -> None:
        self.url = str(url or "")
        super().__init__(f"mission excludes domain: {normalize_domain(self.url)}")


__all__ = [
    "ExcludedDomainError",
    "domain_matches",
    "normalize_domain",
    "normalize_domains",
]
