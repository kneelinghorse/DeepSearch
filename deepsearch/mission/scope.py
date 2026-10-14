"""Versioned authored bounds and deterministic evidence admission (no providers)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urldefrag, urlsplit

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from deepsearch.domain_policy import domain_matches

SCOPE_VERSION: Literal["authored-scope-v1"] = "authored-scope-v1"
_HTTP_URL = TypeAdapter(AnyHttpUrl)
_NUMBER = r"(?:\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten)"
_NUMBER_WORDS = dict(zip(
    "one two three four five six seven eight nine ten".split(), range(1, 11)
))


class AuthoredScopeError(ValueError):
    """An authored boundary is invalid or has contradictory requirements."""


def page_identity(value: str) -> str:
    """Canonical HTTP page identity; fragments collapse, origins/queries do not."""
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise AuthoredScopeError("source URL must not contain credentials")
    canonical = str(_HTTP_URL.validate_python(value))
    if len(canonical) > 4096:
        raise AuthoredScopeError("source URL exceeds 4096 characters")
    return urldefrag(canonical)[0]


class AuthoredScope(BaseModel):
    """Hard limits, separate from ordinary reference seeds and authority hints."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["authored-scope-v1"] = SCOPE_VERSION
    reference_urls: list[str] = Field(default_factory=list)
    restriction: Literal["unrestricted", "domains", "exact_pages"] = "unrestricted"
    allowed_urls: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    min_words: int | None = Field(default=None, ge=0, strict=True)
    max_words: int | None = Field(default=None, ge=1, strict=True)
    max_sources: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> AuthoredScope:
        if (self.min_words is not None and self.max_words is not None
                and self.min_words > self.max_words):
            raise AuthoredScopeError("min_words exceeds max_words")
        for url in [*self.reference_urls, *self.allowed_urls]:
            page_identity(url)
        for domain in self.allowed_domains:
            parsed = urlsplit("https://" + domain)
            if (not parsed.hostname or parsed.hostname != domain or parsed.path
                    or parsed.query or parsed.fragment or parsed.port is not None
                    or len(domain) > 253
                    or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                           for label in domain.split("."))):
                raise AuthoredScopeError("allowed_domains must contain lowercase hostnames")
        if self.restriction == "exact_pages" and not self.allowed_urls:
            raise AuthoredScopeError("exact_pages requires allowed_urls")
        if self.restriction == "domains" and not self.allowed_domains:
            raise AuthoredScopeError("domains restriction requires allowed_domains")
        if self.restriction == "unrestricted" and (self.allowed_urls or self.allowed_domains):
            raise AuthoredScopeError("unrestricted scope cannot carry an allowlist")
        if self.restriction == "domains" and self.allowed_urls:
            raise AuthoredScopeError("allowed_urls requires exact_pages restriction")
        if self.allowed_domains and any(not self.domain_allowed(u) for u in self.allowed_urls):
            raise AuthoredScopeError("allowed page conflicts with domain restriction")
        return self

    def domain_allowed(self, url: str) -> bool:
        """Match explicitly allowed hostnames and their subdomains, without www folding."""
        host = urlsplit(url).hostname or ""
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

    def allows(self, url: str) -> bool:
        """Check a source identity without consuming its consultation budget."""
        try:
            key = page_identity(url)
        except ValueError:
            return False
        if self.restriction == "exact_pages":
            return key in {page_identity(u) for u in self.allowed_urls}
        return self.restriction == "unrestricted" or self.domain_allowed(key)

    def prompt(self) -> str:
        """Render the same policy for actor, retained-memory finalization and critic."""
        lines = [f"Authored scope ({self.version}): {self.restriction}."]
        if self.allowed_urls:
            lines += ["Consult only these exact pages; use source_fetch, not search:",
                      *self.allowed_urls]
        elif self.allowed_domains:
            lines += ["Consult only these domains and subdomains:", *self.allowed_domains]
        elif self.reference_urls:
            lines += ["Reference seeds (not an allowlist):", *self.reference_urls]
        if self.max_sources is not None:
            lines += [f"At most {self.max_sources} distinct consulted pages, including snippets."]
        if self.min_words is not None or self.max_words is not None:
            lines += [
                f"Final persisted report: minimum {self.min_words or 0} words; "
                f"maximum {self.max_words if self.max_words is not None else 'unbounded'} words.",
                "Count all whitespace-separated tokens, including Markdown, citations, title, "
                "metadata and footer. Leave room for the generated wrapper.",
            ]
        lines += ["Unavailable sources require an explicit partial-result warning; "
                  "never broaden the boundary or claim unseen evidence was consulted."]
        return "\n".join(lines)


def _number(text: str) -> int:
    return _NUMBER_WORDS[text] if text in _NUMBER_WORDS else int(text.replace(",", ""))


def compile_authored_scope(context: Mapping[str, Any]) -> AuthoredScope:
    """Compile structured bounds plus a documented, deliberately small prose grammar.

    Only objective/criteria/constraints/deliverables are instructions. Background
    and reference titles cannot silently turn reference seeds into an allowlist.
    Numeric requirements intersect; inconsistent requirements fail before payment.
    """
    references = context.get("references") or []
    if isinstance(references, Mapping):
        references = list(references.values())
    urls: list[str] = []
    for item in references:
        url = item.get("url") if isinstance(item, Mapping) else item
        if isinstance(item, Mapping) and "url" in item:
            if not isinstance(url, str):
                raise AuthoredScopeError("reference URL must be text")
            page_identity(url)
        if isinstance(url, str) and url.lower().startswith(("https://", "http://")):
            page_identity(url)
            if url not in urls:
                urls.append(url)
    raw = context.get("authored_scope")
    scope = AuthoredScope.model_validate(raw if raw is not None else {})
    data = scope.model_dump()
    data["reference_urls"] = list(dict.fromkeys([*scope.reference_urls, *urls]))
    text_parts = [str(context.get("objective") or "")]
    for key in ("success_criteria", "constraints", "deliverables"):
        value = context.get(key) or []
        text_parts.extend([value] if isinstance(value, str) else value)
    text = "\n".join(str(p) for p in text_parts).lower()
    minima = [scope.min_words] if scope.min_words is not None else []
    maxima = [scope.max_words] if scope.max_words is not None else []
    for low, high in re.findall(rf"\b({_NUMBER})\s*[-–—]\s*({_NUMBER})[-\s]+words?\b", text):
        minima.append(_number(low))
        maxima.append(_number(high))
    for prefix, num in re.findall(
        rf"\b(at least|minimum(?: of)?|at most|no more than|maximum(?: of)?)\s+({_NUMBER})"
        r"[-\s]+words?\b", text,
    ):
        (minima if prefix.startswith(("at least", "minimum")) else maxima).append(_number(num))
    data["min_words"] = max(minima) if minima else None
    data["max_words"] = min(maxima) if maxima else None
    caps = [scope.max_sources] if scope.max_sources is not None else []
    for num in re.findall(
        rf"\b(?:at most|no more than|maximum(?: of)?)\s+({_NUMBER})\s+"
        r"(?:distinct\s+)?(?:primary\s+)?(?:source pages|sources|pages)\b", text,
    ):
        caps.append(_number(num))
    data["max_sources"] = min(caps) if caps else None
    exact = re.search(
        rf"\b(?:stop after|use only|consult only)\s+(?:the\s+)?(?:{_NUMBER}\s+)?"
        r"(?:requested|listed|provided|referenced)\s+(?:source\s+)?(?:pages|sources|urls)\b",
        text,
    )
    domains = re.findall(
        r"\b(?:use|consult) only\s+([a-z0-9][a-z0-9.-]*\.[a-z]{2,})\b", text
    )
    if domains:
        if scope.allowed_domains and set(domains) != set(scope.allowed_domains):
            raise AuthoredScopeError("structured and prose domain restrictions conflict")
        data["allowed_domains"] = list(dict.fromkeys(domains))
        if data["restriction"] == "unrestricted":
            data["restriction"] = "domains"
    if exact:
        if scope.allowed_urls and {page_identity(u) for u in scope.allowed_urls} != {
            page_identity(u) for u in urls
        }:
            raise AuthoredScopeError("structured exact pages conflict with requested references")
        data["restriction"] = "exact_pages"
        data["allowed_urls"] = urls
    compiled = AuthoredScope.model_validate(data)
    if any(domain_matches(u, context.get("excluded_domains")) for u in compiled.allowed_urls):
        raise AuthoredScopeError("allowed page conflicts with excluded domain")
    return compiled


def resolve_authored_scope(state: Mapping[str, Any]) -> AuthoredScope:
    """Resolve compiled policy, or compile a local preview's authoring context."""
    contract = state.get("mission_contract") or {}
    if "authored_scope" in contract:
        return AuthoredScope.model_validate(contract["authored_scope"])
    return compile_authored_scope(state.get("mission_context") or {})


@dataclass
class ScopeAdmission:
    """Attempt-local, lossless consultation ledger shared across evidence consumers."""

    scope: AuthoredScope
    consulted: dict[str, list[str]] = field(default_factory=dict)
    rejected: list[dict[str, str]] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)

    def refusal(self, url: str) -> str | None:
        """Check before a request or stored-body read, without counting a failed fetch."""
        if not self.scope.allows(url):
            return "scope_source_not_allowed"
        key = page_identity(url)
        if (key not in self.consulted and self.scope.max_sources is not None
                and len(self.consulted) >= self.scope.max_sources):
            return "scope_source_limit"
        return None

    def admit(self, url: str, surface: str) -> bool:
        """Count every admitted snippet/body even when never cited in the report."""
        reason = self.refusal(url)
        if reason:
            self.reject(url, surface, reason)
            return False
        surfaces = self.consulted.setdefault(page_identity(url), [])
        if surface not in surfaces:
            surfaces.append(surface)
        return True

    def reject(self, url: str, surface: str, reason: str) -> None:
        """Retain a content-free rejection receipt without counting unseen evidence."""
        try:
            safe_url = page_identity(url)
        except ValueError:
            safe_url = ""
        record = {"url": safe_url, "surface": surface, "reason": reason}
        if record not in self.rejected:
            self.rejected.append(record)

    def search_refusal(self, tool: str, query: str) -> str | None:
        """Reject discovery that cannot obey the boundary before provider requests."""
        if self.scope.restriction == "exact_pages":
            return "scope_exact_pages_no_search"
        if self.scope.allowed_domains:
            if tool == "scholar_search":
                return "scope_scholar_domain_filter_unavailable"
            for domain in re.findall(r"\bsite:([^\s)]+)", query):
                if not self.scope.domain_allowed("https://" + domain):
                    return "scope_search_domain_not_allowed"
        if (self.scope.max_sources is not None
                and len(self.consulted) >= self.scope.max_sources):
            return "scope_source_limit"
        return None

    def filter_sources(
        self, sources: Sequence[Mapping[str, Any]], surface: str,
    ) -> list[dict[str, Any]]:
        """Admit source records in stable input order before any model sees them."""
        return [dict(s) for s in sources if isinstance(s, Mapping)
                and self.admit(str(s.get("url") or ""), surface)]

    def snapshot(self) -> dict[str, Any]:
        """Return durable policy identity, all consultation identities and partial warnings."""
        policy = self.scope.model_dump(mode="json")
        unavailable = dict(self.unavailable)
        for url in self.scope.allowed_urls:
            if page_identity(url) not in self.consulted:
                unavailable.setdefault(url, "not_retrieved")
        return {
            "policy": policy,
            "policy_sha256": hashlib.sha256(json.dumps(
                policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode()).hexdigest(),
            "consulted": [{"url": u, "surfaces": list(s)} for u, s in self.consulted.items()],
            "consulted_count": len(self.consulted),
            "rejected": list(self.rejected),
            "unavailable": [{"url": u, "reason": r} for u, r in unavailable.items()],
        }
