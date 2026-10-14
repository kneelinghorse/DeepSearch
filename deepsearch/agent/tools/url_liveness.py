"""url_liveness — HEAD/GET probe for the CodeAgent.

Standalone tool so the CodeAgent can pre-check a URL before paying the cost of
a full ``source_fetch``, and so the citation liveness gate (S68+) can verify
cited URLs without re-fetching their bodies.

Returns ``{alive, status_code, final_url, error}``. Tries HEAD first; falls
back to GET with ``Range: bytes=0-0`` when the host returns 405 (Method Not
Allowed). S84 m01: any other 4xx triggers a browser-UA GET fallback — bot
walls routinely answer the default UA with 403/429/451 while serving a
browser UA fine — and only the GET's verdict is reported, so a 4xx in the
payload is always GET-confirmed.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx

from deepsearch.domain_policy import ExcludedDomainError, domain_matches, normalize_domains
from deepsearch.mission.scope import AuthoredScopeError, ScopeAdmission

from .source_fetch import UnsafeTargetError, _validate_public_target

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; DeepSearchBot/1.0)"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT_S = 5.0

ERROR_TIMEOUT = "timeout"
ERROR_CONNECT = "connect"
ERROR_INVALID = "invalid"
ERROR_OTHER = "other"
ERROR_EXCLUDED_DOMAIN = "excluded_domain"


def _failed(url: str, error: str) -> dict[str, Any]:
    return {"alive": False, "status_code": 0, "final_url": url, "error": error}


def _fallback_get(
    client: "httpx.Client", url: str, headers: dict[str, str]
) -> tuple[Any, str | None]:
    """One fallback GET → ``(response_or_None, error_category_or_None)``.

    A transport failure here must NOT erase the HTTP status the ladder has
    already obtained — a bot wall that answers the HEAD with 403 and then
    refuses the GET connection would otherwise masquerade as NXDOMAIN and
    satisfy the batch wrapper's connect-twice dead rule (S84 m01 adversarial
    review finding).
    """
    try:
        return client.get(url, headers=headers), None
    except AuthoredScopeError:
        raise
    except ExcludedDomainError:
        raise
    except UnsafeTargetError:
        raise
    except httpx.TimeoutException:
        return None, ERROR_TIMEOUT
    except httpx.ConnectError:
        return None, ERROR_CONNECT
    except Exception as exc:  # noqa: BLE001 — no-raise contract
        logger.warning(
            "url_liveness fallback GET failed category=%s exception=%s",
            ERROR_OTHER,
            type(exc).__name__,
        )
        return None, ERROR_OTHER


def url_liveness(
    url: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    *,
    excluded_domains: Sequence[str] | None = None,
    scope_admission: ScopeAdmission | None = None,
) -> dict[str, Any]:
    """Probe a URL with HEAD; GET fallbacks on 405 and on any other 4xx.

    Returns ``{alive, status_code, final_url, error}``. ``alive`` is True iff
    the final response status is in ``[200, 400)`` and the probe chain
    completed. ``error`` is ``None`` iff the chain completed with an HTTP
    response; otherwise it is ``"timeout"`` / ``"connect"`` (DNS failure or
    connection refused) / ``"invalid"`` (unusable input URL) / ``"other"``.
    When a fallback GET fails at transport level, ``status_code`` preserves
    the last HTTP status obtained (e.g. the HEAD's 403) alongside the error
    category — so ``status_code == 0`` with ``error == "connect"`` means the
    host never answered at all, and a 404/410 with ``error is None`` is
    GET-confirmed. Never raises.

    Fallback ladder (S84 m01):

    * HEAD 405 → bot-UA GET with ``Range: bytes=0-0`` (unchanged S64 path).
    * Any remaining 4xx → plain GET with a browser User-Agent. On success the
      GET's status and final_url replace the prior response's.

    Args:
        url: Absolute URL to probe.
        timeout_s: Per-request timeout in seconds.
        excluded_domains: Mission-local domains rejected before direct or
            redirected requests connect.
    """

    if not isinstance(url, str) or not url.strip():
        return _failed("", ERROR_INVALID)
    bound_excluded_domains = normalize_domains(excluded_domains)
    if domain_matches(url, bound_excluded_domains):
        return _failed(url, ERROR_EXCLUDED_DOMAIN)

    def _guard_request(request: Any) -> None:
        if scope_admission is not None:
            reason = scope_admission.refusal(str(request.url))
            if reason:
                scope_admission.reject(str(request.url), "probe_request", reason)
                raise AuthoredScopeError(reason)
        _validate_public_target(str(request.url))
        if domain_matches(request.url, bound_excluded_domains):
            raise ExcludedDomainError(request.url)

    error: str | None = None
    try:
        client_kwargs: dict[str, Any] = {
            "timeout": timeout_s,
            "follow_redirects": True,
            "headers": {"User-Agent": DEFAULT_USER_AGENT},
        }
        # Validate every actual request, including each redirect hop followed
        # internally by httpx. This is the same public-target boundary used by
        # source_fetch and prevents citation/model URLs reaching loopback,
        # RFC1918/link-local, or cloud-metadata services.
        client_kwargs["event_hooks"] = {"request": [_guard_request]}
        with httpx.Client(**client_kwargs) as client:
            resp = client.head(url)
            if resp.status_code == 405:
                got, error = _fallback_get(client, url, {"Range": "bytes=0-0"})
                if got is not None:
                    resp = got
            if error is None and 400 <= resp.status_code < 500:
                got, error = _fallback_get(
                    client, url, {"User-Agent": BROWSER_USER_AGENT}
                )
                if got is not None:
                    resp = got
    except AuthoredScopeError as exc:
        return _failed(url, str(exc))
    except ExcludedDomainError as exc:
        return _failed(exc.url, ERROR_EXCLUDED_DOMAIN)
    except UnsafeTargetError:
        return _failed(url, ERROR_INVALID)
    except httpx.TimeoutException:
        return _failed(url, ERROR_TIMEOUT)
    except httpx.ConnectError:
        return _failed(url, ERROR_CONNECT)
    except Exception as exc:
        logger.warning(
            "url_liveness failed category=%s exception=%s",
            ERROR_OTHER,
            type(exc).__name__,
        )
        return _failed(url, ERROR_OTHER)

    code = int(resp.status_code)
    return {
        "alive": error is None and 200 <= code < 400,
        "status_code": code,
        "final_url": str(resp.url),
        "error": error,
    }


__all__ = [
    "BROWSER_USER_AGENT",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_USER_AGENT",
    "ERROR_CONNECT",
    "ERROR_EXCLUDED_DOMAIN",
    "ERROR_INVALID",
    "ERROR_OTHER",
    "ERROR_TIMEOUT",
    "url_liveness",
]
