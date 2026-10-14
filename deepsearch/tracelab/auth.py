"""TraceLab service authentication.

This module prefers a long-lived API key and retains JWT login as an explicit
fallback:
- Send ``TRACELAB_API_KEY`` as ``X-API-Key`` when configured
- Login to get an access token only when no API key is configured
- Token caching with expiry tracking
- Auto-refresh before token expiry
- Retry on 401 errors with re-login
- Verify that the resolved principal has TraceLab's ``service`` role

Environment Variables:
    TRACELAB_BASE_URL: Base URL for TraceLab API (e.g., https://api.namozine.com)
    TRACELAB_API_KEY: Preferred service-principal API key
    TRACELAB_USERNAME: Service-principal email for JWT fallback
    TRACELAB_PASSWORD: Password for JWT fallback
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Refresh token 5 minutes before expiry to prevent edge cases
TOKEN_REFRESH_BUFFER_SECONDS = 300


def is_secure_tracelab_url(base_url: str) -> bool:
    """Allow HTTPS everywhere and plain HTTP only on explicit loopback hosts."""

    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host:
        return True
    if parsed.scheme != "http":
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class TokenInfo:
    """JWT token with expiry tracking."""

    access_token: str
    expires_at: float  # Unix timestamp
    token_type: str = "bearer"

    @property
    def is_expired(self) -> bool:
        """Check if token is expired or about to expire."""
        return time.time() > (self.expires_at - TOKEN_REFRESH_BUFFER_SECONDS)

    @property
    def time_until_expiry(self) -> float:
        """Seconds until token expires."""
        return max(0.0, self.expires_at - time.time())


class TraceLabAuthError(Exception):
    """Raised when authentication with TraceLab fails."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class TraceLabAuth:
    """API-key-first service authentication for TraceLab API.

    Handles login, token caching, and automatic refresh. Uses environment
    variables for configuration by default.

    Args:
        base_url: TraceLab API base URL (default: from TRACELAB_BASE_URL env var)
        api_key: Service API key (default: from TRACELAB_API_KEY env var)
        username: Service email for JWT fallback (default: TRACELAB_USERNAME)
        password: Password for authentication (default: from TRACELAB_PASSWORD env var)
        timeout: HTTP timeout in seconds for auth requests

    Example:
        >>> auth = TraceLabAuth()
        >>> headers = auth.get_headers()
        >>> # Use headers for API requests
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.getenv("TRACELAB_BASE_URL", "")).rstrip("/")
        self.api_key = (
            api_key if api_key is not None else os.getenv("TRACELAB_API_KEY", "")
        )
        self.username = (
            username if username is not None else os.getenv("TRACELAB_USERNAME", "")
        )
        self.password = (
            password if password is not None else os.getenv("TRACELAB_PASSWORD", "")
        )
        self.timeout = timeout
        self._token: Optional[TokenInfo] = None
        self._service_role_verified = False

        # Track auth events for telemetry
        self._login_count = 0
        self._refresh_count = 0
        self._last_login_at: Optional[float] = None

    @property
    def is_configured(self) -> bool:
        """Check if auth credentials are configured."""
        return bool(self.base_url and (self.api_key or (self.username and self.password)))

    @property
    def uses_api_key(self) -> bool:
        """Return whether requests use the preferred API-key credential."""

        return bool(self.api_key)

    def get_token(self) -> str:
        """Get valid JWT token, logging in or refreshing as needed.

        Returns:
            Valid JWT access token string.

        Raises:
            TraceLabAuthError: If login fails or credentials are missing.
        """
        if not (self.base_url and self.username and self.password):
            missing = []
            if not self.base_url:
                missing.append("TRACELAB_BASE_URL")
            if not self.username:
                missing.append("TRACELAB_USERNAME")
            if not self.password:
                missing.append("TRACELAB_PASSWORD")
            raise TraceLabAuthError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        if self._token is None or self._token.is_expired:
            self._login()

        return self._token.access_token

    def get_headers(self) -> Dict[str, str]:
        """Get Authorization headers with valid token.

        Returns:
            Dictionary with Authorization header.

        Raises:
            TraceLabAuthError: If authentication fails.
        """
        if not self.base_url:
            raise TraceLabAuthError("Missing required environment variable: TRACELAB_BASE_URL")
        if not is_secure_tracelab_url(self.base_url):
            raise TraceLabAuthError(
                "TraceLab credentials require HTTPS outside localhost"
            )
        if self.api_key:
            return {"X-API-Key": self.api_key}
        return {"Authorization": f"Bearer {self.get_token()}"}

    def require_service_role(self) -> None:
        """Verify once that the configured credential resolves to ``service``.

        Raises:
            TraceLabAuthError: If authentication fails, the response is invalid,
                or the principal is not a TraceLab service account.
        """

        if self._service_role_verified:
            return

        try:
            import httpx
        except ImportError as exc:
            raise TraceLabAuthError(
                "httpx not installed - run: pip install httpx"
            ) from exc

        url = f"{self.base_url}/api/v1/auth/me"
        try:
            response = httpx.get(
                url,
                headers=self.get_headers(),
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise TraceLabAuthError("Service-role validation timed out") from exc
        except httpx.RequestError as exc:
            raise TraceLabAuthError("Service-role validation request failed") from exc

        if response.status_code != 200:
            raise TraceLabAuthError(
                f"Service-role validation failed with status {response.status_code}",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TraceLabAuthError("Service-role validation returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise TraceLabAuthError(
                "Service-role validation returned a non-object JSON payload"
            )

        role = str(payload.get("role") or "").strip().lower()
        if role != "service":
            raise TraceLabAuthError(
                "TraceLab credential is not 'service' role",
                status_code=403,
            )
        self._service_role_verified = True

    def _login(self) -> None:
        """Login to TraceLab to get new JWT token.

        Raises:
            TraceLabAuthError: If login request fails.
        """
        if not is_secure_tracelab_url(self.base_url):
            raise TraceLabAuthError(
                "TraceLab credentials require HTTPS outside localhost"
            )
        try:
            import httpx
        except ImportError as exc:
            raise TraceLabAuthError(
                "httpx not installed - run: pip install httpx"
            ) from exc

        # NOTE: No trailing slash on endpoint per TraceLab B12.2 workaround
        login_url = f"{self.base_url}/api/v1/auth/login"
        logger.debug("Attempting TraceLab login at %s", login_url)

        try:
            response = httpx.post(
                login_url,
                json={"email": self.username, "password": self.password},
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise TraceLabAuthError(
                f"Login request timed out ({type(exc).__name__})"
            ) from exc
        except httpx.RequestError as exc:
            raise TraceLabAuthError(
                f"Login request failed ({type(exc).__name__})"
            ) from exc

        if response.status_code == 401:
            raise TraceLabAuthError(
                "Invalid credentials - check TRACELAB_USERNAME and TRACELAB_PASSWORD",
                status_code=401,
            )

        if response.status_code != 200:
            raise TraceLabAuthError(
                f"Login failed with status {response.status_code}",
                status_code=response.status_code,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise TraceLabAuthError("Invalid JSON response from login") from exc

        access_token = data.get("access_token")
        if not access_token:
            raise TraceLabAuthError("Login response missing access_token")

        expires_in = data.get("expires_in", 86400)  # Default 24 hours
        self._token = TokenInfo(
            access_token=access_token,
            expires_at=time.time() + expires_in,
            token_type=data.get("token_type", "bearer"),
        )

        self._login_count += 1
        self._last_login_at = time.time()
        logger.info(
            "TraceLab login successful (token expires in %.0f seconds)",
            self._token.time_until_expiry,
        )

    def refresh(self) -> None:
        """Refresh token before expiry.

        Falls back to full login if refresh fails.
        """
        if not is_secure_tracelab_url(self.base_url):
            raise TraceLabAuthError(
                "TraceLab credentials require HTTPS outside localhost"
            )
        if self.api_key:
            return
        if self._token is None:
            self._login()
            return

        try:
            import httpx
        except ImportError:
            self._login()
            return

        refresh_url = f"{self.base_url}/api/v1/auth/refresh"
        logger.debug("Attempting TraceLab token refresh")

        try:
            response = httpx.post(
                refresh_url,
                headers={"Authorization": f"Bearer {self._token.access_token}"},
                timeout=self.timeout,
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.warning(
                "Token refresh failed (%s), falling back to login",
                type(exc).__name__,
            )
            self._login()
            return

        if response.status_code == 401:
            # Token already expired, do full login
            logger.info("Token expired, performing full login")
            self._login()
            return

        if response.status_code != 200:
            logger.warning(
                "Token refresh failed with status %d, falling back to login",
                response.status_code,
            )
            self._login()
            return

        try:
            data = response.json()
        except ValueError:
            logger.warning("Invalid JSON in refresh response, falling back to login")
            self._login()
            return

        access_token = data.get("access_token")
        if not access_token:
            logger.warning("Refresh response missing access_token, falling back to login")
            self._login()
            return

        expires_in = data.get("expires_in", 86400)
        self._token = TokenInfo(
            access_token=access_token,
            expires_at=time.time() + expires_in,
            token_type=data.get("token_type", "bearer"),
        )

        self._refresh_count += 1
        logger.info(
            "TraceLab token refreshed (expires in %.0f seconds)",
            self._token.time_until_expiry,
        )

    def invalidate(self) -> None:
        """Invalidate cached token, forcing re-login on next request."""
        self._token = None
        self._service_role_verified = False
        logger.debug("TraceLab token invalidated")

    def get_telemetry(self) -> Dict[str, Any]:
        """Get authentication telemetry for logging.

        Returns:
            Dictionary with login_count, refresh_count, last_login_at.
        """
        return {
            "login_count": self._login_count,
            "refresh_count": self._refresh_count,
            "last_login_at": self._last_login_at,
            "token_valid": self._token is not None and not self._token.is_expired,
            "auth_mode": "api_key" if self.api_key else "jwt",
            "service_role_verified": self._service_role_verified,
        }


# Module-level singleton for convenience
_default_auth: Optional[TraceLabAuth] = None


def get_default_auth() -> TraceLabAuth:
    """Get or create the default TraceLabAuth instance.

    Returns:
        Singleton TraceLabAuth instance configured from environment.
    """
    global _default_auth
    if _default_auth is None:
        _default_auth = TraceLabAuth()
    return _default_auth


def reset_default_auth() -> None:
    """Reset the default auth instance (useful for testing)."""
    global _default_auth
    _default_auth = None


__all__ = [
    "TraceLabAuth",
    "TraceLabAuthError",
    "TokenInfo",
    "get_default_auth",
    "is_secure_tracelab_url",
    "reset_default_auth",
]
