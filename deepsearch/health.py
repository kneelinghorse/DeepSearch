"""Health check utilities for DeepSearch dependencies.

Provides functions to verify that all required services are available
and properly configured before running research missions.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Status of a dependency check."""

    name: str
    healthy: bool
    message: str
    latency_ms: Optional[float] = None


def check_ollama() -> HealthStatus:
    """Check if Ollama service is running and accessible.

    Returns:
        HealthStatus with Ollama availability info.
    """
    try:
        import httpx
    except ImportError:
        return HealthStatus(
            name="Ollama",
            healthy=False,
            message="httpx not installed - run: pip install httpx",
        )

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    start = time.time()

    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            model_names = [m.get("name", "unknown") for m in models[:3]]
            if len(models) > 3:
                model_names.append(f"+{len(models) - 3} more")
            return HealthStatus(
                name="Ollama",
                healthy=True,
                message=f"{len(models)} models available ({', '.join(model_names)})",
                latency_ms=latency,
            )
        else:
            return HealthStatus(
                name="Ollama",
                healthy=False,
                message=f"HTTP {resp.status_code}",
            )

    except httpx.ConnectError:
        return HealthStatus(
            name="Ollama",
            healthy=False,
            message=f"Connection refused at {base_url} - is Ollama running?",
        )
    except httpx.TimeoutException:
        return HealthStatus(
            name="Ollama",
            healthy=False,
            message=f"Connection timeout at {base_url}",
        )
    except Exception as e:
        return HealthStatus(
            name="Ollama",
            healthy=False,
            message=f"Connection failed: {e}",
        )


def check_tracelab() -> HealthStatus:
    """Check whether TraceLab resolves the configured identity as ``service``.

    Returns:
        HealthStatus with TraceLab connectivity info.
    """
    from deepsearch.tracelab.auth import TraceLabAuth, TraceLabAuthError

    auth = TraceLabAuth()
    if not auth.is_configured:
        return HealthStatus(
            name="TraceLab",
            healthy=False,
            message=(
                "TRACELAB_BASE_URL and TRACELAB_API_KEY (or explicit JWT fallback) "
                "not configured"
            ),
        )

    start = time.time()

    try:
        auth.require_service_role()
        latency = (time.time() - start) * 1000
        return HealthStatus(
            name="TraceLab",
            healthy=True,
            message="Service principal verified",
            latency_ms=latency,
        )
    except TraceLabAuthError as exc:
        return HealthStatus(
            name="TraceLab",
            healthy=False,
            message=(
                f"Service authentication failed (HTTP {exc.status_code})"
                if exc.status_code is not None
                else "Service authentication failed"
            ),
        )


def check_tavily() -> HealthStatus:
    """Check if Tavily API key is configured (format validation only).

    Note: Does not make a real request to preserve API quota.

    Returns:
        HealthStatus with Tavily configuration info.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key:
        return HealthStatus(
            name="Tavily",
            healthy=False,
            message="TAVILY_API_KEY not configured",
        )

    if not api_key.startswith("tvly-"):
        return HealthStatus(
            name="Tavily",
            healthy=False,
            message="Invalid API key format (should start with 'tvly-')",
        )

    # Don't make a real request to preserve quota
    return HealthStatus(
        name="Tavily",
        healthy=True,
        message="API key configured (format valid)",
    )


def check_embedding_model() -> HealthStatus:
    """Check if the embedding model is available in Ollama.

    Returns:
        HealthStatus with embedding model availability info.
    """
    try:
        import httpx
    except ImportError:
        return HealthStatus(
            name="Embeddings",
            healthy=False,
            message="httpx not installed",
        )

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("DEEPSEARCH_EMBEDDING_MODEL", "nomic-embed-text")

    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return HealthStatus(
                name="Embeddings",
                healthy=False,
                message="Could not check Ollama models",
            )

        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]

        # Check if embedding model is available
        if model in model_names or any(model in name for name in model_names):
            return HealthStatus(
                name="Embeddings",
                healthy=True,
                message=f"{model} model available",
            )
        else:
            return HealthStatus(
                name="Embeddings",
                healthy=False,
                message=f"{model} not found - run: ollama pull {model}",
            )

    except Exception as e:
        return HealthStatus(
            name="Embeddings",
            healthy=False,
            message=f"Check failed: {e}",
        )


def overall_health() -> Dict[str, any]:
    """Check all dependencies and return summary.

    Returns:
        Dictionary with overall health status and individual check results.
    """
    checks = {
        "ollama": check_ollama(),
        "tracelab": check_tracelab(),
        "tavily": check_tavily(),
        "embeddings": check_embedding_model(),
    }

    all_healthy = all(c.healthy for c in checks.values())

    # Critical and core depend on configured backends, not hardcoded services
    import os

    from deepsearch.config import _resolve_llm_backend

    llm_backend = _resolve_llm_backend(os.getenv("LLM_BACKEND"))
    search_backend = os.getenv("SEARCH_BACKEND", "google").lower()

    # Critical = LLM backend must work
    # Ollama only required when LLM_BACKEND=ollama
    # Gemini uses API key (will fail on API call if invalid, no pre-check needed)
    critical_healthy = True
    if llm_backend == "ollama":
        critical_healthy = checks["ollama"].healthy

    # Core = LLM + Search backend
    # Tavily only required when SEARCH_BACKEND=tavily
    # Google search uses Gemini API (same as LLM, no separate check)
    core_healthy = critical_healthy
    if search_backend == "tavily":
        core_healthy = core_healthy and checks["tavily"].healthy

    return {
        "healthy": all_healthy,
        "critical": critical_healthy,
        "core": core_healthy,
        "checks": {
            name: {
                "healthy": status.healthy,
                "message": status.message,
                "latency_ms": status.latency_ms,
            }
            for name, status in checks.items()
        },
    }


def print_health_report() -> bool:
    """Print formatted health report to stdout.

    Returns:
        True if all systems healthy, False otherwise.
    """
    result = overall_health()

    print("\nDeepSearch Health Check")
    print("=" * 50)

    for name, status in result["checks"].items():
        icon = "\u2705" if status["healthy"] else "\u274c"
        latency = f" ({status['latency_ms']:.0f}ms)" if status.get("latency_ms") else ""
        print(f"{icon} {name.capitalize()}: {status['message']}{latency}")

    print("=" * 50)

    if result["healthy"]:
        print("\u2705 All systems healthy - ready for research")
    elif result["core"]:
        print("\u26a0\ufe0f  Core systems healthy (TraceLab unavailable - uploads disabled)")
    elif result["critical"]:
        print("\u26a0\ufe0f  Degraded - some features unavailable")
    else:
        print("\u274c Critical dependencies unavailable - cannot run research")

    print()
    return result["healthy"]


__all__ = [
    "HealthStatus",
    "check_ollama",
    "check_tracelab",
    "check_tavily",
    "check_embedding_model",
    "overall_health",
    "print_health_report",
]
