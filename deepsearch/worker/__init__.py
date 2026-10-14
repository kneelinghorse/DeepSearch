"""DeepSearch Worker package for TraceLab integration.

Imports are lazy to allow individual modules (e.g., writer) to be
imported without requiring all worker dependencies (asyncpg, aiohttp).
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy-import worker submodules on first access."""
    _imports = {
        # Database
        "DatabaseClient": ".database",
        "DatabaseConfig": ".database",
        "DatabaseError": ".database",
        # Health
        "HealthServer": ".health",
        # Poller
        "MissionPoller": ".poller",
        "MissionRow": ".poller",
        "PollerError": ".poller",
        "PollerStoppedError": ".poller",
        "LeaseAttempt": ".lease",
        "LeasePolicy": ".lease",
        # Converter
        "convert_row_to_state": ".converter",
        "ConversionError": ".converter",
        "DEFAULT_MAX_LOOPS": ".converter",
        "DEFAULT_MIN_LOOPS": ".converter",
        "DEFAULT_TOKEN_LIMIT": ".converter",
        "DEFAULT_DELIVERABLE_FORMAT": ".converter",
        # Writer
        "ResultWriter": ".writer",
        "WriterError": ".writer",
        "StaleLeaseError": ".writer",
        "ExecutionMetadata": ".writer",
        "build_execution_metadata": ".writer",
        # Worker
        "DeepSearchWorker": ".main",
        "WorkerError": ".main",
        "LeaseLostError": ".main",
    }

    if name in _imports:
        import importlib
        module = importlib.import_module(_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatabaseClient",
    "DatabaseConfig",
    "DatabaseError",
    "HealthServer",
    "MissionPoller",
    "MissionRow",
    "PollerError",
    "PollerStoppedError",
    "LeaseAttempt",
    "LeasePolicy",
    "convert_row_to_state",
    "ConversionError",
    "DEFAULT_MAX_LOOPS",
    "DEFAULT_MIN_LOOPS",
    "DEFAULT_TOKEN_LIMIT",
    "DEFAULT_DELIVERABLE_FORMAT",
    "ResultWriter",
    "WriterError",
    "StaleLeaseError",
    "ExecutionMetadata",
    "build_execution_metadata",
    "DeepSearchWorker",
    "WorkerError",
    "LeaseLostError",
]
