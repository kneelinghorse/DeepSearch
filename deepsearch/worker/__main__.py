"""Entry point for DeepSearch worker: python -m deepsearch.worker

Usage:
    python -m deepsearch.worker

Environment Variables:
    DATABASE_URL: PostgreSQL connection URL (required)
    WORKER_POLL_INTERVAL: Seconds between polls (default: 5.0)
    WORKER_LOG_LEVEL: Logging level (default: INFO)
    WORKER_LOG_FORMAT: 'json' for structured logging (default: text)
    LLM_BACKEND: Shared compiler/synthesis/critique provider (deepseek or openai)
    DEEPSEEK_MODEL: deepseek-flash by default; deepseek-v4-pro for explicit rollback
    OPENAI_MODEL: gpt-5.6-luna when LLM_BACKEND=openai
    DEEPSEEK_API_KEY / OPENAI_API_KEY: Credential for the selected provider
    HEALTH_PORT: HTTP port for health endpoint (default: 8080)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from copy import copy
from datetime import datetime, timezone
from typing import Any, Dict

from deepsearch.error_safety import (
    SafeErrorDetails,
    safe_exception_details,
    safe_exception_text,
)


def _record_error_details(record: logging.LogRecord) -> SafeErrorDetails | None:
    """Project ``exc_info`` without formatting its message or traceback."""

    if record.exc_info and isinstance(record.exc_info[1], BaseException):
        return safe_exception_details(record.exc_info[1])
    return None


def _safe_record_message(record: logging.LogRecord) -> str:
    """Format a log message without stringifying exception arguments."""

    if _record_error_details(record) is not None:
        return "Unhandled exception"
    if isinstance(record.msg, BaseException):
        return safe_exception_text(record.msg)
    if not record.args:
        return str(record.msg)
    projected = copy(record)
    if isinstance(record.args, dict):
        projected.args = {
            key: (
                safe_exception_text(value)
                if isinstance(value, BaseException)
                else value
            )
            for key, value in record.args.items()
        }
    else:
        projected.args = tuple(
            safe_exception_text(value) if isinstance(value, BaseException) else value
            for value in record.args
        )
    return projected.getMessage()


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for Railway/cloud deployments."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _safe_record_message(record),
        }

        error_details = _record_error_details(record)
        if error_details is not None:
            log_entry.update(error_details.to_dict())

        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format a record without allowing ``exc_info`` to append a traceback."""

        projected = copy(record)
        projected.msg = _safe_record_message(record)
        projected.args = ()
        projected.exc_info = None
        projected.exc_text = None
        projected.stack_info = None
        error_details = _record_error_details(record)
        if error_details is not None:
            projected.msg = f"{projected.msg}: {error_details.to_text()}"
        return super().format(projected)


def configure_logging() -> None:
    """Configure logging based on environment variables."""
    log_level = os.getenv("WORKER_LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("WORKER_LOG_FORMAT", "text").lower()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level, logging.INFO))

    # Set formatter based on environment
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)

    # Reduce noise from third-party loggers
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main() -> int:
    """Main entry point for the worker."""
    configure_logging()

    logger = logging.getLogger(__name__)
    logger.info("DeepSearch Worker starting")

    # Log key package versions so Railway logs confirm what's actually installed
    try:
        import importlib.metadata as meta
        for pkg in ("langchain-core", "langgraph", "typing_extensions"):
            try:
                logger.info("Package: %s==%s", pkg, meta.version(pkg))
            except meta.PackageNotFoundError:
                logger.warning("Package not found: %s", pkg)
    except Exception:
        pass

    try:
        from .main import DeepSearchWorker

        worker = DeepSearchWorker()

        await worker.run()
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0

    except Exception as exc:
        error_details = safe_exception_details(exc)
        logger.error("Worker failed: %s", error_details.to_text())
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
