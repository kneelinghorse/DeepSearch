"""HTTP health endpoint for DeepSearch worker.

Provides a lightweight HTTP server that exposes worker health status
for Railway orchestration and monitoring systems.

Environment Variables:
    HEALTH_PORT: HTTP port for health endpoint (default: 8080)
"""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import Any, Callable, Dict, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


class HealthServer:
    """Lightweight HTTP server for health checks.

    Runs alongside the main worker to expose health and stats
    via HTTP endpoints for orchestration platforms.

    Usage:
        health = HealthServer(stats_provider=worker.get_health_stats)
        await health.start()
        # ... worker runs ...
        await health.stop()
    """

    def __init__(
        self,
        stats_provider: Callable[[], Dict[str, Any]],
        port: int = 8080,
        host: str = "0.0.0.0",
    ) -> None:
        """Initialize the health server.

        Args:
            stats_provider: Callable that returns current worker stats dict
            port: HTTP port to listen on (default: 8080)
            host: Host to bind to (default: 0.0.0.0)
        """
        self._stats_provider = stats_provider
        self._port = port
        self._host = host
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._started_at: Optional[float] = None

    async def start(self) -> None:
        """Start the health HTTP server.

        Should be called before starting the worker polling loop.
        """
        self._started_at = time.time()

        self._app = web.Application()
        self._app.router.add_get("/health", self._health_handler)
        self._app.router.add_get("/", self._root_handler)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        logger.info(
            "Health server started on http://%s:%d/health",
            self._host,
            self._port,
        )

    async def stop(self) -> None:
        """Stop the health HTTP server gracefully."""
        if self._runner:
            await self._runner.cleanup()
            logger.info("Health server stopped")

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Handle GET /health requests.

        Returns:
            200 OK with JSON stats body
        """
        stats = self._stats_provider()

        # Add uptime calculation
        uptime_seconds = 0.0
        if self._started_at:
            uptime_seconds = time.time() - self._started_at

        ready = stats.get("ready", True) is not False
        response_data = {
            "status": "healthy" if ready else "unhealthy",
            "uptime_seconds": round(uptime_seconds, 2),
            **stats,
        }

        return web.json_response(
            response_data,
            status=HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
        )

    async def _root_handler(self, request: web.Request) -> web.Response:
        """Handle GET / requests with basic info."""
        return web.json_response(
            {
                "service": "deepsearch-worker",
                "health_endpoint": "/health",
            },
            status=HTTPStatus.OK,
        )

    @property
    def port(self) -> int:
        """Get the configured port."""
        return self._port

    @property
    def is_running(self) -> bool:
        """Check if server is currently running."""
        return self._site is not None and self._runner is not None


__all__ = ["HealthServer"]
