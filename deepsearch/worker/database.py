"""PostgreSQL client with connection pooling for TraceLab worker integration.

This module provides an async PostgreSQL client using asyncpg with:
- Connection pooling (configurable size)
- Automatic reconnection on connection loss
- Health check endpoint
- Context manager for safe connection handling
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

try:
    import asyncpg
    from asyncpg import Pool, Connection
except ImportError:
    asyncpg = None  # type: ignore[assignment]
    Pool = None  # type: ignore[assignment,misc]
    Connection = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Raised when connection to database fails."""
    pass


class PoolExhaustedError(DatabaseError):
    """Raised when connection pool is exhausted."""
    pass


@dataclass
class DatabaseConfig:
    """Configuration for database connection.

    Attributes:
        url: PostgreSQL connection URL (from DATABASE_URL env var)
        min_pool_size: Minimum number of connections in pool (default: 2)
        max_pool_size: Maximum number of connections in pool (default: 10)
        connection_timeout: Timeout for acquiring connection in seconds (default: 30)
        command_timeout: Timeout for executing commands in seconds (default: 60)
        retry_attempts: Number of reconnection attempts on failure (default: 3)
        retry_delay: Delay between reconnection attempts in seconds (default: 1.0)
    """
    url: str
    min_pool_size: int = 2
    max_pool_size: int = 10
    connection_timeout: float = 30.0
    command_timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls, url_key: str = "DATABASE_URL") -> "DatabaseConfig":
        """Create config from environment variables.

        Args:
            url_key: Environment variable name for database URL

        Returns:
            DatabaseConfig with settings from environment

        Raises:
            DatabaseError: If DATABASE_URL is not set
        """
        url = os.getenv(url_key)
        if not url:
            raise DatabaseError(f"{url_key} environment variable not set")

        return cls(
            url=url,
            min_pool_size=int(os.getenv("DB_MIN_POOL_SIZE", "2")),
            max_pool_size=int(os.getenv("DB_MAX_POOL_SIZE", "10")),
            connection_timeout=float(os.getenv("DB_CONNECTION_TIMEOUT", "30")),
            command_timeout=float(os.getenv("DB_COMMAND_TIMEOUT", "60")),
        )

    @property
    def host(self) -> str:
        """Extract host from URL."""
        parsed = urlparse(self.url)
        return parsed.hostname or "localhost"

    @property
    def port(self) -> int:
        """Extract port from URL."""
        parsed = urlparse(self.url)
        return parsed.port or 5432

    @property
    def database(self) -> str:
        """Extract database name from URL."""
        parsed = urlparse(self.url)
        path = parsed.path.lstrip("/")
        return path if path else "postgres"


@dataclass
class HealthStatus:
    """Result of database health check."""
    healthy: bool
    message: str
    latency_ms: Optional[float] = None
    pool_size: Optional[int] = None
    pool_free: Optional[int] = None


class DatabaseClient:
    """Async PostgreSQL client with connection pooling.

    Usage:
        async with DatabaseClient.create(config) as client:
            result = await client.fetchone("SELECT 1")

    Or manually:
        client = DatabaseClient(config)
        await client.connect()
        try:
            result = await client.fetchone("SELECT 1")
        finally:
            await client.close()
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize client with config (does not connect).

        Args:
            config: Database connection configuration
        """
        self.config = config
        self._pool: Optional[Pool] = None
        self._connected = False
        self._lock = asyncio.Lock()

    @classmethod
    async def create(cls, config: DatabaseConfig) -> "DatabaseClient":
        """Factory method that creates and connects client.

        Args:
            config: Database connection configuration

        Returns:
            Connected DatabaseClient instance
        """
        client = cls(config)
        await client.connect()
        return client

    @classmethod
    async def from_env(cls, url_key: str = "DATABASE_URL") -> "DatabaseClient":
        """Create connected client from environment variables.

        Args:
            url_key: Environment variable name for database URL

        Returns:
            Connected DatabaseClient instance
        """
        config = DatabaseConfig.from_env(url_key)
        return await cls.create(config)

    async def __aenter__(self) -> "DatabaseClient":
        """Async context manager entry."""
        if not self._connected:
            await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._pool is not None

    async def connect(self) -> None:
        """Establish connection pool to database.

        Raises:
            ConnectionError: If connection fails after all retry attempts
            DatabaseError: If asyncpg is not installed
        """
        if asyncpg is None:
            raise DatabaseError(
                "asyncpg is not installed. Install it with: pip install asyncpg"
            )
        if self._connected:
            return

        async with self._lock:
            if self._connected:  # Double-check after acquiring lock
                return

            last_error: Optional[Exception] = None

            for attempt in range(1, self.config.retry_attempts + 1):
                try:
                    logger.info(
                        "Connecting to PostgreSQL at %s:%d (attempt %d/%d)",
                        self.config.host,
                        self.config.port,
                        attempt,
                        self.config.retry_attempts
                    )

                    self._pool = await asyncpg.create_pool(
                        self.config.url,
                        min_size=self.config.min_pool_size,
                        max_size=self.config.max_pool_size,
                        command_timeout=self.config.command_timeout,
                    )

                    # Test connection with simple query
                    async with self._pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")

                    self._connected = True
                    logger.info(
                        "Connected to PostgreSQL (pool size: %d-%d)",
                        self.config.min_pool_size,
                        self.config.max_pool_size
                    )
                    return

                except (asyncpg.PostgresError, OSError, asyncio.TimeoutError) as e:
                    last_error = e
                    logger.warning(
                        "Connection attempt %d failed: %s",
                        attempt,
                        str(e)
                    )

                    if attempt < self.config.retry_attempts:
                        await asyncio.sleep(self.config.retry_delay * attempt)

            raise ConnectionError(
                f"Failed to connect after {self.config.retry_attempts} attempts: {last_error}"
            )

    async def close(self) -> None:
        """Close connection pool gracefully."""
        if self._pool is not None:
            logger.info("Closing PostgreSQL connection pool")
            await self._pool.close()
            self._pool = None
            self._connected = False

    async def _ensure_connected(self) -> None:
        """Ensure we have a valid connection, reconnecting if needed."""
        if not self._connected or self._pool is None:
            await self.connect()

    async def acquire(self, timeout: Optional[float] = None) -> Connection:
        """Acquire a connection from the pool.

        Args:
            timeout: Optional timeout for acquiring connection

        Returns:
            Database connection

        Raises:
            PoolExhaustedError: If pool is exhausted and timeout expires
            ConnectionError: If not connected
        """
        await self._ensure_connected()

        if self._pool is None:
            raise ConnectionError("Not connected to database")

        try:
            conn = await asyncio.wait_for(
                self._pool.acquire(),
                timeout=timeout or self.config.connection_timeout
            )
            return conn
        except asyncio.TimeoutError:
            raise PoolExhaustedError(
                f"Could not acquire connection within {timeout or self.config.connection_timeout}s"
            )

    async def release(self, connection: Connection) -> None:
        """Release a connection back to the pool.

        Args:
            connection: Connection to release
        """
        if self._pool is not None:
            await self._pool.release(connection)

    async def execute(self, query: str, *args: Any, timeout: Optional[float] = None) -> str:
        """Execute a query without returning results.

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Optional command timeout

        Returns:
            Status string from command
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetchone(self, query: str, *args: Any, timeout: Optional[float] = None) -> Optional[asyncpg.Record]:
        """Fetch a single row from query.

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Optional command timeout

        Returns:
            Single record or None
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def fetchall(self, query: str, *args: Any, timeout: Optional[float] = None) -> list[asyncpg.Record]:
        """Fetch all rows from query.

        Args:
            query: SQL query string
            *args: Query parameters
            timeout: Optional command timeout

        Returns:
            List of records
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchval(self, query: str, *args: Any, column: int = 0, timeout: Optional[float] = None) -> Any:
        """Fetch a single value from query.

        Args:
            query: SQL query string
            *args: Query parameters
            column: Column index to return (default: 0)
            timeout: Optional command timeout

        Returns:
            Single value or None
        """
        await self._ensure_connected()

        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def health_check(self) -> HealthStatus:
        """Check database connection health.

        Returns:
            HealthStatus with connection state and latency
        """
        import time

        if not self._connected or self._pool is None:
            return HealthStatus(
                healthy=False,
                message="Not connected to database"
            )

        try:
            start = time.monotonic()

            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")

            latency_ms = (time.monotonic() - start) * 1000

            return HealthStatus(
                healthy=result == 1,
                message="Database connection healthy",
                latency_ms=round(latency_ms, 2),
                pool_size=self._pool.get_size(),
                pool_free=self._pool.get_idle_size()
            )

        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Health check failed: {str(e)}"
            )

    async def reconnect(self) -> None:
        """Force reconnection to database."""
        await self.close()
        await self.connect()


# Convenience function for health endpoint integration
async def check_database_health(url_key: str = "DATABASE_URL") -> dict[str, Any]:
    """Check database health for API health endpoint.

    Args:
        url_key: Environment variable name for database URL

    Returns:
        Dict with health status for JSON response
    """
    try:
        config = DatabaseConfig.from_env(url_key)
    except DatabaseError as e:
        return {
            "status": "error",
            "message": str(e),
            "healthy": False
        }

    client = DatabaseClient(config)
    try:
        await client.connect()
        health = await client.health_check()
        return {
            "status": "ok" if health.healthy else "error",
            "message": health.message,
            "healthy": health.healthy,
            "latency_ms": health.latency_ms,
            "pool_size": health.pool_size,
            "pool_free": health.pool_free
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "healthy": False
        }
    finally:
        await client.close()
