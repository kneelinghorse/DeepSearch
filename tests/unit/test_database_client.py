"""Unit tests for PostgreSQL database client."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepsearch.worker.database import (
    DatabaseClient,
    DatabaseConfig,
    DatabaseError,
    ConnectionError,
    PoolExhaustedError,
    HealthStatus,
    check_database_health,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = DatabaseConfig(
            url="postgresql://user:pass@localhost:5432/testdb"
        )
        assert config.url == "postgresql://user:pass@localhost:5432/testdb"
        assert config.min_pool_size == 2
        assert config.max_pool_size == 10
        assert config.connection_timeout == 30.0
        assert config.command_timeout == 60.0
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = DatabaseConfig(
            url="postgresql://localhost/db",
            min_pool_size=5,
            max_pool_size=20,
            connection_timeout=60.0,
            command_timeout=120.0,
            retry_attempts=5,
            retry_delay=2.0,
        )
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20
        assert config.connection_timeout == 60.0
        assert config.command_timeout == 120.0
        assert config.retry_attempts == 5
        assert config.retry_delay == 2.0

    def test_from_env_success(self):
        """Test config from environment variables."""
        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://testuser:testpass@dbhost:5433/testdb",
            "DB_MIN_POOL_SIZE": "4",
            "DB_MAX_POOL_SIZE": "15",
            "DB_CONNECTION_TIMEOUT": "45",
            "DB_COMMAND_TIMEOUT": "90",
        }):
            config = DatabaseConfig.from_env()
            assert config.url == "postgresql://testuser:testpass@dbhost:5433/testdb"
            assert config.min_pool_size == 4
            assert config.max_pool_size == 15
            assert config.connection_timeout == 45.0
            assert config.command_timeout == 90.0

    def test_from_env_missing_url(self):
        """Test error when DATABASE_URL not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure DATABASE_URL is not set
            os.environ.pop("DATABASE_URL", None)
            with pytest.raises(DatabaseError) as exc:
                DatabaseConfig.from_env()
            assert "DATABASE_URL" in str(exc.value)

    def test_from_env_custom_key(self):
        """Test config with custom env var key."""
        with patch.dict(os.environ, {
            "CUSTOM_DB_URL": "postgresql://custom@host/db"
        }):
            config = DatabaseConfig.from_env("CUSTOM_DB_URL")
            assert config.url == "postgresql://custom@host/db"

    def test_host_extraction(self):
        """Test host extraction from URL."""
        config = DatabaseConfig(url="postgresql://user:pass@myhost.example.com:5432/db")
        assert config.host == "myhost.example.com"

    def test_port_extraction(self):
        """Test port extraction from URL."""
        config = DatabaseConfig(url="postgresql://user:pass@localhost:5433/db")
        assert config.port == 5433

    def test_port_default(self):
        """Test default port when not specified."""
        config = DatabaseConfig(url="postgresql://user:pass@localhost/db")
        assert config.port == 5432

    def test_database_extraction(self):
        """Test database name extraction from URL."""
        config = DatabaseConfig(url="postgresql://user:pass@localhost:5432/mydb")
        assert config.database == "mydb"

    def test_database_default(self):
        """Test default database when not specified."""
        config = DatabaseConfig(url="postgresql://localhost")
        assert config.database == "postgres"


class TestHealthStatus:
    """Tests for HealthStatus."""

    def test_healthy_status(self):
        """Test healthy status creation."""
        status = HealthStatus(
            healthy=True,
            message="OK",
            latency_ms=1.5,
            pool_size=5,
            pool_free=3
        )
        assert status.healthy is True
        assert status.message == "OK"
        assert status.latency_ms == 1.5
        assert status.pool_size == 5
        assert status.pool_free == 3

    def test_unhealthy_status(self):
        """Test unhealthy status creation."""
        status = HealthStatus(
            healthy=False,
            message="Connection failed"
        )
        assert status.healthy is False
        assert status.latency_ms is None
        assert status.pool_size is None


class TestDatabaseClient:
    """Tests for DatabaseClient."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DatabaseConfig(
            url="postgresql://test:test@localhost:5432/testdb",
            retry_attempts=2,
            retry_delay=0.01,  # Fast retries for tests
        )

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = MagicMock()
        pool.get_size.return_value = 5
        pool.get_idle_size.return_value = 3
        pool.close = AsyncMock()
        pool.release = AsyncMock()

        # Mock connection
        conn = AsyncMock()
        conn.fetchval.return_value = 1
        conn.fetchrow.return_value = {"id": 1, "name": "test"}
        conn.fetch.return_value = [{"id": 1}, {"id": 2}]
        conn.execute.return_value = "INSERT 0 1"

        # Create async context manager mock for acquire
        async_context = AsyncMock()
        async_context.__aenter__.return_value = conn
        async_context.__aexit__.return_value = None
        pool.acquire.return_value = async_context

        return pool

    def test_client_init(self, config):
        """Test client initialization."""
        client = DatabaseClient(config)
        assert client.config == config
        assert not client.is_connected
        assert client._pool is None

    @pytest.mark.asyncio
    async def test_connect_success(self, config, mock_pool):
        """Test successful connection."""
        client = DatabaseClient(config)

        async def create_pool_mock(*args, **kwargs):
            return mock_pool

        with patch("deepsearch.worker.database.asyncpg.create_pool", side_effect=create_pool_mock):
            await client.connect()

        assert client.is_connected
        assert client._pool == mock_pool

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, config, mock_pool):
        """Test connect when already connected."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        # Should return early without calling create_pool
        with patch("deepsearch.worker.database.asyncpg.create_pool") as mock_create:
            await client.connect()
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_retry_on_failure(self, config):
        """Test connection retry on transient failures."""
        client = DatabaseClient(config)

        # Create properly mocked pool with async context manager
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchval.return_value = 1
        async_context = AsyncMock()
        async_context.__aenter__.return_value = conn
        async_context.__aexit__.return_value = None
        pool.acquire.return_value = async_context

        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("Connection refused")
            return pool

        with patch("deepsearch.worker.database.asyncpg.create_pool", side_effect=failing_then_success):
            await client.connect()

        assert client.is_connected
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_connect_all_retries_fail(self, config):
        """Test connection failure after all retries."""
        client = DatabaseClient(config)

        async def always_fail(*args, **kwargs):
            raise OSError("Connection refused")

        with patch("deepsearch.worker.database.asyncpg.create_pool", side_effect=always_fail):
            with pytest.raises(ConnectionError) as exc:
                await client.connect()
            assert "Failed to connect after 2 attempts" in str(exc.value)

    @pytest.mark.asyncio
    async def test_close(self, config, mock_pool):
        """Test closing connection."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        await client.close()

        assert not client.is_connected
        assert client._pool is None
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_not_connected(self, config):
        """Test closing when not connected."""
        client = DatabaseClient(config)
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, config, mock_pool):
        """Test async context manager."""
        with patch("deepsearch.worker.database.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)):
            async with DatabaseClient(config) as client:
                assert client.is_connected

        assert not client.is_connected
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_factory(self, config, mock_pool):
        """Test create factory method."""
        with patch("deepsearch.worker.database.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)):
            client = await DatabaseClient.create(config)
            assert client.is_connected
            await client.close()

    @pytest.mark.asyncio
    async def test_fetchone(self, config, mock_pool):
        """Test fetchone method."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        result = await client.fetchone("SELECT * FROM test WHERE id = $1", 1)
        assert result == {"id": 1, "name": "test"}

    @pytest.mark.asyncio
    async def test_fetchall(self, config, mock_pool):
        """Test fetchall method."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        result = await client.fetchall("SELECT * FROM test")
        assert result == [{"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_fetchval(self, config, mock_pool):
        """Test fetchval method."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        result = await client.fetchval("SELECT COUNT(*) FROM test")
        assert result == 1

    @pytest.mark.asyncio
    async def test_execute(self, config, mock_pool):
        """Test execute method."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        result = await client.execute("INSERT INTO test VALUES ($1)", "value")
        assert result == "INSERT 0 1"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, config, mock_pool):
        """Test health check when healthy."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        status = await client.health_check()

        assert status.healthy is True
        assert "healthy" in status.message.lower()
        assert status.latency_ms is not None
        assert status.pool_size == 5
        assert status.pool_free == 3

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, config):
        """Test health check when not connected."""
        client = DatabaseClient(config)

        status = await client.health_check()

        assert status.healthy is False
        assert "not connected" in status.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_query_fails(self, config, mock_pool):
        """Test health check when query fails."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        # Make the acquire raise an exception
        mock_pool.acquire.side_effect = Exception("Query failed")

        status = await client.health_check()

        assert status.healthy is False
        assert "failed" in status.message.lower()

    @pytest.mark.asyncio
    async def test_reconnect(self, config, mock_pool):
        """Test reconnect method."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        # Create properly mocked new pool
        new_pool = MagicMock()
        new_pool.close = AsyncMock()
        conn = AsyncMock()
        conn.fetchval.return_value = 1
        async_context = AsyncMock()
        async_context.__aenter__.return_value = conn
        async_context.__aexit__.return_value = None
        new_pool.acquire.return_value = async_context

        with patch("deepsearch.worker.database.asyncpg.create_pool", new=AsyncMock(return_value=new_pool)):
            await client.reconnect()

        mock_pool.close.assert_called_once()
        assert client._pool == new_pool

    @pytest.mark.asyncio
    async def test_ensure_connected_reconnects(self, config, mock_pool):
        """Test that operations reconnect if disconnected."""
        client = DatabaseClient(config)
        client._connected = False

        with patch("deepsearch.worker.database.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)):
            await client._ensure_connected()

        assert client.is_connected

    @pytest.mark.asyncio
    async def test_acquire_timeout(self, config, mock_pool):
        """Test acquire connection with timeout."""
        client = DatabaseClient(config)
        client._connected = True
        client._pool = mock_pool

        async def slow_acquire():
            await asyncio.sleep(10)
            return AsyncMock()

        mock_pool.acquire = slow_acquire

        with pytest.raises(PoolExhaustedError):
            await client.acquire(timeout=0.01)


class TestCheckDatabaseHealth:
    """Tests for check_database_health helper function."""

    @pytest.mark.asyncio
    async def test_health_check_missing_url(self):
        """Test health check when DATABASE_URL not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DATABASE_URL", None)
            result = await check_database_health()

        assert result["healthy"] is False
        assert result["status"] == "error"
        assert "DATABASE_URL" in result["message"]

    @pytest.mark.asyncio
    async def test_health_check_connection_success(self):
        """Test health check with successful connection."""
        # Create properly mocked pool
        mock_pool = MagicMock()
        mock_pool.get_size.return_value = 5
        mock_pool.get_idle_size.return_value = 3
        mock_pool.close = AsyncMock()

        conn = AsyncMock()
        conn.fetchval.return_value = 1
        async_context = AsyncMock()
        async_context.__aenter__.return_value = conn
        async_context.__aexit__.return_value = None
        mock_pool.acquire.return_value = async_context

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test@localhost/db"}):
            with patch("deepsearch.worker.database.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)):
                result = await check_database_health()

        assert result["healthy"] is True
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_connection_failure(self):
        """Test health check with connection failure."""
        async def fail_connect(*args, **kwargs):
            raise OSError("Connection refused")

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test@localhost/db"}):
            with patch("deepsearch.worker.database.asyncpg.create_pool", side_effect=fail_connect):
                result = await check_database_health()

        assert result["healthy"] is False
        assert result["status"] == "error"


class TestExceptions:
    """Tests for custom exceptions."""

    def test_database_error(self):
        """Test DatabaseError exception."""
        error = DatabaseError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_connection_error_inheritance(self):
        """Test ConnectionError inherits from DatabaseError."""
        error = ConnectionError("Connection failed")
        assert isinstance(error, DatabaseError)
        assert str(error) == "Connection failed"

    def test_pool_exhausted_error_inheritance(self):
        """Test PoolExhaustedError inherits from DatabaseError."""
        error = PoolExhaustedError("Pool exhausted")
        assert isinstance(error, DatabaseError)
        assert str(error) == "Pool exhausted"
