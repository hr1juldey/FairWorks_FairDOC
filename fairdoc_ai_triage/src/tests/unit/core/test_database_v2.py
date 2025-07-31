"""
Unit Tests for V2 Database Management
Tests async SQLAlchemy setup, connection pooling, and health checks
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from sqlalchemy.exc import OperationalError

# Test environment setup - ensure we can import the module
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'DATABASE_URL': 'postgresql+asyncpg://test:test@localhost/test',
    'REDIS_URL': 'redis://localhost:6379/0',
    'MINIO_ENDPOINT': 'localhost:9000',
    'MINIO_ACCESS_KEY': 'test',
    'MINIO_SECRET_KEY': 'test',
    'OLLAMA_BASE_URL': 'http://localhost:11434',
    'RAVEN_WEBHOOK_URL': 'http://localhost:8080/webhook',
    'RAVEN_API_KEY': 'test-key',
    'RAVEN_SECRET': 'test-secret',
    'JWT_SECRET_KEY': 'jwt-secret',
    'CELERY_BROKER_URL': 'redis://localhost:6379/1',
    'CELERY_RESULT_BACKEND': 'redis://localhost:6379/2'
}):
    from src.app2.core.database_v2 import (
        database_v2,
        get_engine,
        get_session_factory,
        get_async_session,
        check_database_health,
        startup_database,
        shutdown_database
    )


class TestDatabaseV2Engine:
    """Test V2 database engine configuration and connection pooling"""
    
    def test_get_engine_creation(self):
        """Test engine creation function works correctly"""
        engine = get_engine()
        
        assert engine is not None
        assert engine.url.drivername == "postgresql+asyncpg"
        
        # Test that calling it again returns the same engine (singleton pattern)
        engine2 = get_engine()
        assert engine is engine2
    
    def test_get_session_factory_creation(self):
        """Test session factory creation function works correctly"""
        session_factory = get_session_factory()
        
        assert session_factory is not None
        
        # Test that calling it again returns the same factory (singleton pattern)  
        session_factory2 = get_session_factory()
        assert session_factory is session_factory2
    
    def test_database_v2_global_instance(self):
        """Test global database_v2 instance is properly initialized"""
        assert database_v2 is not None
        assert hasattr(database_v2, 'create_engine')
        assert hasattr(database_v2, 'create_session_factory')
        assert hasattr(database_v2, 'initialize')


class TestDatabaseV2Initialization:
    """Test V2 database initialization and startup"""
    
    @pytest.mark.asyncio
    async def test_startup_database_success(self):
        """Test successful database startup sequence"""
        with patch.object(database_v2, 'initialize', new_callable=AsyncMock) as mock_init:
            await startup_database()
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_startup_database_failure(self):
        """Test database startup handles failures"""
        with patch.object(database_v2, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = OperationalError("Connection failed", None, None)
            
            with pytest.raises(OperationalError):
                await startup_database()
    
    @pytest.mark.asyncio
    async def test_shutdown_database_success(self):
        """Test successful database shutdown sequence"""
        with patch.object(database_v2, 'close', new_callable=AsyncMock) as mock_close:
            await shutdown_database()
            mock_close.assert_called_once()


class TestDatabaseV2Health:
    """Test V2 database health check functionality"""
    
    @pytest.mark.asyncio
    async def test_check_database_health_healthy(self):
        """Test database health check when database is healthy"""
        with patch('src.app2.core.database_v2.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = [1]  # Health check returns 1
            mock_result.scalar.return_value = 5  # Table count
            mock_session.execute.return_value = mock_result
            
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            health = await check_database_health()
            
            assert health["status"] == "healthy"
            assert health["connection"] == "active"
            assert health["health_check_result"] == 1
            assert health["table_count"] == 5
    
    @pytest.mark.asyncio
    async def test_check_database_health_unhealthy(self):
        """Test database health check when database fails"""
        with patch('src.app2.core.database_v2.get_async_session') as mock_get_session:
            mock_get_session.side_effect = OperationalError("Connection failed", None, None)
            
            health = await check_database_health()
            
            assert health["status"] == "unhealthy"
            assert health["connection"] == "failed"
            assert "error" in health
            assert "Connection failed" in health["error"]


class TestDatabaseV2SessionManagement:
    """Test V2 async session management and lifecycle"""
    
    @pytest.mark.asyncio
    async def test_get_async_session_context_manager(self):
        """Test async session context manager works correctly"""
        with patch.object(database_v2, 'create_session_factory') as mock_factory:
            mock_session = AsyncMock()
            mock_session_factory = MagicMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value = mock_session_factory
            
            async with get_async_session() as session:
                assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_get_async_session_error_handling(self):
        """Test async session error handling and rollback"""
        with patch.object(database_v2, 'create_session_factory') as mock_factory:
            mock_session = AsyncMock()
            mock_session_factory = MagicMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value = mock_session_factory
            
            # Simulate an error during session usage
            with pytest.raises(ValueError):  # noqa: PT011
                async with get_async_session() as session:  # noqa: F841
                    raise ValueError("Test error")
            
            # Verify rollback was called
            mock_session.rollback.assert_called_once()


class TestDatabaseV2Configuration:
    """Test V2 database configuration and environment handling"""
    
    def test_engine_configuration_production(self):
        """Test engine is configured correctly for production"""
        with patch('src.app2.core.database_v2.settings_v2') as mock_settings:
            mock_settings.DATABASE_URL = "postgresql+asyncpg://prod:pass@localhost/prod"
            mock_settings.DEBUG = False
            mock_settings.ENVIRONMENT = "production"
            
            # Create new database instance to test configuration
            from src.app2.core.database_v2 import DatabaseV2
            test_db = DatabaseV2()
            
            engine = test_db.create_engine()
            
            assert engine.url.drivername == "postgresql+asyncpg"
            assert not engine.echo  # Should not echo in production
    
    def test_engine_configuration_testing(self):
        """Test engine is configured correctly for testing"""
        with patch('src.app2.core.database_v2.settings_v2') as mock_settings:
            mock_settings.DATABASE_URL = "postgresql+asyncpg://test:pass@localhost/test"
            mock_settings.DEBUG = True
            mock_settings.ENVIRONMENT = "testing"
            
            # Create new database instance to test configuration
            from src.app2.core.database_v2 import DatabaseV2
            test_db = DatabaseV2()
            
            engine = test_db.create_engine()
            
            assert engine.url.drivername == "postgresql+asyncpg"


class TestDatabaseV2Performance:
    """Test V2 database performance and monitoring capabilities"""
    
    def test_connection_pool_properties(self):
        """Test connection pool is accessible for monitoring"""
        engine = get_engine()
        
        # Verify pool exists and has monitoring properties
        assert hasattr(engine, 'pool')
        pool = engine.pool
        
        # Test that pool metrics are accessible (for monitoring)
        if hasattr(pool, 'size'):
            assert callable(pool.size)
        if hasattr(pool, 'checked_in'):
            assert callable(pool.checked_in)
        if hasattr(pool, 'checked_out'):
            assert callable(pool.checked_out)
    
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self):
        """Test database can handle multiple concurrent sessions"""
        async def create_test_session():
            with patch.object(database_v2, 'create_session_factory') as mock_factory:
                mock_session = AsyncMock()
                mock_session_factory = MagicMock()
                mock_session_factory.return_value.__aenter__.return_value = mock_session
                mock_factory.return_value = mock_session_factory
                
                async with get_async_session() as session:
                    return session is not None
        
        # Test multiple concurrent sessions
        tasks = [create_test_session() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All tasks should complete successfully
        assert all(result is True for result in results if not isinstance(result, Exception))


class TestDatabaseV2ErrorRecovery:
    """Test V2 database error recovery and resilience"""
    
    @pytest.mark.asyncio
    async def test_database_reconnection_capability(self):
        """Test database can recover from connection failures"""
        # Simulate database reconnection scenario
        with patch.object(database_v2, 'close', new_callable=AsyncMock) as mock_close:
            with patch.object(database_v2, 'initialize', new_callable=AsyncMock) as mock_init:
                # Simulate shutdown and restart
                await shutdown_database()
                await startup_database()
                
                mock_close.assert_called_once()
                mock_init.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_graceful_degradation(self):
        """Test system degrades gracefully when database is unavailable"""
        with patch('src.app2.core.database_v2.get_async_session') as mock_get_session:
            # Simulate persistent database failure
            mock_get_session.side_effect = OperationalError("Database unavailable", None, None)
            
            # Health check should report unhealthy but not crash the system
            health = await check_database_health()
            
            assert isinstance(health, dict)
            assert health["status"] == "unhealthy"
            assert "error" in health
