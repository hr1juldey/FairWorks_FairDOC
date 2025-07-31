"""
Unit Tests for V2 Dependencies Management
Tests FastAPI dependency injection, service initialization, and connection management
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from redis.exceptions import ConnectionError as RedisConnectionError

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
    from src.app2.core.dependencies_v2 import (
        get_db_session,
        init_redis_pool,
        get_redis_client,
        init_services,
        get_medical_agent,
        get_nice_lookup,
        get_conversation_queue,
        get_stakeholder_router,
        get_raven_bridge,
        cleanup_connections,
        check_system_health
    )


class TestDatabaseDependencies:
    """Test V2 database dependency injection and session management"""
    
    @pytest.mark.asyncio
    async def test_get_db_session_success(self):
        """Test successful database session creation and cleanup"""
        with patch('src.app2.core.dependencies_v2.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            mock_session_local.return_value.__aexit__.return_value = None
            
            # get_db_session() is an async generator, not a context manager
            async_gen = get_db_session()
            session = await async_gen.__anext__()
            
            # Use the session variable to avoid unused variable warning
            assert session == mock_session
            # Verify we can perform operations with the session
            assert hasattr(session, 'execute')
            
            # Clean up the generator
            try:
                await async_gen.__anext__()
            except StopAsyncIteration:
                pass
            
            # Verify session lifecycle
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_db_session_error_handling(self):
        """Test database session error handling and rollback"""
        with patch('src.app2.core.dependencies_v2.AsyncSessionLocal') as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            mock_session.commit.side_effect = OperationalError("DB Error", None, None)
            
            # Setup the generator OUTSIDE of pytest.raises()
            async_gen = get_db_session()
            session = await async_gen.__anext__()
            
            # Use session to avoid unused variable warning
            assert session is not None
            
            # ONLY the single statement that should raise the exception
            with pytest.raises(HTTPException, match="Database operation failed"):
                await async_gen.asend(None)
            
            # Verify rollback was called after the exception
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()



    
    def test_database_engine_configuration(self):
        """Test database engine is configured with correct parameters"""
        from src.app2.core.dependencies_v2 import _async_engine
        
        assert _async_engine is not None
        assert _async_engine.url.drivername == "postgresql+asyncpg"
        
        # Check connection pool settings
        pool = _async_engine.pool
        assert pool.size() == 20  # pool_size=20
        assert pool._max_overflow == 0  # max_overflow=0


class TestRedisDependencies:
    """Test V2 Redis dependency injection and connection pooling"""
    
    @pytest.mark.asyncio
    async def test_init_redis_pool_success(self):
        """Test successful Redis connection pool initialization"""
        with patch('src.app2.core.dependencies_v2.ConnectionPool') as mock_pool:
            with patch('src.app2.core.dependencies_v2.Redis') as mock_redis_class:
                mock_redis = AsyncMock()
                mock_redis_class.return_value = mock_redis
                mock_redis.ping.return_value = "PONG"
                
                await init_redis_pool()
                
                mock_pool.from_url.assert_called_once()
                mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_init_redis_pool_failure(self):
        """Test Redis pool initialization failure handling"""
        with patch('src.app2.core.dependencies_v2.ConnectionPool') as mock_pool:
            mock_pool.from_url.side_effect = RedisConnectionError("Connection failed")
            
            with pytest.raises(RedisConnectionError, match="Connection failed"):
                await init_redis_pool()
    
    @pytest.mark.asyncio
    async def test_get_redis_client_success(self):
        """Test successful Redis client retrieval"""
        with patch('src.app2.core.dependencies_v2._redis_client') as mock_client:
            mock_client.ping = AsyncMock(return_value="PONG")
            
            client = await get_redis_client()
            
            assert client == mock_client
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_redis_client_not_initialized(self):
        """Test Redis client when not initialized"""
        with patch('src.app2.core.dependencies_v2._redis_client', None):
            with pytest.raises(HTTPException, match="Redis service unavailable") as exc_info:
                await get_redis_client()
            
            assert exc_info.value.status_code == 503
    
    @pytest.mark.asyncio
    async def test_get_redis_client_connection_failure(self):
        """Test Redis client connection failure"""
        with patch('src.app2.core.dependencies_v2._redis_client') as mock_client:
            mock_client.ping.side_effect = RedisConnectionError("Ping failed")
            
            with pytest.raises(HTTPException, match="Redis connection failed") as exc_info:
                await get_redis_client()
            
            assert exc_info.value.status_code == 503


class TestServiceDependencies:
    """Test V2 service dependency injection and initialization"""
    
    @pytest.mark.asyncio
    async def test_init_services_success(self):
        """Test successful service initialization"""
        with patch('src.app2.core.dependencies_v2.MedicalTriageAgent') as mock_agent:
            with patch('src.app2.core.dependencies_v2.NICELookupService'):
                with patch('src.app2.core.dependencies_v2.ConversationQueue') as mock_queue_class:
                    with patch('src.app2.core.dependencies_v2.StakeholderRouter'):
                        with patch('src.app2.core.dependencies_v2.RavenBridge'):
                            
                            mock_queue_instance = AsyncMock()
                            mock_queue_class.return_value = mock_queue_instance
                            
                            await init_services()
                            
                            # Verify all services were initialized
                            mock_agent.assert_called_once()
                            mock_queue_instance.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_init_services_failure(self):
        """Test service initialization failure handling"""
        with patch('src.app2.core.dependencies_v2.MedicalTriageAgent') as mock_agent:
            mock_agent.side_effect = RuntimeError("Service init failed")
            
            with pytest.raises(RuntimeError, match="Service init failed"):
                await init_services()
    
    @pytest.mark.asyncio
    async def test_get_medical_agent_success(self):
        """Test successful medical agent retrieval"""
        mock_agent_instance = MagicMock()
        
        with patch('src.app2.core.dependencies_v2._medical_agent', mock_agent_instance):
            agent = await get_medical_agent()
            assert agent == mock_agent_instance
    
    @pytest.mark.asyncio
    async def test_get_medical_agent_not_initialized(self):
        """Test medical agent when not initialized"""
        with patch('src.app2.core.dependencies_v2._medical_agent', None):
            with pytest.raises(HTTPException, match="Medical agent not initialized") as exc_info:
                await get_medical_agent()
            
            assert exc_info.value.status_code == 503
    
    @pytest.mark.asyncio
    async def test_get_nice_lookup_success(self):
        """Test successful NICE lookup service retrieval"""
        nice_instance = MagicMock()
        
        with patch('src.app2.core.dependencies_v2._nice_lookup', nice_instance):
            nice = await get_nice_lookup()
            assert nice == nice_instance
    
    @pytest.mark.asyncio
    async def test_get_conversation_queue_success(self):
        """Test successful conversation queue retrieval"""
        queue_instance = MagicMock()
        
        with patch('src.app2.core.dependencies_v2._conversation_queue', queue_instance):
            queue = await get_conversation_queue()
            assert queue == queue_instance


class TestHealthCheckDependencies:
    """Test V2 system health check functionality"""
    
    @pytest.mark.asyncio
    async def test_check_system_health_all_healthy(self):
        """Test system health check when all services are healthy"""
        with patch('src.app2.core.dependencies_v2.AsyncSessionLocal') as mock_session_local:
            with patch('src.app2.core.dependencies_v2._redis_client') as mock_redis:
                with patch('src.app2.core.dependencies_v2._medical_agent', MagicMock()):
                    with patch('src.app2.core.dependencies_v2._nice_lookup', MagicMock()):
                        with patch('src.app2.core.dependencies_v2._conversation_queue', MagicMock()):
                            with patch('src.app2.core.dependencies_v2._stakeholder_router', MagicMock()):
                                
                                # Mock database session
                                mock_session = AsyncMock()
                                mock_session_local.return_value.__aenter__.return_value = mock_session
                                
                                # Mock Redis ping
                                mock_redis.ping = AsyncMock(return_value="PONG")
                                
                                health = await check_system_health()
                                
                                assert health["database"] == "healthy"
                                assert health["redis"] == "healthy"
                                assert health["medical_agent"] == "healthy"
                                assert health["services"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_check_system_health_database_unhealthy(self):
        """Test system health check when database is unhealthy"""
        with patch('src.app2.core.dependencies_v2.AsyncSessionLocal') as mock_session_local:
            mock_session_local.side_effect = OperationalError("DB failed", None, None)
            
            health = await check_system_health()
            
            assert health["database"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_check_system_health_redis_unhealthy(self):
        """Test system health check when Redis is unhealthy"""
        with patch('src.app2.core.dependencies_v2._redis_client') as mock_redis:
            mock_redis.ping.side_effect = RedisConnectionError("Redis failed")
            
            health = await check_system_health()
            
            assert health["redis"] == "unhealthy"


class TestCleanupDependencies:
    """Test V2 cleanup and shutdown functionality"""
    
    @pytest.mark.asyncio
    async def test_cleanup_connections_success(self):
        """Test successful cleanup of all connections"""
        with patch('src.app2.core.dependencies_v2._raven_bridge') as mock_raven:
            with patch('src.app2.core.dependencies_v2._redis_client') as mock_redis:
                with patch('src.app2.core.dependencies_v2._async_engine') as mock_engine:
                    
                    mock_raven.close = AsyncMock()
                    mock_redis.close = AsyncMock()
                    mock_engine.dispose = AsyncMock()
                    
                    await cleanup_connections()
                    
                    mock_raven.close.assert_called_once()
                    mock_redis.close.assert_called_once()
                    mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_connections_with_errors(self):
        """Test cleanup handles errors gracefully"""
        with patch('src.app2.core.dependencies_v2._raven_bridge') as mock_raven:
            mock_raven.close.side_effect = RuntimeError("Cleanup error")
            
            # Should not raise exception - graceful degradation
            await cleanup_connections()


class TestServiceLifecycle:
    """Test V2 service lifecycle management"""
    
    @pytest.mark.asyncio
    async def test_service_startup_sequence(self):
        """Test services start up in correct order"""
        startup_order = []
        
        def track_startup(service_name):
            def wrapper(*args, **kwargs):
                startup_order.append(service_name)
                return MagicMock()
            return wrapper
        
        with patch('src.app2.core.dependencies_v2.MedicalTriageAgent', side_effect=track_startup('agent')):
            with patch('src.app2.core.dependencies_v2.NICELookupService', side_effect=track_startup('nice')):
                with patch('src.app2.core.dependencies_v2.ConversationQueue') as mock_queue_class:
                    with patch('src.app2.core.dependencies_v2.StakeholderRouter', side_effect=track_startup('router')):
                        with patch('src.app2.core.dependencies_v2.RavenBridge', side_effect=track_startup('raven')):
                            
                            # Set up queue mock to track startup order
                            def queue_wrapper(*args, **kwargs):
                                startup_order.append('queue')
                                mock_instance = MagicMock()
                                mock_instance.initialize = AsyncMock()
                                return mock_instance
                            
                            mock_queue_class.side_effect = queue_wrapper
                            
                            await init_services()
                            
                            # Verify startup order
                            expected_order = ['agent', 'nice', 'queue', 'router', 'raven']
                            assert startup_order == expected_order
    
    def test_dependency_injection_pattern(self):
        """Test dependency injection follows FastAPI patterns"""
        # Verify all service getters are async functions
        import inspect
        
        assert inspect.iscoroutinefunction(get_medical_agent)
        assert inspect.iscoroutinefunction(get_nice_lookup)
        assert inspect.iscoroutinefunction(get_conversation_queue)
        assert inspect.iscoroutinefunction(get_stakeholder_router)
        assert inspect.iscoroutinefunction(get_raven_bridge)


class TestConnectionPooling:
    """Test V2 connection pooling behavior and performance"""
    
    def test_database_connection_pool_settings(self):
        """Test database connection pool is configured correctly"""
        from src.app2.core.dependencies_v2 import _async_engine
        
        # Verify production-ready pool settings
        pool = _async_engine.pool
        assert pool.size() == 20  # High concurrency support
        assert pool._max_overflow == 0  # Prevent connection leaks
        assert pool._recycle == 3600  # 1 hour connection recycling
    
    @pytest.mark.asyncio
    async def test_redis_connection_pool_settings(self):
        """Test Redis connection pool configuration"""
        with patch('src.app2.core.dependencies_v2.ConnectionPool') as mock_pool:
            # Create a proper async mock for the connection pool
            mock_connection_pool = AsyncMock()
            mock_pool.from_url.return_value = mock_connection_pool
            
            # Mock the Redis client initialization
            with patch('src.app2.core.dependencies_v2.Redis') as mock_redis_class:
                mock_redis_client = AsyncMock()
                mock_redis_class.return_value = mock_redis_client
                mock_redis_client.ping = AsyncMock(return_value="PONG")
                
                await init_redis_pool()
                
                # Verify pool was created with correct settings
                mock_pool.from_url.assert_called_once()
                call_args = mock_pool.from_url.call_args
                
                assert call_args.kwargs['max_connections'] == 50
                assert call_args.kwargs['retry_on_timeout'] is True
                assert call_args.kwargs['decode_responses'] is True
