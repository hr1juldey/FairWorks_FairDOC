"""
V2 FastAPI Dependencies - Production-Grade Dependency Injection

Provides database sessions, Redis connections, and service initialization
for the Fairdoc AI V2 medical triage system.

Follows FastAPI dependency injection patterns with proper error handling,
connection pooling, and graceful degradation.
"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import structlog
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from redis.asyncio import Redis, ConnectionPool
import asyncio

from src.app2.core.config_v2 import settings_v2
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
from src.app2.services.context.nice_lookup import NICELookupService
from src.app2.services.context.redis_queue import ConversationQueue
from src.app2.services.chat.stakeholder_router import StakeholderRouter
from src.app2.services.chat.raven_bridge import RavenBridge
from src.app2.services.database.initialization_service import initialize_database_on_startup
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Database Dependencies
# ---------------------------------------------------------------------------

# Async SQLAlchemy engine with connection pooling
_async_engine = create_async_engine(
    settings_v2.DATABASE_URL,
    echo=settings_v2.DEBUG,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Async session factory
AsyncSessionLocal = sessionmaker(
    bind=_async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
    autocommit=False,
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide async database session with automatic cleanup.
    
    Yields:
        AsyncSession: SQLAlchemy async session
        
    Raises:
        HTTPException: If database connection fails
    """
    async with AsyncSessionLocal() as session:
        try:
            logger.debug("📊 Database session created")
            yield session
            await session.commit()
            logger.debug("✅ Database session committed")
        except Exception as e:
            await session.rollback()
            logger.error("❌ Database session error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database operation failed"
            ) from e
        finally:
            await session.close()
            logger.debug("🔒 Database session closed")

# ---------------------------------------------------------------------------
# Redis Dependencies  
# ---------------------------------------------------------------------------

# Redis connection pool (initialized on startup)
_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[Redis] = None

async def init_redis_pool():
    """Initialize Redis connection pool on application startup."""
    global _redis_pool, _redis_client
    
    try:
        _redis_pool = ConnectionPool.from_url(
            settings_v2.REDIS_URL,
            decode_responses=True,
            max_connections=50,
            retry_on_timeout=True,
        )
        _redis_client = Redis(connection_pool=_redis_pool)
        
        # Test connection
        await _redis_client.ping()
        logger.info("✅ Redis connection pool initialized")
        
    except Exception as e:
        logger.error("❌ Failed to initialize Redis", error=str(e))
        raise

async def get_redis_client() -> Redis:
    """
    Get Redis client from connection pool.
    
    Returns:
        Redis: Async Redis client
        
    Raises:
        HTTPException: If Redis is unavailable
    """
    if _redis_client is None:
        logger.error("❌ Redis client not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis service unavailable"
        )
    
    try:
        # Test connection health
        await _redis_client.ping()
        logger.debug("📱 Redis client provided")
        return _redis_client
        
    except Exception as e:
        logger.error("❌ Redis connection failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis connection failed"
        ) from e

# ---------------------------------------------------------------------------
# Service Dependencies (Singletons)
# ---------------------------------------------------------------------------

# Global service instances (initialized once)
# Global service instances (initialized once)
_medical_agent: Optional[MedicalTriageAgent] = None
_question_generator: Optional[MedicalQuestionGenerator] = None
_nice_lookup: Optional[NICELookupService] = None
_conversation_queue: Optional[ConversationQueue] = None
_stakeholder_router: Optional[StakeholderRouter] = None
_raven_bridge: Optional[RavenBridge] = None

async def init_services():
    """Initialize all V2 services on application startup."""
    global _medical_agent, _nice_lookup, _conversation_queue, _stakeholder_router, _raven_bridge

    try:
        logger.info("🚀 Initializing V2 services...")
        
        # Initialize database with seed data FIRST (other services depend on this)
        logger.info("🗄️ Initializing database with seed data...")
        await initialize_database_on_startup()
        logger.info("✅ Database initialization completed")

        # Initialize DSPy Medical Agent  
        _medical_agent = MedicalTriageAgent(model_name=settings_v2.FAIRDOC_V2_DSPy_MODEL)
        logger.info("🩺 Medical agent initialized")

        # Initialize Question Generator
        _question_generator = MedicalQuestionGenerator(model_name=settings_v2.FAIRDOC_V2_DSPy_MODEL)
        logger.info("❓ Question generator initialized")


        # Initialize NICE Lookup Service
        _nice_lookup = NICELookupService()
        logger.info("📋 NICE lookup service initialized")

        # Initialize Redis-based Conversation Queue
        _conversation_queue = ConversationQueue()
        await _conversation_queue.initialize()
        logger.info("💬 Conversation queue initialized")

        # Initialize Stakeholder Router
        _stakeholder_router = StakeholderRouter()
        logger.info("🔄 Stakeholder router initialized")

        # Initialize Raven Bridge
        _raven_bridge = RavenBridge()
        logger.info("📤 Raven bridge initialized")

        logger.info("✅ All V2 services initialized successfully")

    except Exception as e:
        logger.error("❌ Service initialization failed", error=str(e))
        logger.error("💥 Failed component during V2 service initialization")
        raise RuntimeError(f"V2 service initialization failed: {str(e)}") from e



async def get_medical_agent() -> MedicalTriageAgent:
    """Get DSPy medical triage agent instance."""
    if _medical_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Medical agent not initialized"
        )
    return _medical_agent

async def get_nice_lookup() -> NICELookupService:
    """Get NICE protocol lookup service instance."""
    if _nice_lookup is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NICE lookup service not initialized"
        )
    return _nice_lookup

async def get_question_generator() -> MedicalQuestionGenerator:
    """Get medical question generator instance."""
    if _question_generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Question generator not initialized"
        )
    return _question_generator

async def get_conversation_queue() -> ConversationQueue:
    """Get Redis conversation queue instance."""
    if _conversation_queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation queue not initialized"
        )
    return _conversation_queue

async def get_stakeholder_router() -> StakeholderRouter:
    """Get stakeholder routing service instance."""
    if _stakeholder_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stakeholder router not initialized"
        )
    return _stakeholder_router

async def get_raven_bridge() -> RavenBridge:
    """Get Raven chat bridge instance."""
    if _raven_bridge is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Raven bridge not initialized"
        )
    return _raven_bridge

# ---------------------------------------------------------------------------
# Cleanup Dependencies
# ---------------------------------------------------------------------------

async def cleanup_connections():
    """Clean up all connections on application shutdown."""
    logger.info("🧹 Cleaning up V2 connections...")
    
    try:
        # Close Raven bridge
        if _raven_bridge:
            await _raven_bridge.close()
            logger.info("📤 Raven bridge closed")
        
        # Close Redis connections
        if _redis_client:
            await _redis_client.close()
            logger.info("📱 Redis client closed")
            
        # Close database engine
        await _async_engine.dispose()
        logger.info("📊 Database engine disposed")
        
        logger.info("✅ All V2 connections cleaned up")
        
    except Exception as e:
        logger.error("❌ Error during cleanup", error=str(e))

# ---------------------------------------------------------------------------
# Health Check Dependencies
# ---------------------------------------------------------------------------

async def check_system_health() -> dict:
    """
    Check health of all V2 system dependencies.
    
    Returns:
        dict: Health status of all services
    """
    health_status = {
        "database": "unknown",
        "redis": "unknown", 
        "medical_agent": "unknown",
        "services": "unknown"
    }
    
    # Check database
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            health_status["database"] = "healthy"
    except Exception:
        health_status["database"] = "unhealthy"
    
    # Check Redis
    try:
        if _redis_client:
            await _redis_client.ping()
            health_status["redis"] = "healthy"
        else:
            health_status["redis"] = "not_initialized"
    except Exception:
        health_status["redis"] = "unhealthy"
    
    # Check medical agent
    try:
        if _medical_agent:
            health_status["medical_agent"] = "healthy"
        else:
            health_status["medical_agent"] = "not_initialized"
    except Exception:
        health_status["medical_agent"] = "unhealthy"
    
    # Check services
    services_healthy = all([
        _nice_lookup is not None,
        _conversation_queue is not None,
        _stakeholder_router is not None
    ])
    health_status["services"] = "healthy" if services_healthy else "unhealthy"
    
    return health_status
