"""
V2 Database Infrastructure - Async SQLAlchemy Foundation

Provides async database engine, session factory, and table management
for the Fairdoc AI V2 medical triage system.

Designed for production with connection pooling, proper error handling,
and seamless integration with FastAPI dependency injection.
"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession, 
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import text, event
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from src.app2.core.config_v2 import settings_v2


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Base Model Class for V2 System
# ---------------------------------------------------------------------------

# Create V2 base class (separate from V1 to avoid conflicts)
BaseV2 = declarative_base()

class DatabaseV2:
    """V2 Database management class with async support"""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = False
    
    def create_engine(self) -> AsyncEngine:
        """Create async SQLAlchemy engine with production settings"""
        if self._engine is not None:
            return self._engine
        
        # Engine configuration for production
        engine_kwargs = {
            "url": settings_v2.DATABASE_URL,
            "echo": settings_v2.DEBUG,
            "future": True,  # Use SQLAlchemy 2.0 style
            "pool_pre_ping": True,  # Validate connections before use
            "pool_recycle": 3600,   # Recycle connections every hour
        }
        
        # Configure connection pool based on environment
        if settings_v2.ENVIRONMENT == "testing":
            # Use NullPool for testing to avoid connection limits
            engine_kwargs.update({
                "poolclass": NullPool,
            })
        else:
            # Production connection pool
            engine_kwargs.update({
                "poolclass": AsyncAdaptedQueuePool,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
            })
        
        self._engine = create_async_engine(**engine_kwargs)
        
        # Add connection event listeners for monitoring
        event.listen(self._engine.sync_engine, "connect", self._on_connect)
        event.listen(self._engine.sync_engine, "checkout", self._on_checkout)
        
        logger.info("🗄️ V2 Database engine created", 
                   pool_size=engine_kwargs.get("pool_size", "unlimited"))
        
        return self._engine
    
    def create_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Create async session factory"""
        if self._session_factory is not None:
            return self._session_factory
        
        if self._engine is None:
            self.create_engine()
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        
        logger.info("📊 V2 Session factory created")
        return self._session_factory
    
    async def initialize(self) -> None:
        """Initialize database connection, create tables, and verify connectivity"""
        if self._initialized:
            return

        try:
            # Create session factory (engine is created internally)
            session_factory = self.create_session_factory()

            # CRITICAL: Create all tables FIRST before any queries
            await self.create_tables()
            logger.info("🏗️ V2 Database tables ensured")

            # Test database connectivity and verify result
            async with session_factory() as session:
                result = await session.execute(text("SELECT 1 as connectivity_test"))
                row = result.fetchone()

                # Verify the test query returned expected result
                if not row or row[0] != 1:
                    raise RuntimeError("Database connectivity test failed")

                await session.commit()

            self._initialized = True
            logger.info("✅ V2 Database initialized successfully")

        except Exception as e:
            logger.error("❌ Failed to initialize V2 database", error=str(e))
            raise

    
    async def create_tables(self) -> None:
        """Create all V2 tables (for development/testing)"""
        if self._engine is None:
            self.create_engine()

        try:
            # Import models here to avoid circular imports
            from src.app2.models.database.nice_protocols import NICEProtocol
            from src.app2.models.database.gold_standards import GoldStandardDialogue
            from src.app2.models.database.conversation_state import ConversationStateV2

            async with self._engine.begin() as conn:
                await conn.run_sync(BaseV2.metadata.create_all)
            logger.info("🏗️ V2 Database tables created")
        except Exception as e:
            logger.error("❌ Failed to create V2 tables", error=str(e))
            raise

    
    async def drop_tables(self) -> None:
        """Drop all V2 tables (for testing cleanup)"""
        if self._engine is None:
            return
            
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(BaseV2.metadata.drop_all)
            
            logger.info("🗑️ V2 Database tables dropped")
            
        except Exception as e:
            logger.error("❌ Failed to drop V2 tables", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close database engine and clean up connections"""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            
            logger.info("🔒 V2 Database connections closed")
    
    @staticmethod
    def _on_connect(dbapi_connection, connection_record):
        """Connection event handler for monitoring"""
        logger.debug("🔌 Database connection established")
    
    @staticmethod  
    def _on_checkout(dbapi_connection, connection_record, connection_proxy):
        """Connection checkout event handler"""
        logger.debug("📤 Database connection checked out from pool")
    
    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get the async database engine"""
        return self._engine
    
    @property
    def session_factory(self) -> Optional[async_sessionmaker[AsyncSession]]:
        """Get the async session factory"""
        return self._session_factory

# ---------------------------------------------------------------------------
# Global Database Instance
# ---------------------------------------------------------------------------

# Singleton database instance for V2 system
database_v2 = DatabaseV2()

# Convenience functions for external use
def get_engine() -> AsyncEngine:
    """Get V2 database engine"""
    return database_v2.create_engine()

def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get V2 session factory"""
    return database_v2.create_session_factory()

# ---------------------------------------------------------------------------
# Async Session Context Manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    
    Usage:
        async with get_async_session() as session:
            result = await session.execute(query)
            await session.commit()
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            logger.debug("📊 V2 Database session created")
            yield session
            
        except Exception as e:
            await session.rollback()
            logger.error("❌ V2 Database session error", error=str(e))
            raise
            
        finally:
            await session.close()
            logger.debug("🔒 V2 Database session closed")

# ---------------------------------------------------------------------------
# Database Health Check
# ---------------------------------------------------------------------------

async def check_database_health() -> dict:
    """
    Check V2 database connectivity and return status.
    
    Returns:
        dict: Health status with connection details
    """
    try:
        async with get_async_session() as session:
            # Test basic connectivity
            result = await session.execute(text("SELECT 1 as health_check"))
            row = result.fetchone()
            
            # Test table access (basic metadata query)
            table_check = await session.execute(
                text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
            )
            table_count = table_check.scalar()
            
            return {
                "status": "healthy",
                "connection": "active",
                "health_check_result": row[0] if row else None,
                "table_count": table_count,
                "engine_pool_size": getattr(database_v2._engine.pool, 'size', 'unlimited') if database_v2._engine else None
            }
            
    except Exception as e:
        logger.error("❌ V2 Database health check failed", error=str(e))
        return {
            "status": "unhealthy", 
            "error": str(e),
            "connection": "failed"
        }

# ---------------------------------------------------------------------------
# Startup/Shutdown Event Handlers
# ---------------------------------------------------------------------------

async def startup_database():
    """Initialize database on application startup"""
    try:
        await database_v2.initialize()
        logger.info("🚀 V2 Database startup completed")
        
    except Exception as e:
        logger.error("❌ V2 Database startup failed", error=str(e))
        raise

async def shutdown_database():
    """Clean up database connections on application shutdown"""
    try:
        await database_v2.close()
        logger.info("👋 V2 Database shutdown completed")
        
    except Exception as e:
        logger.error("❌ V2 Database shutdown error", error=str(e))
