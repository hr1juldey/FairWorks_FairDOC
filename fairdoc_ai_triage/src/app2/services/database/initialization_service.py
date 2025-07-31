"""
Database Initialization Service for Fairdoc AI V2
Handles database setup, seed data loading, and migrations
Single responsibility: Database initialization and seeding
"""

from typing import Dict, List, Any, Optional
import structlog
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.app2.core.database_v2 import get_async_session
from src.app2.models.database.nice_protocols import NICEProtocol, NICE_SEED_DATA
from src.app2.models.database.gold_standards import GoldStandardDialogue
from src.app2.models.database.gold_standards_seed import (
    GOLD_STANDARDS_SEED_DATA,
    validate_gold_standards
)
from src.app2.utils.datetime_utils import utcnow

logger = structlog.get_logger(__name__)

class DatabaseInitializationService:
    """Service for initializing database with seed data and migrations"""
    
    def __init__(self):
        self.initialized = False
        
    async def initialize_database(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Initialize database with all required seed data
        
        Args:
            force_reload: If True, clear and reload all seed data
            
        Returns:
            Dict with initialization results
        """
        if self.initialized and not force_reload:
            logger.info("✅ Database already initialized")
            return {"status": "already_initialized", "skipped": True}
        
        logger.info("🚀 Starting database initialization...")
        
        results = {
            "nice_protocols": {"loaded": 0, "skipped": 0, "errors": 0},
            "gold_standards": {"loaded": 0, "skipped": 0, "errors": 0},
            "total_time_ms": 0
        }
        
        start_time = utcnow()
        
        try:
            async with get_async_session() as session:
                # Load NICE protocols first (required by gold standards)
                nice_results = await self._load_nice_protocols(session, force_reload)
                results["nice_protocols"] = nice_results
                
                # Load gold standards for DSPy training
                gold_results = await self._load_gold_standards(session, force_reload)
                results["gold_standards"] = gold_results
                
                # Commit all changes
                await session.commit()
                
                # Validate loaded data
                await self._validate_loaded_data(session)
                
            self.initialized = True
            end_time = utcnow()
            results["total_time_ms"] = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info("✅ Database initialization completed successfully",
                       nice_protocols=nice_results["loaded"],
                       gold_standards=gold_results["loaded"],
                       time_ms=results["total_time_ms"])
            
            return {"status": "completed", **results}
            
        except Exception as e:
            logger.error("❌ Database initialization failed", error=str(e))
            raise RuntimeError(f"Database initialization failed: {str(e)}")
    
    async def _load_nice_protocols(self, session, force_reload: bool) -> Dict[str, int]:
        """Load NICE protocol seed data into database"""
        results = {"loaded": 0, "skipped": 0, "errors": 0}
        
        logger.info("📋 Loading NICE protocols...", count=len(NICE_SEED_DATA))
        
        if force_reload:
            # Clear existing protocols
            await session.execute(text("DELETE FROM nice_protocols"))
            logger.info("🗑️ Cleared existing NICE protocols")
        
        for protocol_data in NICE_SEED_DATA:
            try:
                # Remove 'category' field since model doesn't have it
                filtered_data = {
                    k: v for k, v in protocol_data.items() 
                    if k != 'category'
                }
                
                # Check if protocol already exists
                existing = await session.execute(
                    text("SELECT id FROM nice_protocols WHERE protocol_code = :code"),
                    {"code": filtered_data["protocol_code"]}
                )
                
                if existing.fetchone() and not force_reload:
                    results["skipped"] += 1
                    continue
                
                # Create new protocol
                protocol = NICEProtocol(**filtered_data)
                session.add(protocol)
                results["loaded"] += 1
                
            except IntegrityError:
                results["skipped"] += 1
                logger.debug("Protocol already exists", 
                           code=protocol_data["protocol_code"])
            except Exception as e:
                results["errors"] += 1
                logger.error("Failed to load protocol", 
                           code=protocol_data.get("protocol_code", "unknown"),
                           error=str(e))
        
        logger.info("📋 NICE protocols loading completed",
                   loaded=results["loaded"],
                   skipped=results["skipped"],
                   errors=results["errors"])
        
        return results
    
    async def _load_gold_standards(self, session, force_reload: bool) -> Dict[str, int]:
        """Load gold standards seed data into database"""
        results = {"loaded": 0, "skipped": 0, "errors": 0}
        
        # Validate gold standards quality first
        validation = validate_gold_standards()
        if not validation.get("coverage_balanced", False):
            logger.warning("⚠️ Gold standards may not be balanced across outcomes")
        
        logger.info("🏆 Loading gold standards...", 
                   count=len(GOLD_STANDARDS_SEED_DATA))
        
        if force_reload:
            # Clear existing gold standards
            await session.execute(text("DELETE FROM gold_standard_dialogues_v2"))
            logger.info("🗑️ Cleared existing gold standards")
        
        for gs_data in GOLD_STANDARDS_SEED_DATA:
            try:
                # Check if standard already exists
                existing = await session.execute(
                    text("SELECT standard_id FROM gold_standard_dialogues_v2 WHERE title = :title"),
                    {"title": gs_data["title"]}
                )
                
                if existing.fetchone() and not force_reload:
                    results["skipped"] += 1
                    continue
                
                # Create new gold standard
                gold_standard = GoldStandardDialogue(**gs_data)
                session.add(gold_standard)
                results["loaded"] += 1
                
            except IntegrityError:
                results["skipped"] += 1
                logger.debug("Gold standard already exists", title=gs_data["title"])
            except Exception as e:
                results["errors"] += 1
                logger.error("Failed to load gold standard",
                           title=gs_data.get("title", "unknown"),
                           error=str(e))
        
        logger.info("🏆 Gold standards loading completed",
                   loaded=results["loaded"],
                   skipped=results["skipped"],
                   errors=results["errors"])
        
        return results
    
    async def _validate_loaded_data(self, session) -> None:
        """Validate that seed data was loaded correctly"""
        # Check NICE protocols count
        nice_count = await session.execute(
            text("SELECT COUNT(*) FROM nice_protocols")
        )
        nice_total = nice_count.scalar()
        
        # Check gold standards count
        gold_count = await session.execute(
            text("SELECT COUNT(*) FROM gold_standard_dialogues_v2")
        )
        gold_total = gold_count.scalar()
        
        logger.info("📊 Database validation completed",
                   nice_protocols=nice_total,
                   gold_standards=gold_total)
        
        # Verify minimum required data
        if nice_total < 10:
            raise RuntimeError(f"Insufficient NICE protocols loaded: {nice_total} < 10")
        
        if gold_total < 3:
            raise RuntimeError(f"Insufficient gold standards loaded: {gold_total} < 3")
    
    async def get_initialization_status(self) -> Dict[str, Any]:
        """Get current database initialization status"""
        try:
            async with get_async_session() as session:
                # Count loaded records
                nice_count = await session.execute(
                    text("SELECT COUNT(*) FROM nice_protocols")
                )
                gold_count = await session.execute(
                    text("SELECT COUNT(*) FROM gold_standard_dialogues_v2")
                )
                
                return {
                    "initialized": self.initialized,
                    "nice_protocols_count": nice_count.scalar(),
                    "gold_standards_count": gold_count.scalar(),
                    "status": "ready" if self.initialized else "pending"
                }
        except Exception as e:
            return {
                "initialized": False,
                "error": str(e),
                "status": "error"
            }

# Singleton instance for dependency injection
database_initialization = DatabaseInitializationService()

async def initialize_database_on_startup() -> None:
    """Convenience function to initialize database during app startup"""
    await database_initialization.initialize_database()
