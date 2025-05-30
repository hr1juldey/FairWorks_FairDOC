"""
Fairdoc AI v1 - Clean FastAPI Application
Medical-grade triage system with NHS compliance
Senior backend architecture with proper separation of concerns
"""

# === SMART IMPORT SETUP - ADD TO TOP OF FILE ===
import sys
import os
from pathlib import Path

# Setup paths once to prevent double imports
if not hasattr(sys, '_fairdoc_paths_setup'):
    current_dir = Path(__file__).parent
    v1_dir = current_dir
    backend_dir = v1_dir.parent
    project_root = backend_dir.parent
    
    paths_to_add = [str(project_root), str(backend_dir), str(v1_dir)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    sys._fairdoc_paths_setup = True

# Setup universal imports first
import import_config  # This sets up all paths

# Standard imports first
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import uvicorn

# Smart internal imports with fallbacks
try:
    # Try absolute imports first
    from core.config import get_v1_settings
    from core.middleware import configure_middleware
    from core.exceptions import configure_exception_handlers
    
    # API routers
    from api.auth import auth_router
    from api.protocols import protocols_router
    from api.cases import cases_router
    from api.files import files_router
    
    # Database and health
    from data.database import get_database_health, init_database
    
except ImportError:
    # Fallback to relative imports
    from .core.config import get_v1_settings
    from .core.middleware import configure_middleware
    from .core.exceptions import configure_exception_handlers
    
    # API routers
    from .api.auth import auth_router
    from .api.protocols import protocols_router
    from .api.cases import cases_router
    from .api.files import files_router
    
    # Database and health
    from .data.database import get_database_health, init_database

# === END SMART IMPORT SETUP ===

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

settings = get_v1_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_FILE_PATH) if settings.LOG_TO_FILE else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# =============================================================================
# APPLICATION LIFECYCLE MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    logger.info("🚀 Starting Fairdoc AI v1...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("✅ Database initialized")
        
        # Test database health
        health = get_database_health()
        if health.get("status") == "healthy":
            logger.info("✅ Database health check passed")
            logger.info(f"✅ NICE questions loaded: {health.get('tables', {}).get('nice_questions', 0)}")
        else:
            logger.warning("⚠️ Database health check failed")
        
        # Log configuration
        logger.info(f"✅ Environment: {settings.ENVIRONMENT}")
        logger.info(f"✅ Debug mode: {settings.DEBUG}")
        logger.info(f"✅ API prefix: {settings.API_V1_PREFIX}")
        logger.info(f"✅ CORS origins: {len(settings.get_cors_origins())} configured")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Fairdoc AI v1...")

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================


app = FastAPI(
    title=settings.APP_NAME,
    description="NHS-compliant medical triage system with comprehensive NICE protocols",
    version=settings.APP_VERSION,
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================

configure_middleware(app, settings)

# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

configure_exception_handlers(app)

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Comprehensive system health check"""
    try:
        # Get database health from Phase 1 infrastructure
        health_data = get_database_health()
        
        # Enhance with v1 API specific information
        health_data.update({
            "api_name": settings.APP_NAME,
            "api_version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "features": {
                "nhs_authentication": True,
                "nice_protocols": True,
                "doctor_verification": True,
                "medical_permissions": True,
                "master_password_security": True,
                "ai_scoring": True,
                "file_upload": True,
                "pdf_generation": True,
                "background_jobs": True,
                "websocket_support": settings.WEBSOCKET_ENABLED,
                "emergency_services": settings.EMERGENCY_SERVICE_ENABLED
            },
            "endpoints": {
                "authentication": f"{settings.API_V1_PREFIX}/auth",
                "protocols": f"{settings.API_V1_PREFIX}/protocols",
                "case_reports": f"{settings.API_V1_PREFIX}/case-reports",
                "files": f"{settings.API_V1_PREFIX}/files",
                "documentation": f"{settings.API_V1_PREFIX}/docs"
            },
            "protocols_count": settings.NICE_PROTOCOLS_COUNT,
            "rate_limiting": settings.RATE_LIMIT_ENABLED
        })
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "api_version": settings.APP_VERSION,
            "timestamp": "error"
        }

@app.get("/")
async def root():
    """API root endpoint with system information"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "description": "NHS-compliant medical triage with comprehensive NICE protocols",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "documentation": f"{settings.API_V1_PREFIX}/docs",
        "health_check": "/health",
        "authentication": f"{settings.API_V1_PREFIX}/auth",
        "features": {
            "nhs_authentication": "Full NHS ID and GMC verification",
            "doctor_authentication": "GMC-verified medical professionals",
            "nice_protocols": f"{settings.NICE_PROTOCOLS_COUNT} comprehensive protocols",
            "medical_permissions": "Role-based access control",
            "master_password": "Medical-grade security",
            "real_time_chat": "WebSocket support" if settings.WEBSOCKET_ENABLED else "REST API",
            "emergency_services": "Mock hospital/ambulance" if settings.EMERGENCY_SERVICE_ENABLED else "Disabled"
        },
        "status": "operational",
        "uptime": "healthy"
    }

# =============================================================================
# API ROUTER REGISTRATION
# =============================================================================


# Register authentication router
app.include_router(
    auth_router,
    prefix=settings.API_V1_PREFIX,
    tags=["Authentication"]
)

# Register additional routers
app.include_router(protocols_router, prefix=settings.API_V1_PREFIX, tags=["NICE Protocols"])
app.include_router(cases_router, prefix=settings.API_V1_PREFIX, tags=["Case Management"])
app.include_router(files_router, prefix=settings.API_V1_PREFIX, tags=["File Management"])

# =============================================================================
# DEVELOPMENT SERVER
# =============================================================================

if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "app_v1:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1
    )
