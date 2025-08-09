"""
Fairdoc AI V2 FastAPI Application Entry Point

Production-ready FastAPI app with V2 medical triage system
Handles startup/shutdown, middleware, and service initialization

File: src/app2/main_v2.py
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.app2.core.config_v2 import settings_v2
from src.app2.core.dependencies_v2 import (
    init_redis_pool,
    init_services,
    cleanup_connections
)
from src.app2.core.database_v2 import startup_database, shutdown_database
from src.app2.api.v2.router_v2 import api_router

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Application Lifecycle Management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifecycle with proper startup/shutdown
    Ensures all services are initialized before serving requests
    and cleanly shutdown when application stops
    """
    logger.info("🚀 Starting Fairdoc AI V2 application...")
    
    try:
        # Startup sequence
        await startup_database()
        await init_redis_pool()
        await init_services()
        logger.info("✅ Fairdoc AI V2 application started successfully")
        
        # Application is ready to serve requests
        yield
        
    except Exception as e:
        logger.error("❌ Failed to start V2 application", error=str(e))
        raise
    finally:
        # Shutdown sequence
        logger.info("👋 Shutting down Fairdoc AI V2 application...")
        try:
            await cleanup_connections()
            await shutdown_database()
            logger.info("✅ V2 application shutdown completed")
        except Exception as e:
            logger.error("❌ Error during V2 shutdown", error=str(e))

# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create FastAPI application with V2 medical triage system
    
    Returns:
        FastAPI: Configured application instance
    """
    # Create FastAPI app with V2 configuration
    app = FastAPI(
        title="Fairdoc AI Triage System V2",
        description="Multi-turn medical triage with DSPy agents and NICE protocols",
        version="v2.6-stable",
        docs_url="/docs" if settings_v2.DEBUG else None,
        redoc_url="/redoc" if settings_v2.DEBUG else None,
        openapi_url="/openapi.json" if settings_v2.DEBUG else None,
        lifespan=lifespan
    )

    # Configure middleware stack
    _configure_middleware(app)

    # Mount V2 API router
    app.include_router(api_router)

    # Add global exception handlers
    _configure_exception_handlers(app)

    logger.info("🏗️ FastAPI V2 application created")
    return app

def _configure_middleware(app: FastAPI) -> None:
    """Configure middleware stack for production"""
    
    # CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings_v2.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Gzip compression for large responses
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request logging middleware with timing (ADDED FROM ROUTER)
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all V2 API requests with timing and performance metrics"""
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            "📨 V2 API Request",
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log successful response
            logger.info(
                "✅ V2 API Response",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time_ms=round(process_time * 1000, 2)
            )
            
            # Add processing time header for client debugging
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                "❌ V2 API Error",
                method=request.method,
                path=request.url.path,
                error=str(e),
                process_time_ms=round(process_time * 1000, 2)
            )
            raise

    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

    # Request ID middleware for tracing
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        # Add to structured logging context
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

    logger.info("🛡️ Middleware configured")

def _configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers"""
    
    @app.exception_handler(500)
    async def internal_server_error_handler(request: Request, exc: Exception):
        logger.error(
            "❌ Internal server error",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "detail": "An unexpected error occurred" if not settings_v2.DEBUG else str(exc)
                }
            }
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": 404,
                    "message": "Not found",
                    "path": str(request.url.path)
                }
            }
        )

    logger.info("⚠️ Exception handlers configured")

# ---------------------------------------------------------------------------
# Application Instance
# ---------------------------------------------------------------------------

# Create the FastAPI application
app = create_app()

# ---------------------------------------------------------------------------
# Development Server
# ---------------------------------------------------------------------------

async def main():
    """Run development server with hot reload"""
    if settings_v2.ENVIRONMENT == "development":
        config = uvicorn.Config(
            "src.app2.main_v2:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        logger.info("🚀 Starting V2 development server on http://0.0.0.0:8000")
        await server.serve()
    else:
        logger.warning("⚠️ Use production WSGI server (gunicorn) for non-development environments")

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configure structured logging for development
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Run the application
    asyncio.run(main())
