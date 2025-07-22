"""
Fairdoc AI Triage System - Main Application Entry Point
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from src.app.api.v1.router import api_router
from src.app.core.config import settings
from src.app.core.context.manager import FairdocContextManager
from src.app.core.database import init_db
from src.app.core.logging import configure_logging
from src.app.services.ai.ollama_service import OllamaService
from src.app.services.chat.raven_integration import RavenChatService

# Configure structured logging
configure_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events"""
    
    # Startup
    logger.info("🏥 Starting Fairdoc AI Triage System...")
    
    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")
    
    # Initialize AI services
    ollama_service = OllamaService()
    await ollama_service.initialize()
    app.state.ollama = ollama_service
    logger.info("✅ Ollama service initialized")
    
    # Initialize context manager
    context_manager = FairdocContextManager()
    await context_manager.initialize()
    app.state.context_manager = context_manager
    logger.info("✅ Context manager initialized")
    
    # Initialize Raven Chat integration
    raven_service = RavenChatService()
    await raven_service.initialize()
    app.state.raven_chat = raven_service
    logger.info("✅ Raven Chat service initialized")
    
    logger.info("🚀 Fairdoc AI Triage System started successfully!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Fairdoc AI Triage System...")
    
    # Cleanup resources
    await context_manager.cleanup()
    await ollama_service.cleanup()
    await raven_service.cleanup()
    
    logger.info("👋 Fairdoc AI Triage System shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Fairdoc AI Triage System",
    description="AI-powered emergency healthcare triage and assistance platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint for container monitoring"""
    return {
        "status": "healthy",
        "service": "Fairdoc AI Triage System",
        "version": "0.1.0"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error("Unhandled exception", exc_info=exc, path=request.url.path)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please contact support."
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_level="info"
    )
