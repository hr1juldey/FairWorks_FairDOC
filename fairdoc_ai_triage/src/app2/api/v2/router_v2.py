"""
V2 API Router - FastAPI Router Configuration

Mounts all V2 endpoints with proper middleware, error handling,
and dependency injection for the Fairdoc AI medical triage system.

Designed for production with comprehensive logging, rate limiting,
and health monitoring capabilities.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from typing import Dict, Any
import time
from datetime import datetime

from src.app2.core.config_v2 import settings_v2
from src.app2.core.dependencies_v2 import (
    get_medical_agent,
    get_nice_lookup,
    get_conversation_queue,
    get_stakeholder_router,
    get_raven_bridge,
    check_system_health
)

# Import V2 endpoint routers
from src.app2.api.v2.endpoints.multiturn_chat import router as chat_router
from src.app2.api.v2.endpoints.admin_dashboard import router as admin_router
from src.app2.api.v2.endpoints.evaluation_metrics import router as metrics_router

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Main V2 API Router
# ---------------------------------------------------------------------------

api_router = APIRouter(
    prefix="/api/v2",
    tags=["v2"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

# ---------------------------------------------------------------------------
# Global Error Handlers
# ---------------------------------------------------------------------------

@api_router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging"""
    logger.error(
        "❌ HTTP Exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

@api_router.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with structured logging"""
    logger.error(
        "❌ Unexpected Exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path)
            }
        }
    )

# ---------------------------------------------------------------------------
# Request Middleware
# ---------------------------------------------------------------------------

@api_router.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all V2 API requests with timing"""
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
        
        # Add processing time header
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

# ---------------------------------------------------------------------------
# Health Check Endpoints
# ---------------------------------------------------------------------------

@api_router.get("/health", tags=["health"])
async def health_check():
    """V2 System health check endpoint"""
    try:
        health_status = await check_system_health()
        
        # Determine overall health
        is_healthy = all(
            status != "unhealthy" 
            for status in health_status.values()
        )
        
        response_data = {
            "status": "healthy" if is_healthy else "degraded",
            "version": "v2.6-stable",
            "timestamp": datetime.utcnow().isoformat(),
            "services": health_status,
            "environment": settings_v2.ENVIRONMENT
        }
        
        status_code = 200 if is_healthy else 503
        
        logger.info(
            "🏥 V2 Health check",
            overall_status=response_data["status"],
            services=health_status
        )
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
        
    except Exception as e:
        logger.error("❌ Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@api_router.get("/health/ready", tags=["health"])
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Quick service availability check
        health_status = await check_system_health()
        
        critical_services = ["database", "redis", "medical_agent"]
        ready = all(
            health_status.get(service) == "healthy"
            for service in critical_services
        )
        
        if ready:
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "services": health_status,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    except Exception as e:
        logger.error("❌ Readiness check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )

@api_router.get("/health/live", tags=["health"])
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "v2.6-stable"
    }

# ---------------------------------------------------------------------------
# Mount Endpoint Routers
# ---------------------------------------------------------------------------

# Main chat functionality
api_router.include_router(
    chat_router,
    prefix="/medical",
    tags=["medical-chat"]
)

# Admin dashboard endpoints
api_router.include_router(
    admin_router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_medical_agent)]  # Ensure services are initialized
)

# Evaluation and metrics
api_router.include_router(
    metrics_router,
    prefix="/metrics",
    tags=["evaluation"]
)

# ---------------------------------------------------------------------------
# API Information Endpoint
# ---------------------------------------------------------------------------

@api_router.get("/info", tags=["meta"])
async def api_info():
    """V2 API information and capabilities"""
    return {
        "name": "Fairdoc AI Triage System V2",
        "version": "v2.6-stable",
        "description": "Multi-turn medical triage with DSPy agents",
        "features": {
            "multi_turn_conversations": True,
            "nice_protocol_integration": True,
            "emergency_detection": True,
            "stakeholder_routing": True,
            "redis_state_management": True,
            "dspy_optimization": True
        },
        "endpoints": {
            "medical_chat": "/api/v2/medical/chat",
            "admin_dashboard": "/api/v2/admin/dashboard",
            "evaluation_metrics": "/api/v2/metrics/evaluate",
            "health_check": "/api/v2/health"
        },
        "documentation": {
            "openapi": "/docs",
            "redoc": "/redoc"
        },
        "environment": settings_v2.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

# ---------------------------------------------------------------------------
# Export router for main application
# ---------------------------------------------------------------------------

__all__ = ["api_router"]
