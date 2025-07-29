"""
V2 API Router - FastAPI Router Configuration

Mounts all V2 endpoints with proper dependency injection
for the Fairdoc AI medical triage system.

Single responsibility: API routing only (no middleware/exceptions)
File: src/app2/api/v2/router_v2.py
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import structlog

from src.app2.core.config_v2 import settings_v2
from src.app2.core.dependencies_v2 import (
    get_medical_agent,
    get_nice_lookup,
    get_conversation_queue,
    get_stakeholder_router,
    get_raven_bridge,
    check_system_health
)
from src.app2.utils.datetime_utils import utcnow_iso

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
            "timestamp": utcnow_iso(),
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
                "timestamp": utcnow_iso()
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
            return {"status": "ready", "timestamp": utcnow_iso()}
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "services": health_status,
                    "timestamp": utcnow_iso()
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
        "timestamp": utcnow_iso(),
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
        "timestamp": utcnow_iso()
    }

# ---------------------------------------------------------------------------
# Export router for main application
# ---------------------------------------------------------------------------

__all__ = ["api_router"]
