"""
Fairdoc AI Health Check Endpoints
"""

import asyncio
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.app.core.database import get_db_session
from src.app.models.schemas.chat import HealthCheckResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Basic health check endpoint
    """
    return HealthCheckResponse(
        status="healthy",
        service="Fairdoc AI Triage System",
        version="0.1.0",
        timestamp=datetime.utcnow()
    )


@router.get("/detailed", response_model=HealthCheckResponse)
async def detailed_health_check(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Detailed health check with component status
    """
    health_status = HealthCheckResponse()
    
    try:
        # Check database connectivity
        result = await db.execute(text("SELECT 1"))
        if result.scalar() == 1:
            health_status.database = "connected"
        else:
            health_status.database = "error"
            health_status.status = "degraded"
            
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        health_status.database = "error"
        health_status.status = "unhealthy"
    
    # Additional checks can be added for Redis, AI services, etc.
    
    if health_status.status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status.dict()
        )
    
    return health_status
