"""
Admin Dashboard API Endpoints

Stub implementation for admin functionality
Single responsibility: Admin interface endpoints

File: src/app2/api/v2/endpoints/admin_dashboard.py
"""

from fastapi import APIRouter

from src.app2.utils.datetime_utils import utcnow_timestamp
from typing import Dict, Any
import structlog

from src.app2.core.dependencies_v2 import get_medical_agent
from src.app2.models.schemas.multiturn_chat import ConversationStatus

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/dashboard", tags=["admin"])
async def get_admin_dashboard():
    """Admin dashboard overview - stub implementation"""
    return {
        "status": "active",
        "version": "v2.6-stable",
        "timestamp": utcnow_timestamp,
        "stats": {
            "active_conversations": 0,
            "completed_today": 0,
            "emergency_alerts": 0,
            "system_health": "healthy"
        },
        "message": "Admin dashboard stub - implementation pending"
    }


@router.get("/conversations", tags=["admin"])
async def list_conversations(
    limit: int = 50,
    status_filter: ConversationStatus = None
):
    """List active conversations - stub implementation"""
    return {
        "conversations": [],
        "total": 0,
        "limit": limit,
        "status_filter": status_filter,
        "message": "Conversation listing stub - implementation pending"
    }


@router.get("/system/health", tags=["admin"])
async def system_health_detailed():
    """Detailed system health for admin - stub implementation"""
    return {
        "overall_status": "healthy",
        "services": {
            "database": "healthy",
            "redis": "healthy", 
            "medical_agent": "healthy",
            "nice_lookup": "healthy"
        },
        "metrics": {
            "uptime_seconds": 0,
            "memory_usage_mb": 0,
            "active_connections": 0
        },
        "timestamp": utcnow_timestamp,
        "message": "System health stub - implementation pending"
    }


@router.post("/conversations/{conversation_id}/intervention", tags=["admin"])
async def admin_intervention(
    conversation_id: str,
    intervention_type: str,
    notes: str = None
):
    """Admin intervention in conversation - stub implementation"""
    return {
        "conversation_id": conversation_id,
        "intervention_type": intervention_type,
        "notes": notes,
        "timestamp": utcnow_timestamp,
        "status": "queued",
        "message": "Admin intervention stub - implementation pending"
    }


@router.get("/alerts/emergency", tags=["admin"])
async def get_emergency_alerts(limit: int = 20):
    """Get recent emergency alerts - stub implementation"""
    return {
        "alerts": [],
        "total": 0,
        "limit": limit,
        "timestamp": utcnow_timestamp,
        "message": "Emergency alerts stub - implementation pending"
    }
