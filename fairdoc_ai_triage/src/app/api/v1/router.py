"""
Fairdoc AI API Router Configuration
"""

from fastapi import APIRouter

from src.app.api.v1.endpoints import chat, health, thinking

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["chat"]
)

api_router.include_router(
    health.router,
    prefix="/health", 
    tags=["health"]
)

api_router.include_router(
    thinking.router,
    prefix="/thinking",
    tags=["thinking", "observability"]
)
