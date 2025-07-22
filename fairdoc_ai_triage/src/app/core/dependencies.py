"""
Fairdoc AI Dependency Injection
"""

from fastapi import Request
from typing import Annotated

from src.app.core.context.manager import FairdocContextManager
from src.app.services.ai.ollama_service import OllamaService
from src.app.services.chat.raven_integration import RavenChatService


def get_context_manager(request: Request) -> FairdocContextManager:
    """Get context manager from app state"""
    return request.app.state.context_manager


def get_ollama_service(request: Request) -> OllamaService:
    """Get Ollama service from app state"""
    return request.app.state.ollama


def get_raven_service(request: Request) -> RavenChatService:
    """Get Raven Chat service from app state"""
    return request.app.state.raven_chat
