"""
Fairdoc AI Chat API Endpoints
"""

import asyncio
import time
from uuid import uuid4
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.core.context.manager import FairdocContextManager
from src.app.core.database import get_db_session
from src.app.core.dependencies import get_context_manager, get_ollama_service
from src.app.models.schemas.chat import ChatMessageRequest, ChatMessageResponse
from src.app.services.ai.ollama_service import OllamaService
from src.app.services.chat.raven_integration import RavenChatService

logger = structlog.get_logger(__name__)
router = APIRouter()


def confidence_to_int(confidence: Optional[float]) -> Optional[int]:
    """Convert float confidence (0-1) to integer (0-100)"""
    if confidence is None:
        return None
    return int(confidence * 100)


@router.post("/message", response_model=ChatMessageResponse)
async def process_chat_message(
    request: ChatMessageRequest,
    db: AsyncSession = Depends(get_db_session),
    context_manager: FairdocContextManager = Depends(get_context_manager),
    ollama_service: OllamaService = Depends(get_ollama_service),
):
    """
    Process incoming chat message and return AI response
    """
    start_time = time.time()
    message_id = str(uuid4())
    
    try:
        logger.info("Processing chat message", 
                   user_id=request.user_id, 
                   message_id=message_id)
        
        # Get or create conversation context
        context = await context_manager.get_conversation_context(
            conversation_id=request.session_id or f"conv_{request.user_id}_{int(time.time())}",
            user_id=request.user_id
        )
        
        # Process message with AI service
        ai_response = await ollama_service.process_message(
            message=request.message,
            context=context,
            user_id=request.user_id
        )
        
        # Route to appropriate stakeholder
        routing = await context_manager.route_stakeholder(
            conversation_id=context.conversation_id,
            query=request.message,
            context=context
        )
        
        # Update conversation context with converted confidence values
        await context_manager.update_conversation(
            conversation_id=context.conversation_id,
            message={
                "text": request.message,
                "user_id": request.user_id,
                "timestamp": request.timestamp.isoformat(),
                "metadata": request.metadata
            },
            ai_response={
                **ai_response,
                "intent_confidence": confidence_to_int(ai_response.get("intent_confidence"))
            },
            extracted_entities={}
        )
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Build response (keeping float for API response)
        response = ChatMessageResponse(
            message_id=message_id,
            response=ai_response.get("text", "I'm here to help with your healthcare needs."),
            intent=ai_response.get("intent"),
            intent_confidence=ai_response.get("intent_confidence"),  # Keep original float
            stakeholder_route=routing.stakeholder_type,
            urgency_level=routing.urgency_level,
            estimated_wait_time=routing.estimated_response_time,
            context_used={"conversation_length": len(context.messages)},
            model_used=ai_response.get("model", "default"),
            response_time_ms=response_time_ms
        )
        
        logger.info("Chat message processed successfully",
                   message_id=message_id,
                   response_time_ms=response_time_ms,
                   stakeholder=routing.stakeholder_type)
        
        return response
        
    except Exception as e:
        logger.error("Error processing chat message", 
                    error=str(e), 
                    message_id=message_id,
                    user_id=request.user_id)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message. Please try again."
        )


@router.get("/session/{session_id}")
async def get_conversation_history(
    session_id: str,
    context_manager: FairdocContextManager = Depends(get_context_manager)
):
    """
    Retrieve conversation history for a session
    """
    try:
        return {"message": "Conversation history endpoint - coming soon"}
        
    except Exception as e:
        logger.error("Error retrieving conversation history", 
                    error=str(e), 
                    session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )
