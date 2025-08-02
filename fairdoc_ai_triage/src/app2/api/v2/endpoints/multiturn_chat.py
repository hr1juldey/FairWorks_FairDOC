"""
Multi-turn Medical Chat API Endpoints

Lean FastAPI layer that delegates business logic to ChatOrchestrator service.
Handles HTTP requests/responses, background tasks, and error handling.

Single responsibility: HTTP API interface for medical chat
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
import structlog
from datetime import datetime
from src.app2.utils.datetime_utils import utcnow_timestamp
from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatRequest,
    MultiTurnChatResponse,
    StakeholderRole
)
from src.app2.services.chat.chat_orchestrator import ChatOrchestrator
from src.app2.services.chat.emergency_handler import EmergencyHandler
from src.app2.services.chat.persistence_handler import PersistenceHandler
from src.app2.core.dependencies_v2 import (
    get_medical_agent,
    get_conversation_queue,
    get_nice_lookup,
    get_stakeholder_router
)

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency injection for services
async def get_chat_orchestrator(
    medical_agent=Depends(get_medical_agent),
    conversation_queue=Depends(get_conversation_queue),
    nice_lookup=Depends(get_nice_lookup),
    stakeholder_router=Depends(get_stakeholder_router)
) -> ChatOrchestrator:
    """Get initialized ChatOrchestrator with injected dependencies"""
    orchestrator = ChatOrchestrator()
    
    # Inject dependencies (avoiding duplicate initialization)
    orchestrator.medical_agent = medical_agent
    orchestrator.conversation_queue = conversation_queue
    orchestrator.nice_lookup = nice_lookup
    orchestrator.stakeholder_router = stakeholder_router
    
    return orchestrator

async def get_emergency_handler() -> EmergencyHandler:
    """Get EmergencyHandler instance"""
    return EmergencyHandler()

async def get_persistence_handler() -> PersistenceHandler:
    """Get PersistenceHandler instance"""
    return PersistenceHandler()

@router.post("/chat", response_model=MultiTurnChatResponse)
async def process_multiturn_chat(
    request: MultiTurnChatRequest,
    background_tasks: BackgroundTasks,
    chat_orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
    emergency_handler: EmergencyHandler = Depends(get_emergency_handler),
    persistence_handler: PersistenceHandler = Depends(get_persistence_handler)
):
    """
    Main multi-turn medical chat endpoint
    
    Delegates business logic to ChatOrchestrator and handles background tasks
    """
    try:
        # Delegate business logic to orchestrator
        orchestration_result = await chat_orchestrator.process_conversation_turn(request)
        
        # Schedule background tasks based on orchestration result
        await _schedule_background_tasks(
            orchestration_result,
            request.user_id,
            background_tasks,
            emergency_handler,
            persistence_handler
        )
        
        # Build and return API response
        response = chat_orchestrator.build_chat_response(orchestration_result)
        
        logger.info("✅ Chat API request completed",
                   conversation_id=response.conversation_id,
                   outcome=response.medical_outcome,
                   turn=response.turn_number)
        
        return response
        
    except ValueError as e:
        # Business logic errors (conversation not found, etc.)
        logger.warning("⚠️ Chat request validation error", error=str(e))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        
    except Exception as e:
        # Unexpected errors
        logger.error("❌ Chat API error", error=str(e), user_id=request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process conversation: {str(e)}"
        )

@router.get("/chat/{conversation_id}/state")
async def get_conversation_state(
    conversation_id: str,
    chat_orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
):
    """Get current conversation state"""
    state = await chat_orchestrator.get_conversation_state(conversation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return state

@router.get("/chat/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    chat_orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
):
    """Get conversation history"""
    state = await chat_orchestrator.get_conversation_state(conversation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "turns": state.get("conversation_history", []),
        "status": state["status"],
        "outcome": state["current_outcome"]
    }

@router.post("/chat/{conversation_id}/reset")
async def reset_conversation(
    conversation_id: str,
    chat_orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
):
    """Reset conversation state (for testing/debugging)"""
    await chat_orchestrator.reset_conversation(conversation_id)
    return {"message": f"Conversation {conversation_id} reset successfully"}

@router.get("/health")
async def health_check(
    chat_orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
):
    """Health check endpoint for chat services"""
    try:
        services_health = await chat_orchestrator.check_service_health()
        
        return {
            "status": "healthy",
            "services": services_health,
            "timestamp": utcnow_timestamp
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )
@router.post("/chat/emergency-test", response_model=MultiTurnChatResponse)
async def emergency_detection_test(
    symptoms: str = "severe chest pain radiating to left arm with sweating",
    chat_orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
):
    """Test endpoint for emergency detection"""
    
    request = MultiTurnChatRequest(
        user_message=symptoms,
        stakeholder_role=StakeholderRole.PATIENT,
        stakeholder_id="test_emergency_patient"
    )
    
    orchestration_result = await chat_orchestrator.process_conversation_turn(request)
    response = chat_orchestrator.build_chat_response(orchestration_result)
    
    return response

async def _schedule_background_tasks(
    orchestration_result: dict,
    user_id: str,
    background_tasks: BackgroundTasks,
    emergency_handler: EmergencyHandler,
    persistence_handler: PersistenceHandler
):
    """Schedule background tasks based on orchestration result"""
    
    conversation_id = orchestration_result["conversation_id"]
    
    # Handle emergency alerts
    if orchestration_result.get("requires_emergency_alert"):
        background_tasks.add_task(
            emergency_handler.handle_emergency_alert,
            conversation_id,
            user_id,
            orchestration_result["agent_result"]
        )
    
    # Handle conversation persistence
    if orchestration_result.get("requires_persistence"):
        background_tasks.add_task(
            persistence_handler.save_conversation_to_postgresql,
            conversation_id,
            orchestration_result["updated_state"]
        )
