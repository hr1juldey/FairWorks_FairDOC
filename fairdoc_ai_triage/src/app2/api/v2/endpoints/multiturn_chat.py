"""
Multi-turn Medical Chat API Endpoint - Main V2 Interface
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import structlog
from datetime import datetime

from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.context.redis_queue import ConversationQueue
from src.app2.services.context.nice_lookup import NICELookupService
from src.app2.services.chat.stakeholder_router import StakeholderRouter

logger = structlog.get_logger(__name__)
router = APIRouter()

class MultiTurnChatRequest(BaseModel):
    user_id: str = Field(..., description="Patient user ID")
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    stakeholder_type: str = Field(default="patient", description="patient|doctor|admin")

class MultiTurnChatResponse(BaseModel):
    conversation_id: str
    agent_response: Optional[str] = None
    next_question: Optional[str] = None
    medical_outcome: str  # emergency|routine_doctor|self_care|inconclusive|spam
    confidence_score: int
    is_conversation_complete: bool
    turn_number: int
    red_flags: List[str] = []
    reasoning: str
    estimated_completion_turns: Optional[int] = None

# Initialize services (would be done via dependency injection in production)
medical_agent = MedicalTriageAgent()
conversation_queue = ConversationQueue()
nice_lookup = NICELookupService()
stakeholder_router = StakeholderRouter()

@router.post("/chat", response_model=MultiTurnChatResponse)
async def process_multiturn_chat(
    request: MultiTurnChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Process multi-turn medical conversation with progressive questioning
    """
    try:
        logger.info("🩺 Processing multi-turn chat",
                   user_id=request.user_id,
                   conversation_id=request.conversation_id,
                   stakeholder=request.stakeholder_type)
        
        # Handle new conversation
        if not request.conversation_id:
            conversation_id = await conversation_queue.start_conversation(
                user_id=request.user_id,
                initial_symptoms=request.message
            )
        else:
            conversation_id = request.conversation_id
        
        # Get conversation state
        conversation_state = await conversation_queue.get_conversation_state(conversation_id)
        if not conversation_state:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Lookup relevant NICE protocols
        nice_context = await nice_lookup.find_relevant_protocols(request.message)
        
        # Process with medical agent
        agent_result = await medical_agent.process_turn(
            symptoms=request.message,
            nice_context=nice_context["protocol_text"],
            # In full implementation, would reconstruct DSPy.History from conversation_state
        )
        
        # Update conversation state
        updated_state = await conversation_queue.update_conversation_turn(
            conversation_id=conversation_id,
            user_response=request.message,
            agent_result=agent_result
        )
        
        # Route message to appropriate stakeholders
        routes = await stakeholder_router.route_message(
            conversation_id=conversation_id,
            from_stakeholder=request.stakeholder_type,
            message=request.message,
            medical_outcome=agent_result["outcome"]
        )
        print(routes)  # remove it
        # Handle emergency routing in background
        if agent_result["outcome"] == "emergency":
            background_tasks.add_task(
                handle_emergency_alert,
                conversation_id,
                request.user_id,
                agent_result
            )
        
        # Handle completed conversations
        if agent_result["is_complete"]:
            background_tasks.add_task(
                save_conversation_to_postgresql,
                conversation_id,
                updated_state
            )
        
        response = MultiTurnChatResponse(
            conversation_id=conversation_id,
            agent_response=agent_result.get("next_question", "Thank you for the information."),
            next_question=agent_result.get("next_question"),
            medical_outcome=agent_result["outcome"],
            confidence_score=agent_result["confidence"],
            is_conversation_complete=agent_result["is_complete"],
            turn_number=updated_state["turn_count"],
            red_flags=agent_result.get("red_flags", []),
            reasoning=agent_result["reasoning"],
            estimated_completion_turns=estimate_remaining_turns(updated_state)
        )
        
        logger.info("✅ Multi-turn chat processed successfully",
                   conversation_id=conversation_id,
                   outcome=agent_result["outcome"],
                   turn=updated_state["turn_count"])
        
        return response
        
    except Exception as e:
        logger.error("❌ Error processing multi-turn chat",
                    error=str(e),
                    user_id=request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process conversation"
        )

async def handle_emergency_alert(conversation_id: str, user_id: str, agent_result: Dict):
    """Background task to handle emergency alerts"""
    # In production: send alerts to on-call doctors, trigger escalation workflows
    logger.critical("🚨 EMERGENCY DETECTED",
                    conversation_id=conversation_id,
                    user_id=user_id,
                    red_flags=agent_result.get("red_flags"))

async def save_conversation_to_postgresql(conversation_id: str, state: Dict):
    """Background task to persist completed conversations"""
    # In production: save to PostgreSQL for long-term storage and training
    logger.info("💾 Saving completed conversation",
               conversation_id=conversation_id,
               turns=state["turn_count"])

def estimate_remaining_turns(state: Dict) -> Optional[int]:
    """Estimate how many more questions needed"""
    current_turns = state["turn_count"]
    outcome = state["current_outcome"]
    
    if outcome == "inconclusive" and current_turns < 3:
        return 3 - current_turns
    elif outcome == "inconclusive" and current_turns < 5:
        return 2
    else:
        return 1  # Should conclude soon
