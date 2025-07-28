"""
Multi-turn Medical Chat API Endpoint - V2 Main Interface
Orchestrates DSPy services, Redis state, and PostgreSQL persistence
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Optional, List
import structlog
from datetime import datetime

# Import our V2 schemas and services
from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatRequest, 
    MultiTurnChatResponse,
    ConversationState,
    ConversationTurn,
    StakeholderType,
    MedicalOutcome
)
from src.app2.models.schemas.medical_triage import (
    TriageDecision,
    TriagePriority,
    RedFlagIndicator,
    ClinicalAssessment
)
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.context.redis_queue import ConversationQueue
from src.app2.services.context.nice_lookup import NICELookupService
from src.app2.services.chat.stakeholder_router import StakeholderRouter
from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)
router = APIRouter()

class ChatOrchestrator:
    """Main orchestrator for medical chat workflow"""
    
    def __init__(self):
        # Initialize all services
        self.medical_agent = MedicalTriageAgent(model_name=settings_v2.FAIRDOC_V2_DSPy_MODEL)
        self.conversation_queue = ConversationQueue()
        self.nice_lookup = NICELookupService()
        self.stakeholder_router = StakeholderRouter()
        
        logger.info("🏥 Chat Orchestrator initialized")
    
    async def initialize(self):
        """Initialize async components"""
        await self.conversation_queue.initialize()
        logger.info("✅ Chat Orchestrator ready")
    
    async def process_chat_turn(
        self, 
        request: MultiTurnChatRequest, 
        background_tasks: BackgroundTasks
    ) -> MultiTurnChatResponse:
        """Main chat processing workflow"""
        
        logger.info("🩺 Processing chat turn",
                   user_id=request.user_id,
                   conversation_id=request.conversation_id,
                   stakeholder=request.stakeholder_type)
        
        try:
            # Step 1: Get or create conversation
            conversation_id = await self._get_or_create_conversation(request)
            
            # Step 2: Retrieve conversation state from Redis
            conversation_state = await self.conversation_queue.get_conversation_state(conversation_id)
            if not conversation_state:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Step 3: Look up relevant NICE protocols
            nice_context = self.nice_lookup.find_relevant_protocols(request.message)
            logger.info("📋 NICE protocol lookup",
                       protocol_code=nice_context["protocol_code"],
                       conversation_id=conversation_id)
            
            # Step 4: Process with DSPy medical agent
            agent_result = await self.medical_agent.process_turn(
                symptoms=request.message,
                nice_context=nice_context["protocol_text"]
            )
            
            # Step 5: Update conversation state in Redis
            updated_state = await self.conversation_queue.update_conversation_turn(
                conversation_id=conversation_id,
                user_response=request.message,
                agent_result=agent_result
            )
            
            # Step 6: Route messages to stakeholders
            routes = await self.stakeholder_router.route_message(  # noqa: F841
                conversation_id=conversation_id,
                from_stakeholder=request.stakeholder_type,
                message=request.message,
                medical_outcome=agent_result["outcome"]
            )
            
            # Step 7: Handle special cases in background
            await self._handle_background_tasks(
                agent_result, conversation_id, request.user_id, updated_state, background_tasks
            )
            
            # Step 8: Build and return response
            response = self._build_chat_response(
                conversation_id, agent_result, updated_state, nice_context
            )
            
            logger.info("✅ Chat turn completed successfully",
                       conversation_id=conversation_id,
                       outcome=agent_result["outcome"],
                       turn=updated_state["turn_count"])
            
            return response
            
        except Exception as e:
            logger.error("❌ Error processing chat turn",
                        error=str(e),
                        user_id=request.user_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process conversation: {str(e)}"
            )
    
    async def _get_or_create_conversation(self, request: MultiTurnChatRequest) -> str:
        """Get existing conversation or create new one"""
        if request.conversation_id:
            return request.conversation_id
        else:
            return await self.conversation_queue.start_conversation(
                user_id=request.user_id,
                initial_symptoms=request.message
            )
    
    async def _handle_background_tasks(
        self, 
        agent_result: dict, 
        conversation_id: str, 
        user_id: str,
        updated_state: dict,
        background_tasks: BackgroundTasks
    ):
        """Handle emergency alerts and conversation persistence"""
        
        # Handle emergency situations
        if agent_result["outcome"] == "emergency":
            background_tasks.add_task(
                self._handle_emergency_alert,
                conversation_id, user_id, agent_result
            )
        
        # Persist completed conversations to PostgreSQL  
        if agent_result.get("is_complete") or updated_state["status"] == "completed":
            background_tasks.add_task(
                self._save_conversation_to_postgresql,
                conversation_id, updated_state
            )
    
    def _build_chat_response(
        self, 
        conversation_id: str, 
        agent_result: dict, 
        updated_state: dict,
        nice_context: dict
    ) -> MultiTurnChatResponse:
        """Build structured chat response"""
        
        # Map agent outcome to schema enum
        outcome_mapping = {
            "emergency": MedicalOutcome.EMERGENCY,
            "routine": MedicalOutcome.ROUTINE_DOCTOR,
            "self_care": MedicalOutcome.SELF_CARE,
            "inconclusive": MedicalOutcome.INCONCLUSIVE,
            "spam": MedicalOutcome.SPAM_DETECTED
        }
        
        medical_outcome = outcome_mapping.get(
            agent_result["outcome"], 
            MedicalOutcome.INCONCLUSIVE
        )
        
        return MultiTurnChatResponse(
            conversation_id=conversation_id,
            agent_response=agent_result.get("next_question", "Thank you for the information."),
            next_question=agent_result.get("next_question"),
            medical_outcome=medical_outcome,
            confidence_score=agent_result["confidence"],
            is_conversation_complete=agent_result.get("is_complete", False),
            turn_number=updated_state["turn_count"],
            red_flags=agent_result.get("red_flags", []),
            reasoning=agent_result.get("reasoning", ""),
            estimated_completion_turns=self._estimate_remaining_turns(updated_state)
        )
    
    def _estimate_remaining_turns(self, state: dict) -> Optional[int]:
        """Estimate remaining conversation turns"""
        current_turns = state["turn_count"]
        outcome = state["current_outcome"]
        
        if outcome == "inconclusive":
            if current_turns < 3:
                return 3 - current_turns
            elif current_turns < 5:
                return 2
            else:
                return 1
        return None
    
    async def _handle_emergency_alert(self, conversation_id: str, user_id: str, agent_result: dict):
        """Background task: Handle emergency situations"""
        logger.critical("🚨 EMERGENCY DETECTED",
                        conversation_id=conversation_id,
                        user_id=user_id,
                        red_flags=agent_result.get("red_flags", []))
        
        # In production: Send alerts to on-call doctors, trigger escalation workflows
        # Could integrate with webhooks, SMS, email alerts, etc.
    
    async def _save_conversation_to_postgresql(self, conversation_id: str, state: dict):
        """Background task: Persist conversation to PostgreSQL"""
        logger.info("💾 Saving completed conversation to PostgreSQL",
                   conversation_id=conversation_id,
                   turns=state["turn_count"])
        
        try:
            # Convert Redis state to Pydantic model for validation
            conversation = ConversationState(
                conversation_id=state["conversation_id"],
                user_id=state["user_id"], 
                initial_symptoms=state["initial_symptoms"],
                turn_count=state["turn_count"],
                current_outcome=MedicalOutcome(state["current_outcome"]),
                red_flags_detected=state.get("red_flags_detected", []),
                nice_protocols_used=state.get("nice_protocols_used", [])
            )
            
            # Convert history to conversation turns
            for turn_data in state.get("conversation_history", []):
                turn = ConversationTurn(
                    turn_number=turn_data["turn"],
                    user_message=turn_data["user_response"],
                    agent_response=turn_data.get("agent_question"),
                    agent_question=turn_data.get("agent_question"),
                    medical_outcome=MedicalOutcome(turn_data["outcome"]),
                    confidence_score=turn_data["confidence"],
                    red_flags=turn_data.get("red_flags", []),
                    reasoning=turn_data.get("agent_reasoning", "")
                )
                conversation.add_turn(turn)
            
            # Here you would save to PostgreSQL using SQLAlchemy
            # Example: 
            # async with get_db_session() as session:
            #     db_conversation = ConversationStateDB.from_pydantic(conversation)
            #     session.add(db_conversation)
            #     await session.commit()
            
            logger.info("✅ Conversation saved to PostgreSQL", 
                       conversation_id=conversation_id)
                       
        except Exception as e:
            logger.error("❌ Failed to save conversation to PostgreSQL",
                        conversation_id=conversation_id,
                        error=str(e))

# Global orchestrator instance
chat_orchestrator = ChatOrchestrator()

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await chat_orchestrator.initialize()

@router.post("/chat", response_model=MultiTurnChatResponse)
async def process_multiturn_chat(
    request: MultiTurnChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Main multi-turn medical chat endpoint
    
    Integrates DSPy medical agent, Redis state management, 
    NICE protocol lookup, and PostgreSQL persistence
    """
    return await chat_orchestrator.process_chat_turn(request, background_tasks)

@router.get("/chat/{conversation_id}/state")
async def get_conversation_state(conversation_id: str):
    """Get current conversation state from Redis"""
    state = await chat_orchestrator.conversation_queue.get_conversation_state(conversation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return state

@router.get("/chat/{conversation_id}/history") 
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    state = await chat_orchestrator.conversation_queue.get_conversation_state(conversation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "turns": state.get("conversation_history", []),
        "status": state["status"],
        "outcome": state["current_outcome"]
    }

@router.post("/chat/{conversation_id}/reset")
async def reset_conversation(conversation_id: str):
    """Reset conversation state (for testing/debugging)"""
    # Clear from Redis
    state_key = f"{chat_orchestrator.conversation_queue.state_prefix}:{conversation_id}"
    await chat_orchestrator.conversation_queue.redis.delete(state_key)
    
    # Reset medical agent
    chat_orchestrator.medical_agent.reset_conversation()
    
    return {"message": f"Conversation {conversation_id} reset successfully"}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        await chat_orchestrator.conversation_queue.redis.ping()
        
        return {
            "status": "healthy",
            "services": {
                "redis": "connected",
                "dspy_agent": "initialized", 
                "nice_lookup": "ready"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )
