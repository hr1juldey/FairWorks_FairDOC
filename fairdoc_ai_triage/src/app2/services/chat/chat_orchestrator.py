"""
Chat Orchestration Service - Business Logic Layer

Handles multi-turn medical conversation workflow coordination
between DSPy agents, Redis state, NICE protocols, and stakeholder routing.

Single responsibility: Orchestrate medical chat business logic
"""

from typing import Optional, Dict, Any
import structlog
from datetime import datetime

from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatRequest,
    MultiTurnChatResponse,
    ConversationState,
    ConversationTurn,
    MedicalOutcome
)
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.context.redis_queue import ConversationQueue
from src.app2.services.context.nice_lookup import NICELookupService
from src.app2.services.chat.stakeholder_router import StakeholderRouter
from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)

class ChatOrchestrator:
    """
    Medical chat workflow orchestrator
    
    Coordinates DSPy agent, Redis state, NICE protocols, and stakeholder routing
    for multi-turn medical conversations
    """
    
    def __init__(self, question_generator=None):
        # Initialize service dependencies
        from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
        if question_generator is None:
            question_generator = MedicalQuestionGenerator(model_name=settings_v2.FAIRDOC_V2_DSPy_MODEL)
        
        self.medical_agent = MedicalTriageAgent(
            model_name=settings_v2.FAIRDOC_V2_DSPy_MODEL,
            question_generator=question_generator
        )    
        self.conversation_queue = ConversationQueue()
        self.nice_lookup = NICELookupService()
        self.stakeholder_router = StakeholderRouter()
        logger.info("🏥 Chat Orchestrator initialized")
    
    async def initialize(self) -> None:
        """Initialize async service components"""
        await self.conversation_queue.initialize()
        logger.info("✅ Chat Orchestrator ready")
    
    async def process_conversation_turn(
        self,
        request: MultiTurnChatRequest
    ) -> Dict[str, Any]:
        """
        Process one turn of medical conversation
        
        Returns orchestration result with all necessary data
        for API response building
        """
        logger.info("🩺 Processing conversation turn",
                   user_id=request.stakeholder_id,
                   conversation_id=request.conversation_id,
                   stakeholder=request.stakeholder_role)
        
        # Step 1: Get or create conversation
        conversation_id = await self._get_or_create_conversation(request)
        
        # Step 2: Retrieve conversation state
        conversation_state = await self.conversation_queue.get_conversation_state(conversation_id)
        if not conversation_state:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Step 3: Look up NICE protocols
        nice_context = self.nice_lookup.find_relevant_protocols(request.user_message)
        
        # Step 4: Process with DSPy medical agent
        agent_result = await self.medical_agent.process_turn(
            symptoms=request.user_message,
            nice_context=nice_context["protocol_text"]
        )
        
        # Step 5: Update conversation state
        updated_state = await self.conversation_queue.update_conversation_turn(
            conversation_id=conversation_id,
            user_response=request.user_message,
            agent_result=agent_result
        )

        
        # Step 6: Route messages to stakeholders
        message_routes = await self.stakeholder_router.route_message(
            conversation_id=conversation_id,
            from_stakeholder=request.stakeholder_role,
            message=request.user_message,
            medical_outcome=agent_result["outcome"]
        )
        
        logger.info("✅ Conversation turn processed",
                   conversation_id=conversation_id,
                   outcome=agent_result["outcome"],
                   turn=updated_state["turn_count"])
        
        return {
            "conversation_id": conversation_id,
            "agent_result": agent_result,
            "updated_state": updated_state,
            "nice_context": nice_context,
            "message_routes": message_routes,
            "requires_emergency_alert": agent_result["outcome"] == "emergency",
            "requires_persistence": (
                agent_result.get("is_complete", False) or 
                updated_state["status"] == "completed"
            )
        }
    
    async def get_conversation_state(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation state from Redis"""
        return await self.conversation_queue.get_conversation_state(conversation_id)
    
    async def reset_conversation(self, conversation_id: str) -> None:
        """Reset conversation state"""
        state_key = f"{self.conversation_queue.state_prefix}:{conversation_id}"
        await self.conversation_queue.redis.delete(state_key)
        self.medical_agent.reset_conversation()
        
        logger.info("🔄 Conversation reset", conversation_id=conversation_id)
    
    async def check_service_health(self) -> Dict[str, str]:
        """Check health of all orchestrated services"""
        try:
            await self.conversation_queue.redis.ping()
            return {
                "redis": "connected",
                "dspy_agent": "initialized", 
                "nice_lookup": "ready",
                "stakeholder_router": "ready"
            }
        except Exception as e:
            return {
                "redis": f"error: {str(e)}",
                "overall_status": "unhealthy"
            }
    
    def build_chat_response(
        self,
        orchestration_result: Dict[str, Any]
    ) -> MultiTurnChatResponse:
        """Build structured chat response from orchestration result"""
        
        agent_result = orchestration_result["agent_result"]
        updated_state = orchestration_result["updated_state"]
        conversation_id = orchestration_result["conversation_id"]
        
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
    
    async def _get_or_create_conversation(self, request: MultiTurnChatRequest) -> str:
        """Get existing conversation or create new one"""
        if request.conversation_id:
            # Check if conversation exists in Redis
            existing_state = await self.conversation_queue.get_conversation_state(str(request.conversation_id))
            if existing_state:
                return str(request.conversation_id)
            else:
                # Conversation ID provided but doesn't exist - create it
                conversation_id = await self.conversation_queue.start_conversation(
                    user_id=request.stakeholder_id,
                    initial_symptoms=request.user_message
                )
                # Update the conversation with the requested ID
                await self.conversation_queue._override_conversation_id(
                    old_id=conversation_id,
                    new_id=str(request.conversation_id)
                )
                return str(request.conversation_id)
        else:
            # Create new conversation
            return await self.conversation_queue.start_conversation(
                user_id=request.stakeholder_id,
                initial_symptoms=request.user_message
            )


    
    def _estimate_remaining_turns(self, state: dict) -> Optional[int]:
        """Estimate remaining conversation turns based on current state"""
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
