"""
Chat Orchestration Service - Business Logic Layer

Handles multi-turn medical conversation workflow coordination
between DSPy agents, Redis state, NICE protocols, and stakeholder routing.

Single responsibility: Orchestrate medical chat business logic
"""

from typing import Optional, Dict, Any
import structlog
from datetime import datetime
import dspy
from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatRequest,
    MultiTurnChatResponse,
    ConversationState,
    ConversationTurn,
    MedicalOutcome,
    ConversationStatus
)
from src.app2.models.schemas.medical_triage import (

    RedFlagIndicator
)
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.context.redis_queue import ConversationQueue
from src.app2.services.context.nice_lookup import NICELookupService
from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
from src.app2.services.chat.stakeholder_router import StakeholderRouter
from src.app2.core.config_v2 import settings_v2
from src.app2.core.dspy_config_v2 import ensure_dspy_configured
# Add at top
from src.app2.utils.conversation_logger import conversation_logger

logger = structlog.get_logger(__name__)

class ChatOrchestrator:
    """
    Medical chat workflow orchestrator
    
    Coordinates DSPy agent, Redis state, NICE protocols, and stakeholder routing
    for multi-turn medical conversations
    """
    
    def __init__(self, question_generator=None):
        # Initialize service dependencies
        
        
        # NEW: Ensure DSPy is configured centrally before creating any DSPy agents

        model_name = settings_v2.FAIRDOC_V2_DSPy_MODEL
        
        # More resilient DSPy configuration - don't fail completely if async context issues
        try:
            if not ensure_dspy_configured(model_name):
                logger.warning("⚠️ DSPy configuration returned False, but continuing with initialization")
        except Exception as e:
            logger.warning(f"⚠️ DSPy configuration issue in async context: {e}. Continuing with initialization.")

        
        # ADD: Initialize missing attributes BEFORE any other setup
        self.conversation_queue = ConversationQueue()
        self.nice_lookup = NICELookupService()
        self.stakeholder_router = StakeholderRouter()
        
        # Initialize components - they will handle their own DSPy configuration if needed
        if question_generator is None:
            try:
                question_generator = MedicalQuestionGenerator(model_name=model_name)
            except Exception as e:
                logger.warning(f"⚠️ Question generator initialization issue: {e}")
                question_generator = None  # Will create fallback later

        try:
            self.medical_agent = MedicalTriageAgent(
                model_name=model_name,
                question_generator=question_generator
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize medical agent: {e}")
            # Create a minimal fallback that won't break the orchestrator
            self.medical_agent = None

    
    async def initialize(self) -> None:
        """Initialize async service components"""
        await self.conversation_queue.initialize()
        logger.info("✅ Chat Orchestrator ready")
    

    def _build_validated_history(self, conversation_state: Dict) -> dspy.History:
        """FIX: Build validated conversation history"""
        history = dspy.History(messages=[])
        
        for turn in conversation_state.get("conversation_history", []):
            # Validate turn data before adding
            if self._is_valid_turn(turn):
                history.messages.append({
                    "turn": turn.get("turn", 0),
                    "current_symptoms": turn.get("user_response", ""),
                    "outcome_classification": turn.get("outcome", "inconclusive"),
                    "reasoning": turn.get("agent_reasoning", ""),
                    "confidence": turn.get("confidence", 50),
                    "red_flags": turn.get("red_flags", [])
                })
        
        return history

    

    def _is_valid_turn(self, turn: Dict) -> bool:
        """Validate turn data structure"""
        required_fields = ["turn", "user_response", "outcome"]
        return all(field in turn for field in required_fields)


    async def process_conversation_turn(self, request: MultiTurnChatRequest) -> Dict[str, Any]:
        """Process one turn of medical conversation with proper context management"""
        
        logger.info("🩺 Processing conversation turn",
                    user_id=request.stakeholder_id,
                    conversation_id=request.conversation_id,
                    stakeholder=request.stakeholder_role)

        # Step 0: Ensure medical agent is available
        if self.medical_agent is None:
            logger.error("❌ Medical agent not initialized")
            return {
                "conversation_id": str(request.conversation_id or "unknown"),
                "agent_result": {
                    "outcome": "inconclusive",
                    "confidence": 30,
                    "next_question": "I'm experiencing technical difficulties. Please try again.",
                    "reasoning": "System initialization error",
                    "red_flags": [],
                    "is_complete": False
                },
                "error": "Medical agent not available",
                "message_routes": [],  # ADDED
                "context_maintained": False,  # ADDED
                "requires_emergency_alert": False,
                "requires_persistence": False
            }

        # Step 1: Get or create conversation
        conversation_id = await self._get_or_create_conversation(request)

        # Step 2: Retrieve conversation state AND HISTORY
        conversation_state = await self.conversation_queue.get_conversation_state(conversation_id)
        if not conversation_state:
            raise ValueError(f"Conversation {conversation_id} not found")

        # ✅ FIX: Build proper DSPy history with validation
        conversation_history = self._build_validated_history(conversation_state)

        # Step 3: Look up NICE protocols with enhanced matching
        nice_context = self.nice_lookup.find_relevant_protocols(request.user_message)

        # Step 4: Process with DSPy medical agent WITH PROPER ASYNC CONTEXT
        try:
            # ✅ FIX: Use context manager for async DSPy operations
            from src.app2.core.dspy_config_v2 import get_llm_provider            
            llm_provider = get_llm_provider()
            llm_instance = llm_provider.get_llm(settings_v2.DSPY_MODEL_NAME)
            
            # Use context manager instead of configure in async environment
            with dspy.context(lm=llm_instance):
                agent_result = await self.medical_agent.process_turn(
                    symptoms=request.user_message,
                    nice_context=nice_context["protocol_text"],
                    history=conversation_history  # ✅ PASS VALIDATED CONTEXT
                )

        except Exception as e:
            logger.error(f"❌ Medical agent processing error: {e}")
            # Provide fallback response to prevent conversation failure
            agent_result = {
                "outcome": "inconclusive",
                "confidence": 40,
                "next_question": "I need more information about your symptoms. Could you describe them in more detail?",
                "reasoning": f"Processing error handled: {str(e)[:50]}",
                "red_flags": [],
                "is_complete": False,
                "error_handled": True
            }

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
                    turn=updated_state["turn_count"],
                    context_turns=len(conversation_history.messages))  # ✅ LOG CONTEXT
        
        # ADD: Clean conversation logging
        try:
            conversation_logger.log_conversation_turn(
                conversation_id=conversation_id,
                turn_number=updated_state["turn_count"],
                patient_message=request.user_message,
                agent_response=agent_result.get("next_question", "Assessment complete"),
                medical_assessment=agent_result,
                patient_info={
                    "age": getattr(request, 'patient_age', None),
                    "gender": getattr(request, 'patient_gender', None),
                    "stakeholder": request.stakeholder_role
                }
            )
        except Exception as e:
            logger.warning(f"Conversation logging failed: {e}")
        return {
            "conversation_id": conversation_id,
            "agent_result": agent_result,
            "updated_state": updated_state,
            "nice_context": nice_context,
            "message_routes": message_routes,
            "context_maintained": len(conversation_history.messages) > 0,  # ✅ TRACK CONTEXT
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
        health = {}
        
        # Check Redis
        try:
            await self.conversation_queue.redis.ping()
            health["redis"] = "connected"
        except Exception as e:
            health["redis"] = f"error: {str(e)}"
        
        # Check Medical Agent
        if self.medical_agent is not None:
            health["dspy_agent"] = "initialized"
        else:
            health["dspy_agent"] = "not_initialized"
        
        # Check other services
        try:
            health["nice_lookup"] = "ready"
            health["stakeholder_router"] = "ready"
        except Exception as e:
            health["services"] = f"error: {str(e)}"
        
        # Overall status
        if any("error" in str(v) or "not_initialized" in str(v) for v in health.values()):
            health["overall_status"] = "degraded" 
        else:
            health["overall_status"] = "healthy"
        
        return health

    
    def build_chat_response(self, orchestration_result: Dict[str, Any]) -> MultiTurnChatResponse:
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

        # Convert red flags to proper format
        red_flags_detected = []
        for flag in agent_result.get("red_flags", []):
            if isinstance(flag, str):
                red_flags_detected.append(RedFlagIndicator(symptom=flag, critical=True))
            else:
                red_flags_detected.append(flag)

        # Determine conversation status
        if agent_result.get("is_complete", False):
            conv_status = ConversationStatus.COMPLETED
        elif medical_outcome == MedicalOutcome.EMERGENCY:
            conv_status = ConversationStatus.ESCALATED
        else:
            conv_status = ConversationStatus.IN_PROGRESS

        return MultiTurnChatResponse(
            conversation_id=conversation_id,
            agent_message=agent_result.get("next_question", "Thank you for the information."),
            next_question=agent_result.get("next_question"),
            medical_outcome=medical_outcome,
            confidence_score=float(agent_result["confidence"]),
            red_flags_detected=red_flags_detected,  # ✅ CORRECT FIELD
            requires_human_review=medical_outcome in [MedicalOutcome.EMERGENCY, MedicalOutcome.ROUTINE_DOCTOR],
            is_emergency=medical_outcome == MedicalOutcome.EMERGENCY,
            conversation_status=conv_status,  # ✅ CORRECT FIELD
            turn_count=updated_state["turn_count"],  # ✅ CORRECT FIELD
            relevant_protocols=orchestration_result.get("nice_context", {}).get("protocol_code", "").split(","),
            notify_stakeholders=[],  # Add proper logic later
            processing_time_ms=orchestration_result.get("processing_time_ms"),
            model_version="v2.6-stable"
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
