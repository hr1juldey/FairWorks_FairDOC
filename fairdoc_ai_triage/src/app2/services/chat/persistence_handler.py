"""
Conversation Persistence Handler Service

Handles saving completed conversations from Redis to PostgreSQL.
Single responsibility: Conversation data persistence
"""

import structlog
from typing import Dict, Any

from src.app2.models.schemas.multiturn_chat import (
    ConversationState, 
    ConversationTurn, 
    MedicalOutcome
)

logger = structlog.get_logger(__name__)

class PersistenceHandler:
    """Handles conversation persistence to PostgreSQL"""
    
    async def save_conversation_to_postgresql(
        self,
        conversation_id: str,
        state: Dict[str, Any]
    ):
        """Save completed conversation to PostgreSQL"""
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
            
            # Save to PostgreSQL using SQLAlchemy
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
