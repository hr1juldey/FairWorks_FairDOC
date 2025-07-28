"""
Conversation State Database Model for Fairdoc AI V2

SQLAlchemy model mirroring Redis ConversationQueue JSON structure
with ORM-friendly columns and helper methods (≤200 LOC)
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Text, JSON, Integer, DateTime, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship
from pydantic import BaseModel

from src.app.core.database import Base
from src.app2.models.schemas.multiturn_chat import (
    ConversationStatus, 
    MedicalOutcome, 
    StakeholderType,
    ConversationState as ConversationStateSchema,
    ConversationTurn as ConversationTurnSchema
)

class ConversationStateDB(Base):
    """
    PostgreSQL model for multi-turn medical conversation persistence
    Mirrors Redis ConversationQueue structure with ORM optimizations
    """
    __tablename__ = "conversation_state_v2"
    
    # Primary identification
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(String(100), nullable=False, unique=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Conversation status and metadata
    status = Column(String(20), nullable=False, default="active", index=True)
    initial_symptoms = Column(Text, nullable=False)
    current_outcome = Column(String(30), nullable=False, default="inconclusive")
    turn_count = Column(Integer, nullable=False, default=0)
    
    # JSON fields for complex data (Redis-style storage)
    conversation_history = Column(JSON, nullable=False, default=list)
    nice_protocols_used = Column(JSON, nullable=False, default=list)
    red_flags_detected = Column(JSON, nullable=False, default=list)
    active_stakeholders = Column(JSON, nullable=False, default=list)
    
    # Flags and booleans
    requires_human_review = Column(Boolean, nullable=False, default=False)
    is_emergency = Column(Boolean, nullable=False, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_activity = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Database indexes for optimized queries
    __table_args__ = (
        Index('idx_conv_user_status', 'user_id', 'status'),
        Index('idx_conv_outcome', 'current_outcome'),
        Index('idx_conv_activity', 'last_activity'),
        Index('idx_conv_emergency', 'is_emergency'),
    )
    
    @classmethod
    def from_redis_state(cls, redis_data: Dict[str, Any]) -> ConversationStateDB:
        """Create DB instance from Redis conversation state"""
        return cls(
            conversation_id=redis_data["conversation_id"],
            user_id=redis_data["user_id"],
            status=redis_data.get("status", "active"),
            initial_symptoms=redis_data["initial_symptoms"],
            current_outcome=redis_data.get("current_outcome", "inconclusive"),
            turn_count=redis_data.get("turn_count", 0),
            conversation_history=redis_data.get("conversation_history", []),
            nice_protocols_used=redis_data.get("nice_protocols_used", []),
            red_flags_detected=redis_data.get("red_flags_detected", []),
            active_stakeholders=redis_data.get("active_stakeholders", ["patient", "fairdoc_agent"]),
            requires_human_review=redis_data.get("requires_human_review", False),
            is_emergency=redis_data.get("current_outcome") == "emergency",
            created_at=datetime.fromisoformat(redis_data["created_at"].replace('Z', '+00:00')),
            last_activity=datetime.fromisoformat(redis_data["last_activity"].replace('Z', '+00:00')),
            completed_at=(
                datetime.fromisoformat(redis_data["completed_at"].replace('Z', '+00:00'))
                if redis_data.get("completed_at") else None
            )
        )
    
    @classmethod
    def from_pydantic(cls, pydantic_state: ConversationStateSchema) -> ConversationStateDB:
        """Create DB instance from Pydantic schema"""
        return cls(
            conversation_id=pydantic_state.conversation_id,
            user_id=pydantic_state.user_id,
            status=pydantic_state.status.value,
            initial_symptoms=pydantic_state.initial_symptoms,
            current_outcome=pydantic_state.current_outcome.value,
            turn_count=pydantic_state.turn_count,
            conversation_history=[turn.model_dump() for turn in pydantic_state.turns],
            nice_protocols_used=pydantic_state.nice_protocols_used,
            red_flags_detected=pydantic_state.red_flags_detected,
            active_stakeholders=[s.value for s in pydantic_state.active_stakeholders],
            requires_human_review=pydantic_state.requires_human_review,
            is_emergency=pydantic_state.current_outcome == MedicalOutcome.EMERGENCY,
            created_at=pydantic_state.created_at,
            last_activity=pydantic_state.last_activity,
            completed_at=pydantic_state.completed_at
        )
    
    def to_pydantic(self) -> ConversationStateSchema:
        """Convert DB instance to Pydantic schema"""
        turns = [
            ConversationTurnSchema(**turn_data) 
            for turn_data in self.conversation_history
        ]
        
        return ConversationStateSchema(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            status=ConversationStatus(self.status),
            initial_symptoms=self.initial_symptoms,
            current_outcome=MedicalOutcome(self.current_outcome),
            turn_count=self.turn_count,
            turns=turns,
            nice_protocols_used=self.nice_protocols_used,
            red_flags_detected=self.red_flags_detected,
            active_stakeholders=[StakeholderType(s) for s in self.active_stakeholders],
            requires_human_review=self.requires_human_review,
            created_at=self.created_at,
            last_activity=self.last_activity,
            completed_at=self.completed_at
        )
    
    def to_redis_format(self) -> Dict[str, Any]:
        """Convert DB instance to Redis-compatible format"""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "status": self.status,
            "turn_count": self.turn_count,
            "initial_symptoms": self.initial_symptoms,
            "current_outcome": self.current_outcome,
            "conversation_history": self.conversation_history,
            "nice_protocols_used": self.nice_protocols_used,
            "red_flags_detected": self.red_flags_detected,
            "active_stakeholders": self.active_stakeholders,
            "requires_human_review": self.requires_human_review,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    def update_from_turn(self, turn_data: Dict[str, Any]) -> None:
        """Update conversation state with new turn data"""
        # Add turn to history
        self.conversation_history.append(turn_data)
        self.turn_count = len(self.conversation_history)
        self.current_outcome = turn_data.get("outcome", "inconclusive")
        self.last_activity = datetime.now(timezone.utc)
        
        # Update emergency flag
        self.is_emergency = self.current_outcome == "emergency"
        
        # Mark for human review if needed
        if self.current_outcome in ["emergency", "routine_doctor"]:
            self.requires_human_review = True
        
        # Mark completion
        if self.current_outcome in ["emergency", "routine_doctor", "self_care", "spam_detected"]:
            self.status = "completed"
            self.completed_at = datetime.now(timezone.utc)
    
    def get_latest_turn(self) -> Optional[Dict[str, Any]]:
        """Get the most recent conversation turn"""
        return self.conversation_history[-1] if self.conversation_history else None
    
    def is_complete(self) -> bool:
        """Check if conversation has reached completion"""
        return self.status == "completed"
    
    def __repr__(self) -> str:
        return (
            f"<ConversationStateDB("
            f"id={self.conversation_id}, "
            f"user={self.user_id}, "
            f"status={self.status}, "
            f"turns={self.turn_count})>"
        )
