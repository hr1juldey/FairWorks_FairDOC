"""
V2 Conversation State Database Model
SQLAlchemy ORM model for persisting conversation state
Mirrors Redis ConversationQueue for long-term storage
File: src/app2/models/database/conversation_state.py
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, Integer, Float, Boolean, 
    Text, JSON, ForeignKey, Index, CheckConstraint, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

from src.app2.models.schemas.medical_triage import MedicalOutcome, RedFlagIndicator
from src.app2.models.schemas.multiturn_chat import ConversationStatus, StakeholderRole, ChatProvider

# Base class for all V2 database models
Base = declarative_base()


class ConversationStateV2(Base):
    """
    Main conversation state table for V2 system
    Mirrors Redis queue structure for PostgreSQL persistence
    Optimized for emergency retrieval and analytics
    """
    __tablename__ = "conversation_states_v2"
    
    # Primary identifiers
    conversation_id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        doc="Unique conversation identifier matching Redis queue"
    )
    
    # Stakeholder information
    stakeholder_role = Column(
        SQLEnum(StakeholderRole, name="stakeholder_role_enum"),
        nullable=False,
        default=StakeholderRole.PATIENT,
        doc="Primary stakeholder for this conversation"
    )
    stakeholder_id = Column(
        String(100),
        nullable=True,
        index=True,
        doc="External stakeholder ID (phone, user_id, etc.)"
    )
    
    # Chat platform context
    chat_provider = Column(
        SQLEnum(ChatProvider, name="chat_provider_enum"),
        nullable=False,
        default=ChatProvider.API_DIRECT,
        doc="Which chat platform originated this conversation"
    )
    provider_metadata = Column(
        JSONB,
        nullable=False,
        default={},
        doc="Provider-specific context (phone numbers, chat IDs)"
    )
    
    # Conversation lifecycle
    status = Column(
        SQLEnum(ConversationStatus, name="conversation_status_enum"),
        nullable=False,
        default=ConversationStatus.NEW,
        index=True,
        doc="Current conversation state"
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        index=True,
        doc="When conversation was initiated"
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        onupdate=func.now(),
        index=True,
        doc="Last modification timestamp"
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="When conversation reached final state"
    )
    
    # Patient demographics
    patient_age = Column(
        Integer,
        nullable=True,
        doc="Patient age for clinical context"
    )
    patient_gender = Column(
        String(20),
        nullable=True,
        doc="Patient gender"
    )
    
    # Triage assessment results
    final_outcome = Column(
        SQLEnum(MedicalOutcome, name="medical_outcome_enum"),
        nullable=True,
        index=True,
        doc="Final triage decision if conversation completed"
    )
    confidence_score = Column(
        Float,
        nullable=False,
        default=0.0,
        doc="Final confidence in triage assessment (0-100)"
    )
    
    # Safety indicators
    red_flags_detected = Column(
        JSONB,
        nullable=False,
        default=[],
        doc="JSON array of RedFlagIndicator values"
    )
    requires_human_review = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="Flag for human clinician review needed"
    )
    is_emergency = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="Emergency escalation flag"
    )
    emergency_notified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="When emergency alerts were sent"
    )
    
    # Conversation metrics
    turn_count = Column(
        Integer,
        nullable=False,
        default=1,
        doc="Total number of conversational turns"
    )
    total_processing_time_ms = Column(
        Integer,
        nullable=True,
        doc="Cumulative processing time across all turns"
    )
    
    # NICE protocol context
    relevant_protocols = Column(
        JSONB,
        nullable=False,
        default=[],
        doc="JSON array of matching NICE protocol IDs"
    )
    
    # Conversation history (denormalized for performance)
    conversation_turns = Column(
        JSONB,
        nullable=False,
        default=[],
        doc="Complete conversation history as JSON array"
    )
    
    # Model versioning
    model_version = Column(
        String(20),
        nullable=False,
        default="v2.6-stable",
        doc="Version of triage system used"
    )
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 100', 
                       name='valid_confidence_score'),
        CheckConstraint('patient_age IS NULL OR (patient_age >= 0 AND patient_age <= 120)', 
                       name='valid_patient_age'),
        CheckConstraint('turn_count >= 1', 
                       name='valid_turn_count'),
        
        # Indexes for common queries
        Index('idx_conversations_emergency', 'is_emergency', 'created_at'),
        Index('idx_conversations_status_updated', 'status', 'updated_at'),
        Index('idx_conversations_stakeholder', 'stakeholder_id', 'chat_provider'),
        Index('idx_conversations_review_pending', 'requires_human_review', 'created_at'),
        Index('idx_conversations_outcome', 'final_outcome', 'completed_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'conversation_id': str(self.conversation_id),
            'stakeholder_role': self.stakeholder_role.value if self.stakeholder_role else None,
            'stakeholder_id': self.stakeholder_id,
            'chat_provider': self.chat_provider.value if self.chat_provider else None,
            'status': self.status.value if self.status else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'final_outcome': self.final_outcome.value if self.final_outcome else None,
            'confidence_score': self.confidence_score,
            'turn_count': self.turn_count,
            'is_emergency': self.is_emergency,
            'requires_human_review': self.requires_human_review,
            'relevant_protocols': self.relevant_protocols or []
        }

    def add_conversation_turn(self, turn_data: Dict[str, Any]) -> None:
        """Add a new turn to the conversation history"""
        turns = list(self.conversation_turns or [])
        turns.append(turn_data)
        self.conversation_turns = turns
        self.turn_count = len(turns)
        self.updated_at = func.now()

    def mark_emergency(self) -> None:
        """Mark conversation as emergency and update flags"""
        self.is_emergency = True
        self.requires_human_review = True
        self.emergency_notified_at = func.now()
        self.updated_at = func.now()

    def complete_conversation(self, final_outcome: MedicalOutcome, confidence: float) -> None:
        """Mark conversation as completed with final assessment"""
        self.status = ConversationStatus.COMPLETED
        self.final_outcome = final_outcome
        self.confidence_score = confidence
        self.completed_at = func.now()
        self.updated_at = func.now()

    @classmethod
    def get_active_conversations(cls, session) -> List['ConversationStateV2']:
        """Get all active (non-completed) conversations"""
        return session.query(cls).filter(
            cls.status.in_([
                ConversationStatus.NEW,
                ConversationStatus.IN_PROGRESS,
                ConversationStatus.AWAITING_RESPONSE
            ])
        ).order_by(cls.updated_at.desc()).all()

    @classmethod
    def get_emergency_conversations(cls, session, hours_back: int = 24) -> List['ConversationStateV2']:
        """Get recent emergency conversations for review"""
        cutoff = func.now() - func.interval(f'{hours_back} hours')
        return session.query(cls).filter(
            cls.is_emergency,
            cls.created_at >= cutoff
        ).order_by(cls.created_at.desc()).all()

    @classmethod
    def get_pending_review(cls, session) -> List['ConversationStateV2']:
        """Get conversations awaiting human review"""
        return session.query(cls).filter(
            cls.requires_human_review,
            cls.status != ConversationStatus.COMPLETED
        ).order_by(cls.created_at.asc()).all()

    def __repr__(self) -> str:
        return f"<ConversationStateV2(id={self.conversation_id}, status={self.status}, turns={self.turn_count})>"
