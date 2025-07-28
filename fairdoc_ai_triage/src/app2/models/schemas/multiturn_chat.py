"""
Multi-turn Chat Schema Models for Fairdoc AI V2
Pydantic v2 models for conversation management and API contracts
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

class StakeholderType(str, Enum):
    """Types of stakeholders in medical conversations"""
    PATIENT = "patient"
    DOCTOR = "doctor" 
    ADMIN = "admin"
    FAIRDOC_AGENT = "fairdoc_agent"

class MessagePriority(str, Enum):
    """Message routing priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"

class ConversationStatus(str, Enum):
    """Conversation lifecycle status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"

class MedicalOutcome(str, Enum):
    """Medical triage outcome classifications"""
    EMERGENCY = "emergency"
    ROUTINE_DOCTOR = "routine_doctor"
    SELF_CARE = "self_care"
    INCONCLUSIVE = "inconclusive"
    SPAM_DETECTED = "spam_detected"

# === Request/Response Models for API ===

class MultiTurnChatRequest(BaseModel):
    """Request model for multi-turn chat API"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    user_id: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="Unique identifier for the patient/user"
    )
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="User's message or symptom description"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Existing conversation ID for follow-up messages"
    )
    stakeholder_type: StakeholderType = Field(
        default=StakeholderType.PATIENT,
        description="Type of stakeholder sending the message"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message_content(cls, v: str) -> str:
        """Ensure message has meaningful content"""
        if not v.strip():
            raise ValueError("Message cannot be empty or only whitespace")
        return v.strip()

class MultiTurnChatResponse(BaseModel):
    """Response model for multi-turn chat API"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    conversation_id: str = Field(
        ...,
        description="Unique conversation identifier"
    )
    agent_response: Optional[str] = Field(
        None,
        description="Agent's response message to the user"
    )
    next_question: Optional[str] = Field(
        None,
        description="Next question to ask patient, null if conversation complete"
    )
    medical_outcome: MedicalOutcome = Field(
        ...,
        description="Current medical triage classification"
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence level in the medical outcome (0-100)"
    )
    is_conversation_complete: bool = Field(
        ...,
        description="Whether the conversation has reached a conclusion"
    )
    turn_number: int = Field(
        ...,
        ge=1,
        description="Current turn number in the conversation"
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="List of concerning symptoms detected"
    )
    reasoning: str = Field(
        default="",
        description="Medical reasoning behind the outcome classification"
    )
    estimated_completion_turns: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated number of turns remaining"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )

# === Internal Conversation Models ===

class ConversationTurn(BaseModel):
    """Individual turn in a multi-turn conversation"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    turn_number: int = Field(..., ge=1)
    user_message: str = Field(..., min_length=1)
    agent_response: Optional[str] = None
    agent_question: Optional[str] = None
    medical_outcome: MedicalOutcome
    confidence_score: int = Field(..., ge=0, le=100)
    red_flags: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    nice_protocol_used: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ConversationState(BaseModel):
    """Complete state of a multi-turn medical conversation"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    conversation_id: str = Field(
        default_factory=lambda: f"conv_{uuid4().hex[:12]}",
        description="Unique conversation identifier"
    )
    user_id: str = Field(..., min_length=1)
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    
    # Conversation metadata
    initial_symptoms: str = Field(..., min_length=1)
    current_outcome: MedicalOutcome = Field(default=MedicalOutcome.INCONCLUSIVE)
    turn_count: int = Field(default=0, ge=0)
    
    # Conversation history
    turns: List[ConversationTurn] = Field(default_factory=list)
    nice_protocols_used: List[str] = Field(default_factory=list)
    red_flags_detected: List[str] = Field(default_factory=list)
    
    # Stakeholder tracking
    active_stakeholders: List[StakeholderType] = Field(
        default_factory=lambda: [StakeholderType.PATIENT, StakeholderType.FAIRDOC_AGENT]
    )
    requires_human_review: bool = Field(default=False)
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new turn to the conversation"""
        self.turns.append(turn)
        self.turn_count = len(self.turns)
        self.current_outcome = turn.medical_outcome
        self.last_activity = datetime.now(timezone.utc)
        
        # Update red flags
        self.red_flags_detected.extend(turn.red_flags)
        
        # Mark completion if needed
        if turn.medical_outcome in [MedicalOutcome.EMERGENCY, MedicalOutcome.ROUTINE_DOCTOR, MedicalOutcome.SELF_CARE]:
            self.status = ConversationStatus.COMPLETED
            self.completed_at = datetime.now(timezone.utc)
    
    def get_latest_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent conversation turn"""
        return self.turns[-1] if self.turns else None
    
    def is_complete(self) -> bool:
        """Check if conversation has reached completion"""
        return self.status == ConversationStatus.COMPLETED

# === Message Routing Models ===

class MessageRoute(BaseModel):
    """Model for routing messages between stakeholders"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    from_stakeholder: StakeholderType
    to_stakeholder: StakeholderType
    message_content: str = Field(..., min_length=1)
    conversation_id: str
    priority: MessagePriority = Field(default=MessagePriority.MEDIUM)
    requires_human_review: bool = Field(default=False)
    medical_outcome: Optional[MedicalOutcome] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# === Conversation Analytics Models ===

class ConversationMetrics(BaseModel):
    """Analytics model for conversation performance tracking"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    conversation_id: str
    total_turns: int = Field(..., ge=0)
    completion_time_seconds: Optional[int] = Field(None, ge=0)
    final_outcome: Optional[MedicalOutcome] = None
    confidence_scores: List[int] = Field(default_factory=list)
    red_flags_count: int = Field(default=0, ge=0)
    nice_protocols_used: List[str] = Field(default_factory=list)
    stakeholders_involved: List[StakeholderType] = Field(default_factory=list)
    average_confidence: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    def calculate_metrics(self, conversation: ConversationState) -> None:
        """Calculate metrics from conversation state"""
        self.total_turns = conversation.turn_count
        self.final_outcome = conversation.current_outcome
        self.red_flags_count = len(conversation.red_flags_detected)
        self.nice_protocols_used = conversation.nice_protocols_used
        self.stakeholders_involved = conversation.active_stakeholders
        
        # Calculate completion time
        if conversation.completed_at and conversation.created_at:
            self.completion_time_seconds = int(
                (conversation.completed_at - conversation.created_at).total_seconds()
            )
        
        # Calculate average confidence
        confidence_scores = [turn.confidence_score for turn in conversation.turns]
        if confidence_scores:
            self.confidence_scores = confidence_scores
            self.average_confidence = sum(confidence_scores) / len(confidence_scores)
