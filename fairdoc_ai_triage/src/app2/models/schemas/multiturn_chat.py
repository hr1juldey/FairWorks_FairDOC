"""
V2 Multi-turn Chat API Request/Response DTOs
Pydantic v2 models for FastAPI endpoints
File: src/app2/models/schemas/multiturn_chat.py
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict
from .medical_triage import (
    MedicalOutcome, 
    RedFlagIndicator, 
    TriageDecision, 
    ConversationTurn
)


class ChatProvider(str, Enum):
    """Supported chat providers for V2 system"""
    RAVEN = "raven"
    TELEGRAM = "telegram" 
    WHATSAPP = "whatsapp"
    API_DIRECT = "api_direct"


class StakeholderRole(str, Enum):
    """Who is participating in the conversation"""
    PATIENT = "patient"
    FAMILY_MEMBER = "family_member"
    DOCTOR = "doctor"
    ADMIN = "admin"
    TRIAGE_AGENT = "triage_agent"

class MessagePriority(str, Enum):
    """Message routing priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"

class ConversationStatus(str, Enum):
    """Current state of the conversation"""
    NEW = "new"
    IN_PROGRESS = "in_progress"
    AWAITING_RESPONSE = "awaiting_response"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"


class MultiTurnChatRequest(BaseModel):
    """
    Incoming request for V2 chat endpoint
    Maps to medical triage flow
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False
    )
    
    # Core identifiers
    conversation_id: Optional[UUID] = Field(
        default_factory=uuid4,
        description="Unique conversation identifier for session continuity"
    )
    user_message: str = Field(
        min_length=1,
        max_length=1000,
        description="Patient's current message or symptom description"
    )
    
    # Stakeholder context
    stakeholder_role: StakeholderRole = Field(
        default=StakeholderRole.PATIENT,
        description="Role of the person sending the message"
    )
    stakeholder_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="External ID for the person (phone number, user_id, etc.)"
    )
    
    # Chat provider metadata
    chat_provider: ChatProvider = Field(
        default=ChatProvider.API_DIRECT,
        description="Which chat platform this message came from"
    )
    provider_metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Provider-specific context (phone numbers, chat IDs, etc.)"
    )
    
    # Optional context
    patient_age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Patient age for better triage assessment"
    )
    patient_gender: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Patient gender for clinical context"
    )
    is_emergency_override: bool = Field(
        default=False,
        description="Force emergency escalation bypass normal triage"
    )


class MultiTurnChatResponse(BaseModel):
    """
    Response from V2 chat system
    Contains agent decision + next steps
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True
    )
    
    # Response identifiers
    conversation_id: UUID = Field(
        description="Session identifier matching the request"
    )
    response_id: UUID = Field(
        default_factory=uuid4,
        description="Unique ID for this specific response"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this response was generated"
    )
    
    # Agent response content
    agent_message: Optional[str] = Field(
        default=None,
        description="Triage agent's response or follow-up question"
    )
    next_question: Optional[str] = Field(
        default=None,
        description="Specific follow-up question if conversation continues"
    )
    
    # Triage assessment
    current_assessment: Optional[TriageDecision] = Field(
        default=None,
        description="Current medical assessment from DSPy agent"
    )
    medical_outcome: Optional[MedicalOutcome] = Field(
        default=None,
        description="Final outcome if triage is complete"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=100.0,
        default=0.0,
        description="Confidence in current assessment (0-100)"
    )
    
    # Safety flags
    red_flags_detected: List[RedFlagIndicator] = Field(
        default_factory=list,
        description="Any red flag symptoms identified"
    )
    requires_human_review: bool = Field(
        default=False,
        description="Whether human clinician review is needed"
    )
    is_emergency: bool = Field(
        default=False,
        description="Emergency flag for immediate escalation"
    )
    
    # Conversation state
    conversation_status: ConversationStatus = Field(
        default=ConversationStatus.IN_PROGRESS,
        description="Current state of the conversation"
    )
    turn_count: int = Field(
        ge=1,
        description="Number of conversational turns so far"
    )
    
    # NICE protocol context
    relevant_protocols: List[str] = Field(
        default_factory=list,
        description="NICE protocol IDs that match current symptoms"
    )
    
    # Routing information
    notify_stakeholders: List[StakeholderRole] = Field(
        default_factory=list,
        description="Which stakeholders should be notified of this response"
    )
    
    # Metadata for client
    processing_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="How long the triage agent took to respond"
    )
    model_version: str = Field(
        default="v2.6-stable",
        description="Version of the triage system that generated this response"
    )


class ConversationHistoryRequest(BaseModel):
    """Request to fetch conversation history"""
    model_config = ConfigDict(frozen=True)
    
    conversation_id: UUID = Field(
        description="Conversation to retrieve"
    )
    include_metadata: bool = Field(
        default=False,
        description="Whether to include processing metadata"
    )


class ConversationHistoryResponse(BaseModel):
    """Response containing full conversation history"""
    model_config = ConfigDict(frozen=True)
    
    conversation_id: UUID
    conversation_turns: List[ConversationTurn] = Field(
        description="Complete history of conversation turns"
    )
    final_outcome: Optional[MedicalOutcome] = Field(
        default=None,
        description="Final triage outcome if conversation completed"
    )
    total_turns: int = Field(
        ge=1,
        description="Total number of turns in conversation"
    )
    created_at: datetime
    completed_at: Optional[datetime] = Field(default=None)


class EmergencyAlertPayload(BaseModel):
    """
    Payload sent to external webhooks when emergency detected
    Used for SMS/email notifications
    """
    model_config = ConfigDict(frozen=True)
    
    alert_id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    patient_context: dict = Field(
        description="Safe patient context (no PHI details)"
    )
    red_flags: List[RedFlagIndicator]
    alert_timestamp: datetime = Field(default_factory=datetime.now)
    severity_level: str = Field(
        regex="^(HIGH|CRITICAL)$",
        description="Alert severity for escalation routing"
    )
    recommended_action: str = Field(
        description="Human-readable next steps"
    )
