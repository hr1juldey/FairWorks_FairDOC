"""
Chat and conversation models for Fairdoc Medical AI Backend.
Handles real-time messaging, WebSocket connections, conversation threads, and AI-human interactions.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator
from enum import Enum

from datamodels.base_models import (
    BaseEntity, BaseResponse, TimestampMixin, UUIDMixin,
    ValidationMixin, MetadataMixin, RiskLevel, UrgencyLevel
)

# ============================================================================
# CHAT ENUMS AND TYPES
# ============================================================================

class MessageType(str, Enum):
    """Types of messages in the chat system."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    FILE = "file"
    SYSTEM = "system"
    AI_ASSESSMENT = "ai_assessment"
    DOCTOR_NOTE = "doctor_note"
    PRESCRIPTION = "prescription"
    REFERRAL = "referral"
    EMERGENCY_ALERT = "emergency_alert"

class SenderType(str, Enum):
    """Types of message senders."""
    PATIENT = "patient"
    AI_ASSISTANT = "ai_assistant"
    DOCTOR = "doctor"
    NURSE = "nurse"
    SYSTEM = "system"
    ADMIN = "admin"

class ConversationStatus(str, Enum):
    """Status of conversation threads."""
    ACTIVE = "active"
    WAITING_PATIENT = "waiting_patient"
    WAITING_DOCTOR = "waiting_doctor"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    EMERGENCY = "emergency"

class ChatSessionType(str, Enum):
    """Types of chat sessions."""
    INITIAL_TRIAGE = "initial_triage"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    SECOND_OPINION = "second_opinion"
    ROUTINE_CHECK = "routine_check"
    SPECIALIST_CONSULTATION = "specialist_consultation"

class WebSocketConnectionStatus(str, Enum):
    """WebSocket connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    IDLE = "idle"

class AIInteractionType(str, Enum):
    """Types of AI interactions."""
    SYMPTOM_ASSESSMENT = "symptom_assessment"
    RISK_EVALUATION = "risk_evaluation"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    TREATMENT_SUGGESTION = "treatment_suggestion"
    FOLLOW_UP_QUESTION = "follow_up_question"
    CLARIFICATION = "clarification"
    HANDOFF_SUMMARY = "handoff_summary"

# ============================================================================
# MESSAGE MODELS
# ============================================================================

class MessageContent(BaseModel):
    """Content structure for different message types."""
    text: Optional[str] = Field(None, max_length=5000, description="Text content")
    file_url: Optional[str] = Field(None, description="File URL for attachments")
    file_type: Optional[str] = Field(None, description="MIME type of file")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('text')
    def validate_text_content(cls, v, values):
        """Ensure text content exists for text messages."""
        return v.strip() if v else v

class AIAssessmentContent(BaseModel):
    """Specialized content for AI assessment messages."""
    assessment_type: AIInteractionType
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: Optional[RiskLevel] = None
    urgency: Optional[UrgencyLevel] = None
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    next_questions: List[str] = Field(default_factory=list)
    differential_diagnoses: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('red_flags')
    def validate_red_flags(cls, v):
        """Ensure red flags trigger appropriate urgency."""
        if v and len(v) > 0:
            # Red flags should increase urgency
            return v
        return v

class DoctorNoteContent(BaseModel):
    """Content structure for doctor notes and assessments."""
    clinical_assessment: str = Field(..., min_length=10, max_length=5000)
    diagnosis: Optional[str] = Field(None, max_length=1000)
    treatment_plan: Optional[str] = Field(None, max_length=2000)
    follow_up_instructions: Optional[str] = Field(None, max_length=1000)
    prescription_details: Optional[str] = Field(None, max_length=1000)
    referral_needed: bool = Field(default=False)
    referral_specialty: Optional[str] = Field(None, max_length=100)
    urgency_override: Optional[UrgencyLevel] = None
    
    @validator('clinical_assessment')
    def validate_assessment_completeness(cls, v):
        """Ensure clinical assessment is comprehensive."""
        if len(v.strip()) < 10:
            raise ValueError('Clinical assessment must be detailed')
        return v.strip()

class ChatMessage(BaseEntity, ValidationMixin, MetadataMixin):
    """Core chat message model."""
    
    # Message identification
    conversation_id: UUID = Field(..., description="Reference to conversation thread")
    sender_id: UUID = Field(..., description="ID of message sender")
    sender_type: SenderType
    recipient_id: Optional[UUID] = Field(None, description="Specific recipient (optional)")
    
    # Message content
    message_type: MessageType
    content: Union[MessageContent, AIAssessmentContent, DoctorNoteContent] = Field(..., description="Message content")
    
    # Message context
    reply_to_message_id: Optional[UUID] = Field(None, description="Reply thread reference")
    thread_depth: int = Field(default=0, ge=0, description="Thread nesting level")
    
    # Delivery and status
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    is_edited: bool = Field(default=False)
    edited_at: Optional[datetime] = None
    is_deleted: bool = Field(default=False)
    deleted_at: Optional[datetime] = None
    
    # Medical context
    clinical_context: Dict[str, Any] = Field(default_factory=dict)
    tagged_conditions: List[str] = Field(default_factory=list)
    mentioned_symptoms: List[str] = Field(default_factory=list)
    
    # AI processing
    ai_processed: bool = Field(default=False)
    ai_processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    bias_checked: bool = Field(default=False)
    bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    def mark_delivered(self):
        """Mark message as delivered."""
        self.delivered_at = datetime.utcnow()
    
    def mark_read(self):
        """Mark message as read."""
        self.read_at = datetime.utcnow()
    
    def edit_message(self, new_content: Union[MessageContent, AIAssessmentContent, DoctorNoteContent]):
        """Edit message content with audit trail."""
        self.content = new_content
        self.is_edited = True
        self.edited_at = datetime.utcnow()
    
    def soft_delete(self):
        """Soft delete message for compliance."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()

# ============================================================================
# CONVERSATION MODELS
# ============================================================================

class ConversationParticipant(BaseModel):
    """Participant in a conversation."""
    user_id: UUID
    role: SenderType
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    left_at: Optional[datetime] = None
    is_active: bool = Field(default=True)
    last_seen: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)
    
    def leave_conversation(self):
        """Mark participant as left."""
        self.is_active = False
        self.left_at = datetime.utcnow()

class ConversationSummary(BaseModel):
    """Summary of conversation for handoffs and records."""
    chief_complaint: str = Field(..., max_length=500)
    key_symptoms: List[str] = Field(default_factory=list)
    assessment_summary: Optional[str] = Field(None, max_length=2000)
    risk_assessment: Optional[RiskLevel] = None
    recommendations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    
    # AI metrics
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    bias_assessment: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time: Optional[float] = None
    
    # Clinical metrics
    clinical_reviewed: bool = Field(default=False)
    clinical_reviewer: Optional[str] = None
    clinical_review_time: Optional[datetime] = None

class Conversation(BaseEntity, ValidationMixin, MetadataMixin):
    """Main conversation thread model."""
    
    # Basic identification
    patient_id: UUID = Field(..., description="Primary patient in conversation")
    session_type: ChatSessionType
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    
    # Participants
    participants: List[ConversationParticipant] = Field(default_factory=list)
    current_handler: Optional[UUID] = Field(None, description="Current responsible clinician")
    
    # Conversation metadata
    title: str = Field(..., max_length=200, description="Conversation title/subject")
    priority: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    estimated_duration: Optional[int] = Field(None, description="Expected duration in minutes")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Message tracking
    total_messages: int = Field(default=0)
    unread_count: int = Field(default=0)
    last_message_id: Optional[UUID] = None
    
    # Clinical context
    medical_record_id: Optional[UUID] = Field(None, description="Associated medical record")
    consultation_id: Optional[UUID] = Field(None, description="Associated consultation")
    summary: Optional[ConversationSummary] = None
    
    # AI interaction tracking
    ai_interactions: int = Field(default=0)
    human_takeover: bool = Field(default=False)
    human_takeover_reason: Optional[str] = None
    human_takeover_at: Optional[datetime] = None
    
    def add_participant(self, user_id: UUID, role: SenderType) -> ConversationParticipant:
        """Add participant to conversation."""
        participant = ConversationParticipant(user_id=user_id, role=role)
        self.participants.append(participant)
        return participant
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()
    
    def escalate_to_human(self, reason: str, handler_id: UUID):
        """Escalate conversation to human handler."""
        self.human_takeover = True
        self.human_takeover_reason = reason
        self.human_takeover_at = datetime.utcnow()
        self.current_handler = handler_id
        self.status = ConversationStatus.ESCALATED
    
    def complete_conversation(self, summary: ConversationSummary):
        """Complete conversation with summary."""
        self.status = ConversationStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.summary = summary

# ============================================================================
# WEBSOCKET CONNECTION MODELS
# ============================================================================

class WebSocketConnection(BaseEntity):
    """WebSocket connection tracking."""
    
    # Connection details
    user_id: UUID
    connection_id: str = Field(..., description="Unique connection identifier")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    
    # Connection status
    status: WebSocketConnectionStatus = Field(default=WebSocketConnectionStatus.CONNECTED)
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    last_ping: Optional[datetime] = None
    last_pong: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    
    # Connection metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Activity tracking
    messages_sent: int = Field(default=0)
    messages_received: int = Field(default=0)
    bytes_sent: int = Field(default=0)
    bytes_received: int = Field(default=0)
    
    # Quality metrics
    latency_ms: Optional[float] = None
    packet_loss: float = Field(default=0.0, ge=0.0, le=1.0)
    connection_quality: Optional[str] = Field(None, description="poor/fair/good/excellent")
    
    def update_ping(self):
        """Update ping timestamp."""
        self.last_ping = datetime.utcnow()
    
    def update_pong(self):
        """Update pong timestamp and calculate latency."""
        now = datetime.utcnow()
        self.last_pong = now
        if self.last_ping:
            self.latency_ms = (now - self.last_ping).total_seconds() * 1000
    
    def disconnect(self, reason: Optional[str] = None):
        """Mark connection as disconnected."""
        self.status = WebSocketConnectionStatus.DISCONNECTED
        self.disconnected_at = datetime.utcnow()
        if reason:
            self.metadata["disconnect_reason"] = reason

# ============================================================================
# AI CONVERSATION MODELS
# ============================================================================

class AIConversationState(BaseModel):
    """State management for AI conversation flow."""
    
    # Current conversation context
    conversation_id: UUID
    current_step: str = Field(default="greeting")
    completed_steps: List[str] = Field(default_factory=list)
    
    # Patient information gathering
    symptoms_collected: Dict[str, Any] = Field(default_factory=dict)
    vital_signs_collected: Dict[str, Any] = Field(default_factory=dict)
    medical_history_collected: Dict[str, Any] = Field(default_factory=dict)
    
    # Assessment progress
    assessment_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    information_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_indicators: List[str] = Field(default_factory=list)
    
    # Decision points
    needs_human_review: bool = Field(default=False)
    escalation_triggers: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    
    # Timing
    conversation_duration: timedelta = Field(default=timedelta(0))
    last_update: datetime = Field(default_factory=datetime.utcnow)

class AIInteractionLog(BaseEntity):
    """Log of AI interactions for learning and improvement."""
    
    conversation_id: UUID
    interaction_type: AIInteractionType
    
    # Input context
    user_input: str = Field(..., max_length=2000)
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    
    # AI processing
    model_used: str = Field(..., description="AI model identifier")
    processing_time_ms: float = Field(..., ge=0)
    tokens_used: int = Field(default=0)
    
    # AI output
    ai_response: str = Field(..., max_length=5000)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    risk_assessment: Optional[RiskLevel] = None
    
    # Quality metrics
    user_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="1-5 rating")
    clinical_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Learning feedback
    human_override: bool = Field(default=False)
    human_feedback: Optional[str] = Field(None, max_length=1000)
    outcome_accuracy: Optional[bool] = None
    
    # Follow-up
    led_to_escalation: bool = Field(default=False)
    escalation_reason: Optional[str] = None

# ============================================================================
# CHAT SESSION MODELS
# ============================================================================

class ChatSession(BaseEntity, ValidationMixin, MetadataMixin):
    """Overall chat session encompassing multiple conversations."""
    
    # Session identification
    patient_id: UUID
    session_type: ChatSessionType
    
    # Session state
    is_active: bool = Field(default=True)
    session_token: Optional[str] = Field(None, description="Secure session token")
    
    # Conversations in this session
    conversations: List[UUID] = Field(default_factory=list, description="Conversation IDs")
    active_conversation_id: Optional[UUID] = None
    
    # Session timing
    session_started_at: datetime = Field(default_factory=datetime.utcnow)
    session_ended_at: Optional[datetime] = None
    total_duration: Optional[timedelta] = None
    
    # Participants over session lifetime
    all_participants: List[UUID] = Field(default_factory=list)
    current_clinician: Optional[UUID] = None
    
    # Session outcomes
    final_assessment: Optional[ConversationSummary] = None
    patient_satisfaction: Optional[int] = Field(None, ge=1, le=5)
    session_notes: Optional[str] = Field(None, max_length=2000)
    
    def add_conversation(self, conversation_id: UUID):
        """Add conversation to session."""
        self.conversations.append(conversation_id)
        self.active_conversation_id = conversation_id
    
    def end_session(self, final_assessment: ConversationSummary):
        """End the chat session."""
        self.is_active = False
        self.session_ended_at = datetime.utcnow()
        self.total_duration = self.session_ended_at - self.session_started_at
        self.final_assessment = final_assessment

# ============================================================================
# NOTIFICATION MODELS
# ============================================================================

class ChatNotification(BaseEntity):
    """Real-time notifications for chat events."""
    
    # Notification target
    recipient_id: UUID
    conversation_id: Optional[UUID] = None
    
    # Notification content
    notification_type: str = Field(..., description="message, escalation, emergency, etc.")
    title: str = Field(..., max_length=100)
    message: str = Field(..., max_length=500)
    
    # Notification status
    is_read: bool = Field(default=False)
    is_delivered: bool = Field(default=False)
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    # Priority and urgency
    priority: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    expires_at: Optional[datetime] = None
    
    # Action context
    action_required: bool = Field(default=False)
    action_url: Optional[str] = None
    action_label: Optional[str] = None

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class ConversationListResponse(BaseResponse):
    """Response for conversation list queries."""
    conversations: List[Conversation]
    total_count: int
    active_count: int
    unread_count: int

class MessageListResponse(BaseResponse):
    """Response for message list queries."""
    messages: List[ChatMessage]
    conversation_id: UUID
    total_count: int
    has_more: bool

class WebSocketEventResponse(BaseModel):
    """WebSocket event response format."""
    event_type: str
    conversation_id: Optional[UUID] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ============================================================================
# CONVERSATION ANALYTICS
# ============================================================================

class ConversationAnalytics(BaseModel):
    """Analytics data for conversation performance."""
    
    # Time metrics
    average_response_time: float = Field(..., description="Average AI response time in seconds")
    patient_wait_time: float = Field(..., description="Average patient wait time")
    conversation_duration: float = Field(..., description="Total conversation time")
    
    # Interaction metrics
    total_messages: int
    ai_messages: int
    human_messages: int
    escalation_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Quality metrics
    patient_satisfaction: Optional[float] = Field(None, ge=1.0, le=5.0)
    clinical_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    bias_score: float = Field(..., ge=0.0, le=1.0)
    
    # Outcome metrics
    successful_resolutions: int
    required_follow_up: int
    emergency_escalations: int



# Default conversation flow templates
DEFAULT_CONVERSATION_FLOWS = {
    ChatSessionType.INITIAL_TRIAGE: [
        "greeting",
        "chief_complaint",
        "symptom_assessment",
        "vital_signs",
        "medical_history",
        "risk_assessment",
        "recommendations",
        "handoff_or_completion"
    ],
    ChatSessionType.EMERGENCY: [
        "emergency_assessment",
        "immediate_action_check",
        "emergency_services",
        "continuous_monitoring"
    ],
    ChatSessionType.FOLLOW_UP: [
        "greeting",
        "status_check",
        "symptom_update",
        "treatment_compliance",
        "outcome_assessment"
    ]
}
