"""
Enhanced multi-modal chat and conversation models for Fairdoc Medical AI Backend.
Supports text, audio, images, emojis, medical files, and real-time communication.
Fixed for Pydantic V2 with proper imports and enum usage.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Union, Literal
from uuid import UUID
from pydantic import Field, field_validator
from enum import Enum

# Fixed imports from basemodels with all required components
from datamodels.base_models import (
    BaseEntity, BaseResponse, TimestampMixin, UUIDMixin,
    ValidationMixin, MetadataMixin, RiskLevel, UrgencyLevel,
    Gender, Ethnicity  # Added Gender and Ethnicity
)

# ============================================================================
# MULTI-MODAL ENUMS AND TYPES
# ============================================================================

class MessageType(str, Enum):
    """Enhanced message types for multi-modal support."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"
    FILE = "file"
    EMOJI = "emoji"
    VOICE_NOTE = "voice_note"
    PAIN_SCALE = "pain_scale"
    MEDICAL_IMAGE = "medical_image"
    DICOM_IMAGE = "dicom_image"
    DOCUMENT = "document"
    LOCATION = "location"
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

class SpeechToTextStatus(str, Enum):
    """Speech-to-text processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TRANSCRIBING = "transcribing"

class EmojiCategory(str, Enum):
    """Emoji categories for medical context."""
    PAIN_EXPRESSION = "pain_expression"
    MOOD = "mood"
    BODY_PART = "body_part"
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    GENERAL = "general"

# ============================================================================
# MULTI-MODAL CONTENT MODELS WITH PROPER VALIDATION
# ============================================================================

class AudioContent(TimestampMixin, ValidationMixin):
    """Audio message content with speech-to-text support."""
    audio_url: str = Field(..., description="URL to audio file")
    audio_format: str = Field(..., description="mp3, wav, m4a, ogg, webm")
    duration_seconds: float = Field(..., ge=0, le=300, description="Max 5 minutes")
    file_size_bytes: int = Field(..., ge=0)
    
    # Speech-to-text results
    transcription_status: SpeechToTextStatus = Field(default=SpeechToTextStatus.PENDING)
    transcribed_text: Optional[str] = Field(None, description="Auto-transcribed text")
    transcription_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    language_detected: Optional[str] = Field(None, description="Detected language code")
    
    # Audio analysis
    volume_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average volume")
    noise_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Background noise")
    emotion_detected: Optional[str] = Field(None, description="Detected emotion in voice")
    
    # Medical context using enums
    urgency_detected: Optional[UrgencyLevel] = None
    keywords_extracted: List[str] = Field(default_factory=list)
    contains_medical_terms: bool = Field(default=False)
    
    @field_validator('audio_url')
    @classmethod
    def validate_audio_url(cls, v: str) -> str:
        """Validate audio URL format."""
        if not v.startswith(('http://', 'https://', 'data:', 'blob:')):
            raise ValueError('Invalid audio URL format')
        return v

class ImageContent(TimestampMixin, ValidationMixin):
    """Image message content with medical image analysis."""
    image_url: str = Field(..., description="URL to image file")
    image_format: str = Field(..., description="jpeg, png, webp, tiff, bmp, svg")
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    file_size_bytes: int = Field(..., ge=0)
    
    # Medical image classification
    medical_image_type: Optional[str] = Field(None, description="xray, ct_scan, mri, ultrasound, etc.")
    is_medical_image: bool = Field(default=False)
    contains_phi: bool = Field(default=False, description="Contains Personal Health Information")
    
    # Image analysis results
    ai_analysis_status: str = Field(default="pending")
    ai_analysis_results: Dict[str, Any] = Field(default_factory=dict)
    detected_objects: List[str] = Field(default_factory=list)
    medical_findings: List[str] = Field(default_factory=list)
    
    # DICOM metadata (for medical images)
    dicom_metadata: Optional[Dict[str, Any]] = Field(None)
    patient_position: Optional[str] = None
    study_date: Optional[datetime] = None
    modality: Optional[str] = None
    
    # Image quality metrics
    sharpness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    brightness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    contrast_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Annotation data with timestamps
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    regions_of_interest: List[Dict[str, Any]] = Field(default_factory=list)
    
    @field_validator('image_url')
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        """Validate image URL format."""
        if not v.startswith(('http://', 'https://', 'data:', 'blob:')):
            raise ValueError('Invalid image URL format')
        return v

class EmojiContent(TimestampMixin, ValidationMixin):
    """Emoji content with medical context mapping."""
    emoji_unicode: str = Field(..., description="Unicode emoji character")
    emoji_name: str = Field(..., description="Emoji name/description")
    category: EmojiCategory
    
    # Medical context mapping
    pain_scale_value: Optional[int] = Field(None, ge=1, le=10, description="Pain scale 1-10")
    mood_indicator: Optional[str] = Field(None, description="Happy, sad, worried, etc.")
    symptom_reference: Optional[str] = Field(None, description="Referenced symptom")
    body_part_reference: Optional[str] = Field(None, description="Referenced body part")
    
    # Cultural context
    cultural_meaning: Optional[str] = Field(None, description="Cultural interpretation")
    alternate_meanings: List[str] = Field(default_factory=list)
    
    @field_validator('emoji_unicode')
    @classmethod
    def validate_emoji_unicode(cls, v: str) -> str:
        """Validate emoji unicode format."""
        if not v or len(v) < 1:
            raise ValueError('Invalid emoji unicode')
        return v

class PainScaleContent(TimestampMixin, ValidationMixin):
    """Interactive pain scale content with RiskLevel mapping."""
    scale_type: Literal["numeric", "faces", "colors"] = "numeric"
    scale_value: int = Field(..., ge=0, le=10)
    scale_description: str
    
    # Visual representation
    emoji_representation: Optional[str] = None
    color_code: Optional[str] = None
    face_image_url: Optional[str] = None
    
    # Context
    body_part: Optional[str] = None
    pain_quality: Optional[str] = Field(None, description="Sharp, dull, burning, etc.")
    pain_timing: Optional[str] = Field(None, description="Constant, intermittent, etc.")
    triggers: List[str] = Field(default_factory=list)
    relieving_factors: List[str] = Field(default_factory=list)
    
    # Risk assessment using RiskLevel enum
    pain_risk_level: Optional[RiskLevel] = Field(None, description="Pain-based risk assessment")
    
    @field_validator('pain_risk_level')
    @classmethod
    def calculate_pain_risk_level(cls, v: Optional[RiskLevel], info) -> Optional[RiskLevel]:
        """Auto-calculate risk level based on pain scale value."""
        if hasattr(info, 'data') and 'scale_value' in info.data:
            pain_value = info.data['scale_value']
            if pain_value >= 8:
                return RiskLevel.HIGH
            elif pain_value >= 6:
                return RiskLevel.MODERATE
            elif pain_value >= 3:
                return RiskLevel.LOW
            else:
                return RiskLevel.LOW
        return v

class DocumentContent(TimestampMixin, ValidationMixin):
    """Document/file content."""
    file_url: str = Field(..., description="URL to document file")
    file_name: str = Field(..., max_length=255)
    file_type: str = Field(..., description="MIME type")
    file_size_bytes: int = Field(..., ge=0)
    
    # Document analysis
    document_type: Optional[str] = Field(None, description="Lab report, prescription, etc.")
    extracted_text: Optional[str] = Field(None, description="OCR extracted text")
    medical_data_extracted: Dict[str, Any] = Field(default_factory=dict)
    
    # Security
    scanned_for_malware: bool = Field(default=False)
    contains_sensitive_data: bool = Field(default=False)
    access_permissions: List[str] = Field(default_factory=list)

class LocationContent(TimestampMixin, ValidationMixin):
    """Location/GPS content for emergency services."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy_meters: Optional[float] = Field(None, ge=0)
    
    # Address information
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    
    # Emergency context using UrgencyLevel
    is_emergency_location: bool = Field(default=False)
    emergency_urgency: Optional[UrgencyLevel] = None
    nearest_hospital: Optional[str] = None
    estimated_ems_time: Optional[int] = Field(None, description="Minutes for EMS arrival")

# ============================================================================
# ENHANCED MESSAGE MODEL WITH DEMOGRAPHICS
# ============================================================================

class MultiModalMessage(BaseEntity, ValidationMixin, MetadataMixin):
    """Enhanced message model with proper timestamp, UUID, and demographic tracking."""
    
    # Message identification (UUID handled by BaseEntity)
    conversation_id: UUID = Field(..., description="Reference to conversation thread")
    sender_id: UUID = Field(..., description="ID of message sender")
    sender_type: SenderType = Field(..., description="Type of sender")
    recipient_id: Optional[UUID] = Field(None, description="Specific recipient")
    
    # Sender demographics (using enums from basemodels)
    sender_gender: Optional[Gender] = Field(None, description="Sender gender for bias tracking")
    sender_ethnicity: Optional[Ethnicity] = Field(None, description="Sender ethnicity for bias tracking")
    sender_age_group: Optional[str] = Field(None, description="Age group: young, middle, senior")
    
    # Message content (Union type for multi-modal support)
    message_type: MessageType
    content: Union[
        str,  # Simple text
        AudioContent,
        ImageContent,
        EmojiContent,
        PainScaleContent,
        DocumentContent,
        LocationContent,
        Dict[str, Any]  # For complex/custom content
    ] = Field(..., description="Multi-modal message content")
    
    # Rich text formatting
    formatted_text: Optional[str] = Field(None, description="HTML/Markdown formatted text")
    mentions: List[UUID] = Field(default_factory=list, description="@mentioned users")
    hashtags: List[str] = Field(default_factory=list, description="#tags for categorization")
    
    # Message context
    reply_to_message_id: Optional[UUID] = Field(None, description="Reply thread reference")
    thread_depth: int = Field(default=0, ge=0, description="Thread nesting level")
    is_forwarded: bool = Field(default=False)
    original_message_id: Optional[UUID] = None
    
    # Delivery and status (using TimestampMixin from BaseEntity)
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    is_edited: bool = Field(default=False)
    edited_at: Optional[datetime] = None
    
    # Medical context using RiskLevel
    clinical_context: Dict[str, Any] = Field(default_factory=dict)
    tagged_conditions: List[str] = Field(default_factory=list)
    mentioned_symptoms: List[str] = Field(default_factory=list)
    urgency_indicators: List[str] = Field(default_factory=list)
    message_risk_level: Optional[RiskLevel] = Field(None, description="Risk level of message content")
    
    # AI processing
    ai_processed: bool = Field(default=False)
    ai_processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    bias_checked: bool = Field(default=False)
    bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Multi-modal processing status
    processing_status: Dict[str, str] = Field(default_factory=dict)
    ai_analysis_complete: bool = Field(default=False)
    transcription_complete: bool = Field(default=False)
    image_analysis_complete: bool = Field(default=False)
    
    # Accessibility
    alt_text: Optional[str] = Field(None, description="Alternative text for images")
    audio_description: Optional[str] = Field(None, description="Audio description for visual content")
    translation_available: bool = Field(default=False)
    translations: Dict[str, str] = Field(default_factory=dict)  # language_code: translated_text
    
    def mark_delivered(self):
        """Mark message as delivered with timestamp."""
        self.delivered_at = lambda: datetime.now(timezone.utc)()
        self.update_timestamp()  # From TimestampMixin
    
    def mark_read(self):
        """Mark message as read with timestamp."""
        self.read_at = lambda: datetime.now(timezone.utc)()
        self.update_timestamp()
    
    def assess_message_risk(self) -> RiskLevel:
        """Assess risk level based on message content."""
        if self.message_type == MessageType.PAIN_SCALE and isinstance(self.content, PainScaleContent):
            return self.content.pain_risk_level or RiskLevel.LOW
        
        # Check for emergency keywords
        emergency_keywords = ['can\'t breathe', 'chest pain', 'severe pain', 'emergency', 'help']
        if self.message_type == MessageType.TEXT and isinstance(self.content, str):
            content_lower = self.content.lower()
            if any(keyword in content_lower for keyword in emergency_keywords):
                return RiskLevel.HIGH
        
        return RiskLevel.LOW
    
    def update_risk_assessment(self):
        """Update message risk level and save timestamp."""
        self.message_risk_level = self.assess_message_risk()
        self.update_timestamp()

# ============================================================================
# CONVERSATION MODELS WITH DEMOGRAPHICS
# ============================================================================

class ConversationParticipant(TimestampMixin, UUIDMixin):
    """Participant in a conversation with demographic tracking."""
    user_id: UUID
    role: SenderType
    
    # Demographics for bias monitoring (using enums from basemodels)
    gender: Optional[Gender] = None
    ethnicity: Optional[Ethnicity] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    
    is_active: bool = Field(default=True)
    last_seen: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)
    left_at: Optional[datetime] = None
    
    def leave_conversation(self) -> None:
        """Mark participant as left with timestamp."""
        self.is_active = False
        self.left_at = lambda: datetime.now(timezone.utc)()
        self.update_timestamp()

class ConversationSummary(TimestampMixin, ValidationMixin):
    """Summary of conversation with comprehensive risk and bias assessment."""
    chief_complaint: str = Field(..., max_length=500)
    key_symptoms: List[str] = Field(default_factory=list)
    assessment_summary: Optional[str] = Field(None, max_length=2000)
    
    # Using enums from basemodels for risk assessment
    risk_assessment: Optional[RiskLevel] = None
    urgency_level: Optional[UrgencyLevel] = None
    
    recommendations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    
    # AI metrics
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    bias_assessment: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time: Optional[float] = None
    
    # Demographic bias analysis
    demographic_bias_detected: bool = Field(default=False)
    bias_factors: List[str] = Field(default_factory=list)
    
    # Clinical review tracking
    clinical_reviewed: bool = Field(default=False)
    clinical_reviewer: Optional[str] = None
    clinical_review_time: Optional[datetime] = None

class MultiModalConversation(BaseEntity, ValidationMixin, MetadataMixin):
    """Enhanced conversation with multi-modal support and demographic tracking."""
    
    # Basic identification (UUID and timestamps handled by BaseEntity)
    patient_id: UUID = Field(..., description="Primary patient in conversation")
    session_type: ChatSessionType
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    
    # Patient demographics for bias monitoring
    patient_gender: Optional[Gender] = None
    patient_ethnicity: Optional[Ethnicity] = None
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    
    # Participants with proper timestamp tracking
    participants: List[ConversationParticipant] = Field(default_factory=list)
    current_handler: Optional[UUID] = Field(None, description="Current responsible clinician")
    
    # Conversation metadata
    title: str = Field(..., max_length=200, description="Conversation title/subject")
    
    # Using enums from basemodels
    priority: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    overall_risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    
    estimated_duration: Optional[timedelta] = Field(None, description="Expected duration")
    
    # Timing (created_at and updated_at handled by BaseEntity)
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Message tracking
    total_messages: int = Field(default=0)
    unread_count: int = Field(default=0)
    last_message_id: Optional[UUID] = None
    
    # Multi-modal capabilities
    supported_modes: List[MessageType] = Field(
        default_factory=lambda: [
            MessageType.TEXT, MessageType.AUDIO, MessageType.IMAGE,
            MessageType.EMOJI, MessageType.PAIN_SCALE, MessageType.DOCUMENT
        ]
    )
    
    # Content analysis summary
    media_summary: Dict[str, int] = Field(
        default_factory=lambda: {
            "total_messages": 0,
            "text_messages": 0,
            "audio_messages": 0,
            "image_messages": 0,
            "emoji_count": 0,
            "pain_scale_entries": 0
        }
    )
    
    # Medical content tracking with risk levels
    symptoms_mentioned: List[str] = Field(default_factory=list)
    pain_scores_recorded: List[Dict[str, Any]] = Field(default_factory=list)
    risk_escalations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Bias monitoring
    bias_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    demographic_bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # AI interaction tracking
    ai_interactions: int = Field(default=0)
    human_takeover: bool = Field(default=False)
    human_takeover_reason: Optional[str] = None
    human_takeover_at: Optional[datetime] = None
    
    def add_participant(self, user_id: UUID, role: SenderType,
                       gender: Optional[Gender] = None,
                       ethnicity: Optional[Ethnicity] = None,
                       age: Optional[int] = None) -> ConversationParticipant:
        """Add participant with demographic tracking."""
        participant = ConversationParticipant(
            user_id=user_id,
            role=role,
            gender=gender,
            ethnicity=ethnicity,
            age=age
        )
        self.participants.append(participant)
        self.update_timestamp()
        return participant
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = lambda: datetime.now(timezone.utc)()
        self.update_timestamp()
    
    def escalate_conversation(self, reason: str, risk_level: RiskLevel) -> None:
        """Escalate conversation with risk level tracking."""
        self.status = ConversationStatus.ESCALATED
        self.overall_risk_level = risk_level
        self.human_takeover = True
        self.human_takeover_reason = reason
        self.human_takeover_at = lambda: datetime.now(timezone.utc)()
        
        # Track escalation
        self.risk_escalations.append({
            "timestamp": lambda: datetime.now(timezone.utc)().isoformat(),
            "reason": reason,
            "risk_level": risk_level.value,
            "escalated_from": self.priority.value
        })
        self.update_timestamp()
    
    def add_bias_alert(self, bias_type: str, severity: float, details: Dict[str, Any]) -> None:
        """Add bias detection alert."""
        alert = {
            "timestamp": lambda: datetime.now(timezone.utc)().isoformat(),
            "bias_type": bias_type,
            "severity": severity,
            "details": details
        }
        self.bias_alerts.append(alert)
        self.update_timestamp()
    
    def add_pain_score(self, score: int, risk_level: RiskLevel):
        """Add pain score with risk level tracking."""
        pain_entry = {
            "score": score,
            "risk_level": risk_level.value,
            "timestamp": lambda: datetime.now(timezone.utc)().isoformat()
        }
        self.pain_scores_recorded.append(pain_entry)
        
        # Update overall risk if pain indicates higher risk
        if risk_level.value > self.overall_risk_level.value:
            self.overall_risk_level = risk_level
        
        self.update_timestamp()

# ============================================================================
# WEBSOCKET CONNECTION WITH PROPER INHERITANCE
# ============================================================================

class WebSocketConnection(BaseEntity):
    """WebSocket connection tracking with UUID and timestamp support."""
    
    # Connection details (UUID handled by BaseEntity)
    user_id: UUID
    connection_id: str = Field(..., description="Unique connection identifier")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    
    # User demographics for connection tracking
    user_gender: Optional[Gender] = None
    user_ethnicity: Optional[Ethnicity] = None
    
    # Connection status
    status: WebSocketConnectionStatus = Field(default=WebSocketConnectionStatus.CONNECTED)
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
        self.last_ping = lambda: datetime.now(timezone.utc)()
        self.update_timestamp()
    
    def update_pong(self):
        """Update pong timestamp and calculate latency."""
        def now():
            return datetime.now(timezone.utc)()
        self.last_pong = now
        if self.last_ping:
            self.latency_ms = (now - self.last_ping).total_seconds() * 1000
        self.update_timestamp()
    
    def disconnect(self, reason: Optional[str] = None):
        """Mark connection as disconnected with timestamp."""
        self.status = WebSocketConnectionStatus.DISCONNECTED
        self.disconnected_at = lambda: datetime.now(timezone.utc)()
        if reason:
            self.add_metadata("disconnect_reason", reason)
        self.update_timestamp()

# ============================================================================
# RESPONSE MODELS WITH BIAS TRACKING
# ============================================================================

class ConversationListResponse(BaseResponse):
    """Response for conversation list queries with bias metrics."""
    conversations: List[MultiModalConversation]
    total_count: int
    active_count: int
    unread_count: int
    risk_distribution: Dict[str, int] = Field(
        default_factory=lambda: {
            RiskLevel.LOW.value: 0,
            RiskLevel.MODERATE.value: 0,
            RiskLevel.HIGH.value: 0,
            RiskLevel.CRITICAL.value: 0
        }
    )
    # Bias metrics by demographics
    bias_metrics_by_gender: Dict[str, float] = Field(default_factory=dict)
    bias_metrics_by_ethnicity: Dict[str, float] = Field(default_factory=dict)

class MessageListResponse(BaseResponse):
    """Response for message list queries with bias tracking."""
    messages: List[MultiModalMessage]
    conversation_id: UUID
    total_count: int
    has_more: bool
    highest_risk_level: Optional[RiskLevel] = None
    average_bias_score: Optional[float] = Field(None, ge=0.0, le=1.0)

# ============================================================================
# EMOJI MAPPING WITH DEMOGRAPHICS CONSIDERATION
# ============================================================================


MEDICAL_EMOJI_MAPPING = {
    # Pain scale emojis with RiskLevel mapping
    "😭": {"pain_scale": 10, "risk_level": RiskLevel.HIGH, "description": "Severe pain"},
    "😰": {"pain_scale": 8, "risk_level": RiskLevel.HIGH, "description": "Intense pain"},
    "😣": {"pain_scale": 6, "risk_level": RiskLevel.MODERATE, "description": "Moderate pain"},
    "🙂": {"pain_scale": 3, "risk_level": RiskLevel.LOW, "description": "Mild pain"},
    "😊": {"pain_scale": 1, "risk_level": RiskLevel.LOW, "description": "Minimal pain"},
    
    # Emergency emojis
    "🚨": {"urgency": UrgencyLevel.EMERGENT, "risk_level": RiskLevel.CRITICAL, "context": "emergency"},
    "🆘": {"urgency": UrgencyLevel.EMERGENT, "risk_level": RiskLevel.CRITICAL, "context": "help_needed"},
    "⚠️": {"urgency": UrgencyLevel.URGENT, "risk_level": RiskLevel.HIGH, "context": "warning"},
    
    # Body parts with risk context
    "💓": {"body_part": "heart", "context": "cardiac", "risk_level": RiskLevel.MODERATE},
    "🫁": {"body_part": "lungs", "context": "respiratory", "risk_level": RiskLevel.MODERATE},
    "🧠": {"body_part": "brain", "context": "neurological", "risk_level": RiskLevel.HIGH},
    
    # Symptoms with risk levels
    "🤒": {"symptom": "fever", "risk_level": RiskLevel.MODERATE},
    "🤢": {"symptom": "nausea", "risk_level": RiskLevel.LOW},
    "😵": {"symptom": "dizziness", "risk_level": RiskLevel.MODERATE},
}


# Default conversation flow templates with bias-aware design
BIAS_AWARE_CONVERSATION_FLOWS = {
    ChatSessionType.INITIAL_TRIAGE: [
        {
            "type": "text",
            "message": "Hello! I'm here to help assess your symptoms. How are you feeling today?",
            "bias_check": "greeting_neutrality"
        },
        {
            "type": "pain_scale",
            "message": "Can you rate your pain on a scale of 1-10?",
            "bias_check": "pain_scale_cultural_sensitivity"
        },
        {
            "type": "audio",
            "message": "Feel free to describe your symptoms in your own words",
            "bias_check": "language_accessibility"
        },
        {
            "type": "demographic_optional",
            "message": "To provide better care, may I ask about your background? (This is optional)",
            "bias_check": "voluntary_demographic_collection"
        }
    ]
}
