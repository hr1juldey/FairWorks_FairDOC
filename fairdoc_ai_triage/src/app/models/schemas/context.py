"""
Fairdoc AI Context Management Schemas
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

def utcnow():
    return datetime.now(timezone.utc)

class ConversationMessage(BaseModel):
    """Single conversation message"""
    timestamp: str  # Changed from datetime to string (ISO format)
    user_message: Dict[str, Any]
    ai_response: Dict[str, Any]
    medical_extraction: Dict[str, Any] = {}  # Renamed from extracted_entities
    
    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"


class ConversationContext(BaseModel):
    """Complete conversation context"""
    conversation_id: str
    user_id: str
    session_id: Optional[str] = None
    
    messages: List[Dict[str, Any]] = []  # Changed to Dict for now, can be ConversationMessage later
    healthcare_context: Dict[str, Any] = {}
    intent_history: List[Dict[str, Any]] = []
    stakeholder_interactions: List[Dict[str, Any]] = []
    
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"

class UserProfile(BaseModel):  # Changed from UserMedicalProfile
    """User profile information"""
    user_id: str
    preferred_language: str = "english"
    communication_style: str = "standard"
    privacy_level: str = "standard"
    
    interaction_history: List[Dict[str, Any]] = []
    preferences: Dict[str, Any] = {}
    
    created_at: datetime
    updated_at: Optional[datetime] = None

class SessionState(BaseModel):
    """Current session state"""
    session_id: str
    user_id: str
    
    current_intent: Optional[str] = None
    current_context: Dict[str, Any] = {}
    conversation_flow: List[str] = []
    
    start_time: datetime
    last_activity: datetime
    interaction_count: int = 0
    is_active: bool = True

class StakeholderRoute(BaseModel):
    """Stakeholder routing decision"""
    stakeholder_type: str  # doctor, lab, admin, ai
    urgency_level: str     # low, medium, high, critical
    confidence: int = Field(ge=0, le=100)  # ← CHANGED: From float(0-1) to int(0-100)
    reasoning: str
    estimated_response_time: int  # seconds
    
    created_at: datetime = Field(default_factory=utcnow)
    metadata: Dict[str, Any] = {}
