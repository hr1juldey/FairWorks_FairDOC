"""
Fairdoc AI Chat API Schemas
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    """Incoming chat message request"""
    user_id: str = Field(..., description="Unique user identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message text")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    # Message context
    message_type: str = Field(default="text", description="Message type: text, voice, image")
    language: Optional[str] = Field(default="english", description="Message language")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "message": "I have been experiencing headaches for the past few days",
                "session_id": "session_abc123",
                "metadata": {"source": "raven_chat", "urgency": "low"},
                "message_type": "text",
                "language": "english"
            }
        }


class ChatMessageResponse(BaseModel):
    """Chat message response"""
    message_id: str = Field(..., description="Unique message identifier")
    response: str = Field(..., description="AI response text")
    
    # Response metadata
    intent: Optional[str] = Field(None, description="Detected user intent")
    intent_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Routing information
    stakeholder_route: Optional[str] = Field(None, description="Recommended stakeholder")
    urgency_level: Optional[str] = Field(None, description="Assessed urgency level")
    estimated_wait_time: Optional[int] = Field(None, description="Estimated response time in seconds")
    
    # Response context
    context_used: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    
    # Technical metadata
    model_used: Optional[str] = None
    response_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg_789xyz",
                "response": "I understand you've been experiencing headaches. Can you describe the pain - is it throbbing, sharp, or dull?",
                "intent": "symptom_assessment",
                "intent_confidence": 0.85,
                "stakeholder_route": "doctor",
                "urgency_level": "medium",
                "estimated_wait_time": 1800,
                "suggestions": ["Describe pain type", "Mention duration", "List any triggers"],
                "model_used": "healthcare_llm",
                "response_time_ms": 245
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str = "Fairdoc AI Triage System"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Component health
    database: str = "connected"
    redis: str = "connected"
    ai_service: str = "ready"
    
    uptime_seconds: Optional[int] = None
