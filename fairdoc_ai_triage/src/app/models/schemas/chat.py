"""
Fairdoc AI Chat API Schemas with Thinking Process Support
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

def utcnow():
    return datetime.now(timezone.utc)


class ThinkingProcessData(BaseModel):
    """Thinking process analysis data"""
    content: str = Field(..., description="Raw thinking process content")
    word_count: int = Field(ge=0, description="Word count of thinking process")
    safety_flags: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    confidence_indicators: Dict[str, Any] = Field(default_factory=dict)
    medical_considerations: List[str] = Field(default_factory=list)
    timestamp: str = Field(..., description="When thinking was processed")


class SafetySummary(BaseModel):
    """Safety assessment summary"""
    overall_safety_level: str = Field(..., description="Overall safety assessment")
    total_flags: int = Field(ge=0, description="Total safety flags detected")
    high_severity_count: int = Field(ge=0, description="High severity flags")
    reasoning_quality: str = Field(..., description="Quality of reasoning process")
    requires_review: bool = Field(..., description="Whether human review is needed")
    generated_at: str = Field(..., description="When summary was generated")


class ChatMessageRequest(BaseModel):
    """Incoming chat message request"""
    user_id: str = Field(..., description="Unique user identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message text")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = Field(default_factory=utcnow)
    
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
    """Chat message response with thinking process support"""
    message_id: str = Field(..., description="Unique message identifier")
    response: str = Field(..., description="Clean AI response text (thinking removed)")
    
    # Response metadata
    intent: Optional[str] = Field(None, description="Detected user intent")
    intent_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Routing information
    stakeholder_route: Optional[str] = Field(None, description="Recommended stakeholder")
    urgency_level: Optional[str] = Field(None, description="Assessed urgency level")
    estimated_wait_time: Optional[int] = Field(None, description="Estimated response time in seconds")
    
    # Enhanced AI metadata
    thinking_process: Optional[ThinkingProcessData] = Field(None, description="AI thinking process data")
    safety_summary: Optional[SafetySummary] = Field(None, description="Safety assessment summary")
    
    # Response context
    context_used: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    
    # Technical metadata
    model_used: Optional[str] = None
    response_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg_789xyz",
                "response": "I understand you've been experiencing headaches. Can you describe the pain?",
                "intent": "symptom_assessment",
                "intent_confidence": 0.85,
                "stakeholder_route": "doctor",
                "urgency_level": "medium",
                "estimated_wait_time": 1800,
                "thinking_process": {
                    "content": "The user mentions headaches lasting several days...",
                    "word_count": 45,
                    "safety_flags": [],
                    "reasoning_quality": "high"
                },
                "safety_summary": {
                    "overall_safety_level": "safe",
                    "requires_review": False
                },
                "model_used": "deepseek-r1:8b",
                "response_time_ms": 245
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str = "Fairdoc AI Triage System"
    version: str = "0.1.0" 
    timestamp: datetime = Field(default_factory=utcnow)
    
    # Component health
    database: str = "connected"
    redis: str = "connected"
    ai_service: str = "ready"
    
    uptime_seconds: Optional[int] = None
