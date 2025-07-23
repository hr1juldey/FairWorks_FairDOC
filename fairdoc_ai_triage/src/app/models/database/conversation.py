"""
Fairdoc AI Conversation Database Models
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, JSON, Text, Integer
from sqlalchemy.dialects.postgresql import UUID

from src.app.core.database import Base

def utcnow():
    """Return timezone-aware UTC now"""
    return datetime.now(timezone.utc)

class ConversationModel(Base):
    """Database model for storing conversations"""
    
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    
    # Conversation content
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=True)
    
    # Context and metadata
    conversation_context = Column(JSON, nullable=True)
    user_context = Column(JSON, nullable=True)
    ai_context = Column(JSON, nullable=True)
    
    # Routing and stakeholder info
    stakeholder_type = Column(String(50), nullable=True)
    urgency_level = Column(String(20), nullable=True)
    routing_confidence = Column(Integer, nullable=True)
    
    # Intent and analysis
    detected_intent = Column(String(100), nullable=True)
    intent_confidence = Column(Integer, nullable=True)
    extracted_entities = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Response metrics
    response_time_ms = Column(Integer, nullable=True)
    model_used = Column(String(100), nullable=True)


class UserSessionModel(Base):
    """Database model for user sessions"""
    
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Session metadata
    session_data = Column(JSON, nullable=True)
    user_profile = Column(JSON, nullable=True)
    preferences = Column(JSON, nullable=True)
    
    # Session tracking
    first_interaction = Column(DateTime(timezone=True), default=utcnow)
    last_interaction = Column(DateTime(timezone=True), default=utcnow)
    interaction_count = Column(Integer, default=0)
    
    # Session state
    is_active = Column(String(10), default="active")
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
