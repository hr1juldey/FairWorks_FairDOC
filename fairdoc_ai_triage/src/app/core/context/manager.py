"""
Medical Context Manager - Core Intelligence Component
Priority implementation for Phase 1
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel
from redis.asyncio import Redis

from src.app.core.config import settings
from src.app.models.database.conversation import ConversationModel
from src.app.models.schemas.context import (
    ConversationContext,
    UserMedicalProfile,
    SessionState,
    StakeholderRoute
)

logger = structlog.get_logger(__name__)


class MedicalContextManager:
    """
    Medical Context Manager - The AI brain's memory system
    
    Responsibilities:
    - Track conversation history and context
    - Maintain user medical profiles
    - Manage session states across interactions
    - Route queries to appropriate stakeholders
    - Learn from interactions for continuous improvement
    """
    
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.conversation_memory: Dict[str, ConversationContext] = {}
        self.user_profiles: Dict[str, UserMedicalProfile] = {}
        self.session_states: Dict[str, SessionState] = {}
        self.routing_decisions: Dict[str, StakeholderRoute] = {}
        
    async def initialize(self):
        """Initialize context manager with Redis connection"""
        try:
            self.redis_client = Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Context Manager connected to Redis")
            
            # Load existing contexts from Redis
            await self._load_contexts()
            
        except Exception as e:
            logger.error("❌ Failed to initialize Context Manager", error=str(e))
            raise
    
    async def get_conversation_context(
        self, 
        conversation_id: str,
        user_id: str
    ) -> ConversationContext:
        """
        Retrieve or create conversation context
        """
        context_key = f"conversation:{conversation_id}"
        
        # Try to get from memory first
        if context_key in self.conversation_memory:
            return self.conversation_memory[context_key]
        
        # Try to get from Redis
        context_data = await self.redis_client.get(context_key)
        if context_data:
            context = ConversationContext.model_validate_json(context_data)
            self.conversation_memory[context_key] = context
            return context
        
        # Create new context
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            messages=[],
            medical_context={},
            intent_history=[],
            stakeholder_interactions=[]
        )
        
        # Store in memory and Redis
        self.conversation_memory[context_key] = context
        await self._save_context(context_key, context)
        
        logger.info("📝 Created new conversation context", 
                   conversation_id=conversation_id, user_id=user_id)
        
        return context
    
    async def update_conversation(
        self,
        conversation_id: str,
        message: Dict[str, Any],
        ai_response: Dict[str, Any],
        extracted_medical_info: Dict[str, Any]
    ) -> ConversationContext:
        """
        Update conversation context with new message and AI response
        """
        context = await self.get_conversation_context(conversation_id, message.get("user_id"))
        
        # Add message to context
        context.messages.append({
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": message,
            "ai_response": ai_response,
            "medical_extraction": extracted_medical_info
        })
        
        # Update medical context with extracted information
        context.medical_context.update(extracted_medical_info)
        
        # Update intent if detected
        if "intent" in ai_response:
            context.intent_history.append({
                "intent": ai_response["intent"],
                "confidence": ai_response.get("intent_confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Save updated context
        context_key = f"conversation:{conversation_id}"
        self.conversation_memory[context_key] = context
        await self._save_context(context_key, context)
        
        logger.info("🔄 Updated conversation context", 
                   conversation_id=conversation_id, 
                   messages_count=len(context.messages))
        
        return context
    
    async def route_stakeholder(
        self,
        conversation_id: str,
        query: str,
        context: ConversationContext
    ) -> StakeholderRoute:
        """
        Intelligent routing to appropriate stakeholder based on query complexity
        """
        # Analyze query complexity and medical urgency
        routing_decision = await self._analyze_routing_requirements(
            query, context
        )
        
        # Store routing decision
        route_key = f"route:{conversation_id}:{datetime.utcnow().timestamp()}"
        self.routing_decisions[route_key] = routing_decision
        
        # Save to Redis with expiration
        await self.redis_client.setex(
            route_key,
            timedelta(hours=24).total_seconds(),
            routing_decision.model_dump_json()
        )
        
        logger.info("🎯 Stakeholder routing decision made",
                   conversation_id=conversation_id,
                   stakeholder=routing_decision.stakeholder_type,
                   urgency=routing_decision.urgency_level)
        
        return routing_decision
    
    async def get_user_profile(self, user_id: str) -> UserMedicalProfile:
        """
        Get or create user medical profile
        """
        profile_key = f"user_profile:{user_id}"
        
        # Check memory first
        if profile_key in self.user_profiles:
            return self.user_profiles[profile_key]
        
        # Check Redis
        profile_data = await self.redis_client.get(profile_key)
        if profile_data:
            profile = UserMedicalProfile.model_validate_json(profile_data)
            self.user_profiles[profile_key] = profile
            return profile
        
        # Create new profile
        profile = UserMedicalProfile(
            user_id=user_id,
            created_at=datetime.utcnow(),
            medical_history={},
            preferences={},
            risk_factors=[],
            medications=[],
            allergies=[]
        )
        
        self.user_profiles[profile_key] = profile
        await self._save_user_profile(profile_key, profile)
        
        return profile
    
    async def _analyze_routing_requirements(
        self,
        query: str,
        context: ConversationContext
    ) -> StakeholderRoute:
        """
        Analyze query to determine appropriate stakeholder routing
        """
        # Simple rule-based routing (Phase 1)
        # Will be enhanced with ML models in later phases
        
        query_lower = query.lower()
        urgency_keywords = ["emergency", "urgent", "pain", "chest pain", "difficulty breathing"]
        doctor_keywords = ["diagnosis", "treatment", "medication", "symptoms"]
        lab_keywords = ["test results", "blood work", "lab report", "x-ray"]
        admin_keywords = ["appointment", "schedule", "billing", "insurance"]
        
        if any(keyword in query_lower for keyword in urgency_keywords):
            return StakeholderRoute(
                stakeholder_type="doctor",
                urgency_level="high",
                confidence=0.9,
                reasoning="Emergency or urgent medical keywords detected",
                estimated_response_time=300  # 5 minutes
            )
        elif any(keyword in query_lower for keyword in doctor_keywords):
            return StakeholderRoute(
                stakeholder_type="doctor",
                urgency_level="medium",
                confidence=0.8,
                reasoning="Medical consultation required",
                estimated_response_time=1800  # 30 minutes
            )
        elif any(keyword in query_lower for keyword in lab_keywords):
            return StakeholderRoute(
                stakeholder_type="lab",
                urgency_level="low",
                confidence=0.7,
                reasoning="Lab results interpretation needed",
                estimated_response_time=3600  # 1 hour
            )
        elif any(keyword in query_lower for keyword in admin_keywords):
            return StakeholderRoute(
                stakeholder_type="admin",
                urgency_level="low",
                confidence=0.6,
                reasoning="Administrative request",
                estimated_response_time=7200  # 2 hours
            )
        else:
            return StakeholderRoute(
                stakeholder_type="ai",
                urgency_level="low",
                confidence=0.5,
                reasoning="General query - AI assistance sufficient",
                estimated_response_time=30  # 30 seconds
            )
    
    async def _save_context(self, key: str, context: ConversationContext):
        """Save context to Redis with expiration"""
        await self.redis_client.setex(
            key,
            timedelta(days=30).total_seconds(),  # Keep for 30 days
            context.model_dump_json()
        )
    
    async def _save_user_profile(self, key: str, profile: UserMedicalProfile):
        """Save user profile to Redis"""
        await self.redis_client.set(key, profile.model_dump_json())
    
    async def _load_contexts(self):
        """Load existing contexts from Redis on startup"""
        try:
            # Load conversation contexts
            keys = await self.redis_client.keys("conversation:*")
            for key in keys:
                context_data = await self.redis_client.get(key)
                if context_data:
                    context = ConversationContext.model_validate_json(context_data)
                    self.conversation_memory[key] = context
            
            logger.info(f"📚 Loaded {len(keys)} conversation contexts from Redis")
            
        except Exception as e:
            logger.error("❌ Error loading contexts from Redis", error=str(e))
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("🧹 Context Manager cleanup completed")
