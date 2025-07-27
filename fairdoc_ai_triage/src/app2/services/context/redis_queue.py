"""
Redis-based Conversation State Management for Multi-turn Medical Conversations
"""
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from redis.asyncio import Redis
import structlog
from src.app.core.config import settings

logger = structlog.get_logger(__name__)

class ConversationQueue:
    """Redis-powered conversation state management"""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.queue_prefix = "fairdoc:v2:conversation"
        self.state_prefix = "fairdoc:v2:state"
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        await self.redis.ping()
        logger.info("✅ V2 Conversation Queue initialized")
    
    async def start_conversation(self, user_id: str, initial_symptoms: str) -> str:
        """Start new multi-turn conversation"""
        conversation_id = f"conv_{user_id}_{int(datetime.now().timestamp())}"
        
        conversation_state = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "status": "active",
            "turn_count": 1,
            "initial_symptoms": initial_symptoms,
            "current_outcome": "inconclusive",
            "conversation_history": [],
            "nice_protocols_used": [],
            "red_flags_detected": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
        
        # Store in Redis with 24h expiration
        await self.redis.setex(
            f"{self.state_prefix}:{conversation_id}",
            int(timedelta(hours=24).total_seconds()),
            json.dumps(conversation_state)
        )
        
        # Add to processing queue
        await self.redis.lpush(f"{self.queue_prefix}:pending", conversation_id)
        
        logger.info("🔄 Started new conversation", 
                   conversation_id=conversation_id, user_id=user_id)
        return conversation_id
    
    async def update_conversation_turn(self, 
                                     conversation_id: str, 
                                     user_response: str,
                                     agent_result: Dict) -> Dict:
        """Update conversation with new turn"""
        
        state_data = await self.redis.get(f"{self.state_prefix}:{conversation_id}")
        if not state_data:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        state = json.loads(state_data)
        state["turn_count"] += 1
        state["last_activity"] = datetime.now(timezone.utc).isoformat()
        state["current_outcome"] = agent_result["outcome"]
        
        # Add turn to history
        turn_data = {
            "turn": state["turn_count"],
            "user_response": user_response,
            "agent_question": agent_result.get("next_question"),
            "agent_reasoning": agent_result.get("reasoning"),
            "outcome": agent_result["outcome"],
            "confidence": agent_result["confidence"],
            "red_flags": agent_result.get("red_flags", []),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        state["conversation_history"].append(turn_data)
        
        # Update red flags
        if agent_result.get("red_flags"):
            state["red_flags_detected"].extend(agent_result["red_flags"])
        
        # Check if conversation is complete
        if agent_result.get("is_complete") or agent_result["outcome"] in ["emergency", "spam_detected"]:
            state["status"] = "completed"
            # Move to completed queue for PostgreSQL persistence
            await self.redis.lpush(f"{self.queue_prefix}:completed", conversation_id)
        
        # Update Redis state
        await self.redis.setex(
            f"{self.state_prefix}:{conversation_id}",
            int(timedelta(hours=24).total_seconds()),
            json.dumps(state)
        )
        
        return state
    
    async def get_conversation_state(self, conversation_id: str) -> Optional[Dict]:
        """Retrieve conversation state"""
        state_data = await self.redis.get(f"{self.state_prefix}:{conversation_id}")
        return json.loads(state_data) if state_data else None
