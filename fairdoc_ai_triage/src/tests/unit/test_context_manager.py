"""
Unit tests for Medical Context Manager
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.app.core.context.manager import MedicalContextManager
from src.app.models.schemas.context import ConversationContext, StakeholderRoute


@pytest.mark.asyncio
class TestMedicalContextManager:
    """Test suite for Medical Context Manager"""
    
    async def test_initialize_context_manager(self):
        """Test context manager initialization"""
        context_manager = MedicalContextManager()
        
        # Mock Redis client
        context_manager.redis_client = AsyncMock()
        context_manager.redis_client.ping = AsyncMock()
        
        await context_manager.initialize()
        
        assert context_manager.redis_client is not None
        context_manager.redis_client.ping.assert_called_once()
    
    async def test_create_new_conversation_context(self):
        """Test creating new conversation context"""
        context_manager = MedicalContextManager()
        context_manager.redis_client = AsyncMock()
        context_manager.redis_client.get = AsyncMock(return_value=None)
        context_manager.redis_client.setex = AsyncMock()
        
        conversation_id = "conv_123"
        user_id = "user_456"
        
        context = await context_manager.get_conversation_context(
            conversation_id, user_id
        )
        
        assert context.conversation_id == conversation_id
        assert context.user_id == user_id
        assert len(context.messages) == 0
        assert isinstance(context.created_at, datetime)
    
    async def test_stakeholder_routing_emergency(self):
        """Test stakeholder routing for emergency queries"""
        context_manager = MedicalContextManager()
        context_manager.redis_client = AsyncMock()
        context_manager.redis_client.setex = AsyncMock()
        
        # Mock conversation context
        context = ConversationContext(
            conversation_id="conv_123",
            user_id="user_456",
            created_at=datetime.utcnow(),
            messages=[],
            medical_context={},
            intent_history=[],
            stakeholder_interactions=[]
        )
        
        emergency_query = "I'm having severe chest pain and difficulty breathing"
        
        route = await context_manager.route_stakeholder(
            "conv_123", emergency_query, context
        )
        
        assert route.stakeholder_type == "doctor"
        assert route.urgency_level == "high"
        assert route.confidence >= 0.8
        assert "emergency" in route.reasoning.lower() or "urgent" in route.reasoning.lower()
    
    async def test_stakeholder_routing_admin(self):
        """Test stakeholder routing for administrative queries"""
        context_manager = MedicalContextManager()
        context_manager.redis_client = AsyncMock()
        context_manager.redis_client.setex = AsyncMock()
        
        context = ConversationContext(
            conversation_id="conv_123",
            user_id="user_456",
            created_at=datetime.utcnow(),
            messages=[],
            medical_context={},
            intent_history=[],
            stakeholder_interactions=[]
        )
        
        admin_query = "I need to schedule an appointment for next week"
        
        route = await context_manager.route_stakeholder(
            "conv_123", admin_query, context
        )
        
        assert route.stakeholder_type == "admin"
        assert route.urgency_level == "low"
    
    async def test_update_conversation_context(self):
        """Test updating conversation context"""
        context_manager = MedicalContextManager()
        context_manager.redis_client = AsyncMock()
        context_manager.redis_client.get = AsyncMock(return_value=None)
        context_manager.redis_client.setex = AsyncMock()
        
        # Create initial context
        conversation_id = "conv_123"
        user_id = "user_456"
        context = await context_manager.get_conversation_context(
            conversation_id, user_id
        )
        
        # Update with new message
        message = {
            "user_id": user_id,
            "text": "I have a headache",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        ai_response = {
            "text": "I understand you have a headache. Can you describe the pain?",
            "intent": "symptom_assessment",
            "intent_confidence": 0.9
        }
        
        medical_info = {
            "symptoms": ["headache"],
            "severity": "mild"
        }
        
        updated_context = await context_manager.update_conversation(
            conversation_id, message, ai_response, medical_info
        )
        
        assert len(updated_context.messages) == 1
        assert updated_context.medical_context["symptoms"] == ["headache"]
        assert len(updated_context.intent_history) == 1
        assert updated_context.intent_history[0]["intent"] == "symptom_assessment"
