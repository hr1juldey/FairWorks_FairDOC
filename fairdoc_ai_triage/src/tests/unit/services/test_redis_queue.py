"""
Redis Queue Integration Tests

Tests Redis conversation state management and queue operations.
Validates conversation persistence and state transitions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json
from datetime import datetime, timezone

from src.app2.services.context.redis_queue import ConversationQueue

class TestRedisConnectionManagement:
    """Test Redis connection initialization and health"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.queue = ConversationQueue()
        
    @pytest.mark.asyncio
    async def test_redis_initialization_success(self):
        """Test successful Redis connection initialization"""
        with patch('redis.asyncio.Redis.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock(return_value=True)
            
            await self.queue.initialize()
            
            assert self.queue.redis is not None
            mock_redis_instance.ping.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_redis_initialization_failure(self):
        """Test Redis connection initialization failure"""
        with patch('redis.asyncio.Redis.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
            
            with pytest.raises(ConnectionError):
                await self.queue.initialize()

class TestConversationCreation:
    """Test conversation creation and initial state management"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.queue = ConversationQueue()
        self.queue.redis = AsyncMock()
        
    @pytest.mark.asyncio
    async def test_start_conversation_success(self):
        """Test successful conversation creation"""
        self.queue.redis.setex = AsyncMock(return_value=True)
        self.queue.redis.lpush = AsyncMock(return_value=1)
        
        conversation_id = await self.queue.start_conversation(
            user_id="test_user_123",
            initial_symptoms="I have chest pain"
        )
        
        assert conversation_id.startswith("conv_test_user_123_")
        self.queue.redis.setex.assert_called_once()
        self.queue.redis.lpush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_conversation_state_structure(self):
        """Test that conversation state has correct structure"""
        self.queue.redis.setex = AsyncMock(return_value=True)
        self.queue.redis.lpush = AsyncMock(return_value=1)
        
        conversation_id = await self.queue.start_conversation(
            user_id="test_user_123", 
            initial_symptoms="I have a headache"
        )
        
        # Get the state that was stored
        call_args = self.queue.redis.setex.call_args
        stored_state = json.loads(call_args[0][2])
        
        # Validate state structure
        assert stored_state["conversation_id"] == conversation_id
        assert stored_state["user_id"] == "test_user_123"
        assert stored_state["initial_symptoms"] == "I have a headache"
        assert stored_state["status"] == "active"
        assert stored_state["turn_count"] == 1
        assert stored_state["current_outcome"] == "inconclusive"
        assert "conversation_history" in stored_state
        assert "created_at" in stored_state

class TestConversationUpdates:
    """Test conversation turn updates and state transitions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.queue = ConversationQueue()
        self.queue.redis = AsyncMock()
        
    @pytest.mark.asyncio
    async def test_update_conversation_turn_success(self):
        """Test successful conversation turn update"""
        # Mock existing conversation state
        existing_state = {
            "conversation_id": "conv_test_123",
            "user_id": "test_user",
            "status": "active",
            "turn_count": 1,
            "current_outcome": "inconclusive",
            "conversation_history": [],
            "red_flags_detected": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
        
        self.queue.redis.get = AsyncMock(return_value=json.dumps(existing_state))
        self.queue.redis.setex = AsyncMock(return_value=True)
        self.queue.redis.lpush = AsyncMock(return_value=1)
        
        agent_result = {
            "outcome": "routine_doctor",
            "confidence": 85,
            "next_question": "Do you have any allergies?",
            "reasoning": "Symptoms suggest routine consultation needed",
            "red_flags": ["persistent_symptoms"],
            "is_complete": False
        }
        
        updated_state = await self.queue.update_conversation_turn(
            conversation_id="conv_test_123",
            user_response="The pain is getting worse",
            agent_result=agent_result
        )
        
        assert updated_state["turn_count"] == 2
        assert updated_state["current_outcome"] == "routine_doctor"
        assert len(updated_state["conversation_history"]) == 1
        assert updated_state["red_flags_detected"] == ["persistent_symptoms"]
    
    @pytest.mark.asyncio
    async def test_conversation_completion_handling(self):
        """Test conversation completion and queue movement"""
        existing_state = {
            "conversation_id": "conv_test_123",
            "user_id": "test_user",
            "status": "active",
            "turn_count": 3,
            "current_outcome": "inconclusive",
            "conversation_history": [],
            "red_flags_detected": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
        
        self.queue.redis.get = AsyncMock(return_value=json.dumps(existing_state))
        self.queue.redis.setex = AsyncMock(return_value=True)
        self.queue.redis.lpush = AsyncMock(return_value=1)
        
        agent_result = {
            "outcome": "emergency",
            "confidence": 95,
            "reasoning": "Emergency situation detected",
            "red_flags": ["severe_symptoms"],
            "is_complete": True
        }
        
        updated_state = await self.queue.update_conversation_turn(
            conversation_id="conv_test_123",
            user_response="I'm having severe chest pain",
            agent_result=agent_result
        )
        
        assert updated_state["status"] == "completed"
        assert updated_state["current_outcome"] == "emergency"
        # Verify moved to completed queue
        self.queue.redis.lpush.assert_called_with(
            f"{self.queue.queue_prefix}:completed", 
            "conv_test_123"
        )
    
    @pytest.mark.asyncio
    async def test_conversation_not_found_error(self):
        """Test error handling when conversation doesn't exist"""
        self.queue.redis.get = AsyncMock(return_value=None)
        
        agent_result = {"outcome": "self_care", "confidence": 70}
        
        with pytest.raises(ValueError, match="Conversation conv_nonexistent not found"):
            await self.queue.update_conversation_turn(
                conversation_id="conv_nonexistent",
                user_response="Test response", 
                agent_result=agent_result
            )

class TestConversationRetrieval:
    """Test conversation state retrieval and queries"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.queue = ConversationQueue()
        self.queue.redis = AsyncMock()
        
    @pytest.mark.asyncio
    async def test_get_conversation_state_success(self):
        """Test successful conversation state retrieval"""
        test_state = {
            "conversation_id": "conv_test_123",
            "user_id": "test_user",
            "status": "active",
            "turn_count": 2,
            "current_outcome": "inconclusive"
        }
        
        self.queue.redis.get = AsyncMock(return_value=json.dumps(test_state))
        
        result = await self.queue.get_conversation_state("conv_test_123")
        
        assert result == test_state
        self.queue.redis.get.assert_called_once_with(
            f"{self.queue.state_prefix}:conv_test_123"
        )
    
    @pytest.mark.asyncio
    async def test_get_conversation_state_not_found(self):
        """Test conversation state retrieval when conversation doesn't exist"""
        self.queue.redis.get = AsyncMock(return_value=None)
        
        result = await self.queue.get_conversation_state("conv_nonexistent")
        
        assert result is None

class TestRedisKeyManagement:
    """Test Redis key management and expiration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.queue = ConversationQueue()
        self.queue.redis = AsyncMock()
        
    @pytest.mark.asyncio
    async def test_conversation_expiration_set(self):
        """Test that conversation states have proper expiration"""
        self.queue.redis.setex = AsyncMock(return_value=True)
        self.queue.redis.lpush = AsyncMock(return_value=1)
        
        await self.queue.start_conversation(
            user_id="test_user",
            initial_symptoms="Test symptoms"
        )
        
        # Verify setex was called with 24 hour expiration
        call_args = self.queue.redis.setex.call_args
        expiration_seconds = call_args[0][1]
        assert expiration_seconds == 86400  # 24 hours in seconds
    
    @pytest.mark.asyncio
    async def test_redis_key_prefix_consistency(self):
        """Test that Redis keys use consistent prefixes"""
        self.queue.redis.setex = AsyncMock(return_value=True)
        self.queue.redis.lpush = AsyncMock(return_value=1)
        
        conversation_id = await self.queue.start_conversation(
            user_id="test_user",
            initial_symptoms="Test symptoms"
        )
        
        # Check state key prefix
        state_call = self.queue.redis.setex.call_args
        state_key = state_call[0][0]
        assert state_key.startswith(f"{self.queue.state_prefix}:")
        # Verify the conversation_id is part of the state key
        assert conversation_id in state_key
        
        # Check queue key prefix  
        queue_call = self.queue.redis.lpush.call_args
        queue_key = queue_call[0][0]
        assert queue_key.startswith(f"{self.queue.queue_prefix}:")
        
        # Verify the conversation_id was added to the active queue
        assert self.queue.redis.lpush.call_args[0][1] == conversation_id

class TestErrorHandling:
    """Test error handling and resilience"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.queue = ConversationQueue()
        self.queue.redis = AsyncMock()
        
    @pytest.mark.asyncio
    async def test_redis_connection_lost_during_operation(self):
        """Test handling of Redis connection loss during operations"""
        self.queue.redis.setex = AsyncMock(side_effect=ConnectionError("Connection lost"))
        
        with pytest.raises(ConnectionError):
            await self.queue.start_conversation(
                user_id="test_user",
                initial_symptoms="Test symptoms"
            )
    
    @pytest.mark.asyncio
    async def test_invalid_json_in_stored_state(self):
        """Test handling of corrupted JSON in stored conversation state"""
        self.queue.redis.get = AsyncMock(return_value="invalid json data")
        
        with pytest.raises(json.JSONDecodeError):
            await self.queue.get_conversation_state("conv_test_123")
    
    @pytest.mark.asyncio
    async def test_malformed_agent_result(self):
        """Test handling of malformed agent result data"""
        existing_state = {
            "conversation_id": "conv_test_123",
            "user_id": "test_user", 
            "status": "active",
            "turn_count": 1,
            "conversation_history": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
        
        self.queue.redis.get = AsyncMock(return_value=json.dumps(existing_state))
        self.queue.redis.setex = AsyncMock(return_value=True)
        
        # Agent result missing required fields
        malformed_agent_result = {"confidence": 70}  # Missing outcome
        
        with pytest.raises(KeyError):
            await self.queue.update_conversation_turn(
                conversation_id="conv_test_123",
                user_response="Test response",
                agent_result=malformed_agent_result
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
