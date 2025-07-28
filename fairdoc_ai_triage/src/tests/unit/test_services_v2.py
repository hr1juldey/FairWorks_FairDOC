"""
Unit Tests for NICELookupService and ConversationQueue

Comprehensive test coverage for core V2 services
Follows pytest-asyncio patterns for async testing
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import fakeredis.aioredis

from src.app2.services.context.nice_lookup import NICELookupService
from src.app2.services.context.redis_queue import ConversationQueue


class TestNICELookupService:
    """Test suite for NICE protocol lookup functionality"""
    
    @pytest.fixture
    def mock_seed_data(self):
        """Mock NICE protocol data for testing"""
        return [
            {
                "protocol_code": "TEST_HEADACHE",
                "condition_name": "Test Headache Protocol",
                "primary_symptoms": ["headache", "head_pain"],
                "initial_questions": ["Where is the pain?"],
                "follow_up_questions": ["How severe is it?"],
                "emergency_criteria": "Sudden severe onset",
                "routine_criteria": "Mild chronic pattern",
                "self_care_criteria": "Mild tension type"
            }
        ]
    
    @pytest.fixture 
    def lookup_service(self, mock_seed_data):
        """Create NICELookupService with mock data"""
        return NICELookupService(seed_data=mock_seed_data)
    
    def test_successful_symptom_match(self, lookup_service):
        """Test successful protocol matching for known symptoms"""
        result = lookup_service.find_relevant_protocols("I have a bad headache")
        
        assert result["protocol_code"] == "TEST_HEADACHE"
        assert "Test Headache Protocol" in result["protocol_text"]
        assert "Where is the pain?" in result["protocol_text"]
    
    def test_case_insensitive_matching(self, lookup_service):
        """Test that symptom matching is case insensitive"""
        result = lookup_service.find_relevant_protocols("HEADACHE pain")
        
        assert result["protocol_code"] == "TEST_HEADACHE"
    
    def test_no_match_fallback(self, lookup_service):
        """Test fallback behavior when no protocol matches"""
        result = lookup_service.find_relevant_protocols("completely unrelated symptoms")
        
        assert result["protocol_code"] == "NONE"
        assert result["protocol_text"] == ""
    
    def test_text_cleaning(self, lookup_service):
        """Test text tokenization and cleaning"""
        # Test with punctuation and numbers
        result = lookup_service.find_relevant_protocols("I have head-pain!!! 123")
        
        assert result["protocol_code"] == "TEST_HEADACHE"
    
    def test_empty_input_handling(self, lookup_service):
        """Test handling of empty or whitespace input"""
        result = lookup_service.find_relevant_protocols("")
        assert result["protocol_code"] == "NONE"
        
        result = lookup_service.find_relevant_protocols("   ")
        assert result["protocol_code"] == "NONE"


class TestConversationQueue:
    """Test suite for Redis conversation queue functionality"""
    
    @pytest.fixture
    async def mock_redis(self):
        """Create fake Redis instance for testing"""
        fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return fake_redis
    
    @pytest.fixture
    async def conversation_queue(self, mock_redis):
        """Create ConversationQueue with mocked Redis"""
        queue = ConversationQueue()
        queue.redis = mock_redis
        return queue
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, conversation_queue):
        """Test starting a new conversation"""
        user_id = "test_user_123"
        symptoms = "I have chest pain"
        
        conversation_id = await conversation_queue.start_conversation(user_id, symptoms)
        
        assert conversation_id.startswith(f"conv_{user_id}_")
        
        # Verify state was stored in Redis
        state_key = f"{conversation_queue.state_prefix}:{conversation_id}"
        stored_data = await conversation_queue.redis.get(state_key)
        assert stored_data is not None
        
        state = json.loads(stored_data)
        assert state["user_id"] == user_id
        assert state["initial_symptoms"] == symptoms
        assert state["status"] == "active"
        assert state["turn_count"] == 1
    
    @pytest.mark.asyncio
    async def test_update_conversation_turn(self, conversation_queue):
        """Test updating conversation with new turn"""
        # First start a conversation
        conversation_id = await conversation_queue.start_conversation("test_user", "headache")
        
        # Prepare agent result
        agent_result = {
            "outcome": "inconclusive",
            "confidence": 75,
            "next_question": "How long have you had this headache?",
            "reasoning": "Need more information",
            "red_flags": []
        }
        
        # Update the conversation
        updated_state = await conversation_queue.update_conversation_turn(
            conversation_id=conversation_id,
            user_response="It started this morning",
            agent_result=agent_result
        )
        
        assert updated_state["turn_count"] == 2
        assert updated_state["current_outcome"] == "inconclusive"
        assert len(updated_state["conversation_history"]) == 1
        
        turn_data = updated_state["conversation_history"][0]
        assert turn_data["user_response"] == "It started this morning"
        assert turn_data["outcome"] == "inconclusive"
        assert turn_data["confidence"] == 75
    
    @pytest.mark.asyncio
    async def test_conversation_completion(self, conversation_queue):
        """Test conversation completion logic"""
        conversation_id = await conversation_queue.start_conversation("test_user", "chest pain")
        
        # Agent result indicating emergency
        agent_result = {
            "outcome": "emergency",
            "confidence": 95,
            "is_complete": True,
            "reasoning": "Potential cardiac event",
            "red_flags": ["crushing_pain"]
        }
        
        updated_state = await conversation_queue.update_conversation_turn(
            conversation_id=conversation_id,
            user_response="Yes, crushing pain radiating to arm",
            agent_result=agent_result
        )
        
        assert updated_state["status"] == "completed"
        assert updated_state["current_outcome"] == "emergency"
        
        # Verify conversation was moved to completed queue
        completed_conversations = await conversation_queue.redis.lrange(
            f"{conversation_queue.queue_prefix}:completed", 0, -1
        )
        assert conversation_id in completed_conversations
    
    @pytest.mark.asyncio
    async def test_get_conversation_state(self, conversation_queue):
        """Test retrieving conversation state"""
        conversation_id = await conversation_queue.start_conversation("test_user", "symptoms")
        
        retrieved_state = await conversation_queue.get_conversation_state(conversation_id)
        
        assert retrieved_state is not None
        assert retrieved_state["conversation_id"] == conversation_id
        assert retrieved_state["initial_symptoms"] == "symptoms"
    
    @pytest.mark.asyncio
    async def test_nonexistent_conversation(self, conversation_queue):
        """Test handling of non-existent conversation"""
        retrieved_state = await conversation_queue.get_conversation_state("nonexistent_id")
        assert retrieved_state is None
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_conversation(self, conversation_queue):
        """Test error handling when updating non-existent conversation"""
        agent_result = {"outcome": "test", "confidence": 50}
        
        with pytest.raises(ValueError, match="Conversation .* not found"):
            await conversation_queue.update_conversation_turn(
                conversation_id="nonexistent_id",
                user_response="test",
                agent_result=agent_result
            )
    
    @pytest.mark.asyncio 
    async def test_red_flags_accumulation(self, conversation_queue):
        """Test that red flags are properly accumulated across turns"""
        conversation_id = await conversation_queue.start_conversation("test_user", "chest pain")
        
        # First turn with red flags
        agent_result_1 = {
            "outcome": "inconclusive", 
            "confidence": 70,
            "red_flags": ["chest_pain"]
        }
        
        await conversation_queue.update_conversation_turn(
            conversation_id=conversation_id,
            user_response="Sharp pain",
            agent_result=agent_result_1
        )
        
        # Second turn with additional red flags
        agent_result_2 = {
            "outcome": "emergency",
            "confidence": 90,
            "red_flags": ["shortness_breath"]
        }
        
        updated_state = await conversation_queue.update_conversation_turn(
            conversation_id=conversation_id,
            user_response="Yes, hard to breathe",
            agent_result=agent_result_2
        )
        
        # Verify red flags are accumulated
        assert "chest_pain" in updated_state["red_flags_detected"]
        assert "shortness_breath" in updated_state["red_flags_detected"]
        assert len(updated_state["red_flags_detected"]) == 2


# Integration test combining both services
class TestServiceIntegration:
    """Integration tests for NICELookupService and ConversationQueue"""
    
    @pytest.mark.asyncio 
    async def test_nice_lookup_integration(self):
        """Test integration between NICE lookup and conversation queue"""
        # Setup services
        nice_service = NICELookupService()
        fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        
        conversation_queue = ConversationQueue()
        conversation_queue.redis = fake_redis
        
        # Start conversation
        conversation_id = await conversation_queue.start_conversation(
            "test_user", "I have a severe headache"
        )
        
        # Look up protocol
        protocol_info = nice_service.find_relevant_protocols("severe headache")
        
        # Simulate agent result using protocol
        agent_result = {
            "outcome": "inconclusive",
            "confidence": 80,
            "next_question": "Is this the worst headache you've ever experienced?",
            "reasoning": f"Using protocol: {protocol_info['protocol_code']}",
            "red_flags": []
        }
        
        # Update conversation
        updated_state = await conversation_queue.update_conversation_turn(
            conversation_id=conversation_id,
            user_response="Yes, very severe pain",
            agent_result=agent_result
        )
        
        # Verify integration
        assert updated_state["turn_count"] == 2
        assert protocol_info["protocol_code"] in agent_result["reasoning"]
        assert updated_state["current_outcome"] == "inconclusive"
