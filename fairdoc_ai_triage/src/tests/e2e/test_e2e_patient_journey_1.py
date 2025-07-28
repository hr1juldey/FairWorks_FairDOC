"""
End-to-End Test: Complete Patient Journey

Tests the full patient triage workflow from initial symptoms 
through multi-turn conversation to final medical outcome.

File: src/tests/e2e/test_e2e_patient_journey.py
"""

import pytest
import asyncio
import json
from httpx import AsyncClient
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for E2E tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_services():
    """Mock external services for E2E testing"""
    
    # Mock Redis with in-memory store
    mock_redis_store = {}
    
    class MockRedis:
        async def get(self, key):
            return mock_redis_store.get(key)
        
        async def set(self, key, value):
            mock_redis_store[key] = value
        
        async def setex(self, key, ttl, value):
            mock_redis_store[key] = value
        
        async def lpush(self, key, value):
            if key not in mock_redis_store:
                mock_redis_store[key] = []
            mock_redis_store[key].append(value)
        
        async def ping(self):
            return True
        
        async def close(self):
            pass
    
    # Mock DSPy Agent responses
    class MockMedicalAgent:
        def __init__(self):
            self.turn_count = 0
            
        async def process_turn(self, symptoms, nice_context=""):
            self.turn_count += 1
            
            # Simulate realistic medical triage logic
            if "headache" in symptoms.lower() and self.turn_count == 1:
                return {
                    "outcome": "inconclusive",
                    "confidence": 65,
                    "next_question": "How long have you had this headache?",
                    "reasoning": "Need duration to assess severity",
                    "red_flags": [],
                    "is_complete": False
                }
            elif "headache" in symptoms.lower() and "hours" in symptoms.lower():
                return {
                    "outcome": "routine_doctor",
                    "confidence": 78,
                    "next_question": None,
                    "reasoning": "Persistent headache warrants medical evaluation",
                    "red_flags": [],
                    "is_complete": True
                }
            elif "chest pain" in symptoms.lower():
                return {
                    "outcome": "emergency",
                    "confidence": 95,
                    "next_question": None,
                    "reasoning": "Chest pain requires immediate medical attention",
                    "red_flags": ["chest_pain", "cardiac_risk"],
                    "is_complete": True
                }
            else:
                return {
                    "outcome": "inconclusive",
                    "confidence": 50,
                    "next_question": "Can you describe your symptoms in more detail?",
                    "reasoning": "Insufficient information for assessment",
                    "red_flags": [],
                    "is_complete": False
                }
    
    mock_redis = MockRedis() 
    mock_agent = MockMedicalAgent()
    
    return {
        "redis": mock_redis,
        "agent": mock_agent,
        "store": mock_redis_store
    }


@pytest.fixture
async def e2e_client(mock_services):
    """FastAPI test client with mocked services"""
    
    # Patch dependencies to use mocks
    with patch('src.app2.services.context.redis_queue.Redis.from_url') as mock_redis_factory, \
         patch('src.app2.services.dspy.medical_agent.MedicalTriageAgent') as mock_agent_class, \
         patch('src.app2.core.dependencies_v2.init_services') as mock_init:
        
        mock_redis_factory.return_value = mock_services["redis"]
        mock_agent_class.return_value = mock_services["agent"]
        mock_init.return_value = None
        
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client


class TestCompletePatientJourney:
    """E2E test for complete patient medical triage journey"""
    
    @pytest.mark.asyncio
    async def test_simple_headache_journey(self, e2e_client, mock_services):
        """Test complete headache triage from start to finish"""
        
        # Step 1: Patient starts conversation with headache
        initial_payload = {
            "user_message": "I have a severe headache",
            "stakeholder_role": "patient", 
            "stakeholder_id": "patient_headache_001",
            "chat_provider": "api_direct"
        }
        
        response1 = await e2e_client.post("/api/v2/medical/chat", json=initial_payload)
        assert response1.status_code == 200
        
        data1 = response1.json()
        conversation_id = data1["conversation_id"]
        
        # Verify initial response structure
        assert "conversation_id" in data1
        assert data1["medical_outcome"] == "need_more_questions"
        assert data1["confidence_score"] == 65
        assert data1["next_question"] == "How long have you had this headache?"
        assert data1["is_conversation_complete"] is False
        assert data1["turn_count"] == 1
        
        # Step 2: Patient provides duration information
        followup_payload = {
            "conversation_id": conversation_id,
            "user_message": "It started about 6 hours ago and keeps getting worse",
            "stakeholder_role": "patient",
            "stakeholder_id": "patient_headache_001",
            "chat_provider": "api_direct"
        }
        
        response2 = await e2e_client.post("/api/v2/medical/chat", json=followup_payload)
        assert response2.status_code == 200
        
        data2 = response2.json()
        
        # Verify conversation completion
        assert data2["conversation_id"] == conversation_id
        assert data2["medical_outcome"] == "routine_doctor_consultation"
        assert data2["confidence_score"] == 78
        assert data2["is_conversation_complete"] is True
        assert data2["turn_count"] == 2
        assert data2["next_question"] is None
        
        # Step 3: Verify conversation state persistence
        state_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == 200
        
        state_data = state_response.json()
        assert state_data["turn_count"] == 2
        assert state_data["status"] == "completed"
        
        # Step 4: Verify conversation history
        history_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
        assert history_response.status_code == 200
        
        history_data = history_response.json()
        assert len(history_data["turns"]) == 2
        assert "headache" in history_data["turns"][0]["user_message"].lower()
        assert "6 hours" in history_data["turns"][1]["user_message"].lower()
    
    @pytest.mark.asyncio
    async def test_emergency_chest_pain_journey(self, e2e_client, mock_services):
        """Test emergency chest pain detection and immediate routing"""
        
        emergency_payload = {
            "user_message": "I have crushing chest pain and difficulty breathing",
            "stakeholder_role": "patient",
            "stakeholder_id": "emergency_patient_001", 
            "chat_provider": "api_direct"
        }
        
        response = await e2e_client.post("/api/v2/medical/chat", json=emergency_payload)
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify immediate emergency classification
        assert data["medical_outcome"] == "emergency_route_to_doctor"
        assert data["confidence_score"] == 95
        assert data["is_conversation_complete"] is True
        assert len(data["red_flags"]) > 0
        assert "chest_pain" in data["red_flags"]
        
        # Emergency should complete in single turn
        assert data["turn_count"] == 1
        assert data["next_question"] is None
    
    @pytest.mark.asyncio
    async def test_health_endpoints_during_journey(self, e2e_client):
        """Test system health endpoints work during patient journey"""
        
        # Test health endpoint
        health_response = await e2e_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # May be unhealthy in test env
        
        health_data = health_response.json()
        assert "status" in health_data
        assert "services" in health_data
        
        # Test API info endpoint
        info_response = await e2e_client.get("/api/v2/info")
        assert info_response.status_code == 200
        
        info_data = info_response.json()
        assert info_data["name"] == "Fairdoc AI Triage System V2"
        assert info_data["version"] == "v2.6-stable"
        assert "endpoints" in info_data
    
    @pytest.mark.asyncio 
    async def test_error_handling_in_journey(self, e2e_client):
        """Test error handling throughout patient journey"""
        
        # Test with invalid conversation ID
        invalid_response = await e2e_client.get("/api/v2/medical/chat/invalid-id/state")
        assert invalid_response.status_code == 404
        
        # Test with malformed request
        malformed_payload = {
            "user_message": "",  # Empty message
            "stakeholder_role": "invalid_role",
            "stakeholder_id": "test_patient"
        }
        
        error_response = await e2e_client.post("/api/v2/medical/chat", json=malformed_payload)
        assert error_response.status_code in [400, 422]  # Validation error
        
        # Test conversation reset
        # First create a valid conversation
        valid_payload = {
            "user_message": "Test symptoms",
            "stakeholder_role": "patient",
            "stakeholder_id": "test_patient"
        }
        
        create_response = await e2e_client.post("/api/v2/medical/chat", json=valid_payload)
        assert create_response.status_code == 200
        
        conversation_id = create_response.json()["conversation_id"]
        
        # Then reset it
        reset_response = await e2e_client.post(f"/api/v2/medical/chat/{conversation_id}/reset")
        assert reset_response.status_code == 200
        
        reset_data = reset_response.json()
        assert "reset successfully" in reset_data["message"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])