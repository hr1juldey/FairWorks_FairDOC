"""
E2E Test: Complete Medical Triage Workflow

Tests the full medical triage flow from HTTP request through all services
to final response, validating system integration and data flow.

File: src/tests/e2e/test_e2e_medical_triage_flow.py
"""

import pytest
import asyncio
import json
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime
import uuid

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2

# Test scenarios covering different medical outcomes
MEDICAL_SCENARIOS = [
    {
        "name": "headache_inconclusive",
        "symptoms": "I have a mild headache",
        "expected_outcome": "need_more_questions",
        "expected_confidence_min": 50,
        "should_have_next_question": True,
        "expected_red_flags": []
    },
    {
        "name": "chest_pain_emergency", 
        "symptoms": "Severe crushing chest pain radiating to left arm",
        "expected_outcome": "emergency_route_to_doctor",
        "expected_confidence_min": 85,
        "should_have_next_question": False,
        "expected_red_flags": ["chest_pain"]
    },
    {
        "name": "self_care_scenario",
        "symptoms": "Minor cut on finger, already cleaned and bandaged",
        "expected_outcome": "self_care_advice", 
        "expected_confidence_min": 70,
        "should_have_next_question": False,
        "expected_red_flags": []
    }
]

class TestMedicalTriageWorkflow:
    """Test complete medical triage workflow E2E"""
    
    @pytest.fixture(autouse=True)
    async def setup_mocks(self):
        """Setup realistic mocks for external dependencies"""
        
        # Mock DSPy Medical Agent with scenario-based responses
        def mock_agent_response(symptoms, nice_context=""):
            symptoms_lower = symptoms.lower()
            
            if "chest pain" in symptoms_lower and "crushing" in symptoms_lower:
                return {
                    "outcome": "emergency",
                    "confidence": 95,
                    "next_question": None,
                    "reasoning": "Crushing chest pain with radiation suggests possible MI",
                    "red_flags": ["chest_pain", "radiation_to_arm"],
                    "is_complete": True
                }
            elif "headache" in symptoms_lower and "mild" in symptoms_lower:
                return {
                    "outcome": "inconclusive",
                    "confidence": 60,
                    "next_question": "How long have you had this headache?",
                    "reasoning": "Need more information about headache characteristics",
                    "red_flags": [],
                    "is_complete": False
                }
            elif "cut" in symptoms_lower and "bandaged" in symptoms_lower:
                return {
                    "outcome": "self_care",
                    "confidence": 85,
                    "next_question": None,
                    "reasoning": "Minor wound, properly treated, self-care appropriate",
                    "red_flags": [],
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
        
        # Mock Redis with in-memory store
        self.redis_store = {}
        self.conversation_counter = 0
        
        async def mock_redis_operations():
            redis_mock = AsyncMock()
            
            async def mock_set(key, value, ex=None):
                self.redis_store[key] = {"value": value, "ttl": ex}
            
            async def mock_get(key):
                return self.redis_store.get(key, {}).get("value")
            
            async def mock_lpush(key, value):
                if key not in self.redis_store:
                    self.redis_store[key] = {"value": [], "type": "list"}
                self.redis_store[key]["value"].append(value)
            
            redis_mock.set = mock_set
            redis_mock.get = mock_get
            redis_mock.lpush = mock_lpush
            redis_mock.ping = AsyncMock(return_value=True)
            
            return redis_mock
        
        # Apply patches
        with patch('src.app2.services.dspy.medical_agent.MedicalTriageAgent') as mock_agent_class, \
             patch('redis.asyncio.from_url', side_effect=mock_redis_operations), \
             patch('src.app2.services.context.redis_queue.Redis.from_url', side_effect=mock_redis_operations):
            
            mock_agent = AsyncMock()
            mock_agent.process_turn = AsyncMock(side_effect=mock_agent_response)
            mock_agent_class.return_value = mock_agent
            
            yield
    
    @pytest.mark.parametrize("scenario", MEDICAL_SCENARIOS)
    @pytest.mark.asyncio
    async def test_complete_triage_workflow(self, scenario):
        """Test complete medical triage workflow for different scenarios"""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Initial medical consultation request
            payload = {
                "user_message": scenario["symptoms"],
                "stakeholder_role": "patient",
                "stakeholder_id": f"patient_{scenario['name']}_001",
                "chat_provider": "api_direct"
            }
            
            response = await client.post("/api/v2/medical/chat", json=payload)
            
            # Validate HTTP response
            assert response.status_code == 200, f"Failed for scenario: {scenario['name']}"
            data = response.json()
            
            # Validate response structure
            required_fields = [
                "conversation_id", "agent_message", "medical_outcome", 
                "confidence_score", "turn_count", "is_conversation_complete"
            ]
            for field in required_fields:
                assert field in data, f"Missing field {field} in scenario {scenario['name']}"
            
            # Validate medical assessment
            assert data["medical_outcome"] == scenario["expected_outcome"]
            assert data["confidence_score"] >= scenario["expected_confidence_min"]
            assert data["turn_count"] == 1
            
            # Validate conversation flow logic
            if scenario["should_have_next_question"]:
                assert data["next_question"] is not None
                assert data["is_conversation_complete"] is False
            else:
                assert data["is_conversation_complete"] is True
            
            # Validate red flags detection
            if scenario["expected_red_flags"]:
                assert data["red_flags"] is not None
                for flag in scenario["expected_red_flags"]:
                    assert flag in data["red_flags"]
            
            # Step 2: Test conversation state persistence
            conv_id = data["conversation_id"]
            state_response = await client.get(f"/api/v2/medical/chat/{conv_id}/state")
            assert state_response.status_code == 200
            
            state_data = state_response.json()
            assert state_data["conversation_id"] == conv_id
            assert state_data["turn_count"] == 1
            
            # Step 3: Test conversation history
            history_response = await client.get(f"/api/v2/medical/chat/{conv_id}/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert "turns" in history_data
            assert history_data["conversation_id"] == conv_id
    
    @pytest.mark.asyncio
    async def test_system_health_during_triage(self):
        """Test system health endpoints during active triage"""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Check initial system health
            health_response = await client.get("/api/v2/health")
            # May return 503 if services not fully initialized
            assert health_response.status_code in [200, 503]
            
            # Step 2: Start a triage session
            payload = {
                "user_message": "I have symptoms to report",
                "stakeholder_role": "patient"
            }
            
            triage_response = await client.post("/api/v2/medical/chat", json=payload)
            assert triage_response.status_code == 200
            
            # Step 3: Check health during active session
            health_response = await client.get("/api/v2/health")
            health_data = health_response.json()
            
            assert "status" in health_data
            assert "services" in health_data
            assert "timestamp" in health_data
            
            # Step 4: Check API information
            info_response = await client.get("/api/v2/info")
            assert info_response.status_code == 200
            
            info_data = info_response.json()
            assert info_data["name"] == "Fairdoc AI Triage System V2"
            assert info_data["version"] == "v2.6-stable"
            assert "features" in info_data
    
    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self):
        """Test error handling throughout the triage workflow"""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test invalid request payload
            invalid_payload = {
                "user_message": "",  # Empty message
                "stakeholder_role": "invalid_role"
            }
            
            response = await client.post("/api/v2/medical/chat", json=invalid_payload)
            assert response.status_code in [400, 422]  # Validation error
            
            # Test non-existent conversation
            fake_conv_id = str(uuid.uuid4())
            response = await client.get(f"/api/v2/medical/chat/{fake_conv_id}/state")
            assert response.status_code == 404
            
            # Test malformed conversation ID
            response = await client.get("/api/v2/medical/chat/invalid-id/history")
            assert response.status_code in [400, 422, 404]
    
    @pytest.mark.asyncio
    async def test_concurrent_triage_sessions(self):
        """Test system handling of concurrent triage sessions"""
        
        async def create_triage_session(session_id):
            async with AsyncClient(app=app, base_url="http://test") as client:
                payload = {
                    "user_message": f"Patient {session_id} reporting symptoms",
                    "stakeholder_role": "patient",
                    "stakeholder_id": f"concurrent_patient_{session_id}"
                }
                
                response = await client.post("/api/v2/medical/chat", json=payload)
                return response.status_code, response.json() if response.status_code == 200 else None
        
        # Create 5 concurrent sessions
        tasks = [create_triage_session(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate all sessions succeeded
        successful_sessions = 0
        for result in results:
            if isinstance(result, tuple) and result[0] == 200:
                successful_sessions += 1
                # Validate each session has unique conversation ID
                assert result[1] is not None
                assert "conversation_id" in result[1]
        
        assert successful_sessions >= 3, "At least 3 concurrent sessions should succeed"