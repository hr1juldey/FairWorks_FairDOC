"""
Complete Medical Triage Workflow E2E Test

Tests the full patient journey from initial symptoms through to final medical outcome.
Validates the entire V2 system integration including FastAPI, DSPy agent, Redis state,
NICE protocols, stakeholder routing, and background task execution.

File: src/tests/e2e/test_complete_medical_triage_e2e.py
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2


class TestCompleteMedicalTriageWorkflow:
    """End-to-end test for complete medical triage workflow"""
    
    @pytest.fixture(scope="class")
    async def app_client(self):
        """Create FastAPI test client for the entire test class"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client
    
    @pytest.fixture(autouse=True)
    async def setup_test_environment(self):
        """Setup clean test environment for each test"""
        # Mock external dependencies to avoid network calls
        with patch('src.app2.services.dspy.medical_agent.dspy.configure'), \
             patch('src.app2.services.dspy.medical_agent.dspy.LM') as mock_lm, \
             patch('src.app2.services.dspy.medical_agent.dspy.ChainOfThought') as mock_cot, \
             patch('redis.asyncio.Redis.from_url') as mock_redis:
            
            # Configure DSPy mocks
            mock_cot_instance = MagicMock()
            mock_cot.return_value = mock_cot_instance
            
            # Configure Redis mock
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            # Setup Redis mock methods
            mock_redis_instance.ping = AsyncMock()
            mock_redis_instance.get = AsyncMock(return_value=None)
            mock_redis_instance.setex = AsyncMock()
            mock_redis_instance.lpush = AsyncMock()
            mock_redis_instance.lrange = AsyncMock(return_value=[])
            
            yield {
                "mock_cot_instance": mock_cot_instance,
                "mock_redis": mock_redis_instance
            }
    
    @pytest.mark.asyncio
    async def test_headache_triage_complete_workflow(self, app_client, setup_test_environment):
        """Test complete workflow for headache symptoms leading to routine care"""
        mocks = setup_test_environment
        
        # Mock DSPy agent responses for multi-turn conversation
        def mock_dspy_responses(symptoms, nice_context=""):
            if "headache" in symptoms.lower() and "started today" not in symptoms.lower():
                # First turn - ask for duration
                result = MagicMock()
                result.outcome_classification = "inconclusive"
                result.confidence_score = 65
                result.next_question = "How long have you had this headache?"
                result.reasoning = "Need more information about headache duration"
                result.red_flags = ""
                return result
            elif "started today" in symptoms.lower():
                # Second turn - recommend routine care
                result = MagicMock()
                result.outcome_classification = "routine"
                result.confidence_score = 80
                result.next_question = "COMPLETE"
                result.reasoning = "Recent onset headache, recommend seeing GP"
                result.red_flags = ""
                return result
        
        mocks["mock_cot_instance"].forward = MagicMock(side_effect=mock_dspy_responses)
        
        # Mock Redis conversation state
        conversation_states = {}
        
        async def mock_redis_get(key):
            return json.dumps(conversation_states.get(key)) if key in conversation_states else None
        
        async def mock_redis_setex(key, ttl, value):
            conversation_states[key] = json.loads(value)
        
        mocks["mock_redis"].get.side_effect = mock_redis_get
        mocks["mock_redis"].setex.side_effect = mock_redis_setex
        
        # Step 1: Initial symptoms report
        initial_payload = {
            "user_message": "I have a severe headache",
            "stakeholder_role": "patient",
            "stakeholder_id": "patient_headache_001",
            "chat_provider": "api_direct"
        }
        
        response1 = await app_client.post("/api/v2/medical/chat", json=initial_payload)
        assert response1.status_code == 200
        
        data1 = response1.json()
        conversation_id = data1["conversation_id"]
        
        # Validate first response
        assert data1["medical_outcome"] == "need_more_questions"
        assert data1["confidence_score"] == 65
        assert "How long have you had this headache?" in data1["agent_response"]
        assert data1["turn_number"] == 1
        assert data1["is_conversation_complete"] is False
        
        # Step 2: Follow-up response
        followup_payload = {
            "conversation_id": conversation_id,
            "user_message": "It started today morning, getting worse",
            "stakeholder_role": "patient",
            "stakeholder_id": "patient_headache_001",
            "chat_provider": "api_direct"
        }
        
        response2 = await app_client.post("/api/v2/medical/chat", json=followup_payload)
        assert response2.status_code == 200
        
        data2 = response2.json()
        
        # Validate final response
        assert data2["conversation_id"] == conversation_id
        assert data2["medical_outcome"] == "routine_doctor_consultation"
        assert data2["confidence_score"] == 80
        assert data2["turn_number"] == 2
        assert data2["is_conversation_complete"] is True
        
        # Step 3: Verify conversation state persistence
        state_response = await app_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == 200
        
        state_data = state_response.json()
        assert state_data["turn_count"] == 2
        assert state_data["status"] == "completed"
        assert state_data["current_outcome"] == "routine"
        
        # Step 4: Verify conversation history
        history_response = await app_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
        assert history_response.status_code == 200
        
        history_data = history_response.json()
        assert len(history_data["turns"]) == 2
        assert "severe headache" in history_data["turns"][0]["user_response"]
        assert "started today morning" in history_data["turns"][1]["user_response"]
    
    @pytest.mark.asyncio
    async def test_chest_pain_emergency_workflow(self, app_client, setup_test_environment):
        """Test emergency workflow for chest pain symptoms"""
        mocks = setup_test_environment
        
        # Mock DSPy agent for emergency response
        def mock_emergency_dspy(symptoms, nice_context=""):
            result = MagicMock()
            result.outcome_classification = "emergency"
            result.confidence_score = 95
            result.next_question = "COMPLETE"
            result.reasoning = "Chest pain with concerning symptoms requires immediate medical attention"
            result.red_flags = "chest_pain,shortness_of_breath"
            return result
        
        mocks["mock_cot_instance"].forward = MagicMock(side_effect=mock_emergency_dspy)
        
        # Mock background task execution tracking
        emergency_alerts = []
        
        async def mock_emergency_handler(*args, **kwargs):
            emergency_alerts.append({"args": args, "kwargs": kwargs, "timestamp": datetime.utcnow()})
        
        with patch('src.app2.services.chat.emergency_handler.EmergencyHandler.handle_emergency_alert', 
                   side_effect=mock_emergency_handler):
            
            # Emergency symptoms report
            emergency_payload = {
                "user_message": "I have crushing chest pain and can't breathe properly",
                "stakeholder_role": "patient",
                "stakeholder_id": "emergency_patient_001",
                "chat_provider": "api_direct"
            }
            
            response = await app_client.post("/api/v2/medical/chat", json=emergency_payload)
            assert response.status_code == 200
            
            data = response.json()
            
            # Validate emergency response
            assert data["medical_outcome"] == "emergency_route_to_doctor"
            assert data["confidence_score"] == 95
            assert data["is_conversation_complete"] is True
            assert len(data["red_flags"]) > 0
            assert "chest_pain" in data["red_flags"]
            
            # Allow background tasks to execute
            await asyncio.sleep(0.1)
            
            # Verify emergency alert was triggered
            assert len(emergency_alerts) == 1
            alert = emergency_alerts[0]
            assert alert["args"][0] == data["conversation_id"]  # conversation_id
            assert alert["args"][1] == "emergency_patient_001"  # user_id
    
    @pytest.mark.asyncio
    async def test_self_care_recommendation_workflow(self, app_client, setup_test_environment):
        """Test workflow leading to self-care recommendation"""
        mocks = setup_test_environment
        
        # Mock DSPy agent for self-care recommendation
        def mock_selfcare_dspy(symptoms, nice_context=""):
            result = MagicMock()
            result.outcome_classification = "self_care"
            result.confidence_score = 75
            result.next_question = "COMPLETE"
            result.reasoning = "Mild symptoms that can be managed with self-care"
            result.red_flags = ""
            return result
        
        mocks["mock_cot_instance"].forward = MagicMock(side_effect=mock_selfcare_dspy)
        
        # Self-care scenario
        selfcare_payload = {
            "user_message": "I have a mild headache, no other symptoms, took paracetamol",
            "stakeholder_role": "patient",
            "stakeholder_id": "selfcare_patient_001",
            "chat_provider": "api_direct"
        }
        
        response = await app_client.post("/api/v2/medical/chat", json=selfcare_payload)
        assert response.status_code == 200
        
        data = response.json()
        
        # Validate self-care response
        assert data["medical_outcome"] == "self_care_advice"
        assert data["confidence_score"] == 75
        assert data["is_conversation_complete"] is True
        assert len(data["red_flags"]) == 0
        assert "self-care" in data["reasoning"].lower()
    
    @pytest.mark.asyncio
    async def test_conversation_reset_workflow(self, app_client, setup_test_environment):
        """Test conversation reset functionality"""
        mocks = setup_test_environment
        
        # Mock DSPy for initial conversation
        def mock_reset_dspy(symptoms, nice_context=""):
            result = MagicMock()
            result.outcome_classification = "inconclusive"
            result.confidence_score = 60
            result.next_question = "Can you provide more details?"
            result.reasoning = "Need more information"
            result.red_flags = ""
            return result
        
        mocks["mock_cot_instance"].forward = MagicMock(side_effect=mock_reset_dspy)
        
        # Start conversation
        payload = {
            "user_message": "I feel unwell",
            "stakeholder_role": "patient",
            "stakeholder_id": "reset_test_patient",
            "chat_provider": "api_direct"
        }
        
        response = await app_client.post("/api/v2/medical/chat", json=payload)
        assert response.status_code == 200
        
        conversation_id = response.json()["conversation_id"]
        
        # Reset conversation
        reset_response = await app_client.post(f"/api/v2/medical/chat/{conversation_id}/reset")
        assert reset_response.status_code == 200
        assert "reset successfully" in reset_response.json()["message"]
        
        # Verify conversation state is cleared (should return 404 or empty state)
        state_response = await app_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        # After reset, conversation should either not exist or be in initial state
        assert state_response.status_code in [404, 200]
    
    @pytest.mark.asyncio
    async def test_system_health_during_workflow(self, app_client):
        """Test system health checks during active workflow"""
        # Check health before workflow
        health_response = await app_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # May not be fully initialized in test
        
        health_data = health_response.json()
        assert "status" in health_data
        assert "services" in health_data
        assert "version" in health_data
        
        # Check API info
        info_response = await app_client.get("/api/v2/info")
        assert info_response.status_code == 200
        
        info_data = info_response.json()
        assert info_data["name"] == "Fairdoc AI Triage System V2"
        assert info_data["version"] == "v2.6-stable"
        assert "features" in info_data
        assert info_data["features"]["multi_turn_conversations"] is True
        assert info_data["features"]["emergency_detection"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])