"""
E2E Test: Complete Medical Triage Flow

Tests the entire V2 medical triage system from HTTP request to final response,
including all service orchestration, state management, and business logic.

File: src/tests/e2e/test_e2e_medical_triage_complete_flow.py
"""

import pytest
import asyncio
import os
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime
import uuid
import json

# Set test environment
os.environ.update({
    "ENVIRONMENT": "testing",
    "DATABASE_URL": "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_fairdoc_e2e",
    "REDIS_URL": "redis://localhost:6379/14",
    "FAIRDOC_V2_ENABLED": "true",
    "DEBUG": "true"
})

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2


class TestCompleteTriageFlow:
    """End-to-end test for complete medical triage workflow"""
    
    @pytest.fixture(autouse=True)
    async def setup_test_environment(self):
        """Setup test environment with mocked external dependencies"""
        
        # Mock Ollama/DSPy responses
        self.mock_dspy_responses = {
            "headache": {
                "outcome": "inconclusive",
                "confidence": 70,
                "next_question": "How severe is the pain on a scale of 1-10?",
                "reasoning": "Headache requires severity assessment",
                "red_flags": [],
                "is_complete": False
            },
            "severe_headache": {
                "outcome": "routine_doctor",
                "confidence": 85,
                "next_question": None,
                "reasoning": "Severe headache warrants medical evaluation",
                "red_flags": ["severe_pain"],
                "is_complete": True
            },
            "chest_pain": {
                "outcome": "emergency",
                "confidence": 95,
                "next_question": None,
                "reasoning": "Chest pain requires immediate medical attention",
                "red_flags": ["chest_pain", "cardiac_risk"],
                "is_complete": True
            }
        }
        
        # Mock medical agent
        async def mock_process_turn(symptoms, nice_context=""):
            symptoms_lower = symptoms.lower()
            if "chest pain" in symptoms_lower:
                return self.mock_dspy_responses["chest_pain"]
            elif "severe" in symptoms_lower and "headache" in symptoms_lower:
                return self.mock_dspy_responses["severe_headache"]
            elif "headache" in symptoms_lower:
                return self.mock_dspy_responses["headache"]
            else:
                return {
                    "outcome": "inconclusive",
                    "confidence": 50,
                    "next_question": "Can you describe your symptoms more specifically?",
                    "reasoning": "Need more specific symptom information",
                    "red_flags": [],
                    "is_complete": False
                }
        
        # Mock all heavy dependencies
        with patch('src.app2.core.dependencies_v2.init_redis_pool') as mock_redis_init, \
             patch('src.app2.core.dependencies_v2.init_services') as mock_services_init, \
             patch('src.app2.core.database_v2.startup_database') as mock_db_startup, \
             patch('src.app2.services.dspy.medical_agent.MedicalTriageAgent.process_turn', 
                   side_effect=mock_process_turn) as mock_agent:
            
            mock_redis_init.return_value = None
            mock_services_init.return_value = None
            mock_db_startup.return_value = None
            
            yield
    
    @pytest.mark.asyncio
    async def test_headache_to_routine_doctor_flow(self):
        """Test complete flow: Initial headache → Follow-up → Routine doctor recommendation"""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # STEP 1: Initial symptom report
            initial_payload = {
                "user_message": "I have a headache",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_e2e_001",
                "chat_provider": "api_direct"
            }
            
            response1 = await client.post("/api/v2/medical/chat", json=initial_payload)
            
            # Validate initial response
            assert response1.status_code == 200
            data1 = response1.json()
            
            # Extract conversation details
            conversation_id = data1["conversation_id"]
            assert conversation_id is not None
            assert conversation_id.startswith("conv_")
            
            # Validate triage decision
            assert data1["medical_outcome"] == "need_more_questions"
            assert data1["confidence_score"] == 70
            assert data1["next_question"] == "How severe is the pain on a scale of 1-10?"
            assert data1["is_conversation_complete"] is False
            assert data1["turn_count"] == 1
            assert len(data1["red_flags"]) == 0
            
            # STEP 2: Follow-up response
            followup_payload = {
                "conversation_id": conversation_id,
                "user_message": "It's severe, about an 8 out of 10",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_e2e_001",
                "chat_provider": "api_direct"
            }
            
            response2 = await client.post("/api/v2/medical/chat", json=followup_payload)
            
            # Validate follow-up response
            assert response2.status_code == 200
            data2 = response2.json()
            
            # Verify conversation continuity
            assert data2["conversation_id"] == conversation_id
            assert data2["turn_count"] == 2
            
            # Validate final triage decision
            assert data2["medical_outcome"] == "routine_doctor_consultation"
            assert data2["confidence_score"] == 85
            assert data2["is_conversation_complete"] is True
            assert "severe_pain" in data2["red_flags"]
            assert data2["next_question"] is None
            
            # STEP 3: Verify conversation state persistence
            state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            assert state_response.status_code == 200
            
            state_data = state_response.json()
            assert state_data["turn_count"] == 2
            assert state_data["status"] == "completed"
            assert state_data["current_outcome"] == "routine_doctor"
            
            # STEP 4: Verify conversation history
            history_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert len(history_data["turns"]) == 2
            assert history_data["turns"][0]["user_message"] == "I have a headache"
            assert history_data["turns"][1]["user_message"] == "It's severe, about an 8 out of 10"
    
    @pytest.mark.asyncio
    async def test_system_health_and_info_endpoints(self):
        """Test system health and API info endpoints work correctly"""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Test health endpoint
            health_response = await client.get("/api/v2/health")
            # May return 503 if services not fully initialized, which is acceptable
            assert health_response.status_code in [200, 503]
            
            health_data = health_response.json()
            assert "status" in health_data
            assert "version" in health_data
            assert "services" in health_data
            assert health_data["version"] == "v2.6-stable"
            
            # Test readiness endpoint
            ready_response = await client.get("/api/v2/health/ready")
            assert ready_response.status_code in [200, 503]
            
            # Test liveness endpoint
            live_response = await client.get("/api/v2/health/live")
            assert live_response.status_code == 200
            live_data = live_response.json()
            assert live_data["status"] == "alive"
            
            # Test API info endpoint
            info_response = await client.get("/api/v2/info")
            assert info_response.status_code == 200
            
            info_data = info_response.json()
            assert info_data["name"] == "Fairdoc AI Triage System V2"
            assert info_data["version"] == "v2.6-stable"
            assert "features" in info_data
            assert info_data["features"]["multi_turn_conversations"] is True
            assert info_data["features"]["emergency_detection"] is True
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_requests(self):
        """Test system handles invalid requests gracefully"""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Test missing required fields
            invalid_payload = {
                "stakeholder_role": "patient"
                # Missing user_message
            }
            
            response = await client.post("/api/v2/medical/chat", json=invalid_payload)
            assert response.status_code == 422  # Validation error
            
            # Test invalid conversation ID
            invalid_conv_response = await client.get(
                "/api/v2/medical/chat/invalid-uuid/state"
            )
            assert invalid_conv_response.status_code == 404
            
            # Test nonexistent conversation history
            fake_uuid = str(uuid.uuid4())
            fake_history_response = await client.get(
                f"/api/v2/medical/chat/{fake_uuid}/history"
            )
            assert fake_history_response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])