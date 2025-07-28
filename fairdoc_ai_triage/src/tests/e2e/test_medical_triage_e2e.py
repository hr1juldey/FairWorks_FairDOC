# End-to-End Medical Triage Workflow Test
# File: src/tests/e2e/test_medical_triage_e2e.py

"""
E2E test for complete medical triage workflow covering:
- Patient symptom input → DSPy analysis → Medical outcome
- NICE protocol integration → Redis state management  
- Background task processing → External integrations

Tests the critical path that a real patient would experience.
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi import status
import json
import time
from typing import Dict, Any
from unittest.mock import patch, AsyncMock

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async e2e tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def e2e_client():
    """HTTP client for end-to-end testing."""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def mock_external_services():
    """Mock external services for isolated e2e testing."""
    with patch('src.app2.services.chat.raven_bridge.httpx.AsyncClient.post') as mock_raven, \
         patch('src.app2.services.dspy.medical_agent.dspy.LM') as mock_dspy:
        
        # Mock Raven webhook responses
        mock_raven.return_value.status_code = 200
        mock_raven.return_value.json.return_value = {"status": "delivered"}
        
        # Mock DSPy responses will be handled per test
        yield {
            "raven": mock_raven,
            "dspy": mock_dspy
        }


class TestMedicalTriageE2E:
    """End-to-end medical triage workflow tests."""
    
    @pytest.mark.asyncio
    async def test_routine_headache_workflow(self, e2e_client, mock_external_services):
        """Test complete workflow for routine headache case."""
        
        # Step 1: Initial patient message
        initial_payload = {
            "user_message": "I have a mild headache that started this morning",
            "stakeholder_role": "patient",
            "stakeholder_id": "patient_routine_001",
            "chat_provider": "api_direct"
        }
        
        response1 = await e2e_client.post("/api/v2/medical/chat", json=initial_payload)
        assert response1.status_code == status.HTTP_200_OK
        
        data1 = response1.json()
        conversation_id = data1["conversation_id"]
        
        # Verify initial response structure
        assert data1["medical_outcome"] in ["need_more_questions", "routine_doctor_consultation"]
        assert data1["confidence_score"] >= 0
        assert data1["turn_count"] == 1
        assert data1["is_conversation_complete"] is False
        
        # Step 2: Follow-up response
        followup_payload = {
            "conversation_id": conversation_id,
            "user_message": "It's a dull ache, about 4/10 pain level",
            "stakeholder_role": "patient",
            "stakeholder_id": "patient_routine_001",
            "chat_provider": "api_direct"
        }
        
        response2 = await e2e_client.post("/api/v2/medical/chat", json=followup_payload)
        assert response2.status_code == status.HTTP_200_OK
        
        data2 = response2.json()
        assert data2["conversation_id"] == conversation_id
        assert data2["turn_count"] >= 2
        
        # Step 3: Verify conversation state persistence
        state_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == status.HTTP_200_OK
        
        state_data = state_response.json()
        assert state_data["conversation_id"] == conversation_id
        assert state_data["turn_count"] >= 2
        assert "conversation_history" in state_data
        
        # Step 4: Verify conversation history
        history_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
        assert history_response.status_code == status.HTTP_200_OK
        
        history_data = history_response.json()
        assert len(history_data["turns"]) >= 1
        assert history_data["status"] in ["active", "completed"]

    @pytest.mark.asyncio 
    async def test_emergency_chest_pain_workflow(self, e2e_client, mock_external_services):
        """Test complete emergency escalation workflow."""
        
        # Emergency symptoms should trigger immediate escalation
        emergency_payload = {
            "user_message": "I have severe crushing chest pain radiating to my left arm",
            "stakeholder_role": "patient", 
            "stakeholder_id": "patient_emergency_001",
            "chat_provider": "api_direct"
        }
        
        # Track time for response speed validation
        start_time = time.time()
        response = await e2e_client.post("/api/v2/medical/chat", json=emergency_payload)
        response_time = time.time() - start_time
        
        assert response.status_code == status.HTTP_200_OK
        assert response_time < 5.0  # Should respond quickly in emergencies
        
        data = response.json()
        conversation_id = data["conversation_id"]
        
        # Verify emergency detection
        assert data["medical_outcome"] == "emergency_route_to_doctor"
        assert data["confidence_score"] >= 80  # High confidence for clear emergency
        assert len(data["red_flags"]) > 0  # Should detect red flags
        assert data["is_conversation_complete"] is True  # Emergency = immediate completion
        
        # Verify emergency reasoning is present
        assert data["reasoning"] is not None
        assert len(data["reasoning"]) > 10
        
        # Verify background task execution (webhook calls)
        # Note: In real e2e, this would check actual webhook delivery
        mock_external_services["raven"].assert_called()
        
        # Verify conversation was marked as emergency in state
        state_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        state_data = state_response.json()
        assert state_data["current_outcome"] == "emergency"
        assert state_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_multi_turn_diagnostic_workflow(self, e2e_client, mock_external_services):
        """Test multi-turn conversation leading to diagnosis."""
        
        conversation_turns = [
            {
                "message": "I've been feeling unwell",
                "expected_outcome": "need_more_questions"
            },
            {
                "message": "I have fever and body aches",
                "expected_outcome": "need_more_questions"  
            },
            {
                "message": "The fever is 102F and I have chills",
                "expected_outcome": "routine_doctor_consultation"
            }
        ]
        
        conversation_id = None
        
        for i, turn in enumerate(conversation_turns):
            payload = {
                "user_message": turn["message"],
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_multiturn_001", 
                "chat_provider": "api_direct"
            }
            
            if conversation_id:
                payload["conversation_id"] = conversation_id
            
            response = await e2e_client.post("/api/v2/medical/chat", json=payload)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            
            if i == 0:
                conversation_id = data["conversation_id"]
            else:
                assert data["conversation_id"] == conversation_id
            
            # Verify turn progression
            assert data["turn_count"] == i + 1
            
            # Verify expected outcome or progression
            if i < len(conversation_turns) - 1:
                # Should continue asking questions
                assert data["is_conversation_complete"] is False
                assert data["next_question"] is not None
            else:
                # Final turn should have clear outcome
                assert data["medical_outcome"] in [
                    "routine_doctor_consultation", 
                    "self_care_advice",
                    "emergency_route_to_doctor"
                ]
        
        # Verify complete conversation history
        history_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
        history_data = history_response.json()
        
        assert len(history_data["turns"]) == len(conversation_turns)
        
        # Verify conversation progression in history
        for i, turn_data in enumerate(history_data["turns"]):
            assert conversation_turns[i]["message"] in turn_data.get("user_response", "")

    @pytest.mark.asyncio
    async def test_system_health_during_load(self, e2e_client):
        """Test system health endpoints under load."""
        
        # Test health endpoints are responsive
        health_response = await e2e_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # May be degraded but should respond
        
        health_data = health_response.json()
        assert "status" in health_data
        assert "services" in health_data
        assert "timestamp" in health_data
        
        # Test readiness probe
        ready_response = await e2e_client.get("/api/v2/health/ready")
        assert ready_response.status_code in [200, 503]
        
        # Test liveness probe
        live_response = await e2e_client.get("/api/v2/health/live")
        assert live_response.status_code == 200
        
        live_data = live_response.json()
        assert live_data["status"] == "alive"
        
        # Test API info endpoint
        info_response = await e2e_client.get("/api/v2/info")
        assert info_response.status_code == 200
        
        info_data = info_response.json()
        assert info_data["name"] == "Fairdoc AI Triage System V2"
        assert info_data["version"] == "v2.6-stable"
        assert "endpoints" in info_data

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, e2e_client):
        """Test system error handling and graceful degradation."""
        
        # Test invalid conversation ID
        invalid_id = "invalid-conversation-id"
        response = await e2e_client.get(f"/api/v2/medical/chat/{invalid_id}/state")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        error_data = response.json()
        assert "error" in error_data
        
        # Test malformed request payload
        malformed_payload = {
            "invalid_field": "invalid_value"
        }
        
        response = await e2e_client.post("/api/v2/medical/chat", json=malformed_payload) 
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test empty message handling
        empty_payload = {
            "user_message": "",
            "stakeholder_role": "patient",
            "stakeholder_id": "test_patient"
        }
        
        response = await e2e_client.post("/api/v2/medical/chat", json=empty_payload)
        # Should either reject (422) or handle gracefully (200 with error response)
        assert response.status_code in [200, 422]
        
        if response.status_code == 200:
            data = response.json()
            # Should indicate inability to process
            assert data["medical_outcome"] in ["need_more_questions", "spam_or_irrelevant"]

    @pytest.mark.asyncio
    async def test_conversation_reset_functionality(self, e2e_client):
        """Test conversation reset and cleanup."""
        
        # Start a conversation
        initial_payload = {
            "user_message": "I have symptoms",
            "stakeholder_role": "patient",
            "stakeholder_id": "reset_test_patient"
        }
        
        response = await e2e_client.post("/api/v2/medical/chat", json=initial_payload)
        data = response.json()
        conversation_id = data["conversation_id"]
        
        # Verify conversation exists
        state_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == 200
        
        # Reset conversation
        reset_response = await e2e_client.post(f"/api/v2/medical/chat/{conversation_id}/reset")
        assert reset_response.status_code == 200
        
        reset_data = reset_response.json()
        assert "reset successfully" in reset_data["message"].lower()
        
        # Verify conversation state is cleared (may return 404 or empty state)
        post_reset_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        # Implementation may vary - either 404 or empty state
        assert post_reset_response.status_code in [200, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])