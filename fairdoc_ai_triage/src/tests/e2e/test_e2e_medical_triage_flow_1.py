"""
E2E Test: Complete Medical Triage Flow
=====================================

Tests the full patient conversation journey from initial symptoms
through to final medical outcome, including all service interactions.

File: src/tests/e2e/test_e2e_medical_triage_flow.py
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from httpx import AsyncClient
from fastapi import status

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.models.schemas.multiturn_chat import (
    StakeholderRole,
    ChatProvider,
    MedicalOutcome
)

@pytest.mark.asyncio
@pytest.mark.e2e
class TestCompleteTriageFlow:
    """End-to-end medical triage conversation flow"""
    
    async def test_headache_triage_complete_flow(self):
        """Test complete headache triage from symptoms to final outcome"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            patient_id = f"e2e_patient_{uuid.uuid4()}"
            conversation_id = None
            
            # Turn 1: Initial headache symptoms
            initial_payload = {
                "user_message": "I have a severe headache that started suddenly",
                "stakeholder_role": "patient",
                "stakeholder_id": patient_id,
                "chat_provider": "api_direct"
            }
            
            response = await client.post("/api/v2/medical/chat", json=initial_payload)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            conversation_id = data["conversation_id"]
            
            # Validate initial response structure
            assert "agent_response" in data
            assert "medical_outcome" in data
            assert "confidence_score" in data
            assert "turn_number" in data
            assert data["turn_number"] == 1
            
            # Should ask follow-up questions for headache
            if data["medical_outcome"] == "need_more_questions":
                assert data["next_question"] is not None
                assert data["is_conversation_complete"] is False
                
                # Turn 2: Patient provides duration info
                followup_payload = {
                    "conversation_id": conversation_id,
                    "user_message": "It started about 2 hours ago, very sudden onset",
                    "stakeholder_role": "patient",
                    "stakeholder_id": patient_id,
                    "chat_provider": "api_direct"
                }
                
                response2 = await client.post("/api/v2/medical/chat", json=followup_payload)
                assert response2.status_code == 200
                
                data2 = response2.json()
                assert data2["conversation_id"] == conversation_id
                assert data2["turn_number"] == 2
                
                # Should continue asking or provide outcome
                if not data2["is_conversation_complete"]:
                    # Turn 3: Additional symptoms
                    final_payload = {
                        "conversation_id": conversation_id,
                        "user_message": "Yes, and I have neck stiffness and light sensitivity",
                        "stakeholder_role": "patient",
                        "stakeholder_id": patient_id,
                        "chat_provider": "api_direct"
                    }
                    
                    response3 = await client.post("/api/v2/medical/chat", json=final_payload)
                    assert response3.status_code == 200
                    
                    data3 = response3.json()
                    # These symptoms should trigger emergency or doctor consultation
                    assert data3["medical_outcome"] in ["emergency", "routine_doctor_consultation"]
                    
                    if data3["medical_outcome"] == "emergency":
                        assert len(data3["red_flags"]) > 0
                        assert "neck_stiffness" in data3.get("red_flags", []) or \
                               "photophobia" in data3.get("red_flags", [])
            
            # Validate conversation state endpoint
            state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            assert state_response.status_code == 200
            
            state_data = state_response.json()
            assert state_data["conversation_id"] == conversation_id
            assert state_data["turn_count"] >= 1
            assert state_data["user_id"] == patient_id
            
            # Validate conversation history endpoint
            history_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert len(history_data["turns"]) >= 1
            assert history_data["conversation_id"] == conversation_id

    async def test_chest_pain_emergency_flow(self):
        """Test chest pain symptoms triggering immediate emergency response"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            emergency_payload = {
                "user_message": "I have crushing chest pain radiating to my left arm and I'm sweating profusely",
                "stakeholder_role": "patient",
                "stakeholder_id": f"emergency_patient_{uuid.uuid4()}",
                "chat_provider": "api_direct"
            }
            
            response = await client.post("/api/v2/medical/chat", json=emergency_payload)
            assert response.status_code == 200
            
            data = response.json()
            
            # Emergency symptoms should be detected immediately
            assert data["medical_outcome"] == "emergency"
            assert data["confidence_score"] >= 80  # High confidence for clear emergency
            assert data["is_conversation_complete"] is True  # No more questions needed
            
            # Should have red flags
            red_flags = data.get("red_flags", [])
            assert len(red_flags) > 0
            assert any("chest" in flag.lower() for flag in red_flags)
            
            # Should have clear reasoning
            assert "emergency" in data["reasoning"].lower() or "urgent" in data["reasoning"].lower()

    async def test_self_care_advice_flow(self):
        """Test minor symptoms leading to self-care advice"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            minor_payload = {
                "user_message": "I have a mild headache and I think it's from dehydration",
                "stakeholder_role": "patient",
                "stakeholder_id": f"selfcare_patient_{uuid.uuid4()}",
                "chat_provider": "api_direct"
            }
            
            response = await client.post("/api/v2/medical/chat", json=minor_payload)
            assert response.status_code == 200
            
            data = response.json()
            
            # May initially be inconclusive, but follow-up should lead to self-care
            if data["medical_outcome"] == "need_more_questions":
                followup_payload = {
                    "conversation_id": data["conversation_id"],
                    "user_message": "It's very mild, started after I didn't drink water for several hours",
                    "stakeholder_role": "patient",
                    "stakeholder_id": minor_payload["stakeholder_id"],
                    "chat_provider": "api_direct"
                }
                
                response2 = await client.post("/api/v2/medical/chat", json=followup_payload)
                final_data = response2.json()
                
                # Should result in self-care advice
                assert final_data["medical_outcome"] in ["self_care_advice", "need_more_questions"]
                
                if final_data["medical_outcome"] == "self_care_advice":
                    assert "hydrat" in final_data["agent_response"].lower() or \
                           "water" in final_data["agent_response"].lower() or \
                           "rest" in final_data["agent_response"].lower()

    async def test_conversation_reset_flow(self):
        """Test conversation reset functionality works end-to-end"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Start a conversation
            payload = {
                "user_message": "I have symptoms to discuss",
                "stakeholder_role": "patient",
                "stakeholder_id": f"reset_test_{uuid.uuid4()}",
                "chat_provider": "api_direct"
            }
            
            response = await client.post("/api/v2/medical/chat", json=payload)
            conversation_id = response.json()["conversation_id"]
            
            # Reset the conversation
            reset_response = await client.post(f"/api/v2/medical/chat/{conversation_id}/reset")
            assert reset_response.status_code == 200
            
            reset_data = reset_response.json()
            assert "reset successfully" in reset_data["message"].lower()
            
            # Verify conversation state after reset
            # Note: Depending on implementation, this might return 404 or empty state
            state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            # Either conversation is gone (404) or reset to initial state
            assert state_response.status_code in [200, 404]

    async def test_health_and_info_endpoints(self):
        """Test system health and info endpoints work end-to-end"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Test health endpoint
            health_response = await client.get("/api/v2/health")
            assert health_response.status_code in [200, 503]  # May be degraded in test env
            
            health_data = health_response.json()
            assert "status" in health_data
            assert "services" in health_data
            assert "version" in health_data
            
            # Test info endpoint
            info_response = await client.get("/api/v2/info")
            assert info_response.status_code == 200
            
            info_data = info_response.json()
            assert info_data["name"] == "Fairdoc AI Triage System V2"
            assert info_data["version"] == "v2.6-stable"
            assert "endpoints" in info_data
            assert "/api/v2/medical/chat" in str(info_data["endpoints"])

    async def test_error_handling_flow(self):
        """Test error handling works correctly end-to-end"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Test invalid conversation ID
            bad_conv_id = str(uuid.uuid4())
            
            invalid_response = await client.get(f"/api/v2/medical/chat/{bad_conv_id}/state")
            assert invalid_response.status_code == 404
            
            # Test malformed request
            malformed_payload = {
                "user_message": "",  # Empty message
                "stakeholder_role": "invalid_role",
                "stakeholder_id": "test_user"
            }
            
            error_response = await client.post("/api/v2/medical/chat", json=malformed_payload)
            # Should return validation error
            assert error_response.status_code in [400, 422]