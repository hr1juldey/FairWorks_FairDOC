"""
End-to-End Test: Complete Medical Triage Flow

Tests the entire V2 system from HTTP request through DSPy agent to response.
Uses real FastAPI app with real services but controlled test data.

File: src/tests/e2e/test_complete_medical_flow_e2e.py
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock

# V2 Application and core components
from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider

# Test configuration
TEST_USER_ID = "e2e_patient_001"
TEST_CONVERSATION_SCENARIOS = [
    {
        "name": "headache_to_routine_doctor",
        "messages": [
            "I have a headache that started this morning",
            "It's getting worse and I feel nauseous", 
            "Yes, it's throbbing and I can't concentrate"
        ],
        "expected_final_outcome": "routine_doctor"
    },
    {
        "name": "chest_pain_emergency",
        "messages": [
            "I have severe crushing chest pain radiating to my left arm"
        ],
        "expected_final_outcome": "emergency"
    }
]

@pytest.fixture
async def e2e_client():
    """Real FastAPI client for E2E testing"""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

@pytest.fixture
def mock_external_services(monkeypatch):
    """Mock external services but keep core V2 logic intact"""
    
    # Mock Raven bridge to avoid external HTTP calls
    async def mock_send_message(self, message, conversation_id):
        return {"status": "sent", "message_id": f"raven_{conversation_id}"}
    
    # Mock emergency webhook
    async def mock_emergency_alert(self, conv_id, user_id, agent_result):
        print(f"🚨 MOCK EMERGENCY ALERT: {conv_id} - {agent_result.get('red_flags', [])}")
        return {"alert_sent": True}
    
    # Mock PostgreSQL persistence 
    async def mock_save_to_postgres(self, conv_id, state):
        print(f"💾 MOCK POSTGRES SAVE: {conv_id} - Status: {state.get('status', 'unknown')}")
        return {"saved": True}
    
    from src.app2.services.chat import raven_bridge, emergency_handler, persistence_handler
    
    monkeypatch.setattr(raven_bridge.RavenBridge, "send_message", mock_send_message)
    monkeypatch.setattr(emergency_handler.EmergencyHandler, "handle_emergency_alert", mock_emergency_alert)
    monkeypatch.setattr(persistence_handler.PersistenceHandler, "save_conversation_to_postgresql", mock_save_to_postgres)

class TestCompleteMedicalTriageFlow:
    """E2E test for complete medical triage conversation flows"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_headache_to_routine_doctor_flow(self, e2e_client, mock_external_services):
        """Test complete headache conversation leading to routine doctor recommendation"""
        
        conversation_id = None
        conversation_log = []
        
        scenario = TEST_CONVERSATION_SCENARIOS[0]  # headache scenario
        
        for turn_num, message in enumerate(scenario["messages"], 1):
            # Prepare request payload
            payload = {
                "user_message": message,
                "stakeholder_role": StakeholderRole.PATIENT,
                "stakeholder_id": TEST_USER_ID,
                "chat_provider": ChatProvider.API_DIRECT
            }
            
            # Include conversation_id for continuation turns
            if conversation_id:
                payload["conversation_id"] = conversation_id
            
            # Make API request
            response = await e2e_client.post("/api/v2/medical/chat", json=payload)
            
            # Verify response structure
            assert response.status_code == 200, f"Turn {turn_num} failed: {response.text}"
            
            data = response.json()
            
            # Capture conversation_id from first turn
            if turn_num == 1:
                conversation_id = data["conversation_id"]
                assert conversation_id.startswith("conv_"), "Invalid conversation ID format"
            
            # Verify response schema
            required_fields = [
                "conversation_id", "agent_message", "medical_outcome", 
                "confidence_score", "turn_count", "is_conversation_complete"
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Log conversation turn for debugging
            turn_log = {
                "turn": turn_num,
                "user_message": message,
                "agent_response": data.get("agent_message", ""),
                "medical_outcome": data["medical_outcome"],
                "confidence": data["confidence_score"],
                "complete": data["is_conversation_complete"],
                "red_flags": data.get("red_flags", [])
            }
            conversation_log.append(turn_log)
            
            # Verify turn count increments correctly
            assert data["turn_count"] == turn_num, f"Turn count mismatch: expected {turn_num}, got {data['turn_count']}"
            
            # Check if conversation completed
            if data["is_conversation_complete"]:
                break
        
        # Verify final outcome matches scenario expectation
        final_turn = conversation_log[-1]
        assert final_turn["medical_outcome"] == scenario["expected_final_outcome"], \
            f"Expected {scenario['expected_final_outcome']}, got {final_turn['medical_outcome']}"
        
        # Verify conversation state via state endpoint
        state_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == 200
        
        state_data = state_response.json()
        assert state_data["conversation_id"] == conversation_id
        assert state_data["user_id"] == TEST_USER_ID
        
        # Verify conversation history endpoint
        history_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
        assert history_response.status_code == 200
        
        history_data = history_response.json()
        assert len(history_data["turns"]) >= len(scenario["messages"])
        
        # Print detailed conversation log for debugging
        print(f"\n=== COMPLETE CONVERSATION LOG: {scenario['name']} ===")
        for turn in conversation_log:
            print(f"Turn {turn['turn']}: {turn['user_message']}")
            print(f"  → Agent: {turn['agent_response']}")
            print(f"  → Outcome: {turn['medical_outcome']} (confidence: {turn['confidence']}%)")
            print(f"  → Complete: {turn['complete']}, Red flags: {turn['red_flags']}")
        print("=" * 60)
    
    @pytest.mark.asyncio 
    @pytest.mark.e2e
    async def test_emergency_chest_pain_immediate_response(self, e2e_client, mock_external_services):
        """Test emergency chest pain triggers immediate emergency response"""
        
        scenario = TEST_CONVERSATION_SCENARIOS[1]  # chest pain scenario
        
        payload = {
            "user_message": scenario["messages"][0],
            "stakeholder_role": StakeholderRole.PATIENT,
            "stakeholder_id": f"{TEST_USER_ID}_emergency",
            "chat_provider": ChatProvider.API_DIRECT
        }
        
        response = await e2e_client.post("/api/v2/medical/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Emergency should be detected immediately
        assert data["medical_outcome"] == "emergency", f"Expected emergency, got {data['medical_outcome']}"
        assert data["is_conversation_complete"] is True, "Emergency conversation should complete immediately"
        assert data["confidence_score"] >= 85, f"Emergency confidence too low: {data['confidence_score']}"
        
        # Should have red flags
        assert len(data.get("red_flags", [])) > 0, "Emergency scenario should have red flags"
        
        # Verify emergency escalation via health check
        health_response = await e2e_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # Service may show degraded during emergency
        
        print(f"\n=== EMERGENCY RESPONSE LOG ===")
        print(f"Input: {scenario['messages'][0]}")
        print(f"Outcome: {data['medical_outcome']}")
        print(f"Confidence: {data['confidence_score']}%")
        print(f"Red flags: {data.get('red_flags', [])}")
        print(f"Agent message: {data.get('agent_message', 'N/A')}")
        print("=" * 40)
    
    @pytest.mark.asyncio
    @pytest.mark.e2e 
    async def test_conversation_reset_functionality(self, e2e_client, mock_external_services):
        """Test conversation reset and cleanup"""
        
        # Start a conversation
        payload = {
            "user_message": "I have a mild headache",
            "stakeholder_role": StakeholderRole.PATIENT,
            "stakeholder_id": f"{TEST_USER_ID}_reset_test"
        }
        
        response = await e2e_client.post("/api/v2/medical/chat", json=payload)
        assert response.status_code == 200
        
        conversation_id = response.json()["conversation_id"]
        
        # Verify conversation exists
        state_response = await e2e_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == 200
        
        # Reset conversation
        reset_response = await e2e_client.post(f"/api/v2/medical/chat/{conversation_id}/reset")
        assert reset_response.status_code == 200
        
        reset_data = reset_response.json()
        assert "reset successfully" in reset_data["message"].lower()
        
        print(f"\n=== CONVERSATION RESET TEST ===")
        print(f"Conversation ID: {conversation_id}")
        print(f"Reset result: {reset_data['message']}")
        print("=" * 40)
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_api_health_and_info_endpoints(self, e2e_client):
        """Test system health and API information endpoints"""
        
        # Test health endpoint
        health_response = await e2e_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]
        
        health_data = health_response.json()
        assert "status" in health_data
        assert "services" in health_data
        assert "version" in health_data
        
        # Test API info endpoint  
        info_response = await e2e_client.get("/api/v2/info")
        assert info_response.status_code == 200
        
        info_data = info_response.json()
        assert info_data["name"] == "Fairdoc AI Triage System V2"
        assert info_data["version"] == "v2.6-stable"
        assert "features" in info_data
        assert "endpoints" in info_data
        
        print(f"\n=== SYSTEM STATUS ===")
        print(f"Health: {health_data.get('status', 'unknown')}")
        print(f"Version: {info_data['version']}")
        print(f"Features: {list(info_data['features'].keys())}")
        print("=" * 30)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "e2e"])