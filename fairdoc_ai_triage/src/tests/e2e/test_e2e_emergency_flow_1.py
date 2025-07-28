# E2E Test: Emergency Detection and Escalation Flow
# File: src/tests/e2e/test_e2e_emergency_flow.py
"""
End-to-End test for emergency medical triage scenarios.

Tests complete flow from HTTP request through DSPy agent, Redis state management,
background task execution, and external webhook calls.

Provides comprehensive debugging output for system validation.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient
from fastapi import status

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider

# Test configuration
EMERGENCY_SYMPTOMS = [
    "crushing chest pain radiating to left arm",
    "severe chest pain with sweating and nausea", 
    "sudden severe headache worst ever experienced",
    "difficulty breathing with chest tightness"
]

@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
async def test_client():
    """Create test client with real app instance"""
    async with AsyncClient(app=app, base_url="http://localhost:8000") as client:
        yield client

@pytest.fixture
def mock_webhook_calls():
    """Mock external webhook calls to capture emergency alerts"""
    webhook_calls = []
    
    async def capture_webhook(url, **kwargs):
        webhook_calls.append({
            "url": url,
            "method": kwargs.get("method", "POST"),
            "data": kwargs.get("json", {}),
            "timestamp": time.time()
        })
        return AsyncMock(status_code=200)
    
    with patch('httpx.AsyncClient.post', side_effect=capture_webhook):
        yield webhook_calls

class TestEmergencyDetectionFlow:
    """Test emergency detection and escalation workflows"""
    
    @pytest.mark.asyncio
    async def test_chest_pain_emergency_detection(self, test_client, mock_webhook_calls):
        """Test chest pain triggers immediate emergency response"""
        print("\n🚨 TESTING: Chest Pain Emergency Detection")
        
        # Step 1: Send emergency symptoms
        payload = {
            "user_message": EMERGENCY_SYMPTOMS[0],
            "stakeholder_role": StakeholderRole.PATIENT.value,
            "stakeholder_id": "emergency_patient_001",
            "chat_provider": ChatProvider.API_DIRECT.value
        }
        
        print(f"📤 Sending emergency request: {payload['user_message']}")
        start_time = time.time()
        
        response = await test_client.post("/api/v2/medical/chat", json=payload)
        response_time = time.time() - start_time
        
        print(f"⏱️ Response time: {response_time:.3f}s")
        
        # Step 2: Validate immediate response
        assert response.status_code == status.HTTP_200_OK, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        print(f"📥 Response data: {json.dumps(data, indent=2)}")
        
        # Emergency validation
        assert data["medical_outcome"] == "emergency_route_to_doctor", f"Expected emergency, got {data['medical_outcome']}"
        assert data["confidence_score"] >= 85, f"Low confidence for emergency: {data['confidence_score']}"
        assert data["is_conversation_complete"] is True, "Emergency should complete conversation"
        assert len(data["red_flags"]) > 0, "Emergency should have red flags"
        
        print(f"✅ Emergency detected with {data['confidence_score']}% confidence")
        print(f"🚩 Red flags: {data['red_flags']}")
        
        # Step 3: Verify conversation state
        conv_id = data["conversation_id"]
        state_response = await test_client.get(f"/api/v2/medical/chat/{conv_id}/state")
        
        assert state_response.status_code == 200
        state_data = state_response.json()
        
        print(f"💾 Conversation state: {json.dumps(state_data, indent=2)}")
        
        assert state_data["status"] == "completed", f"Expected completed, got {state_data['status']}"
        assert state_data["current_outcome"] == "emergency", f"Wrong outcome in state: {state_data['current_outcome']}"
        
        # Step 4: Verify background tasks were triggered
        await asyncio.sleep(2)  # Allow background tasks to complete
        
        print(f"📞 Webhook calls made: {len(mock_webhook_calls)}")
        for call in mock_webhook_calls:
            print(f"  - {call['method']} {call['url']} at {call['timestamp']}")
        
        # Should have triggered emergency webhook
        emergency_calls = [call for call in mock_webhook_calls if "alert" in call["url"].lower()]
        assert len(emergency_calls) > 0, "Emergency webhook should have been called"
        
        print("✅ Emergency flow completed successfully")

    @pytest.mark.asyncio 
    async def test_multiple_emergency_scenarios(self, test_client):
        """Test different emergency symptoms produce consistent results"""
        print("\n🚨 TESTING: Multiple Emergency Scenarios")
        
        results = []
        
        for i, symptoms in enumerate(EMERGENCY_SYMPTOMS):
            print(f"\n📋 Scenario {i+1}: {symptoms}")
            
            payload = {
                "user_message": symptoms,
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"emergency_patient_{i+1:03d}",
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            response = await test_client.post("/api/v2/medical/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            results.append({
                "symptoms": symptoms,
                "outcome": data["medical_outcome"],
                "confidence": data["confidence_score"],
                "red_flags": data["red_flags"],
                "complete": data["is_conversation_complete"]
            })
            
            print(f"  ➤ Outcome: {data['medical_outcome']}")
            print(f"  ➤ Confidence: {data['confidence_score']}%")
            print(f"  ➤ Red flags: {len(data['red_flags'])}")
        
        # Analyze results
        print(f"\n📊 EMERGENCY DETECTION ANALYSIS:")
        print(f"Total scenarios tested: {len(results)}")
        
        emergency_count = sum(1 for r in results if r["outcome"] == "emergency_route_to_doctor")
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        total_red_flags = sum(len(r["red_flags"]) for r in results)
        
        print(f"Emergency detections: {emergency_count}/{len(results)} ({emergency_count/len(results)*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.1f}%")
        print(f"Total red flags identified: {total_red_flags}")
        
        # All should be classified as emergency
        assert emergency_count == len(results), f"All {len(results)} scenarios should be emergency"
        assert avg_confidence >= 80, f"Average confidence too low: {avg_confidence}"
        
        print("✅ All emergency scenarios handled correctly")

    @pytest.mark.asyncio
    async def test_emergency_conversation_persistence(self, test_client):
        """Test emergency conversations are properly persisted"""
        print("\n💾 TESTING: Emergency Conversation Persistence")
        
        # Create emergency conversation
        payload = {
            "user_message": "severe chest pain and can't breathe",
            "stakeholder_role": StakeholderRole.PATIENT.value,
            "stakeholder_id": "persistence_test_patient",
            "chat_provider": ChatProvider.API_DIRECT.value
        }
        
        response = await test_client.post("/api/v2/medical/chat", json=payload)
        data = response.json()
        conv_id = data["conversation_id"]
        
        # Wait for background persistence
        await asyncio.sleep(3)
        
        # Verify conversation history is accessible
        history_response = await test_client.get(f"/api/v2/medical/chat/{conv_id}/history")
        assert history_response.status_code == 200
        
        history_data = history_response.json()
        print(f"📚 Conversation history: {json.dumps(history_data, indent=2)}")
        
        assert history_data["status"] == "completed"
        assert history_data["outcome"] == "emergency"
        assert len(history_data["turns"]) >= 1
        
        # Verify turn data structure
        turn = history_data["turns"][0]
        required_fields = ["turn_number", "user_message", "agent_response", "medical_outcome", "confidence_score"]
        
        for field in required_fields:
            assert field in turn, f"Missing required field: {field}"
        
        print("✅ Emergency conversation properly persisted")

    @pytest.mark.asyncio
    async def test_system_recovery_after_emergency(self, test_client):
        """Test system can handle new requests after emergency"""
        print("\n🔄 TESTING: System Recovery After Emergency")
        
        # First: Process emergency
        emergency_payload = {
            "user_message": "crushing chest pain",
            "stakeholder_role": StakeholderRole.PATIENT.value,
            "stakeholder_id": "recovery_test_emergency",
            "chat_provider": ChatProvider.API_DIRECT.value
        }
        
        emergency_response = await test_client.post("/api/v2/medical/chat", json=emergency_payload)
        assert emergency_response.status_code == 200
        
        print("✅ Emergency processed successfully")
        
        # Wait briefly
        await asyncio.sleep(1)
        
        # Second: Process normal request
        normal_payload = {
            "user_message": "mild headache for a few hours",
            "stakeholder_role": StakeholderRole.PATIENT.value,
            "stakeholder_id": "recovery_test_normal",
            "chat_provider": ChatProvider.API_DIRECT.value
        }
        
        normal_response = await test_client.post("/api/v2/medical/chat", json=normal_payload)
        assert normal_response.status_code == 200
        
        normal_data = normal_response.json()
        print(f"📋 Normal request outcome: {normal_data['medical_outcome']}")
        
        # Should NOT be emergency
        assert normal_data["medical_outcome"] != "emergency_route_to_doctor"
        assert normal_data["is_conversation_complete"] is False  # Should ask follow-up
        
        print("✅ System recovered and processing normal requests")
        
        # Third: Verify health check
        health_response = await test_client.get("/api/v2/health")
        health_data = health_response.json()
        
        print(f"🏥 System health: {health_data['status']}")
        assert health_response.status_code in [200, 503]  # 503 if services not fully initialized
        
        print("✅ System recovery test completed")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])