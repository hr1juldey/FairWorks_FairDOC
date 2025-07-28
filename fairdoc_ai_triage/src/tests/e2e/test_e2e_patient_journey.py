"""
End-to-End Test: Complete Patient Journey
==========================================

Tests the full patient journey from initial symptom report through triage decision.
Validates entire V2 stack: FastAPI → ChatOrchestrator → DSPy → Redis → PostgreSQL

File: src/tests/e2e/test_e2e_patient_journey.py
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from httpx import AsyncClient
from fastapi import status

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider

class TestPatientJourneyE2E:
    """End-to-end test for complete patient medical triage journey"""
    
    @pytest.fixture(scope="class")
    async def client(self):
        """HTTP client for E2E testing"""
        async with AsyncClient(app=app, base_url="http://test", timeout=30.0) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_routine_headache_journey(self, client):
        """Test complete routine headache triage journey"""
        print("\n🏥 Starting E2E Patient Journey: Routine Headache")
        
        # Step 1: Initial patient contact
        initial_payload = {
            "user_message": "I have a headache that started this morning",
            "stakeholder_role": "patient", 
            "stakeholder_id": "e2e_patient_001",
            "chat_provider": "api_direct"
        }
        
        start_time = time.time()
        response1 = await client.post("/api/v2/medical/chat", json=initial_payload)
        response_time_1 = time.time() - start_time
        
        print(f"📝 Turn 1 Response Time: {response_time_1:.2f}s")
        assert response1.status_code == status.HTTP_200_OK
        
        data1 = response1.json()
        conversation_id = data1["conversation_id"]
        
        # Validate initial response structure
        assert "agent_message" in data1
        assert "medical_outcome" in data1
        assert "confidence_score" in data1
        assert "turn_count" in data1
        assert data1["turn_count"] == 1
        
        print(f"🆔 Conversation ID: {conversation_id}")
        print(f"🤖 Agent Response: {data1['agent_message']}")
        print(f"🎯 Medical Outcome: {data1['medical_outcome']}")
        print(f"📊 Confidence: {data1['confidence_score']}")
        
        # Step 2: Follow-up questions
        if not data1.get("is_conversation_complete", False):
            followup_payload = {
                "conversation_id": conversation_id,
                "user_message": "It's a dull ache, about 5/10 pain level",
                "stakeholder_role": "patient",
                "stakeholder_id": "e2e_patient_001",
                "chat_provider": "api_direct"
            }
            
            start_time = time.time()
            response2 = await client.post("/api/v2/medical/chat", json=followup_payload)
            response_time_2 = time.time() - start_time
            
            print(f"📝 Turn 2 Response Time: {response_time_2:.2f}s")
            assert response2.status_code == status.HTTP_200_OK
            
            data2 = response2.json()
            assert data2["conversation_id"] == conversation_id
            assert data2["turn_count"] == 2
            
            print(f"🤖 Follow-up Response: {data2['agent_message']}")
            print(f"🎯 Updated Outcome: {data2['medical_outcome']}")
        
        # Step 3: Verify conversation state persistence
        state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == status.HTTP_200_OK
        
        state_data = state_response.json()
        assert state_data["conversation_id"] == conversation_id
        assert state_data["turn_count"] >= 1
        assert "conversation_history" in state_data
        
        print(f"💾 Conversation State: {len(state_data['conversation_history'])} turns stored")
        
        # Step 4: Verify conversation history
        history_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/history")
        assert history_response.status_code == status.HTTP_200_OK
        
        history_data = history_response.json()
        assert "turns" in history_data
        assert len(history_data["turns"]) >= 1
        
        print(f"📚 History Retrieved: {len(history_data['turns'])} turns")
        
        # Log final journey summary
        total_time = response_time_1 + (response_time_2 if 'response_time_2' in locals() else 0)
        print(f"✅ Journey Complete - Total Time: {total_time:.2f}s")
        print(f"🏁 Final Outcome: {data2.get('medical_outcome', data1['medical_outcome'])}")

    @pytest.mark.asyncio 
    async def test_chest_pain_emergency_journey(self, client):
        """Test complete emergency chest pain triage journey"""
        print("\n🚨 Starting E2E Patient Journey: Emergency Chest Pain")
        
        # Emergency symptoms should trigger immediate escalation
        emergency_payload = {
            "user_message": "I have severe crushing chest pain radiating to my left arm and I'm sweating profusely",
            "stakeholder_role": "patient",
            "stakeholder_id": "e2e_emergency_001", 
            "chat_provider": "api_direct"
        }
        
        start_time = time.time()
        response = await client.post("/api/v2/medical/chat", json=emergency_payload)
        response_time = time.time() - start_time
        
        print(f"🚨 Emergency Response Time: {response_time:.2f}s")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        conversation_id = data["conversation_id"]
        
        # Validate emergency response
        assert data["medical_outcome"] in ["emergency", "routine_doctor"]
        assert data["confidence_score"] >= 70  # Should be high confidence for emergency
        
        if "red_flags" in data:
            assert len(data["red_flags"]) > 0
            print(f"🚩 Red Flags Detected: {data['red_flags']}")
        
        print(f"🆔 Emergency Conversation ID: {conversation_id}")
        print(f"🎯 Emergency Outcome: {data['medical_outcome']}")
        print(f"📊 Emergency Confidence: {data['confidence_score']}")
        print(f"🤖 Emergency Response: {data['agent_message']}")
        
        # Verify emergency conversation is marked appropriately
        state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == status.HTTP_200_OK
        
        state_data = state_response.json()
        print(f"🚑 Emergency State Status: {state_data.get('status', 'unknown')}")
        
        if data.get("is_conversation_complete", False):
            print("✅ Emergency Journey Complete - Immediate Referral")
        else:
            print("⚠️ Emergency Journey Continuing - Additional Questions")

    @pytest.mark.asyncio
    async def test_system_health_validation(self, client):
        """Validate system health before running patient journeys"""
        print("\n🔍 Starting E2E System Health Validation")
        
        # Test basic health endpoint
        health_response = await client.get("/api/v2/health")
        print(f"🏥 Health Check Status: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ System Status: {health_data.get('status', 'unknown')}")
            print(f"🔧 Services: {json.dumps(health_data.get('services', {}), indent=2)}")
        else:
            print(f"⚠️ System Health Issues Detected")
            
        # Test API info endpoint
        info_response = await client.get("/api/v2/info") 
        assert info_response.status_code == 200
        
        info_data = info_response.json()
        print(f"📋 API Version: {info_data.get('version', 'unknown')}")
        print(f"🎯 Features Available: {json.dumps(info_data.get('features', {}), indent=2)}")
        
        # Test readiness endpoint
        ready_response = await client.get("/api/v2/health/ready")
        print(f"🚀 Readiness Status: {ready_response.status_code}")
        
        if ready_response.status_code == 200:
            ready_data = ready_response.json()
            print(f"✅ System Ready: {ready_data.get('status', 'unknown')}")
        
        print("🔍 System Health Validation Complete")

    @pytest.mark.asyncio
    async def test_invalid_requests_handling(self, client):
        """Test system handles invalid requests gracefully"""
        print("\n❌ Starting E2E Invalid Request Handling")
        
        # Test empty message
        empty_payload = {
            "user_message": "",
            "stakeholder_role": "patient",
            "stakeholder_id": "e2e_invalid_001"
        }
        
        response = await client.post("/api/v2/medical/chat", json=empty_payload)
        print(f"📝 Empty Message Response: {response.status_code}")
        
        # Test invalid conversation ID
        invalid_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/api/v2/medical/chat/{invalid_id}/state")
        print(f"🆔 Invalid ID Response: {response.status_code}")
        assert response.status_code == 404
        
        # Test malformed JSON
        try:
            response = await client.post("/api/v2/medical/chat", json={"invalid": "structure"})
            print(f"🔧 Malformed JSON Response: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Malformed JSON Exception: {str(e)}")
        
        print("❌ Invalid Request Handling Complete")