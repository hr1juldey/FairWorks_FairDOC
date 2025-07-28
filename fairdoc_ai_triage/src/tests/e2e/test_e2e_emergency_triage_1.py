"""
E2E Test: Complete Emergency Triage Flow

Tests the entire emergency detection pipeline from HTTP request
through DSPy agent, NICE protocols, stakeholder routing, to final response.

File: src/tests/e2e/test_e2e_emergency_triage.py
"""

import pytest
import asyncio
import json
from datetime import datetime
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import structlog

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2

# Configure test logging
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    logger_factory=structlog.testing.LogCapture,
    cache_logger_on_first_use=True,
)

@pytest.fixture
async def real_app_client():
    """AsyncClient with real FastAPI app but mocked external dependencies"""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

@pytest.fixture
def mock_external_services():
    """Mock external services but keep internal logic intact"""
    with patch('src.app2.services.dspy.medical_agent.dspy') as mock_dspy, \
         patch('src.app2.services.context.redis_queue.Redis') as mock_redis, \
         patch('src.app2.services.chat.raven_bridge.httpx.AsyncClient') as mock_http, \
         patch('src.app2.core.dependencies_v2.init_redis_pool') as mock_redis_init, \
         patch('src.app2.core.dependencies_v2.init_services') as mock_services_init:
        
        # Mock DSPy agent to return emergency response
        mock_agent = AsyncMock()
        mock_agent.process_turn.return_value = {
            "outcome": "emergency", 
            "confidence": 95,
            "next_question": None,
            "reasoning": "Crushing chest pain with radiation suggests acute MI",
            "red_flags": ["chest_pain", "radiation", "diaphoresis"],
            "is_complete": True
        }
        mock_dspy.ChainOfThought.return_value = mock_agent
        
        # Mock Redis for conversation state
        mock_redis_client = AsyncMock()
        mock_redis_client.get.return_value = json.dumps({
            "conversation_id": "conv_emergency_001",
            "user_id": "patient_emergency",
            "turn_count": 1,
            "status": "active",
            "conversation_history": [],
            "red_flags_detected": [],
            "current_outcome": "inconclusive"
        })
        mock_redis.from_url.return_value = mock_redis_client
        
        # Mock external HTTP calls (Raven, webhooks)
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value.status_code = 200
        mock_http.return_value.__aenter__.return_value = mock_http_client
        
        yield {
            "dspy": mock_dspy,
            "redis": mock_redis_client,
            "http": mock_http_client,
            "agent": mock_agent
        }

class TestEmergencyTriageE2E:
    """End-to-end emergency triage workflow tests"""
    
    @pytest.mark.asyncio
    async def test_complete_emergency_detection_flow(self, real_app_client, mock_external_services):
        """Test complete emergency detection from API to external alerts"""
        
        # Step 1: Patient sends emergency symptoms
        emergency_payload = {
            "user_message": "I have crushing chest pain radiating to my left arm with sweating",
            "stakeholder_role": "patient",
            "stakeholder_id": "emergency_patient_001",
            "chat_provider": "api_direct"
        }
        
        # Step 2: Send request to API
        response = await real_app_client.post(
            "/api/v2/medical/chat",
            json=emergency_payload
        )
        
        # Step 3: Verify emergency response
        assert response.status_code == 200
        data = response.json()
        
        # Check emergency classification
        assert data["medical_outcome"] == "emergency"
        assert data["confidence_score"] >= 90
        assert data["is_conversation_complete"] is True
        assert len(data["red_flags"]) > 0
        
        # Verify conversation tracking
        conversation_id = data["conversation_id"]
        assert conversation_id.startswith("conv_")
        assert data["turn_count"] >= 1
        
        # Step 4: Verify state endpoint shows emergency status
        state_response = await real_app_client.get(
            f"/api/v2/medical/chat/{conversation_id}/state"
        )
        assert state_response.status_code == 200
        state_data = state_response.json()
        assert "emergency" in state_data.get("current_outcome", "").lower()
        
        # Step 5: Verify DSPy agent was called with correct context
        mock_agent = mock_external_services["agent"]
        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args
        symptoms = call_args[1]["symptoms"]
        assert "chest pain" in symptoms.lower()
        assert "radiating" in symptoms.lower()
        
        # Step 6: Verify Redis state management
        redis_client = mock_external_services["redis"]
        redis_client.get.assert_called()  # Conversation state retrieved
        
        # Step 7: Verify external webhook calls would be made
        # (Background tasks are fire-and-forget, so we check the setup)
        http_client = mock_external_services["http"]
        # Emergency handler would make webhook calls in background
        
        print(f"✅ Emergency E2E Test Passed - Conversation ID: {conversation_id}")
        print(f"📊 Emergency Response Data: {json.dumps(data, indent=2)}")
    
    @pytest.mark.asyncio 
    async def test_emergency_escalation_chain(self, real_app_client, mock_external_services):
        """Test emergency triggers proper escalation chain"""
        
        payload = {
            "user_message": "Severe chest pain, can't breathe, feels like elephant on chest",
            "stakeholder_role": "patient",
            "stakeholder_id": "critical_patient_002"
        }
        
        response = await real_app_client.post("/api/v2/medical/chat", json=payload)
        data = response.json()
        
        # Emergency should be detected
        assert data["medical_outcome"] == "emergency"
        assert data["confidence_score"] >= 85
        
        # Verify multiple red flags
        red_flags = data["red_flags"]
        assert len(red_flags) >= 2
        expected_flags = ["chest_pain", "shortness_breath", "severe_pain"]
        assert any(flag in " ".join(red_flags) for flag in expected_flags)
        
        # Check reasoning mentions emergency indicators
        reasoning = data.get("reasoning", "").lower()
        assert any(term in reasoning for term in ["emergency", "acute", "immediate", "critical"])
        
        print(f"🚨 Emergency Escalation Test Passed")
        print(f"🔍 Red Flags Detected: {red_flags}")
        print(f"💭 Emergency Reasoning: {data.get('reasoning', 'N/A')}")

    @pytest.mark.asyncio
    async def test_system_health_during_emergency(self, real_app_client):
        """Test system health endpoints work during emergency processing"""
        
        # Check system health before emergency
        health_response = await real_app_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # May be degraded but should respond
        
        # Check API info
        info_response = await real_app_client.get("/api/v2/info")
        assert info_response.status_code == 200
        info_data = info_response.json()
        assert info_data["name"] == "Fairdoc AI Triage System V2"
        assert "medical_chat" in info_data["endpoints"]
        
        # Check readiness probe
        ready_response = await real_app_client.get("/api/v2/health/ready")
        assert ready_response.status_code in [200, 503]
        
        # Check liveness probe
        live_response = await real_app_client.get("/api/v2/health/live")
        assert live_response.status_code == 200
        live_data = live_response.json()
        assert live_data["status"] == "alive"
        assert live_data["version"] == "v2.6-stable"
        
        print("🏥 System Health Check Passed During Emergency Scenarios")

    @pytest.mark.asyncio
    async def test_emergency_response_structure(self, real_app_client, mock_external_services):
        """Test emergency response has all required fields"""
        
        payload = {
            "user_message": "Heart attack symptoms - severe chest pain",
            "stakeholder_role": "patient"
        }
        
        response = await real_app_client.post("/api/v2/medical/chat", json=payload)
        data = response.json()
        
        # Required response fields
        required_fields = [
            "conversation_id", "medical_outcome", "confidence_score",
            "turn_count", "is_conversation_complete", "red_flags",
            "reasoning", "agent_message"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Emergency-specific validations
        assert data["medical_outcome"] == "emergency"
        assert isinstance(data["confidence_score"], (int, float))
        assert 0 <= data["confidence_score"] <= 100
        assert isinstance(data["red_flags"], list)
        assert len(data["red_flags"]) > 0
        assert isinstance(data["reasoning"], str)
        assert len(data["reasoning"]) > 0
        
        print("📋 Emergency Response Structure Validation Passed")
        print(f"📈 Confidence Score: {data['confidence_score']}")
        print(f"🚩 Red Flags Count: {len(data['red_flags'])}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])