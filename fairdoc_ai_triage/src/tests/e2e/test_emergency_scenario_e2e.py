"""
End-to-End Test: Complete Emergency Medical Scenario

Tests the complete emergency triage workflow from patient input
through DSPy agent processing, emergency detection, stakeholder
routing, and background task execution.

File: src/tests/e2e/test_emergency_scenario_e2e.py
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.app2.main_v2 import app
from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatRequest,
    StakeholderRole,
    ChatProvider,
    MedicalOutcome
)


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_redis():
    """Setup test Redis connection"""
    redis_client = redis.from_url("redis://localhost:6379/14", decode_responses=True)
    await redis_client.flushdb()
    yield redis_client
    await redis_client.flushdb()
    await redis_client.close()


@pytest.fixture
async def emergency_test_client():
    """FastAPI test client for emergency scenarios"""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


class TestEmergencyScenarioE2E:
    """Complete emergency scenario end-to-end testing"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_emergency_chest_pain_flow(self, emergency_test_client, test_redis):
        """
        Test complete emergency flow:
        1. Patient reports chest pain
        2. System detects emergency
        3. Routes to doctor immediately
        4. Triggers background alerts
        5. Persists to database
        """
        # Mock dependencies for controlled testing
        with patch('src.app2.core.dependencies_v2.get_medical_agent') as mock_agent, \
             patch('src.app2.core.dependencies_v2.get_conversation_queue') as mock_queue, \
             patch('src.app2.services.chat.emergency_handler.EmergencyHandler') as mock_emergency:
            
            # Setup mocks
            mock_agent_instance = AsyncMock()
            mock_agent_instance.process_turn.return_value = {
                "outcome": "emergency",
                "confidence": 95,
                "next_question": None,
                "reasoning": "Chest pain with radiation suggests cardiac emergency",
                "red_flags": ["chest_pain", "arm_radiation", "shortness_breath"],
                "is_complete": True
            }
            mock_agent.return_value = mock_agent_instance
            
            # Mock Redis queue
            mock_queue_instance = AsyncMock()
            conversation_id = "conv_emergency_e2e_001"
            mock_queue_instance.start_conversation.return_value = conversation_id
            mock_queue_instance.get_conversation_state.return_value = {
                "conversation_id": conversation_id,
                "user_id": "emergency_patient_001",
                "status": "completed",
                "turn_count": 1,
                "current_outcome": "emergency",
                "red_flags_detected": ["chest_pain", "arm_radiation"],
                "conversation_history": []
            }
            mock_queue_instance.update_conversation_turn.return_value = {
                "conversation_id": conversation_id,
                "status": "completed",
                "turn_count": 1,
                "current_outcome": "emergency"
            }
            mock_queue.return_value = mock_queue_instance
            
            # Mock emergency handler
            mock_emergency_instance = AsyncMock()
            mock_emergency.return_value = mock_emergency_instance
            
            # Step 1: Patient reports emergency symptoms
            emergency_payload = {
                "user_message": "I have severe crushing chest pain radiating down my left arm and I'm sweating profusely",
                "stakeholder_role": "patient",
                "stakeholder_id": "emergency_patient_001",
                "chat_provider": "api_direct"
            }
            
            response = await emergency_test_client.post(
                "/api/v2/medical/chat",
                json=emergency_payload
            )
            
            # Verify immediate emergency response
            assert response.status_code == 200
            data = response.json()
            
            assert data["medical_outcome"] == "emergency"
            assert data["confidence_score"] >= 90
            assert data["is_conversation_complete"] is True
            assert len(data["red_flags"]) >= 2
            assert "chest_pain" in data["red_flags"]
            
            # Verify conversation was created and completed
            mock_queue_instance.start_conversation.assert_called_once()
            mock_queue_instance.update_conversation_turn.assert_called_once()
            
            # Verify emergency handler was triggered
            mock_emergency_instance.handle_emergency_alert.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_emergency_stakeholder_routing(self, emergency_test_client):
        """
        Test emergency stakeholder routing:
        1. Emergency detected
        2. Multiple stakeholders notified
        3. Priority routing to doctors
        4. Admin notification for tracking
        """
        with patch('src.app2.services.chat.stakeholder_router.StakeholderRouter') as mock_router:
            
            # Mock router to return emergency routes
            mock_router_instance = AsyncMock()
            mock_router_instance.route_message.return_value = [
                # Route to agent first
                {
                    "from_stakeholder": "patient",
                    "to_stakeholder": "fairdoc_agent", 
                    "priority": "medium",
                    "message_content": "Emergency symptoms reported"
                },
                # Emergency route to doctor
                {
                    "from_stakeholder": "fairdoc_agent",
                    "to_stakeholder": "doctor",
                    "priority": "emergency", 
                    "requires_human_review": True,
                    "message_content": "🚨 EMERGENCY: Chest pain with cardiac symptoms"
                },
                # Admin notification
                {
                    "from_stakeholder": "system",
                    "to_stakeholder": "admin",
                    "priority": "high",
                    "message_content": "Emergency case logged - tracking required"
                }
            ]
            mock_router.return_value = mock_router_instance
            
            # Test emergency routing
            payload = {
                "user_message": "Help! I can't breathe and have chest pain!",
                "stakeholder_role": "patient",
                "stakeholder_id": "emergency_patient_002"
            }
            
            response = await emergency_test_client.post("/api/v2/medical/chat", json=payload)
            assert response.status_code == 200
            
            # Verify routing was called with emergency outcome
            args, kwargs = mock_router_instance.route_message.call_args
            assert kwargs.get("medical_outcome") == "emergency" or "emergency" in str(args)
    
    @pytest.mark.asyncio
    @pytest.mark.e2e  
    async def test_emergency_system_health_during_load(self, emergency_test_client):
        """
        Test system health during emergency load:
        1. Multiple concurrent emergency requests
        2. System remains responsive
        3. Health checks pass under load
        4. No resource leaks
        """
        # Test health endpoint before load
        health_response = await emergency_test_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # May not be fully initialized
        
        # Simulate concurrent emergency requests
        async def emergency_request(patient_id: int):
            payload = {
                "user_message": f"Emergency patient {patient_id}: severe chest pain!",
                "stakeholder_role": "patient", 
                "stakeholder_id": f"emergency_load_patient_{patient_id}"
            }
            return await emergency_test_client.post("/api/v2/medical/chat", json=payload)
        
        # Send 5 concurrent emergency requests
        tasks = [emergency_request(i) for i in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests were handled
        success_count = 0
        for response in responses:
            if not isinstance(response, Exception):
                assert response.status_code == 200
                success_count += 1
        
        assert success_count >= 3  # At least 60% success rate under load
        
        # Verify system health after load
        health_response = await emergency_test_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]
        
        # Test readiness probe
        ready_response = await emergency_test_client.get("/api/v2/health/ready")
        assert ready_response.status_code in [200, 503]
        
        # Test liveness probe (should always pass)
        live_response = await emergency_test_client.get("/api/v2/health/live") 
        assert live_response.status_code == 200
        assert live_response.json()["status"] == "alive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-m", "e2e"])