"""
E2E Test: Emergency Medical Scenario Flow

Tests the complete end-to-end flow for emergency medical scenarios:
- Patient reports chest pain symptoms
- System detects emergency condition
- Routes to appropriate stakeholders
- Triggers background emergency handling
- Persists conversation data

File: src/tests/e2e/test_e2e_emergency_flow.py
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import structlog

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2

logger = structlog.get_logger(__name__)

class TestEmergencyFlowE2E:
    """End-to-end test for emergency medical scenario"""
    
    @pytest.fixture
    async def client(self):
        """Create HTTP client for E2E testing"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def emergency_patient_data(self):
        """Emergency scenario test data"""
        return {
            "user_message": "I have crushing chest pain radiating to my left arm and I'm sweating profusely",
            "stakeholder_role": "patient",
            "stakeholder_id": "emergency_patient_001",
            "chat_provider": "api_direct"
        }
    
    @pytest.fixture
    def mock_external_services(self):
        """Mock external services for isolated E2E testing"""
        with patch('src.app2.services.chat.raven_bridge.RavenBridge.send_message') as mock_raven, \
             patch('src.app2.services.chat.emergency_handler.EmergencyHandler.handle_emergency_alert') as mock_emergency, \
             patch('src.app2.services.chat.persistence_handler.PersistenceHandler.save_conversation_to_postgresql') as mock_persist:
            
            mock_raven.return_value = {"status": "sent", "message_id": "msg_123"}
            mock_emergency.return_value = True
            mock_persist.return_value = True
            
            yield {
                "raven": mock_raven,
                "emergency": mock_emergency,
                "persistence": mock_persist
            }
    
    @pytest.mark.asyncio
    async def test_emergency_detection_and_routing(self, client, emergency_patient_data, mock_external_services):
        """
        Test complete emergency detection and routing flow
        
        Expected Flow:
        1. Patient reports emergency symptoms
        2. DSPy agent detects emergency
        3. System routes to doctor immediately
        4. Emergency handler triggered
        5. Conversation marked as emergency
        """
        logger.info("🚨 Starting emergency flow E2E test")
        
        start_time = time.time()
        
        # Step 1: Send emergency symptoms
        response = await client.post("/api/v2/medical/chat", json=emergency_patient_data)
        
        # Verify HTTP response
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        logger.info("📨 Emergency chat response received", 
                   conversation_id=data.get("conversation_id"),
                   outcome=data.get("medical_outcome"))
        
        # Step 2: Verify emergency detection
        assert data["medical_outcome"] == "emergency", f"Expected emergency, got {data['medical_outcome']}"
        assert data["is_conversation_complete"] is True, "Emergency should complete conversation immediately"
        assert data["confidence_score"] >= 80, f"Expected high confidence, got {data['confidence_score']}"
        
        # Step 3: Verify emergency indicators
        assert len(data["red_flags"]) > 0, "Emergency should have red flags"
        emergency_flags = ["chest_pain", "cardiac", "crushing", "radiating"]
        assert any(flag in str(data["red_flags"]).lower() for flag in emergency_flags), \
               f"Expected emergency red flags, got {data['red_flags']}"
        
        # Step 4: Verify conversation state
        conversation_id = data["conversation_id"]
        state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == 200
        
        state_data = state_response.json()
        assert state_data["status"] == "completed", f"Expected completed, got {state_data['status']}"
        assert state_data["current_outcome"] == "emergency"
        
        # Step 5: Verify background task triggers
        # Wait brief moment for background tasks to trigger
        await asyncio.sleep(0.1)
        
        # Emergency handler should have been called
        mock_external_services["emergency"].assert_called_once()
        call_args = mock_external_services["emergency"].call_args
        assert "emergency_patient_001" in str(call_args)
        
        # Step 6: Performance verification
        end_time = time.time()
        response_time = end_time - start_time
        assert response_time < 5.0, f"Emergency response too slow: {response_time:.2f}s"
        
        logger.info("✅ Emergency flow E2E test completed successfully", 
                   response_time_ms=round(response_time * 1000, 2))
    
    @pytest.mark.asyncio
    async def test_emergency_stakeholder_routing(self, client, emergency_patient_data):
        """Test emergency stakeholder routing and notifications"""
        
        # Send emergency scenario
        response = await client.post("/api/v2/medical/chat", json=emergency_patient_data)
        data = response.json()
        
        conversation_id = data["conversation_id"]
        
        # Verify conversation history includes routing
        history_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/history")
        assert history_response.status_code == 200
        
        history_data = history_response.json()
        assert len(history_data["turns"]) >= 1
        
        # First turn should be emergency classification
        first_turn = history_data["turns"][0]
        assert first_turn["outcome"] == "emergency"
        assert first_turn["user_response"] == emergency_patient_data["user_message"]
        
        logger.info("📋 Emergency stakeholder routing verified", 
                   conversation_id=conversation_id,
                   turns=len(history_data["turns"]))
    
    @pytest.mark.asyncio
    async def test_emergency_system_health_during_crisis(self, client, emergency_patient_data):
        """Test system health monitoring during emergency scenarios"""
        
        # Check system health before emergency
        health_response = await client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # May be initializing
        
        # Process emergency scenario
        chat_response = await client.post("/api/v2/medical/chat", json=emergency_patient_data)
        assert chat_response.status_code == 200
        
        # Verify system remains healthy after emergency processing
        post_health_response = await client.get("/api/v2/health")
        assert post_health_response.status_code in [200, 503]
        
        health_data = post_health_response.json()
        logger.info("🏥 System health during emergency", 
                   status=health_data.get("status"),
                   services=health_data.get("services", {}))
        
        # System should still be responsive
        info_response = await client.get("/api/v2/info")
        assert info_response.status_code == 200
        
        info_data = info_response.json()
        assert info_data["version"] == "v2.6-stable"
        assert info_data["features"]["emergency_detection"] is True
    
    @pytest.mark.asyncio
    async def test_emergency_data_persistence_flow(self, client, emergency_patient_data, mock_external_services):
        """Test emergency data persistence and recovery"""
        
        # Process emergency
        response = await client.post("/api/v2/medical/chat", json=emergency_patient_data)
        data = response.json()
        conversation_id = data["conversation_id"]
        
        # Verify immediate state persistence
        state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        state_data = state_response.json()
        
        # Emergency should be marked for persistence
        assert state_data["status"] == "completed"
        assert "emergency" in state_data["current_outcome"]
        
        # Verify background persistence was triggered
        await asyncio.sleep(0.1)  # Allow background tasks
        mock_external_services["persistence"].assert_called_once()
        
        # Test conversation reset capability
        reset_response = await client.post(f"/api/v2/medical/chat/{conversation_id}/reset")
        assert reset_response.status_code == 200
        
        reset_data = reset_response.json()
        assert "reset successfully" in reset_data["message"]
        
        logger.info("💾 Emergency data persistence verified", 
                   conversation_id=conversation_id,
                   persistence_triggered=mock_external_services["persistence"].called)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])