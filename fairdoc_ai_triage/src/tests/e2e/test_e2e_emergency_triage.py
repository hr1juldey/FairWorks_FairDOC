"""
End-to-End Emergency Triage Test

Tests complete emergency scenario from patient symptom input through
DSPy agent processing, stakeholder routing, background alerts, and 
PostgreSQL persistence.

File: src/tests/e2e/test_e2e_emergency_triage.py
"""

import pytest
import asyncio
from httpx import AsyncClient
import json
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
import redis.asyncio as redis
from sqlalchemy import text

from src.app2.main_v2 import app
from src.app2.core.database_v2 import get_async_session, database_v2
from src.app2.models.schemas.multiturn_chat import MedicalOutcome


class TestE2EEmergencyTriage:
    """End-to-end emergency triage workflow testing"""
    
    @pytest.fixture(scope="class", autouse=True)
    async def setup_system(self):
        """Setup complete system for e2e testing"""
        # Initialize database
        await database_v2.initialize()
        await database_v2.create_tables()
        
        # Clear Redis test database
        redis_client = redis.from_url("redis://localhost:6379/15", decode_responses=True)
        await redis_client.flushdb()
        await redis_client.close()
        
        yield
        
        # Cleanup
        await database_v2.drop_tables()
        await database_v2.close()
    
    @pytest.fixture
    async def test_client(self):
        """AsyncClient for full HTTP testing"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def mock_emergency_services(self):
        """Mock external emergency services"""
        with patch('src.app2.services.chat.emergency_handler.EmergencyHandler.handle_emergency_alert') as mock_emergency, \
             patch('src.app2.services.chat.raven_bridge.RavenBridge.send_message') as mock_raven:
            
            async def fake_emergency_alert(conv_id, user_id, agent_result):
                # Simulate emergency alert processing
                return {"status": "emergency_alert_sent", "conv_id": conv_id}
            
            async def fake_raven_send(message, recipients):
                # Simulate Raven webhook call
                return {"status": "sent", "message_id": "raven_123"}
            
            mock_emergency.side_effect = fake_emergency_alert
            mock_raven.side_effect = fake_raven_send
            
            yield {"emergency": mock_emergency, "raven": mock_raven}
    
    @pytest.mark.asyncio
    async def test_complete_emergency_flow(self, test_client, mock_emergency_services):
        """Test complete emergency triage flow end-to-end"""
        # Step 1: Patient reports emergency symptoms
        emergency_payload = {
            "user_message": "I have crushing chest pain radiating to my left arm, sweating, and nausea",
            "stakeholder_role": "patient", 
            "stakeholder_id": "emergency_patient_001",
            "chat_provider": "api_direct"
        }
        
        # Make initial request
        response = await test_client.post("/api/v2/medical/chat", json=emergency_payload)
        
        # Verify HTTP response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify emergency detection
        assert response_data["medical_outcome"] == "emergency"
        assert response_data["confidence_score"] >= 80
        assert response_data["is_conversation_complete"] is True
        assert len(response_data["red_flags"]) > 0
        
        conversation_id = response_data["conversation_id"]
        
        # Step 2: Verify conversation state in Redis
        redis_client = redis.from_url("redis://localhost:6379/15", decode_responses=True)
        
        state_key = f"fairdoc:v2:state:{conversation_id}"
        redis_data = await redis_client.get(state_key)
        assert redis_data is not None
        
        state = json.loads(redis_data)
        assert state["status"] == "completed"
        assert state["current_outcome"] == "emergency"
        assert "chest_pain" in str(state["red_flags_detected"]).lower()
        
        # Step 3: Verify conversation moved to completed queue
        completed_conversations = await redis_client.lrange("fairdoc:v2:queue:completed", 0, -1)
        assert conversation_id in completed_conversations
        
        await redis_client.close()
        
        # Step 4: Wait for background tasks to complete
        await asyncio.sleep(1)  # Allow background tasks to execute
        
        # Step 5: Verify emergency handler was called
        mock_emergency_services["emergency"].assert_called_once()
        call_args = mock_emergency_services["emergency"].call_args[0]
        assert call_args[0] == conversation_id  # conversation_id
        assert call_args[1] == "emergency_patient_001"  # user_id
        
        # Step 6: Verify PostgreSQL persistence (background task)
        async with get_async_session() as session:
            # Check if conversation was persisted
            result = await session.execute(
                text("SELECT COUNT(*) FROM conversation_states_v2 WHERE conversation_id = :conv_id"),
                {"conv_id": conversation_id}
            )
            count = result.scalar()
            assert count == 1, "Emergency conversation should be persisted to PostgreSQL"
        
        # Step 7: Test conversation history endpoint
        history_response = await test_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
        assert history_response.status_code == 200
        
        history_data = history_response.json()
        assert history_data["conversation_id"] == conversation_id
        assert history_data["status"] == "completed"
        assert history_data["outcome"] == "emergency"
    
    @pytest.mark.asyncio
    async def test_emergency_stakeholder_routing(self, test_client, mock_emergency_services):
        """Test emergency escalates to all appropriate stakeholders"""
        payload = {
            "user_message": "Severe chest pain, can't breathe, dizzy",
            "stakeholder_role": "patient",
            "stakeholder_id": "emergency_test_002"  
        }
        
        response = await test_client.post("/api/v2/medical/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        conversation_id = data["conversation_id"]
        
        # Emergency should trigger multiple routing paths
        assert data["medical_outcome"] == "emergency"
        
        # Wait for background processing
        await asyncio.sleep(1)
        
        # Verify emergency alert was triggered
        mock_emergency_services["emergency"].assert_called_once()
        
        # Verify Raven notification was sent (stakeholder routing)
        mock_emergency_services["raven"].assert_called()
    
    @pytest.mark.asyncio
    async def test_emergency_conversation_limits(self, test_client):
        """Test emergency scenarios bypass normal conversation turn limits"""
        # Create payload that would normally require multiple turns
        payload = {
            "user_message": "I feel unwell",  # Vague symptom
            "stakeholder_role": "patient",
            "stakeholder_id": "limit_test_patient"
        }
        
        # First turn - should be inconclusive
        response1 = await test_client.post("/api/v2/medical/chat", json=payload)
        assert response1.status_code == 200
        
        data1 = response1.json()
        conv_id = data1["conversation_id"]
        assert data1["medical_outcome"] == "inconclusive"
        
        # Follow-up with emergency symptoms - should immediately escalate
        emergency_payload = {
            "conversation_id": conv_id,
            "user_message": "Now I have severe chest pain and trouble breathing!",
            "stakeholder_role": "patient",
            "stakeholder_id": "limit_test_patient"
        }
        
        response2 = await test_client.post("/api/v2/medical/chat", json=emergency_payload)
        assert response2.status_code == 200
        
        data2 = response2.json()
        assert data2["medical_outcome"] == "emergency"
        assert data2["is_conversation_complete"] is True
        
        # Conversation should be completed despite being only 2 turns
        history_response = await test_client.get(f"/api/v2/medical/chat/{conv_id}/history")
        history_data = history_response.json()
        assert history_data["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_system_health_during_emergency(self, test_client):
        """Test system health endpoints remain responsive during emergency processing"""
        # Start emergency processing
        payload = {
            "user_message": "Heart attack symptoms - chest pain, sweating, shortness of breath",
            "stakeholder_role": "patient"
        }
        
        # Make emergency request (don't await to test concurrency)
        emergency_task = asyncio.create_task(
            test_client.post("/api/v2/medical/chat", json=payload)
        )
        
        # Health check should still work during processing
        health_response = await test_client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # Either healthy or degraded, but responding
        
        # API info should still work
        info_response = await test_client.get("/api/v2/info")
        assert info_response.status_code == 200
        
        # Wait for emergency processing to complete
        emergency_response = await emergency_task
        assert emergency_response.status_code == 200
        
        emergency_data = emergency_response.json()
        assert emergency_data["medical_outcome"] == "emergency"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])