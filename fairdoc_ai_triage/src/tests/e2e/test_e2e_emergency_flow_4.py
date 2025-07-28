"""
E2E Test: Emergency Medical Scenario Flow

Tests the complete emergency detection and handling pipeline from 
patient input through DSPy agent to emergency alerts and stakeholder routing.

File: src/tests/e2e/test_e2e_emergency_flow.py
"""

import pytest
import asyncio
import time
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import json
from datetime import datetime

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2


class TestEmergencyFlowE2E:
    """End-to-end test for emergency medical scenarios"""
    
    @pytest.fixture(autouse=True)
    async def setup_test_environment(self, monkeypatch):
        """Setup test environment with mocked external dependencies"""
        # Mock DSPy agent for deterministic emergency responses
        emergency_responses = {
            "chest pain": {
                "outcome": "emergency",
                "confidence": 95,
                "next_question": None,
                "reasoning": "Potential cardiac event - immediate medical attention required",
                "red_flags": ["chest_pain", "potential_cardiac_event", "shortness_breath"],
                "is_complete": True
            },
            "stroke symptoms": {
                "outcome": "emergency", 
                "confidence": 98,
                "next_question": None,
                "reasoning": "FAST criteria met - possible stroke",
                "red_flags": ["facial_droop", "speech_difficulty", "weakness"],
                "is_complete": True
            }
        }
        
        async def mock_dspy_process(symptoms, nice_context=""):
            for key, response in emergency_responses.items():
                if key in symptoms.lower():
                    return response
            return {
                "outcome": "inconclusive",
                "confidence": 60,
                "next_question": "Can you describe your symptoms in more detail?",
                "reasoning": "Need more information",
                "red_flags": [],
                "is_complete": False
            }
        
        # Mock emergency handler to capture alerts
        self.emergency_alerts = []
        async def mock_emergency_handler(conversation_id, user_id, agent_result):
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "red_flags": agent_result.get("red_flags", []),
                "outcome": agent_result.get("outcome"),
                "confidence": agent_result.get("confidence")
            }
            self.emergency_alerts.append(alert)
        
        # Mock Raven webhook calls
        self.raven_calls = []
        async def mock_raven_send(message_data):
            self.raven_calls.append({
                "timestamp": datetime.utcnow().isoformat(),
                "message": message_data
            })
            return {"status": "sent", "message_id": "test_msg_123"}
        
        # Apply patches
        monkeypatch.setattr(
            "src.app2.services.dspy.medical_agent.MedicalTriageAgent.process_turn",
            mock_dspy_process
        )
        monkeypatch.setattr(
            "src.app2.services.chat.emergency_handler.EmergencyHandler.handle_emergency_alert",
            mock_emergency_handler
        )
        monkeypatch.setattr(
            "src.app2.services.chat.raven_bridge.RavenBridge.send_message",
            mock_raven_send
        )
        
        # Setup in-memory state
        self.conversation_states = {}
        
    @pytest.mark.asyncio
    async def test_chest_pain_emergency_complete_flow(self):
        """Test complete emergency flow for chest pain scenario"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Patient reports chest pain symptoms
            payload = {
                "user_message": "I'm having severe crushing chest pain radiating to my left arm and I'm sweating profusely",
                "stakeholder_role": "patient",
                "stakeholder_id": "emergency_patient_001",
                "chat_provider": "api_direct"
            }
            
            start_time = time.time()
            response = await client.post("/api/v2/medical/chat", json=payload)
            response_time = time.time() - start_time
            
            # Verify response structure
            assert response.status_code == 200
            data = response.json()
            
            # Verify emergency detection
            assert data["medical_outcome"] == "emergency"
            assert data["confidence_score"] >= 90
            assert data["is_conversation_complete"] is True
            assert len(data["red_flags"]) >= 2
            assert "chest_pain" in data["red_flags"]
            
            # Verify conversation metadata
            conversation_id = data["conversation_id"]
            assert conversation_id.startswith("conv_")
            assert data["turn_count"] == 1
            assert data["stakeholder_id"] == "emergency_patient_001"
            
            # Step 2: Verify conversation state was persisted
            state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            assert state_response.status_code == 200
            
            state = state_response.json()
            assert state["status"] == "completed"
            assert state["current_outcome"] == "emergency"
            assert len(state["red_flags_detected"]) >= 2
            
            # Step 3: Verify conversation history
            history_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/history")
            assert history_response.status_code == 200
            
            history = history_response.json()
            assert history["status"] == "completed"
            assert len(history["turns"]) == 1
            
            # Step 4: Verify emergency alerts were triggered
            await asyncio.sleep(0.1)  # Allow background tasks to complete
            assert len(self.emergency_alerts) == 1
            
            alert = self.emergency_alerts[0]
            assert alert["conversation_id"] == conversation_id
            assert alert["outcome"] == "emergency"
            assert "chest_pain" in alert["red_flags"]
            
            # Step 5: Verify response time is acceptable (< 2 seconds for mocked services)
            assert response_time < 2.0
            
            # Step 6: Test system health after emergency
            health_response = await client.get("/api/v2/health")
            assert health_response.status_code in [200, 503]  # May be degraded but responsive
            
    @pytest.mark.asyncio
    async def test_stroke_symptoms_emergency_escalation(self):
        """Test emergency escalation for stroke symptoms"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "user_message": "My face is drooping on one side, I can't speak properly, and my left arm is weak",
                "stakeholder_role": "patient",
                "stakeholder_id": "stroke_patient_002",
                "chat_provider": "api_direct"
            }
            
            response = await client.post("/api/v2/medical/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["medical_outcome"] == "emergency"
            assert data["confidence_score"] >= 95
            assert "facial_droop" in data["red_flags"]
            assert "speech_difficulty" in data["red_flags"]
            
            # Verify immediate completion for high-confidence emergency
            assert data["is_conversation_complete"] is True
            assert data["next_question"] is None
            
    @pytest.mark.asyncio
    async def test_emergency_system_capacity_handling(self):
        """Test system handles multiple concurrent emergency cases"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Create 5 concurrent emergency scenarios
            emergency_scenarios = [
                "Severe chest pain and difficulty breathing",
                "Sudden severe headache and vision problems", 
                "Crushing chest pressure radiating to jaw",
                "Face drooping and can't move right arm",
                "Severe allergic reaction, throat closing"
            ]
            
            tasks = []
            for i, symptoms in enumerate(emergency_scenarios):
                payload = {
                    "user_message": symptoms,
                    "stakeholder_role": "patient",
                    "stakeholder_id": f"concurrent_emergency_{i}",
                    "chat_provider": "api_direct"
                }
                task = client.post("/api/v2/medical/chat", json=payload)
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all responses succeeded
            successful_responses = 0
            emergency_detections = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    continue
                    
                assert response.status_code == 200
                successful_responses += 1
                
                data = response.json()
                if data["medical_outcome"] == "emergency":
                    emergency_detections += 1
            
            # At least 80% should succeed and detect emergencies appropriately
            assert successful_responses >= 4
            assert emergency_detections >= 3
            
            # Verify emergency alerts were generated
            await asyncio.sleep(0.2)  # Allow background processing
            assert len(self.emergency_alerts) >= 3
            
    @pytest.mark.asyncio
    async def test_emergency_to_non_emergency_false_positive_handling(self):
        """Test system handles initial emergency classification that gets refined"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Start with ambiguous symptoms that might initially seem emergency-like
            payload = {
                "user_message": "I have chest discomfort",
                "stakeholder_role": "patient", 
                "stakeholder_id": "ambiguous_patient_003",
                "chat_provider": "api_direct"
            }
            
            response = await client.post("/api/v2/medical/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            # This should be inconclusive, asking for more details
            assert data["medical_outcome"] in ["inconclusive", "emergency"]
            
            # If emergency was detected, verify the system handled it properly
            if data["medical_outcome"] == "emergency":
                assert data["confidence_score"] >= 50
                assert len(data["red_flags"]) >= 1
                
                # Verify conversation was marked complete for emergency
                assert data["is_conversation_complete"] is True
                
                # Check that emergency alert was generated
                await asyncio.sleep(0.1)
                assert len(self.emergency_alerts) >= 1
                
    @pytest.mark.asyncio 
    async def test_emergency_system_monitoring_endpoints(self):
        """Test system monitoring works during emergency scenarios"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Trigger an emergency first
            payload = {
                "user_message": "Severe chest pain, can't breathe",
                "stakeholder_role": "patient",
                "stakeholder_id": "monitoring_test_patient",
                "chat_provider": "api_direct"
            }
            
            await client.post("/api/v2/medical/chat", json=payload)
            
            # Test health endpoint during emergency processing
            health_response = await client.get("/api/v2/health")
            assert health_response.status_code in [200, 503]
            
            health_data = health_response.json()
            assert "status" in health_data
            assert "services" in health_data
            assert "timestamp" in health_data
            
            # Test readiness endpoint
            ready_response = await client.get("/api/v2/health/ready")
            assert ready_response.status_code in [200, 503]
            
            # Test liveness endpoint (should always respond)
            live_response = await client.get("/api/v2/health/live")
            assert live_response.status_code == 200
            assert live_response.json()["status"] == "alive"
            
            # Test API info endpoint 
            info_response = await client.get("/api/v2/info")
            assert info_response.status_code == 200
            
            info_data = info_response.json()
            assert info_data["name"] == "Fairdoc AI Triage System V2"
            assert info_data["version"] == "v2.6-stable"
            assert "features" in info_data
            assert info_data["features"]["emergency_detection"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])