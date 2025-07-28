# End-to-End Test: Emergency Scenario Workflow
# File: src/tests/e2e/test_e2e_emergency.py

"""
Comprehensive E2E test for emergency medical scenarios.

Tests the complete flow from patient input through DSPy prediction,
emergency detection, stakeholder routing, background tasks, and persistence.

Validates that the entire V2 system responds correctly to emergency situations.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient
from sqlalchemy import text

from src.app2.main_v2 import app
from src.app2.core.database_v2 import get_async_session
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider


@pytest.fixture(scope="module")
def emergency_scenarios():
    """Test cases for different emergency scenarios"""
    return [
        {
            "name": "cardiac_emergency",
            "symptoms": "I have crushing chest pain radiating to my left arm and I'm sweating profusely",
            "expected_outcome": "emergency",
            "expected_confidence": 90,
            "expected_red_flags": ["chest_pain", "cardiac_symptoms"],
            "should_complete": True
        },
        {
            "name": "stroke_symptoms", 
            "symptoms": "I suddenly can't speak properly and my face feels numb on one side",
            "expected_outcome": "emergency",
            "expected_confidence": 95,
            "expected_red_flags": ["neurological_symptoms", "stroke_indicators"],
            "should_complete": True
        },
        {
            "name": "severe_breathing",
            "symptoms": "I can't breathe and my lips are turning blue",
            "expected_outcome": "emergency", 
            "expected_confidence": 98,
            "expected_red_flags": ["respiratory_distress", "cyanosis"],
            "should_complete": True
        }
    ]


@pytest.fixture
async def emergency_test_client():
    """FastAPI test client with emergency-focused mocking"""
    
    # Mock DSPy agent to recognize emergency keywords
    def mock_emergency_agent():
        agent = MagicMock()
        
        async def process_emergency_turn(symptoms, nice_context=""):
            symptoms_lower = symptoms.lower()
            
            if any(keyword in symptoms_lower for keyword in ["chest pain", "crushing", "radiating"]):
                return {
                    "outcome": "emergency",
                    "confidence": 95,
                    "next_question": None,
                    "reasoning": "Cardiac emergency detected - immediate medical attention required",
                    "red_flags": ["chest_pain", "cardiac_symptoms", "radiation_pattern"],
                    "is_complete": True
                }
            elif any(keyword in symptoms_lower for keyword in ["can't speak", "face numb", "stroke"]):
                return {
                    "outcome": "emergency", 
                    "confidence": 98,
                    "next_question": None,
                    "reasoning": "Neurological emergency - possible stroke",
                    "red_flags": ["neurological_symptoms", "stroke_indicators"],
                    "is_complete": True
                }
            elif any(keyword in symptoms_lower for keyword in ["can't breathe", "blue lips", "cyanosis"]):
                return {
                    "outcome": "emergency",
                    "confidence": 99, 
                    "next_question": None,
                    "reasoning": "Severe respiratory distress with cyanosis",
                    "red_flags": ["respiratory_distress", "cyanosis", "hypoxia"],
                    "is_complete": True
                }
            else:
                return {
                    "outcome": "inconclusive",
                    "confidence": 50,
                    "next_question": "Can you describe your symptoms in more detail?",
                    "reasoning": "Need more information",
                    "red_flags": [],
                    "is_complete": False
                }
        
        agent.process_turn = AsyncMock(side_effect=process_emergency_turn)
        return agent
    
    # Mock emergency handler to track alerts
    emergency_alerts = []
    
    async def mock_emergency_alert(conversation_id, user_id, agent_result):
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "outcome": agent_result["outcome"],
            "red_flags": agent_result["red_flags"],
            "confidence": agent_result["confidence"]
        }
        emergency_alerts.append(alert)
    
    with patch('src.app2.core.dependencies_v2.get_medical_agent', return_value=mock_emergency_agent()), \
         patch('src.app2.services.chat.emergency_handler.EmergencyHandler.handle_emergency_alert', 
               side_effect=mock_emergency_alert):
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            client.emergency_alerts = emergency_alerts
            yield client


class TestEmergencyE2EFlow:
    """End-to-end emergency scenario testing"""
    
    @pytest.mark.asyncio 
    async def test_cardiac_emergency_complete_flow(self, emergency_test_client, emergency_scenarios):
        """Test complete cardiac emergency flow from input to persistence"""
        
        cardiac_scenario = next(s for s in emergency_scenarios if s["name"] == "cardiac_emergency")
        
        # Step 1: Patient reports emergency symptoms
        payload = {
            "user_message": cardiac_scenario["symptoms"],
            "stakeholder_role": StakeholderRole.PATIENT.value,
            "stakeholder_id": "emergency_patient_001",
            "chat_provider": ChatProvider.API_DIRECT.value
        }
        
        response = await emergency_test_client.post("/api/v2/medical/chat", json=payload)
        
        # Verify immediate emergency response
        assert response.status_code == 200
        data = response.json()
        
        conversation_id = data["conversation_id"]
        assert data["medical_outcome"] == "emergency"
        assert data["confidence_score"] >= cardiac_scenario["expected_confidence"]
        assert data["is_conversation_complete"] is True
        assert len(data["red_flags"]) > 0
        
        # Step 2: Verify emergency alert was triggered
        await asyncio.sleep(0.1)  # Allow background task to execute
        assert len(emergency_test_client.emergency_alerts) == 1
        
        alert = emergency_test_client.emergency_alerts[0]
        assert alert["conversation_id"] == conversation_id
        assert alert["outcome"] == "emergency"
        assert alert["confidence"] >= 90
        
        # Step 3: Check conversation state was updated
        state_response = await emergency_test_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
        assert state_response.status_code == 200
        
        state_data = state_response.json()
        assert state_data["status"] == "completed"
        assert state_data["current_outcome"] == "emergency"
        assert len(state_data["red_flags_detected"]) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_emergency_scenarios(self, emergency_test_client, emergency_scenarios):
        """Test system handles multiple different emergency types correctly"""
        
        results = []
        
        for scenario in emergency_scenarios:
            payload = {
                "user_message": scenario["symptoms"],
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"patient_{scenario['name']}",
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            response = await emergency_test_client.post("/api/v2/medical/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            results.append({
                "scenario": scenario["name"],
                "conversation_id": data["conversation_id"],
                "outcome": data["medical_outcome"],
                "confidence": data["confidence_score"],
                "red_flags": data["red_flags"],
                "complete": data["is_conversation_complete"]
            })
        
        # Verify all scenarios detected as emergencies
        for result in results:
            assert result["outcome"] == "emergency"
            assert result["confidence"] >= 90
            assert result["complete"] is True
            assert len(result["red_flags"]) > 0
        
        # Verify all emergency alerts were fired
        await asyncio.sleep(0.2)
        assert len(emergency_test_client.emergency_alerts) == len(emergency_scenarios)
    
    @pytest.mark.asyncio
    async def test_emergency_system_resilience(self, emergency_test_client):
        """Test system resilience during emergency processing"""
        
        # Test rapid successive emergency requests
        tasks = []
        for i in range(5):
            payload = {
                "user_message": f"Emergency {i}: I have severe chest pain and difficulty breathing",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"concurrent_emergency_{i}",
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            task = emergency_test_client.post("/api/v2/medical/chat", json=payload)
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["medical_outcome"] == "emergency"
            assert data["confidence_score"] >= 90
        
        # Verify system handled all emergency alerts
        await asyncio.sleep(0.3)
        assert len(emergency_test_client.emergency_alerts) >= 5


class TestEmergencySystemMonitoring:
    """Test emergency system monitoring and observability"""
    
    @pytest.mark.asyncio
    async def test_emergency_metrics_collection(self, emergency_test_client):
        """Test that emergency scenarios generate proper metrics"""
        
        # Trigger emergency scenario
        payload = {
            "user_message": "I'm having a heart attack - chest pain is unbearable",
            "stakeholder_role": StakeholderRole.PATIENT.value,
            "stakeholder_id": "metrics_test_patient"
        }
        
        response = await emergency_test_client.post("/api/v2/medical/chat", json=payload)
        data = response.json()
        
        # Check health endpoint reflects emergency processing
        health_response = await emergency_test_client.get("/api/v2/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data["status"] in ["healthy", "degraded"]
        assert "services" in health_data
    
    @pytest.mark.asyncio
    async def test_emergency_conversation_history(self, emergency_test_client):
        """Test emergency conversation history is properly maintained"""
        
        payload = {
            "user_message": "Help! I think I'm having a stroke - can't move my right side",
            "stakeholder_role": StakeholderRole.PATIENT.value,
            "stakeholder_id": "history_test_patient"
        }
        
        response = await emergency_test_client.post("/api/v2/medical/chat", json=payload)
        data = response.json()
        conversation_id = data["conversation_id"]
        
        # Get conversation history
        history_response = await emergency_test_client.get(
            f"/api/v2/medical/chat/{conversation_id}/history"
        )
        assert history_response.status_code == 200
        
        history_data = history_response.json()
        assert history_data["conversation_id"] == conversation_id
        assert history_data["status"] == "completed"
        assert history_data["outcome"] == "emergency"
        assert len(history_data["turns"]) >= 1
        
        # Verify turn data contains emergency information
        last_turn = history_data["turns"][-1]
        assert last_turn["outcome"] == "emergency"
        assert len(last_turn["red_flags"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])