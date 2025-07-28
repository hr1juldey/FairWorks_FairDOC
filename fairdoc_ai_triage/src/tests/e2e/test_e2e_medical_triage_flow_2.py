"""
End-to-End Test: Complete Medical Triage Flow

Tests the full patient journey from initial symptom report through
final medical outcome, including multi-turn conversations, NICE protocol
integration, and proper state management.

Path: src/tests/e2e/test_e2e_medical_triage_flow.py
"""

import pytest
import asyncio
import json
from datetime import datetime
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock

from src.app2.main_v2 import app
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider


@pytest.fixture
async def test_client():
    """Create test client for FastAPI app"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_services():
    """Mock all external services for deterministic testing"""
    
    # Mock DSPy Medical Agent responses
    class MockMedicalAgent:
        def __init__(self):
            self.turn_count = 0
            self.conversation_state = "active"
        
        async def process_turn(self, symptoms, nice_context=""):
            self.turn_count += 1
            
            # Headache progression logic
            if "headache" in symptoms.lower():
                if self.turn_count == 1:
                    return {
                        "outcome": "inconclusive",
                        "confidence": 65,
                        "next_question": "How long have you had this headache?",
                        "reasoning": "Need duration to assess severity",
                        "red_flags": [],
                        "is_complete": False
                    }
                elif "sudden" in symptoms.lower() or "worst ever" in symptoms.lower():
                    return {
                        "outcome": "emergency",
                        "confidence": 90,
                        "next_question": None,
                        "reasoning": "Sudden severe headache - possible SAH",
                        "red_flags": ["sudden_onset", "severe_headache"],
                        "is_complete": True
                    }
                else:
                    return {
                        "outcome": "routine_doctor",
                        "confidence": 75,
                        "next_question": None,
                        "reasoning": "Persistent headache requires medical evaluation",
                        "red_flags": [],
                        "is_complete": True
                    }
        
        def reset_conversation(self):
            self.turn_count = 0
            self.conversation_state = "active"
    
    # Mock Redis Conversation Queue
    class MockConversationQueue:
        def __init__(self):
            self.conversations = {}
            self.next_id = 1
        
        async def initialize(self):
            pass
        
        async def start_conversation(self, user_id, initial_symptoms):
            conv_id = f"conv_{user_id}_{self.next_id}"
            self.next_id += 1
            
            self.conversations[conv_id] = {
                "conversation_id": conv_id,
                "user_id": user_id,
                "initial_symptoms": initial_symptoms,
                "turn_count": 1,
                "status": "active",
                "current_outcome": "inconclusive",
                "conversation_history": [],
                "red_flags_detected": [],
                "created_at": datetime.utcnow().isoformat()
            }
            return conv_id
        
        async def update_conversation_turn(self, conversation_id, user_response, agent_result):
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            conv = self.conversations[conversation_id]
            conv["turn_count"] += 1
            conv["current_outcome"] = agent_result["outcome"]
            
            # Add turn to history
            turn = {
                "turn": conv["turn_count"] - 1,
                "user_response": user_response,
                "agent_question": agent_result.get("next_question"),
                "outcome": agent_result["outcome"],
                "confidence": agent_result["confidence"],
                "red_flags": agent_result.get("red_flags", [])
            }
            conv["conversation_history"].append(turn)
            
            # Update red flags
            for flag in agent_result.get("red_flags", []):
                if flag not in conv["red_flags_detected"]:
                    conv["red_flags_detected"].append(flag)
            
            # Mark as completed if agent says so
            if agent_result.get("is_complete"):
                conv["status"] = "completed"
            
            return conv
        
        async def get_conversation_state(self, conversation_id):
            return self.conversations.get(conversation_id)
    
    # Mock NICE Lookup
    class MockNICELookup:
        def find_relevant_protocols(self, symptoms):
            if "headache" in symptoms.lower():
                return {
                    "protocol_code": "NG127_HEADACHE",
                    "protocol_text": "Condition: Headache\nImmediate red flags: sudden onset, worst headache ever\nRoutine: persistent headache >2 weeks"
                }
            return {"protocol_code": "NONE", "protocol_text": ""}
    
    # Mock Stakeholder Router
    class MockStakeholderRouter:
        async def route_message(self, conversation_id, from_stakeholder, message, medical_outcome=None):
            routes = []
            
            # Basic routing logic
            if from_stakeholder == "patient":
                routes.append(MockRoute("patient", "fairdoc_agent", message))
                
                if medical_outcome == "emergency":
                    routes.append(MockRoute("fairdoc_agent", "doctor", f"🚨 EMERGENCY: {message}", "emergency"))
                elif medical_outcome == "routine_doctor":
                    routes.append(MockRoute("fairdoc_agent", "doctor", f"📋 Consultation needed: {message}"))
            
            return routes
    
    class MockRoute:
        def __init__(self, from_stakeholder, to_stakeholder, message, priority="medium"):
            self.from_stakeholder = from_stakeholder
            self.to_stakeholder = to_stakeholder
            self.message_content = message
            self.priority = priority
            self.requires_human_review = priority == "emergency"
    
    return {
        "medical_agent": MockMedicalAgent(),
        "conversation_queue": MockConversationQueue(),
        "nice_lookup": MockNICELookup(),
        "stakeholder_router": MockStakeholderRouter()
    }


class TestCompleteTriageFlow:
    """Test complete medical triage scenarios end-to-end"""
    
    @pytest.mark.asyncio
    async def test_headache_to_routine_doctor_flow(self, test_client, mock_services):
        """Test: Patient with headache → Follow-up questions → Routine doctor recommendation"""
        
        # Patch dependencies
        with patch('src.app2.core.dependencies_v2.get_medical_agent', return_value=mock_services["medical_agent"]), \
             patch('src.app2.core.dependencies_v2.get_conversation_queue', return_value=mock_services["conversation_queue"]), \
             patch('src.app2.core.dependencies_v2.get_nice_lookup', return_value=mock_services["nice_lookup"]), \
             patch('src.app2.core.dependencies_v2.get_stakeholder_router', return_value=mock_services["stakeholder_router"]):
            
            # Step 1: Initial symptom report
            initial_payload = {
                "user_message": "I have a headache that won't go away",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_001",
                "chat_provider": "api_direct"
            }
            
            response1 = await test_client.post("/api/v2/medical/chat", json=initial_payload)
            assert response1.status_code == 200
            
            data1 = response1.json()
            conversation_id = data1["conversation_id"]
            
            # Verify initial response
            assert data1["medical_outcome"] == "need_more_questions"
            assert data1["confidence_score"] == 65
            assert "How long have you had this headache?" in data1["agent_message"]
            assert data1["is_conversation_complete"] is False
            assert data1["turn_count"] == 1
            
            # Step 2: Follow-up response
            followup_payload = {
                "conversation_id": conversation_id,
                "user_message": "It's been going on for about a week now",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_001",
                "chat_provider": "api_direct"
            }
            
            response2 = await test_client.post("/api/v2/medical/chat", json=followup_payload)
            assert response2.status_code == 200
            
            data2 = response2.json()
            
            # Verify final outcome
            assert data2["conversation_id"] == conversation_id
            assert data2["medical_outcome"] == "routine_doctor_consultation"
            assert data2["confidence_score"] == 75
            assert data2["is_conversation_complete"] is True
            assert data2["turn_count"] == 2
            assert "medical evaluation" in data2["agent_message"]
            
            # Step 3: Verify conversation state
            state_response = await test_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            assert state_response.status_code == 200
            
            state_data = state_response.json()
            assert state_data["status"] == "completed"
            assert state_data["turn_count"] == 2
            assert state_data["current_outcome"] == "routine_doctor"
    
    @pytest.mark.asyncio
    async def test_headache_to_emergency_flow(self, test_client, mock_services):
        """Test: Patient with sudden severe headache → Emergency escalation"""
        
        with patch('src.app2.core.dependencies_v2.get_medical_agent', return_value=mock_services["medical_agent"]), \
             patch('src.app2.core.dependencies_v2.get_conversation_queue', return_value=mock_services["conversation_queue"]), \
             patch('src.app2.core.dependencies_v2.get_nice_lookup', return_value=mock_services["nice_lookup"]), \
             patch('src.app2.core.dependencies_v2.get_stakeholder_router', return_value=mock_services["stakeholder_router"]):
            
            # Step 1: Initial headache report
            response1 = await test_client.post("/api/v2/medical/chat", json={
                "user_message": "I have a headache",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_002"
            })
            
            data1 = response1.json()
            conversation_id = data1["conversation_id"]
            
            # Step 2: Report sudden severe onset
            response2 = await test_client.post("/api/v2/medical/chat", json={
                "conversation_id": conversation_id,
                "user_message": "Actually, it came on suddenly and it's the worst headache I've ever had",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_002"
            })
            
            data2 = response2.json()
            
            # Verify emergency detection
            assert data2["medical_outcome"] == "emergency_route_to_doctor"
            assert data2["confidence_score"] == 90
            assert data2["is_conversation_complete"] is True
            assert len(data2["red_flags"]) > 0
            assert "sudden_onset" in data2["red_flags"]
            assert "possible SAH" in data2["reasoning"]
    
    @pytest.mark.asyncio
    async def test_conversation_history_tracking(self, test_client, mock_services):
        """Test: Verify conversation history is properly tracked"""
        
        with patch('src.app2.core.dependencies_v2.get_medical_agent', return_value=mock_services["medical_agent"]), \
             patch('src.app2.core.dependencies_v2.get_conversation_queue', return_value=mock_services["conversation_queue"]), \
             patch('src.app2.core.dependencies_v2.get_nice_lookup', return_value=mock_services["nice_lookup"]), \
             patch('src.app2.core.dependencies_v2.get_stakeholder_router', return_value=mock_services["stakeholder_router"]):
            
            # Create multi-turn conversation
            response1 = await test_client.post("/api/v2/medical/chat", json={
                "user_message": "I have a headache",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_003"
            })
            
            conversation_id = response1.json()["conversation_id"]
            
            await test_client.post("/api/v2/medical/chat", json={
                "conversation_id": conversation_id,
                "user_message": "It's been going on for 3 days",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_003"
            })
            
            # Check conversation history
            history_response = await test_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert len(history_data["turns"]) == 2
            assert history_data["status"] == "completed"
            assert history_data["outcome"] == "routine_doctor"
            
            # Verify turn details
            first_turn = history_data["turns"][0]
            assert first_turn["user_response"] == "It's been going on for 3 days"
            assert first_turn["outcome"] == "routine_doctor"
            assert first_turn["confidence"] == 75


@pytest.mark.asyncio
async def test_api_health_and_info():
    """Test system health and API information endpoints"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        
        # Test health endpoint
        health_response = await client.get("/api/v2/health")
        assert health_response.status_code in [200, 503]  # May be unhealthy without real services
        
        health_data = health_response.json()
        assert "status" in health_data
        assert "services" in health_data
        assert "version" in health_data
        
        # Test API info endpoint
        info_response = await client.get("/api/v2/info")
        assert info_response.status_code == 200
        
        info_data = info_response.json()
        assert info_data["name"] == "Fairdoc AI Triage System V2"
        assert info_data["version"] == "v2.6-stable"
        assert "endpoints" in info_data
        assert "features" in info_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])