"""
E2E Test: Complete Patient Medical Triage Flow

Tests the full patient journey from initial symptoms to final outcome,
including multi-turn conversations, NICE protocol integration, and
Redis state management.

File: src/tests/e2e/test_e2e_patient_flow.py
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock

from src.app2.main_v2 import app
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider


class TestE2EPatientFlow:
    """End-to-end patient medical triage flow testing"""
    
    @pytest.fixture
    async def test_client(self):
        """Create test client with app2"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def mock_services(self):
        """Mock external services for E2E testing"""
        
        # Mock DSPy responses for different scenarios
        async def mock_dspy_process(symptoms: str, nice_context: str = ""):
            if "headache" in symptoms.lower():
                if "severe" in symptoms.lower() or "worst" in symptoms.lower():
                    return {
                        "outcome": "emergency",
                        "confidence": 90,
                        "next_question": None,
                        "reasoning": "Sudden severe headache - possible subarachnoid hemorrhage",
                        "red_flags": ["sudden_severe_headache", "neurological_emergency"],
                        "is_complete": True
                    }
                else:
                    return {
                        "outcome": "inconclusive", 
                        "confidence": 65,
                        "next_question": "Is this the worst headache you've ever experienced?",
                        "reasoning": "Need to assess headache severity and characteristics",
                        "red_flags": [],
                        "is_complete": False
                    }
            elif "chest pain" in symptoms.lower():
                return {
                    "outcome": "emergency",
                    "confidence": 95,
                    "next_question": None,
                    "reasoning": "Chest pain requires immediate cardiac evaluation",
                    "red_flags": ["chest_pain", "potential_cardiac_event"],
                    "is_complete": True
                }
            else:
                return {
                    "outcome": "routine_doctor",
                    "confidence": 75,
                    "next_question": None,
                    "reasoning": "Symptoms warrant routine medical evaluation",
                    "red_flags": [],
                    "is_complete": True
                }
        
        # Mock Redis for conversation state
        fake_redis_store = {}
        
        async def mock_redis_set(key: str, value: str, ex: int = None):
            fake_redis_store[key] = value
            return True
            
        async def mock_redis_get(key: str):
            return fake_redis_store.get(key)
            
        async def mock_redis_ping():
            return True
            
        mock_redis = AsyncMock()
        mock_redis.set = mock_redis_set
        mock_redis.get = mock_redis_get
        mock_redis.ping = mock_redis_ping
        mock_redis.lpush = AsyncMock(return_value=1)
        mock_redis.lrem = AsyncMock(return_value=1)
        
        return {
            'dspy_process': mock_dspy_process,
            'redis': mock_redis,
            'redis_store': fake_redis_store
        }
    
    @pytest.mark.asyncio
    async def test_complete_headache_conversation_flow(self, test_client, mock_services):
        """Test complete headache triage conversation with multi-turn flow"""
        
        with patch('src.app2.services.dspy.medical_agent.MedicalTriageAgent.process_turn', 
                   side_effect=mock_services['dspy_process']), \
             patch('src.app2.services.context.redis_queue.Redis.from_url', 
                   return_value=mock_services['redis']):
            
            # Step 1: Initial headache complaint
            initial_payload = {
                "user_message": "I have a headache that started this morning",
                "stakeholder_role": "patient", 
                "stakeholder_id": "patient_e2e_001",
                "chat_provider": "api_direct"
            }
            
            response1 = await test_client.post("/api/v2/medical/chat", json=initial_payload)
            
            assert response1.status_code == 200
            data1 = response1.json()
            
            # Verify initial response structure
            assert "conversation_id" in data1
            assert data1["medical_outcome"] == "need_more_questions"
            assert data1["next_question"] is not None
            assert "worst headache" in data1["next_question"].lower()
            assert data1["confidence_score"] == 65
            assert data1["turn_count"] == 1
            assert not data1["is_conversation_complete"]
            
            conversation_id = data1["conversation_id"]
            
            # Step 2: Follow-up response indicating severity
            followup_payload = {
                "conversation_id": conversation_id,
                "user_message": "Yes, this is definitely the worst headache I've ever had. It came on suddenly while I was exercising.",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_e2e_001"
            }
            
            response2 = await test_client.post("/api/v2/medical/chat", json=followup_payload)
            
            assert response2.status_code == 200
            data2 = response2.json()
            
            # Verify escalation to emergency
            assert data2["conversation_id"] == conversation_id
            assert data2["medical_outcome"] == "emergency"
            assert data2["confidence_score"] == 90
            assert data2["is_conversation_complete"] is True
            assert data2["turn_count"] == 2
            assert len(data2["red_flags"]) > 0
            assert "sudden_severe_headache" in data2["red_flags"]
            
            # Step 3: Verify conversation state persistence
            state_response = await test_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            assert state_response.status_code == 200
            
            state_data = state_response.json()
            assert state_data["turn_count"] == 2
            assert state_data["status"] in ["completed", "active"]
            
            # Step 4: Verify conversation history
            history_response = await test_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert len(history_data["turns"]) >= 1
            assert history_data["outcome"] == "emergency"
    
    @pytest.mark.asyncio
    async def test_chest_pain_immediate_emergency(self, test_client, mock_services):
        """Test immediate emergency detection for chest pain"""
        
        with patch('src.app2.services.dspy.medical_agent.MedicalTriageAgent.process_turn',
                   side_effect=mock_services['dspy_process']), \
             patch('src.app2.services.context.redis_queue.Redis.from_url',
                   return_value=mock_services['redis']):
            
            payload = {
                "user_message": "I'm having severe crushing chest pain that radiates to my left arm",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_e2e_002", 
                "chat_provider": "api_direct"
            }
            
            response = await test_client.post("/api/v2/medical/chat", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should immediately classify as emergency
            assert data["medical_outcome"] == "emergency"
            assert data["confidence_score"] == 95
            assert data["is_conversation_complete"] is True
            assert data["turn_count"] == 1
            assert "chest_pain" in data["red_flags"]
            assert "potential_cardiac_event" in data["red_flags"]
    
    @pytest.mark.asyncio  
    async def test_routine_symptoms_flow(self, test_client, mock_services):
        """Test routine symptoms that warrant doctor consultation"""
        
        with patch('src.app2.services.dspy.medical_agent.MedicalTriageAgent.process_turn',
                   side_effect=mock_services['dspy_process']), \
             patch('src.app2.services.context.redis_queue.Redis.from_url',
                   return_value=mock_services['redis']):
            
            payload = {
                "user_message": "I've been having some joint pain and stiffness for the past week",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_e2e_003",
                "chat_provider": "api_direct"
            }
            
            response = await test_client.post("/api/v2/medical/chat", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["medical_outcome"] == "routine_doctor_consultation"
            assert data["confidence_score"] == 75
            assert data["is_conversation_complete"] is True
            assert len(data["red_flags"]) == 0
    
    @pytest.mark.asyncio
    async def test_conversation_reset_functionality(self, test_client, mock_services):
        """Test conversation reset functionality"""
        
        with patch('src.app2.services.dspy.medical_agent.MedicalTriageAgent.process_turn',
                   side_effect=mock_services['dspy_process']), \
             patch('src.app2.services.context.redis_queue.Redis.from_url',
                   return_value=mock_services['redis']):
            
            # Start conversation
            payload = {
                "user_message": "I have a mild headache",
                "stakeholder_role": "patient",
                "stakeholder_id": "patient_e2e_004"
            }
            
            response = await test_client.post("/api/v2/medical/chat", json=payload)
            conversation_id = response.json()["conversation_id"]
            
            # Reset conversation
            reset_response = await test_client.post(f"/api/v2/medical/chat/{conversation_id}/reset")
            assert reset_response.status_code == 200
            
            reset_data = reset_response.json()
            assert "reset successfully" in reset_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_conversation(self, test_client):
        """Test error handling for invalid conversation ID"""
        
        invalid_id = "conv_invalid_12345"
        
        # Test invalid conversation state request
        response = await test_client.get(f"/api/v2/medical/chat/{invalid_id}/state")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "not found" in error_data["detail"].lower()
        
        # Test invalid conversation history request  
        history_response = await test_client.get(f"/api/v2/medical/chat/{invalid_id}/history")
        assert history_response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, test_client):
        """Test system health check endpoint"""
        
        response = await test_client.get("/api/v2/health")
        
        # Should respond regardless of service status
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "timestamp" in data
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_api_info_endpoint(self, test_client):
        """Test API information endpoint"""
        
        response = await test_client.get("/api/v2/info")
        assert response.status_code == 200
        
        data = response.json() 
        assert data["name"] == "Fairdoc AI Triage System V2"
        assert data["version"] == "v2.6-stable"
        assert "endpoints" in data
        assert "features" in data
        
        # Verify key features are listed
        features = data["features"]
        assert features["multi_turn_conversations"] is True
        assert features["nice_protocol_integration"] is True
        assert features["emergency_detection"] is True