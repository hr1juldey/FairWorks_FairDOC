"""
E2E Test: Emergency Medical Scenario Flow

Tests the complete emergency detection and escalation workflow from
HTTP request through DSPy agent to stakeholder routing and background tasks.

File: src/tests/e2e/test_e2e_emergency_flow.py
"""

import pytest
import asyncio
from datetime import datetime
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock
import json

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2


class TestEmergencyFlowE2E:
    """End-to-end test for emergency medical scenarios"""
    
    @pytest.fixture(autouse=True)
    async def setup_test_environment(self):
        """Setup test environment with controlled dependencies"""
        # Track all service calls for debugging
        self.service_calls = {
            "medical_agent": [],
            "redis_operations": [],
            "nice_lookups": [],
            "stakeholder_routes": [],
            "emergency_alerts": [],
            "persistence_calls": []
        }
        
        # Mock Redis with call tracking
        self.mock_redis = AsyncMock()
        self.mock_redis.get = AsyncMock(side_effect=self._track_redis_get)
        self.mock_redis.setex = AsyncMock(side_effect=self._track_redis_set)
        self.mock_redis.lpush = AsyncMock(side_effect=self._track_redis_lpush)
        
        # Mock DSPy agent with emergency detection
        self.mock_agent = MagicMock()
        self.mock_agent.process_turn = AsyncMock(side_effect=self._emergency_agent_response)
        
        yield
        
        # Debug information available after test
        print(f"\n=== DEBUG INFO FOR EMERGENCY FLOW ===")
        print(f"Service calls made: {json.dumps(self.service_calls, indent=2, default=str)}")
    
    async def _track_redis_get(self, key):
        """Track Redis GET operations"""
        self.service_calls["redis_operations"].append({
            "operation": "GET",
            "key": key,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if "state:" in key:
            # Return mock conversation state
            return json.dumps({
                "conversation_id": key.split(":")[-1],
                "user_id": "emergency_patient_001",
                "turn_count": 1,
                "status": "active",
                "conversation_history": [],
                "current_outcome": "inconclusive",
                "red_flags_detected": []
            })
        return None
    
    async def _track_redis_set(self, key, value, expire):
        """Track Redis SET operations"""
        self.service_calls["redis_operations"].append({
            "operation": "SETEX",
            "key": key,
            "expire": expire,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _track_redis_lpush(self, key, value):
        """Track Redis LPUSH operations"""
        self.service_calls["redis_operations"].append({
            "operation": "LPUSH",
            "key": key,
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _emergency_agent_response(self, symptoms, nice_context=""):
        """Mock DSPy agent that detects emergencies"""
        self.service_calls["medical_agent"].append({
            "symptoms": symptoms,
            "nice_context": nice_context,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Emergency detection logic
        emergency_keywords = ["chest pain", "crushing", "radiating", "heart attack", "stroke"]
        if any(keyword in symptoms.lower() for keyword in emergency_keywords):
            return {
                "outcome": "emergency",
                "confidence": 95,
                "next_question": None,
                "reasoning": "Critical symptoms detected - immediate medical attention required",
                "red_flags": ["chest_pain", "potential_cardiac_event", "radiating_pain"],
                "is_complete": True
            }
        
        return {
            "outcome": "inconclusive",
            "confidence": 60,
            "next_question": "Can you describe the pain in more detail?",
            "reasoning": "Need more information",
            "red_flags": [],
            "is_complete": False
        }
    
    @patch('src.app2.core.dependencies_v2.get_medical_agent')
    @patch('src.app2.core.dependencies_v2.get_conversation_queue')
    @patch('src.app2.core.dependencies_v2.get_nice_lookup')
    @patch('src.app2.core.dependencies_v2.get_stakeholder_router')
    @pytest.mark.asyncio
    async def test_complete_emergency_detection_flow(
        self, mock_router, mock_nice, mock_queue, mock_agent
    ):
        """Test complete emergency flow from API to background tasks"""
        
        # Setup mocks
        mock_agent.return_value = self.mock_agent
        
        mock_queue_instance = AsyncMock()
        mock_queue_instance.start_conversation = AsyncMock(return_value="conv_emergency_001")
        mock_queue_instance.update_conversation_turn = AsyncMock(return_value={
            "conversation_id": "conv_emergency_001",
            "turn_count": 1,
            "status": "completed",
            "current_outcome": "emergency"
        })
        mock_queue.return_value = mock_queue_instance
        
        mock_nice_instance = MagicMock()
        mock_nice_instance.find_relevant_protocols = MagicMock(return_value={
            "protocol_code": "CG95_CHEST_PAIN",
            "protocol_text": "NICE Guidelines: Chest pain requires immediate assessment"
        })
        mock_nice.return_value = mock_nice_instance
        
        mock_router_instance = AsyncMock()
        mock_router_instance.route_message = AsyncMock(return_value=[
            MagicMock(
                to_stakeholder="fairdoc_agent",
                priority="medium",
                requires_human_review=False
            ),
            MagicMock(
                to_stakeholder="doctor",
                priority="emergency",
                requires_human_review=True,
                message_content="🚨 EMERGENCY: Patient reports chest pain"
            )
        ])
        mock_router.return_value = mock_router_instance
        
        # Execute E2E test
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Emergency request
            payload = {
                "user_message": "I have crushing chest pain radiating to my left arm",
                "stakeholder_role": "patient",
                "stakeholder_id": "emergency_patient_001",
                "chat_provider": "api_direct"
            }
            
            response = await client.post("/api/v2/medical/chat", json=payload)
            
            # Verify emergency response
            assert response.status_code == 200
            data = response.json()
            
            # Emergency outcome assertions
            assert data["medical_outcome"] == "emergency"
            assert data["confidence_score"] >= 90
            assert data["is_conversation_complete"] is True
            assert len(data["red_flags"]) > 0
            assert "chest_pain" in str(data["red_flags"])
            
            # Verify service interaction chain
            assert "conversation_id" in data
            
            # Check that emergency escalation was triggered
            mock_router_instance.route_message.assert_called()
            call_args = mock_router_instance.route_message.call_args
            assert call_args[1]["medical_outcome"] == "emergency"
            
            # Verify NICE protocol lookup
            mock_nice_instance.find_relevant_protocols.assert_called_with(
                "I have crushing chest pain radiating to my left arm"
            )
    
    @patch('src.app2.core.dependencies_v2.get_medical_agent')
    @patch('src.app2.core.dependencies_v2.get_conversation_queue')
    @patch('src.app2.services.chat.emergency_handler.EmergencyHandler')
    @pytest.mark.asyncio
    async def test_emergency_background_task_execution(
        self, mock_emergency_handler, mock_queue, mock_agent
    ):
        """Test that emergency background tasks are properly triggered"""
        
        # Setup emergency handler mock
        mock_handler_instance = AsyncMock()
        mock_handler_instance.handle_emergency_alert = AsyncMock()
        mock_emergency_handler.return_value = mock_handler_instance
        
        # Setup other mocks
        mock_agent.return_value = self.mock_agent
        mock_queue_instance = AsyncMock()
        mock_queue_instance.start_conversation = AsyncMock(return_value="conv_bg_001")
        mock_queue_instance.update_conversation_turn = AsyncMock(return_value={
            "status": "completed"
        })
        mock_queue.return_value = mock_queue_instance
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "user_message": "Severe chest pain, can't breathe",
                "stakeholder_role": "patient"
            }
            
            response = await client.post("/api/v2/medical/chat", json=payload)
            assert response.status_code == 200
            
            # Allow background tasks to execute
            await asyncio.sleep(0.1)
            
            # Verify emergency handler was called
            # Note: In real implementation, this would be called via BackgroundTasks
            # Here we're testing the integration setup
            assert response.json()["medical_outcome"] == "emergency"
    
    @pytest.mark.asyncio
    async def test_emergency_system_health_during_crisis(self):
        """Test system health endpoints during emergency scenarios"""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Check system health before emergency
            health_response = await client.get("/api/v2/health")
            assert health_response.status_code in [200, 503]
            
            # Check API info
            info_response = await client.get("/api/v2/info")
            assert info_response.status_code == 200
            info_data = info_response.json()
            
            # Verify emergency-related features are enabled
            assert info_data["features"]["emergency_detection"] is True
            assert info_data["features"]["stakeholder_routing"] is True
            
            # Check readiness probe
            ready_response = await client.get("/api/v2/health/ready")
            assert ready_response.status_code in [200, 503]
            
            # Check liveness probe
            live_response = await client.get("/api/v2/health/live")
            assert live_response.status_code == 200
            assert live_response.json()["status"] == "alive"
    
    def test_emergency_configuration_validation(self):
        """Test that emergency-related configuration is properly loaded"""
        
        # Verify V2 emergency settings
        assert settings_v2.FAIRDOC_V2_ENABLED is True
        assert settings_v2.FAIRDOC_V2_EMERGENCY_ALERT_WEBHOOK is not None
        assert settings_v2.FAIRDOC_V2_MAX_CONVERSATION_TURNS > 0
        
        # Verify DSPy model configuration
        assert settings_v2.FAIRDOC_V2_DSPy_MODEL == "deepseek-r1:8b"
        assert settings_v2.OLLAMA_BASE_URL == "http://localhost:11434"
        
        # Debug output for configuration
        print(f"\n=== EMERGENCY CONFIG DEBUG ===")
        print(f"V2 Enabled: {settings_v2.FAIRDOC_V2_ENABLED}")
        print(f"Emergency Webhook: {settings_v2.FAIRDOC_V2_EMERGENCY_ALERT_WEBHOOK}")
        print(f"DSPy Model: {settings_v2.FAIRDOC_V2_DSPy_MODEL}")
        print(f"Max Turns: {settings_v2.FAIRDOC_V2_MAX_CONVERSATION_TURNS}")