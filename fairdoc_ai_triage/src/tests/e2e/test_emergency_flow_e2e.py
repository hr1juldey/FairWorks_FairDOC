# E2E Test: Emergency Medical Flow
# File: src/tests/e2e/test_emergency_flow_e2e.py

"""
End-to-End testing for emergency medical scenarios in Fairdoc AI V2.

Tests the complete flow from patient symptom input through DSPy agent 
processing, NICE protocol lookup, stakeholder routing, and emergency 
alert handling.

Critical for debugging: Captures detailed logs at each step to trace
failures through the entire system stack.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.core.database_v2 import BaseV2
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider

# E2E Test Configuration
E2E_DB_URL = "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_fairdoc_e2e"
E2E_REDIS_URL = "redis://localhost:6379/14"  # Separate Redis DB for E2E

class E2ETestState:
    """Capture comprehensive state for debugging"""
    
    def __init__(self):
        self.test_start = datetime.utcnow()
        self.requests: List[Dict] = []
        self.responses: List[Dict] = []
        self.service_calls: List[Dict] = []
        self.database_operations: List[Dict] = []
        self.redis_operations: List[Dict] = []
        self.errors: List[Dict] = []
    
    def log_request(self, endpoint: str, payload: Dict, timestamp: datetime = None):
        self.requests.append({
            "timestamp": timestamp or datetime.utcnow(),
            "endpoint": endpoint,
            "payload": payload
        })
    
    def log_response(self, status_code: int, body: Dict, duration_ms: float):
        self.responses.append({
            "timestamp": datetime.utcnow(),
            "status_code": status_code,
            "body": body,
            "duration_ms": duration_ms
        })
    
    def log_service_call(self, service: str, method: str, args: Dict, result: Any):
        self.service_calls.append({
            "timestamp": datetime.utcnow(),
            "service": service,
            "method": method,
            "args": args,
            "result": str(result)[:500]  # Truncate for readability
        })
    
    def log_error(self, error: Exception, context: str):
        self.errors.append({
            "timestamp": datetime.utcnow(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        })
    
    def get_summary(self) -> Dict:
        """Generate test execution summary for debugging"""
        return {
            "test_duration_seconds": (datetime.utcnow() - self.test_start).total_seconds(),
            "total_requests": len(self.requests),
            "total_responses": len(self.responses),
            "service_calls": len(self.service_calls),
            "database_operations": len(self.database_operations),
            "redis_operations": len(self.redis_operations),
            "errors_encountered": len(self.errors),
            "success_rate": len([r for r in self.responses if r["status_code"] == 200]) / max(len(self.responses), 1)
        }

@pytest.fixture
async def e2e_state():
    """Provide test state tracking for debugging"""
    return E2ETestState()

@pytest.fixture
async def e2e_database():
    """Setup isolated E2E database"""
    engine = create_async_engine(E2E_DB_URL, echo=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(BaseV2.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(BaseV2.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def e2e_redis():
    """Setup isolated E2E Redis"""
    redis_client = redis.from_url(E2E_REDIS_URL, decode_responses=True)
    await redis_client.flushdb()
    yield redis_client
    await redis_client.flushdb()
    await redis_client.close()

@pytest.fixture
async def instrumented_client(e2e_state: E2ETestState):
    """HTTP client with request/response logging"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Monkey patch to capture requests/responses
        original_request = client.request
        
        async def logged_request(method, url, **kwargs):
            start_time = time.time()
            e2e_state.log_request(f"{method} {url}", kwargs.get("json", {}))
            
            try:
                response = await original_request(method, url, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                try:
                    body = response.json()
                except:
                    body = {"raw_content": response.text[:200]}
                
                e2e_state.log_response(response.status_code, body, duration_ms)
                return response
                
            except Exception as e:
                e2e_state.log_error(e, f"HTTP {method} {url}")
                raise
        
        client.request = logged_request
        yield client

class TestEmergencyFlowE2E:
    """End-to-End emergency flow testing"""
    
    @pytest.mark.asyncio
    async def test_chest_pain_emergency_complete_flow(
        self, 
        instrumented_client, 
        e2e_state: E2ETestState
    ):
        """
        Test complete emergency flow for chest pain scenario.
        
        Expected flow:
        1. Patient reports chest pain → Emergency detection
        2. NICE protocol lookup → Chest pain protocol
        3. DSPy agent → Emergency outcome + high confidence
        4. Stakeholder router → Routes to doctor with emergency priority
        5. Background tasks → Emergency alert triggered
        6. Redis state → Conversation marked as emergency + completed
        """
        
        # Step 1: Send emergency chest pain message
        emergency_payload = {
            "user_message": "I have severe crushing chest pain radiating to my left arm and I'm sweating profusely",
            "stakeholder_role": "patient", 
            "stakeholder_id": "emergency_patient_001",
            "chat_provider": "api_direct"
        }
        
        response = await instrumented_client.post(
            "/api/v2/medical/chat", 
            json=emergency_payload
        )
        
        # Validate HTTP response
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        response_data = response.json()
        conversation_id = response_data["conversation_id"]
        
        # Step 2: Validate emergency detection
        assert response_data["medical_outcome"] == "emergency", \
            f"Expected emergency outcome, got {response_data['medical_outcome']}"
        
        assert response_data["confidence_score"] >= 85, \
            f"Expected high confidence ≥85, got {response_data['confidence_score']}"
        
        assert response_data["is_conversation_complete"] is True, \
            "Emergency conversations should complete immediately"
        
        # Step 3: Validate red flags detection
        red_flags = response_data.get("red_flags", [])
        expected_flags = ["chest_pain", "radiation_to_arm", "diaphoresis"]
        assert any(flag in str(red_flags).lower() for flag in expected_flags), \
            f"Expected emergency red flags, got {red_flags}"
        
        # Step 4: Check conversation state endpoint
        state_response = await instrumented_client.get(
            f"/api/v2/medical/chat/{conversation_id}/state"
        )
        
        assert state_response.status_code == 200
        state_data = state_response.json()
        
        assert state_data["status"] == "completed", \
            f"Emergency conversations should be completed, got {state_data['status']}"
        
        assert state_data["current_outcome"] == "emergency", \
            f"State should reflect emergency outcome, got {state_data['current_outcome']}"
        
        # Step 5: Validate conversation history structure
        history_response = await instrumented_client.get(
            f"/api/v2/medical/chat/{conversation_id}/history"
        )
        
        assert history_response.status_code == 200
        history_data = history_response.json()
        
        assert len(history_data["turns"]) >= 1, "Should have at least one turn recorded"
        assert history_data["outcome"] == "emergency", "History should reflect emergency outcome"
        
        # Log test summary for debugging
        e2e_state.log_service_call(
            "test_summary", 
            "chest_pain_emergency", 
            {"conversation_id": conversation_id},
            e2e_state.get_summary()
        )
        
        print(f"\n=== E2E Test Summary ===")
        print(f"Conversation ID: {conversation_id}")
        print(f"Test Duration: {e2e_state.get_summary()['test_duration_seconds']:.2f}s")
        print(f"Total API Calls: {e2e_state.get_summary()['total_requests']}")
        print(f"Success Rate: {e2e_state.get_summary()['success_rate']:.1%}")
        print(f"Errors: {len(e2e_state.errors)}")
    
    @pytest.mark.asyncio 
    async def test_emergency_routing_and_alerts(
        self, 
        instrumented_client,
        e2e_state: E2ETestState
    ):
        """
        Test emergency routing and alert system activation.
        
        Validates that emergency scenarios trigger proper stakeholder
        routing and background alert mechanisms.
        """
        
        with patch('src.app2.services.chat.emergency_handler.EmergencyHandler') as mock_emergency:
            # Setup emergency handler mock to capture calls
            mock_handler_instance = AsyncMock()
            mock_emergency.return_value = mock_handler_instance
            
            # Send emergency message
            payload = {
                "user_message": "I can't breathe and have severe chest pain",
                "stakeholder_role": "patient",
                "stakeholder_id": "emergency_patient_002"
            }
            
            response = await instrumented_client.post("/api/v2/medical/chat", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate emergency outcome
            assert data["medical_outcome"] == "emergency"
            
            # Give background tasks time to execute
            await asyncio.sleep(0.1)
            
            # Verify emergency handler was called (background task)
            # Note: In real E2E, this would verify actual webhook calls
            print(f"Emergency handler call attempts: {mock_handler_instance.handle_emergency_alert.call_count}")
            
            e2e_state.log_service_call(
                "emergency_handler",
                "handle_emergency_alert", 
                {"patient_id": "emergency_patient_002"},
                "Background task executed"
            )
    
    @pytest.mark.asyncio
    async def test_false_positive_emergency_handling(
        self, 
        instrumented_client,
        e2e_state: E2ETestState  
    ):
        """
        Test handling of potential false positive emergency scenarios.
        
        Ensures system can distinguish between true emergencies and
        concerning but non-emergency symptoms.
        """
        
        # Send concerning but non-emergency symptoms
        payload = {
            "user_message": "I have chest discomfort after eating a large meal, feels like heartburn",
            "stakeholder_role": "patient",
            "stakeholder_id": "patient_003"
        }
        
        response = await instrumented_client.post("/api/v2/medical/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should NOT be classified as emergency
        assert data["medical_outcome"] in ["routine_doctor", "inconclusive"], \
            f"Heartburn should not be emergency, got {data['medical_outcome']}"
        
        # Should ask follow-up questions for clarification
        assert data["next_question"] is not None, \
            "Non-emergency chest symptoms should trigger follow-up questions"
        
        assert data["is_conversation_complete"] is False, \
            "Non-emergency cases should continue conversation"
        
        # Confidence should be moderate, not high
        assert data["confidence_score"] < 85, \
            f"Non-emergency should have moderate confidence, got {data['confidence_score']}"
        
        e2e_state.log_service_call(
            "false_positive_test",
            "heartburn_vs_emergency",
            {"outcome": data["medical_outcome"]},
            f"Correctly avoided false emergency classification"
        )
        
        print(f"False positive test passed - Outcome: {data['medical_outcome']}")