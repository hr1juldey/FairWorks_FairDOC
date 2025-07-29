"""
E2E Test: Emergency Medical Detection and Response Flow

Tests the complete emergency detection pipeline from patient input through
DSPy agent processing, emergency classification, stakeholder routing,
and background alert processing.

File: src/tests/e2e/test_e2e_emergency_detection.py
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timezone
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import structlog

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider

logger = structlog.get_logger(__name__)



def utcnow():
    """
    Return timezone-aware UTC datetime
    
    This function replaces the deprecated datetime.utcnow() method
    with the recommended timezone-aware approach using UTC timezone.
    
    Returns:
        datetime: Current UTC time with timezone information
    """
    return datetime.now(timezone.utc)



class TestEmergencyDetectionE2E:
    """Complete emergency detection and response testing"""
    
    @pytest.fixture(autouse=True)
    async def setup_emergency_environment(self):
        """Setup environment with comprehensive debug tracking"""
        self.test_session_id = f"emergency_test_{int(time.time())}"
        self.debug_info = {
            "test_start": utcnow().isoformat(),
            "api_calls": [],
            "service_interactions": [],
            "background_tasks": [],
            "system_metrics": {}
        }
        
        # Mock external services with debug tracking
        self.emergency_alerts = []
        self.webhook_calls = []
        
        async def track_emergency_alert(conv_id, user_id, agent_result):
            alert = {
                "timestamp": utcnow().isoformat(),
                "conversation_id": conv_id,
                "user_id": user_id,
                "red_flags": agent_result.get("red_flags", []),
                "confidence": agent_result.get("confidence", 0),
                "outcome": agent_result.get("outcome")
            }
            self.emergency_alerts.append(alert)
            self.debug_info["background_tasks"].append({"type": "emergency_alert", "data": alert})
            
        async def track_webhook_call(url, **kwargs):
            call = {
                "timestamp": utcnow().isoformat(),
                "url": url,
                "method": kwargs.get("method", "POST"),
                "payload": kwargs.get("json", {})
            }
            self.webhook_calls.append(call)
            self.debug_info["background_tasks"].append({"type": "webhook", "data": call})
            return AsyncMock(status_code=200)
        
        with patch('src.app2.services.chat.emergency_handler.EmergencyHandler.handle_emergency_alert', 
                   side_effect=track_emergency_alert), \
             patch('httpx.AsyncClient.post', side_effect=track_webhook_call):
            yield
    
    @pytest.mark.asyncio
    async def test_cardiac_emergency_immediate_detection(self):
        """Test cardiac emergency symptoms trigger immediate emergency response"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # === PHASE 1: Emergency Symptom Report ===
            emergency_payload = {
                "user_message": "I have crushing chest pain radiating to my left arm, I'm sweating and feel nauseous",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"{self.test_session_id}_cardiac_patient",
                "chat_provider": ChatProvider.API_DIRECT.value,
                "metadata": {
                    "severity": "high",
                    "location": "emergency_department_waiting"
                }
            }
            
            start_time = time.time()
            logger.info("🚨 Starting cardiac emergency detection test", payload=emergency_payload)
            
            response = await client.post("/api/v2/medical/chat", json=emergency_payload)
            response_time = time.time() - start_time
            
            self.debug_info["api_calls"].append({
                "endpoint": "/api/v2/medical/chat",
                "method": "POST",
                "response_time": response_time,
                "status_code": response.status_code,
                "request_payload": emergency_payload
            })
            
            # === PHASE 2: Validate Emergency Response ===
            assert response.status_code == 200, f"API call failed: {response.status_code} - {response.text}"
            
            data = response.json()
            conversation_id = data["conversation_id"]
            
            self.debug_info["api_calls"][-1]["response_data"] = data
            
            # Emergency detection validations
            assert data["medical_outcome"] == "emergency", f"Expected emergency, got: {data['medical_outcome']}"
            assert data["confidence_score"] >= 85, f"Emergency confidence too low: {data['confidence_score']}%"
            assert data["is_conversation_complete"] is True, "Emergency should complete conversation immediately"
            assert response_time < 5.0, f"Emergency response too slow: {response_time:.2f}s"
            
            # Red flags validation
            red_flags = data.get("red_flags", [])
            assert len(red_flags) > 0, "Emergency should have red flags"
            expected_flags = ["chest_pain", "radiation", "nausea", "diaphoresis"]
            flag_matches = [flag for flag in expected_flags if any(flag in rf.lower() for rf in red_flags)]
            assert len(flag_matches) >= 2, f"Expected emergency red flags, got: {red_flags}"
            
            logger.info("✅ Emergency detected successfully", 
                       confidence=data["confidence_score"],
                       red_flags=red_flags,
                       response_time=response_time)
            
            # === PHASE 3: Verify Conversation State ===
            state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            assert state_response.status_code == 200
            
            state_data = state_response.json()
            assert state_data["status"] == "completed", f"Expected completed, got: {state_data['status']}"
            assert state_data["current_outcome"] == "emergency"
            
            self.debug_info["service_interactions"].append({
                "service": "conversation_state",
                "operation": "get_state",
                "conversation_id": conversation_id,
                "result": state_data
            })
            
            # === PHASE 4: Verify Background Task Execution ===
            await asyncio.sleep(0.5)  # Allow background tasks to execute
            
            assert len(self.emergency_alerts) >= 1, "Emergency alert should have been triggered"
            alert = self.emergency_alerts[0]
            assert alert["conversation_id"] == conversation_id
            assert alert["outcome"] == "emergency"
            assert "chest_pain" in str(alert["red_flags"]).lower()
            
            # === PHASE 5: Test System Health During Emergency ===
            health_response = await client.get("/api/v2/health")
            health_data = health_response.json()
            
            self.debug_info["system_metrics"]["health_during_emergency"] = {
                "status_code": health_response.status_code,
                "health_data": health_data,
                "services_available": health_data.get("services", {})
            }
            
            # System should remain responsive
            assert health_response.status_code in [200, 503], "Health endpoint should respond during emergency"
            
            logger.info("🏥 System health verified during emergency processing")
    
    @pytest.mark.asyncio
    async def test_stroke_symptoms_neurological_emergency(self):
        """Test stroke symptoms trigger immediate neurological emergency response"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            stroke_payload = {
                "user_message": "Suddenly I can't speak properly and the right side of my face feels numb",
                "stakeholder_role": StakeholderRole.PATIENT.value,
                "stakeholder_id": f"{self.test_session_id}_stroke_patient",
                "chat_provider": ChatProvider.API_DIRECT.value
            }
            
            response = await client.post("/api/v2/medical/chat", json=stroke_payload)
            data = response.json()
            
            # Neurological emergency should be detected
            assert data["medical_outcome"] == "emergency"
            assert data["confidence_score"] >= 90, "Stroke symptoms should have very high confidence"
            
            # Should have neurological red flags
            red_flags = data.get("red_flags", [])
            neuro_indicators = ["speech", "facial", "neurological", "stroke", "weakness"]
            assert any(indicator in " ".join(red_flags).lower() for indicator in neuro_indicators)
            
            logger.info("🧠 Neurological emergency detected", red_flags=red_flags)
    
    @pytest.mark.asyncio
    async def test_emergency_system_load_handling(self):
        """Test system handles multiple concurrent emergency cases"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            emergency_scenarios = [
                "Severe chest pain with shortness of breath",
                "Sudden severe headache worst of my life",
                "Can't breathe and lips turning blue",
                "Crushing chest pressure radiating to jaw",
                "Sudden weakness on left side of body"
            ]
            
            # Create concurrent emergency requests
            tasks = []
            for i, symptoms in enumerate(emergency_scenarios):
                payload = {
                    "user_message": symptoms,
                    "stakeholder_role": StakeholderRole.PATIENT.value,
                    "stakeholder_id": f"{self.test_session_id}_concurrent_{i}",
                    "chat_provider": ChatProvider.API_DIRECT.value
                }
                task = client.post("/api/v2/medical/chat", json=payload)
                tasks.append((symptoms, task))
            
            # Execute all requests concurrently
            start_time = time.time()
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze concurrent emergency handling
            successful_emergencies = 0
            failed_requests = 0
            
            for i, result in enumerate(results):
                symptom = tasks[i][0]  # Retrieve corresponding symptom
                if isinstance(result, Exception):
                    failed_requests += 1
                    logger.error(f"❌ Request {i} failed for symptom: '{symptom}' with error: {result}")
                    continue
                
                if result.status_code == 200:
                    data = result.json()
                    if data["medical_outcome"] == "emergency":
                        successful_emergencies += 1
            
            self.debug_info["system_metrics"]["concurrent_emergency_test"] = {
                "total_requests": len(emergency_scenarios),
                "successful_emergencies": successful_emergencies,
                "failed_requests": failed_requests,
                "total_time": total_time,
                "avg_time_per_request": total_time / len(emergency_scenarios)
            }
            
            # Performance assertions
            assert successful_emergencies >= len(emergency_scenarios) * 0.8, "80% of emergencies should be detected"
            assert failed_requests <= 1, "At most 1 request should fail under load"
            assert total_time < 10.0, f"Concurrent emergency processing too slow: {total_time:.2f}s"
            
            logger.info("⚡ Concurrent emergency handling test completed",
                       successful=successful_emergencies,
                       failed=failed_requests,
                       total_time=total_time)
    
    def teardown_method(self):
        """Print comprehensive debug information after each test"""
        self.debug_info["test_end"] = utcnow().isoformat()
        self.debug_info["total_emergency_alerts"] = len(self.emergency_alerts)
        self.debug_info["total_webhook_calls"] = len(self.webhook_calls)
        
        print(f"\n{'=' * 80}")
        print("EMERGENCY DETECTION E2E TEST DEBUG REPORT")
        print(f"{'=' * 80}")
        print(f"Session ID: {self.test_session_id}")
        print(f"Test Duration: {self.debug_info['test_start']} → {self.debug_info['test_end']}")
        print(f"API Calls Made: {len(self.debug_info['api_calls'])}")
        print(f"Emergency Alerts Triggered: {len(self.emergency_alerts)}")
        print(f"Background Tasks Executed: {len(self.debug_info['background_tasks'])}")
        
        if self.debug_info["api_calls"]:
            print("\nAPI CALL PERFORMANCE:")
            for call in self.debug_info["api_calls"]:
                print(f"  {call['method']} {call['endpoint']}: {call['response_time']:.3f}s (Status: {call['status_code']})")
        
        if self.emergency_alerts:
            print("\nEMERGENCY ALERTS:")
            for alert in self.emergency_alerts:
                print(f"  {alert['timestamp']}: {alert['outcome']} (Confidence: {alert['confidence']}%)")
                print(f"    Red Flags: {alert['red_flags']}")
        
        print(f"{'=' * 80}\n")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
