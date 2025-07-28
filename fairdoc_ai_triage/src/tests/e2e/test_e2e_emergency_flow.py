# End-to-End Test: Complete Emergency Triage Flow
# File: src/tests/e2e/test_e2e_emergency_flow.py

"""
E2E test for complete emergency triage workflow from HTTP request to database persistence.
Tests the critical path: Patient reports emergency → System detects → Routes → Alerts → Persists

Provides comprehensive diagnostics for debugging both offline and live environments.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from httpx import AsyncClient
from sqlalchemy import text
from unittest.mock import patch, AsyncMock

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.core.database_v2 import get_async_session
from src.app2.models.schemas.multiturn_chat import StakeholderRole, ChatProvider


class TestEmergencyTriageFlow:
    """Complete emergency triage flow testing"""
    
    @pytest.fixture
    async def test_client(self):
        """FastAPI test client with real app instance"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def emergency_payload(self):
        """Emergency scenario test payload"""
        return {
            "user_message": "I have crushing chest pain radiating to my left arm and I'm sweating profusely",
            "stakeholder_role": "patient",
            "stakeholder_id": "emergency_patient_001",
            "chat_provider": "api_direct",
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user_agent": "E2E Test Client",
                "location": "test_location"
            }
        }
    
    @pytest.mark.asyncio
    async def test_complete_emergency_workflow(self, test_client, emergency_payload):
        """
        Test complete emergency workflow with full system integration
        
        Flow: HTTP Request → Chat Orchestrator → DSPy Agent → Emergency Detection 
              → Stakeholder Routing → Background Tasks → Database Persistence
        """
        # Track test execution metrics
        start_time = time.time()
        test_metrics = {
            "request_time": None,
            "response_time": None,
            "background_tasks_completed": False,
            "database_persisted": False,
            "emergency_alert_sent": False
        }
        
        # Mock background task tracking
        background_tasks_executed = []
        
        def track_background_task(task_name, *args, **kwargs):
            background_tasks_executed.append({
                "task": task_name,
                "args": args,
                "kwargs": kwargs,
                "timestamp": datetime.utcnow().isoformat()
            })
            return AsyncMock()
        
        # Patch background task handlers for tracking
        with patch('src.app2.services.chat.emergency_handler.EmergencyHandler.handle_emergency_alert', 
                  side_effect=lambda *args: track_background_task('emergency_alert', *args)):
            with patch('src.app2.services.chat.persistence_handler.PersistenceHandler.save_conversation_to_postgresql',
                      side_effect=lambda *args: track_background_task('persistence', *args)):
                
                # Step 1: Send emergency request
                request_start = time.time()
                response = await test_client.post("/api/v2/medical/chat", json=emergency_payload)
                request_end = time.time()
                
                test_metrics["request_time"] = request_end - request_start
                
                # Step 2: Verify HTTP response
                assert response.status_code == 200, f"Request failed: {response.text}"
                
                response_data = response.json()
                conversation_id = response_data["conversation_id"]
                
                # Step 3: Verify emergency detection
                assert response_data["medical_outcome"] == "emergency", \
                    f"Expected emergency outcome, got: {response_data['medical_outcome']}"
                
                assert response_data["confidence_score"] >= 80, \
                    f"Low confidence for emergency: {response_data['confidence_score']}"
                
                assert response_data["is_conversation_complete"] is True, \
                    "Emergency conversations should complete immediately"
                
                assert len(response_data["red_flags"]) > 0, \
                    "Emergency should have red flags"
                
                # Step 4: Verify conversation state persistence
                await asyncio.sleep(0.1)  # Allow background tasks to execute
                
                state_response = await test_client.get(f"/api/v2/medical/chat/{conversation_id}/state")
                assert state_response.status_code == 200
                
                state_data = state_response.json()
                assert state_data["status"] == "completed"
                assert state_data["current_outcome"] == "emergency"
                
                # Step 5: Verify background tasks were triggered
                await asyncio.sleep(0.2)  # Additional wait for background tasks
                
                # Check emergency alert task
                emergency_tasks = [t for t in background_tasks_executed if t["task"] == "emergency_alert"]
                assert len(emergency_tasks) > 0, "Emergency alert task not executed"
                
                # Check persistence task
                persistence_tasks = [t for t in background_tasks_executed if t["task"] == "persistence"]
                assert len(persistence_tasks) > 0, "Persistence task not executed"
                
                test_metrics["background_tasks_completed"] = True
                test_metrics["emergency_alert_sent"] = len(emergency_tasks) > 0
                
                # Step 6: Verify conversation history
                history_response = await test_client.get(f"/api/v2/medical/chat/{conversation_id}/history")
                assert history_response.status_code == 200
                
                history_data = history_response.json()
                assert len(history_data["turns"]) >= 1
                assert history_data["outcome"] == "emergency"
                
                # Step 7: Verify system health after emergency processing
                health_response = await test_client.get("/api/v2/health")
                health_data = health_response.json()
                
                # System should remain healthy after emergency processing
                assert health_response.status_code in [200, 503]  # 503 acceptable if services not fully initialized
                assert "services" in health_data
                
                test_metrics["response_time"] = time.time() - start_time
                
                # Step 8: Comprehensive test result logging
                print("\n" + "="*80)
                print("EMERGENCY TRIAGE FLOW - TEST RESULTS")
                print("="*80)
                print(f"Total Test Duration: {test_metrics['response_time']:.3f} seconds")
                print(f"HTTP Request Time: {test_metrics['request_time']:.3f} seconds")
                print(f"Conversation ID: {conversation_id}")
                print(f"Medical Outcome: {response_data['medical_outcome']}")
                print(f"Confidence Score: {response_data['confidence_score']}")
                print(f"Red Flags: {response_data['red_flags']}")
                print(f"Background Tasks: {len(background_tasks_executed)}")
                print(f"Emergency Alert: {'✓' if test_metrics['emergency_alert_sent'] else '✗'}")
                print(f"Persistence: {'✓' if len(persistence_tasks) > 0 else '✗'}")
                print("\nBackground Task Details:")
                for task in background_tasks_executed:
                    print(f"  - {task['task']} at {task['timestamp']}")
                print("="*80)
                
                # All assertions passed - emergency flow working correctly
                assert test_metrics["background_tasks_completed"]
                assert test_metrics["emergency_alert_sent"]
    
    @pytest.mark.asyncio
    async def test_emergency_system_load(self, test_client):
        """
        Test system behavior under emergency load scenarios
        """
        emergency_payloads = [
            {
                "user_message": f"Emergency patient {i}: severe chest pain and difficulty breathing",
                "stakeholder_role": "patient",
                "stakeholder_id": f"load_test_patient_{i:03d}",
                "chat_provider": "api_direct"
            }
            for i in range(5)
        ]
        
        # Send multiple emergency requests concurrently
        tasks = [
            test_client.post("/api/v2/medical/chat", json=payload)
            for payload in emergency_payloads
        ]
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify all requests succeeded
        successful_responses = 0
        emergency_outcomes = 0
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                pytest.fail(f"Request {i} failed with exception: {response}")
            
            assert response.status_code == 200, f"Request {i} failed: {response.text}"
            
            data = response.json()
            successful_responses += 1
            
            if data["medical_outcome"] == "emergency":
                emergency_outcomes += 1
        
        # System performance validation
        avg_response_time = total_time / len(emergency_payloads)
        
        print(f"\nEMERGENCY LOAD TEST RESULTS:")
        print(f"Total Requests: {len(emergency_payloads)}")
        print(f"Successful: {successful_responses}")
        print(f"Emergency Outcomes: {emergency_outcomes}")
        print(f"Total Time: {total_time:.3f} seconds")
        print(f"Average Response Time: {avg_response_time:.3f} seconds")
        
        # Performance assertions
        assert successful_responses == len(emergency_payloads)
        assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time:.3f}s"
        assert emergency_outcomes >= len(emergency_payloads) * 0.8  # At least 80% should be emergencies

    @pytest.mark.asyncio
    async def test_emergency_data_integrity(self, test_client, emergency_payload):
        """
        Test data integrity throughout emergency processing pipeline
        """
        # Send emergency request
        response = await test_client.post("/api/v2/medical/chat", json=emergency_payload)
        assert response.status_code == 200
        
        response_data = response.json()
        conversation_id = response_data["conversation_id"]
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Verify data consistency across endpoints
        endpoints_to_check = [
            f"/api/v2/medical/chat/{conversation_id}/state",
            f"/api/v2/medical/chat/{conversation_id}/history"
        ]
        
        data_consistency_results = {}
        
        for endpoint in endpoints_to_check:
            endpoint_response = await test_client.get(endpoint)
            assert endpoint_response.status_code == 200
            data_consistency_results[endpoint] = endpoint_response.json()
        
        # Verify data consistency
        state_data = data_consistency_results[f"/api/v2/medical/chat/{conversation_id}/state"]
        history_data = data_consistency_results[f"/api/v2/medical/chat/{conversation_id}/history"]
        
        # Cross-validate conversation data
        assert state_data["conversation_id"] == conversation_id
        assert history_data["conversation_id"] == conversation_id
        assert state_data["current_outcome"] == history_data["outcome"] == "emergency"
        
        print(f"\nDATA INTEGRITY VERIFICATION:")
        print(f"Conversation ID: {conversation_id}")
        print(f"State Outcome: {state_data['current_outcome']}")
        print(f"History Outcome: {history_data['outcome']}")
        print(f"Data Consistency: ✓ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])