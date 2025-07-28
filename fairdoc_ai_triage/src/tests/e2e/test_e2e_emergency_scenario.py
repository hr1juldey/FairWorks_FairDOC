"""
E2E Test: Emergency Medical Scenario Flow

Tests complete emergency detection workflow from patient input
through DSPy agent, stakeholder routing, and alert generation.

File: src/tests/e2e/test_e2e_emergency_scenario.py
"""

import pytest
import asyncio
import time
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import json

from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2


class TestEmergencyScenarioE2E:
    """
    End-to-end test for emergency medical scenarios
    
    Flow: Patient → Emergency Detection → Doctor Alert → Background Tasks
    """
    
    @pytest.fixture(autouse=True)
    async def setup_e2e_environment(self):
        """Setup realistic E2E environment with minimal mocking"""
        # Mock only external services, keep internal logic intact
        with patch('src.app2.services.chat.raven_bridge.httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value.status_code = 200
            
            with patch('src.app2.services.dspy.medical_agent.dspy.ChainOfThought') as mock_dspy:
                # Configure DSPy to return emergency responses for chest pain
                mock_predictor = AsyncMock()
                mock_result = type('DSPyResult', (), {
                    'outcome_classification': 'emergency',
                    'confidence_score': 95,
                    'next_question': 'COMPLETE',
                    'reasoning': 'Chest pain with radiation indicates possible MI',
                    'red_flags': 'chest_pain,radiation,diaphoresis'
                })()
                mock_predictor.return_value = mock_result
                mock_dspy.return_value = mock_predictor
                
                yield
    
    @pytest.mark.asyncio
    async def test_complete_emergency_workflow(self):
        """
        Test complete emergency workflow from patient input to alert generation
        """
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # === Phase 1: Initial Emergency Report ===
            emergency_payload = {
                "user_message": "I have crushing chest pain radiating to my left arm and I'm sweating profusely",
                "stakeholder_role": "patient",
                "stakeholder_id": "emergency_patient_001",
                "chat_provider": "api_direct"
            }
            
            start_time = time.time()
            response = await client.post("/api/v2/medical/chat", json=emergency_payload)
            response_time = time.time() - start_time
            
            # Verify immediate response
            assert response.status_code == 200
            assert response_time < 5.0, f"Emergency response too slow: {response_time}s"
            
            data = response.json()
            conversation_id = data["conversation_id"]
            
            # === Phase 2: Verify Emergency Detection ===
            assert data["medical_outcome"] == "emergency", f"Expected emergency, got {data['medical_outcome']}"
            assert data["confidence_score"] >= 90, f"Emergency confidence too low: {data['confidence_score']}"
            assert data["is_conversation_complete"] is True, "Emergency should complete conversation"
            assert len(data["red_flags"]) > 0, "Emergency should have red flags"
            
            # Verify red flags contain expected emergency indicators
            red_flags = data["red_flags"]
            emergency_indicators = ["chest_pain", "radiation"]
            assert any(flag in red_flags for flag in emergency_indicators), f"Missing emergency indicators in {red_flags}"
            
            print(f"✅ Emergency detected in {response_time:.2f}s with {data['confidence_score']}% confidence")
            
            # === Phase 3: Verify Conversation State ===
            state_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/state")
            assert state_response.status_code == 200
            
            state_data = state_response.json()
            assert state_data["status"] == "completed", "Emergency conversation should be completed"
            assert state_data["current_outcome"] == "emergency"
            
            # === Phase 4: Verify History Recording ===
            history_response = await client.get(f"/api/v2/medical/chat/{conversation_id}/history")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert len(history_data["turns"]) >= 1, "Should have at least one conversation turn"
            
            first_turn = history_data["turns"][0] if history_data["turns"] else {}
            assert "crushing chest pain" in first_turn.get("user_message", "").lower()
            
            print(f"✅ Emergency workflow completed for conversation {conversation_id}")
    
    @pytest.mark.asyncio
    async def test_emergency_response_time_performance(self):
        """
        Test that emergency scenarios are processed within acceptable time limits
        """
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            emergency_scenarios = [
                "I'm having severe chest pain and difficulty breathing",
                "I think I'm having a heart attack - chest pain and nausea",
                "Crushing chest pain radiating down my left arm"
            ]
            
            response_times = []
            
            for scenario in emergency_scenarios:
                payload = {
                    "user_message": scenario,
                    "stakeholder_role": "patient",
                    "stakeholder_id": f"perf_test_{len(response_times)}",
                    "chat_provider": "api_direct"
                }
                
                start_time = time.time()
                response = await client.post("/api/v2/medical/chat", json=payload)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                assert response.status_code == 200
                data = response.json()
                assert data["medical_outcome"] == "emergency"
                
                print(f"✅ Emergency '{scenario[:30]}...' processed in {response_time:.2f}s")
            
            # Performance assertions
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            assert avg_response_time < 3.0, f"Average emergency response time too high: {avg_response_time:.2f}s"
            assert max_response_time < 5.0, f"Maximum emergency response time too high: {max_response_time:.2f}s"
            
            print(f"✅ Performance test passed - Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_emergency_vs_non_emergency_classification(self):
        """
        Test that system correctly distinguishes emergency from non-emergency scenarios
        """
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            test_cases = [
                {
                    "message": "I have crushing chest pain and can't breathe",
                    "expected_outcome": "emergency",
                    "min_confidence": 85
                },
                {
                    "message": "I have a mild headache that started today",
                    "expected_outcome": "inconclusive",
                    "max_confidence": 70
                },
                {
                    "message": "I need to schedule an appointment for next week",
                    "expected_outcome": "inconclusive",
                    "max_confidence": 40
                }
            ]
            
            results = []
            
            for i, case in enumerate(test_cases):
                payload = {
                    "user_message": case["message"],
                    "stakeholder_role": "patient",
                    "stakeholder_id": f"classification_test_{i}",
                    "chat_provider": "api_direct"
                }
                
                response = await client.post("/api/v2/medical/chat", json=payload)
                assert response.status_code == 200
                
                data = response.json()
                outcome = data["medical_outcome"]
                confidence = data["confidence_score"]
                
                # Verify expected outcome
                if case["expected_outcome"] == "emergency":
                    assert outcome == "emergency", f"Expected emergency for '{case['message']}', got {outcome}"
                    assert confidence >= case["min_confidence"], f"Emergency confidence too low: {confidence}%"
                else:
                    assert outcome != "emergency", f"False emergency for '{case['message']}'"
                    if "max_confidence" in case:
                        assert confidence <= case["max_confidence"], f"Non-emergency confidence too high: {confidence}%"
                
                results.append({
                    "message": case["message"][:50],
                    "outcome": outcome,
                    "confidence": confidence,
                    "expected": case["expected_outcome"]
                })
                
                print(f"✅ Classification: '{case['message'][:30]}...' → {outcome} ({confidence}%)")
            
            # Verify we tested both emergency and non-emergency cases
            emergency_count = sum(1 for r in results if r["outcome"] == "emergency")
            non_emergency_count = len(results) - emergency_count
            
            assert emergency_count >= 1, "Should test at least one emergency case"
            assert non_emergency_count >= 1, "Should test at least one non-emergency case"
            
            print(f"✅ Classification test completed: {emergency_count} emergency, {non_emergency_count} non-emergency")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])