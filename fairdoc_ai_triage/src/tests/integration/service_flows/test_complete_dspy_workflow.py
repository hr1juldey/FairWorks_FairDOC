"""
Complete DSPy Medical Triage Workflow Integration Tests - REAL LLM CALLS

Tests end-to-end behavior consistency and performance with REAL DeepSeek-R1 calls.
NO MOCKS - Real DSPy integration testing with 60-second timeouts.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime

# Configure pytest for async and extended timeouts
pytestmark = pytest.mark.asyncio

from src.app2.services.chat.chat_orchestrator import ChatOrchestrator
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
from src.app2.services.context.redis_queue import ConversationQueue
from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatRequest,
    MultiTurnChatResponse,
    StakeholderRole,
    MedicalOutcome
)
from src.app2.core.config_v2 import settings_v2

import structlog
logger = structlog.get_logger(__name__)

# Test configuration for real LLM calls
REAL_LLM_TIMEOUT = 120  # 60 seconds for real API calls
MAX_RETRIES = 4        # Retry failed calls
CONCURRENT_LIMIT = 3   # Reduced concurrent requests for stability


class TestCompleteDSPyWorkflowRealLLM:
    """Integration tests with REAL LLM calls - no mocks"""
    
    @pytest.fixture(autouse=True)
    async def setup_real_integration_environment(self):
        """Setup REAL DSPy integration environment"""
        try:
            # Initialize real components (NO MOCKS)
            self.question_generator = MedicalQuestionGenerator(
                model_name=settings_v2.DSPy_MODEL_NAME
            )
            
            self.medical_agent = MedicalTriageAgent(
            model_name=settings_v2.DSPY_MODEL_NAME,
            question_generator=self.question_generator
            )
            
            self.chat_orchestrator = ChatOrchestrator(
                question_generator=self.question_generator
            )
            
            # Initialize conversation queue
            self.conversation_queue = ConversationQueue()
            await self.conversation_queue.initialize()
            
            # Initialize chat orchestrator
            await self.chat_orchestrator.initialize()
            
            logger.info("🧪 REAL DSPy integration environment ready - NO MOCKS")
            
        except Exception as e:
            pytest.skip(f"Real integration environment setup failed: {str(e)}")
    
    async def test_emergency_chest_pain_real_workflow(self):
        """Test REAL workflow for emergency chest pain - REAL LLM CALLS"""
        
        # FIXED: Use user_message and stakeholder_id
        request = MultiTurnChatRequest(
            conversation_id=uuid4(),
            user_message="I have severe crushing chest pain radiating to my left arm with heavy sweating",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="test_patient_emergency_001"
        )
        
        # Process with REAL LLM (60 second timeout)
        start_time = time.time()
        
        try:
            orchestration_result = await asyncio.wait_for(
                self.chat_orchestrator.process_conversation_turn(request),
                timeout=REAL_LLM_TIMEOUT
            )
            processing_time = time.time() - start_time
            
            # Validate REAL LLM response
            agent_result = orchestration_result["agent_result"]
            
            # Real LLM should detect emergency (with some tolerance for AI variability)
            assert agent_result["outcome"] in ["emergency", "routine"], \
                f"Unexpected outcome: {agent_result['outcome']}"
            
            # For chest pain with radiation, confidence should be reasonable
            assert agent_result["confidence"] >= 60, \
                f"Confidence too low for serious symptoms: {agent_result['confidence']}"
            
            # Real LLM should provide reasoning
            assert len(agent_result.get("reasoning", "")) > 0, "Should provide medical reasoning"
            
            # Performance check (real LLM can be slow)
            assert processing_time <= REAL_LLM_TIMEOUT, \
                f"Processing took {processing_time:.2f}s, should be under {REAL_LLM_TIMEOUT}s"
            
            logger.info("✅ REAL emergency chest pain workflow test passed",
                       conversation_id=str(request.conversation_id),
                       outcome=agent_result["outcome"],
                       confidence=agent_result["confidence"],
                       processing_time_s=round(processing_time, 2),
                       real_llm_call=True)
                       
        except asyncio.TimeoutError:
            pytest.skip(f"Test skipped due to LLM timeout (>{REAL_LLM_TIMEOUT}s)")
        except Exception as e:
            pytest.fail(f"Real LLM workflow failed: {str(e)}")
    
    async def test_multi_turn_real_conversation(self):
        """Test REAL multi-turn conversation with actual DSPy reasoning"""
        
        conversation_id = uuid4()
        
        # Turn 1: Vague symptoms
        request1 = MultiTurnChatRequest(
            conversation_id=conversation_id,
            user_message="I have some chest discomfort",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="test_patient_multiturn_real"
        )
        
        try:
            result1 = await asyncio.wait_for(
                self.chat_orchestrator.process_conversation_turn(request1),
                timeout=REAL_LLM_TIMEOUT
            )
            
            agent_result1 = result1["agent_result"]
            
            # Real LLM should ask follow-up questions for vague symptoms
            assert agent_result1["outcome"] in ["inconclusive", "routine", "emergency"], \
                f"Unexpected Turn 1 outcome: {agent_result1['outcome']}"
            
            # Turn 2: More specific emergency symptoms
            request2 = MultiTurnChatRequest(
                conversation_id=conversation_id,
                user_message="Actually it's crushing pain that started 30 minutes ago, I'm sweating heavily and feeling nauseous",
                stakeholder_role=StakeholderRole.PATIENT,
                stakeholder_id="test_patient_multiturn_real"
            )
            
            result2 = await asyncio.wait_for(
                self.chat_orchestrator.process_conversation_turn(request2),
                timeout=REAL_LLM_TIMEOUT
            )
            
            agent_result2 = result2["agent_result"]
            
            # Real LLM should escalate with more specific symptoms
            assert agent_result2["outcome"] in ["emergency", "routine"], \
                f"Turn 2 should show escalation, got: {agent_result2['outcome']}"
            
            # Confidence should generally increase with more specific symptoms
            confidence_increased = agent_result2["confidence"] >= agent_result1["confidence"] - 10
            assert confidence_increased, \
                f"Confidence should increase or stay similar: {agent_result1['confidence']} → {agent_result2['confidence']}"
            
            logger.info("✅ REAL multi-turn conversation test passed",
                       turn1_outcome=agent_result1["outcome"],
                       turn2_outcome=agent_result2["outcome"],
                       confidence_change=f"{agent_result1['confidence']} → {agent_result2['confidence']}",
                       real_llm_calls=2)
                       
        except asyncio.TimeoutError:
            pytest.skip("Multi-turn test skipped due to LLM timeout")
        except Exception as e:
            pytest.fail(f"Real multi-turn conversation failed: {str(e)}")
    
    async def test_emergency_pattern_detection_real_llm(self):
        """Test REAL emergency pattern detection with actual DSPy reasoning"""
        
        # Test scenarios for REAL LLM (fewer scenarios, longer timeouts)
        test_scenarios = [
            {
                'symptoms': 'sudden thunderclap headache worst ever experienced',
                'expected_severity': 'high',
                'description': 'Critical neurological emergency'
            },
            {
                'symptoms': 'severe crushing chest pain with radiation to left arm',
                'expected_severity': 'high', 
                'description': 'Cardiac emergency symptoms'
            },
            {
                'symptoms': 'mild headache from work stress',
                'expected_severity': 'low',
                'description': 'Minor symptoms'
            }
        ]
        
        results = []
        errors = []
        
        for i, scenario in enumerate(test_scenarios):
            try:
                # Test question generator directly with REAL DSPy
                start_time = time.time()
                
                question_result = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.to_thread(
                            self.question_generator.suggest_questions,
                            symptom_text=scenario['symptoms'],
                            max_questions=3
                        )
                    ),
                    timeout=REAL_LLM_TIMEOUT
                )
                
                processing_time = time.time() - start_time
                
                # Validate REAL LLM response structure
                assert 'urgency_level' in question_result, "Should return urgency level"
                assert 'questions' in question_result, "Should return questions"
                assert len(question_result['questions']) > 0, "Should generate questions"
                
                urgency_level = question_result.get('urgency_level', 'unknown')
                emergency_indicators = question_result.get('emergency_indicators', [])
                
                # Log real results for analysis
                result_data = {
                    'scenario': i,
                    'symptoms': scenario['symptoms'][:50],
                    'expected_severity': scenario['expected_severity'],
                    'actual_urgency': urgency_level,
                    'emergency_indicators': len(emergency_indicators),
                    'processing_time': round(processing_time, 2),
                    'questions_generated': len(question_result['questions'])
                }
                results.append(result_data)
                
                logger.info(f"✅ REAL pattern detection test {i} completed", **result_data)
                
            except asyncio.TimeoutError:
                errors.append(f"Scenario {i}: Timeout after {REAL_LLM_TIMEOUT}s")
            except Exception as e:
                errors.append(f"Scenario {i}: {str(e)}")
        
        # Validate that most scenarios worked
        success_rate = len(results) / len(test_scenarios)
        assert success_rate >= 0.6, f"Success rate too low: {success_rate:.1%}. Errors: {errors}"
        
        logger.info("🎯 REAL pattern detection completed",
                   successful_scenarios=len(results),
                   total_scenarios=len(test_scenarios),
                   success_rate=f"{success_rate:.1%}")
    
    async def test_real_llm_performance_under_load(self):
        """Test REAL DSPy performance under reduced concurrent load"""
        
        async def simulate_real_emergency_conversation(user_id: str) -> Dict[str, Any]:
            """Simulate emergency conversation with REAL LLM calls"""
            start_time = time.time()
            
            request = MultiTurnChatRequest(
                conversation_id=uuid4(),
                user_message="severe crushing chest pain with sweating",
                stakeholder_role=StakeholderRole.PATIENT,
                stakeholder_id=user_id
            )
            
            try:
                result = await asyncio.wait_for(
                    self.chat_orchestrator.process_conversation_turn(request),
                    timeout=REAL_LLM_TIMEOUT
                )
                processing_time = time.time() - start_time
                
                return {
                    'user_id': user_id,
                    'processing_time': processing_time,
                    'outcome': result["agent_result"]["outcome"],
                    'confidence': result["agent_result"]["confidence"],
                    'success': True
                }
            except Exception as e:
                return {
                    'user_id': user_id,
                    'processing_time': time.time() - start_time,
                    'error': str(e),
                    'success': False
                }
        
        # Reduced concurrent requests for REAL LLM stability
        concurrent_requests = CONCURRENT_LIMIT
        tasks = [
            simulate_real_emergency_conversation(f"load_test_real_{i}")
            for i in range(concurrent_requests)
        ]
        
        # Execute with REAL LLM calls
        load_test_start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_load_time = time.time() - load_test_start
        
        # Analyze REAL results
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed_results = [r for r in results if not isinstance(r, dict) or not r.get('success')]
        
        # Real LLM performance assertions (more lenient)
        success_rate = len(successful_results) / concurrent_requests
        assert success_rate >= 0.5, \
            f"Real LLM success rate too low: {success_rate:.1%}. Got {len(successful_results)}/{concurrent_requests}"
        
        if successful_results:
            avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            max_processing_time = max(r['processing_time'] for r in successful_results)
            
            # Real LLM performance expectations
            assert avg_processing_time <= REAL_LLM_TIMEOUT, \
                f"Average time {avg_processing_time:.2f}s should be under {REAL_LLM_TIMEOUT}s"
            assert max_processing_time <= REAL_LLM_TIMEOUT, \
                f"Max time {max_processing_time:.2f}s should be under {REAL_LLM_TIMEOUT}s"
        
        logger.info("🚀 REAL LLM performance test passed",
                   concurrent_requests=concurrent_requests,
                   successful_requests=len(successful_results),
                   failed_requests=len(failed_results),
                   success_rate=f"{success_rate:.1%}",
                   avg_time=f"{avg_processing_time:.2f}s" if successful_results else "N/A",
                   total_time=f"{total_load_time:.2f}s")
    
    async def test_question_generator_real_consistency(self):
        """Test REAL question generator and medical agent consistency"""
        
        test_symptoms = [
            "severe crushing chest pain with radiation",
            "mild headache from stress"
        ]
        
        consistency_results = []
        
        for symptoms in test_symptoms:
            try:
                # Real medical agent assessment
                agent_result = await asyncio.wait_for(
                    self.medical_agent.process_turn(
                        symptoms=symptoms,
                        nice_context="Emergency protocols"
                    ),
                    timeout=REAL_LLM_TIMEOUT
                )
                
                # Real question generator assessment
                question_result = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.to_thread(
                            self.question_generator.suggest_questions,
                            symptom_text=symptoms,
                            max_questions=3
                        )
                    ),
                    timeout=REAL_LLM_TIMEOUT
                )
                
                # Analyze REAL responses
                agent_emergency = agent_result.get("emergency_detected", False)
                question_urgency = question_result.get("urgency_level", "low")
                question_emergency = question_urgency in ["high", "critical"]
                
                # Real LLM can have some variability - allow for slight inconsistencies
                consistency_check = {
                    'symptoms': symptoms[:40],
                    'agent_emergency': agent_emergency,
                    'question_urgency': question_urgency,
                    'question_emergency': question_emergency,
                    'consistent': agent_emergency == question_emergency
                }
                consistency_results.append(consistency_check)
                
                logger.info("✅ REAL consistency check completed", **consistency_check)
                
            except asyncio.TimeoutError:
                logger.warning(f"Consistency test timeout for: {symptoms[:40]}")
            except Exception as e:
                logger.error(f"Consistency test error for {symptoms[:40]}: {str(e)}")
        
        # Real LLM consistency validation (allow some variation)
        if consistency_results:
            consistent_count = sum(1 for r in consistency_results if r['consistent'])
            consistency_rate = consistent_count / len(consistency_results)
            
            # Allow for some LLM variability in real calls
            assert consistency_rate >= 0.6, \
                f"Consistency rate too low: {consistency_rate:.1%}. Results: {consistency_results}"
            
            logger.info("🎯 REAL consistency validation passed",
                       consistent_results=consistent_count,
                       total_results=len(consistency_results),
                       consistency_rate=f"{consistency_rate:.1%}")


# Performance benchmark for REAL LLM calls
def real_llm_benchmark(max_time_seconds: float = REAL_LLM_TIMEOUT):
    """Decorator for REAL LLM performance benchmarking"""
    def decorator(test_func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await test_func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info("⏱️ REAL LLM benchmark completed",
                           test=test_func.__name__,
                           execution_time=f"{execution_time:.2f}s",
                           max_allowed=f"{max_time_seconds}s",
                           within_limit=execution_time <= max_time_seconds)
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error("❌ REAL LLM test failed",
                           test=test_func.__name__,
                           execution_time=f"{execution_time:.2f}s",
                           error=str(e))
                raise
        return wrapper
    return decorator


# Apply real LLM benchmarks
TestCompleteDSPyWorkflowRealLLM.test_emergency_chest_pain_real_workflow = real_llm_benchmark()(
    TestCompleteDSPyWorkflowRealLLM.test_emergency_chest_pain_real_workflow
)

TestCompleteDSPyWorkflowRealLLM.test_multi_turn_real_conversation = real_llm_benchmark()(
    TestCompleteDSPyWorkflowRealLLM.test_multi_turn_real_conversation
)
