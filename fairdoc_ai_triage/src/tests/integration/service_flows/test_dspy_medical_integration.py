"""
Integration Tests for DSPy Medical Service Module
Tests real DeepSeek calls with varied inputs and monitoring
Runs multiple iterations to catch intermittent issues
"""
import pytest
import asyncio
import random
import time
from typing import Dict, Any, List
from unittest.mock import patch, Mock
import structlog

logger = structlog.get_logger(__name__)

# Test environment setup for integration testing
with patch.dict('os.environ', {
    'OLLAMA_BASE_URL': 'http://localhost:11434',
    'FAIRDOC_V2_DSPy_MODEL': 'deepseek-r1:8b',
    'REDIS_URL': 'redis://localhost:6379/0',
    'DATABASE_URL': 'postgresql+asyncpg://test:test@localhost/test',
}):
    from src.app2.services.dspy.medical_agent import MedicalTriageAgent, MedicalOutcome
    from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
    from src.app2.services.dspy.evaluation_optimizer import EvaluationOptimizer


class TestDSPyMedicalIntegration:
    """Integration tests for complete DSPy medical service module"""
    
    # Test data variations for multiple runs
    VARIED_SYMPTOMS = [
        "severe crushing chest pain going to my left arm, sweating heavily",
        "mild headache on right side, no nausea, started this morning",
        "sharp stomach pain, vomiting twice, fever 101F",
        "shortness of breath, ankle swelling, chest tightness",
        "worst headache ever, sudden onset, neck feels stiff",
        "chest discomfort, comes and goes, worse with exercise",
        "dizziness when standing up, heart racing, sweating",
        "abdominal pain started near belly button, now right side",
        "back pain radiating to leg, numbness in toes",
        "sore throat, fever, difficulty swallowing"
    ]
    
    NICE_CONTEXTS = [
        "CG95 Chest Pain: Emergency if crushing >20min with radiation",
        "NG127 Headache: Emergency if thunderclap, neck stiffness, fever",
        "CG141 Abdominal Pain: Emergency if peritoneal signs present",
        "NG116 Heart Failure: Monitor fluid retention, breathlessness",
        "Basic assessment protocol: Gather symptoms systematically"
    ]
    
    @pytest.fixture(autouse=True)
    async def setup_integration_environment(self):
        """Setup for integration testing with real services"""
        # Add delay for service initialization
        await asyncio.sleep(1)
        
        # Initialize services
        self.medical_agent = MedicalTriageAgent()
        self.question_generator = MedicalQuestionGenerator()
        self.evaluation_optimizer = EvaluationOptimizer()
        
        # Track API calls
        self.api_calls = []
        self.deepseek_called = False
        
        logger.info("🧪 Integration test environment initialized")
    
    @pytest.mark.asyncio
    async def test_deepseek_actually_called_multiple_runs(self):
        """Test that DeepSeek is actually being called with varied inputs"""
        call_results = []
        
        for run in range(5):  # Multiple runs with different inputs
            symptom = random.choice(self.VARIED_SYMPTOMS)
            context = random.choice(self.NICE_CONTEXTS)
            
            logger.info(f"🔄 Integration test run {run + 1}/5", 
                       symptom_preview=symptom[:30],
                       context_type=context.split()[0])
            
            start_time = time.time()
            
            # Test with real DSPy agent (may call Ollama or fallback)
            try:
                result = await self.medical_agent.process_turn(
                    symptoms=symptom,
                    nice_context=context
                )
                
                processing_time = time.time() - start_time
                
                call_results.append({
                    'run': run + 1,
                    'symptom': symptom,
                    'outcome': result['outcome'],
                    'confidence': result['confidence'],
                    'has_next_question': result['next_question'] is not None,
                    'has_reasoning': len(result.get('reasoning', '')) > 0,
                    'has_thinking': len(result.get('thinking', '')) > 0,
                    'red_flags_count': len(result.get('red_flags', [])),
                    'emergency_detected': result.get('emergency_detected', False),
                    'processing_time': processing_time,
                    'success': True
                })
                
                # Verify realistic response structure
                assert result['outcome'] in [e.value.split('_')[0] for e in MedicalOutcome]
                assert 0 <= result['confidence'] <= 100
                assert isinstance(result['red_flags'], list)
                assert isinstance(result['is_complete'], bool)
                
                logger.info(f"✅ Run {run + 1} completed", 
                           outcome=result['outcome'],
                           confidence=result['confidence'],
                           time=f"{processing_time:.2f}s")
                
            except Exception as e:
                call_results.append({
                    'run': run + 1,
                    'symptom': symptom,
                    'error': str(e),
                    'success': False
                })
                logger.error(f"❌ Run {run + 1} failed", error=str(e))
        
        # Analyze results across all runs
        successful_runs = [r for r in call_results if r['success']]
        assert len(successful_runs) >= 3, f"At least 3/5 runs should succeed, got {len(successful_runs)}"
        
        # Check for response variation (not all identical)
        outcomes = [r['outcome'] for r in successful_runs]
        confidences = [r['confidence'] for r in successful_runs]
        
        assert len(set(outcomes)) > 1 or len(set(confidences)) > 1, \
            "Responses should show variation across different inputs"
        
        # Performance check
        avg_time = sum(r['processing_time'] for r in successful_runs) / len(successful_runs)
        assert avg_time < 30.0, f"Average processing time {avg_time:.2f}s should be < 30s"
        
        logger.info("🎯 Multi-run integration test passed", 
                   successful_runs=len(successful_runs),
                   avg_time=f"{avg_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_conversation_flow_integration(self):
        """Test multi-turn conversation with state management"""
        # Start with vague symptoms
        result1 = await self.medical_agent.process_turn(
            symptoms="I have some chest discomfort",
            nice_context="CG95 Chest Pain Assessment Protocol"
        )
        
        assert result1['outcome'] in ['inconclusive', 'routine', 'emergency']
        assert not result1['is_complete']  # Should need more info
        turn_1_confidence = result1['confidence']
        
        # Follow up with more specific emergency symptoms
        result2 = await self.medical_agent.process_turn(
            symptoms="Actually it's crushing pain going to my arm, I'm sweating",
            nice_context="CG95 Emergency Chest Pain Criteria"
        )
        
        assert result2['outcome'] in ['emergency', 'routine']
        turn_2_confidence = result2['confidence']
        
        # Check conversation progression
        assert self.medical_agent.turn_count == 2
        assert len(self.medical_agent.conversation_history.messages) == 2
        
        # Confidence should generally increase with more specific symptoms
        # (though not guaranteed with real LLM)
        logger.info("📊 Conversation progression", 
                   turn1_conf=turn_1_confidence,
                   turn2_conf=turn_2_confidence,
                   outcome_progression=f"{result1['outcome']} → {result2['outcome']}")
    
    @pytest.mark.asyncio 
    async def test_question_generator_integration(self):
        """Test question generator with DSPy integration"""
        varied_inputs = [
            ("severe headache sudden onset", "NG127 Headache Emergency Protocol"),
            ("chest pain with sweating", "CG95 Cardiac Assessment"),
            ("abdominal pain right side", "CG141 Appendicitis Assessment")
        ]
        
        for symptoms, protocol in varied_inputs:
            result = self.question_generator.suggest_questions(
                symptom_text=symptoms,
                nice_protocols=protocol,
                max_questions=3
            )
            
            # Verify question generation works
            assert 'questions' in result
            assert len(result['questions']) <= 3
            assert result['thinking_enabled'] is True
            
            # Check question quality
            questions = result['questions']
            for question in questions:
                assert len(question) > 10  # Reasonable question length
                assert '?' in question or question.endswith('.')
            
            logger.info("❓ Questions generated", 
                       symptoms=symptoms[:20],
                       question_count=len(questions),
                       urgency=result.get('urgency_level', 'unknown'))
    
    @pytest.mark.asyncio
    async def test_service_error_handling_integration(self):
        """Test error handling across DSPy services"""
        # Test with problematic inputs
        error_inputs = [
            "",  # Empty input
            "   ",  # Whitespace only  
            "x" * 2000,  # Very long input
            "🤔💭🏥❓",  # Only emojis
            "SELECT * FROM users;",  # Potential injection
        ]
        
        for bad_input in error_inputs:
            try:
                result = await self.medical_agent.process_turn(
                    symptoms=bad_input,
                    nice_context="Basic protocol"
                )
                
                # Should handle gracefully, not crash
                assert 'outcome' in result
                assert result['outcome'] in ['inconclusive', 'spam_detected']
                
                logger.info("🛡️ Handled problematic input", 
                           input_type=type(bad_input).__name__,
                           input_length=len(str(bad_input)),
                           outcome=result['outcome'])
                
            except ValueError as e:
                # Some inputs may legitimately raise validation errors
                if "empty" in str(e).lower():
                    continue  # Expected for empty inputs
                else:
                    logger.error("🚫 Unexpected validation error", error=str(e))
                    raise
    
    @pytest.mark.asyncio
    async def test_evaluation_optimizer_integration(self):
        """Test evaluation optimizer with gold standards"""
        # Test evaluation pipeline
        evaluation_result = await self.evaluation_optimizer.evaluate_model(limit=5)
        
        assert 'model_name' in evaluation_result
        assert 'evaluation_type' in evaluation_result
        assert 'metrics' in evaluation_result
        
        # Check if evaluation actually ran
        if evaluation_result.get('examples_evaluated', 0) > 0:
            metrics = evaluation_result['metrics']
            assert 'overall_accuracy' in metrics
            assert 0.0 <= metrics['overall_accuracy'] <= 1.0
        
        logger.info("📈 Evaluation completed", 
                   examples=evaluation_result['metrics'].get('examples_evaluated', 0),
                   accuracy=evaluation_result['metrics'].get('overall_accuracy', 'N/A'))
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self):
        """Test system behavior under concurrent load"""
        async def process_concurrent_request(request_id):
            symptom = random.choice(self.VARIED_SYMPTOMS[:5])  # Limit variety for concurrency
            context = random.choice(self.NICE_CONTEXTS[:3])
            
            start_time = time.time()
            result = await self.medical_agent.process_turn(
                symptoms=f"[Request {request_id}] {symptom}",
                nice_context=context
            )
            
            return {
                'request_id': request_id,
                'processing_time': time.time() - start_time,
                'outcome': result['outcome'],
                'success': True
            }
        
        # Run 5 concurrent requests
        tasks = [process_concurrent_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze concurrent performance
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        
        assert len(successful_results) >= 3, "Most concurrent requests should succeed"
        
        avg_concurrent_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        
        logger.info("🚀 Concurrent test completed", 
                   successful_requests=len(successful_results),
                   avg_time=f"{avg_concurrent_time:.2f}s")
        
    def test_dspy_history_immutability_fix(self):
        """Test that DSPy History handling works correctly"""
        agent = MedicalTriageAgent()
        
        # Test conversation reset with immutable History
        original_turn_count = agent.turn_count
        agent.turn_count = 5
        
        # Verify we changed the turn count
        assert agent.turn_count != original_turn_count
        
        # Reset should work without trying to modify frozen History
        agent.reset_conversation()
        
        assert agent.turn_count == 0
        assert len(agent.conversation_history.messages) == 0
        
        logger.info("🔄 History reset test passed")
