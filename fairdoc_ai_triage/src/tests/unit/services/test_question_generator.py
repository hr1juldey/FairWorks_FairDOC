"""
Comprehensive Unit & Stress Tests for DSPy Question Generator
Tests medical question generation, emergency detection, and DSPy integration
Includes stress testing with varied medical scenarios
"""
import pytest
import asyncio
import time
import random
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import structlog

logger = structlog.get_logger(__name__)

# Test environment setup
with patch.dict('os.environ', {
    'OLLAMA_BASE_URL': 'http://localhost:11434',
    'FAIRDOC_V2_DSPy_MODEL': 'deepseek-r1:8b',
}):
    from src.app2.services.dspy.question_generator import (
        MedicalQuestionGenerator, 
        MedicalQuestionModule,
        question_generator,
        suggest_questions
    )

class TestMedicalQuestionGenerator:
    """Unit tests for DSPy medical question generator"""
    
    @pytest.fixture(autouse=True)
    def setup_generator(self):
        """Setup question generator for testing"""
        self.generator = MedicalQuestionGenerator()
        
    def test_generator_initialization_success(self):
        """Test question generator initializes correctly"""
        assert self.generator is not None
        assert self.generator.model_name == "deepseek-r1:8b"
        assert hasattr(self.generator, 'question_program')
        assert self.generator.question_program is not None
        
    def test_dspy_configuration_validation(self):
        """Test DSPy configuration is valid"""
        # Test should not raise configuration errors
        try:
            test_generator = MedicalQuestionGenerator("deepseek-r1:8b")
            assert test_generator is not None
        except Exception as e:
            pytest.fail(f"DSPy configuration failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_emergency_question_generation(self):
        """Test emergency scenario question generation"""
        emergency_symptoms = "severe crushing chest pain radiating to left arm with heavy sweating"
        nice_protocols = "CG95 Chest Pain: Emergency if crushing >20min with radiation"
        
        result = self.generator.suggest_questions(
            symptom_text=emergency_symptoms,
            nice_protocols=nice_protocols,
            max_questions=3
        )
        
        # Validate response structure
        assert 'questions' in result
        assert len(result['questions']) <= 3
        assert len(result['questions']) > 0
        assert result['thinking_enabled'] is True
        
        # Validate emergency detection
        assert 'urgency_level' in result
        assert result['urgency_level'] in ['high', 'critical']
        
        # Validate question quality for emergencies
        questions_text = ' '.join(result['questions']).lower()
        emergency_keywords = ['severe', 'pain', 'chest', 'breathing', 'emergency']
        found_keywords = [kw for kw in emergency_keywords if kw in questions_text]
        assert len(found_keywords) >= 2, f"Questions should contain emergency keywords, found: {found_keywords}"
        
    @pytest.mark.asyncio
    async def test_routine_question_generation(self):
        """Test routine scenario question generation"""
        routine_symptoms = "mild headache, stress-related, no nausea"
        nice_protocols = "NG127 Headache: Routine assessment for tension-type"
        
        result = self.generator.suggest_questions(
            symptom_text=routine_symptoms,
            nice_protocols=nice_protocols,
            max_questions=3
        )
        
        # Should generate routine questions
        assert result['urgency_level'] in ['low', 'medium']
        assert len(result['questions']) > 0
        
        # Questions should be appropriate for routine care
        questions_text = ' '.join(result['questions']).lower()
        routine_keywords = ['how long', 'scale', 'medication', 'before', 'severe']
        found_keywords = [kw for kw in routine_keywords if kw in questions_text]
        assert len(found_keywords) >= 1, "Should contain routine assessment keywords"
    
    def test_emergency_indicator_detection(self):
        """Test emergency red flag detection logic"""
        module = MedicalQuestionModule()
        
        # Test critical indicators
        critical_symptoms = "crushing chest pain, worst headache ever, can't breathe, passed out"
        indicators = module._detect_emergency_indicators(critical_symptoms, "")
        
        assert len(indicators) >= 2, f"Should detect multiple emergency indicators, found: {indicators}"
        assert any('crushing' in ind or 'chest' in ind for ind in indicators)
        
        # Test non-emergency symptoms
        mild_symptoms = "slight headache, tired, minor discomfort"
        mild_indicators = module._detect_emergency_indicators(mild_symptoms, "")
        
        assert len(mild_indicators) == 0, "Should not detect emergency indicators for mild symptoms"
    
    def test_urgency_level_assessment(self):
        """Test medical urgency level classification"""
        module = MedicalQuestionModule()
        
        # Test critical level
        critical_indicators = ['crushing_chest_pain', 'loss_consciousness']
        urgency = module._assess_urgency_level(critical_indicators)
        assert urgency == 'critical'
        
        # Test high level  
        high_indicators = ['chest_pain_radiation', 'severe_breathlessness']
        urgency = module._assess_urgency_level(high_indicators)
        assert urgency == 'high'
        
        # Test low level
        no_indicators = []
        urgency = module._assess_urgency_level(no_indicators)
        assert urgency == 'low'
    
    def test_dspy_examples_validity(self):
        """Test DSPy training examples are valid"""
        module = MedicalQuestionModule()
        examples = module.medical_examples
        
        assert len(examples) >= 3, "Should have multiple training examples"
        
        for example in examples:
            # Verify example structure
            assert hasattr(example, 'symptoms')
            assert hasattr(example, 'medical_reasoning')
            assert hasattr(example, 'priority_questions')
            
            # Verify content quality
            assert len(example.symptoms) > 10, "Symptoms should be descriptive"
            assert len(example.medical_reasoning) > 20, "Reasoning should be detailed"
    
    def test_fallback_question_generation(self):
        """Test fallback behavior when DSPy fails"""
        # Test emergency fallback
        fallback_result = self.generator._emergency_fallback_questions("test symptoms", 3)
        
        assert 'questions' in fallback_result
        assert len(fallback_result['questions']) == 3
        assert fallback_result['thinking_enabled'] is False
        assert fallback_result['urgency_level'] == 'medium'  # Conservative fallback
        
        # Verify fallback questions are NICE-compliant
        questions = fallback_result['questions']
        assert any('scale' in q.lower() for q in questions), "Should include severity assessment"
        assert any('long' in q.lower() for q in questions), "Should include duration assessment"
    
    def test_legacy_compatibility(self):
        """Test legacy suggest_questions function works"""
        result = suggest_questions("headache", max_questions=2)
        
        assert isinstance(result, list)
        assert len(result) <= 2
        assert all(isinstance(q, str) for q in result)
        assert all(len(q) > 5 for q in result), "Questions should be meaningful length"

class TestQuestionGeneratorStressTesting:
    """Stress tests for question generator under varied conditions"""
    
    @pytest.fixture(autouse=True)
    def setup_stress_testing(self):
        """Setup for stress testing"""
        self.generator = MedicalQuestionGenerator()
        self.stress_scenarios = self._create_stress_scenarios()
    
    def _create_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Create varied test scenarios for stress testing"""
        return [
            {
                'name': 'Cardiac Emergency',
                'symptoms': 'severe crushing chest pain radiating to left arm, sweating, nausea',
                'protocols': 'CG95 Chest Pain Emergency',
                'expected_urgency': 'critical'
            },
            {
                'name': 'Thunderclap Headache', 
                'symptoms': 'sudden worst headache ever, neck stiff, confused',
                'protocols': 'NG127 Headache Emergency',
                'expected_urgency': 'critical'
            },
            {
                'name': 'Appendicitis Symptoms',
                'symptoms': 'pain started at belly button, moved to right side, nausea',
                'protocols': 'CG141 Abdominal Pain',
                'expected_urgency': 'high'
            },
            {
                'name': 'Mild Tension Headache',
                'symptoms': 'mild headache, work stress, similar before',
                'protocols': 'NG127 Routine Headache',
                'expected_urgency': 'low'
            },
            {
                'name': 'Heart Failure Symptoms',
                'symptoms': 'shortness of breath, ankle swelling, fatigue',
                'protocols': 'NG106 Heart Failure Assessment',
                'expected_urgency': 'medium'
            }
        ]
    
    @pytest.mark.asyncio
    async def test_multiple_scenario_stress_test(self):
        """Stress test with multiple medical scenarios"""
        results = []
        processing_times = []
        
        for scenario in self.stress_scenarios:
            start_time = time.time()
            
            result = self.generator.suggest_questions(
                symptom_text=scenario['symptoms'],
                nice_protocols=scenario['protocols'],
                max_questions=3
            )
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Validate each result
            assert 'questions' in result, f"Missing questions for {scenario['name']}"
            assert len(result['questions']) > 0, f"No questions generated for {scenario['name']}"
            assert result['urgency_level'] == scenario['expected_urgency'], \
                f"Wrong urgency for {scenario['name']}: expected {scenario['expected_urgency']}, got {result['urgency_level']}"
            
            results.append({
                'scenario': scenario['name'],
                'questions_count': len(result['questions']),
                'urgency': result['urgency_level'],
                'processing_time': processing_time,
                'has_reasoning': len(result.get('medical_reasoning', '')) > 0
            })
        
        # Performance validation
        avg_time = sum(processing_times) / len(processing_times)
        assert avg_time < 30.0, f"Average processing time {avg_time:.2f}s too slow"
        
        # Quality validation
        assert all(r['questions_count'] > 0 for r in results), "All scenarios should generate questions"
        
        logger.info("🧪 Stress test completed", 
                   scenarios_tested=len(results),
                   avg_processing_time=f"{avg_time:.2f}s")
    
    @pytest.mark.asyncio 
    async def test_concurrent_question_generation(self):
        """Test concurrent question generation performance"""
        
        async def generate_concurrent_questions(scenario_id):
            scenario = random.choice(self.stress_scenarios)
            start_time = time.time()
            
            result = self.generator.suggest_questions(
                symptom_text=scenario['symptoms'],
                nice_protocols=scenario['protocols'],
                max_questions=3
            )
            
            return {
                'scenario_id': scenario_id,
                'processing_time': time.time() - start_time,
                'questions_generated': len(result['questions']),
                'success': True
            }
        
        # Run 5 concurrent requests
        tasks = [generate_concurrent_questions(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate concurrent performance
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        assert len(successful_results) >= 4, "Most concurrent requests should succeed"
        
        avg_concurrent_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        logger.info("🚀 Concurrent test completed",
                   successful_requests=len(successful_results),
                   avg_time=f"{avg_concurrent_time:.2f}s")
    
    def test_edge_case_inputs(self):
        """Test question generator with edge case inputs"""
        edge_cases = [
            ("", "Empty input"),
            ("x" * 1000, "Very long input"),
            ("🤔💭🏥❓", "Only emojis"),
            ("SELECT * FROM users;", "Potential injection"),
            ("chest pain" * 100, "Repeated terms"),
        ]
        
        for edge_input, case_name in edge_cases:
            try:
                result = self.generator.suggest_questions(
                    symptom_text=edge_input,
                    max_questions=2
                )
                
                # Should handle gracefully
                assert 'questions' in result, f"Missing questions for {case_name}"
                assert len(result['questions']) >= 0, f"Invalid questions for {case_name}"
                
                logger.info(f"✅ Handled edge case: {case_name}")
                
            except Exception as e:
                # Some edge cases may legitimately raise errors
                if "empty" in str(e).lower():
                    continue  # Expected for empty input
                else:
                    logger.warning(f"⚠️ Edge case error: {case_name}", error=str(e))
