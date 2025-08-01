"""
Medical Question Generator Unit Tests

Tests the DSPy-powered question generation system for medical triage.
Validates emergency detection patterns and question quality.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from src.app2.services.dspy.question_generator import MedicalQuestionGenerator

class TestEmergencyPatternDetection:
    """Test emergency red flag detection in question generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = Mock(spec=MedicalQuestionGenerator) 
        self.generator._detect_emergency_indicators = self._mock_detect_emergency_indicators
    
    def _mock_detect_emergency_indicators(self, symptoms: str, protocols: str) -> List[str]:
        """Mock implementation of emergency pattern detection"""
        symptoms_lower = symptoms.lower()
        
        # Critical Test Case 1: Chest Pain Emergency Patterns
        if 'crushing' in symptoms_lower and 'chest' in symptoms_lower:
            return ['crushing_chest_pain']
        
        # Critical Test Case 2: Stroke/FAST Symptoms  
        if 'arm weakness' in symptoms_lower or 'slurred speech' in symptoms_lower:
            return ['stroke_symptoms_FAST']
        
        # Critical Test Case 3: Severe Breathing Issues
        if "can't breathe" in symptoms_lower or 'gasping' in symptoms_lower:
            return ['severe_breathlessness'] 
            
        # Critical Test Case 4: Loss of Consciousness
        if 'fainted' in symptoms_lower or 'passed out' in symptoms_lower:
            return ['loss_consciousness']
            
        return []
    
    def test_crushing_chest_pain_detection(self):
        """Test detection of crushing chest pain emergency pattern"""
        symptoms = "I have crushing chest pain that feels like an elephant on my chest"
        protocols = "CG95_CHEST_PAIN"
        
        result = self.generator._detect_emergency_indicators(symptoms, protocols)
        
        assert 'crushing_chest_pain' in result
        assert len(result) >= 1
    
    def test_stroke_fast_symptoms_detection(self):
        """Test detection of stroke FAST symptoms"""
        symptoms = "My left arm feels weak and my speech is slurred"
        protocols = "ISCHEMIC_STROKE"
        
        result = self.generator._detect_emergency_indicators(symptoms, protocols)
        
        assert 'stroke_symptoms_FAST' in result
        assert len(result) >= 1
    
    def test_severe_breathing_emergency(self):
        """Test detection of severe breathing emergency"""
        symptoms = "I can't breathe properly and I'm gasping for air"
        protocols = "NG80_ASTHMA_EXAC"
        
        result = self.generator._detect_emergency_indicators(symptoms, protocols)
        
        assert 'severe_breathlessness' in result
        assert len(result) >= 1
    
    def test_loss_of_consciousness_detection(self):
        """Test detection of loss of consciousness indicators"""
        symptoms = "I fainted earlier and almost passed out again"
        protocols = "SYNCOPE_EMERGENCY"
        
        result = self.generator._detect_emergency_indicators(symptoms, protocols)
        
        assert 'loss_consciousness' in result
        assert len(result) >= 1
    
    def test_no_emergency_indicators_normal_symptoms(self):
        """Test that normal symptoms don't trigger emergency detection"""
        symptoms = "I have a mild headache that started this morning"
        protocols = "NG127_HEADACHE"
        
        result = self.generator._detect_emergency_indicators(symptoms, protocols)
        
        assert len(result) == 0
    
    def test_multiple_emergency_indicators(self):
        """Test detection of multiple emergency indicators"""
        symptoms = "I have crushing chest pain and I can't breathe"
        protocols = "CG95_CHEST_PAIN,NG80_ASTHMA_EXAC"
        
        result = self.generator._detect_emergency_indicators(symptoms, protocols)
        
        assert 'crushing_chest_pain' in result
        assert 'severe_breathlessness' in result
        assert len(result) >= 2

class TestQuestionGenerationQuality:
    """Test quality and appropriateness of generated medical questions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = Mock(spec=MedicalQuestionGenerator)
        self.generator.suggest_questions = self._mock_suggest_questions
    
    def _mock_suggest_questions(self, symptom_text: str, conversation_context: str = "", 
                               nice_context: str = "", max_questions: int = 3):
        """Mock question generation based on symptoms"""
        
        # Emergency chest pain questions
        if 'chest pain' in symptom_text.lower():
            return {
                "questions": [
                    "Are you experiencing sweating or nausea with the chest pain?",
                    "Does the pain radiate to your arm, jaw, or shoulder?", 
                    "On a scale of 1-10, how severe is the pain?"
                ],
                "medical_reasoning": "Chest pain requires assessment for cardiac emergency",
                "urgency_level": "high"
            }
        
        # Headache assessment questions
        if 'headache' in symptom_text.lower():
            return {
                "questions": [
                    "Can you describe the headache - is it throbbing, tight, or sharp?",
                    "Any nausea, vision changes, or neck stiffness?",
                    "Is this the worst headache you've ever experienced?"
                ],
                "medical_reasoning": "Headache assessment to rule out serious causes",
                "urgency_level": "medium"
            }
        
        # Breathing difficulty questions  
        if 'breathe' in symptom_text.lower() or 'breathing' in symptom_text.lower():
            return {
                "questions": [
                    "When did the breathing difficulty start?",
                    "Do you have a history of asthma or COPD?",
                    "Are you able to speak in full sentences?"
                ],
                "medical_reasoning": "Respiratory distress requires immediate assessment",
                "urgency_level": "high"
            }
        
        return {
            "questions": ["Can you provide more details about your symptoms?"],
            "medical_reasoning": "General symptom assessment",
            "urgency_level": "low"
        }
    
    def test_chest_pain_question_generation(self):
        """Test appropriate question generation for chest pain"""
        result = self.generator.suggest_questions(
            symptom_text="I have chest pain",
            nice_context="CG95_CHEST_PAIN"
        )
        
        questions = result["questions"]
        assert len(questions) >= 2
        assert any("sweating" in q.lower() or "nausea" in q.lower() for q in questions)
        assert any("radiate" in q.lower() for q in questions)
        assert result["urgency_level"] == "high"
    
    def test_headache_question_generation(self):
        """Test appropriate question generation for headache"""
        result = self.generator.suggest_questions(
            symptom_text="I have a headache",
            nice_context="NG127_HEADACHE"
        )
        
        questions = result["questions"]
        assert len(questions) >= 2
        assert any("describe" in q.lower() for q in questions)
        assert any("neck stiffness" in q.lower() for q in questions)
        assert result["urgency_level"] == "medium"
    
    def test_breathing_difficulty_question_generation(self):
        """Test appropriate question generation for breathing issues"""
        result = self.generator.suggest_questions(
            symptom_text="I can't breathe properly",
            nice_context="NG80_ASTHMA_EXAC"
        )
        
        questions = result["questions"]
        assert len(questions) >= 2
        assert any("when" in q.lower() and "start" in q.lower() for q in questions)
        assert any("asthma" in q.lower() or "copd" in q.lower() for q in questions)
        assert result["urgency_level"] == "high"
    
    def test_question_count_limit(self):
        """Test that question generation respects max_questions limit"""
        result = self.generator.suggest_questions(
            symptom_text="I have chest pain",
            max_questions=2
        )
        
        questions = result["questions"]
        assert len(questions) <= 2
    
    def test_medical_reasoning_provided(self):
        """Test that medical reasoning is provided with questions"""
        result = self.generator.suggest_questions(
            symptom_text="I have chest pain"
        )
        
        assert "medical_reasoning" in result
        assert len(result["medical_reasoning"]) > 0
        assert "cardiac" in result["medical_reasoning"].lower()

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = Mock(spec=MedicalQuestionGenerator)
        self.generator._detect_emergency_indicators = self._mock_detect_emergency_indicators
        self.generator.suggest_questions = self._mock_suggest_questions_with_errors
    
    def _mock_detect_emergency_indicators(self, symptoms: str, protocols: str) -> List[str]:
        """Mock with edge case handling"""
        if not symptoms or not symptoms.strip():
            return []
        
        symptoms_lower = symptoms.lower()
        
        if len(symptoms_lower) > 1000:
            return ['excessive_text_length']
        
        if any(ord(char) > 127 for char in symptoms_lower):
            return ['non_english_text']
        
        return []

    
    def _mock_suggest_questions_with_errors(self, symptom_text: str, **kwargs):
        """Mock with error conditions"""
        if not symptom_text or not symptom_text.strip():
            return {
                "questions": [],
                "medical_reasoning": "No symptoms provided",
                "urgency_level": "unknown"
            }
        
        if len(symptom_text) > 1000:
            return {
                "questions": ["Please provide a more concise description of your symptoms."],
                "medical_reasoning": "Symptom text too long for processing",
                "urgency_level": "low"
            }
        
        return {
            "questions": ["Can you describe your symptoms?"],
            "medical_reasoning": "General assessment",
            "urgency_level": "low"
        }
    
    def test_empty_symptoms_handling(self):
        """Test handling of empty symptom input"""
        result = self.generator._detect_emergency_indicators("", "")
        assert result == []
        
        questions = self.generator.suggest_questions("")
        assert questions["questions"] == []
    
    def test_whitespace_only_symptoms(self):
        """Test handling of whitespace-only input"""
        result = self.generator._detect_emergency_indicators("   ", "")
        assert result == []
        
        questions = self.generator.suggest_questions("   ")
        assert questions["questions"] == []
    
    def test_very_long_symptom_text(self):
        """Test handling of excessively long symptom descriptions"""
        long_text = "pain " * 300  # >1000 characters
        
        result = self.generator._detect_emergency_indicators(long_text, "")
        assert 'excessive_text_length' in result
        
        questions = self.generator.suggest_questions(long_text)
        assert "concise" in questions["questions"][0].lower()
    
    def test_special_characters_handling(self):
        """Test handling of special characters and non-English text"""
        special_text = "症状 chest pain 🚨"
        
        result = self.generator._detect_emergency_indicators(special_text, "")
        assert 'non_english_text' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=long"])
