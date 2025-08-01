"""
Unit Tests for DSPy Medical Agent
Tests medical reasoning, conversation state, and DSPy integration
Production-grade testing with proper mocking and error scenarios
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import dspy
from typing import Dict, Any


# Test environment setup
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'DATABASE_URL': 'postgresql+asyncpg://test:test@localhost/test',
    'REDIS_URL': 'redis://localhost:6379/0',
    'MINIO_ENDPOINT': 'localhost:9000',
    'MINIO_ACCESS_KEY': 'test',
    'MINIO_SECRET_KEY': 'test',
    'OLLAMA_BASE_URL': 'http://localhost:11434',
    'RAVEN_WEBHOOK_URL': 'http://localhost:8080/webhook',
    'RAVEN_API_KEY': 'test-key',
    'RAVEN_SECRET': 'test-secret',
    'JWT_SECRET_KEY': 'jwt-secret',
    'CELERY_BROKER_URL': 'redis://localhost:6379/1',
    'CELERY_RESULT_BACKEND': 'redis://localhost:6379/2'
}):
    from src.app2.services.dspy.medical_agent import (
        MedicalTriageAgent,
        MedicalTriageSignature,
        MedicalOutcome,
        ConversationTurn
    )


@pytest.fixture
def mock_dspy_response():
    """Mock DSPy prediction response"""
    response = Mock()
    response.outcome_classification = "emergency"
    response.confidence_score = 85
    response.next_question = "Do you have chest pain radiating to your arm?"
    response.reasoning = "Patient shows signs of possible cardiac emergency"
    response.red_flags = "chest_pain, shortness_of_breath"
    return response


@pytest.fixture
def mock_dspy_complete_response():
    """Mock DSPy response indicating completion"""
    response = Mock()
    response.outcome_classification = "self_care"
    response.confidence_score = 90
    response.next_question = "COMPLETE"
    response.reasoning = "Minor headache, recommend rest and hydration"
    response.red_flags = ""
    return response


@pytest.fixture
async def dspy_warmup_sleep():
    """Add sleep for DSPy first-time initialization"""
    await asyncio.sleep(2)  # 2 second warmup


class TestMedicalTriageAgent:
    """Test DSPy medical agent core functionality"""
    
    @patch('src.app2.services.dspy.medical_agent.dspy.LM')
    @patch('src.app2.services.dspy.medical_agent.dspy.configure')
    def test_agent_initialization_success(self, mock_configure, mock_lm_class):
        """Test successful agent initialization with DSPy configuration"""
        mock_lm_class.return_value = Mock()
        
        agent = MedicalTriageAgent(model_name="test-model")
        
        assert agent.model_name == "test-model"
        assert agent.turn_count == 0
        assert isinstance(agent.conversation_history, dspy.History)
        mock_lm_class.assert_called_once_with(
            'ollama_chat/test-model',
            api_base='http://localhost:11434',
            api_key='',
            thinking=True,
            stream=False
        )
        mock_configure.assert_called_once()
    
    @patch('src.app2.services.dspy.medical_agent.dspy.LM')
    @patch('src.app2.services.dspy.medical_agent.dspy.configure')
    def test_agent_initialization_failure(self, mock_configure, mock_lm_class):
        """Test agent initialization handles DSPy configuration errors"""
        mock_lm_class.side_effect = Exception("DSPy configuration failed")
        
        with pytest.raises(Exception, match="DSPy configuration failed"):
            MedicalTriageAgent()
    
    @pytest.mark.asyncio
    async def test_process_turn_emergency_detection(self, mock_dspy_response, dspy_warmup_sleep):
        """Test medical turn processing with emergency outcome"""
        agent = MedicalTriageAgent()
        
        # Mock the DSPy program
        mock_program_result = {
            'medical_reasoning': mock_dspy_response,
            'emergency_analysis': Mock(is_emergency=True, critical_flags="chest_pain")
        }
        agent.triage_program = Mock(return_value=mock_program_result)

        result = await agent.process_turn(
            symptoms="severe chest pain radiating to left arm",
            nice_context="Chest pain protocol: Consider cardiac emergency"
        )
        
        assert result["outcome"] == "emergency"
        assert result["confidence"] == 85
        assert result["next_question"] == "Do you have chest pain radiating to your arm?"
        assert "chest_pain" in result["red_flags"]
        assert result["is_complete"] is False
        assert agent.turn_count == 1
    
    @pytest.mark.asyncio
    async def test_process_turn_conversation_completion(self, mock_dspy_complete_response):
        """Test conversation completion when agent returns COMPLETE"""
        agent = MedicalTriageAgent()
        
        mock_program_result = {
            'medical_reasoning': mock_dspy_complete_response,
            'emergency_analysis': Mock(is_emergency=False, critical_flags="")
        }
        agent.triage_program = Mock(return_value=mock_program_result)
        
        result = await agent.process_turn(
            symptoms="mild headache",
            nice_context="Headache assessment protocol"
        )
        
        assert result["outcome"] == "self"
        assert result["confidence"] == 90
        assert result["next_question"] is None
        assert result["is_complete"] is True
        assert len(result["red_flags"]) == 0
    
    @pytest.mark.asyncio
    async def test_process_turn_invalid_symptoms(self):
        """Test error handling for empty symptoms"""
        agent = MedicalTriageAgent()
        
        with pytest.raises(ValueError, match="Symptoms cannot be empty"):
            await agent.process_turn(
                symptoms="",
                nice_context="Any protocol"
            )
        
        with pytest.raises(ValueError, match="Symptoms cannot be empty"):
            await agent.process_turn(
                symptoms="   ",  # Whitespace only
                nice_context="Any protocol"
            )
    
    @pytest.mark.asyncio
    async def test_process_turn_dspy_error_handling(self):
        """Test graceful error handling when DSPy program fails"""
        agent = MedicalTriageAgent()
        agent.triage_program = Mock(side_effect=Exception("DSPy model error"))
        
        result = await agent.process_turn(
            symptoms="chest pain",
            nice_context="Chest pain protocol"
        )
        
        # Should return error response instead of crashing
        assert result["outcome"] == "inconclusive"
        assert result["confidence"] == 0
        assert "error" in result["reasoning"].lower()
        assert result["next_question"] is not None
        assert result["is_complete"] is False
    
    def test_parse_response_validation(self):
        """Test response parsing with boundary conditions"""
        agent = MedicalTriageAgent()
        
        # Test confidence bounds
        mock_medical_result = Mock()
        mock_medical_result.outcome_classification = "emergency"
        mock_medical_result.confidence_score = 150  # Above 100
        mock_medical_result.next_question = "Test question"
        mock_medical_result.red_flags = "flag1, flag2"
        
        mock_emergency_result = Mock()
        mock_emergency_result.is_emergency = False
        mock_emergency_result.critical_flags = ""
        
        parsed = agent._parse_dspy_response(mock_medical_result, mock_emergency_result)
        assert parsed["confidence"] == 100  # Clamped to 100
        
        # Test negative confidence
        mock_medical_result.confidence_score = -10
        parsed = agent._parse_dspy_response(mock_medical_result, mock_emergency_result)
        assert parsed["confidence"] == 0  # Clamped to 0
    
    def test_conversation_history_tracking(self):
        """Test conversation history updates correctly"""
        agent = MedicalTriageAgent()
        history = dspy.History(messages=[])
        
        response = {
            "outcome": "inconclusive",
            "confidence": 75,
            "next_question": "Any other symptoms?",
            "reasoning": "Need more information",
            "red_flags": ["mild_concern"]
        }
        
        agent._update_history(history, "test symptoms", response)
        
        assert len(history.messages) == 1
        turn_data = history.messages[0]
        assert turn_data["current_symptoms"] == "test symptoms"
        assert turn_data["outcome_classification"] == "inconclusive"
        assert turn_data["confidence"] == 75
    
    def test_conversation_reset(self):
        """Test conversation state reset functionality"""
        agent = MedicalTriageAgent()
        agent.turn_count = 5
        
        # Mock history with messages
        agent.conversation_history.messages = [{"test": "data"}]
        
        agent.reset_conversation()
        
        assert agent.turn_count == 0
        assert len(agent.conversation_history.messages) == 0
        assert isinstance(agent.conversation_history, dspy.History)
    
    def test_conversation_summary(self):
        """Test conversation summary generation"""
        agent = MedicalTriageAgent()
        agent.turn_count = 3
        
        # Mock history with messages
        mock_history = Mock()
        mock_history.messages = [
            {"outcome_classification": "inconclusive"},
            {"outcome_classification": "routine"},
            {"outcome_classification": "emergency"}
        ]
        agent.conversation_history = mock_history
        
        summary = agent.get_conversation_summary()
        
        assert summary["turns"] == 3
        assert summary["history_length"] == 3
        assert summary["last_outcome"] == "emergency"


class TestMedicalTriageSignature:
    """Test DSPy signature structure and field validation"""
    
    def test_signature_fields_defined(self):
        """Test that all required fields are properly defined"""
        # DSPy signatures have fields in __annotations__
        fields = MedicalTriageSignature.__annotations__
        
        # Input fields
        assert 'current_symptoms' in fields
        assert 'conversation_history' in fields
        assert 'nice_protocols' in fields
        
        # Output fields  
        assert 'outcome_classification' in fields
        assert 'confidence_score' in fields
        assert 'next_question' in fields
        assert 'medical_reasoning' in fields
        assert 'red_flags' in fields


class TestMedicalOutcomeEnum:
    """Test medical outcome enumeration"""
    
    def test_medical_outcome_values(self):
        """Test all expected medical outcomes are defined"""
        expected_enum_names = [
            "EMERGENCY",
            "ROUTINE_DOCTOR",
            "SELF_CARE", 
            "INCONCLUSIVE",
            "SPAM_DETECTED"
        ]
        
        for enum_name in expected_enum_names:
            assert hasattr(MedicalOutcome, enum_name)
    
    def test_medical_outcome_string_values(self):
        """Test medical outcome enum string values"""
        assert MedicalOutcome.EMERGENCY.value == "emergency_route_to_doctor"
        assert MedicalOutcome.ROUTINE_DOCTOR.value == "routine_doctor_consultation"
        assert MedicalOutcome.SELF_CARE.value == "self_care_advice"
