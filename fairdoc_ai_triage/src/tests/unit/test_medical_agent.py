"""
Comprehensive test suite for Medical Triage Agent
Test-driven development approach with mocking for isolated testing
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import dspy
from src.app2.services.dspy.medical_agent import (
    MedicalTriageAgent, 
    MedicalOutcome,
    MedicalTriageSignature,
    ConversationTurn
)

class TestMedicalTriageAgent:
    """Test suite for Medical Triage Agent"""
    
    @pytest.fixture
    def mock_dspy_result(self):
        """Mock DSPy prediction result"""
        mock_result = Mock()
        mock_result.outcome_classification = "emergency"
        mock_result.confidence_score = 85
        mock_result.next_question = "Do you have chest pain radiating to your arm?"
        mock_result.reasoning = "Patient reports chest pain with concerning symptoms"
        mock_result.red_flags = "chest_pain, shortness_of_breath"
        return mock_result
    
    @pytest.fixture 
    def agent(self):
        """Create agent instance with mocked DSPy"""
        with patch('dspy.configure'), \
             patch('dspy.LM'), \
             patch('dspy.ChainOfThought'):
            return MedicalTriageAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.model_name == "deepseek-r1:8b"
        assert agent.turn_count == 0
        assert len(agent.conversation_history.messages) == 0
    
    def test_custom_model_initialization(self):
        """Test agent with custom model"""
        with patch('dspy.configure'), \
             patch('dspy.LM'), \
             patch('dspy.ChainOfThought'):
            agent = MedicalTriageAgent(model_name="llama3:8b")
            assert agent.model_name == "llama3:8b"
    
    @pytest.mark.asyncio
    async def test_process_turn_basic(self, agent, mock_dspy_result):
        """Test basic turn processing"""
        agent.predict = Mock(return_value=mock_dspy_result)
        
        result = await agent.process_turn(
            symptoms="I have chest pain",
            nice_context="Chest pain protocol"
        )
        
        assert result["outcome"] == "emergency"
        assert result["confidence"] == 85
        assert result["next_question"] == "Do you have chest pain radiating to your arm?"
        assert "chest_pain" in result["red_flags"]
        assert not result["is_complete"]
        assert agent.turn_count == 1
    
    @pytest.mark.asyncio
    async def test_process_turn_complete(self, agent):
        """Test turn processing when conversation is complete"""
        mock_result = Mock()
        mock_result.outcome_classification = "self_care"
        mock_result.confidence_score = 75
        mock_result.next_question = "COMPLETE"
        mock_result.reasoning = "Minor headache, recommend rest"
        mock_result.red_flags = ""
        
        agent.predict = Mock(return_value=mock_result)
        
        result = await agent.process_turn(
            symptoms="Mild headache, feel better now",
            nice_context="Headache protocol"
        )
        
        assert result["outcome"] == "self_care"
        assert result["is_complete"] is True
        assert result["next_question"] is None
    
    @pytest.mark.asyncio
    async def test_empty_symptoms_error(self, agent):
        """Test error handling for empty symptoms"""
        with pytest.raises(ValueError, match="Symptoms cannot be empty"):
            await agent.process_turn("", "protocol")
        
        with pytest.raises(ValueError, match="Symptoms cannot be empty"):
            await agent.process_turn("   ", "protocol")
    
    @pytest.mark.asyncio
    async def test_dspy_prediction_error(self, agent):
        """Test error handling when DSPy prediction fails"""
        agent.predict = Mock(side_effect=Exception("Model error"))
        
        result = await agent.process_turn(
            symptoms="chest pain",
            nice_context="protocol"
        )
        
        assert result["outcome"] == "inconclusive"
        assert result["confidence"] == 0
        assert "error" in result["reasoning"].lower()
        assert result["is_complete"] is False
    
    def test_parse_response_valid(self, agent):
        """Test parsing valid DSPy response"""
        mock_result = Mock()
        mock_result.outcome_classification = "routine"
        mock_result.confidence_score = 70
        mock_result.next_question = "How long have you had these symptoms?"
        mock_result.reasoning = "Non-urgent symptoms"
        mock_result.red_flags = "fever, persistent_cough"
        
        result = agent._parse_response(mock_result)
        
        assert result["outcome"] == "routine"
        assert result["confidence"] == 70
        assert result["red_flags"] == ["fever", "persistent_cough"]
        assert not result["is_complete"]
    
    def test_parse_response_invalid_outcome(self, agent):
        """Test parsing response with invalid outcome"""
        mock_result = Mock()
        mock_result.outcome_classification = "invalid_outcome"
        mock_result.confidence_score = 80
        mock_result.next_question = "Question"
        mock_result.reasoning = "Reasoning"
        mock_result.red_flags = ""
        
        result = agent._parse_response(mock_result)
        
        assert result["outcome"] == "inconclusive"  # Should default
    
    def test_parse_response_invalid_confidence(self, agent):
        """Test parsing response with invalid confidence"""
        mock_result = Mock()
        mock_result.outcome_classification = "emergency"
        mock_result.confidence_score = "invalid"
        mock_result.next_question = "Question"
        mock_result.reasoning = "Reasoning"
        mock_result.red_flags = ""
        
        result = agent._parse_response(mock_result)
        
        assert result["confidence"] == 50  # Should default
    
    def test_parse_response_confidence_bounds(self, agent):
        """Test confidence score boundary handling"""
        mock_result = Mock()
        mock_result.outcome_classification = "emergency"
        mock_result.next_question = "Question"
        mock_result.reasoning = "Reasoning"
        mock_result.red_flags = ""
        
        # Test upper bound
        mock_result.confidence_score = 150
        result = agent._parse_response(mock_result)
        assert result["confidence"] == 100
        
        # Test lower bound
        mock_result.confidence_score = -10
        result = agent._parse_response(mock_result) 
        assert result["confidence"] == 0
    
    def test_update_history(self, agent):
        """Test conversation history updates correctly"""
        history = dspy.History(messages=[])
        response = {
            "outcome": "emergency",
            "confidence": 85,
            "next_question": "Next question?",
            "reasoning": "Medical reasoning",
            "red_flags": ["chest_pain"]
        }
        
        agent.turn_count = 1
        agent._update_history(history, "chest pain", response)
        
        assert len(history.messages) == 1
        turn = history.messages[0]
        assert turn["turn"] == 1
        assert turn["current_symptoms"] == "chest pain"
        assert turn["outcome_classification"] == "emergency"
    
    def test_reset_conversation(self, agent):
        """Test conversation reset"""
        # Simulate some conversation history
        agent.turn_count = 3
        agent.conversation_history.messages = [{"test": "data"}]
        
        agent.reset_conversation()
        
        assert agent.turn_count == 0
        assert len(agent.conversation_history.messages) == 0
    
    def test_get_conversation_summary_empty(self, agent):
        """Test conversation summary when empty"""
        summary = agent.get_conversation_summary()
        
        assert summary["turns"] == 0
        assert summary["history_length"] == 0
        assert summary["last_outcome"] is None
    
    def test_get_conversation_summary_with_history(self, agent):
        """Test conversation summary with history"""
        agent.turn_count = 2
        agent.conversation_history.messages = [
            {"outcome_classification": "inconclusive"},
            {"outcome_classification": "routine"}
        ]
        
        summary = agent.get_conversation_summary()
        
        assert summary["turns"] == 2
        assert summary["history_length"] == 2
        assert summary["last_outcome"] == "routine"
    
    def test_create_error_response(self, agent):
        """Test error response creation"""
        error_response = agent._create_error_response("Test error")
        
        assert error_response["outcome"] == "inconclusive"
        assert error_response["confidence"] == 0
        assert "error" in error_response["reasoning"].lower()
        assert error_response["is_complete"] is False
        assert isinstance(error_response["red_flags"], list)

class TestMedicalTriageSignature:
    """Test DSPy signature definition"""
    
    def test_signature_fields(self):
        """Test signature has correct fields"""
        signature = MedicalTriageSignature
        
        # Test input fields exist
        assert hasattr(signature, 'current_symptoms')
        assert hasattr(signature, 'conversation_history') 
        assert hasattr(signature, 'nice_protocols')
        
        # Test output fields exist
        assert hasattr(signature, 'outcome_classification')
        assert hasattr(signature, 'confidence_score')
        assert hasattr(signature, 'next_question')
        assert hasattr(signature, 'reasoning')
        assert hasattr(signature, 'red_flags')

class TestMedicalOutcome:
    """Test medical outcome enumeration"""
    
    def test_outcome_values(self):
        """Test all expected outcomes are defined"""
        assert MedicalOutcome.EMERGENCY == "emergency_route_to_doctor"
        assert MedicalOutcome.ROUTINE_DOCTOR == "routine_doctor_consultation"
        assert MedicalOutcome.SELF_CARE == "self_care_advice"
        assert MedicalOutcome.INCONCLUSIVE == "need_more_questions"
        assert MedicalOutcome.SPAM_DETECTED == "spam_or_irrelevant"
    
    def test_outcome_count(self):
        """Test expected number of outcomes"""
        assert len(list(MedicalOutcome)) == 5

# Integration test markers
@pytest.mark.integration 
class TestMedicalTriageIntegration:
    """Integration tests requiring actual DSPy/Ollama setup"""
    
    @pytest.mark.skip(reason="Requires Ollama running locally")
    @pytest.mark.asyncio
    async def test_real_conversation_flow(self):
        """Test real conversation with Ollama"""
        agent = MedicalTriageAgent()
        
        # First turn
        result1 = await agent.process_turn(
            symptoms="I have a severe headache that started suddenly",
            nice_context="NICE headache protocol"
        )
        
        assert result1["outcome"] in ["emergency", "routine", "inconclusive"]
        assert 0 <= result1["confidence"] <= 100
        
        # Second turn if not complete
        if not result1["is_complete"]:
            result2 = await agent.process_turn(
                symptoms="Yes, it's the worst headache I've ever had",
                nice_context="NICE headache protocol"
            )
            
            assert isinstance(result2, dict)
            assert "outcome" in result2
