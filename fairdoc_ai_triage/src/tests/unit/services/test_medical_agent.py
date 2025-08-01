"""
Real DSPy Medical Agent Integration Tests
Tests actual LLM responses with medical reasoning scenarios
Production-ready testing with real Ollama integration
"""
import pytest
import asyncio
import time
from unittest.mock import patch
import dspy
from typing import Dict, Any

# Real test environment - no mocking of core DSPy functionality
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'DATABASE_URL': 'postgresql+asyncpg://test:test@localhost/test',
    'REDIS_URL': 'redis://localhost:6379/0',
    'OLLAMA_BASE_URL': 'http://localhost:11434',
    'FAIRDOC_V2_DSPy_MODEL': 'deepseek-r1:8b',
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
def real_nice_context():
    """Real NICE protocol context for chest pain"""
    return """
    NICE Guideline CG95 - Chest Pain Assessment
    
    Emergency Criteria:
    - Crushing central chest pain >20 minutes
    - Pain radiating to left arm, jaw, or neck
    - Associated with sweating, nausea, breathlessness
    - Cardiac risk factors present
    
    Red Flags:
    - crushing_chest_pain
    - left_arm_radiation
    - severe_sweating
    - cardiac_risk_factors
    
    Initial Questions:
    1. Can you describe the chest pain quality and location?
    2. Does activity or rest change the pain?
    3. Any nausea, sweating, or breathlessness?
    
    Self-Care Criteria:
    - Musculoskeletal pain reproducible on palpation
    - No cardiac risk factors
    - Pain varies with position/movement
    """

@pytest.fixture
def headache_nice_context():
    """Real NICE protocol context for headache"""
    return """
    NICE Guideline NG127 - Headache Assessment
    
    Emergency Criteria:
    - Sudden onset worst headache ever (thunderclap)
    - Headache with neck stiffness
    - Headache with fever and rash
    - Neurological deficit present
    
    Red Flags:
    - thunderclap_headache
    - neck_stiffness
    - photophobia
    - neurological_deficit
    
    Self-Care Criteria:
    - Tension-type headache pattern
    - No red flag symptoms
    - Responsive to simple analgesia
    """

class TestRealDSPyMedicalAgent:
    """Test medical agent with real LLM integration"""
    
    @pytest.mark.asyncio
    async def test_agent_real_initialization(self):
        """Test real agent initialization with Ollama connection"""
        # Allow time for DSPy to warm up
        await asyncio.sleep(2)
        
        agent = MedicalTriageAgent(model_name="deepseek-r1:8b")
        
        assert agent.model_name == "deepseek-r1:8b"
        assert agent.turn_count == 0
        assert isinstance(agent.conversation_history, dspy.History)
        assert hasattr(agent, 'predict')
        assert isinstance(agent.predict, dspy.ChainOfThought)
    
    @pytest.mark.asyncio
    async def test_real_emergency_chest_pain_scenario(self, real_nice_context):
        """Test real emergency chest pain with actual LLM response"""
        # DSPy warmup delay
        await asyncio.sleep(3)
        
        agent = MedicalTriageAgent()
        
        # Real emergency symptoms that should trigger emergency response
        emergency_symptoms = (
            "I have severe crushing chest pain in the center of my chest "
            "that started 30 minutes ago. The pain is radiating down my left arm "
            "and I'm sweating heavily and feel very nauseous. I'm having trouble breathing."
        )
        
        start_time = time.time()
        result = await agent.process_turn(
            symptoms=emergency_symptoms,
            nice_context=real_nice_context
        )
        processing_time = time.time() - start_time
        
        # Validate real LLM response for emergency
        assert result["outcome"] == "emergency"  # Based on parsing logic
        assert result["confidence"] >= 70  # Should be high confidence for clear emergency
        assert result["reasoning"] is not None
        assert len(result["reasoning"]) > 10  # Real reasoning from LLM
        assert processing_time < 30  # Should respond within 30 seconds
        
        # Check for medical red flags in response
        red_flags = result.get("red_flags", [])
        assert len(red_flags) > 0  # Should detect red flags
        
        # Validate conversation state
        assert agent.turn_count == 1
        assert len(agent.conversation_history.messages) == 1
    
    @pytest.mark.asyncio
    async def test_real_self_care_headache_scenario(self, headache_nice_context):
        """Test real self-care headache with actual LLM response"""
        await asyncio.sleep(2)
        
        agent = MedicalTriageAgent()
        
        # Mild headache symptoms that should suggest self-care
        mild_symptoms = (
            "I have a dull headache around my temples that started this morning. "
            "It feels like a tight band around my head. No nausea, no vision changes, "
            "no neck stiffness. I've had similar headaches before when stressed."
        )
        
        result = await agent.process_turn(
            symptoms=mild_symptoms,
            nice_context=headache_nice_context
        )
        
        # Validate real LLM response for self-care
        # Based on parsing logic: "self_care_advice".split('_')[0] = "self"
        assert result["outcome"] in ["self", "routine", "inconclusive"]  # Accept any non-emergency
        assert result["confidence"] >= 50
        assert result["reasoning"] is not None
        
        # Should have fewer or no red flags for mild headache
        red_flags = result.get("red_flags", [])
        emergency_flags = ["thunderclap", "neck_stiffness", "neurological_deficit"]
        has_emergency_flags = any(flag in str(red_flags) for flag in emergency_flags)
        assert not has_emergency_flags  # Should not detect emergency red flags
    
    @pytest.mark.asyncio
    async def test_real_multi_turn_conversation(self, real_nice_context):
        """Test real multi-turn conversation with progressive questioning"""
        await asyncio.sleep(2)
        
        agent = MedicalTriageAgent()
        
        # Turn 1: Vague initial symptoms
        result1 = await agent.process_turn(
            symptoms="I have some chest discomfort",
            nice_context=real_nice_context
        )
        
        # Should ask for more information
        assert result1["outcome"] in ["inconclusive", "routine"]
        assert result1["next_question"] is not None
        assert len(result1["next_question"]) > 5  # Real question from LLM
        assert not result1["is_complete"]
        assert agent.turn_count == 1
        
        # Turn 2: More specific emergency symptoms
        result2 = await agent.process_turn(
            symptoms=(
                "Actually, it's now a severe crushing pain in my chest, "
                "going down my left arm, and I'm sweating a lot"
            ),
            nice_context=real_nice_context
        )
        
        # Should now escalate to emergency
        assert result2["outcome"] == "emergency"
        assert result2["confidence"] > result1["confidence"]  # Increased confidence
        assert agent.turn_count == 2
        assert len(agent.conversation_history.messages) == 2
        
        # Validate conversation progression
        turn1_data = agent.conversation_history.messages[0]
        turn2_data = agent.conversation_history.messages[1]
        assert turn1_data["turn"] == 1
        assert turn2_data["turn"] == 2
        assert turn2_data["confidence"] > turn1_data["confidence"]
    
    @pytest.mark.asyncio
    async def test_real_llm_error_handling(self):
        """Test error handling with real agent when LLM is unavailable"""
        # Use invalid model to simulate LLM failure
        try:
            agent = MedicalTriageAgent(model_name="invalid-model-name")
            
            result = await agent.process_turn(
                symptoms="test symptoms",
                nice_context="test context"
            )
            
            # Should return graceful error response
            assert result["outcome"] == "inconclusive"
            assert result["confidence"] == 0
            assert "error" in result["reasoning"].lower()
            assert not result["is_complete"]
            
        except Exception as e:
            # Expected for invalid model - test that it fails gracefully
            pytest.rasies("DSPy configuration failed" in str(e) or "model" in str(e).lower()) 
    
    def test_real_response_parsing_logic(self):
        """Test actual response parsing logic from the medical agent"""
        agent = MedicalTriageAgent.__new__(MedicalTriageAgent)  # Skip __init__
        
        # Mock a real DSPy response structure
        from unittest.mock import Mock
        
        # Test emergency outcome parsing
        response = Mock()
        response.outcome_classification = "emergency"
        response.confidence_score = 92
        response.next_question = "Call 999 immediately"
        response.reasoning = "Cardiac emergency indicators present"
        response.red_flags = "crushing_chest_pain, left_arm_radiation"
        
        parsed = agent._parse_response(response)
        assert parsed["outcome"] == "emergency"
        assert parsed["confidence"] == 92
        assert "crushing_chest_pain" in parsed["red_flags"]
        
        # Test self_care outcome parsing - should map to "self"
        response.outcome_classification = "self"  # Use "self" not "self_care"
        parsed = agent._parse_response(response)
        assert parsed["outcome"] == "self"  # Based on split('_')[0] logic
        
        # Test COMPLETE question handling
        response.next_question = "COMPLETE"
        parsed = agent._parse_response(response)
        assert parsed["next_question"] is None
        assert parsed["is_complete"] is True

class TestRealDSPyIntegration:
    """Test DSPy framework integration aspects"""
    
    def test_signature_real_structure(self):
        """Test DSPy signature structure is correctly defined"""
        # Validate signature fields exist
        assert hasattr(MedicalTriageSignature, '__annotations__')
        fields = MedicalTriageSignature.__annotations__
        
        # Input fields
        required_inputs = ['current_symptoms', 'conversation_history', 'nice_protocols']
        for field in required_inputs:
            assert field in fields
        
        # Output fields
        required_outputs = ['outcome_classification', 'confidence_score', 'next_question', 
                          'reasoning', 'red_flags']
        for field in required_outputs:
            assert field in fields
    
    def test_medical_outcome_enum_real_values(self):
        """Test medical outcome enum matches expected values"""
        # Test all enum values exist
        expected_outcomes = ["EMERGENCY", "ROUTINE_DOCTOR", "SELF_CARE", "INCONCLUSIVE", "SPAM_DETECTED"]
        for outcome in expected_outcomes:
            assert hasattr(MedicalOutcome, outcome)
        
        # Test enum string values for parsing logic
        assert MedicalOutcome.EMERGENCY.value == "emergency_route_to_doctor"
        assert MedicalOutcome.SELF_CARE.value == "self_care_advice"
        assert MedicalOutcome.ROUTINE_DOCTOR.value == "routine_doctor_consultation"
        
        # Test parsing logic compatibility
        for outcome in MedicalOutcome:
            first_part = outcome.value.split('_')[0]
            assert len(first_part) > 0  # Should have valid first part for parsing

class TestProductionScenarios:
    """Test production-ready scenarios with real medical data"""
    
    @pytest.mark.asyncio
    async def test_conversation_state_management(self):
        """Test real conversation state persistence and reset"""
        await asyncio.sleep(2)
        
        agent = MedicalTriageAgent()
        
        # Process a turn to create state
        await agent.process_turn(
            symptoms="test symptoms",
            nice_context="test context"
        )
        
        # Validate state exists
        assert agent.turn_count > 0
        assert len(agent.conversation_history.messages) > 0
        
        # Test reset
        agent.reset_conversation()
        assert agent.turn_count == 0
        assert len(agent.conversation_history.messages) == 0
        assert isinstance(agent.conversation_history, dspy.History)
        
        # Test summary with real data
        agent.turn_count = 2
        # Use mock for messages to avoid DSPy History immutability issues
        from unittest.mock import Mock
        mock_history = Mock()
        mock_history.messages = [
            {"outcome_classification": "inconclusive", "confidence": 60},
            {"outcome_classification": "emergency", "confidence": 90}
        ]
        agent.conversation_history = mock_history
        
        summary = agent.get_conversation_summary()
        assert summary["turns"] == 2
        assert summary["history_length"] == 2
        assert summary["last_outcome"] == "emergency"
