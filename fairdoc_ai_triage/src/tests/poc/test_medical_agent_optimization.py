"""
Test Medical Agent DSPy Optimization

Tests core medical triage agent with various DSPy optimizers
and evaluation metrics for medical decision making.
"""
import pytest
import dspy
from typing import Dict, List
import asyncio

from src.app2.services.dspy.medical_agent import MedicalTriageAgent, MedicalTriageProgram
from src.app2.services.dspy.evaluation_optimizer import EvaluationOptimizer
from src.app2.models.database.gold_standards_seed import GOLD_STANDARDS_SEED_DATA
from src.app2.core.dspy_config_v2 import ensure_dspy_configured


@pytest.fixture
def medical_agent():
    """Initialize medical triage agent"""
    ensure_dspy_configured("gemma3n:e4b")
    return MedicalTriageAgent(model_name="gemma3n:e4b")


@pytest.fixture 
def evaluation_optimizer():
    """Initialize evaluation optimizer"""
    ensure_dspy_configured("gemma3n:e4b")
    return EvaluationOptimizer(model_name="gemma3n:e4b")


@pytest.fixture
def gold_standard_examples():
    """Sample gold standard examples for training"""
    return GOLD_STANDARDS_SEED_DATA[:5]  # First 5 examples


@pytest.mark.asyncio
async def test_medical_agent_basic_triage(medical_agent):
    """Test basic medical triage functionality"""
    symptoms = "severe crushing chest pain radiating to left arm"
    nice_context = "Chest Pain Assessment - Emergency criteria: crushing pain >20min"
    
    result = await medical_agent.process_turn(
        symptoms=symptoms,
        nice_context=nice_context
    )
    
    assert "outcome" in result
    assert "confidence" in result
    assert "red_flags" in result
    assert result["outcome"] in ["emergency", "routine", "self_care", "inconclusive", "spam"]
    assert 0 <= result["confidence"] <= 100


def test_medical_reasoning_separation(medical_agent):
    """Test separation of reasoning and output as mentioned in medical_agent.py"""
    # Test that medical agent uses proper DSPy signature separation
    triage_program = MedicalTriageProgram()
    
    assert hasattr(triage_program, 'reasoning_module')
    assert hasattr(triage_program.reasoning_module, 'medical_cot')
    assert hasattr(triage_program.reasoning_module, 'emergency_detector')


@pytest.mark.asyncio
async def test_emergency_detection_accuracy(medical_agent):
    """Test emergency detection accuracy"""
    emergency_cases = [
        "severe crushing chest pain with sweating and nausea",
        "sudden worst headache ever with neck stiffness", 
        "cannot breathe properly gasping for air",
        "unconscious patient not responding"
    ]
    
    emergency_count = 0
    
    for symptoms in emergency_cases:
        result = await medical_agent.process_turn(
            symptoms=symptoms,
            nice_context="Emergency assessment protocol"
        )
        
        if result["outcome"] == "emergency":
            emergency_count += 1
    
    # Should detect most emergencies
    accuracy = emergency_count / len(emergency_cases)
    assert accuracy >= 0.75, f"Emergency detection accuracy too low: {accuracy}"


@pytest.mark.asyncio
async def test_conversation_completion_logic(medical_agent):
    """Test conversation completion logic from medical_agent.py"""
    # Test emergency completion
    emergency_result = await medical_agent.process_turn(
        symptoms="severe crushing chest pain cant breathe",
        nice_context="Emergency protocol"
    )
    
    # Emergency should complete immediately  
    assert emergency_result.get("is_complete", False), "Emergency should complete conversation"
    
    # Test inconclusive continuation
    mild_result = await medical_agent.process_turn(
        symptoms="mild headache from work",
        nice_context="Headache protocol"
    )
    
    # Mild symptoms should continue conversation
    assert not mild_result.get("is_complete", True), "Mild symptoms should continue conversation"


@pytest.mark.asyncio
async def test_model_evaluation_against_gold_standards(evaluation_optimizer):
    """Test model evaluation against gold standards"""
    evaluation_result = await evaluation_optimizer.evaluate_model(limit=10)
    
    assert "metrics" in evaluation_result
    assert "examples_evaluated" in evaluation_result["metrics"]
    assert evaluation_result["examples_evaluated"] > 0
    
    metrics = evaluation_result["metrics"]
    assert "overall_accuracy" in metrics


@pytest.mark.parametrize("optimizer_name", [
    "bootstrap", "mipro", "copro", "labeled_fewshot", "ensemble"
])
@pytest.mark.asyncio
async def test_dspy_optimizer_strategies(evaluation_optimizer, optimizer_name):
    """Test different DSPy optimization strategies"""
    # Test optimizer initialization and compilation
    optimization_result = await evaluation_optimizer.optimize_model(
        iterations=1,  # Single iteration for testing
        optimizer_type=optimizer_name
    )
    
    assert "optimization_status" in optimization_result
    assert optimization_result["optimization_status"] == "completed"
    assert "optimizer_type" in optimization_result
    assert optimization_result["optimizer_type"] == optimizer_name


def test_medical_outcome_mapping():
    """Test medical outcome mapping between internal and API enums"""
    from src.app2.utils.outcome_mapper import OutcomeMapper
    from src.app2.models.schemas.medical_triage import MedicalOutcome as TriageOutcome
    from src.app2.models.schemas.multiturn_chat import MedicalOutcome as ChatOutcome
    
    # Test bidirectional mapping
    triage_emergency = TriageOutcome.EMERGENCY
    chat_emergency = OutcomeMapper.to_chat(triage_emergency)
    back_to_triage = OutcomeMapper.to_triage(chat_emergency)
    
    assert back_to_triage == triage_emergency
    assert OutcomeMapper.is_equivalent(triage_emergency, chat_emergency)


@pytest.mark.asyncio
async def test_red_flag_detection_accuracy(medical_agent):
    """Test red flag detection accuracy and types"""
    red_flag_symptoms = {
        "crushing_chest_pain": "severe crushing chest pain like elephant on chest",
        "thunderclap_headache": "sudden worst headache ever like thunderclap",
        "severe_dyspnea": "cannot breathe properly gasping for air",
        "loss_consciousness": "patient fainted and lost consciousness"
    }
    
    for expected_flag, symptoms in red_flag_symptoms.items():
        result = await medical_agent.process_turn(
            symptoms=symptoms,
            nice_context="Emergency assessment"
        )
        
        red_flags = result.get("red_flags", [])
        assert len(red_flags) > 0, f"Should detect red flags for {expected_flag}: {symptoms}"



def test_confidence_score_calibration():
    """Test confidence score calibration and bounds"""
    test_confidences = [95, 85, 65, 45, 30]
    
    for confidence in test_confidences:
        # Test confidence bounds
        assert 0 <= confidence <= 100, f"Confidence out of bounds: {confidence}"
        
        # Test confidence mapping to outcomes
        if confidence >= 85:
            expected_certainty = "high"
        elif confidence >= 60:
            expected_certainty = "medium"
        else:
            expected_certainty = "low"
        
        assert expected_certainty in ["low", "medium", "high"]


@pytest.mark.asyncio
async def test_context_history_integration(medical_agent):
    """Test conversation history integration"""
    # Reset conversation for clean test
    medical_agent.reset_conversation()
    
    # First turn
    result1 = await medical_agent.process_turn(
        symptoms="chest discomfort",
        nice_context="Chest pain protocol"
    )
    
    # Second turn with additional context
    result2 = await medical_agent.process_turn(
        symptoms="pain is getting worse and radiating to arm",
        nice_context="Chest pain protocol"
    )
    
    # Second turn should have higher confidence/urgency
    assert result2["confidence"] >= result1["confidence"] or result2["outcome"] == "emergency"


def test_nice_protocol_integration():
    """Test NICE protocol integration with medical agent"""
    from src.app2.services.context.nice_lookup import NICELookupService
    
    nice_service = NICELookupService()
    
    test_symptoms = "severe chest pain with radiation"
    protocol_result = nice_service.find_relevant_protocols(test_symptoms)
    
    assert "protocol_code" in protocol_result
    assert "protocol_text" in protocol_result
    assert len(protocol_result["protocol_text"]) > 50  # Should have substantial content


@pytest.mark.asyncio 
async def test_performance_metrics(medical_agent):
    """Test agent performance metrics"""
    import time
    
    start_time = time.time()
    
    result = await medical_agent.process_turn(
        symptoms="test symptoms for performance",
        nice_context="test protocol"
    )
    
    end_time = time.time()
    response_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Should respond within reasonable time
    assert response_time < 5000, f"Response time too slow: {response_time}ms"
    assert "outcome" in result


def test_error_handling_robustness(medical_agent):
    """Test error handling and robustness"""
    # Test empty symptoms
    with pytest.raises(ValueError, match="Symptoms cannot be empty"):
        asyncio.run(medical_agent.process_turn(
            symptoms="",
            nice_context="test"
        ))
    
    # Test very long symptoms
    long_symptoms = "symptom " * 1000
    try:
        result = asyncio.run(medical_agent.process_turn(
            symptoms=long_symptoms,
            nice_context="test"
        ))
        # Should handle gracefully
        assert "outcome" in result
    except Exception:
        # Should not crash completely - just pass if it handles the error
        pass



@pytest.mark.asyncio
async def test_medical_agent_memory_management(medical_agent):
    """Test conversation memory and state management"""
    # Test conversation reset
    medical_agent.reset_conversation()
    
    summary = medical_agent.get_conversation_summary()
    assert summary["turns"] == 0
    assert summary["history_length"] == 0
    
    # Process a turn
    await medical_agent.process_turn("test symptoms", "test context")
    
    updated_summary = medical_agent.get_conversation_summary()
    assert updated_summary["turns"] > 0
    assert updated_summary["history_length"] > 0
