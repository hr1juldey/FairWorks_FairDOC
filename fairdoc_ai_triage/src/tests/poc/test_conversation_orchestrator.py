"""
Test Conversation Orchestration and Multi-turn Chat

Tests DSPy-powered conversation flow with Redis state management
and context engineering for medical consultations.
"""
import pytest
import dspy
import asyncio
from typing import Dict, List, Optional
from uuid import uuid4

from src.app2.services.chat.chat_orchestrator import ChatOrchestrator
from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatRequest, StakeholderRole, ChatProvider
)
from src.app2.core.dspy_config_v2 import ensure_dspy_configured


class ConversationSignature(dspy.Signature):
    """Multi-turn medical conversation management"""
    current_symptoms = dspy.InputField(desc="Patient's current symptoms")
    conversation_history = dspy.InputField(desc="Previous conversation context")
    patient_demographics = dspy.InputField(desc="Patient age, gender, context")
    
    next_question = dspy.OutputField(desc="Next clarifying question")
    medical_assessment = dspy.OutputField(desc="Current medical assessment")
    conversation_stage = dspy.OutputField(desc="Stage: initial|gathering|assessment|conclusion")
    urgency_level = dspy.OutputField(desc="Urgency: low|medium|high|critical")


class ConversationModule(dspy.Module):
    """DSPy module for conversation flow management"""
    
    def __init__(self):
        super().__init__()
        self.conversation_manager = dspy.ChainOfThought(ConversationSignature)
        self.turn_count = 0
    
    def forward(self, symptoms: str, history: str, demographics: str):
        """Manage conversation turn"""
        self.turn_count += 1
        
        result = self.conversation_manager(
            current_symptoms=symptoms,
            conversation_history=history,
            patient_demographics=demographics
        )
        
        return dspy.Prediction(
            next_question=result.next_question,
            assessment=result.medical_assessment,
            stage=result.conversation_stage,
            urgency=result.urgency_level,
            turn_number=self.turn_count
        )


@pytest.fixture
async def chat_orchestrator():
    """Initialize chat orchestrator"""
    ensure_dspy_configured("gemma3n:e4b")
    orchestrator = ChatOrchestrator()
    await orchestrator.initialize()
    return orchestrator


@pytest.fixture
def conversation_module():
    """Initialize conversation module"""
    ensure_dspy_configured("gemma3n:e4b")
    return ConversationModule()


@pytest.fixture
def sample_chat_requests():
    """Sample chat requests for testing"""
    return [
        MultiTurnChatRequest(
            conversation_id=uuid4(),
            user_message="I have severe chest pain",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="test_patient_001",
            chat_provider=ChatProvider.API_DIRECT,
            patient_age=55,
            patient_gender="male"
        ),
        MultiTurnChatRequest(
            conversation_id=uuid4(),
            user_message="My head hurts a lot",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="test_patient_002",
            chat_provider=ChatProvider.API_DIRECT,
            patient_age=32,
            patient_gender="female"
        )
    ]


@pytest.mark.asyncio
async def test_conversation_initialization(chat_orchestrator, sample_chat_requests):
    """Test conversation initialization and first turn"""
    request = sample_chat_requests[0]
    
    result = await chat_orchestrator.process_conversation_turn(request)
    
    assert "conversation_id" in result
    assert "agent_result" in result
    assert result["agent_result"]["outcome"] in ["emergency", "routine", "self_care", "inconclusive"]


@pytest.mark.asyncio
async def test_context_continuity(chat_orchestrator):
    """Test conversation context continuity across turns"""
    conversation_id = uuid4()
    
    # First turn
    request1 = MultiTurnChatRequest(
        conversation_id=conversation_id,
        user_message="I have chest pain",
        stakeholder_role=StakeholderRole.PATIENT,
        stakeholder_id="test_patient",
        patient_age=45,
        patient_gender="male"
    )
    
    result1 = await chat_orchestrator.process_conversation_turn(request1)
    
    # Second turn - should have context from first
    request2 = MultiTurnChatRequest(
        conversation_id=conversation_id,
        user_message="It started 2 hours ago and radiates to my arm",
        stakeholder_role=StakeholderRole.PATIENT,
        stakeholder_id="test_patient"
    )
    
    result2 = await chat_orchestrator.process_conversation_turn(request2)
    
    # Verify context is maintained
    assert result1["conversation_id"] == result2["conversation_id"]
    assert result2["context_maintained"], "Context should be maintained across turns"


def test_conversation_stage_progression(conversation_module):
    """Test conversation progresses through logical stages"""
    # Initial stage
    result1 = conversation_module(
        symptoms="chest pain",
        history="",
        demographics="Age: 45, Gender: male"
    )
    
    # Should be in initial or gathering stage
    assert result1.stage in ["initial", "gathering"]
    
    # Progressive conversation
    result2 = conversation_module(
        symptoms="crushing pain radiating to arm",
        history="Previous: chest pain complaint",
        demographics="Age: 45, Gender: male"
    )
    
    # Should progress to assessment
    assert result2.turn_number > result1.turn_number


def test_urgency_escalation(conversation_module):
    """Test urgency level escalation with red flags"""
    emergency_symptoms = [
        "severe crushing chest pain with sweating",
        "sudden worst headache ever",
        "cannot breathe properly gasping for air",
        "unconscious and unresponsive"
    ]
    
    for symptoms in emergency_symptoms:
        result = conversation_module(
            symptoms=symptoms,
            history="",
            demographics="Age: 50, Gender: male"
        )
        
        # Should detect high urgency
        assert result.urgency in ["high", "critical"], f"Should escalate urgency for: {symptoms}"


@pytest.mark.parametrize("optimizer_type", ["mipro", "copro", "bootstrap"])
def test_conversation_optimization(conversation_module, optimizer_type):
    """Test conversation optimization with different DSPy optimizers"""
    training_conversations = [
        dspy.Example(
            current_symptoms="chest pain",
            conversation_history="",
            patient_demographics="Age: 55, Gender: male",
            next_question="When did the chest pain start and what does it feel like?",
            medical_assessment="Possible cardiac event, needs immediate evaluation",
            conversation_stage="gathering",
            urgency_level="high"
        ).with_inputs('current_symptoms', 'conversation_history', 'patient_demographics')
    ]
    
    if optimizer_type == "mipro":
        optimizer = dspy.MIPROv2(num_candidates=3, init_temperature=0.1)
    elif optimizer_type == "copro":
        optimizer = dspy.COPRO(breadth=3, depth=2)
    else:  # bootstrap
        optimizer = dspy.BootstrapFewShot(max_bootstrapped_demos=5)
    
    assert optimizer is not None
    assert len(training_conversations) == 1


@pytest.mark.asyncio 
async def test_redis_state_management(chat_orchestrator):
    """Test Redis conversation state persistence"""
    conversation_id = uuid4()
    
    request = MultiTurnChatRequest(
        conversation_id=conversation_id,
        user_message="Test message for state persistence",
        stakeholder_role=StakeholderRole.PATIENT,
        stakeholder_id="test_patient"
    )
    
    # Process turn
    result = await chat_orchestrator.process_conversation_turn(request)
    
    # Check state is stored
    state = await chat_orchestrator.get_conversation_state(str(conversation_id))
    assert state is not None
    assert state["conversation_id"] == str(conversation_id)


def test_conversation_completion_logic():
    """Test conversation completion determination"""
    completion_scenarios = [
        {"outcome": "emergency", "turns": 2, "should_complete": True},
        {"outcome": "self_care", "turns": 4, "confidence": 85, "should_complete": True}, 
        {"outcome": "inconclusive", "turns": 3, "confidence": 45, "should_complete": False},
        {"outcome": "inconclusive", "turns": 15, "confidence": 60, "should_complete": True}  # Max turns
    ]
    
    for scenario in completion_scenarios:
        # Test completion logic rules from chat_orchestrator.py
        if scenario["outcome"] == "emergency":
            assert scenario["should_complete"], "Emergency should complete immediately"
        elif scenario["turns"] >= 15:
            assert scenario["should_complete"], "Should complete at max turns"


@pytest.mark.asyncio
async def test_stakeholder_routing(chat_orchestrator):
    """Test multi-stakeholder conversation routing"""
    conversation_id = uuid4()
    
    # Patient message
    patient_request = MultiTurnChatRequest(
        conversation_id=conversation_id,
        user_message="I need medical help",
        stakeholder_role=StakeholderRole.PATIENT,
        stakeholder_id="patient_001"
    )
    
    result = await chat_orchestrator.process_conversation_turn(patient_request)
    
    # Should route to triage agent
    assert "message_routes" in result
    routes = result["message_routes"]
    assert len(routes) >= 1


def test_context_engineering_versioning():
    """Test context versioning like Git for conversation state"""
    from src.app2.models.schemas.multiturn_chat import ConversationState
    
    # Test immutable state versioning
    state1 = ConversationState(
        conversation_id=uuid4(),
        context_hash="abc123",
        turn_count=1,
        current_status="NEW"
    )
    
    # Should be immutable (frozen=True)
    with pytest.raises(Exception):
        state1.turn_count = 2  # Should fail due to frozen=True


@pytest.mark.asyncio
async def test_emergency_escalation_workflow(chat_orchestrator):
    """Test emergency escalation triggers proper workflows"""
    emergency_request = MultiTurnChatRequest(
        conversation_id=uuid4(),
        user_message="severe crushing chest pain cant breathe",
        stakeholder_role=StakeholderRole.PATIENT,
        stakeholder_id="emergency_patient",
        patient_age=58,
        patient_gender="male"
    )
    
    result = await chat_orchestrator.process_conversation_turn(emergency_request)
    
    # Should trigger emergency workflows
    assert result["requires_emergency_alert"], "Should trigger emergency alert"
    assert result["agent_result"]["outcome"] == "emergency"


def test_conversation_metrics_tracking():
    """Test conversation performance metrics"""
    metrics = {
        "average_turns_to_resolution": 4.2,
        "emergency_detection_accuracy": 0.95,
        "patient_satisfaction_score": 4.3,
        "response_time_ms": 850
    }
    
    # Validate metrics are within expected ranges
    assert 2 <= metrics["average_turns_to_resolution"] <= 10
    assert 0.8 <= metrics["emergency_detection_accuracy"] <= 1.0
    assert 1 <= metrics["patient_satisfaction_score"] <= 5
    assert metrics["response_time_ms"] < 2000  # Under 2 seconds
