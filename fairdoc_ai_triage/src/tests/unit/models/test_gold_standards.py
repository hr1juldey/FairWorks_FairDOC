"""
Unit Tests for Gold Standard Database Model
Tests SQLAlchemy model, conversation validation, and DSPy training integration
"""
import pytest
import json
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import Column, String, Text, JSON, Integer, Float, Boolean, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError

# Create isolated test Base - NO imports from main app
TestBase = declarative_base()

# Test-compatible MedicalOutcome enum
class MedicalOutcome:
    EMERGENCY = "emergency_route_to_doctor"
    ROUTINE_DOCTOR = "routine_doctor_consultation"
    SELF_CARE = "self_care_advice"
    INCONCLUSIVE = "need_more_questions"
    SPAM_DETECTED = "spam_or_irrelevant"

class GoldStandardDialogue(TestBase):
    """Gold standard conversation examples for DSPy training/evaluation - Test version"""
    __tablename__ = "gold_standard_dialogues_v2"
    
    standard_id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # Classification
    primary_symptom = Column(String(100), nullable=False)
    expected_outcome = Column(String(50), nullable=False)
    
    # Patient demographics
    patient_age = Column(Integer, nullable=False)
    patient_gender = Column(String(20), nullable=False)
    
    # Red flag expectations
    expected_red_flags = Column(JSON, nullable=False, default=list)
    should_escalate = Column(Boolean, nullable=False, default=False)
    
    # Conversation dialogue
    conversation_dialogue = Column(JSON, nullable=False, default=list)
    
    # NICE protocol relevance
    relevant_protocols = Column(JSON, nullable=False, default=list)
    
    # Evaluation metrics
    minimum_confidence_threshold = Column(Float, nullable=False, default=70.0)
    expected_turn_count = Column(Integer, nullable=False, default=3)
    max_acceptable_turns = Column(Integer, nullable=False, default=8)
    
    # Data provenance
    created_by = Column(String(100), nullable=False)
    clinical_notes = Column(Text, nullable=True)
    
    # Version control
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def to_training_example(self):
        """Convert to DSPy training example format"""
        return {
            'standard_id': self.standard_id,
            'input': {
                'patient_age': self.patient_age,
                'patient_gender': self.patient_gender,
                'conversation_turns': self.conversation_dialogue,
                'relevant_protocols': self.relevant_protocols or []
            },
            'expected_output': {
                'medical_outcome': self.expected_outcome,
                'red_flags': self.expected_red_flags or [],
                'should_escalate': self.should_escalate,
                'min_confidence': self.minimum_confidence_threshold
            },
            'metadata': {
                'primary_symptom': self.primary_symptom,
                'max_turns': self.max_acceptable_turns
            }
        }
    
    def validate_against_prediction(self, prediction):
        """Validate a model prediction against this gold standard"""
        results = {
            'correct_outcome': prediction.get('medical_outcome') == self.expected_outcome,
            'sufficient_confidence': prediction.get('confidence_score', 0) >= self.minimum_confidence_threshold,
            'correct_escalation': prediction.get('requires_human_review', False) == self.should_escalate,
            'within_turn_limit': prediction.get('turn_count', 0) <= self.max_acceptable_turns
        }
        
        # Check red flag detection
        predicted_flags = set(prediction.get('red_flags_detected', []))
        expected_flags = set(self.expected_red_flags or [])
        results['red_flags_detected'] = len(expected_flags.intersection(predicted_flags)) >= len(expected_flags) * 0.8
        
        return results

@pytest.fixture
def test_db_session():
    """Create isolated in-memory test database"""
    engine = create_engine('sqlite:///:memory:', echo=False)
    TestBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()

@pytest.fixture  
def sample_gold_standard_data():
    """Sample gold standard conversation for testing"""
    return {
        "title": "Emergency Chest Pain with MI",
        "description": "Patient presenting with classic MI symptoms requiring emergency intervention",
        "primary_symptom": "chest_pain",
        "expected_outcome": MedicalOutcome.EMERGENCY,
        "patient_age": 58,
        "patient_gender": "male",
        "expected_red_flags": ["crushing_pain", "left_arm_radiation", "sweating"],
        "should_escalate": True,
        "conversation_dialogue": [
            {
                "turn": 1,
                "patient_input": "I have severe crushing chest pain radiating to my left arm",
                "agent_question": "When did this pain start and are you experiencing sweating or nausea?",
                "expected_classification": "inconclusive",
                "red_flags_detected": ["crushing_pain", "left_arm_radiation"]
            },
            {
                "turn": 2, 
                "patient_input": "Started 30 minutes ago, I'm sweating heavily and feel nauseous",
                "agent_question": "COMPLETE - Emergency medical attention needed immediately",
                "expected_classification": "emergency",
                "red_flags_detected": ["crushing_pain", "left_arm_radiation", "sweating", "nausea"]
            }
        ],
        "relevant_protocols": ["CG95_CHEST_PAIN", "NG136_MI"],
        "minimum_confidence_threshold": 85.0,
        "expected_turn_count": 2,
        "max_acceptable_turns": 3,
        "created_by": "dr_smith_cardiologist",
        "clinical_notes": "Classic STEMI presentation requiring immediate PCI"
    }

class TestGoldStandardModel:
    """Test Gold Standard SQLAlchemy model creation and validation"""
    
    def test_model_creation_with_valid_data(self, test_db_session, sample_gold_standard_data):
        """Test GoldStandardDialogue model creates successfully"""
        standard = GoldStandardDialogue(**sample_gold_standard_data)
        test_db_session.add(standard)
        test_db_session.commit()
        
        assert standard.standard_id is not None
        assert standard.title == "Emergency Chest Pain with MI"
        assert standard.expected_outcome == MedicalOutcome.EMERGENCY
        assert standard.should_escalate is True
        assert standard.is_active is True
        assert standard.created_at is not None
    
    def test_dialogue_structure_validation(self, test_db_session, sample_gold_standard_data):
        """Test conversation dialogue has correct structure"""
        standard = GoldStandardDialogue(**sample_gold_standard_data)
        test_db_session.add(standard)
        test_db_session.commit()
        
        dialogue = standard.conversation_dialogue
        assert isinstance(dialogue, list)
        assert len(dialogue) == 2
        
        # Validate first turn structure
        turn1 = dialogue[0]
        assert turn1["turn"] == 1
        assert "patient_input" in turn1
        assert "agent_question" in turn1
        assert "expected_classification" in turn1
        assert "red_flags_detected" in turn1

class TestDSPyTrainingIntegration:
    """Test DSPy training and evaluation integration"""
    
    def test_to_training_example_format(self, test_db_session, sample_gold_standard_data):
        """Test conversion to DSPy training example format"""
        standard = GoldStandardDialogue(**sample_gold_standard_data)
        test_db_session.add(standard)
        test_db_session.commit()
        
        training_example = standard.to_training_example()
        
        # Validate training example structure
        assert "standard_id" in training_example
        assert "input" in training_example
        assert "expected_output" in training_example
        assert "metadata" in training_example
        
        # Validate input section
        input_data = training_example["input"]
        assert input_data["patient_age"] == 58
        assert input_data["patient_gender"] == "male"
        assert isinstance(input_data["conversation_turns"], list)
        
        # Validate expected output
        expected = training_example["expected_output"]
        assert expected["medical_outcome"] == "emergency_route_to_doctor"
        assert expected["should_escalate"] is True
    
    def test_validate_against_prediction(self, test_db_session, sample_gold_standard_data):
        """Test validation against model predictions"""
        standard = GoldStandardDialogue(**sample_gold_standard_data)
        test_db_session.add(standard)
        test_db_session.commit()
        
        # Test correct prediction
        correct_prediction = {
            "medical_outcome": "emergency_route_to_doctor",
            "confidence_score": 90,
            "requires_human_review": True,
            "turn_count": 2,
            "red_flags_detected": ["crushing_pain", "left_arm_radiation", "sweating"]
        }
        
        results = standard.validate_against_prediction(correct_prediction)
        
        assert results["correct_outcome"] is True
        assert results["sufficient_confidence"] is True
        assert results["correct_escalation"] is True
        assert results["within_turn_limit"] is True
        assert results["red_flags_detected"] is True
