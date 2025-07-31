"""
Unit Tests for Gold Standard Database Model
Tests SQLAlchemy model, conversation validation, and DSPy training integration
"""
import pytest
import json
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from unittest.mock import patch

# Test environment setup - isolated database  
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'DATABASE_URL': 'sqlite:///:memory:',
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
    from src.app2.models.database.gold_standards import GoldStandardDialogue, Base
    from src.app2.models.schemas.medical_triage import MedicalOutcome


@pytest.fixture
def test_db_session():
    """Create isolated in-memory test database"""
    engine = create_engine('sqlite:///:memory:', echo=False)
    Base.metadata.create_all(engine)
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
    
    def test_constraint_validations(self, test_db_session, sample_gold_standard_data):
        """Test database constraints work correctly"""
        # Test invalid confidence threshold
        invalid_data = sample_gold_standard_data.copy()
        invalid_data["minimum_confidence_threshold"] = 150.0  # > 100
        
        standard = GoldStandardDialogue(**invalid_data)
        test_db_session.add(standard)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
    
    def test_turn_count_constraints(self, test_db_session, sample_gold_standard_data):
        """Test turn count constraints"""
        invalid_data = sample_gold_standard_data.copy()
        invalid_data["expected_turn_count"] = 10
        invalid_data["max_acceptable_turns"] = 5  # Less than expected
        
        standard = GoldStandardDialogue(**invalid_data)
        test_db_session.add(standard)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()


class TestConversationDialogueValidation:
    """Test conversation dialogue JSON structure and validation"""
    
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
    
    def test_medical_outcome_progression(self, test_db_session, sample_gold_standard_data):
        """Test conversation shows proper medical outcome progression"""
        standard = GoldStandardDialogue(**sample_gold_standard_data)
        test_db_session.add(standard)
        test_db_session.commit()
        
        dialogue = standard.conversation_dialogue
        
        # Should progress from inconclusive to emergency
        assert dialogue[0]["expected_classification"] == "inconclusive"
        assert dialogue[1]["expected_classification"] == "emergency"
        
        # Red flags should accumulate
        turn1_flags = dialogue[0]["red_flags_detected"]
        turn2_flags = dialogue[1]["red_flags_detected"]
        assert len(turn2_flags) > len(turn1_flags)


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
        assert expected["medical_outcome"] == "emergency"
        assert expected["should_escalate"] is True
    
    def test_validate_against_prediction(self, test_db_session, sample_gold_standard_data):
        """Test validation against model predictions"""
        standard = GoldStandardDialogue(**sample_gold_standard_data)
        test_db_session.add(standard)
        test_db_session.commit()
        
        # Test correct prediction
        correct_prediction = {
            "medical_outcome": "emergency",
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
    
    def test_validation_failure_cases(self, test_db_session, sample_gold_standard_data):
        """Test validation with incorrect predictions"""
        standard = GoldStandardDialogue(**sample_gold_standard_data)
        test_db_session.add(standard)
        test_db_session.commit()
        
        # Test incorrect prediction
        wrong_prediction = {
            "medical_outcome": "self_care",  # Wrong outcome
            "confidence_score": 60,  # Too low confidence
            "requires_human_review": False,  # Wrong escalation
            "turn_count": 5,  # Too many turns
            "red_flags_detected": []  # Missed red flags
        }
        
        results = standard.validate_against_prediction(wrong_prediction)
        
        assert results["correct_outcome"] is False
        assert results["sufficient_confidence"] is False
        assert results["correct_escalation"] is False
        assert results["within_turn_limit"] is False
        assert results["red_flags_detected"] is False


class TestGoldStandardQueries:
    """Test gold standard query methods for training/evaluation"""
    
    def test_get_training_set(self, test_db_session):
        """Test retrieving active gold standards for training"""
        # Create multiple standards
        for i in range(5):
            standard = GoldStandardDialogue(
                title=f"Test Case {i}",
                description=f"Test case number {i}",
                primary_symptom="headache",
                expected_outcome=MedicalOutcome.ROUTINE_DOCTOR,
                patient_age=30 + i,
                patient_gender="female",
                conversation_dialogue=[],
                minimum_confidence_threshold=70.0,
                expected_turn_count=3,
                created_by="test_creator"
            )
            test_db_session.add(standard)
        
        test_db_session.commit()
        
        # Test training set retrieval
        training_set = GoldStandardDialogue.get_training_set(
            test_db_session, 
            symptom_filter="headache", 
            limit=3
        )
        
        assert len(training_set) == 3
        assert all(std.primary_symptom == "headache" for std in training_set)
        assert all(std.is_active is True for std in training_set)
    
    def test_get_emergency_examples(self, test_db_session, sample_gold_standard_data):
        """Test retrieving emergency-specific examples"""
        # Create emergency standard
        emergency_standard = GoldStandardDialogue(**sample_gold_standard_data)
        test_db_session.add(emergency_standard)
        
        # Create non-emergency standard
        routine_data = sample_gold_standard_data.copy()
        routine_data["title"] = "Routine Headache"
        routine_data["expected_outcome"] = MedicalOutcome.ROUTINE_DOCTOR
        routine_standard = GoldStandardDialogue(**routine_data)
        test_db_session.add(routine_standard)
        
        test_db_session.commit()
        
        emergency_examples = GoldStandardDialogue.get_emergency_examples(test_db_session)
        
        assert len(emergency_examples) == 1
        assert emergency_examples[0].expected_outcome == MedicalOutcome.EMERGENCY
        assert emergency_examples[0].should_escalate is True


class TestGoldStandardPerformance:
    """Test performance aspects for training/evaluation workflows"""
    
    def test_bulk_training_example_generation(self, test_db_session):
        """Test bulk generation of training examples"""
        # Create multiple standards
        standards = []
        for i in range(10):
            standard = GoldStandardDialogue(
                title=f"Bulk Test {i}",
                description=f"Bulk test case {i}",
                primary_symptom="chest_pain",
                expected_outcome=MedicalOutcome.EMERGENCY,
                patient_age=40 + i,
                patient_gender="male" if i % 2 == 0 else "female",
                conversation_dialogue=[{"turn": 1, "test": "data"}],
                minimum_confidence_threshold=80.0,
                expected_turn_count=2,
                created_by="bulk_creator"
            )
            standards.append(standard)
            test_db_session.add(standard)
        
        test_db_session.commit()
        
        # Test bulk conversion to training examples
        import time
        start_time = time.time()
        
        training_examples = [std.to_training_example() for std in standards]
        
        conversion_time = time.time() - start_time
        
        # Should be fast bulk conversion
        assert conversion_time < 0.1
        assert len(training_examples) == 10
        assert all("standard_id" in ex for ex in training_examples)
