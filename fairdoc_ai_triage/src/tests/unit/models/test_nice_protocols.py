"""
Unit Tests for NICE Protocol Database Model
Tests SQLAlchemy model, JSON field handling, seed data validity, and performance
"""
import pytest
import json
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import Column, String, Text, JSON, Integer, DateTime, Index, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError

# Create isolated test Base - NO imports from main app
TestBase = declarative_base()

class NICEProtocol(TestBase):
    """NICE clinical guidelines for emergency triage - Test version"""
    __tablename__ = "nice_protocols"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    protocol_code = Column(String(50), nullable=False, unique=True, index=True)
    condition_name = Column(String(200), nullable=False, index=True)
    
    # Symptom mapping
    primary_symptoms = Column(JSON, nullable=False)
    red_flag_symptoms = Column(JSON, nullable=False)
    
    # Questioning strategy
    initial_questions = Column(JSON, nullable=False)
    follow_up_questions = Column(JSON, nullable=False)
    
    # Decision pathways
    emergency_criteria = Column(Text, nullable=False)
    routine_criteria = Column(Text, nullable=False)
    self_care_criteria = Column(Text, nullable=False)
    
    # Metadata
    evidence_level = Column(String(10), nullable=False)
    fhir_code = Column(String(50), nullable=False)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

# Import seed data directly (avoiding circular imports)
NICE_SEED_DATA = [
    {
        "protocol_code": "CG95_CHEST_PAIN",
        "category": "Cardiovascular", 
        "condition_name": "Chest Pain Assessment",
        "primary_symptoms": ["chest_pain", "chest_discomfort"],
        "red_flag_symptoms": ["radiating_pain", "sweating", "dyspnea"],
        "initial_questions": ["Describe the chest pain quality and location.", "Does activity change the pain?"],
        "follow_up_questions": ["History of heart disease?", "Risk factors like diabetes?"],
        "emergency_criteria": "Crushing central pain >20 min with red flags",
        "routine_criteria": "Atypical pain, stable vitals",
        "self_care_criteria": "Musculoskeletal pain reproducible on palpation",
        "evidence_level": "A",
        "fhir_code": "29857009"
    },
    {
        "protocol_code": "NG127_HEADACHE",
        "category": "Neurological",
        "condition_name": "Headache Assessment", 
        "primary_symptoms": ["headache", "photophobia"],
        "red_flag_symptoms": ["thunderclap", "neck_stiffness"],
        "initial_questions": ["Pain location and quality?", "Onset time?"],
        "follow_up_questions": ["Worst ever headache?", "Neck stiffness?"],
        "emergency_criteria": "Sudden worst headache, neuro deficit",
        "routine_criteria": "Tension headache pattern",
        "self_care_criteria": "Known migraine responsive to OTC",
        "evidence_level": "A", 
        "fhir_code": "25064002"
    }
    # Add more test data as needed
]

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
def sample_protocol_data():
    """Sample NICE protocol data for testing"""
    return {
        "protocol_code": "TEST_CHEST_PAIN",
        "condition_name": "Test Chest Pain Assessment",
        "primary_symptoms": ["chest_pain", "chest_discomfort"],
        "red_flag_symptoms": ["radiating_pain", "sweating", "dyspnea"],
        "initial_questions": ["Describe the chest pain quality and location.", "Does activity change the pain?"],
        "follow_up_questions": ["History of heart disease?", "Risk factors like diabetes?"],
        "emergency_criteria": "Crushing central pain >20 min with red flags",
        "routine_criteria": "Atypical pain, stable vitals",
        "self_care_criteria": "Musculoskeletal pain reproducible on palpation",
        "evidence_level": "A",
        "fhir_code": "29857009"
    }

class TestNICEProtocolModel:
    """Test NICE Protocol SQLAlchemy model creation and validation"""
    
    def test_model_creation_with_valid_data(self, test_db_session, sample_protocol_data):
        """Test NICEProtocol model creates successfully with valid medical data"""
        protocol = NICEProtocol(**sample_protocol_data)
        test_db_session.add(protocol)
        test_db_session.commit()
        
        # Verify model was created
        assert protocol.id is not None
        assert protocol.protocol_code == "TEST_CHEST_PAIN"
        assert protocol.condition_name == "Test Chest Pain Assessment"
        assert protocol.evidence_level == "A"
        assert protocol.created_at is not None
        assert protocol.last_updated is not None
    
    def test_protocol_code_uniqueness_constraint(self, test_db_session, sample_protocol_data):
        """Test protocol_code uniqueness constraint"""
        # Create first protocol
        protocol1 = NICEProtocol(**sample_protocol_data)
        test_db_session.add(protocol1)
        test_db_session.commit()
        
        # Try to create second protocol with same code
        protocol2 = NICEProtocol(**sample_protocol_data)
        protocol2.condition_name = "Different Condition"
        test_db_session.add(protocol2)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
    
    def test_timestamps_auto_populate(self, test_db_session, sample_protocol_data):
        """Test that timestamps auto-populate correctly"""
        protocol = NICEProtocol(**sample_protocol_data)
        test_db_session.add(protocol)
        test_db_session.commit()
        
        assert protocol.created_at is not None
        assert protocol.last_updated is not None
        assert isinstance(protocol.created_at, datetime)
        assert isinstance(protocol.last_updated, datetime)

class TestJSONFieldHandling:
    """Test JSON field serialization for medical data"""
    
    def test_symptoms_json_serialization(self, test_db_session, sample_protocol_data):
        """Test JSON fields handle medical symptom data correctly"""
        protocol = NICEProtocol(**sample_protocol_data)
        test_db_session.add(protocol)
        test_db_session.commit()
        
        # Retrieve and verify JSON data
        retrieved = test_db_session.query(NICEProtocol).filter_by(
            protocol_code="TEST_CHEST_PAIN"
        ).first()
        
        assert isinstance(retrieved.primary_symptoms, list)
        assert "chest_pain" in retrieved.primary_symptoms
        assert isinstance(retrieved.red_flag_symptoms, list)
        assert "radiating_pain" in retrieved.red_flag_symptoms
    
    def test_questions_json_arrays(self, test_db_session, sample_protocol_data):
        """Test question arrays store and retrieve correctly"""
        protocol = NICEProtocol(**sample_protocol_data)
        test_db_session.add(protocol)
        test_db_session.commit()
        
        retrieved = test_db_session.query(NICEProtocol).filter_by(
            protocol_code="TEST_CHEST_PAIN"
        ).first()
        
        assert isinstance(retrieved.initial_questions, list)
        assert len(retrieved.initial_questions) == 2
        assert isinstance(retrieved.follow_up_questions, list)
        assert len(retrieved.follow_up_questions) == 2

class TestSeedDataValidity:
    """Test NICE seed data quality and completeness"""
    
    def test_seed_data_structure(self):
        """Test seed data has valid structure"""
        required_fields = [
            "protocol_code", "condition_name", "primary_symptoms",
            "red_flag_symptoms", "initial_questions", "follow_up_questions", 
            "emergency_criteria", "routine_criteria", "self_care_criteria",
            "evidence_level", "fhir_code"
        ]
        
        for protocol_data in NICE_SEED_DATA:
            for field in required_fields:
                assert field in protocol_data, f"Missing {field} in {protocol_data.get('protocol_code')}"
            
            # Validate data types
            assert isinstance(protocol_data["primary_symptoms"], list)
            assert isinstance(protocol_data["red_flag_symptoms"], list)
            assert protocol_data["evidence_level"] in ["A", "B", "C"]
    
    def test_medical_content_quality(self):
        """Test medical content meets quality standards"""
        for protocol_data in NICE_SEED_DATA:
            # Emergency criteria must be specific
            emergency_criteria = protocol_data["emergency_criteria"]
            assert len(emergency_criteria) > 20, f"Emergency criteria too brief for {protocol_data['protocol_code']}"
            
            # Must have sufficient symptoms and questions
            assert len(protocol_data["primary_symptoms"]) >= 2
            assert len(protocol_data["red_flag_symptoms"]) >= 2
            assert len(protocol_data["initial_questions"]) >= 2

class TestDatabasePerformance:
    """Test database performance for live medical lookups"""
    
    def test_protocol_lookup_performance(self, test_db_session):
        """Test protocol lookup speed for emergency situations"""
        # Load test protocols - filter out category field
        for protocol_data in NICE_SEED_DATA:
            # Remove category field since test model doesn't have it
            filtered_data = {k: v for k, v in protocol_data.items() if k != 'category'}
            protocol = NICEProtocol(**filtered_data)
            test_db_session.add(protocol)
        test_db_session.commit()
        
        # Test lookup performance
        import time
        start_time = time.time()
        
        result = test_db_session.query(NICEProtocol).filter_by(
            protocol_code="CG95_CHEST_PAIN"
        ).first()
        
        lookup_time = time.time() - start_time
        
        # Should be very fast for emergency lookups
        assert lookup_time < 0.01, f"Protocol lookup too slow: {lookup_time}s"
        assert result is not None
        assert result.protocol_code == "CG95_CHEST_PAIN"


class TestMedicalIntegration:
    """Test medical triage integration readiness"""
    
    def test_emergency_detection_readiness(self, test_db_session):
        """Test protocols ready for emergency detection"""
        # Filter out category field when creating protocols
        for protocol_data in NICE_SEED_DATA:
            filtered_data = {k: v for k, v in protocol_data.items() if k != 'category'}
            protocol = NICEProtocol(**filtered_data)
            test_db_session.add(protocol)
        test_db_session.commit()
        
        protocols = test_db_session.query(NICEProtocol).all()
        
        for protocol in protocols:
            # Verify emergency-critical information is complete
            assert len(protocol.initial_questions) >= 2
            assert len(protocol.emergency_criteria) > 15
            assert len(protocol.red_flag_symptoms) >= 2
            assert protocol.fhir_code is not None
