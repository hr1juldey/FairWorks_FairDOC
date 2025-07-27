"""
NICE Protocol Database Models for Medical Emergency Guidelines
"""
from sqlalchemy import Column, String, Text, JSON, Integer, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
from uuid import uuid4
from datetime import datetime, timezone
from src.app.core.database import Base

class NICEProtocol(Base):
    """NICE clinical guidelines for emergency triage"""
    __tablename__ = "nice_protocols"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    protocol_code = Column(String(50), nullable=False, unique=True, index=True)
    condition_name = Column(String(200), nullable=False, index=True)
    
    # Symptom mapping
    primary_symptoms = Column(JSON, nullable=False)  # ["headache", "chest_pain"]
    red_flag_symptoms = Column(JSON, nullable=False)  # Emergency indicators
    
    # Questioning strategy
    initial_questions = Column(JSON, nullable=False)  # First questions to ask
    follow_up_questions = Column(JSON, nullable=False)  # Progressive questions
    
    # Decision pathways
    emergency_criteria = Column(Text, nullable=False)  # When to route to emergency
    routine_criteria = Column(Text, nullable=False)    # When routine doctor visit
    self_care_criteria = Column(Text, nullable=False)  # When self-care appropriate
    
    # Metadata
    evidence_level = Column(String(10), nullable=False)  # A, B, C evidence quality
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Database indexes for fast lookup
    __table_args__ = (
        Index('idx_nice_symptoms', 'primary_symptoms'),
        Index('idx_nice_condition', 'condition_name'),
    )

# Pre-populate with essential protocols
NICE_SEED_DATA = [
    {
        "protocol_code": "NG127_HEADACHE",
        "condition_name": "Headache Assessment",
        "primary_symptoms": ["headache", "head_pain", "migraine"],
        "red_flag_symptoms": ["sudden_onset", "worst_headache_ever", "neck_stiffness", "fever", "confusion"],
        "initial_questions": [
            "Can you describe where exactly the pain is and whether it feels like a tight band or pressure?",
            "When did this headache first start, and how long does it usually last?",
            "Have you experienced any nausea, vomiting, or sensitivity to light?"
        ],
        "follow_up_questions": [
            "Is this the worst headache you've ever experienced?",
            "Do you have any neck stiffness or fever?",
            "Have you had any recent head injuries or vision changes?"
        ],
        "emergency_criteria": "Sudden onset worst headache ever, neck stiffness, fever >38.5C, confusion, vision changes",
        "routine_criteria": "Chronic headache pattern, mild-moderate intensity, no red flags",
        "self_care_criteria": "Mild tension headache, known triggers, responds to over-counter medication",
        "evidence_level": "A"
    },
    {
        "protocol_code": "CG95_CHEST_PAIN", 
        "condition_name": "Chest Pain Assessment",
        "primary_symptoms": ["chest_pain", "chest_discomfort", "heart_pain"],
        "red_flag_symptoms": ["crushing_pain", "left_arm_radiation", "sweating", "shortness_breath", "nausea"],
        "initial_questions": [
            "Is the pain crushing or heavy, and does it radiate to your left arm, neck, or jaw?",
            "Are you experiencing sweating, nausea, or shortness of breath?",
            "Does the pain worsen with exertion or improve with rest?"
        ],
        "follow_up_questions": [
            "How long has this pain been present?",
            "Do you have a history of heart problems or high blood pressure?",
            "Can you reproduce the pain by pressing on your chest wall?"
        ],
        "emergency_criteria": "Crushing chest pain with radiation, sweating, SOB, suspected MI",
        "routine_criteria": "Atypical chest pain, no cardiac risk factors, stable symptoms",
        "self_care_criteria": "Musculoskeletal pain, reproducible by movement, no cardiac symptoms",
        "evidence_level": "A"
    }
]
