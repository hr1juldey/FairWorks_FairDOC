"""
V2 Gold Standards Database Model  
SQLAlchemy model for storing evaluation/training data
Seed data for DSPy optimization and model evaluation
File: src/app2/models/database/gold_standards.py
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, Integer, Float, Boolean,
    Text, Index, CheckConstraint, Enum as SQLEnum
)

from sqlalchemy.orm import validates  # ✅ Correct import

from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB

from sqlalchemy.sql import func
from src.app2.core.database_v2 import BaseV2 as Base
from src.app2.models.schemas.medical_triage import MedicalOutcome, RedFlagIndicator
from src.app2.models.schemas.multiturn_chat import StakeholderRole




class GoldStandardDialogue(Base):
    """
    Gold standard conversation examples for DSPy training/evaluation
    Contains expert-labeled triage conversations for model optimization
    """
    __tablename__ = "gold_standard_dialogues_v2"
    
    # Primary identifiers
    standard_id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        doc="Unique identifier for this gold standard example"
    )
    
    # Metadata
    title = Column(
        String(200),
        nullable=False,
        doc="Human-readable title (e.g., 'Chest Pain Emergency Case')"
    )
    description = Column(
        Text,
        nullable=False,
        doc="Detailed description of the clinical scenario"
    )
    
    # Classification
    primary_symptom = Column(
        String(100),
        nullable=False,
        index=True,
        doc="Main presenting symptom category"
    )
    expected_outcome = Column(
        String(50),  # Store as string instead of enum
        nullable=False,
        index=True,
        doc="Expert-labeled correct triage outcome"
    )

    @validates('expected_outcome')
    def validate_outcome(self, key, outcome):
        """Validate and convert outcome to database string value"""
        valid_values = [
            'emergency_route_to_doctor',
            'routine_doctor_consultation', 
            'self_care_advice',
            'need_more_questions',
            'spam_or_irrelevant'
        ]
        
        if isinstance(outcome, MedicalOutcome):
            value = outcome.value
        else:
            value = str(outcome)
            
        if value not in valid_values:
            raise ValueError(f"Invalid medical outcome: {value}")
            
        return value

    
    # Patient demographics for this scenario
    patient_age = Column(
        Integer,
        nullable=False,
        doc="Patient age in this scenario"
    )
    patient_gender = Column(
        String(20),
        nullable=False,
        doc="Patient gender in this scenario"
    )
    
    # Red flag expectations
    expected_red_flags = Column(
        JSONB,
        nullable=False,
        default=[],
        doc="JSON array of RedFlagIndicator values that should be detected"
    )
    should_escalate = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="Whether this case should trigger human review"
    )
    
    # The actual conversation dialogue
    conversation_dialogue = Column(
        JSONB,
        nullable=False,
        doc="Complete conversation as JSON array of turns with expected agent responses"
    )
    
    # NICE protocol relevance
    relevant_protocols = Column(
        JSONB,
        nullable=False,
        default=[],
        doc="JSON array of NICE protocol IDs relevant to this case"
    )
    
    # Evaluation metrics
    minimum_confidence_threshold = Column(
        Float,
        nullable=False,
        default=70.0,
        doc="Minimum confidence score expected for this case"
    )
    expected_turn_count = Column(
        Integer,
        nullable=False,
        default=3,
        doc="Expected number of turns to reach correct diagnosis"
    )
    max_acceptable_turns = Column(
        Integer,
        nullable=False,
        default=8,
        doc="Maximum turns before considering evaluation failed"
    )
    
    # Data provenance
    created_by = Column(
        String(100),
        nullable=False,
        doc="Who created this gold standard (clinician ID, system, etc.)"
    )
    reviewed_by = Column(
        String(100),
        nullable=True,
        doc="Clinical expert who reviewed/approved this case"
    )
    clinical_notes = Column(
        Text,
        nullable=True,
        doc="Additional clinical context or reasoning notes"
    )
    
    # Version control
    version = Column(
        String(10),
        nullable=False,
        default="1.0",
        doc="Version of this gold standard case"
    )
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        doc="Whether to include in current training/evaluation sets"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        doc="When this standard was created"
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        onupdate=func.now(),
        doc="Last modification timestamp"
    )
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('minimum_confidence_threshold >= 0 AND minimum_confidence_threshold <= 100', 
                       name='valid_confidence_threshold'),
        CheckConstraint('patient_age >= 0 AND patient_age <= 120', 
                       name='valid_patient_age'),
        CheckConstraint('expected_turn_count >= 1 AND expected_turn_count <= max_acceptable_turns', 
                       name='valid_turn_counts'),
        
        # Indexes for training queries
        Index('idx_gold_standards_active', 'is_active', 'primary_symptom'),
        Index('idx_gold_standards_outcome', 'expected_outcome', 'is_active'),
        Index('idx_gold_standards_escalation', 'should_escalate', 'expected_outcome'),
        Index('idx_gold_standards_symptom', 'primary_symptom', 'patient_age'),
    )

    def to_training_example(self) -> Dict[str, Any]:
        """Convert to DSPy training example format"""
        return {
            'standard_id': str(self.standard_id),
            'input': {
                'patient_age': self.patient_age,
                'patient_gender': self.patient_gender,
                'conversation_turns': self.conversation_dialogue,
                'relevant_protocols': self.relevant_protocols or []
            },
            'expected_output': {
                'medical_outcome': self.expected_outcome.value,
                'red_flags': self.expected_red_flags or [],
                'should_escalate': self.should_escalate,
                'min_confidence': self.minimum_confidence_threshold
            },
            'metadata': {
                'primary_symptom': self.primary_symptom,
                'max_turns': self.max_acceptable_turns,
                'version': self.version
            }
        }

    def validate_against_prediction(self, prediction: Dict[str, Any]) -> Dict[str, bool]:
        """Validate a model prediction against this gold standard"""
        results = {
            'correct_outcome': prediction.get('medical_outcome') == self.expected_outcome.value,
            'sufficient_confidence': prediction.get('confidence_score', 0) >= self.minimum_confidence_threshold,
            'correct_escalation': prediction.get('requires_human_review', False) == self.should_escalate,
            'within_turn_limit': prediction.get('turn_count', 0) <= self.max_acceptable_turns
        }
        
        # Check red flag detection
        predicted_flags = set(prediction.get('red_flags_detected', []))
        expected_flags = set(self.expected_red_flags or [])
        results['red_flags_detected'] = len(expected_flags.intersection(predicted_flags)) >= len(expected_flags) * 0.8
        
        return results

    @classmethod 
    def get_training_set(cls, session, symptom_filter: Optional[str] = None, limit: int = 50) -> List['GoldStandardDialogue']:
        """Get active gold standards for training"""
        query = session.query(cls).filter(cls.is_active)
        
        if symptom_filter:
            query = query.filter(cls.primary_symptom == symptom_filter)
            
        return query.order_by(cls.created_at.desc()).limit(limit).all()

    @classmethod
    def get_evaluation_set(cls, session, outcome_filter: Optional[MedicalOutcome] = None) -> List['GoldStandardDialogue']:
        """Get gold standards for model evaluation"""
        query = session.query(cls).filter(cls.is_active)
        
        if outcome_filter:
            query = query.filter(cls.expected_outcome == outcome_filter)
            
        return query.order_by(cls.primary_symptom, cls.patient_age).all()

    @classmethod
    def get_emergency_examples(cls, session) -> List['GoldStandardDialogue']:
        """Get gold standards specifically for emergency scenarios"""
        return session.query(cls).filter(
            cls.is_active,
            cls.expected_outcome == MedicalOutcome.EMERGENCY
        ).order_by(cls.created_at.desc()).all()

    def __repr__(self) -> str:
        return f"<GoldStandardDialogue(id={self.standard_id}, symptom={self.primary_symptom}, outcome={self.expected_outcome})>"
