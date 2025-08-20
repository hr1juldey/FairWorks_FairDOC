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
    Text, Index, CheckConstraint, Enum as SQLEnum, select
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
        """FIX: Proper enum conversion with fallback using key parameter"""
        from src.app2.utils.outcome_mapper import OutcomeMapper
        import structlog
        
        logger = structlog.get_logger(__name__)
        
        try:
            if isinstance(outcome, str):
                # Convert string to proper enum first
                mapped_outcome = OutcomeMapper.to_triage(outcome)
                logger.debug(f"Validated {key}: '{outcome}' -> '{mapped_outcome.value}'")
                return mapped_outcome.value
            elif hasattr(outcome, 'value'):
                logger.debug(f"Validated {key}: enum value '{outcome.value}'")
                return outcome.value
            else:
                outcome_str = str(outcome)
                logger.debug(f"Validated {key}: converted to string '{outcome_str}'")
                return outcome_str
        except Exception as e:
            logger.warning(f"Validation failed for field '{key}' with value '{outcome}': {e}")
            logger.info(f"Using fallback value for field '{key}': 'need_more_questions'")
            return "need_more_questions"  # Safe fallback


    
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
        """Convert to DSPy training example format with robust field handling"""
        from src.app2.utils.outcome_mapper import OutcomeMapper
        
        # Safe handling of expected_outcome field
        try:
            if isinstance(self.expected_outcome, str):
                outcome_value = self.expected_outcome
            elif hasattr(self.expected_outcome, 'value'):
                outcome_value = self.expected_outcome.value
            else:
                outcome_value = str(self.expected_outcome)
        except Exception:
            outcome_value = "need_more_questions"  # Safe fallback
        
        return {
            'standard_id': str(self.standard_id),
            'title': self.title,
            'conversation_dialogue': self.conversation_dialogue,
            'expected_outcome': outcome_value,  # FIX: Robust field access
            'relevant_protocols': self.relevant_protocols or [],
            'patient_context': {
                'age': self.patient_age,
                'gender': self.patient_gender,
                'primary_symptom': self.primary_symptom
            },
            'evaluation_criteria': {
                'expected_red_flags': self.expected_red_flags or [],
                'should_escalate': self.should_escalate,
                'min_confidence': self.minimum_confidence_threshold,
                'max_turns': self.max_acceptable_turns
            },
            'metadata': {
                'version': self.version,
                'created_by': self.created_by,
                'clinical_notes': self.clinical_notes
            }
        }

    def validate_against_prediction(self, prediction: Dict[str, Any]) -> Dict[str, bool]:
        """Validate a model prediction against this gold standard with robust field handling"""
        from src.app2.utils.outcome_mapper import OutcomeMapper
        
        try:
            # Safe comparison of expected vs predicted outcome
            expected_outcome = self.expected_outcome
            if isinstance(expected_outcome, str):
                expected_value = expected_outcome
            elif hasattr(expected_outcome, 'value'):
                expected_value = expected_outcome.value
            else:
                expected_value = str(expected_outcome)
            
            predicted_outcome = prediction.get('medical_outcome', 'unknown')
            
            # Normalize both for comparison using OutcomeMapper
            try:
                expected_normalized = OutcomeMapper.canonical(expected_value)
                predicted_normalized = OutcomeMapper.canonical(predicted_outcome)
                outcome_correct = expected_normalized == predicted_normalized
            except Exception:
                # Fallback to direct string comparison
                outcome_correct = str(expected_value).lower() == str(predicted_outcome).lower()
            
            results = {
                'correct_outcome': outcome_correct,
                'sufficient_confidence': prediction.get('confidence_score', 0) >= self.minimum_confidence_threshold,
                'correct_escalation': prediction.get('requires_human_review', False) == self.should_escalate,
                'within_turn_limit': prediction.get('turn_count', 0) <= self.max_acceptable_turns
            }
            
            # Check red flag detection with safe handling
            predicted_flags = set(prediction.get('red_flags_detected', []))
            expected_flags = set(self.expected_red_flags or [])
            
            if expected_flags:
                results['red_flags_detected'] = len(expected_flags.intersection(predicted_flags)) >= len(expected_flags) * 0.8
            else:
                results['red_flags_detected'] = True  # No flags expected
            
            return results
            
        except Exception as e:
            # Safe fallback results
            return {
                'correct_outcome': False,
                'sufficient_confidence': False,
                'correct_escalation': False,
                'within_turn_limit': True,
                'red_flags_detected': False,
                'validation_error': str(e)
            }



    # NEW (async compatible)
    @classmethod
    async def get_training_set(cls, session, symptom_filter: Optional[str] = None, limit: int = 50) -> List['GoldStandardDialogue']:
        """Get active gold standards for training"""
        
        stmt = select(cls).filter(cls.is_active)
        if symptom_filter:
            stmt = stmt.filter(cls.primary_symptom == symptom_filter)
        stmt = stmt.order_by(cls.created_at.desc()).limit(limit)
        
        result = await session.execute(stmt)
        return result.scalars().all()

    @classmethod
    async def get_evaluation_set(cls, session, outcome_filter: Optional[MedicalOutcome] = None, limit: int = 50) -> List['GoldStandardDialogue']:
        """Get gold standards for model evaluation"""
        from sqlalchemy import select
        
        stmt = select(cls).filter(cls.is_active)
        if outcome_filter:
            stmt = stmt.filter(cls.expected_outcome == outcome_filter)
        stmt = stmt.order_by(cls.primary_symptom, cls.patient_age)
        if limit:
            stmt = stmt.limit(limit)
        
        result = await session.execute(stmt)
        return result.scalars().all()

    @classmethod
    async def get_emergency_examples(cls, session) -> List['GoldStandardDialogue']:
        """Get gold standards specifically for emergency scenarios"""
        from sqlalchemy import select
        
        stmt = select(cls).filter(
            cls.is_active,
            cls.expected_outcome == MedicalOutcome.EMERGENCY
        ).order_by(cls.created_at.desc())
        
        result = await session.execute(stmt)
        return result.scalars().all()

    def __repr__(self) -> str:
        return f"<GoldStandardDialogue(id={self.standard_id}, symptom={self.primary_symptom}, outcome={self.expected_outcome})>"
