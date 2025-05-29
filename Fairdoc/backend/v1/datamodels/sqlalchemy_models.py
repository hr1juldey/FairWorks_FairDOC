"""
SQLAlchemy Database Models
Data structure definitions for medical triage system
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.dialects.postgresql import JSONB
import uuid
from datetime import datetime

# SQLAlchemy base
Base = declarative_base()


class CaseReportDB(Base):
    """Case report database model with proper JSON mutation tracking"""
    __tablename__ = "case_reports"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(
        String(50),
        unique=True,
        index=True,
        default=lambda: str(uuid.uuid4())
    )
    patient_id = Column(String(50), index=True, nullable=False)
    session_id = Column(String(50), nullable=True)
    
    # Demographics
    age = Column(Integer, nullable=False)
    gender = Column(String(50), nullable=False)
    pregnancy_status = Column(Boolean, nullable=True)
    ethnicity = Column(String(100), nullable=True)
    postcode_sector = Column(String(10), nullable=True)
    
    # Clinical data
    chief_complaint = Column(Text, nullable=False)
    presenting_complaint_category = Column(String(100), nullable=False)
    
    # Vital signs (Mutable JSON for proper change detection)
    vital_signs = Column(MutableDict.as_mutable(JSONB), nullable=True)
    medical_history = Column(MutableDict.as_mutable(JSONB), nullable=True)
    chest_pain_assessment = Column(MutableDict.as_mutable(JSONB), nullable=True)
    chest_pain_red_flags = Column(MutableDict.as_mutable(JSONB), nullable=True)
    associated_symptoms = Column(MutableDict.as_mutable(JSONB), nullable=True)
    ai_assessment = Column(MutableDict.as_mutable(JSONB), nullable=True)
    
    # Scoring for ML/triage
    urgency_score = Column(Float, nullable=True)
    importance_score = Column(Float, nullable=True)
    
    # Case management
    status = Column(String(20), default="created", nullable=False)
    created_at = Column(DateTime, default=datetime.datetime, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.datetime,
        onupdate=datetime.datetime,
        nullable=False
    )
    completed_at = Column(DateTime, nullable=True)
    
    # Quality metrics
    data_quality_score = Column(Float, nullable=True)
    protocol_compliance = Column(Boolean, nullable=True)
    clinical_review_required = Column(Boolean, default=False, nullable=False)
    pdf_report_url = Column(String(500), nullable=True)
    
    # Relationships
    responses = relationship(
        "PatientResponseDB",
        back_populates="case_report",
        cascade="all, delete-orphan"
    )
    uploaded_files = relationship(
        "UploadedFileDB",
        back_populates="case_report",
        cascade="all, delete-orphan"
    )


class NICEProtocolQuestionDB(Base):
    """Enhanced NICE protocol questions with comprehensive JSON support"""
    __tablename__ = "nice_protocol_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, unique=True, index=True, nullable=False)
    protocol_id = Column(Integer, nullable=True, index=True)
    protocol_name = Column(String(200), nullable=True)
    nice_guideline = Column(String(50), nullable=True)
    urgency_category = Column(String(50), nullable=True, index=True)
    condition_type = Column(String(100), nullable=True, index=True)
    category = Column(String(50), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(20), nullable=False)
    
    # Question configuration (Mutable JSON)
    options = Column(MutableDict.as_mutable(JSONB), nullable=True)
    validation_rules = Column(MutableDict.as_mutable(JSONB), nullable=True)
    next_question_logic = Column(MutableDict.as_mutable(JSONB), nullable=True)
    
    # Question metadata
    is_required = Column(Boolean, default=True, nullable=False)
    is_red_flag = Column(Boolean, default=False, nullable=False)
    order_index = Column(Integer, nullable=False, index=True)
    condition_specific = Column(String(50), nullable=True)
    scoring_weight = Column(Float, nullable=True)
    protocol_version = Column(String(20), default="v1.0", nullable=False)
    nice_guideline_ref = Column(String(50), nullable=True)
    clinical_rationale = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.datetime,
        onupdate=datetime.datetime,
        nullable=False
    )
    
    # Relationships
    responses = relationship("PatientResponseDB", back_populates="question")


class PatientResponseDB(Base):
    """Patient responses database model"""
    __tablename__ = "patient_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(
        String(50),
        unique=True,
        index=True,
        default=lambda: str(uuid.uuid4())
    )
    case_report_id = Column(
        Integer,
        ForeignKey("case_reports.id"),
        nullable=False,
        index=True
    )
    question_id = Column(
        Integer,
        ForeignKey("nice_protocol_questions.question_id"),
        nullable=False,
        index=True
    )
    response_text = Column(Text, nullable=False)
    response_value = Column(MutableDict.as_mutable(JSONB), nullable=True)
    confidence_level = Column(Float, nullable=True)
    response_time_seconds = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime, nullable=False)
    
    # Relationships
    case_report = relationship("CaseReportDB", back_populates="responses")
    question = relationship("NICEProtocolQuestionDB", back_populates="responses")


class UploadedFileDB(Base):
    """Uploaded files database model"""
    __tablename__ = "uploaded_files"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(
        String(50),
        unique=True,
        index=True,
        default=lambda: str(uuid.uuid4())
    )
    case_report_id = Column(
        Integer,
        ForeignKey("case_reports.id"),
        nullable=False,
        index=True
    )
    filename = Column(String(255), nullable=False)
    file_type = Column(String(100), nullable=False)
    file_category = Column(String(50), nullable=False)
    minio_url = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime, nullable=False)
    description = Column(Text, nullable=True)
    ai_analysis = Column(MutableDict.as_mutable(JSONB), nullable=True)
    
    # Relationships
    case_report = relationship("CaseReportDB", back_populates="uploaded_files")


# Export models
__all__ = [
    "Base",
    "CaseReportDB",
    "NICEProtocolQuestionDB",
    "PatientResponseDB",
    "UploadedFileDB",
]
