"""
V1 Database Operations and Connection Management
Clean data access layer for NHS medical triage system
Proper separation of concerns - operations only, models imported separately
"""
# === SMART IMPORT SETUP - ADD TO TOP OF FILE ===
import sys
import os
from pathlib import Path

# Setup paths once to prevent double imports
if not hasattr(sys, '_fairdoc_paths_setup'):
    current_dir = Path(__file__).parent
    v1_dir = current_dir.parent
    backend_dir = v1_dir.parent
    project_root = backend_dir.parent
    
    paths_to_add = [str(project_root), str(backend_dir), str(v1_dir)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    sys._fairdoc_paths_setup = True

# Standard imports first
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Generator, Dict, List, Any, Optional
from datetime import datetime
import json
import logging

# Smart internal imports with fallbacks
try:
    # Try absolute imports first
    from datamodels.sqlalchemy_models import Base, CaseReportDB, NICEProtocolQuestionDB
    from core.config import get_v1_settings
except ImportError:
    # Fallback to relative imports
    from datamodels.sqlalchemy_models import Base, CaseReportDB, NICEProtocolQuestionDB
    from core.config import get_v1_settings

# === END SMART IMPORT SETUP ===

from pathlib import Path

# Import models from proper location (separation of concerns)
from datamodels.sqlalchemy_models import (
    Base,
    CaseReportDB,
    NICEProtocolQuestionDB,
    PatientResponseDB,
    UploadedFileDB,
    # NHS-specific models if they exist
    UserDB,
    UserSessionDB,
    DoctorProfileDB
)

# Import configuration
from core.config import get_v1_settings

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION AND CONNECTION
# =============================================================================

settings = get_v1_settings()

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600    # Recycle connections every hour
)

# Create sessionmaker
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Keep objects accessible after commit
)

# =============================================================================
# DATABASE CONNECTION UTILITIES
# =============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get database session
    Ensures proper connection management and cleanup
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def test_database_connection() -> bool:
    """Test database connectivity and basic functionality"""
    try:
        db = SessionLocal()
        
        # Test basic connection
        result = db.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        logger.info("✅ Database connection successful")
        logger.info(f"📊 PostgreSQL version: {version.split(',')[0]}")
        
        # Test table existence
        table_counts = get_table_counts(db)
        logger.info("📋 Table status:")
        for table_name, count in table_counts.items():
            logger.info(f"   {table_name}: {count} records")
        
        db.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False

def get_table_counts(db: Session) -> Dict[str, int]:
    """Get record counts for all main tables"""
    try:
        return {
            "case_reports": db.query(CaseReportDB).count(),
            "nice_questions": db.query(NICEProtocolQuestionDB).count(),
            "patient_responses": db.query(PatientResponseDB).count(),
            "uploaded_files": db.query(UploadedFileDB).count(),
            "users": db.query(UserDB).count() if UserDB else 0,
            "user_sessions": db.query(UserSessionDB).count() if UserSessionDB else 0,
            "doctor_profiles": db.query(DoctorProfileDB).count() if DoctorProfileDB else 0
        }
    except Exception as e:
        logger.warning(f"Could not get all table counts: {e}")
        return {"error": "Could not fetch counts"}

# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

async def init_database() -> bool:
    """
    Initialize database tables and load NICE protocol data
    Returns True if successful, False otherwise
    """
    try:
        logger.info("🔨 Initializing V1 database.")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Load NICE protocols
        protocols_loaded = await load_nice_protocols()
        if protocols_loaded:
            logger.info("✅ NICE protocols loaded successfully")
        else:
            logger.warning("⚠️ NICE protocols loading failed")
        
        # Verify initialization
        if test_database_connection():
            logger.info("✅ Database initialization completed successfully")
            return True
        else:
            logger.error("❌ Database initialization verification failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

async def load_nice_protocols() -> bool:
    """Load NICE protocols from JSON file into database"""
    try:
        db = SessionLocal()
        
        # Check if protocols already exist
        existing_count = db.query(NICEProtocolQuestionDB).count()
        if existing_count > 0:
            logger.info(f"📋 NICE protocols already loaded ({existing_count} questions)")
            db.close()
            return True
        
        # Load from JSON file
        protocols_data = load_protocols_from_json()
        if not protocols_data:
            logger.error("❌ No protocols data found")
            db.close()
            return False
        
        # Insert protocols
        questions_inserted = insert_protocols_into_db(db, protocols_data)
        
        if questions_inserted > 0:
            logger.info(f"✅ Loaded {questions_inserted} NICE protocol questions")
            log_protocol_summary(db)
            db.close()
            return True
        else:
            logger.error("❌ No questions were inserted")
            db.close()
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to load NICE protocols: {e}")
        return False

def load_protocols_from_json() -> Optional[Dict[str, Any]]:
    """Load protocols from JSON file with fallback options"""
    
    # Try multiple possible locations for the JSON file
    possible_paths = [
        Path(__file__).parent / "nice_questions.json",
        Path(__file__).parent / "nice_protocols_complete.json",
        Path(__file__).parent.parent / "data" / "nice_questions.json",
        settings.DATA_DIR / "nice_questions.json"
    ]
    
    for json_path in possible_paths:
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    logger.info(f"📋 Loaded protocols from: {json_path}")
                    return data
            except Exception as e:
                logger.warning(f"⚠️ Failed to load from {json_path}: {e}")
                continue
    
    # If no file found, return fallback data
    logger.warning("⚠️ No protocol JSON files found, using fallback data")
    return get_fallback_protocols()

def insert_protocols_into_db(db: Session, protocols_data: Dict[str, Any]) -> int:
    """Insert protocol questions from JSON data into database"""
    
    questions_inserted = 0
    
    try:
        # Handle comprehensive JSON structure
        if "protocols" in protocols_data:
            protocols = protocols_data["protocols"]
            
            for protocol in protocols:
                protocol_id = protocol.get("protocol_id")
                protocol_name = protocol.get("protocol_name")
                nice_guideline = protocol.get("nice_guideline")
                urgency_category = protocol.get("urgency_category")
                condition_type = protocol.get("condition_type")
                
                questions = protocol.get("questions", [])
                
                for question_data in questions:
                    try:
                        # Create question with protocol context
                        question = NICEProtocolQuestionDB(
                            # Core question fields
                            question_id=question_data["question_id"],
                            category=question_data["category"],
                            question_text=question_data["question_text"],
                            question_type=question_data["question_type"],
                            options=question_data.get("options"),
                            validation_rules=question_data.get("validation_rules"),
                            is_required=question_data.get("is_required", True),
                            is_red_flag=question_data.get("is_red_flag", False),
                            order_index=question_data["order_index"],
                            
                            # Protocol context
                            protocol_id=protocol_id,
                            protocol_name=protocol_name,
                            nice_guideline=nice_guideline,
                            urgency_category=urgency_category,
                            condition_type=condition_type,
                            
                            # Additional fields
                            condition_specific=question_data.get("condition_specific"),
                            scoring_weight=question_data.get("scoring_weight"),
                            nice_guideline_ref=question_data.get("nice_guideline_ref"),
                            clinical_rationale=question_data.get("clinical_rationale")
                        )
                        
                        db.add(question)
                        questions_inserted += 1
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to insert question {question_data.get('question_id')}: {e}")
                        continue
        
        # Handle simple structure (fallback)
        elif "questions" in protocols_data:
            questions = protocols_data["questions"]
            
            for question_data in questions:
                try:
                    question = NICEProtocolQuestionDB(
                        question_id=question_data["question_id"],
                        category=question_data["category"],
                        question_text=question_data["question_text"],
                        question_type=question_data["question_type"],
                        options=question_data.get("options"),
                        validation_rules=question_data.get("validation_rules"),
                        is_required=question_data.get("is_required", True),
                        is_red_flag=question_data.get("is_red_flag", False),
                        order_index=question_data["order_index"],
                        scoring_weight=question_data.get("scoring_weight")
                    )
                    
                    db.add(question)
                    questions_inserted += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to insert question {question_data.get('question_id')}: {e}")
                    continue
        
        # Commit all questions
        db.commit()
        logger.info(f"✅ Successfully inserted {questions_inserted} questions")
        
    except Exception as e:
        logger.error(f"❌ Failed to insert protocols: {e}")
        db.rollback()
        questions_inserted = 0
    
    return questions_inserted

def log_protocol_summary(db: Session):
    """Log summary of loaded protocols"""
    try:
        # Get protocol distribution
        protocol_stats = (
            db.query(
                NICEProtocolQuestionDB.protocol_name,
                NICEProtocolQuestionDB.urgency_category,
                func.count(NICEProtocolQuestionDB.id).label('question_count')
            )
            .group_by(
                NICEProtocolQuestionDB.protocol_name,
                NICEProtocolQuestionDB.urgency_category
            )
            .order_by(NICEProtocolQuestionDB.protocol_id)
            .all()
        )
        
        logger.info("📊 Protocol Summary:")
        for name, urgency, count in protocol_stats:
            protocol_name = name or "Unknown Protocol"
            urgency_cat = urgency or "No Category"
            logger.info(f"   {protocol_name} ({urgency_cat}): {count} questions")
        
        # Get red flag count
        red_flag_count = (
            db.query(NICEProtocolQuestionDB)
            .filter(NICEProtocolQuestionDB.is_red_flag)
            .count()
        )
        logger.info(f"🚩 Red flag questions: {red_flag_count}")
        
    except Exception as e:
        logger.warning(f"Could not generate protocol summary: {e}")

def get_fallback_protocols() -> Dict[str, Any]:
    """Fallback protocol data for basic functionality"""
    return {
        "protocol_collection": {
            "name": "Basic Medical Triage Protocols",
            "version": "v1.0-fallback",
            "total_protocols": 2
        },
        "protocols": [
            {
                "protocol_id": 1,
                "protocol_name": "Basic Chest Pain Assessment",
                "nice_guideline": "NICE CG95",
                "urgency_category": "111_urgent",
                "condition_type": "cardiovascular",
                "questions": [
                    {
                        "question_id": 1,
                        "category": "demographics",
                        "question_text": "What is your age?",
                        "question_type": "number",
                        "validation_rules": {"min": 0, "max": 150, "required": True},
                        "is_required": True,
                        "is_red_flag": False,
                        "order_index": 1,
                        "scoring_weight": 0.2
                    },
                    {
                        "question_id": 2,
                        "category": "pain_assessment",
                        "question_text": "How severe is your chest pain (0-10)?",
                        "question_type": "scale_1_10",
                        "options": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                        "validation_rules": {"min": 0, "max": 10, "required": True},
                        "is_required": True,
                        "is_red_flag": True,
                        "order_index": 2,
                        "scoring_weight": 0.4
                    }
                ]
            },
            {
                "protocol_id": 2,
                "protocol_name": "Basic Shortness of Breath Assessment",
                "nice_guideline": "NICE CG191",
                "urgency_category": "111_urgent",
                "condition_type": "respiratory",
                "questions": [
                    {
                        "question_id": 3,
                        "category": "respiratory_assessment",
                        "question_text": "How severe is your breathing difficulty?",
                        "question_type": "multiple_choice",
                        "options": ["Mild", "Moderate", "Severe"],
                        "validation_rules": {"required": True},
                        "is_required": True,
                        "is_red_flag": True,
                        "order_index": 1,
                        "scoring_weight": 0.4
                    }
                ]
            }
        ]
    }

# =============================================================================
# ENHANCED CRUD OPERATIONS
# =============================================================================

class CaseReportCRUD:
    """Enhanced CRUD operations for case reports"""

    @staticmethod
    def create_case_report(
        db: Session,
        patient_id: str,
        age: int,
        gender: str,
        chief_complaint: str,
        **kwargs
    ) -> CaseReportDB:
        """Create new case report with comprehensive data handling"""
        try:
            case_report = CaseReportDB(
                patient_id=patient_id,
                age=age,
                gender=gender,
                chief_complaint=chief_complaint,
                presenting_complaint_category=kwargs.get("presenting_complaint_category", "triage_assessment"),
                vital_signs=kwargs.get("vital_signs", {}),
                medical_history=kwargs.get("medical_history", {}),
                ethnicity=kwargs.get("ethnicity"),
                postcode_sector=kwargs.get("postcode_sector"),
                pregnancy_status=kwargs.get("pregnancy_status")
            )
            
            db.add(case_report)
            db.commit()
            db.refresh(case_report)
            
            logger.info(f"✅ Created case report: {case_report.case_id}")
            return case_report
            
        except Exception as e:
            logger.error(f"❌ Failed to create case report: {e}")
            db.rollback()
            raise

    @staticmethod
    def get_case_report(db: Session, case_id: str) -> Optional[CaseReportDB]:
        """Get case report by case_id"""
        try:
            return (
                db.query(CaseReportDB)
                .filter(CaseReportDB.case_id == case_id)
                .first()
            )
        except Exception as e:
            logger.error(f"❌ Failed to get case report {case_id}: {e}")
            return None

    @staticmethod
    def update_case_report(
        db: Session,
        case_id: str,
        **kwargs
    ) -> Optional[CaseReportDB]:
        """Update case report with proper JSON field handling"""
        try:
            case_report = (
                db.query(CaseReportDB)
                .filter(CaseReportDB.case_id == case_id)
                .first()
            )
            
            if not case_report:
                return None
            
            # Handle JSON field updates with proper mutation detection
            json_fields = [
                "vital_signs", "medical_history", "ai_assessment",
                "chest_pain_assessment", "chest_pain_red_flags",
                "associated_symptoms"
            ]
            
            for key, value in kwargs.items():
                if hasattr(case_report, key):
                    if key in json_fields and isinstance(value, dict):
                        # For JSON fields, merge with existing data
                        current_value = getattr(case_report, key) or {}
                        current_value.update(value)
                        setattr(case_report, key, current_value)
                    else:
                        setattr(case_report, key, value)
            
            case_report.updated_at = datetime.datetime()
            db.commit()
            db.refresh(case_report)
            
            logger.info(f"✅ Updated case report: {case_id}")
            return case_report
            
        except Exception as e:
            logger.error(f"❌ Failed to update case report {case_id}: {e}")
            db.rollback()
            return None

    @staticmethod
    def list_case_reports(
        db: Session,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[CaseReportDB]:
        """List case reports with pagination and filtering"""
        try:
            query = db.query(CaseReportDB)
            
            if status_filter:
                query = query.filter(CaseReportDB.status == status_filter)
            
            return (
                query
                .order_by(CaseReportDB.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to list case reports: {e}")
            return []

class NICEQuestionCRUD:
    """Enhanced CRUD operations for NICE protocol questions"""

    @staticmethod
    def get_questions_by_protocol(
        db: Session,
        protocol_id: int
    ) -> List[NICEProtocolQuestionDB]:
        """Get questions by protocol ID with proper ordering"""
        try:
            return (
                db.query(NICEProtocolQuestionDB)
                .filter(NICEProtocolQuestionDB.protocol_id == protocol_id)
                .order_by(NICEProtocolQuestionDB.order_index)
                .all()
            )
        except Exception as e:
            logger.error(f"❌ Failed to get questions for protocol {protocol_id}: {e}")
            return []

    @staticmethod
    def get_questions_by_urgency(
        db: Session,
        urgency_category: str
    ) -> List[NICEProtocolQuestionDB]:
        """Get questions by urgency category"""
        try:
            return (
                db.query(NICEProtocolQuestionDB)
                .filter(NICEProtocolQuestionDB.urgency_category == urgency_category)
                .order_by(
                    NICEProtocolQuestionDB.protocol_id,
                    NICEProtocolQuestionDB.order_index
                )
                .all()
            )
        except Exception as e:
            logger.error(f"❌ Failed to get questions for urgency {urgency_category}: {e}")
            return []

    @staticmethod
    def get_red_flag_questions(db: Session) -> List[NICEProtocolQuestionDB]:
        """Get all red flag questions across protocols"""
        try:
            return (
                db.query(NICEProtocolQuestionDB)
                .filter(NICEProtocolQuestionDB.is_red_flag)
                .order_by(
                    NICEProtocolQuestionDB.protocol_id,
                    NICEProtocolQuestionDB.order_index
                )
                .all()
            )
        except Exception as e:
            logger.error(f"❌ Failed to get red flag questions: {e}")
            return []

    @staticmethod
    def get_question_by_id(
        db: Session,
        question_id: int
    ) -> Optional[NICEProtocolQuestionDB]:
        """Get question by question_id"""
        try:
            return (
                db.query(NICEProtocolQuestionDB)
                .filter(NICEProtocolQuestionDB.question_id == question_id)
                .first()
            )
        except Exception as e:
            logger.error(f"❌ Failed to get question {question_id}: {e}")
            return None

class PatientResponseCRUD:
    """Enhanced CRUD operations for patient responses"""

    @staticmethod
    def create_response(
        db: Session,
        case_report_id: int,
        question_id: int,
        response_text: str,
        response_value: Optional[Dict[str, Any]] = None
    ) -> PatientResponseDB:
        """Create patient response"""
        try:
            response = PatientResponseDB(
                case_report_id=case_report_id,
                question_id=question_id,
                response_text=response_text,
                response_value=response_value or {}
            )
            
            db.add(response)
            db.commit()
            db.refresh(response)
            
            logger.info(f"✅ Created response for case {case_report_id}, question {question_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to create response: {e}")
            db.rollback()
            raise

    @staticmethod
    def create_multiple_responses(
        db: Session,
        case_report_id: int,
        responses: List[Dict[str, Any]]
    ) -> List[PatientResponseDB]:
        """Create multiple patient responses in batch"""
        try:
            created_responses = []
            
            for response_data in responses:
                response = PatientResponseDB(
                    case_report_id=case_report_id,
                    question_id=response_data["question_id"],
                    response_text=response_data["response_text"],
                    response_value=response_data.get("response_value", {})
                )
                
                db.add(response)
                created_responses.append(response)
            
            db.commit()
            
            for response in created_responses:
                db.refresh(response)
            
            logger.info(f"✅ Created {len(created_responses)} responses for case {case_report_id}")
            return created_responses
            
        except Exception as e:
            logger.error(f"❌ Failed to create multiple responses: {e}")
            db.rollback()
            raise

    @staticmethod
    def get_responses_for_case(
        db: Session,
        case_report_id: int
    ) -> List[PatientResponseDB]:
        """Get all responses for a case report"""
        try:
            return (
                db.query(PatientResponseDB)
                .filter(PatientResponseDB.case_report_id == case_report_id)
                .order_by(PatientResponseDB.timestamp)
                .all()
            )
        except Exception as e:
            logger.error(f"❌ Failed to get responses for case {case_report_id}: {e}")
            return []

# =============================================================================
# DATABASE HEALTH AND MONITORING
# =============================================================================

def get_database_health() -> Dict[str, Any]:
    """Comprehensive database health check with detailed metrics"""
    try:
        db = SessionLocal()
        
        # Basic connection test
        result = db.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        
        # Get table counts
        table_counts = get_table_counts(db)
        
        # Protocol analysis
        protocol_stats = (
            db.query(
                NICEProtocolQuestionDB.protocol_name,
                NICEProtocolQuestionDB.urgency_category,
                func.count(NICEProtocolQuestionDB.id).label('count')
            )
            .group_by(
                NICEProtocolQuestionDB.protocol_name,
                NICEProtocolQuestionDB.urgency_category
            )
            .all()
        )
        
        # Red flag analysis
        red_flag_count = (
            db.query(NICEProtocolQuestionDB)
            .filter(NICEProtocolQuestionDB.is_red_flag)
            .count()
        )
        
        # Recent activity (last 24 hours)
        from datetime import timedelta
        yesterday = datetime.datetime() - timedelta(days=1)
        recent_cases = (
            db.query(CaseReportDB)
            .filter(CaseReportDB.created_at >= yesterday)
            .count()
        )
        
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.datetime().isoformat(),
            "database": {
                "version": version.split(',')[0],
                "connection": "active",
                "pool_size": settings.DATABASE_POOL_SIZE,
                "max_overflow": settings.DATABASE_MAX_OVERFLOW
            },
            "tables": table_counts,
            "protocols": [
                {
                    "name": name or "Unknown Protocol",
                    "urgency_category": urgency or "No Category",
                    "question_count": count
                }
                for name, urgency, count in protocol_stats
            ],
            "metrics": {
                "total_protocols": len(protocol_stats),
                "red_flag_questions": red_flag_count,
                "recent_cases_24h": recent_cases,
                "total_questions": table_counts.get("nice_questions", 0)
            },
            "features": {
                "json_fields": "mutable_tracking_enabled",
                "crud_operations": "enhanced",
                "error_handling": "comprehensive",
                "logging": "enabled"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.datetime().isoformat(),
            "error": str(e),
            "connection": "failed"
        }

# =============================================================================
# EXPORT - CLEAN INTERFACE
# =============================================================================


__all__ = [
    # Database setup
    "engine",
    "SessionLocal",
    "get_db",
    
    # Initialization
    "init_database",
    "test_database_connection",
    "get_database_health",
    
    # CRUD Operations
    "CaseReportCRUD",
    "NICEQuestionCRUD",
    "PatientResponseCRUD",
    
    # Utilities
    "load_nice_protocols",
    "get_table_counts"
]
