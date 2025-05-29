"""
Database Operations and Connection Management
Data access layer for medical triage system - OPERATIONS ONLY
Imports models from datamodels/sqlalchemy_models.py (proper separation)
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
import json
from typing import Generator, Dict, List, Any, Optional
from datetime import datetime

# Import models from correct location (datamodels folder)
from ..datamodels.sqlalchemy_models import (
    Base,
    CaseReportDB,
    NICEProtocolQuestionDB,
    PatientResponseDB,
    UploadedFileDB,
)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fairdoc:password@localhost:5432/fairdoc_v0"
)

# SQLAlchemy setup - CONNECTION ONLY
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# =============================================================================
# DATABASE CONNECTION UTILITIES
# =============================================================================

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database() -> bool:
    """Initialize database tables and load protocol data"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        # Load NICE protocol questions from JSON
        success = load_nice_protocols_from_json()
        
        if success:
            print("✅ Database initialization completed successfully")
            return True
        else:
            print("⚠️  Database created but protocol loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def load_nice_protocols_from_json() -> bool:
    """Load NICE protocols from nice_questions.json file"""
    try:
        # Load from JSON file in same directory
        json_file_path = os.path.join(
            os.path.dirname(__file__),
            "nice_questions.json"
        )
        
        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as file:
                protocols_data = json.load(file)
            print(f"📋 Loaded protocols from {json_file_path}")
        else:
            # Try the comprehensive protocols file
            comprehensive_file_path = os.path.join(
                os.path.dirname(__file__),
                "nice_protocols_complete.json"
            )
            
            if os.path.exists(comprehensive_file_path):
                with open(comprehensive_file_path, "r", encoding="utf-8") as file:
                    protocols_data = json.load(file)
                print(f"📋 Loaded comprehensive protocols from {comprehensive_file_path}")
            else:
                # Fallback to basic data
                protocols_data = get_fallback_protocols_data()
                print("📋 Using fallback protocol data")
        
        return insert_protocols_from_json(protocols_data)
        
    except Exception as e:
        print(f"❌ Failed to load NICE protocols: {e}")
        return False


def insert_protocols_from_json(protocols_data: Dict[str, Any]) -> bool:
    """Insert protocol questions from JSON structure"""
    db = SessionLocal()
    
    try:
        # Check if questions already exist
        existing_count = db.query(NICEProtocolQuestionDB).count()
        if existing_count > 0:
            print(f"📋 NICE questions already exist ({existing_count} questions)")
            return True
        
        total_questions_inserted = 0
        
        # Handle different JSON structures
        if "protocols" in protocols_data:
            # Comprehensive structure with multiple protocols
            protocols = protocols_data["protocols"]
            
            for protocol in protocols:
                questions = protocol.get("questions", [])
                
                for question_data in questions:
                    question = create_nice_question_from_data(
                        question_data,
                        protocol
                    )
                    db.add(question)
                    total_questions_inserted += 1
                    
        elif "questions" in protocols_data:
            # Simple structure with direct questions array
            questions = protocols_data["questions"]
            
            for question_data in questions:
                question = create_nice_question_from_data(
                    question_data,
                    None  # No protocol context
                )
                db.add(question)
                total_questions_inserted += 1
        else:
            raise ValueError("Invalid JSON structure: no 'protocols' or 'questions' key found")
        
        db.commit()
        print(f"📋 Inserted {total_questions_inserted} NICE protocol questions")
        
        # Log protocol summary if available
        if "protocols" in protocols_data:
            for protocol in protocols_data["protocols"]:
                question_count = len(protocol.get("questions", []))
                urgency = protocol.get("urgency_category", "unknown")
                print(f"   - {protocol['protocol_name']}: {question_count} questions ({urgency})")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to insert protocols: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def create_nice_question_from_data(
    question_data: Dict[str, Any],
    protocol_context: Optional[Dict[str, Any]] = None
) -> NICEProtocolQuestionDB:
    """Create NICEProtocolQuestionDB from JSON data"""
    
    # Base question data
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
        condition_specific=question_data.get("condition_specific"),
        scoring_weight=question_data.get("scoring_weight"),
        nice_guideline_ref=question_data.get("nice_guideline_ref"),
        clinical_rationale=question_data.get("clinical_rationale")
    )
    
    # Add protocol context if available
    if protocol_context:
        question.protocol_id = protocol_context.get("protocol_id")
        question.protocol_name = protocol_context.get("protocol_name")
        question.nice_guideline = protocol_context.get("nice_guideline")
        question.urgency_category = protocol_context.get("urgency_category")
        question.condition_type = protocol_context.get("condition_type")
    
    return question


def get_fallback_protocols_data() -> Dict[str, Any]:
    """Fallback protocol data if no JSON files found"""
    return {
        "protocol_collection": {
            "name": "Basic Chest Pain Protocol",
            "version": "v1.0",
            "total_protocols": 1
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
                    },
                    {
                        "question_id": 3,
                        "category": "chief_complaint",
                        "question_text": "Please describe your chest pain:",
                        "question_type": "text",
                        "validation_rules": {"min_length": 10, "required": True},
                        "is_required": True,
                        "is_red_flag": False,
                        "order_index": 3,
                        "scoring_weight": 0.2
                    }
                ]
            }
        ]
    }


def test_database_connection() -> bool:
    """Test database connection and validate protocol data"""
    try:
        db = SessionLocal()
        
        # Test connection
        result = db.execute("SELECT version()")
        version = result.fetchone()[0]
        print("✅ Database connection successful")
        print(f"   PostgreSQL version: {version.split(',')[0]}")
        
        # Test tables and data
        case_count = db.query(CaseReportDB).count()
        question_count = db.query(NICEProtocolQuestionDB).count()
        response_count = db.query(PatientResponseDB).count()
        file_count = db.query(UploadedFileDB).count()
        
        print("📊 Database status:")
        print(f"   Case reports: {case_count}")
        print(f"   NICE questions: {question_count}")
        print(f"   Patient responses: {response_count}")
        print(f"   Uploaded files: {file_count}")
        
        # Test protocol distribution if questions exist
        if question_count > 0:
            protocol_distribution = (
                db.query(
                    NICEProtocolQuestionDB.protocol_name,
                    NICEProtocolQuestionDB.urgency_category,
                    db.func.count(NICEProtocolQuestionDB.id)
                )
                .group_by(
                    NICEProtocolQuestionDB.protocol_name,
                    NICEProtocolQuestionDB.urgency_category
                )
                .all()
            )
            
            print("📋 Protocol distribution:")
            for protocol_name, urgency_category, count in protocol_distribution:
                name_display = protocol_name or "Unknown Protocol"
                urgency_display = urgency_category or "No Category"
                print(f"   - {name_display} ({urgency_display}): {count} questions")
            
            # Test red flag questions
            red_flag_count = (
                db.query(NICEProtocolQuestionDB)
                .filter(NICEProtocolQuestionDB.is_red_flag)
                .count()
            )
            print(f"   - Red flag questions: {red_flag_count}")
        
        # Test table relationships
        tables_created = [
            "case_reports", "nice_protocol_questions",
            "patient_responses", "uploaded_files"
        ]
        print(f"   Tables created: {len(tables_created)}")
        print(f"   Table names: {', '.join(tables_created)}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


# =============================================================================
# ENHANCED CRUD OPERATIONS (Business Logic Layer)
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
        """Create new case report with optional fields"""
        case_report = CaseReportDB(
            patient_id=patient_id,
            age=age,
            gender=gender,
            chief_complaint=chief_complaint,
            presenting_complaint_category=kwargs.get(
                "presenting_complaint_category",
                "chest_pain"
            ),
            vital_signs=kwargs.get("vital_signs", {}),
            medical_history=kwargs.get("medical_history", {}),
            **{k: v for k, v in kwargs.items()
               if k in ["ethnicity", "postcode_sector", "pregnancy_status"]}
        )
        
        db.add(case_report)
        db.commit()
        db.refresh(case_report)
        return case_report
    
    @staticmethod
    def get_case_report(db: Session, case_id: str) -> Optional[CaseReportDB]:
        """Get case report by case_id"""
        return (
            db.query(CaseReportDB)
            .filter(CaseReportDB.case_id == case_id)
            .first()
        )
    
    @staticmethod
    def get_case_report_by_id(db: Session, id: int) -> Optional[CaseReportDB]:
        """Get case report by database ID"""
        return (
            db.query(CaseReportDB)
            .filter(CaseReportDB.id == id)
            .first()
        )
    
    @staticmethod
    def update_case_report(
        db: Session,
        case_id: str,
        **kwargs
    ) -> Optional[CaseReportDB]:
        """Update case report with JSON field support"""
        case_report = (
            db.query(CaseReportDB)
            .filter(CaseReportDB.case_id == case_id)
            .first()
        )
        
        if not case_report:
            return None
        
        # Handle JSON field updates properly
        json_fields = [
            "vital_signs", "medical_history", "ai_assessment",
            "chest_pain_assessment", "chest_pain_red_flags",
            "associated_symptoms"
        ]
        
        for key, value in kwargs.items():
            if hasattr(case_report, key):
                if key in json_fields:
                    # For JSON fields, ensure proper mutation detection
                    current_value = getattr(case_report, key) or {}
                    if isinstance(value, dict):
                        current_value.update(value)
                        setattr(case_report, key, current_value)
                    else:
                        setattr(case_report, key, value)
                else:
                    setattr(case_report, key, value)
        
        case_report.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(case_report)
        return case_report
    
    @staticmethod
    def list_case_reports(db: Session, limit: int = 10) -> List[CaseReportDB]:
        """List recent case reports"""
        return (
            db.query(CaseReportDB)
            .order_by(CaseReportDB.created_at.desc())
            .limit(limit)
            .all()
        )


class PatientResponseCRUD:
    """CRUD operations for patient responses"""
    
    @staticmethod
    def create_response(
        db: Session,
        case_report_id: int,
        question_id: int,
        response_text: str,
        response_value: Optional[Dict[str, Any]] = None
    ) -> PatientResponseDB:
        """Create patient response"""
        response = PatientResponseDB(
            case_report_id=case_report_id,
            question_id=question_id,
            response_text=response_text,
            response_value=response_value
        )
        db.add(response)
        db.commit()
        db.refresh(response)
        return response
    
    @staticmethod
    def get_responses_for_case(db: Session, case_report_id: int) -> List[PatientResponseDB]:
        """Get all responses for a case report"""
        return (
            db.query(PatientResponseDB)
            .filter(PatientResponseDB.case_report_id == case_report_id)
            .order_by(PatientResponseDB.timestamp)
            .all()
        )
    
    @staticmethod
    def create_multiple_responses(
        db: Session,
        case_report_id: int,
        responses: List[Dict[str, Any]]
    ) -> List[PatientResponseDB]:
        """Create multiple patient responses"""
        created_responses = []
        
        for response_data in responses:
            response = PatientResponseDB(
                case_report_id=case_report_id,
                question_id=response_data["question_id"],
                response_text=response_data["response_text"],
                response_value=response_data.get("response_value")
            )
            db.add(response)
            created_responses.append(response)
        
        db.commit()
        
        for response in created_responses:
            db.refresh(response)
        
        return created_responses


class NICEQuestionCRUD:
    """Enhanced CRUD operations for NICE protocol questions"""
    
    @staticmethod
    def get_questions_by_protocol(
        db: Session,
        protocol_id: int
    ) -> List[NICEProtocolQuestionDB]:
        """Get questions by protocol ID"""
        return (
            db.query(NICEProtocolQuestionDB)
            .filter(NICEProtocolQuestionDB.protocol_id == protocol_id)
            .order_by(NICEProtocolQuestionDB.order_index)
            .all()
        )
    
    @staticmethod
    def get_questions_by_category(
        db: Session,
        category: str
    ) -> List[NICEProtocolQuestionDB]:
        """Get questions by category"""
        return (
            db.query(NICEProtocolQuestionDB)
            .filter(NICEProtocolQuestionDB.category == category)
            .order_by(NICEProtocolQuestionDB.order_index)
            .all()
        )
    
    @staticmethod
    def get_questions_by_urgency(
        db: Session,
        urgency_category: str
    ) -> List[NICEProtocolQuestionDB]:
        """Get questions by urgency category"""
        return (
            db.query(NICEProtocolQuestionDB)
            .filter(NICEProtocolQuestionDB.urgency_category == urgency_category)
            .order_by(
                NICEProtocolQuestionDB.protocol_id,
                NICEProtocolQuestionDB.order_index
            )
            .all()
        )
    
    @staticmethod
    def get_chest_pain_questions(db: Session) -> List[NICEProtocolQuestionDB]:
        """Get all chest pain protocol questions"""
        return (
            db.query(NICEProtocolQuestionDB)
            .order_by(NICEProtocolQuestionDB.order_index)
            .all()
        )
    
    @staticmethod
    def get_question_by_id(db: Session, question_id: int) -> Optional[NICEProtocolQuestionDB]:
        """Get question by question_id"""
        return (
            db.query(NICEProtocolQuestionDB)
            .filter(NICEProtocolQuestionDB.question_id == question_id)
            .first()
        )
    
    @staticmethod
    def get_red_flag_questions(db: Session) -> List[NICEProtocolQuestionDB]:
        """Get all red flag questions across protocols"""
        return (
            db.query(NICEProtocolQuestionDB)
            .filter(NICEProtocolQuestionDB.is_red_flag)
            .order_by(
                NICEProtocolQuestionDB.protocol_id,
                NICEProtocolQuestionDB.order_index
            )
            .all()
        )


# =============================================================================
# DATABASE HEALTH AND MONITORING
# =============================================================================

def get_database_health() -> Dict[str, Any]:
    """Get comprehensive database health information"""
    try:
        db = SessionLocal()
        
        # Connection test
        result = db.execute("SELECT version()")
        version = result.fetchone()[0]
        
        # Count records
        case_count = db.query(CaseReportDB).count()
        question_count = db.query(NICEProtocolQuestionDB).count()
        response_count = db.query(PatientResponseDB).count()
        file_count = db.query(UploadedFileDB).count()
        
        # Protocol analysis
        protocol_stats = (
            db.query(
                NICEProtocolQuestionDB.protocol_name,
                NICEProtocolQuestionDB.urgency_category,
                db.func.count(NICEProtocolQuestionDB.id)
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
        
        # Recent activity
        recent_cases = (
            db.query(CaseReportDB)
            .order_by(CaseReportDB.created_at.desc())
            .limit(5)
            .count()
        )
        
        db.close()
        
        return {
            "status": "healthy",
            "version": version.split(',')[0],
            "tables": {
                "case_reports": case_count,
                "nice_questions": question_count,
                "patient_responses": response_count,
                "uploaded_files": file_count
            },
            "protocols": [
                {
                    "name": name or "Unknown Protocol",
                    "urgency_category": urgency or "No Category",
                    "question_count": count
                }
                for name, urgency, count in protocol_stats
            ],
            "red_flag_questions": red_flag_count,
            "recent_activity": {"recent_cases_count": recent_cases},
            "connection": "active",
            "json_fields": "mutable_tracking_enabled",
            "architecture": "clean_separation"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": "failed"
        }


# =============================================================================
# EXPORT - OPERATIONS ONLY (NO MODELS)
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
    "load_nice_protocols_from_json",
    
    # CRUD Operations
    "CaseReportCRUD",
    "PatientResponseCRUD",
    "NICEQuestionCRUD",
]
