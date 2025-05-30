"""
V1 Case Management API
Complete CRUD operations for medical case reports
Integrates with NICE protocols and ML scoring system
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

# Import our infrastructure
from ..data.database import get_db, CaseReportCRUD, PatientResponseCRUD
from ..datamodels.medical_model import CaseReport, PatientResponse, TriageCoordinate
from ..datamodels.sqlalchemy_models import CaseReportDB, PatientResponseDB
from ..core.security import get_current_active_user, require_permission
from ..datamodels.auth_models import UserDB

logger = logging.getLogger(__name__)

# Initialize router
cases_router = APIRouter(prefix="/cases", tags=["Case Management"])

# =============================================================================
# CASE REPORT CRUD OPERATIONS
# =============================================================================

@cases_router.post("/", response_model=Dict[str, Any])
async def create_case_report(
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    chief_complaint: Optional[str] = None,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create new case report
    Links to authenticated user for patient tracking
    """
    try:
        # Use user information if patient details not provided
        case_report = CaseReportCRUD.create_case_report(
            db=db,
            patient_id=current_user.user_id,
            age=patient_age or 0,
            gender=patient_gender or "unknown",
            chief_complaint=chief_complaint or "Medical triage assessment",
            presenting_complaint_category="triage_assessment"
        )
        
        logger.info(f"Created case report {case_report.case_id} for user {current_user.username}")
        
        return {
            "case_id": case_report.case_id,
            "patient_id": case_report.patient_id,
            "status": case_report.status,
            "created_at": case_report.created_at.isoformat(),
            "message": "Case report created successfully",
            "next_steps": [
                "Begin NICE protocol assessment",
                "Submit patient responses",
                "Upload supporting documents if available"
            ],
            "api_endpoints": {
                "update": f"/api/v1/cases/{case_report.case_id}",
                "responses": f"/api/v1/cases/{case_report.case_id}/responses",
                "files": f"/api/v1/cases/{case_report.case_id}/files"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create case report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create case report"
        )

@cases_router.get("/{case_id}", response_model=Dict[str, Any])
async def get_case_report(
    case_id: str,
    include_responses: bool = Query(True, description="Include patient responses"),
    include_ai_assessment: bool = Query(True, description="Include AI assessment"),
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed case report with optional related data
    Includes patient responses and AI assessment if available
    """
    try:
        # Get case report
        case_report = CaseReportCRUD.get_case_report(db, case_id)
        
        if not case_report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case report {case_id} not found"
            )
        
        # Check access permissions (users can only see their own cases unless medical staff)
        medical_roles = ["doctor", "admin", "developer"]
        if (case_report.patient_id != current_user.user_id and
            current_user.role not in medical_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this case report"
            )
        
        # Base case information
        case_data = {
            "case_id": case_report.case_id,
            "patient_id": case_report.patient_id,
            "status": case_report.status,
            "demographics": {
                "age": case_report.age,
                "gender": case_report.gender,
                "pregnancy_status": case_report.pregnancy_status,
                "ethnicity": case_report.ethnicity,
                "postcode_sector": case_report.postcode_sector
            },
            "clinical_data": {
                "chief_complaint": case_report.chief_complaint,
                "presenting_complaint_category": case_report.presenting_complaint_category,
                "vital_signs": case_report.vital_signs or {},
                "medical_history": case_report.medical_history or {},
                "chest_pain_assessment": case_report.chest_pain_assessment or {},
                "associated_symptoms": case_report.associated_symptoms or {}
            },
            "timestamps": {
                "created_at": case_report.created_at.isoformat(),
                "updated_at": case_report.updated_at.isoformat(),
                "completed_at": case_report.completed_at.isoformat() if case_report.completed_at else None
            },
            "scores": {
                "urgency_score": case_report.urgency_score,
                "importance_score": case_report.importance_score
            },
            "pdf_report_url": case_report.pdf_report_url
        }
        
        # Include patient responses if requested
        if include_responses:
            responses = PatientResponseCRUD.get_responses_for_case(db, case_report.id)
            case_data["responses"] = [
                {
                    "question_id": r.question_id,
                    "response_text": r.response_text,
                    "response_value": r.response_value,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in responses
            ]
            case_data["response_count"] = len(responses)
        
        # Include AI assessment if requested and available
        if include_ai_assessment and case_report.ai_assessment:
            case_data["ai_assessment"] = case_report.ai_assessment
        
        return case_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get case report {case_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve case report"
        )

@cases_router.put("/{case_id}", response_model=Dict[str, Any])
async def update_case_report(
    case_id: str,
    update_data: Dict[str, Any],
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update case report with new information
    Supports updating demographics, symptoms, and clinical data
    """
    try:
        # Verify case exists and user has access
        case_report = CaseReportCRUD.get_case_report(db, case_id)
        
        if not case_report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case report {case_id} not found"
            )
        
        # Check access permissions
        medical_roles = ["doctor", "admin", "developer"]
        if (case_report.patient_id != current_user.user_id and
            current_user.role not in medical_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to update this case report"
            )
        
        # Update case report
        updated_case = CaseReportCRUD.update_case_report(
            db=db,
            case_id=case_id,
            **update_data
        )
        
        if not updated_case:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update case report"
            )
        
        logger.info(f"Updated case report {case_id} by user {current_user.username}")
        
        return {
            "case_id": updated_case.case_id,
            "status": updated_case.status,
            "updated_at": updated_case.updated_at.isoformat(),
            "message": "Case report updated successfully",
            "updated_fields": list(update_data.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update case report {case_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update case report"
        )

# =============================================================================
# PATIENT RESPONSES MANAGEMENT
# =============================================================================

@cases_router.post("/{case_id}/responses", response_model=Dict[str, Any])
async def submit_patient_responses(
    case_id: str,
    responses: List[Dict[str, Any]],
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Submit patient responses to NICE protocol questions
    Supports both single and batch response submission
    """
    try:
        # Verify case exists and user has access
        case_report = CaseReportCRUD.get_case_report(db, case_id)
        
        if not case_report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case report {case_id} not found"
            )
        
        # Check access permissions
        if (case_report.patient_id != current_user.user_id and
            current_user.role not in ["doctor", "admin", "developer"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to submit responses for this case"
            )
        
        # Validate response format
        if not responses or not isinstance(responses, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Responses must be a non-empty list"
            )
        
        # Create responses using batch operation
        created_responses = PatientResponseCRUD.create_multiple_responses(
            db=db,
            case_report_id=case_report.id,
            responses=responses
        )
        
        # Update case status if this was the final submission
        if len(created_responses) > 0:
            # Simple logic: if we have responses, update status to "in_progress"
            CaseReportCRUD.update_case_report(
                db=db,
                case_id=case_id,
                status="in_progress"
            )
        
        logger.info(f"Submitted {len(created_responses)} responses for case {case_id}")
        
        return {
            "case_id": case_id,
            "responses_submitted": len(created_responses),
            "status": "responses_saved",
            "message": f"Successfully submitted {len(created_responses)} responses",
            "next_steps": [
                "Continue with remaining questions",
                "Upload supporting documents if needed",
                "Complete assessment for AI analysis"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit responses for case {case_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit patient responses"
        )

@cases_router.get("/{case_id}/responses", response_model=List[Dict[str, Any]])
async def get_case_responses(
    case_id: str,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all responses for a case report
    Returns responses with question context
    """
    try:
        # Verify case exists and user has access
        case_report = CaseReportCRUD.get_case_report(db, case_id)
        
        if not case_report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case report {case_id} not found"
            )
        
        # Check access permissions
        medical_roles = ["doctor", "admin", "developer"]
        if (case_report.patient_id != current_user.user_id and
            current_user.role not in medical_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to view responses for this case"
            )
        
        # Get responses
        responses = PatientResponseCRUD.get_responses_for_case(db, case_report.id)
        
        # Format responses with question context
        formatted_responses = []
        for response in responses:
            # Get question details
            from ..data.database import NICEQuestionCRUD
            question = NICEQuestionCRUD.get_question_by_id(db, response.question_id)
            
            response_data = {
                "response_id": response.id,
                "question_id": response.question_id,
                "response_text": response.response_text,
                "response_value": response.response_value,
                "timestamp": response.timestamp.isoformat()
            }
            
            if question:
                response_data["question_context"] = {
                    "question_text": question.question_text,
                    "category": question.category,
                    "is_red_flag": question.is_red_flag,
                    "scoring_weight": question.scoring_weight
                }
            
            formatted_responses.append(response_data)
        
        return formatted_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get responses for case {case_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve case responses"
        )

# =============================================================================
# CASE PROCESSING AND AI ANALYSIS
# =============================================================================

@cases_router.post("/{case_id}/process", response_model=Dict[str, Any])
async def process_case_report(
    case_id: str,
    background_tasks: BackgroundTasks,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Trigger AI processing of complete case report
    Queues background processing for ML analysis and scoring
    """
    try:
        # Verify case exists and user has access
        case_report = CaseReportCRUD.get_case_report(db, case_id)
        
        if not case_report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case report {case_id} not found"
            )
        
        # Check access permissions
        if (case_report.patient_id != current_user.user_id and
            current_user.role not in ["doctor", "admin", "developer"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to process this case"
            )
        
        # Check if case has sufficient data for processing
        responses = PatientResponseCRUD.get_responses_for_case(db, case_report.id)
        
        if len(responses) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Case has no responses. Please submit patient responses first."
            )
        
        # Update case status to processing
        CaseReportCRUD.update_case_report(
            db=db,
            case_id=case_id,
            status="processing",
            ai_assessment={"processing_started": lambda: datetime.now(timezone.utc)().isoformat()}
        )
        
        # Queue background processing (placeholder for Celery task)
        # In a full implementation, this would trigger:
        # process_case_report_task.delay(case_report.id)
        
        # For MVP, add a simple background task
        background_tasks.add_task(
            simulate_case_processing,
            case_report.id,
            len(responses)
        )
        
        logger.info(f"Started processing case {case_id} with {len(responses)} responses")
        
        return {
            "case_id": case_id,
            "status": "processing_started",
            "processing_info": {
                "responses_count": len(responses),
                "estimated_completion": "2-3 minutes",
                "processing_steps": [
                    "Analyzing patient responses",
                    "Calculating urgency/importance scores",
                    "Generating AI recommendations",
                    "Creating PDF report"
                ]
            },
            "check_status_endpoint": f"/api/v1/cases/{case_id}",
            "message": "Case processing started. Check status for updates."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process case {case_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start case processing"
        )

async def simulate_case_processing(case_report_id: int, response_count: int):
    """
    Simulate case processing for MVP
    In production, this would be replaced by Celery tasks
    """
    import asyncio
    
    # Simulate processing delay
    await asyncio.sleep(5)
    
    try:
        # Get database session
        from ..data.database import SessionLocal
        db = SessionLocal()
        
        # Simple rule-based scoring for MVP
        urgency_score = min(0.8, response_count * 0.1)  # Simple scoring
        importance_score = min(0.7, response_count * 0.08)
        
        # Create mock AI assessment
        ai_assessment = {
            "processing_completed": lambda: datetime.now(timezone.utc)().isoformat(),
            "urgency_score": urgency_score,
            "importance_score": importance_score,
            "predicted_conditions": ["Chest pain assessment", "Cardiovascular evaluation"],
            "recommended_actions": ["Continue monitoring", "Contact GP if symptoms persist"],
            "reasoning": f"Assessment based on {response_count} patient responses using rule-based scoring",
            "confidence_level": 0.75,
            "coordinates": {
                "x": urgency_score,
                "y": importance_score,
                "quadrant": "medium_priority" if urgency_score < 0.5 else "high_priority"
            }
        }
        
        # Update case report
        case_report = db.query(CaseReportDB).filter(CaseReportDB.id == case_report_id).first()
        if case_report:
            case_report.status = "completed"
            case_report.urgency_score = urgency_score
            case_report.importance_score = importance_score
            case_report.ai_assessment = ai_assessment
            case_report.completed_at = lambda: datetime.now(timezone.utc)()
            
            db.commit()
            
        db.close()
        logger.info(f"Completed processing case {case_report_id}")
        
    except Exception as e:
        logger.error(f"Case processing simulation failed: {e}")

# =============================================================================
# CASE LISTING AND MANAGEMENT
# =============================================================================

@cases_router.get("/", response_model=Dict[str, Any])
async def list_user_cases(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[str] = Query(None),
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List cases for current user with pagination
    Medical staff can see all cases, patients see only their own
    """
    try:
        # Determine access scope based on user role
        medical_roles = ["doctor", "admin", "developer"]
        
        if current_user.role in medical_roles:
            # Medical staff can see all cases
            cases = CaseReportCRUD.list_case_reports(
                db=db,
                limit=limit,
                offset=offset,
                status_filter=status_filter
            )
        else:
            # Patients see only their own cases
            # Note: We'd need to modify CaseReportCRUD to support patient_id filtering
            cases = db.query(CaseReportDB).filter(
                CaseReportDB.patient_id == current_user.user_id
            )
            
            if status_filter:
                cases = cases.filter(CaseReportDB.status == status_filter)
            
            cases = cases.order_by(CaseReportDB.created_at.desc()).offset(offset).limit(limit).all()
        
        # Format cases for response
        formatted_cases = []
        for case in cases:
            formatted_cases.append({
                "case_id": case.case_id,
                "patient_id": case.patient_id,
                "status": case.status,
                "chief_complaint": case.chief_complaint,
                "urgency_score": case.urgency_score,
                "importance_score": case.importance_score,
                "created_at": case.created_at.isoformat(),
                "updated_at": case.updated_at.isoformat(),
                "completed_at": case.completed_at.isoformat() if case.completed_at else None
            })
        
        return {
            "cases": formatted_cases,
            "pagination": {
                "total": len(formatted_cases),
                "limit": limit,
                "offset": offset,
                "has_more": len(formatted_cases) == limit
            },
            "filters": {
                "status_filter": status_filter,
                "available_statuses": ["created", "in_progress", "processing", "completed", "archived"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list cases for user {current_user.username}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve case list"
        )


# Export router
__all__ = ["cases_router"]
