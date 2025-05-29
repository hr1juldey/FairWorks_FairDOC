"""
V1 NICE Protocols API
Complete endpoint suite for NHS NICE protocol access and question management
Integrates with our 10 comprehensive protocols and ML-ready scoring system
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging

# Import our infrastructure
from ..data.database import get_db, NICEQuestionCRUD
from ..datamodels.medical_model import (
    NICEProtocolQuestion, TriageCoordinate,
    PatientResponse, CaseReport
)
from ..datamodels.sqlalchemy_models import NICEProtocolQuestionDB
from ..core.security import get_current_active_user
from ..datamodels.auth_models import UserDB

logger = logging.getLogger(__name__)

# Initialize router
protocols_router = APIRouter(prefix="/protocols", tags=["NICE Protocols"])

# =============================================================================
# NICE PROTOCOL DISCOVERY ENDPOINTS
# =============================================================================

@protocols_router.get("/", response_model=Dict[str, Any])
async def list_all_protocols(
    include_stats: bool = Query(True, description="Include protocol statistics"),
    urgency_filter: Optional[str] = Query(None, description="Filter by urgency category"),
    db: Session = Depends(get_db)
):
    """
    List all available NICE protocols with comprehensive metadata
    Returns overview of all 10 protocols loaded from our JSON data
    """
    try:
        # Get unique protocols with statistics
        from sqlalchemy import func
        
        base_query = (
            db.query(
                NICEProtocolQuestionDB.protocol_id,
                NICEProtocolQuestionDB.protocol_name,
                NICEProtocolQuestionDB.nice_guideline,
                NICEProtocolQuestionDB.urgency_category,
                NICEProtocolQuestionDB.condition_type,
                func.count(NICEProtocolQuestionDB.id).label('total_questions'),
                func.sum(
                    func.case(
                        (NICEProtocolQuestionDB.is_red_flag, 1),
                        else_=0
                    )
                ).label('red_flag_count'),
                func.sum(
                    func.case(
                        (NICEProtocolQuestionDB.is_required, 1),
                        else_=0
                    )
                ).label('required_questions')
            )
            .group_by(
                NICEProtocolQuestionDB.protocol_id,
                NICEProtocolQuestionDB.protocol_name,
                NICEProtocolQuestionDB.nice_guideline,
                NICEProtocolQuestionDB.urgency_category,
                NICEProtocolQuestionDB.condition_type
            )
        )
        
        # Apply urgency filter if specified
        if urgency_filter:
            base_query = base_query.filter(
                NICEProtocolQuestionDB.urgency_category == urgency_filter
            )
        
        protocols_data = base_query.order_by(NICEProtocolQuestionDB.protocol_id).all()
        
        if not protocols_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No NICE protocols found. Please check database initialization."
            )
        
        # Format response
        protocols = []
        for p in protocols_data:
            protocol = {
                "protocol_id": p.protocol_id,
                "protocol_name": p.protocol_name,
                "nice_guideline": p.nice_guideline,
                "urgency_category": p.urgency_category,
                "condition_type": p.condition_type,
                "endpoints": {
                    "questions": f"/api/v1/protocols/{p.protocol_id}/questions",
                    "start_assessment": f"/api/v1/protocols/{p.protocol_id}/start",
                    "red_flags": f"/api/v1/protocols/{p.protocol_id}/red-flags"
                }
            }
            
            if include_stats:
                protocol.update({
                    "statistics": {
                        "total_questions": p.total_questions,
                        "red_flag_questions": p.red_flag_count,
                        "required_questions": p.required_questions,
                        "completion_time_estimate": f"{p.total_questions * 30} seconds"
                    }
                })
            
            protocols.append(protocol)
        
        # Summary statistics
        summary = {
            "total_protocols": len(protocols),
            "total_questions": sum(p["statistics"]["total_questions"] for p in protocols if include_stats),
            "total_red_flags": sum(p["statistics"]["red_flag_questions"] for p in protocols if include_stats),
            "urgency_categories": list(set(p["urgency_category"] for p in protocols)),
            "condition_types": list(set(p["condition_type"] for p in protocols))
        }
        
        return {
            "protocols": protocols,
            "summary": summary,
            "api_info": {
                "version": "v1",
                "total_endpoints": len(protocols) * 3,
                "supported_features": [
                    "question_flow_management",
                    "red_flag_detection",
                    "ml_scoring_coordinates",
                    "real_time_assessment"
                ]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list protocols: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve NICE protocols"
        )

@protocols_router.get("/{protocol_id}/questions", response_model=Dict[str, Any])
async def get_protocol_questions(
    protocol_id: int,
    include_metadata: bool = Query(True, description="Include scoring metadata"),
    db: Session = Depends(get_db)
):
    """
    Get all questions for a specific NICE protocol
    Returns structured questions ready for chat interface integration
    """
    try:
        # Get questions using our enhanced CRUD
        questions = NICEQuestionCRUD.get_questions_by_protocol(db, protocol_id)
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protocol {protocol_id} not found or contains no questions"
            )
        
        # Get protocol metadata from first question
        first_question = questions[0]
        protocol_info = {
            "protocol_id": protocol_id,
            "protocol_name": first_question.protocol_name,
            "nice_guideline": first_question.nice_guideline,
            "urgency_category": first_question.urgency_category,
            "condition_type": first_question.condition_type,
            "total_questions": len(questions),
            "required_questions": sum(1 for q in questions if q.is_required),
            "red_flag_questions": sum(1 for q in questions if q.is_red_flag)
        }
        
        # Format questions for frontend consumption
        formatted_questions = []
        for q in questions:
            question_data = {
                "question_id": q.question_id,
                "category": q.category,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "options": q.options,
                "validation_rules": q.validation_rules,
                "is_required": q.is_required,
                "is_red_flag": q.is_red_flag,
                "order_index": q.order_index,
                "ui_config": {
                    "input_type": q.question_type,
                    "placeholder": f"Please answer: {q.question_text}",
                    "error_message": "This question is required" if q.is_required else None,
                    "warning_message": "⚠️ This is a red flag symptom" if q.is_red_flag else None
                }
            }
            
            if include_metadata:
                question_data.update({
                    "scoring_metadata": {
                        "weight": q.scoring_weight,
                        "condition_specific": q.condition_specific,
                        "nice_guideline_ref": q.nice_guideline_ref,
                        "clinical_rationale": q.clinical_rationale
                    }
                })
            
            formatted_questions.append(question_data)
        
        return {
            "protocol": protocol_info,
            "questions": formatted_questions,
            "flow_control": {
                "start_endpoint": f"/api/v1/protocols/{protocol_id}/start",
                "submit_endpoint": f"/api/v1/protocols/{protocol_id}/submit",
                "next_question_logic": "sequential_with_branching",
                "completion_criteria": "all_required_questions_answered"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get protocol {protocol_id} questions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve protocol questions \n {e}"
        )

@protocols_router.get("/{protocol_id}/red-flags", response_model=Dict[str, Any])
async def get_red_flag_questions(
    protocol_id: int,
    db: Session = Depends(get_db)
):
    """
    Get red flag questions for emergency detection
    Critical for immediate triage decision making
    """
    try:
        # Get all questions for protocol
        all_questions = NICEQuestionCRUD.get_questions_by_protocol(db, protocol_id)
        
        if not all_questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protocol {protocol_id} not found"
            )
        
        # Filter red flag questions
        red_flag_questions = [q for q in all_questions if q.is_red_flag]
        
        if not red_flag_questions:
            return {
                "protocol_id": protocol_id,
                "protocol_name": all_questions[0].protocol_name,
                "red_flag_questions": [],
                "emergency_guidance": {
                    "message": "This protocol has no specific red flag questions",
                    "action": "Continue with standard assessment"
                }
            }
        
        # Format red flag questions
        red_flags = []
        for q in red_flag_questions:
            red_flags.append({
                "question_id": q.question_id,
                "question_text": q.question_text,
                "category": q.category,
                "clinical_rationale": q.clinical_rationale,
                "urgency_action": "immediate_assessment" if q.urgency_category == "999_emergency" else "urgent_assessment",
                "scoring_impact": q.scoring_weight
            })
        
        return {
            "protocol_id": protocol_id,
            "protocol_name": all_questions[0].protocol_name,
            "urgency_category": all_questions[0].urgency_category,
            "red_flag_questions": red_flags,
            "emergency_guidance": {
                "total_red_flags": len(red_flags),
                "escalation_threshold": 1,  # Any red flag triggers escalation
                "emergency_contact": "999" if all_questions[0].urgency_category == "999_emergency" else "111",
                "message": "Positive red flag responses require immediate clinical assessment"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get red flags for protocol {protocol_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve red flag questions"
        )

# =============================================================================
# PROTOCOL FILTERING AND SEARCH
# =============================================================================

@protocols_router.get("/by-urgency/{urgency_category}", response_model=Dict[str, Any])
async def get_protocols_by_urgency(
    urgency_category: str,
    db: Session = Depends(get_db)
):
    """
    Get protocols filtered by urgency category
    Categories: 111_standard, 111_urgent, 999_emergency
    """
    try:
        # Validate urgency category
        valid_categories = ["111_standard", "111_urgent", "999_emergency"]
        if urgency_category not in valid_categories:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid urgency category. Must be one of: {valid_categories}"
            )
        
        # Get questions for this urgency category
        questions = NICEQuestionCRUD.get_questions_by_urgency(db, urgency_category)
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No protocols found for urgency category: {urgency_category}"
            )
        
        # Group by protocol
        protocols = {}
        for q in questions:
            pid = q.protocol_id
            if pid not in protocols:
                protocols[pid] = {
                    "protocol_id": pid,
                    "protocol_name": q.protocol_name,
                    "nice_guideline": q.nice_guideline,
                    "condition_type": q.condition_type,
                    "urgency_category": q.urgency_category,
                    "questions": [],
                    "red_flag_count": 0
                }
            
            protocols[pid]["questions"].append({
                "question_id": q.question_id,
                "category": q.category,
                "question_text": q.question_text,
                "is_red_flag": q.is_red_flag
            })
            
            if q.is_red_flag:
                protocols[pid]["red_flag_count"] += 1
        
        return {
            "urgency_category": urgency_category,
            "category_description": {
                "111_standard": "Standard NHS 111 triage protocols",
                "111_urgent": "Urgent NHS 111 assessment required",
                "999_emergency": "Emergency protocols requiring immediate response"
            }.get(urgency_category, "Unknown category"),
            "protocols": list(protocols.values()),
            "summary": {
                "total_protocols": len(protocols),
                "total_questions": sum(len(p["questions"]) for p in protocols.values()),
                "total_red_flags": sum(p["red_flag_count"] for p in protocols.values())
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get protocols by urgency {urgency_category}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve protocols by urgency"
        )

@protocols_router.get("/search", response_model=Dict[str, Any])
async def search_protocols(
    query: str = Query(..., min_length=3, description="Search term"),
    search_type: str = Query("all", description="Search in: all, questions, conditions"),
    db: Session = Depends(get_db)
):
    """
    Search across protocols and questions
    Enables dynamic protocol discovery based on symptoms/conditions
    """
    try:
        from sqlalchemy import or_, func
        
        # Build search query based on type
        base_query = db.query(NICEProtocolQuestionDB)
        
        if search_type == "questions":
            search_query = base_query.filter(
                NICEProtocolQuestionDB.question_text.ilike(f"%{query}%")
            )
        elif search_type == "conditions":
            search_query = base_query.filter(
                or_(
                    NICEProtocolQuestionDB.condition_type.ilike(f"%{query}%"),
                    NICEProtocolQuestionDB.protocol_name.ilike(f"%{query}%")
                )
            )
        else:  # search_type == "all"
            search_query = base_query.filter(
                or_(
                    NICEProtocolQuestionDB.question_text.ilike(f"%{query}%"),
                    NICEProtocolQuestionDB.protocol_name.ilike(f"%{query}%"),
                    NICEProtocolQuestionDB.condition_type.ilike(f"%{query}%"),
                    NICEProtocolQuestionDB.category.ilike(f"%{query}%"),
                    NICEProtocolQuestionDB.clinical_rationale.ilike(f"%{query}%")
                )
            )
        
        results = search_query.order_by(
            NICEProtocolQuestionDB.protocol_id,
            NICEProtocolQuestionDB.order_index
        ).all()
        
        if not results:
            return {
                "query": query,
                "search_type": search_type,
                "results": [],
                "summary": {
                    "total_matches": 0,
                    "protocols_matched": 0,
                    "suggestions": [
                        "Try broader search terms",
                        "Check spelling",
                        "Use medical terminology"
                    ]
                }
            }
        
        # Group results by protocol
        protocols_matched = {}
        for result in results:
            pid = result.protocol_id
            if pid not in protocols_matched:
                protocols_matched[pid] = {
                    "protocol_id": pid,
                    "protocol_name": result.protocol_name,
                    "condition_type": result.condition_type,
                    "urgency_category": result.urgency_category,
                    "matched_questions": []
                }
            
            protocols_matched[pid]["matched_questions"].append({
                "question_id": result.question_id,
                "question_text": result.question_text,
                "category": result.category,
                "is_red_flag": result.is_red_flag
            })
        
        return {
            "query": query,
            "search_type": search_type,
            "results": list(protocols_matched.values()),
            "summary": {
                "total_matches": len(results),
                "protocols_matched": len(protocols_matched),
                "query_processed": query.lower().strip()
            }
        }
        
    except Exception as e:
        logger.error(f"Protocol search failed for query '{query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Protocol search service error"
        )

# =============================================================================
# PROTOCOL ASSESSMENT WORKFLOW
# =============================================================================

@protocols_router.post("/{protocol_id}/start", response_model=Dict[str, Any])
async def start_protocol_assessment(
    protocol_id: int,
    current_user: UserDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Initialize protocol assessment workflow
    Creates case report and returns first question
    """
    try:
        # Verify protocol exists
        questions = NICEQuestionCRUD.get_questions_by_protocol(db, protocol_id)
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protocol {protocol_id} not found"
            )
        
        # Create new case report for this assessment
        from ..data.database import CaseReportCRUD
        
        case_report = CaseReportCRUD.create_case_report(
            db=db,
            patient_id=current_user.user_id,  # Link to authenticated user
            age=0,  # Will be updated from first demographics question
            gender="unknown",  # Will be updated from demographics
            chief_complaint=f"Assessment using {questions[0].protocol_name}",
            presenting_complaint_category=questions[0].condition_type
        )
        
        # Get first question
        first_question = min(questions, key=lambda q: q.order_index)
        
        return {
            "assessment_started": True,
            "case_id": case_report.case_id,
            "protocol": {
                "protocol_id": protocol_id,
                "protocol_name": first_question.protocol_name,
                "urgency_category": first_question.urgency_category,
                "total_questions": len(questions)
            },
            "first_question": {
                "question_id": first_question.question_id,
                "category": first_question.category,
                "question_text": first_question.question_text,
                "question_type": first_question.question_type,
                "options": first_question.options,
                "is_required": first_question.is_required,
                "is_red_flag": first_question.is_red_flag,
                "order_index": first_question.order_index
            },
            "workflow": {
                "total_questions": len(questions),
                "current_question": 1,
                "progress_percentage": round((1 / len(questions)) * 100, 1),
                "submit_endpoint": f"/api/v1/protocols/{protocol_id}/submit",
                "next_question_endpoint": f"/api/v1/protocols/{protocol_id}/next"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start protocol {protocol_id} assessment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start protocol assessment"
        )


# Export router
__all__ = ["protocols_router"]
