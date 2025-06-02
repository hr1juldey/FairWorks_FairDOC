# msp/state/state_manager.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

# Import all state modules
from .core_states import AppState, ChatMessage, Role
from .clinical_states import ClinicalAnalysisState, BiasMonitoringState
from .patient_states import PatientDataState
from .document_states import UploadedFileState
from .ui_states import ReportTabsState

# Import database manager
from .database_manager import (
    save_message_to_db, 
    load_chat_history_from_db,
    save_comprehensive_session_data_to_db as db_save_session,
    load_session_data_from_db,
    save_file_to_db
)

# Import serializers
from .state_serializers import (
    clinical_analysis_to_json,
    patient_data_to_json,
    patient_data_to_dict,
    bias_monitoring_to_json,
    load_clinical_analysis_from_dict,
    load_patient_data_from_dict,
    load_bias_monitoring_from_dict
)

# Re-export all state classes so other files can import them from here
__all__ = [
    'AppState', 'ChatMessage', 'Role',
    'ClinicalAnalysisState', 'BiasMonitoringState',
    'PatientDataState', 'UploadedFileState', 'ReportTabsState',
    'initialize_app_state', 'add_chat_message', 'update_clinical_analysis',
    'add_emergency_alert', 'update_patient_data', 'add_uploaded_file',
    'save_clinical_report', 'save_comprehensive_session_data_to_db',
    'update_report_state_in_app_and_db'
]

def initialize_app_state(session_id: str = "default_session"):
    """Initialize comprehensive application state from database."""
    state = me.state(AppState)
    state.session_id = session_id
    
    # Load chat history
    if not state.chat_history:
        state.chat_history = load_chat_history_from_db(session_id)
        if not state.chat_history:
            add_chat_message("assistant", "Hello! I'm your NHS Digital Triage assistant. How can I help you today?", session_id)
    
    # Load session data from database
    session_data = load_session_data_from_db(session_id)
    if session_data:
        # Basic session data
        state.current_input = session_data.get("current_input", "")
        state.is_bot_typing = bool(session_data.get("is_bot_typing", 0))
        state.report_data_json = session_data.get("report_data_json")
        state.report_is_generating = bool(session_data.get("report_is_generating", 0))
        state.report_generation_progress = session_data.get("report_generation_progress", "")
        state.report_generation_complete = bool(session_data.get("report_generation_complete", 0))
        state.report_error_message = session_data.get("report_error_message")
        
        # Load clinical analysis data
        clinical_json = session_data.get("clinical_analysis_json")
        if clinical_json:
            try:
                clinical_data = json.loads(clinical_json)
                load_clinical_analysis_from_dict(state.clinical_analysis, clinical_data)
            except json.JSONDecodeError:
                print("Warning: Could not parse clinical analysis JSON")
        
        # Load EHR data
        state.ehr_data_json = session_data.get("ehr_data_json")
        
        # Load patient data
        patient_json = session_data.get("patient_data_json")
        if patient_json:
            try:
                patient_data = json.loads(patient_json)
                load_patient_data_from_dict(state.patient_data, patient_data)
            except json.JSONDecodeError:
                print("Warning: Could not parse patient data JSON")
        
        # Load uploaded files
        files_json = session_data.get("uploaded_documents_json")
        if files_json:
            try:
                files_data = json.loads(files_json)
                state.uploaded_files = [UploadedFileState(**file_data) for file_data in files_data]
            except json.JSONDecodeError:
                print("Warning: Could not parse uploaded documents JSON")
        
        # Load bias monitoring data
        bias_json = session_data.get("bias_monitoring_json")
        if bias_json:
            try:
                bias_data = json.loads(bias_json)
                load_bias_monitoring_from_dict(state.bias_monitoring, bias_data)
            except json.JSONDecodeError:
                print("Warning: Could not parse bias monitoring JSON")
        
        # Load report tab state
        active_tab = session_data.get("active_report_tab", "overview")
        state.report_tabs.active_tab = active_tab
    else:
        # Save initial state if no session data exists
        save_comprehensive_session_data_to_db()

def add_chat_message(role: Role, content: str, session_id: str):
    """Add chat message and update clinical analysis if needed."""
    state = me.state(AppState)
    if not state.chat_history:
        state.chat_history = []
    
    timestamp_str = datetime.now().strftime("%I:%M %p").lstrip("0")
    message_id_str = f"{session_id}_{datetime.now().timestamp()}_{len(state.chat_history)}"
    
    new_message = ChatMessage(
        role=role,
        content=content,
        timestamp=timestamp_str,
        message_id=message_id_str
    )
    
    state.chat_history.append(new_message)
    save_message_to_db(session_id, new_message)
    
    # Trigger clinical analysis update if user message
    if role == "user":
        update_clinical_analysis()

def update_clinical_analysis():
    """Update clinical analysis based on current chat history."""
    from utils.clinical_analysis import calculate_urgency_score, generate_clinical_summary
    
    state = me.state(AppState)
    
    # Convert chat history to format expected by clinical analysis
    chat_data = [{"role": msg.role, "content": msg.content} for msg in state.chat_history]
    patient_data_dict = patient_data_to_dict(state.patient_data) if state.patient_data else None
    
    # Calculate urgency and risk assessment
    urgency_data = calculate_urgency_score(chat_data, patient_data_dict)
    
    # Update clinical analysis state
    clinical = state.clinical_analysis
    clinical.urgency_score = urgency_data.get("urgency_score", 0.0)
    clinical.risk_level = urgency_data.get("risk_level", "ROUTINE")
    clinical.risk_color = urgency_data.get("risk_color", "#4CAF50")
    clinical.recommended_action = urgency_data.get("recommended_action", "")
    clinical.flagged_phrases = urgency_data.get("flagged_phrases", [])
    clinical.risk_factors = urgency_data.get("risk_factors", [])
    clinical.analysis_timestamp = urgency_data.get("analysis_timestamp", "")
    
    # Generate clinical summary
    clinical_summary = generate_clinical_summary(chat_data, urgency_data)
    clinical.nice_protocol = clinical_summary.get("nice_protocol", "")
    
    # Check for emergency escalation
    if clinical.risk_level in ["IMMEDIATE", "URGENT"]:
        state.emergency_escalation = True
        add_emergency_alert(clinical.risk_level, clinical.recommended_action)
    
    # Save updated state
    save_comprehensive_session_data_to_db()

def add_emergency_alert(risk_level: str, action: str):
    """Add emergency alert to active alerts."""
    state = me.state(AppState)
    
    alert = {
        "id": str(uuid.uuid4()),
        "type": "emergency" if risk_level == "IMMEDIATE" else "urgent",
        "title": f"{risk_level} Medical Attention Required",
        "message": f"Recommended action: {action}",
        "timestamp": datetime.now().isoformat(),
        "dismissed": False
    }
    
    state.active_alerts.append(alert)

def update_patient_data(patient_updates: Dict[str, Any]):
    """Update patient data from EHR or manual input."""
    state = me.state(AppState)
    patient = state.patient_data
    
    # Update patient fields if provided
    for field, value in patient_updates.items():
        if hasattr(patient, field):
            setattr(patient, field, value)
    
    save_comprehensive_session_data_to_db()

def add_uploaded_file(file_data: me.UploadedFile) -> str:
    """Add uploaded file to state and database."""
    state = me.state(AppState)
    
    file_id = str(uuid.uuid4())
    
    uploaded_file = UploadedFileState(
        file_id=file_id,
        original_filename=file_data.name,
        file_size=file_data.size,
        mime_type=file_data.mime_type,
        upload_timestamp=datetime.now().isoformat(),
        file_category="medical_document"
    )
    
    state.uploaded_files.append(uploaded_file)
    
    # Save file data to database
    save_file_to_db(state.session_id, uploaded_file, file_data.getvalue())
    save_comprehensive_session_data_to_db()
    
    return file_id

def save_clinical_report(report_data: Dict[str, Any], report_type: str = "clinical_assessment") -> str:
    """Save generated clinical report to database."""
    state = me.state(AppState)
    report_id = f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    # Use database manager
    import sqlite3
    from .database_manager import DATABASE_FILE
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO clinical_reports (report_id, session_id, report_type, report_data_json, created_at, updated_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        report_id,
        state.session_id,
        report_type,
        json.dumps(report_data),
        datetime.now().isoformat(),
        datetime.now().isoformat(),
        "active"
    ))
    
    conn.commit()
    conn.close()
    
    return report_id

def save_comprehensive_session_data_to_db():
    """Save complete session state to database using modular serializers."""
    state = me.state(AppState)
    
    # Prepare data for database
    session_data = {
        "session_id": state.session_id,
        "current_input": state.current_input,
        "is_bot_typing": state.is_bot_typing,
        "report_data_json": state.report_data_json,
        "report_is_generating": state.report_is_generating,
        "report_generation_progress": state.report_generation_progress,
        "report_generation_complete": state.report_generation_complete,
        "report_error_message": state.report_error_message,
        "clinical_analysis_json": clinical_analysis_to_json(state.clinical_analysis),
        "ehr_data_json": state.ehr_data_json,
        "uploaded_documents_json": json.dumps([{
            "file_id": f.file_id,
            "original_filename": f.original_filename,
            "file_size": f.file_size,
            "mime_type": f.mime_type,
            "upload_timestamp": f.upload_timestamp,
            "file_category": f.file_category
        } for f in state.uploaded_files]) if state.uploaded_files else None,
        "patient_data_json": patient_data_to_json(state.patient_data),
        "active_report_tab": state.report_tabs.active_tab if state.report_tabs else "overview",
        "bias_monitoring_json": bias_monitoring_to_json(state.bias_monitoring)
    }
    
    # Use database manager
    db_save_session(session_data)

def update_report_state_in_app_and_db(report_data_dict: Optional[Dict] = None,
                                     is_generating: Optional[bool] = None,
                                     generation_progress: Optional[str] = None,
                                     generation_complete: Optional[bool] = None,
                                     error_message: Optional[str] = None):
    """Update report state in AppState and database."""
    state = me.state(AppState)
    
    if report_data_dict is not None:
        state.report_data_json = json.dumps(report_data_dict)
    if is_generating is not None:
        state.report_is_generating = is_generating
    if generation_progress is not None:
        state.report_generation_progress = generation_progress
    if generation_complete is not None:
        state.report_generation_complete = generation_complete
    if error_message is not None:
        state.report_error_message = error_message
    elif error_message == "":
        state.report_error_message = None
    
    save_comprehensive_session_data_to_db()
