# msp/state/state_manager.py

from utils.path_setup import setup_project_paths
setup_project_paths()

import mesop as me
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
import sqlite3
import json
import uuid

Role = Literal["user", "assistant", "system"]
DATABASE_FILE = "msp_chat_state.db"

def _get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> List[str]:
    """Helper function to get column names of a table."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]

def init_db():
    """Initialize database with all required tables and columns."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Create chat_messages table (if not exists)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            message_id TEXT NOT NULL UNIQUE
        )
    ''')
    
    # Create app_sessions table (if not exists)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS app_sessions (
            session_id TEXT PRIMARY KEY,
            current_input TEXT,
            is_bot_typing INTEGER,
            last_active TEXT,
            report_data_json TEXT,
            report_is_generating INTEGER,
            report_generation_progress TEXT,
            report_generation_complete INTEGER,
            report_error_message TEXT,
            clinical_analysis_json TEXT,
            ehr_data_json TEXT,
            uploaded_documents_json TEXT,
            patient_data_json TEXT,
            active_report_tab TEXT,
            bias_monitoring_json TEXT
        )
    ''')
    
    # Create clinical_reports table for storing generated reports
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clinical_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            report_type TEXT NOT NULL,
            report_data_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
    ''')
    
    # Create uploaded_files table for document management
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            mime_type TEXT NOT NULL,
            file_data BLOB,
            upload_timestamp TEXT NOT NULL,
            file_category TEXT DEFAULT 'medical_document'
        )
    ''')
    
    # Schema migration for existing tables
    existing_columns = _get_table_columns(cursor, "app_sessions")
    new_columns_to_add = {
        "report_data_json": "TEXT",
        "report_is_generating": "INTEGER",
        "report_generation_progress": "TEXT", 
        "report_generation_complete": "INTEGER",
        "report_error_message": "TEXT",
        "clinical_analysis_json": "TEXT",
        "ehr_data_json": "TEXT",
        "uploaded_documents_json": "TEXT",
        "patient_data_json": "TEXT",
        "active_report_tab": "TEXT",
        "bias_monitoring_json": "TEXT"
    }
    
    for col_name, col_type in new_columns_to_add.items():
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE app_sessions ADD COLUMN {col_name} {col_type};")
                print(f"Added column '{col_name}' to 'app_sessions' table.")
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not add column '{col_name}': {e}")
    
    conn.commit()
    conn.close()

# Initialize DB when module is loaded
init_db()

@me.stateclass
class ChatMessage:
    role: Role = "user"
    content: str = ""
    timestamp: str = ""
    message_id: str = ""

@me.stateclass
class ClinicalAnalysisState:
    """State for clinical analysis results"""
    urgency_score: float = 0.0
    risk_level: str = "ROUTINE"
    risk_color: str = "#4CAF50"
    recommended_action: str = ""
    # FIXED: Use empty lists instead of None to prevent deserialization errors
    flagged_phrases: List[Dict[str, Any]] = None
    risk_factors: List[str] = None
    analysis_timestamp: str = ""
    nice_protocol: str = ""
    clinical_entities: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize None list fields to prevent deserialization errors"""
        if self.flagged_phrases is None:
            self.flagged_phrases = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.clinical_entities is None:
            self.clinical_entities = []

@me.stateclass  
class PatientDataState:
    """State for patient demographics and medical history"""
    nhs_number: str = ""
    name: str = ""
    age: Optional[int] = None
    gender: str = ""
    birth_date: str = ""
    address: str = ""
    phone: str = ""
    # FIXED: Use empty lists instead of None
    allergies: List[str] = None
    current_medications: List[str] = None
    medical_conditions: List[Dict[str, Any]] = None
    pregnancy_status: bool = False
    
    def __post_init__(self):
        """Initialize None list fields to prevent deserialization errors"""
        if self.allergies is None:
            self.allergies = []
        if self.current_medications is None:
            self.current_medications = []
        if self.medical_conditions is None:
            self.medical_conditions = []

@me.stateclass
class UploadedFileState:
    """State for uploaded medical documents"""
    file_id: str = ""
    original_filename: str = ""
    file_size: int = 0
    mime_type: str = ""
    upload_timestamp: str = ""
    file_category: str = "medical_document"

@me.stateclass
class BiasMonitoringState:
    """State for AI bias monitoring and fairness metrics"""
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    individual_fairness: float = 0.0
    counterfactual_fairness: float = 0.0
    overall_fairness_score: float = 0.0
    # FIXED: Use empty list instead of None
    bias_flags: List[str] = None
    monitoring_timestamp: str = ""
    
    def __post_init__(self):
        """Initialize None list fields to prevent deserialization errors"""
        if self.bias_flags is None:
            self.bias_flags = []

@me.stateclass
class ReportTabsState:
    """State for report tab management"""
    active_tab: str = "overview"
    # FIXED: Use empty dict instead of None
    tabs_expanded: Dict[str, bool] = None
    
    def __post_init__(self):
        """Initialize None dict fields to prevent deserialization errors"""
        if self.tabs_expanded is None:
            self.tabs_expanded = {"ehr_overview": True, "nice_protocol": True}

@me.stateclass
class AppState:
    """Main application state with comprehensive medical triage data"""
    
    # Navigation and UI state
    current_page: str = "chat"
    home_page_viewed: bool = False
    selected_feature: str = ""
    newsletter_subscribed: bool = False
    
    # Chat functionality - FIXED: Use empty list instead of None
    chat_history: List[ChatMessage] = None
    current_input: str = ""
    is_bot_typing: bool = False
    session_id: str = "default_session"
    
    # Report generation state
    report_data_json: Optional[str] = None
    report_is_generating: bool = False
    report_generation_progress: str = ""
    report_generation_complete: bool = False
    report_error_message: Optional[str] = None
    
    # Clinical analysis data - FIXED: Initialize with actual instance
    clinical_analysis: ClinicalAnalysisState = None
    
    # Patient data - FIXED: Initialize with actual instance
    patient_data: PatientDataState = None
    
    # EHR integration
    ehr_data_json: Optional[str] = None
    
    # Document management - FIXED: Use empty list instead of None
    uploaded_files: List[UploadedFileState] = None
    
    # Bias monitoring - FIXED: Initialize with actual instance
    bias_monitoring: BiasMonitoringState = None
    
    # Report UI state - FIXED: Initialize with actual instance
    report_tabs: ReportTabsState = None
    
    # Clinical alerts and notifications - FIXED: Use empty list instead of None
    active_alerts: List[Dict[str, Any]] = None
    emergency_escalation: bool = False
    
    def __post_init__(self):
        """Initialize all nested objects and None fields to prevent deserialization errors"""
        # Initialize list fields
        if self.chat_history is None:
            self.chat_history = []
        if self.uploaded_files is None:
            self.uploaded_files = []
        if self.active_alerts is None:
            self.active_alerts = []
        
        # Initialize nested state objects - CRITICAL FIX
        if self.clinical_analysis is None:
            self.clinical_analysis = ClinicalAnalysisState()
        if self.patient_data is None:
            self.patient_data = PatientDataState()
        if self.bias_monitoring is None:
            self.bias_monitoring = BiasMonitoringState()
        if self.report_tabs is None:
            self.report_tabs = ReportTabsState()

def initialize_app_state(session_id: str = "default_session"):
    """Initialize comprehensive application state from database."""
    state = me.state(AppState)
    state.session_id = session_id
    
    # The __post_init__ method will handle initialization of nested objects
    # No need for manual initialization here since it's done in __post_init__
    
    # Load chat history
    if not state.chat_history:  # Changed from None check
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
                load_clinical_analysis_from_dict(clinical_data)
            except json.JSONDecodeError:
                print("Warning: Could not parse clinical analysis JSON")
        
        # Load EHR data
        state.ehr_data_json = session_data.get("ehr_data_json")
        
        # Load patient data
        patient_json = session_data.get("patient_data_json")
        if patient_json:
            try:
                patient_data = json.loads(patient_json)
                load_patient_data_from_dict(patient_data)
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
                load_bias_monitoring_from_dict(bias_data)
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
    if not state.chat_history:  # Changed from None check
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
    
    # Update clinical analysis state - Now safe since object exists
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
    
    # Now safe since patient_data is always initialized
    patient = state.patient_data
    
    # Update patient fields if provided
    if "nhs_number" in patient_updates:
        patient.nhs_number = patient_updates["nhs_number"]
    if "name" in patient_updates:
        patient.name = patient_updates["name"]
    if "age" in patient_updates:
        patient.age = patient_updates["age"]
    if "gender" in patient_updates:
        patient.gender = patient_updates["gender"]
    if "birth_date" in patient_updates:
        patient.birth_date = patient_updates["birth_date"]
    if "address" in patient_updates:
        patient.address = patient_updates["address"]
    if "phone" in patient_updates:
        patient.phone = patient_updates["phone"]
    if "allergies" in patient_updates:
        patient.allergies = patient_updates["allergies"]
    if "current_medications" in patient_updates:
        patient.current_medications = patient_updates["current_medications"]
    if "medical_conditions" in patient_updates:
        patient.medical_conditions = patient_updates["medical_conditions"]
    
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

# Database helper functions
def save_message_to_db(session_id: str, message: ChatMessage):
    """Save chat message to database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_messages (session_id, role, content, timestamp, message_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, message.role, message.content, message.timestamp, message.message_id))
    
    conn.commit()
    conn.close()

def load_chat_history_from_db(session_id: str) -> List[ChatMessage]:
    """Load chat history from database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT role, content, timestamp, message_id FROM chat_messages
        WHERE session_id = ? ORDER BY id ASC
    ''', (session_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [ChatMessage(role=row[0], content=row[1], timestamp=row[2], message_id=row[3]) for row in rows]

def save_comprehensive_session_data_to_db():
    """Save complete session state to database."""
    state = me.state(AppState)
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Serialize complex state objects to JSON
    clinical_json = clinical_analysis_to_json(state.clinical_analysis)
    patient_json = patient_data_to_json(state.patient_data)
    files_json = json.dumps([{
        "file_id": f.file_id,
        "original_filename": f.original_filename,
        "file_size": f.file_size,
        "mime_type": f.mime_type,
        "upload_timestamp": f.upload_timestamp,
        "file_category": f.file_category
    } for f in state.uploaded_files]) if state.uploaded_files else None
    bias_json = bias_monitoring_to_json(state.bias_monitoring)
    active_tab = state.report_tabs.active_tab if state.report_tabs else "overview"
    
    cursor.execute('''
        INSERT OR REPLACE INTO app_sessions (
            session_id, current_input, is_bot_typing, last_active,
            report_data_json, report_is_generating, report_generation_progress,
            report_generation_complete, report_error_message,
            clinical_analysis_json, ehr_data_json, uploaded_documents_json,
            patient_data_json, active_report_tab, bias_monitoring_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        state.session_id,
        state.current_input,
        1 if state.is_bot_typing else 0,
        datetime.now().isoformat(),
        state.report_data_json,
        1 if state.report_is_generating else 0,
        state.report_generation_progress,
        1 if state.report_generation_complete else 0,
        state.report_error_message,
        clinical_json,
        state.ehr_data_json,
        files_json,
        patient_json,
        active_tab,
        bias_json
    ))
    
    conn.commit()
    conn.close()

def load_session_data_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session data from database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM app_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None

def save_file_to_db(session_id: str, file_state: UploadedFileState, file_data: bytes):
    """Save uploaded file data to database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO uploaded_files (
            file_id, session_id, original_filename, file_size, 
            mime_type, file_data, upload_timestamp, file_category
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        file_state.file_id,
        session_id,
        file_state.original_filename,
        file_state.file_size,
        file_state.mime_type,
        file_data,
        file_state.upload_timestamp,
        file_state.file_category
    ))
    
    conn.commit()
    conn.close()

# Serialization helper functions
def clinical_analysis_to_json(clinical: Optional[ClinicalAnalysisState]) -> Optional[str]:
    """Convert clinical analysis state to JSON."""
    if not clinical:
        return None
    
    return json.dumps({
        "urgency_score": clinical.urgency_score,
        "risk_level": clinical.risk_level,
        "risk_color": clinical.risk_color,
        "recommended_action": clinical.recommended_action,
        "flagged_phrases": clinical.flagged_phrases or [],
        "risk_factors": clinical.risk_factors or [],
        "analysis_timestamp": clinical.analysis_timestamp,
        "nice_protocol": clinical.nice_protocol,
        "clinical_entities": clinical.clinical_entities or []
    })

def load_clinical_analysis_from_dict(data: Dict[str, Any]):
    """Load clinical analysis from dictionary."""
    state = me.state(AppState)
    clinical = state.clinical_analysis
    
    clinical.urgency_score = data.get("urgency_score", 0.0)
    clinical.risk_level = data.get("risk_level", "ROUTINE")
    clinical.risk_color = data.get("risk_color", "#4CAF50")
    clinical.recommended_action = data.get("recommended_action", "")
    clinical.flagged_phrases = data.get("flagged_phrases", [])
    clinical.risk_factors = data.get("risk_factors", [])
    clinical.analysis_timestamp = data.get("analysis_timestamp", "")
    clinical.nice_protocol = data.get("nice_protocol", "")
    clinical.clinical_entities = data.get("clinical_entities", [])

def patient_data_to_json(patient: Optional[PatientDataState]) -> Optional[str]:
    """Convert patient data state to JSON."""
    if not patient:
        return None
    
    return json.dumps({
        "nhs_number": patient.nhs_number,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "birth_date": patient.birth_date,
        "address": patient.address,
        "phone": patient.phone,
        "allergies": patient.allergies or [],
        "current_medications": patient.current_medications or [],
        "medical_conditions": patient.medical_conditions or [],
        "pregnancy_status": patient.pregnancy_status
    })

def patient_data_to_dict(patient: Optional[PatientDataState]) -> Optional[Dict[str, Any]]:
    """Convert patient data state to dictionary."""
    if not patient:
        return None
    
    return {
        "nhs_number": patient.nhs_number,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "birth_date": patient.birth_date,
        "address": patient.address,
        "phone": patient.phone,
        "allergies": patient.allergies or [],
        "current_medications": patient.current_medications or [],
        "medical_conditions": patient.medical_conditions or [],
        "pregnancy_status": patient.pregnancy_status
    }

def load_patient_data_from_dict(data: Dict[str, Any]):
    """Load patient data from dictionary."""
    state = me.state(AppState)
    patient = state.patient_data
    
    patient.nhs_number = data.get("nhs_number", "")
    patient.name = data.get("name", "")
    patient.age = data.get("age")
    patient.gender = data.get("gender", "")
    patient.birth_date = data.get("birth_date", "")
    patient.address = data.get("address", "")
    patient.phone = data.get("phone", "")
    patient.allergies = data.get("allergies", [])
    patient.current_medications = data.get("current_medications", [])
    patient.medical_conditions = data.get("medical_conditions", [])
    patient.pregnancy_status = data.get("pregnancy_status", False)

def bias_monitoring_to_json(bias: Optional[BiasMonitoringState]) -> Optional[str]:
    """Convert bias monitoring state to JSON."""
    if not bias:
        return None
    
    return json.dumps({
        "demographic_parity": bias.demographic_parity,
        "equalized_odds": bias.equalized_odds,
        "individual_fairness": bias.individual_fairness,
        "counterfactual_fairness": bias.counterfactual_fairness,
        "overall_fairness_score": bias.overall_fairness_score,
        "bias_flags": bias.bias_flags or [],
        "monitoring_timestamp": bias.monitoring_timestamp
    })

def load_bias_monitoring_from_dict(data: Dict[str, Any]):
    """Load bias monitoring from dictionary."""
    state = me.state(AppState)
    bias = state.bias_monitoring
    
    bias.demographic_parity = data.get("demographic_parity", 0.0)
    bias.equalized_odds = data.get("equalized_odds", 0.0)
    bias.individual_fairness = data.get("individual_fairness", 0.0)
    bias.counterfactual_fairness = data.get("counterfactual_fairness", 0.0)
    bias.overall_fairness_score = data.get("overall_fairness_score", 0.0)
    bias.bias_flags = data.get("bias_flags", [])
    bias.monitoring_timestamp = data.get("monitoring_timestamp", "")

# Convenience functions for backward compatibility
def save_session_data_to_db(session_id: str, current_input: str, is_bot_typing: bool, 
                           report_data_json: Optional[str] = None, 
                           report_is_generating: bool = False,
                           report_generation_progress: str = "",
                           report_generation_complete: bool = False,
                           report_error_message: Optional[str] = None):
    """Backward compatibility function for basic session data saving."""
    save_comprehensive_session_data_to_db()

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
