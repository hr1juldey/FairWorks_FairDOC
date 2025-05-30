# File: Fairdoc\frontend\msp\utils\state_manager.py

"""
State Management for Fairdoc AI Chat Interface
Handles user conversation, case reports, and UI state
"""

import mesop as me
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

@dataclass
class Message:
    id: str
    content: str
    is_user: bool
    timestamp: datetime
    message_type: str = "text"  # text, image, file, alert
    file_url: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class CaseReport:
    case_id: str
    patient_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: str = ""
    symptoms: List[str] = None
    uploaded_files: List[str] = None
    urgency_score: float = 0.0
    importance_score: float = 0.0
    status: str = "in_progress"  # in_progress, completed, emergency
    created_at: datetime = None
    
    def __post_init__(self):
        if self.symptoms is None:
            self.symptoms = []
        if self.uploaded_files is None:
            self.uploaded_files = []
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class AppState:
    # Chat state
    messages: List[Message] = None
    current_input: str = ""
    
    # Question flow state
    current_question_index: int = 0
    questions_completed: bool = False
    awaiting_file_upload: bool = False
    
    # Case report state
    case_report: Optional[CaseReport] = None
    
    # UI state
    is_loading: bool = False
    show_emergency_alert: bool = False
    current_page: str = "chat"  # chat, report, emergency
    
    # Mock backend state
    processing_files: bool = False
    ai_analysis_complete: bool = False
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.case_report is None:
            self.case_report = CaseReport(
                case_id=str(uuid.uuid4()),
                patient_name="Patient",
                chief_complaint="Left chest pain with coughing and wheezing"
            )

# State management functions
def get_state() -> AppState:
    """Get current app state"""
    if not hasattr(me.state, "app_state"):
        me.state.app_state = AppState()
    return me.state.app_state

def add_message(content: str, is_user: bool, message_type: str = "text", **kwargs):
    """Add new message to chat"""
    state = get_state()
    message = Message(
        id=str(uuid.uuid4()),
        content=content,
        is_user=is_user,
        timestamp=datetime.now(),
        message_type=message_type,
        **kwargs
    )
    state.messages.append(message)

def update_case_report(**kwargs):
    """Update case report fields"""
    state = get_state()
    for key, value in kwargs.items():
        if hasattr(state.case_report, key):
            setattr(state.case_report, key, value)

def set_loading(is_loading: bool):
    """Set loading state"""
    state = get_state()
    state.is_loading = is_loading

def trigger_emergency_alert():
    """Trigger emergency alert"""
    state = get_state()
    state.show_emergency_alert = True
    state.case_report.status = "emergency"
    state.current_page = "emergency"
