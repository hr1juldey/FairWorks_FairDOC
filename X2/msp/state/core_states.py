# msp/state/core_states.py

import mesop as me
from typing import List, Literal, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

# Import types only for type checking to avoid circular imports
if TYPE_CHECKING:
    from .clinical_states import ClinicalAnalysisState, BiasMonitoringState
    from .patient_states import PatientDataState
    from .document_states import UploadedFileState
    from .ui_states import ReportTabsState

Role = Literal["user", "assistant", "system"]

@me.stateclass
class ChatMessage:
    role: Role = "user"
    content: str = ""
    timestamp: str = ""
    message_id: str = ""

@me.stateclass
class AppState:
    """Main application state with comprehensive medical triage data"""
    
    # Navigation and UI state
    current_page: str = "chat"
    home_page_viewed: bool = False
    selected_feature: str = ""
    newsletter_subscribed: bool = False
    
    # Chat functionality
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
    
    # FIXED: Proper type hints using TYPE_CHECKING imports
    clinical_analysis: Optional['ClinicalAnalysisState'] = None
    patient_data: Optional['PatientDataState'] = None
    uploaded_files: Optional[List['UploadedFileState']] = None
    bias_monitoring: Optional['BiasMonitoringState'] = None
    report_tabs: Optional['ReportTabsState'] = None
    
    # EHR integration
    ehr_data_json: Optional[str] = None
    
    # Clinical alerts and notifications
    active_alerts: List[Dict[str, Any]] = None
    emergency_escalation: bool = False
    
    def __post_init__(self):
        """Initialize all nested objects and None fields"""
        # Import here to avoid circular imports (runtime imports)
        from .clinical_states import ClinicalAnalysisState, BiasMonitoringState
        from .document_states import UploadedFileState
        from .ui_states import ReportTabsState
        from .patient_states import PatientDataState
        
        # Initialize list fields
        if self.chat_history is None:
            self.chat_history = []
        if self.uploaded_files is None:
            self.uploaded_files = []
        if self.active_alerts is None:
            self.active_alerts = []
        
        # Initialize nested state objects
        if self.clinical_analysis is None:
            self.clinical_analysis = ClinicalAnalysisState()
        if self.patient_data is None:
            self.patient_data = PatientDataState()
        if self.bias_monitoring is None:
            self.bias_monitoring = BiasMonitoringState()
        if self.report_tabs is None:
            self.report_tabs = ReportTabsState()
