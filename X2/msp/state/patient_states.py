# msp/state/patient_states.py

import mesop as me
from typing import List, Optional, Dict, Any

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
    allergies: List[str] = None
    current_medications: List[str] = None
    medical_conditions: List[Dict[str, Any]] = None
    pregnancy_status: bool = False
    
    def __post_init__(self):
        """Initialize None list fields"""
        if self.allergies is None:
            self.allergies = []
        if self.current_medications is None:
            self.current_medications = []
        if self.medical_conditions is None:
            self.medical_conditions = []
