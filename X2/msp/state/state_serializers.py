# msp/state/state_serializers.py

import json
from typing import Dict, Any, Optional

def clinical_analysis_to_json(clinical) -> Optional[str]:
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

def patient_data_to_json(patient) -> Optional[str]:
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

def patient_data_to_dict(patient) -> Optional[Dict[str, Any]]:
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

def bias_monitoring_to_json(bias) -> Optional[str]:
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

def load_clinical_analysis_from_dict(clinical_state, data: Dict[str, Any]):
    """Load clinical analysis from dictionary."""
    clinical_state.urgency_score = data.get("urgency_score", 0.0)
    clinical_state.risk_level = data.get("risk_level", "ROUTINE")
    clinical_state.risk_color = data.get("risk_color", "#4CAF50")
    clinical_state.recommended_action = data.get("recommended_action", "")
    clinical_state.flagged_phrases = data.get("flagged_phrases", [])
    clinical_state.risk_factors = data.get("risk_factors", [])
    clinical_state.analysis_timestamp = data.get("analysis_timestamp", "")
    clinical_state.nice_protocol = data.get("nice_protocol", "")
    clinical_state.clinical_entities = data.get("clinical_entities", [])

def load_patient_data_from_dict(patient_state, data: Dict[str, Any]):
    """Load patient data from dictionary."""
    patient_state.nhs_number = data.get("nhs_number", "")
    patient_state.name = data.get("name", "")
    patient_state.age = data.get("age")
    patient_state.gender = data.get("gender", "")
    patient_state.birth_date = data.get("birth_date", "")
    patient_state.address = data.get("address", "")
    patient_state.phone = data.get("phone", "")
    patient_state.allergies = data.get("allergies", [])
    patient_state.current_medications = data.get("current_medications", [])
    patient_state.medical_conditions = data.get("medical_conditions", [])
    patient_state.pregnancy_status = data.get("pregnancy_status", False)

def load_bias_monitoring_from_dict(bias_state, data: Dict[str, Any]):
    """Load bias monitoring from dictionary."""
    bias_state.demographic_parity = data.get("demographic_parity", 0.0)
    bias_state.equalized_odds = data.get("equalized_odds", 0.0)
    bias_state.individual_fairness = data.get("individual_fairness", 0.0)
    bias_state.counterfactual_fairness = data.get("counterfactual_fairness", 0.0)
    bias_state.overall_fairness_score = data.get("overall_fairness_score", 0.0)
    bias_state.bias_flags = data.get("bias_flags", [])
    bias_state.monitoring_timestamp = data.get("monitoring_timestamp", "")
