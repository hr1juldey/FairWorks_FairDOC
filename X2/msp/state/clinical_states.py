# msp/state/clinical_states.py

import mesop as me
from typing import List, Dict, Any

@me.stateclass
class ClinicalAnalysisState:
    """State for clinical analysis results"""
    urgency_score: float = 0.0
    risk_level: str = "ROUTINE"
    risk_color: str = "#4CAF50"
    recommended_action: str = ""
    flagged_phrases: List[Dict[str, Any]] = None
    risk_factors: List[str] = None
    analysis_timestamp: str = ""
    nice_protocol: str = ""
    clinical_entities: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize None list fields"""
        if self.flagged_phrases is None:
            self.flagged_phrases = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.clinical_entities is None:
            self.clinical_entities = []

@me.stateclass
class BiasMonitoringState:
    """State for AI bias monitoring and fairness metrics"""
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    individual_fairness: float = 0.0
    counterfactual_fairness: float = 0.0
    overall_fairness_score: float = 0.0
    bias_flags: List[str] = None
    monitoring_timestamp: str = ""
    
    def __post_init__(self):
        """Initialize None list fields"""
        if self.bias_flags is None:
            self.bias_flags = []
