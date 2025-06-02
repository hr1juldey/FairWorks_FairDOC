# msp/state/clinical_states.py

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ClinicalAnalysisState:
    """State for clinical analysis results"""
    urgency_score: float = 0.0
    risk_level: str = "ROUTINE"
    risk_color: str = "#4CAF50"
    recommended_action: str = ""
    flagged_phrases: List[Dict[str, Any]] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    analysis_timestamp: str = ""
    nice_protocol: str = ""
    clinical_entities: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BiasMonitoringState:
    """State for AI bias monitoring and fairness metrics"""
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    individual_fairness: float = 0.0
    counterfactual_fairness: float = 0.0
    overall_fairness_score: float = 0.0
    bias_flags: List[str] = field(default_factory=list)
    monitoring_timestamp: str = ""
