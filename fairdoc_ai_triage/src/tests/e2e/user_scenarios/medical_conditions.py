# Medical Conditions - Comprehensive Medical Condition Database
"""
Medical conditions mapped to patient profiles with expected outcomes
Each condition includes symptom progression, red flags, and NICE protocol mappings
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog

from src.app2.models.schemas.medical_triage import MedicalOutcome

logger = structlog.get_logger(__name__)


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SeverityLevel(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class TrustLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
@dataclass
class SymptomProgression:
    """How symptoms develop during conversation"""
    initial: str                    # First symptoms mentioned
    intermediate: List[str]         # Symptoms revealed with questioning
    severe: List[str]               # Severe symptoms if condition worsens
    timeline: str                   # How symptoms developed over time
    triggers: List[str]             # What triggers or worsens symptoms


@dataclass 
class MedicalCondition:
    """Complete medical condition for testing"""
    condition_id: str
    condition_name: str
    expected_outcome: MedicalOutcome
    severity: SeverityLevel

    # Clinical characteristics
    symptom_progression: SymptomProgression
    red_flag_indicators: List[str]
    relevant_protocols: List[str]   # NICE protocol codes
    
    # Testing parameters
    minimum_turns_expected: int         # Minimum turns to identify condition
    maximum_turns_acceptable: int       # Maximum acceptable turns
    emergency_detection_critical: bool  # Whether missing emergency is critical failure
    confidence_threshold: float         # Expected final confidence
    
    # Patient behavior modifiers
    pain_level: int                  # 0-10 pain scale
    urgency_perception: UrgencyLevel          # How urgent patient perceives condition
    communication_difficulty: float  # 0-1, how hard condition makes communication
    
    def get_progressive_symptoms(self, turn_number: int) -> List[str]:
        """Get symptoms that should be available at specific turn"""
        if turn_number == 1:
            return [self.symptom_progression.initial]
        elif turn_number <= 3:
            return [self.symptom_progression.initial] + self.symptom_progression.intermediate[:2]
        else:
            all_symptoms = [self.symptom_progression.initial] + self.symptom_progression.intermediate
            if turn_number > 5 or self.expected_outcome == MedicalOutcome.EMERGENCY:
                all_symptoms.extend(self.symptom_progression.severe)
            return all_symptoms
    
    def should_reveal_red_flag(self, turn_number: int, agent_question: str) -> bool:
        """Determine if red flag should be revealed based on context"""
        
        # Emergency conditions reveal red flags earlier
        if self.expected_outcome == MedicalOutcome.EMERGENCY:
            if turn_number <= 2:
                return True
        
        # Red flags revealed if directly asked about
        question_lower = agent_question.lower()
        for red_flag in self.red_flag_indicators:
            if any(keyword in question_lower for keyword in red_flag.lower().split('_')):
                return True
        
        # Progressive revelation for non-emergency
        if turn_number >= 4:
            return True
        
        return False

# Comprehensive Medical Conditions Database
MEDICAL_CONDITIONS = {
    
    "acute_mi_stemi": MedicalCondition(
        condition_id="acute_mi_stemi",
        condition_name="ST-Elevation Myocardial Infarction",
        expected_outcome=MedicalOutcome.EMERGENCY,
        severity=SeverityLevel.CRITICAL,
        
        symptom_progression=SymptomProgression(
            initial="severe chest pain like someone sitting on my chest",
            intermediate=[
                "pain going to left arm and jaw",
                "heavy sweating even though I'm cold", 
                "feeling very nauseous and dizzy",
                "feeling like I might faint"
            ],
            severe=[
                "can barely breathe properly",
                "pain is worst I've ever felt in my life",
                "feel like I'm going to die"
            ],
            timeline="started suddenly 45 minutes ago and getting worse",
            triggers=["was climbing stairs at work", "had been stressed lately"]
        ),
        
        red_flag_indicators=[
            "crushing_chest_pain",
            "left_arm_radiation", 
            "severe_sweating",
            "nausea_with_chest_pain",
            "worst_pain_ever"
        ],
        
        relevant_protocols=["CG95_CHEST_PAIN", "NG136_MI"],
        minimum_turns_expected=1,
        maximum_turns_acceptable=3,
        emergency_detection_critical=True,
        confidence_threshold=90.0,
        
        pain_level=9,
        urgency_perception=UrgencyLevel.CRITICAL,
        communication_difficulty=0.3  # Pain makes it hard to focus
    ),
    
    "acute_appendicitis": MedicalCondition(
        condition_id="acute_appendicitis", 
        condition_name="Acute Appendicitis",
        expected_outcome=MedicalOutcome.EMERGENCY,
        severity=SeverityLevel.SEVERE,
        
        symptom_progression=SymptomProgression(
            initial="stomach pain that started around my belly button",
            intermediate=[
                "pain moved to right side lower abdomen",
                "feeling nauseous and vomited twice",
                "can't walk properly because of pain",
                "pain gets worse when I cough or move"
            ],
            severe=[
                "pain is unbearable now",
                "can't stand up straight",
                "feeling feverish and very sick"
            ],
            timeline="started this morning around navel, moved to right side in last 2 hours",
            triggers=["started after eating breakfast", "getting worse with any movement"]
        ),
        
        red_flag_indicators=[
            "pain_migration",
            "mcburney_point_tenderness", 
            "rebound_tenderness",
            "fever_with_abdominal_pain",
            "guarding"
        ],
        
        relevant_protocols=["CG141_APPENDICITIS"],
        minimum_turns_expected=2,
        maximum_turns_acceptable=4,
        emergency_detection_critical=True,
        confidence_threshold=85.0,
        
        pain_level=8,
        urgency_perception=UrgencyLevel.HIGH,
        communication_difficulty=0.2
    ),
    
    "postpartum_depression": MedicalCondition(
        condition_id="postpartum_depression",
        condition_name="Postpartum Depression", 
        expected_outcome=MedicalOutcome.ROUTINE_DOCTOR,
        severity=SeverityLevel.MODERATE,

        symptom_progression=SymptomProgression(
            initial="feeling very low and sad most of the time since baby was born",
            intermediate=[
                "crying for no reason almost daily",
                "not feeling connected to baby like I should",
                "very tired but can't sleep even when baby sleeps",
                "feel like I'm failing as a mother"
            ],
            severe=[
                "sometimes think everyone would be better off without me",
                "difficulty taking care of baby properly",
                "feel guilty all the time about everything"
            ],
            timeline="started about 3 weeks after baby born, getting worse last few days",
            triggers=["baby crying", "family comments", "when alone"]
        ),
        
        red_flag_indicators=[
            "postpartum_sadness",
            "bonding_difficulties",
            "maternal_guilt", 
            "sleep_disturbance"
        ],
        
        relevant_protocols=["CG192_ANTENATAL_POSTNATAL_MENTAL_HEALTH"],
        minimum_turns_expected=3,
        maximum_turns_acceptable=6,
        emergency_detection_critical=False,
        confidence_threshold=75.0,
        
        pain_level=2,
        urgency_perception=UrgencyLevel.MEDIUM,
        communication_difficulty=0.4  # Emotional state affects communication
    ),
    
    "hypertension_management": MedicalCondition(
        condition_id="hypertension_management",
        condition_name="Hypertension Follow-up",
        expected_outcome=MedicalOutcome.ROUTINE_DOCTOR,
        severity=SeverityLevel.MODERATE,

        symptom_progression=SymptomProgression(
            initial="my blood pressure readings have been high lately",
            intermediate=[
                "getting headaches more often especially in morning",
                "sometimes feel dizzy when I stand up",
                "feeling tired even with less work",
                "family says I look tired"
            ],
            severe=[
                "severe headache yesterday that worried me",
                "vision felt blurry for few minutes"
            ],
            timeline="BP been high for 2 weeks, symptoms started 4-5 days ago",
            triggers=["work stress", "not taking medicines regularly", "eating too much salt"]
        ),
        
        red_flag_indicators=[
            "severe_morning_headaches",
            "vision_changes",
            "chest_tightness"
        ],
        
        relevant_protocols=["NG19_HYPERTENSION", "NG136_HTN_CRISIS"],
        minimum_turns_expected=3,
        maximum_turns_acceptable=5,
        emergency_detection_critical=False,
        confidence_threshold=70.0,
        
        pain_level=3,
        urgency_perception=UrgencyLevel.MEDIUM,
        communication_difficulty=0.1
    ),
    
    "breast_lump_concern": MedicalCondition(
        condition_id="breast_lump_concern",
        condition_name="Breast Lump Evaluation",
        expected_outcome=MedicalOutcome.ROUTINE_DOCTOR,
        severity=SeverityLevel.MODERATE,
        
        symptom_progression=SymptomProgression(
            initial="found a lump in my left breast during self-examination",
            intermediate=[
                "lump feels hard and doesn't move much",
                "no pain but very worried about cancer",
                "no family history of breast cancer",
                "periods have been regular"
            ],
            severe=[
                "haven't been sleeping properly thinking about it",
                "checking the lump multiple times daily"
            ],
            timeline="noticed it 1 week ago during monthly self-exam",
            triggers=["monthly self-examination", "reading about breast cancer online"]
        ),
        
        red_flag_indicators=[
            "breast_lump",
            "lump_characteristics",
            "family_history_check"
        ],
        
        relevant_protocols=["CG164_FAMILIAL_BREAST_CANCER"],
        minimum_turns_expected=4,
        maximum_turns_acceptable=6,
        emergency_detection_critical=False, 
        confidence_threshold=75.0,
        
        pain_level=1,
        urgency_perception=UrgencyLevel.HIGH,    # High anxiety despite not emergency
        communication_difficulty=0.3  # Anxiety affects communication
    ),
    
    "recurring_uti": MedicalCondition(
        condition_id="recurring_uti",
        condition_name="Recurrent Urinary Tract Infection",
        expected_outcome=MedicalOutcome.ROUTINE_DOCTOR,
        severity=SeverityLevel.MODERATE,

        symptom_progression=SymptomProgression(
            initial="burning sensation when passing urine since 2 days",
            intermediate=[
                "passing urine very frequently but small amounts",
                "urine smells bad and looks cloudy",
                "lower abdomen feels heavy and uncomfortable",
                "this is happening again after 2 months"
            ],
            severe=[
                "some blood in urine this morning",
                "feeling feverish since yesterday evening"
            ],
            timeline="symptoms started 2 days ago, similar episode 2 months back",
            triggers=["not drinking enough water", "holding urine for long time"]
        ),
        
        red_flag_indicators=[
            "recurrent_uti",
            "hematuria",
            "fever_with_uti"
        ],
        
        relevant_protocols=["CG109_URINARY_INCONTINENCE", "NICE_UTI_MANAGEMENT"],
        minimum_turns_expected=3,
        maximum_turns_acceptable=5,
        emergency_detection_critical=False,
        confidence_threshold=80.0,
        
        pain_level=4,
        urgency_perception=UrgencyLevel.MEDIUM,
        communication_difficulty=0.2
    ),
    
    "tension_headache": MedicalCondition(
        condition_id="tension_headache",
        condition_name="Tension-Type Headache",
        expected_outcome=MedicalOutcome.SELF_CARE,
        severity=SeverityLevel.MILD,

        symptom_progression=SymptomProgression(
            initial="having headache like tight band around head since morning",
            intermediate=[
                "not throbbing but constant dull ache",
                "both sides of head affected",
                "gets worse with work stress",
                "had similar headaches before"
            ],
            severe=[
                "took paracetamol but still there",
                "affecting my work concentration"
            ],
            timeline="started this morning after stressful meeting, similar pattern before",
            triggers=["work deadline stress", "not enough sleep", "skipped breakfast"]
        ),
        
        red_flag_indicators=[],  # No red flags for simple tension headache
        
        relevant_protocols=["NG127_HEADACHE"],
        minimum_turns_expected=2,
        maximum_turns_acceptable=4,
        emergency_detection_critical=False,
        confidence_threshold=70.0,
        
        pain_level=5,
        urgency_perception=UrgencyLevel.LOW,
        communication_difficulty=0.1
    ),
    
    "gastroenteritis_viral": MedicalCondition(
        condition_id="gastroenteritis_viral", 
        condition_name="Viral Gastroenteritis",
        expected_outcome=MedicalOutcome.SELF_CARE,
        severity=SeverityLevel.MILD,
        
        symptom_progression=SymptomProgression(
            initial="loose motions and stomach upset since yesterday",
            intermediate=[
                "vomited 3-4 times yesterday, better today",
                "stomach cramping but bearable",
                "feeling weak and tired",
                "able to take some fluids"
            ],
            severe=[
                "worried about dehydration",
                "haven't eaten solid food since yesterday"
            ],
            timeline="started suddenly yesterday after eating street food",
            triggers=["street food", "might be something I ate"]
        ),
        
        red_flag_indicators=[],  # Mild gastroenteritis without red flags
        
        relevant_protocols=["NICE_GASTROENTERITIS"],
        minimum_turns_expected=2,
        maximum_turns_acceptable=4,
        emergency_detection_critical=False,
        confidence_threshold=75.0,
        
        pain_level=4,
        urgency_perception=UrgencyLevel.LOW,
        communication_difficulty=0.1
    )
}

def get_condition(condition_id: str) -> Optional[MedicalCondition]:
    """Get medical condition by ID"""
    return MEDICAL_CONDITIONS.get(condition_id)

def get_all_condition_ids() -> List[str]:
    """Get all available condition IDs"""
    return list(MEDICAL_CONDITIONS.keys())

def get_conditions_by_outcome(outcome: MedicalOutcome) -> List[MedicalCondition]:
    """Get all conditions with specific expected outcome"""
    return [condition for condition in MEDICAL_CONDITIONS.values() 
            if condition.expected_outcome == outcome]

def get_emergency_conditions() -> List[MedicalCondition]:
    """Get all emergency conditions"""
    return get_conditions_by_outcome(MedicalOutcome.EMERGENCY)

def get_routine_conditions() -> List[MedicalCondition]:
    """Get all routine consultation conditions"""
    return get_conditions_by_outcome(MedicalOutcome.ROUTINE_DOCTOR)

def get_self_care_conditions() -> List[MedicalCondition]:
    """Get all self-care conditions"""
    return get_conditions_by_outcome(MedicalOutcome.SELF_CARE)

def get_conditions_by_urgency(urgency: str) -> List[MedicalCondition]:
    """Get conditions by patient perceived urgency"""
    return [condition for condition in MEDICAL_CONDITIONS.values()
            if condition.urgency_perception == urgency]

def get_high_pain_conditions() -> List[MedicalCondition]:
    """Get conditions with high pain levels (7+)"""
    return [condition for condition in MEDICAL_CONDITIONS.values()
            if condition.pain_level >= 7]

def validate_medical_conditions() -> Dict[str, Any]:
    """Validate medical conditions database for testing completeness"""
    
    validation_results = {
        "total_conditions": len(MEDICAL_CONDITIONS),
        "outcome_distribution": {},
        "urgency_distribution": {},
        "pain_level_distribution": {},
        "red_flag_coverage": {},
        "protocol_coverage": {},
        "issues": []
    }
    
    # Outcome distribution
    outcome_dist = {}
    for condition in MEDICAL_CONDITIONS.values():
        outcome = condition.expected_outcome.value
        outcome_dist[outcome] = outcome_dist.get(outcome, 0) + 1
    validation_results["outcome_distribution"] = outcome_dist
    
    # Urgency distribution
    urgency_dist = {}
    for condition in MEDICAL_CONDITIONS.values():
        urgency = condition.urgency_perception
        urgency_dist[urgency] = urgency_dist.get(urgency, 0) + 1
    validation_results["urgency_distribution"] = urgency_dist
    
    # Pain level distribution
    pain_ranges = {"0-3": 0, "4-6": 0, "7-10": 0}
    for condition in MEDICAL_CONDITIONS.values():
        if condition.pain_level <= 3:
            pain_ranges["0-3"] += 1
        elif condition.pain_level <= 6:
            pain_ranges["4-6"] += 1
        else:
            pain_ranges["7-10"] += 1
    validation_results["pain_level_distribution"] = pain_ranges
    
    # Red flag coverage
    total_red_flags = 0
    conditions_with_red_flags = 0
    for condition in MEDICAL_CONDITIONS.values():
        if condition.red_flag_indicators:
            conditions_with_red_flags += 1
            total_red_flags += len(condition.red_flag_indicators)
    
    validation_results["red_flag_coverage"] = {
        "total_red_flags": total_red_flags,
        "conditions_with_red_flags": conditions_with_red_flags,
        "conditions_without_red_flags": len(MEDICAL_CONDITIONS) - conditions_with_red_flags
    }
    
    # Protocol coverage
    all_protocols = set()
    for condition in MEDICAL_CONDITIONS.values():
        all_protocols.update(condition.relevant_protocols)
    validation_results["protocol_coverage"] = {
        "unique_protocols": len(all_protocols),
        "protocols": sorted(list(all_protocols))
    }
    
    # Validation checks
    emergency_count = outcome_dist.get("emergency_route_to_doctor", 0)
    if emergency_count < 2:
        validation_results["issues"].append("Need at least 2 emergency conditions")
    
    routine_count = outcome_dist.get("routine_doctor_consultation", 0) 
    if routine_count < 2:
        validation_results["issues"].append("Need at least 2 routine consultation conditions")
    
    self_care_count = outcome_dist.get("self_care_advice", 0)
    if self_care_count < 2:
        validation_results["issues"].append("Need at least 2 self-care conditions")
    
    if pain_ranges["7-10"] < 2:
        validation_results["issues"].append("Need at least 2 high-pain conditions for testing")
    
    if conditions_with_red_flags < 4:
        validation_results["issues"].append("Need more conditions with red flags for detection testing")
    
    # Check for critical emergency conditions
    critical_emergency_count = len([c for c in MEDICAL_CONDITIONS.values() 
                                   if c.emergency_detection_critical])
    if critical_emergency_count < 2:
        validation_results["issues"].append("Need at least 2 critical emergency detection conditions")
    
    return validation_results

def get_condition_test_matrix() -> Dict[str, List[str]]:
    """Get test matrix mapping condition characteristics to condition IDs"""
    
    matrix = {
        "emergency_critical": [],
        "emergency_non_critical": [],
        "routine_high_anxiety": [],
        "routine_low_anxiety": [],
        "self_care_simple": [],
        "self_care_complex": [],
        "high_pain": [],
        "low_pain": [],
        "many_red_flags": [],
        "few_red_flags": []
    }
    
    for condition_id, condition in MEDICAL_CONDITIONS.items():
        # Emergency classification
        if condition.expected_outcome == MedicalOutcome.EMERGENCY:
            if condition.emergency_detection_critical:
                matrix["emergency_critical"].append(condition_id)
            else:
                matrix["emergency_non_critical"].append(condition_id)
        
        # Routine classification  
        elif condition.expected_outcome == MedicalOutcome.ROUTINE_DOCTOR:
            if condition.urgency_perception in ["high", "very_high"]:
                matrix["routine_high_anxiety"].append(condition_id)
            else:
                matrix["routine_low_anxiety"].append(condition_id)
        
        # Self-care classification
        elif condition.expected_outcome == MedicalOutcome.SELF_CARE:
            if condition.maximum_turns_acceptable <= 3:
                matrix["self_care_simple"].append(condition_id)
            else:
                matrix["self_care_complex"].append(condition_id)
        
        # Pain classification
        if condition.pain_level >= 7:
            matrix["high_pain"].append(condition_id)
        elif condition.pain_level <= 3:
            matrix["low_pain"].append(condition_id)
        
        # Red flag classification
        if len(condition.red_flag_indicators) >= 4:
            matrix["many_red_flags"].append(condition_id)
        elif len(condition.red_flag_indicators) <= 1:
            matrix["few_red_flags"].append(condition_id)
    
    return matrix

def get_balanced_condition_sample(count: int = 6) -> List[str]:
    """Get balanced sample of conditions for comprehensive testing"""
    
    all_conditions = list(MEDICAL_CONDITIONS.keys())
    
    if count >= len(all_conditions):
        return all_conditions
    
    # Ensure we get at least one from each outcome category
    emergency_conditions = [cid for cid, c in MEDICAL_CONDITIONS.items() 
                           if c.expected_outcome == MedicalOutcome.EMERGENCY]
    routine_conditions = [cid for cid, c in MEDICAL_CONDITIONS.items()
                         if c.expected_outcome == MedicalOutcome.ROUTINE_DOCTOR]  
    self_care_conditions = [cid for cid, c in MEDICAL_CONDITIONS.items()
                           if c.expected_outcome == MedicalOutcome.SELF_CARE]
    
    balanced_sample = []
    
    # Add at least one emergency (prioritize critical)
    critical_emergency = [cid for cid in emergency_conditions 
                         if MEDICAL_CONDITIONS[cid].emergency_detection_critical]
    if critical_emergency:
        balanced_sample.append(critical_emergency[0])
    elif emergency_conditions:
        balanced_sample.append(emergency_conditions[0])
    
    # Add at least one routine
    if routine_conditions:
        balanced_sample.append(routine_conditions[0])
    
    # Add at least one self-care
    if self_care_conditions:
        balanced_sample.append(self_care_conditions[0])
    
    # Fill remaining slots with diverse conditions
    remaining_slots = count - len(balanced_sample)
    remaining_conditions = [cid for cid in all_conditions if cid not in balanced_sample]
    
    # Try to add variety in pain levels, red flags, etc.
    if remaining_slots > 0 and remaining_conditions:
        import random
        additional = random.sample(remaining_conditions, min(remaining_slots, len(remaining_conditions)))
        balanced_sample.extend(additional)
    
    return balanced_sample[:count]

def simulate_symptom_revelation(condition_id: str, turn_number: int, agent_question: str = "") -> Dict[str, Any]:
    """Simulate how symptoms would be revealed during conversation"""
    
    condition = get_condition(condition_id)
    if not condition:
        return {"error": f"Condition {condition_id} not found"}
    
    available_symptoms = condition.get_progressive_symptoms(turn_number)
    should_reveal_red_flags = condition.should_reveal_red_flag(turn_number, agent_question)
    
    red_flags_to_reveal = []
    if should_reveal_red_flags:
        # Reveal red flags based on turn number and question relevance
        if turn_number <= 2 and condition.expected_outcome == MedicalOutcome.EMERGENCY:
            red_flags_to_reveal = condition.red_flag_indicators[:2]  # Most critical first
        elif turn_number >= 3:
            red_flags_to_reveal = condition.red_flag_indicators
    
    return {
        "condition_id": condition_id,
        "turn_number": turn_number, 
        "available_symptoms": available_symptoms,
        "red_flags_revealed": red_flags_to_reveal,
        "patient_urgency": condition.urgency_perception,
        "pain_level": condition.pain_level,
        "communication_difficulty": condition.communication_difficulty
    }

def get_condition_symptoms_at_turn(condition_id: str, turn_number: int) -> str:
    """Get symptoms that should be available at specific turn for a condition"""
    condition = get_condition(condition_id)
    if not condition:
        return "Unknown condition symptoms"
    
    symptoms = condition.get_progressive_symptoms(turn_number)
    return "; ".join(symptoms) if symptoms else "No specific symptoms for this turn"
