"""
Gold Standards Seed Data for DSPy Training and Evaluation
Expert-labeled complete conversations for medical triage model optimization
"""

from src.app2.models.schemas.medical_triage import MedicalOutcome

GOLD_STANDARDS_SEED_DATA = [
    {
        "title": "Emergency Chest Pain - STEMI",
        "description": "58-year-old male with classic ST-elevation MI symptoms requiring immediate PCI",
        "primary_symptom": "chest_pain",
        "expected_outcome": MedicalOutcome.EMERGENCY,
        "patient_age": 58,
        "patient_gender": "male",
        "expected_red_flags": ["crushing_chest_pain", "left_arm_radiation", "severe_sweating"],
        "should_escalate": True,
        "conversation_dialogue": [
            {
                "turn": 1,
                "user_message": "I have severe crushing chest pain going to my left arm",
                "agent_question": "When did this start and are you experiencing sweating or nausea?",
                "expected_classification": "inconclusive",
                "red_flags_detected": ["crushing_chest_pain", "left_arm_radiation"],
                "confidence_score": 65
            },
            {
                "turn": 2,
                "user_message": "Started 45 minutes ago, I'm sweating heavily and feel very nauseous",
                "agent_question": "EMERGENCY - Call 999 immediately. You need emergency medical attention now.",
                "expected_classification": "emergency",
                "red_flags_detected": ["crushing_chest_pain", "left_arm_radiation", "severe_sweating"],
                "confidence_score": 95
            }
        ],
        "relevant_protocols": ["CG95_CHEST_PAIN", "NG136_MI"],
        "minimum_confidence_threshold": 90.0,
        "expected_turn_count": 2,
        "max_acceptable_turns": 3,
        "created_by": "dr_smith_cardiologist",
        "clinical_notes": "Classic STEMI presentation requiring immediate PCI"
    },
    {
        "title": "Routine Headache Assessment",
        "description": "32-year-old female with tension headache, suitable for self-care",
        "primary_symptom": "headache",
        "expected_outcome": MedicalOutcome.SELF_CARE,
        "patient_age": 32,
        "patient_gender": "female",
        "expected_red_flags": [],
        "should_escalate": False,
        "conversation_dialogue": [
            {
                "turn": 1,
                "user_message": "I have a headache that started this morning",
                "agent_question": "Can you describe the pain - is it throbbing, tight, or sharp?",
                "expected_classification": "inconclusive",
                "red_flags_detected": [],
                "confidence_score": 40
            },
            {
                "turn": 2,
                "user_message": "It's like a tight band around my head, not too severe",
                "agent_question": "Any nausea, vision changes, or neck stiffness?",
                "expected_classification": "inconclusive", 
                "red_flags_detected": [],
                "confidence_score": 50
            },
            {
                "turn": 3,
                "user_message": "No, just the tight feeling. I had similar headaches before",
                "agent_question": "COMPLETE - This sounds like tension headache. Rest, hydration, and paracetamol should help.",
                "expected_classification": "self_care",
                "red_flags_detected": [],
                "confidence_score": 85
            }
        ],
        "relevant_protocols": ["NG127_HEADACHE"],
        "minimum_confidence_threshold": 75.0,
        "expected_turn_count": 3,
        "max_acceptable_turns": 4,
        "created_by": "dr_jones_gp",
        "clinical_notes": "Typical tension headache pattern"
    },
    {
        "title": "Emergency Respiratory Distress",
        "description": "65-year-old with severe asthma exacerbation requiring immediate treatment",
        "primary_symptom": "breathing_difficulty",
        "expected_outcome": MedicalOutcome.EMERGENCY,
        "patient_age": 65,
        "patient_gender": "female",
        "expected_red_flags": ["severe_dyspnea", "silent_chest", "exhaustion"],
        "should_escalate": True,
        "conversation_dialogue": [
            {
                "turn": 1,
                "user_message": "I can't breathe properly, struggling to speak",
                "agent_question": "When did this breathing difficulty start and do you have asthma?",
                "expected_classification": "inconclusive",
                "red_flags_detected": ["severe_dyspnea"],
                "confidence_score": 70
            },
            {
                "turn": 2,
                "user_message": "Yes asthma, started 2 hours ago, getting worse, used inhaler many times",
                "agent_question": "EMERGENCY - Call 999 now. This sounds like severe asthma attack.",
                "expected_classification": "emergency",
                "red_flags_detected": ["severe_dyspnea", "exhaustion"],
                "confidence_score": 92
            }
        ],
        "relevant_protocols": ["NG80_ASTHMA_EXAC"],
        "minimum_confidence_threshold": 85.0,
        "expected_turn_count": 2,
        "max_acceptable_turns": 2,
        "created_by": "dr_respiratory_expert",
        "clinical_notes": "Severe asthma exacerbation with poor response to bronchodilators"
    }
    # Add 47 more examples following this pattern to reach 50 total
]

def get_gold_standards_by_outcome(outcome: MedicalOutcome) -> list:
    """Filter gold standards by expected outcome"""
    return [gs for gs in GOLD_STANDARDS_SEED_DATA if gs["expected_outcome"] == outcome]

def get_gold_standards_by_symptom(symptom: str) -> list:
    """Filter gold standards by primary symptom"""
    return [gs for gs in GOLD_STANDARDS_SEED_DATA if gs["primary_symptom"] == symptom]

def validate_gold_standards() -> dict:
    """Validate gold standards data quality"""
    total = len(GOLD_STANDARDS_SEED_DATA)
    emergency_count = len(get_gold_standards_by_outcome(MedicalOutcome.EMERGENCY))
    routine_count = len(get_gold_standards_by_outcome(MedicalOutcome.ROUTINE_DOCTOR))
    self_care_count = len(get_gold_standards_by_outcome(MedicalOutcome.SELF_CARE))
    
    return {
        "total_examples": total,
        "emergency_examples": emergency_count,
        "routine_examples": routine_count,
        "self_care_examples": self_care_count,
        "coverage_balanced": emergency_count >= 15 and routine_count >= 15 and self_care_count >= 15
    }
