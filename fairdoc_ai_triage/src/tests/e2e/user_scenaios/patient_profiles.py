# Patient Profiles - Diverse Indian Patient Demographics
"""
Comprehensive patient profiles representing diverse Indian demographics
Each profile includes medical conditions, communication styles, and socioeconomic backgrounds
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class Education(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GRADUATE = "graduate"
    POSTGRADUATE = "postgraduate"

class EconomicStatus(str, Enum):
    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    MIDDLE = "middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"

class EmotionalState(str, Enum):
    CALM = "calm"
    ANXIOUS = "anxious"
    WORRIED = "worried"
    FEARFUL = "fearful"
    CONFUSED = "confused"
    SAD = "sad"
    HOPEFUL = "hopeful"

@dataclass
class CommunicationStyle:
    """Communication style characteristics"""
    english_proficiency: str  # "basic", "intermediate", "advanced"
    local_language_influence: int  # 0-100, how much local language affects English
    typing_speed: str  # "slow", "moderate", "fast"
    vocabulary: str  # "simple", "moderate", "complex"
    sentence_structure: str  # "basic", "moderate", "complex"
    cultural_expressions: List[str]  # Common expressions in their local context
    tech_comfort: str  # "low", "medium", "high"

@dataclass
class PatientProfile:
    """Complete patient profile for testing"""
    patient_id: str
    name: str
    age: int
    gender: Gender
    
    # Geographic and cultural
    city: str
    state: str
    region: str  # North, South, East, West, Central
    
    # Socioeconomic
    education: Education
    economic_status: EconomicStatus
    occupation: str
    
    # Medical background
    medical_condition: str  # Links to MedicalCondition ID
    previous_medical_experience: str
    health_anxiety_level: int  # 0-100
    trust_in_technology: int  # 0-100
    family_influence: int  # 0-100, how much family affects decisions
    
    # Communication and behavior
    communication_style: CommunicationStyle
    primary_emotion: EmotionalState
    stress_response: str  # How they respond to medical stress
    
    def get_emotional_modifiers(self) -> Dict[str, Any]:
        """Get emotional modifiers for conversation behavior"""
        
        base_modifiers = {
            "message_length_multiplier": 1.0,
            "repetition_likelihood": 0.1,
            "pause_duration_multiplier": 1.0,
            "anxiety_escalation_rate": 0.1
        }
        
        if self.primary_emotion == EmotionalState.ANXIOUS:
            base_modifiers.update({
                "message_length_multiplier": 1.3,
                "repetition_likelihood": 0.4,
                "anxiety_escalation_rate": 0.3
            })
        elif self.primary_emotion == EmotionalState.FEARFUL:
            base_modifiers.update({
                "message_length_multiplier": 0.7,
                "pause_duration_multiplier": 1.8,
                "anxiety_escalation_rate": 0.5
            })
        elif self.primary_emotion == EmotionalState.CONFUSED:
            base_modifiers.update({
                "repetition_likelihood": 0.3,
                "pause_duration_multiplier": 1.4
            })
        
        # Adjust for health anxiety
        if self.health_anxiety_level > 70:
            base_modifiers["message_length_multiplier"] *= 1.2
            base_modifiers["repetition_likelihood"] += 0.2
        
        return base_modifiers

# Comprehensive Patient Profiles Database
PATIENT_PROFILES = {
    
    "mumbai_tech_male": PatientProfile(
        patient_id="mumbai_tech_male",
        name="Arjun Sharma",
        age=29,
        gender=Gender.MALE,
        
        city="Mumbai",
        state="Maharashtra", 
        region="West",
        
        education=Education.POSTGRADUATE,
        economic_status=EconomicStatus.UPPER_MIDDLE,
        occupation="Software Engineer",
        
        medical_condition="acute_mi_stemi",  # Heart attack - emergency
        previous_medical_experience="Limited, mostly online consultations",
        health_anxiety_level=60,
        trust_in_technology=85,
        family_influence=40,
        
        communication_style=CommunicationStyle(
            english_proficiency="advanced",
            local_language_influence=20,
            typing_speed="fast",
            vocabulary="complex",
            sentence_structure="complex",
            cultural_expressions=["yaar", "bhai", "actually", "basically"],
            tech_comfort="high"
        ),
        
        primary_emotion=EmotionalState.ANXIOUS,
        stress_response="Becomes very detailed and analytical, asks many questions"
    ),
    
    "delhi_housewife": PatientProfile(
        patient_id="delhi_housewife",
        name="Priya Gupta",
        age=34,
        gender=Gender.FEMALE,
        
        city="Delhi",
        state="Delhi",
        region="North",
        
        education=Education.GRADUATE,
        economic_status=EconomicStatus.MIDDLE,
        occupation="Homemaker",
        
        medical_condition="postpartum_depression",  # Routine consultation
        previous_medical_experience="Regular visits to family doctor, gynecologist",
        health_anxiety_level=75,
        trust_in_technology=45,
        family_influence=80,
        
        communication_style=CommunicationStyle(
            english_proficiency="intermediate",
            local_language_influence=40,
            typing_speed="moderate",
            vocabulary="moderate",
            sentence_structure="moderate",
            cultural_expressions=["ji", "haan", "acha", "pareshani", "ghar wale"],
            tech_comfort="medium"
        ),
        
        primary_emotion=EmotionalState.SAD,
        stress_response="Tends to be hesitant, seeks family approval, emotional"
    ),
    
    "chennai_senior": PatientProfile(
        patient_id="chennai_senior",
        name="K. Venkatesh",
        age=67,
        gender=Gender.MALE,
        
        city="Chennai",
        state="Tamil Nadu",
        region="South",
        
        education=Education.SECONDARY,
        economic_status=EconomicStatus.LOWER_MIDDLE,
        occupation="Retired clerk",
        
        medical_condition="hypertension_management",  # Routine consultation  
        previous_medical_experience="Regular BP monitoring, multiple doctor visits",
        health_anxiety_level=85,
        trust_in_technology=25,
        family_influence=90,
        
        communication_style=CommunicationStyle(
            english_proficiency="basic",
            local_language_influence=70,
            typing_speed="slow",
            vocabulary="simple",
            sentence_structure="basic",
            cultural_expressions=["sir", "madam", "please", "one minute", "family problem"],
            tech_comfort="low"
        ),
        
        primary_emotion=EmotionalState.WORRIED,
        stress_response="Very respectful, concerned about cost, involves family in decisions"
    ),
    
    "kolkata_student": PatientProfile(
        patient_id="kolkata_student",
        name="Ritu Das",
        age=21,
        gender=Gender.FEMALE,
        
        city="Kolkata",
        state="West Bengal",
        region="East",
        
        education=Education.GRADUATE,
        economic_status=EconomicStatus.LOWER_MIDDLE,
        occupation="College student",
        
        medical_condition="acute_appendicitis",  # Emergency
        previous_medical_experience="College health center, very limited",
        health_anxiety_level=90,
        trust_in_technology=70,
        family_influence=85,
        
        communication_style=CommunicationStyle(
            english_proficiency="intermediate",
            local_language_influence=50,
            typing_speed="fast",
            vocabulary="moderate",
            sentence_structure="moderate",
            cultural_expressions=["didi", "bhalo", "ektu", "problem", "family ke bolo"],
            tech_comfort="high"
        ),
        
        primary_emotion=EmotionalState.FEARFUL,
        stress_response="Panics easily, wants to call family immediately, very scared"
    ),
    
    "bangalore_professional": PatientProfile(
        patient_id="bangalore_professional",
        name="Deepika Rao",
        age=28,
        gender=Gender.FEMALE,
        
        city="Bangalore",
        state="Karnataka",
        region="South",
        
        education=Education.POSTGRADUATE,
        economic_status=EconomicStatus.UPPER_MIDDLE,
        occupation="Marketing Manager",
        
        medical_condition="breast_lump_concern",  # Routine consultation
        previous_medical_experience="Regular health checkups, corporate health programs",
        health_anxiety_level=70,
        trust_in_technology=80,
        family_influence=50,
        
        communication_style=CommunicationStyle(
            english_proficiency="advanced",
            local_language_influence=25,
            typing_speed="fast",
            vocabulary="complex",
            sentence_structure="complex",
            cultural_expressions=["actually", "you know", "right", "exactly"],
            tech_comfort="high"
        ),
        
        primary_emotion=EmotionalState.WORRIED,
        stress_response="Very articulate about concerns, asks direct questions, research-oriented"
    ),
    
    "hyderabad_elderly": PatientProfile(
        patient_id="hyderabad_elderly",
        name="Lakshmi Devi",
        age=58,
        gender=Gender.FEMALE,
        
        city="Hyderabad",
        state="Telangana",
        region="South",
        
        education=Education.PRIMARY,
        economic_status=EconomicStatus.LOW,
        occupation="Domestic worker",
        
        medical_condition="recurring_uti",  # Routine consultation
        previous_medical_experience="Government hospital visits, local clinic",
        health_anxiety_level=95,
        trust_in_technology=15,
        family_influence=95,
        
        communication_style=CommunicationStyle(
            english_proficiency="basic",
            local_language_influence=80,
            typing_speed="slow",
            vocabulary="simple",
            sentence_structure="basic",
            cultural_expressions=["amma", "ayya", "kastam", "paisa problem", "doctor garu"],
            tech_comfort="low"
        ),
        
        primary_emotion=EmotionalState.CONFUSED,
        stress_response="Very hesitant, worried about costs, needs simple explanations"
    ),
    
    "pune_working_male": PatientProfile(
        patient_id="pune_working_male",
        name="Rohit Patil",
        age=42,
        gender=Gender.MALE,
        
        city="Pune",
        state="Maharashtra",
        region="West",
        
        education=Education.GRADUATE,
        economic_status=EconomicStatus.MIDDLE,
        occupation="Factory supervisor",
        
        medical_condition="tension_headache",  # Self-care
        previous_medical_experience="Occasional clinic visits, self-medication",
        health_anxiety_level=40,
        trust_in_technology=55,
        family_influence=60,
        
        communication_style=CommunicationStyle(
            english_proficiency="intermediate",
            local_language_influence=45,
            typing_speed="moderate",
            vocabulary="moderate",
            sentence_structure="moderate",
            cultural_expressions=["saheb", "kaam", "tension", "ghar", "paisa"],
            tech_comfort="medium"
        ),
        
        primary_emotion=EmotionalState.CALM,
        stress_response="Practical approach, wants quick solutions, cost-conscious"
    ),
    
    "jaipur_rural_migrant": PatientProfile(
        patient_id="jaipur_rural_migrant",
        name="Manish Kumar",
        age=35,
        gender=Gender.MALE,
        
        city="Jaipur",
        state="Rajasthan",
        region="North",
        
        education=Education.PRIMARY,
        economic_status=EconomicStatus.LOW,
        occupation="Construction worker",
        
        medical_condition="gastroenteritis_viral",  # Self-care
        previous_medical_experience="Village doctor, government hospital emergency visits",
        health_anxiety_level=80,
        trust_in_technology=30,
        family_influence=85,
        
        communication_style=CommunicationStyle(
            english_proficiency="basic",
            local_language_influence=75,
            typing_speed="slow",
            vocabulary="simple",
            sentence_structure="basic",
            cultural_expressions=["sahab", "bimari", "dawa", "ghar jana", "paisa nahi"],
            tech_comfort="low"
        ),
        
        primary_emotion=EmotionalState.WORRIED,
        stress_response="Very concerned about work loss, family support, prefers familiar treatments"
    )
}

def get_patient_profile(patient_id: str) -> Optional[PatientProfile]:
    """Get patient profile by ID"""
    return PATIENT_PROFILES.get(patient_id)

def get_all_patient_ids() -> List[str]:
    """Get all available patient IDs"""
    return list(PATIENT_PROFILES.keys())

def get_patients_by_city(city: str) -> List[PatientProfile]:
    """Get all patients from specific city"""
    return [profile for profile in PATIENT_PROFILES.values() if profile.city.lower() == city.lower()]

def get_patients_by_economic_status(status: EconomicStatus) -> List[PatientProfile]:
    """Get patients by economic status"""
    return [profile for profile in PATIENT_PROFILES.values() if profile.economic_status == status]

def get_patients_by_condition_outcome(expected_outcome: str) -> List[PatientProfile]:
    """Get patients whose conditions have specific expected outcome"""
    from medical_conditions import get_condition
    
    matching_patients = []
    for profile in PATIENT_PROFILES.values():
        condition = get_condition(profile.medical_condition)
        if condition and condition.expected_outcome.value == expected_outcome:
            matching_patients.append(profile)
    
    return matching_patients

def get_diverse_patient_sample(count: int = 4) -> List[str]:
    """Get diverse sample of patient IDs for testing"""
    
    all_ids = get_all_patient_ids()
    
    if count >= len(all_ids):
        return all_ids
    
    # Try to get diverse sample across:
    # - Different cities/regions
    # - Different economic statuses  
    # - Different medical outcomes
    # - Different age groups
    # - Different genders
    
    diverse_sample = []
    used_cities = set()
    used_economic_statuses = set()
    used_conditions = set()
    
    # First, try to get one from each major category
    for profile in PATIENT_PROFILES.values():
        if len(diverse_sample) >= count:
            break
            
        # Check diversity criteria
        city_diverse = profile.city not in used_cities
        economic_diverse = profile.economic_status not in used_economic_statuses
        condition_diverse = profile.medical_condition not in used_conditions
        
        # Add if meets diversity criteria
        if city_diverse or economic_diverse or condition_diverse:
            diverse_sample.append(profile.patient_id)
            used_cities.add(profile.city)
            used_economic_statuses.add(profile.economic_status)
            used_conditions.add(profile.medical_condition)
    
    # If we need more, add randomly from remaining
    remaining_ids = [pid for pid in all_ids if pid not in diverse_sample]
    if len(diverse_sample) < count and remaining_ids:
        additional_needed = count - len(diverse_sample)
        diverse_sample.extend(random.sample(remaining_ids, min(additional_needed, len(remaining_ids))))
    
    return diverse_sample[:count]

def get_emergency_patients() -> List[PatientProfile]:
    """Get all patients with emergency conditions"""
    return get_patients_by_condition_outcome("emergency_route_to_doctor")

def get_routine_patients() -> List[PatientProfile]:
    """Get all patients with routine consultation needs"""
    return get_patients_by_condition_outcome("routine_doctor_consultation")

def get_self_care_patients() -> List[PatientProfile]:
    """Get all patients with self-care conditions"""
    return get_patients_by_condition_outcome("self_care_advice")

def validate_patient_profiles() -> Dict[str, Any]:
    """Validate patient profiles for completeness and diversity"""
    
    validation_results = {
        "total_profiles": len(PATIENT_PROFILES),
        "geographic_diversity": {},
        "economic_diversity": {},
        "age_distribution": {},
        "gender_distribution": {},
        "condition_outcome_distribution": {},
        "communication_diversity": {},
        "issues": []
    }
    
    # Geographic diversity
    cities = {}
    regions = {}
    for profile in PATIENT_PROFILES.values():
        cities[profile.city] = cities.get(profile.city, 0) + 1
        regions[profile.region] = regions.get(profile.region, 0) + 1
    
    validation_results["geographic_diversity"] = {
        "cities": cities,
        "regions": regions
    }
    
    # Economic diversity
    economic_dist = {}
    for profile in PATIENT_PROFILES.values():
        economic_dist[profile.economic_status.value] = economic_dist.get(profile.economic_status.value, 0) + 1
    validation_results["economic_diversity"] = economic_dist
    
    # Age distribution
    age_groups = {"20-30": 0, "31-40": 0, "41-50": 0, "51-60": 0, "60+": 0}
    for profile in PATIENT_PROFILES.values():
        if profile.age <= 30:
            age_groups["20-30"] += 1
        elif profile.age <= 40:
            age_groups["31-40"] += 1
        elif profile.age <= 50:
            age_groups["41-50"] += 1
        elif profile.age <= 60:
            age_groups["51-60"] += 1
        else:
            age_groups["60+"] += 1
    validation_results["age_distribution"] = age_groups
    
    # Gender distribution
    gender_dist = {}
    for profile in PATIENT_PROFILES.values():
        gender_dist[profile.gender.value] = gender_dist.get(profile.gender.value, 0) + 1
    validation_results["gender_distribution"] = gender_dist
    
    # Condition outcome distribution
    from medical_conditions import get_condition
    outcome_dist = {}
    for profile in PATIENT_PROFILES.values():
        condition = get_condition(profile.medical_condition)
        if condition:
            outcome = condition.expected_outcome.value
            outcome_dist[outcome] = outcome_dist.get(outcome, 0) + 1
    validation_results["condition_outcome_distribution"] = outcome_dist
    
    # Communication diversity
    english_levels = {}
    tech_comfort_levels = {}
    for profile in PATIENT_PROFILES.values():
        eng_level = profile.communication_style.english_proficiency
        english_levels[eng_level] = english_levels.get(eng_level, 0) + 1
        
        tech_level = profile.communication_style.tech_comfort
        tech_comfort_levels[tech_level] = tech_comfort_levels.get(tech_level, 0) + 1
    
    validation_results["communication_diversity"] = {
        "english_proficiency": english_levels,
        "tech_comfort": tech_comfort_levels
    }
    
    # Validation checks
    if len(PATIENT_PROFILES) < 6:
        validation_results["issues"].append("Need at least 6 patient profiles for comprehensive testing")
    
    if len(regions) < 3:
        validation_results["issues"].append("Need patients from at least 3 different regions")
    
    emergency_count = outcome_dist.get("emergency_route_to_doctor", 0)
    if emergency_count < 2:
        validation_results["issues"].append("Need at least 2 emergency condition patients")
    
    routine_count = outcome_dist.get("routine_doctor_consultation", 0)
    if routine_count < 2:
        validation_results["issues"].append("Need at least 2 routine consultation patients")
    
    self_care_count = outcome_dist.get("self_care_advice", 0)
    if self_care_count < 2:
        validation_results["issues"].append("Need at least 2 self-care patients")
    
    basic_english_count = english_levels.get("basic", 0)
    if basic_english_count < 2:
        validation_results["issues"].append("Need at least 2 patients with basic English proficiency")
    
    return validation_results

def get_test_patient_combinations() -> List[Dict[str, Any]]:
    """Get strategic patient combinations for testing"""
    
    combinations = [
        {
            "name": "Emergency Detection Mix",
            "description": "Mix of emergency and non-emergency to test detection accuracy",
            "patient_ids": [
                "mumbai_tech_male",      # Emergency (MI)
                "kolkata_student",       # Emergency (appendicitis)
                "pune_working_male",     # Self-care (headache)
                "delhi_housewife"        # Routine (postpartum depression)
            ]
        },
        {
            "name": "Communication Diversity",
            "description": "Different English proficiency and tech comfort levels",
            "patient_ids": [
                "bangalore_professional",  # Advanced English, high tech
                "chennai_senior",          # Basic English, low tech
                "hyderabad_elderly",       # Basic English, very low tech
                "jaipur_rural_migrant"     # Basic English, low tech
            ]
        },
        {
            "name": "Socioeconomic Spectrum",
            "description": "Different economic backgrounds and health anxiety levels",
            "patient_ids": [
                "mumbai_tech_male",       # Upper middle, moderate anxiety
                "delhi_housewife",        # Middle, high anxiety
                "hyderabad_elderly",      # Low, very high anxiety
                "bangalore_professional"  # Upper middle, high anxiety
            ]
        },
        {
            "name": "Age and Gender Mix",
            "description": "Diverse age groups and gender representation",
            "patient_ids": [
                "kolkata_student",       # Young female
                "mumbai_tech_male",      # Young male
                "pune_working_male",     # Middle-aged male
                "hyderabad_elderly"      # Elderly female
            ]
        }
    ]
    
    return combinations
