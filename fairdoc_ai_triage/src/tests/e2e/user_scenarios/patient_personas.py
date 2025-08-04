# fairdoc_ai_triage/src/tests/e2e/user_scenarios/patient_personas.py
"""
Patient Persona Lego Blocks for E2E Testing
Creates diverse Indian patient profiles with realistic socioeconomic backgrounds
"""

import dspy
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import random

class City(str, Enum):
    MUMBAI = "mumbai"
    DELHI = "delhi" 
    BANGALORE = "bangalore"
    CHENNAI = "chennai"
    KOLKATA = "kolkata"
    HYDERABAD = "hyderabad"
    PUNE = "pune"
    AHMEDABAD = "ahmedabad"

class EconomicStrength(str, Enum):
    LOWER = "lower"           # Daily wage, minimal education
    LOWER_MIDDLE = "lower_middle"  # Small business, moderate education
    UPPER_MIDDLE = "upper_middle"  # Professional, good education
    UPPER = "upper"           # Business owner, excellent education

class Mood(str, Enum):
    CALM = "calm"
    ANXIOUS = "anxious"
    AFRAID = "afraid"
    CONFUSED = "confused"
    WORRIED = "worried"
    DISBELIEF = "disbelief"
    SAD = "sad"
    PANICKED = "panicked"

class EnglishProficiency(str, Enum):
    BASIC = "basic"           # Local words mixed with English
    INTERMEDIATE = "intermediate"  # Good but with local grammar patterns
    ADVANCED = "advanced"     # Fluent Indian English

@dataclass
class PatientPersona:
    """Complete patient persona with all characteristics"""
    name: str
    age: int
    gender: str
    city: City
    economic_strength: EconomicStrength
    mood: Mood
    english_proficiency: EnglishProficiency
    medical_condition: str
    expected_outcome: str
    communication_style: Dict[str, Any]
    
    def get_typing_pattern(self) -> Dict[str, Any]:
        """Get typing patterns based on mood and background"""
        patterns = {
            "pause_range": (1, 3),   # seconds between messages
            "typo_rate": 0.02,       # percentage of typos
            "repeat_rate": 0.0,      # percentage of repeated words
            "abbreviation_use": 0.1  # percentage of abbreviations
        }
        
        # Modify based on mood
        if self.mood == Mood.ANXIOUS:
            patterns["typo_rate"] = 0.08
            patterns["repeat_rate"] = 0.15
            patterns["pause_range"] = (0.5, 2)
        elif self.mood == Mood.AFRAID:
            patterns["typo_rate"] = 0.12
            patterns["repeat_rate"] = 0.20
            patterns["pause_range"] = (2, 5)
        elif self.mood == Mood.PANICKED:
            patterns["typo_rate"] = 0.15
            patterns["repeat_rate"] = 0.25
            patterns["pause_range"] = (0.2, 1)
        elif self.mood == Mood.CONFUSED:
            patterns["pause_range"] = (3, 8)
            patterns["repeat_rate"] = 0.10
        
        # Modify based on education/economic strength
        if self.economic_strength == EconomicStrength.LOWER:
            patterns["typo_rate"] += 0.05
            patterns["abbreviation_use"] = 0.05
        elif self.economic_strength == EconomicStrength.UPPER:
            patterns["typo_rate"] = max(0.01, patterns["typo_rate"] - 0.02)
            patterns["abbreviation_use"] = 0.15
            
        return patterns

class PatientPersonaFactory:
    """Factory to create diverse patient personas"""
    
    # City-specific characteristics
    CITY_PROFILES = {
        City.MUMBAI: {
            "local_terms": ["yaar", "boss", "bindaas", "jhatka"],
            "grammar_patterns": ["I am having", "It is paining", "From morning only"],
            "cultural_context": "cosmopolitan, time-conscious, competitive"
        },
        City.DELHI: {
            "local_terms": ["bhai", "paaji", "tension", "jugaad"],
            "grammar_patterns": ["Main hun", "Pain ho raha hai", "Subah se"],
            "cultural_context": "political awareness, direct communication"
        },
        City.BANGALORE: {
            "local_terms": ["guru", "maga", "scene", "tech"],
            "grammar_patterns": ["I am", "Pain is there", "From morning"],
            "cultural_context": "tech-savvy, health-conscious, methodical"
        },
        City.CHENNAI: {
            "local_terms": ["da", "pa", "semma", "gethu"],
            "grammar_patterns": ["I am having", "Pain is coming", "Morning-aa irundhu"],
            "cultural_context": "traditional values, family-oriented, conservative"
        },
        City.KOLKATA: {
            "local_terms": ["dada", "mishti", "bhalo", "ektu"],
            "grammar_patterns": ["Ami achi", "Byatha hocche", "Shokal theke"],
            "cultural_context": "intellectual discussion, emotional expression"
        },
        City.HYDERABAD: {
            "local_terms": ["bhai", "macha", "nakko", "babu"],
            "grammar_patterns": ["Nenu unna", "Pain vastundi", "Udayam nunchi"],
            "cultural_context": "laid-back, tech growth, traditional food habits"
        },
        City.PUNE: {
            "local_terms": ["bhau", "kaka", "kai", "mitra"],
            "grammar_patterns": ["Mi aahe", "Dukhtoy", "Sakal pasun"],
            "cultural_context": "educational hub, young population, cultural mix"
        },
        City.AHMEDABAD: {
            "local_terms": ["bhai", "ben", "maja", "thik"],
            "grammar_patterns": ["Hu chhu", "Dukhe che", "Savare thi"],
            "cultural_context": "business-minded, traditional, vegetarian"
        }
    }
    
    # Names by region and gender
    NAMES = {
        "mumbai": {
            "male": ["Rajesh", "Amit", "Vikram", "Arjun", "Rohit"],
            "female": ["Priya", "Sneha", "Kavya", "Meera", "Anita"]
        },
        "delhi": {
            "male": ["Suresh", "Manish", "Deepak", "Nitin", "Pankaj"],
            "female": ["Sunita", "Rekha", "Pooja", "Neetu", "Seema"]
        },
        "bangalore": {
            "male": ["Suresh", "Ravi", "Kiran", "Prasad", "Arun"],
            "female": ["Lakshmi", "Divya", "Shanti", "Bharati", "Geetha"]
        },
        "chennai": {
            "male": ["Raman", "Kumar", "Selvam", "Murugan", "Arjun"],
            "female": ["Kamala", "Radha", "Sita", "Devi", "Shyamala"]
        },
        "kolkata": {
            "male": ["Subhash", "Amit", "Ranjan", "Tapan", "Partha"],
            "female": ["Shilpa", "Ruma", "Swati", "Monika", "Jayanti"]
        },
        "hyderabad": {
            "male": ["Venkat", "Krishna", "Ramesh", "Naveen", "Srinivas"],
            "female": ["Kavitha", "Madhavi", "Suneetha", "Vasanti", "Jyothi"]
        },
        "pune": {
            "male": ["Sachin", "Rahul", "Amol", "Sandeep", "Nikhil"],
            "female": ["Ashwini", "Shweta", "Archana", "Vaishali", "Manisha"]
        },
        "ahmedabad": {
            "male": ["Mehul", "Kiran", "Jignesh", "Hitesh", "Bhavesh"],
            "female": ["Hiral", "Nisha", "Riddhi", "Foram", "Krupa"]
        }
    }
    
    @classmethod
    def create_patient(cls, 
                      city: City,
                      medical_condition: str,
                      expected_outcome: str,
                      **kwargs) -> PatientPersona:
        """Create a patient persona with specified characteristics"""
        
        # Random characteristics if not specified
        gender = kwargs.get('gender', random.choice(['male', 'female']))
        age = kwargs.get('age', random.randint(25, 65))
        economic_strength = kwargs.get('economic_strength', random.choice(list(EconomicStrength)))
        mood = kwargs.get('mood', random.choice(list(Mood)))
        
        # English proficiency based on economic strength
        proficiency_map = {
            EconomicStrength.LOWER: EnglishProficiency.BASIC,
            EconomicStrength.LOWER_MIDDLE: random.choice([EnglishProficiency.BASIC, EnglishProficiency.INTERMEDIATE]),
            EconomicStrength.UPPER_MIDDLE: EnglishProficiency.INTERMEDIATE,
            EconomicStrength.UPPER: EnglishProficiency.ADVANCED
        }
        english_proficiency = kwargs.get('english_proficiency', proficiency_map[economic_strength])
        
        # Select name based on city and gender
        name = random.choice(cls.NAMES[city.value][gender])
        
        # Communication style based on all factors
        communication_style = cls._get_communication_style(
            city, economic_strength, english_proficiency, mood
        )
        
        return PatientPersona(
            name=name,
            age=age,
            gender=gender,
            city=city,
            economic_strength=economic_strength,
            mood=mood,
            english_proficiency=english_proficiency,
            medical_condition=medical_condition,
            expected_outcome=expected_outcome,
            communication_style=communication_style
        )
    
    @classmethod
    def _get_communication_style(cls, 
                                city: City,
                                economic_strength: EconomicStrength,
                                proficiency: EnglishProficiency,
                                mood: Mood) -> Dict[str, Any]:
        """Generate communication style based on characteristics"""
        
        city_profile = cls.CITY_PROFILES[city]
        
        style = {
            "local_terms": city_profile["local_terms"],
            "grammar_patterns": city_profile["grammar_patterns"],
            "cultural_context": city_profile["cultural_context"],
            "formality_level": "low",
            "verbosity": "medium",
            "directness": "medium"
        }
        
        # Adjust based on economic strength
        if economic_strength == EconomicStrength.UPPER:
            style["formality_level"] = "high"
            style["verbosity"] = "high"
        elif economic_strength == EconomicStrength.LOWER:
            style["formality_level"] = "low"
            style["verbosity"] = "low"
            style["directness"] = "high"
        
        # Adjust based on English proficiency
        if proficiency == EnglishProficiency.BASIC:
            style["local_term_frequency"] = 0.3
            style["grammar_error_rate"] = 0.2
        elif proficiency == EnglishProficiency.INTERMEDIATE:
            style["local_term_frequency"] = 0.15
            style["grammar_error_rate"] = 0.1
        else:  # ADVANCED
            style["local_term_frequency"] = 0.05
            style["grammar_error_rate"] = 0.02
        
        # Adjust based on mood
        if mood in [Mood.ANXIOUS, Mood.AFRAID, Mood.PANICKED]:
            style["verbosity"] = "high"
            style["directness"] = "high"
        elif mood == Mood.CONFUSED:
            style["verbosity"] = "low"
            style["directness"] = "low"
        
        return style

    @classmethod
    def create_diverse_patient_set(cls) -> List[PatientPersona]:
        """Create 8 diverse patients from different cities with various conditions"""
        
        patient_configs = [
            {
                "city": City.MUMBAI,
                "medical_condition": "chest_pain_emergency",
                "expected_outcome": "emergency_route_to_doctor",
                "gender": "male",
                "age": 52,
                "economic_strength": EconomicStrength.UPPER_MIDDLE,
                "mood": Mood.ANXIOUS
            },
            {
                "city": City.DELHI, 
                "medical_condition": "tension_headache",
                "expected_outcome": "self_care_advice",
                "gender": "female",
                "age": 34,
                "economic_strength": EconomicStrength.LOWER_MIDDLE,
                "mood": Mood.WORRIED
            },
            {
                "city": City.BANGALORE,
                "medical_condition": "respiratory_distress",
                "expected_outcome": "emergency_route_to_doctor", 
                "gender": "female",
                "age": 42,
                "economic_strength": EconomicStrength.UPPER,
                "mood": Mood.AFRAID
            },
            {
                "city": City.CHENNAI,
                "medical_condition": "abdominal_pain",
                "expected_outcome": "routine_doctor_consultation",
                "gender": "male",
                "age": 28,
                "economic_strength": EconomicStrength.LOWER,
                "mood": Mood.CONFUSED
            },
            {
                "city": City.KOLKATA,
                "medical_condition": "diabetes_management",
                "expected_outcome": "routine_doctor_consultation",
                "gender": "male",
                "age": 58,
                "economic_strength": EconomicStrength.LOWER_MIDDLE,
                "mood": Mood.CALM
            },
            {
                "city": City.HYDERABAD,
                "medical_condition": "skin_rash",
                "expected_outcome": "self_care_advice",
                "gender": "female",
                "age": 29,
                "economic_strength": EconomicStrength.UPPER_MIDDLE,
                "mood": Mood.DISBELIEF
            },
            {
                "city": City.PUNE,
                "medical_condition": "migraine_severe",
                "expected_outcome": "routine_doctor_consultation",
                "gender": "female",
                "age": 31,
                "economic_strength": EconomicStrength.UPPER,
                "mood": Mood.SAD
            },
            {
                "city": City.AHMEDABAD,
                "medical_condition": "fever_dengue_suspect",
                "expected_outcome": "need_more_questions",
                "gender": "male",
                "age": 26,
                "economic_strength": EconomicStrength.LOWER,
                "mood": Mood.PANICKED
            }
        ]
        
        patients = []
        for config in patient_configs:
            patient = cls.create_patient(**config)
            patients.append(patient)
            
        return patients
