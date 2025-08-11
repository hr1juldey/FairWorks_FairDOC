# tests/training_poc/patient_generator.py
"""
Dynamic Patient Persona Generator - POC Implementation

Implements a simplified version of the comprehensive persona modeling system
for rapid prototyping and testing. Based on persona_system_spec.md but optimized
for 3-4 hour POC timeline.

Key Features:
- Generates diverse, realistic patient personas
- Implements basic correlation patterns
- Creates medical training scenarios
- Fast generation for testing purposes
"""

import random
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Simplified data structures for POC
@dataclass
class PatientPersona:
    """Simplified patient persona for POC"""
    id: str
    age_group: int  # 1-5 scale
    location: int   # 1-5 scale (rural to urban)
    symptom_severity: int  # 1-5 scale
    education: int  # 1-5 scale
    tech_savviness: int  # 1-5 scale
    emotion_state: int  # 1-5 scale
    communication_style: str
    primary_language: str
    medical_history: List[str]
    current_symptoms: Dict[str, str]
    expected_outcome: str  # emergency, routine, self_care
    
    def to_training_example(self) -> Dict[str, Any]:
        """Convert persona to training example"""
        # Create realistic patient message
        severity_words = {
            1: "mild", 2: "moderate", 3: "concerning", 4: "severe", 5: "unbearable"
        }
        
        emotion_words = {
            1: "calm", 2: "slightly worried", 3: "concerned", 4: "anxious", 5: "panicked"
        }
        
        # Build symptom description
        symptom_desc = []
        for symptom, intensity in self.current_symptoms.items():
            symptom_desc.append(f"{severity_words.get(self.symptom_severity, 'moderate')} {symptom}")
        
        patient_message = f"I'm feeling {emotion_words.get(self.emotion_state, 'concerned')} because I have {', '.join(symptom_desc)}. "
        
        # Add communication style variation
        if self.communication_style == "hesitant":
            patient_message += "I'm not sure if this is serious but..."
        elif self.communication_style == "direct":
            patient_message += "I need to know what's wrong."
        elif self.communication_style == "detailed":
            patient_message += f"This started yesterday and I also have a history of {', '.join(self.medical_history[:2]) if self.medical_history else 'no major issues'}."
        
        return {
            "user_message": patient_message,
            "expected_outcome": self.expected_outcome,
            "persona_metadata": {
                "age_group": self.age_group,
                "location": self.location,
                "education": self.education,
                "severity": self.symptom_severity
            }
        }

@dataclass 
class PatientScenario:
    """Medical scenario derived from persona"""
    scenario_id: str
    persona: PatientPersona
    conversation_turns: List[Dict[str, str]]
    complexity_level: str  # simple, moderate, complex
    expected_turn_count: int
    
    @classmethod
    def from_persona(cls, persona: PatientPersona) -> 'PatientScenario':
        """Create scenario from persona"""
        # Determine complexity based on persona attributes
        complexity_score = (persona.symptom_severity + persona.emotion_state) / 2
        
        if complexity_score <= 2:
            complexity = "simple"
            expected_turns = 1
        elif complexity_score <= 4:
            complexity = "moderate" 
            expected_turns = 2
        else:
            complexity = "complex"
            expected_turns = 3
            
        # Generate initial conversation turn
        initial_turn = persona.to_training_example()
        
        return cls(
            scenario_id=f"scenario_{persona.id}",
            persona=persona,
            conversation_turns=[initial_turn],
            complexity_level=complexity,
            expected_turn_count=expected_turns
        )

class PersonaGenerator:
    """Generates diverse patient personas for training"""
    
    def __init__(self):
        self.correlation_patterns = {
            # Age patterns
            ("age_group", "symptom_severity"): 0.4,     # Older people tend to have more severe symptoms
            ("age_group", "tech_savviness"): -0.3,      # Older people less tech savvy
            
            # Socioeconomic patterns  
            ("education", "tech_savviness"): 0.5,       # More educated = more tech savvy
            ("location", "tech_savviness"): 0.4,        # Urban = more tech savvy
            ("education", "communication_style"): 0.3,  # More educated = more articulate
            
            # Medical patterns
            ("symptom_severity", "emotion_state"): 0.6,  # More severe = more emotional
            ("age_group", "medical_history"): 0.5,       # Older = more medical history
        }
        
        self.symptom_templates = {
            "chest_pain": {
                "variations": ["chest pain", "chest discomfort", "pressure in chest"],
                "severities": {1: "mild chest discomfort", 5: "crushing chest pain"},
                "likely_outcome": {1: "self_care", 2: "self_care", 3: "routine", 4: "emergency", 5: "emergency"}
            },
            "headache": {
                "variations": ["headache", "head pain", "migraine"],
                "severities": {1: "mild headache", 5: "worst headache ever"},
                "likely_outcome": {1: "self_care", 2: "self_care", 3: "routine", 4: "routine", 5: "emergency"}
            },
            "fever": {
                "variations": ["fever", "high temperature", "feeling hot"],
                "severities": {1: "low fever", 5: "very high fever"},
                "likely_outcome": {1: "self_care", 2: "routine", 3: "routine", 4: "routine", 5: "emergency"}
            },
            "abdominal_pain": {
                "variations": ["stomach pain", "belly pain", "abdominal pain"],
                "severities": {1: "mild stomach ache", 5: "severe abdominal pain"},
                "likely_outcome": {1: "self_care", 2: "routine", 3: "routine", 4: "routine", 5: "emergency"}
            },
            "breathing_difficulty": {
                "variations": ["trouble breathing", "shortness of breath", "can't breathe"],
                "severities": {1: "mild shortness of breath", 5: "severe breathing difficulty"},
                "likely_outcome": {1: "routine", 2: "routine", 3: "emergency", 4: "emergency", 5: "emergency"}
            }
        }
        
        self.communication_styles = ["hesitant", "direct", "detailed", "anxious", "calm"]
        self.languages = ["English", "Hindi", "Tamil", "Telugu", "Bengali", "Marathi"]
        self.medical_histories = [
            [], ["diabetes"], ["hypertension"], ["heart disease"], ["asthma"],
            ["diabetes", "hypertension"], ["heart disease", "diabetes"],
            ["previous surgery"], ["medication allergies"], ["family history of heart disease"]
        ]
    
    async def generate_diverse_batch(self, count: int, correlation_strength: float = 0.8,
                                   include_edge_cases: bool = True) -> List[PatientPersona]:
        """Generate diverse batch of personas"""
        personas = []
        
        # Generate correlated base characteristics
        base_chars = self._generate_correlated_characteristics(count, correlation_strength)
        
        for i in range(count):
            persona = self._create_persona_from_base(base_chars[i], i)
            personas.append(persona)
        
        # Add edge cases if requested
        if include_edge_cases:
            edge_cases = self._generate_edge_cases(max(5, count // 20))
            personas.extend(edge_cases)
        
        return personas
    
    def _generate_correlated_characteristics(self, count: int, strength: float) -> List[Dict[str, int]]:
        """Generate correlated characteristics using simplified approach"""
        characteristics = []
        
        for _ in range(count):
            # Start with random base values
            age = random.randint(1, 5)
            location = random.randint(1, 5)
            education = random.randint(1, 5)
            
            # Apply correlations
            tech_savviness = max(1, min(5, int(
                education * 0.4 + location * 0.3 - age * 0.2 + random.normal(0, 0.5)
            )))
            
            symptom_severity = max(1, min(5, int(
                age * 0.3 + random.normal(2.5, 1)
            )))
            
            emotion_state = max(1, min(5, int(
                symptom_severity * 0.6 + random.normal(0, 0.5)
            )))
            
            characteristics.append({
                "age_group": age,
                "location": location,
                "education": education,
                "tech_savviness": tech_savviness,
                "symptom_severity": symptom_severity,
                "emotion_state": emotion_state
            })
        
        return characteristics
    
    def _create_persona_from_base(self, base_chars: Dict[str, int], index: int) -> PatientPersona:
        """Create full persona from base characteristics"""
        
        # Select primary symptom
        symptom_type = random.choice(list(self.symptom_templates.keys()))
        symptom_data = self.symptom_templates[symptom_type]
        
        # Create symptom description
        severity = base_chars["symptom_severity"]
        symptom_desc = random.choice(symptom_data["variations"])
        intensity_modifier = random.choice(["", "persistent ", "recurring ", "sharp ", "dull "])
        
        current_symptoms = {
            symptom_desc: f"{intensity_modifier}{symptom_desc}"
        }
        
        # Add secondary symptoms occasionally
        if random.random() < 0.3:
            secondary = random.choice(["nausea", "dizziness", "fatigue", "sweating"])
            current_symptoms[secondary] = secondary
        
        # Determine expected outcome based on symptom + severity
        outcome_mapping = symptom_data["likely_outcome"]
        base_outcome = outcome_mapping.get(severity, "routine")
        
        # Add some randomness and correlation adjustments
        if base_chars["emotion_state"] >= 4 and severity >= 4:
            expected_outcome = "emergency"
        elif severity <= 2 and base_chars["emotion_state"] <= 2:
            expected_outcome = "self_care"
        else:
            expected_outcome = base_outcome
        
        # Select communication style based on education and emotion
        if base_chars["education"] >= 4:
            comm_style = random.choice(["direct", "detailed"])
        elif base_chars["emotion_state"] >= 4:
            comm_style = random.choice(["anxious", "hesitant"])
        else:
            comm_style = random.choice(self.communication_styles)
        
        # Select language based on location (simplified)
        if base_chars["location"] <= 2:  # Rural
            primary_language = random.choice(["Hindi", "Tamil", "Telugu", "Bengali"])
        else:  # Urban
            primary_language = random.choice(["English", "Hindi"])
        
        # Select medical history based on age
        if base_chars["age_group"] >= 4:
            medical_history = random.choice(self.medical_histories[-5:])  # More complex histories
        elif base_chars["age_group"] <= 2:
            medical_history = random.choice(self.medical_histories[:3])   # Simpler histories
        else:
            medical_history = random.choice(self.medical_histories)
        
        return PatientPersona(
            id=f"persona_{index}_{uuid.uuid4().hex[:8]}",
            age_group=base_chars["age_group"],
            location=base_chars["location"],
            symptom_severity=base_chars["symptom_severity"],
            education=base_chars["education"],
            tech_savviness=base_chars["tech_savviness"],
            emotion_state=base_chars["emotion_state"],
            communication_style=comm_style,
            primary_language=primary_language,
            medical_history=medical_history,
            current_symptoms=current_symptoms,
            expected_outcome=expected_outcome
        )
    
    def _generate_edge_cases(self, count: int) -> List[PatientPersona]:
        """Generate edge case personas for testing robustness"""
        edge_cases = []
        
        edge_case_templates = [
            # Very young, very severe
            {"age_group": 1, "symptom_severity": 5, "emotion_state": 5, "case_type": "pediatric_emergency"},
            # Very old, mild symptoms but high risk
            {"age_group": 5, "symptom_severity": 2, "emotion_state": 3, "case_type": "geriatric_risk"},
            # High tech, low education (inconsistent)
            {"age_group": 3, "tech_savviness": 5, "education": 1, "case_type": "tech_paradox"},
            # Rural emergency
            {"location": 1, "symptom_severity": 5, "tech_savviness": 1, "case_type": "rural_emergency"},
            # Urban anxiety (mild symptoms, high emotion)
            {"location": 5, "symptom_severity": 2, "emotion_state": 5, "case_type": "urban_anxiety"}
        ]
        
        for i, template in enumerate(edge_case_templates[:count]):
            # Fill in missing values with defaults
            full_template = {
                "age_group": template.get("age_group", 3),
                "location": template.get("location", 3),
                "education": template.get("education", 3),
                "tech_savviness": template.get("tech_savviness", 3),
                "symptom_severity": template.get("symptom_severity", 3),
                "emotion_state": template.get("emotion_state", 3)
            }
            
            persona = self._create_persona_from_base(full_template, f"edge_{i}")
            # Mark as edge case
            persona.id = f"edge_case_{i}_{persona.id}"
            edge_cases.append(persona)
        
        return edge_cases
    
    def analyze_persona_diversity(self, personas: List[PatientPersona]) -> Dict[str, Any]:
        """Analyze diversity of generated personas"""
        if not personas:
            return {"error": "No personas to analyze"}
        
        # Calculate distributions
        age_dist = [p.age_group for p in personas]
        location_dist = [p.location for p in personas]
        severity_dist = [p.symptom_severity for p in personas]
        outcome_dist = [p.expected_outcome for p in personas]
        
        return {
            "total_personas": len(personas),
            "age_distribution": {i: age_dist.count(i) for i in range(1, 6)},
            "location_distribution": {i: location_dist.count(i) for i in range(1, 6)},
            "severity_distribution": {i: severity_dist.count(i) for i in range(1, 6)},
            "outcome_distribution": {
                outcome: outcome_dist.count(outcome) 
                for outcome in ["emergency", "routine", "self_care"]
            },
            "diversity_metrics": {
                "age_entropy": self._calculate_entropy(age_dist),
                "location_entropy": self._calculate_entropy(location_dist),
                "severity_entropy": self._calculate_entropy(severity_dist),
                "outcome_balance": min(outcome_dist.count(o) for o in ["emergency", "routine", "self_care"]) / len(personas)
            }
        }
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution"""
        from collections import Counter
        counts = Counter(values)
        total = len(values)
        entropy = 0
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy

# Quick test function
async def test_persona_generation():
    """Test persona generation"""
    generator = PersonaGenerator()
    
    print("🧪 Testing Persona Generation...")
    personas = await generator.generate_diverse_batch(count=20, include_edge_cases=True)
    
    print(f"✅ Generated {len(personas)} personas")
    
    # Analyze diversity
    analysis = generator.analyze_persona_diversity(personas)
    print("📊 Diversity Analysis:\n")
    print(f"   Age distribution: {analysis['age_distribution']}")
    print(f"   Outcome distribution: {analysis['outcome_distribution']}")
    print(f"   Age entropy: {analysis['diversity_metrics']['age_entropy']:.2f}")
    print(f"   Outcome balance: {analysis['diversity_metrics']['outcome_balance']:.2f}")
    
    # Show sample personas
    print("\n📋 Sample Personas:")
    for i, persona in enumerate(personas[:3]):
        example = persona.to_training_example()
        print(f"   {i + 1}. {example['user_message'][:100]}...")
        print(f"      Expected: {example['expected_outcome']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_persona_generation())
