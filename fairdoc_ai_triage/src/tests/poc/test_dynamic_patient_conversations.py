# src/tests/poc/test_dynamic_patient_conversations.py
"""
Advanced Dynamic Multi-Turn Patient Conversation Testing System

Uses DSPy signatures for everything:
- Dynamic disease generation based on gene profiles
- Realistic communication style generation
- Progressive symptom development
- Behavioral modeling with epistatic interactions

Generates 100s of unique patients with realistic conversations every run.
"""

import pytest
import asyncio
import json
import time
import dspy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from uuid import uuid4
import numpy as np
import pandas as pd
from pathlib import Path

from src.app2.services.chat.chat_orchestrator import ChatOrchestrator
from src.app2.core.config_v2 import settings_v2
from tests.poc.data.patient_genes import (
    PATIENT_GENE_EXPRESSIONS,
    EPISTATIC_INTERACTIONS,
    REGIONAL_PATTERNS,
    WEALTH_PSYCHOLOGY_CATEGORIES,
    DISEASE_RISK_MULTIPLIERS
)

# ==================== DSPy SIGNATURES ====================

class DiseaseGenerationSignature(dspy.Signature):
    """Generate realistic medical conditions based on patient gene profile"""
    gene_profile = dspy.InputField(desc="Complete 20-factor patient gene profile")
    regional_risk_factors = dspy.InputField(desc="Regional disease risk multipliers")
    wealth_psychology = dspy.InputField(desc="Wealth psychology category effects")
    
    primary_condition = dspy.OutputField(desc="Primary medical condition")
    severity_level = dspy.OutputField(desc="Severity: mild|moderate|severe|critical")
    symptom_progression = dspy.OutputField(desc="List of symptoms that progress over time")
    risk_factors = dspy.OutputField(desc="Personal risk factors contributing to condition")
    prognosis = dspy.OutputField(desc="Expected disease progression")

class CommunicationStyleSignature(dspy.Signature):
    """Generate realistic communication style based on gene profile and psychology"""
    gene_profile = dspy.InputField(desc="Patient's socio-demographic profile")
    wealth_psychology = dspy.InputField(desc="Wealth psychology category")
    emergency_stress_level = dspy.InputField(desc="Current stress level 1-10")
    cultural_context = dspy.InputField(desc="Cultural and regional context")
    
    communication_style = dspy.OutputField(desc="Overall communication approach")
    language_preference = dspy.OutputField(desc="Preferred language and complexity")
    emotional_expression = dspy.OutputField(desc="How emotions are expressed")
    authority_relationship = dspy.OutputField(desc="Relationship with medical authority")

class InitialMessageSignature(dspy.Signature):
    """Generate realistic initial patient message"""
    symptoms = dspy.InputField(desc="Current symptoms")
    communication_style = dspy.InputField(desc="Patient communication profile")
    cultural_context = dspy.InputField(desc="Cultural and regional background")
    urgency_perception = dspy.InputField(desc="How urgent patient perceives situation")
    
    initial_message = dspy.OutputField(desc="Natural, realistic opening message")
    emotional_tone = dspy.OutputField(desc="Underlying emotional tone")
    information_sharing_level = dspy.OutputField(desc="How much detail shared initially")

class FollowupMessageSignature(dspy.Signature):
    """Generate realistic follow-up messages with symptom progression"""
    conversation_history = dspy.InputField(desc="Previous conversation turns")
    current_symptoms = dspy.InputField(desc="Current symptom state")
    agent_question = dspy.InputField(desc="Last agent question")
    patient_psychology = dspy.InputField(desc="Patient psychological profile")
    symptom_progression = dspy.InputField(desc="How symptoms are evolving")
    
    followup_message = dspy.OutputField(desc="Natural follow-up response")
    symptom_evolution = dspy.OutputField(desc="How symptoms have changed")
    emotional_state = dspy.OutputField(desc="Patient's current emotional state")
    information_revealed = dspy.OutputField(desc="New information patient shares")

class BehavioralPhenotypingSignature(dspy.Signature):
    """Calculate behavioral phenotype from gene interactions"""
    main_effects = dspy.InputField(desc="Main genetic effects (βᵢ × Geneᵢ)")
    epistatic_interactions = dspy.InputField(desc="Gene-gene interactions (γᵢⱼ × Geneᵢ × Geneⱼ)")
    environmental_factors = dspy.InputField(desc="Environmental modifiers")
    
    behavioral_phenotype = dspy.OutputField(desc="Predicted behavior pattern")
    decision_making_style = dspy.OutputField(desc="How patient makes medical decisions")
    family_involvement_level = dspy.OutputField(desc="Degree of family consultation")
    trust_in_healthcare = dspy.OutputField(desc="Trust level in medical system")

# ==================== ENHANCED DATA CLASSES ====================

@dataclass
class GeneticProfile:
    """Complete 20-factor genetic profile with computed interactions"""
    # Core demographics
    age: int
    gender: str
    education_level: str
    income_quintile: int
    ethnicity: str
    state_region: str
    city_tier: str
    
    # Healthcare access
    health_insurance: bool
    distance_to_healthcare: float
    smartphone_access: bool
    internet_literacy: int
    
    # Lifestyle factors
    tobacco_use: str
    alcohol_consumption: str
    physical_activity: int
    occupation: str
    marital_status: str
    
    # Environmental
    residence_type: str
    caste_category: str
    time_of_day: str
    season: str
    
    # Computed interactions
    epistatic_score: float
    behavioral_phenotype_score: float
    wealth_psychology_category: str

@dataclass
class DynamicPatient:
    """Dynamically generated patient with full gene profile"""
    patient_id: str
    genetic_profile: GeneticProfile
    
    # Medical condition (DSPy generated)
    primary_condition: str
    severity_level: str
    symptoms_progression: List[str]
    risk_factors: List[str]
    prognosis: str
    
    # Behavioral traits (DSPy generated)
    communication_style: str
    language_preference: str
    emotional_expression: str
    decision_making_style: str
    family_involvement: float
    trust_in_healthcare: float
    
    # Hidden labels (for evaluation)
    expected_outcome: str
    red_flags_present: List[str]
    optimal_turn_count: int

@dataclass
class ConversationTurn:
    """Enhanced conversation turn with full context"""
    turn_number: int
    timestamp: str
    patient_message: str
    emotional_tone: str
    information_revealed: str
    
    agent_response: Optional[str]
    agent_next_question: Optional[str]
    
    # Medical assessment
    outcome: str
    confidence: int
    confidence_about: str
    red_flags: List[str]
    reasoning: str
    
    # Conversation state
    is_complete: bool
    completion_reason: Optional[str]
    
    # Progression tracking
    symptom_evolution: str
    emotional_state: str

@dataclass
class PatientConversation:
    """Complete patient conversation with enhanced metrics"""
    conversation_id: str
    patient: DynamicPatient
    turns: List[ConversationTurn]
    
    # Performance metrics
    total_duration_seconds: float
    final_outcome: str
    accuracy_score: float
    confidence_progression: List[int]
    red_flag_detection_accuracy: float
    
    # Advanced metrics
    symptom_progression_accuracy: float
    behavioral_consistency_score: float
    communication_authenticity: float

# ==================== DSPy MODULES ====================

class DiseaseGenerator(dspy.Module):
    """Generate realistic diseases using DSPy"""
    
    def __init__(self):
        super().__init__()
        self.disease_predictor = dspy.ChainOfThought(DiseaseGenerationSignature)
    
    def forward(self, genetic_profile: GeneticProfile) -> Dict[str, Any]:
        """Generate disease based on genetic profile and risk factors"""
        
        # Get regional risk factors
        regional_risks = DISEASE_RISK_MULTIPLIERS.get(
            genetic_profile.state_region, 
            DISEASE_RISK_MULTIPLIERS["north"]
        )
        
        # Get wealth psychology effects
        wealth_effects = WEALTH_PSYCHOLOGY_CATEGORIES.get(
            genetic_profile.wealth_psychology_category,
            WEALTH_PSYCHOLOGY_CATEGORIES["aspirational_poor"]
        )
        
        # Format inputs for DSPy
        profile_str = (f"Age: {genetic_profile.age}, Gender: {genetic_profile.gender}, "
                      f"Education: {genetic_profile.education_level}, Region: {genetic_profile.state_region}, "
                      f"Income: Q{genetic_profile.income_quintile}, Insurance: {genetic_profile.health_insurance}")
        
        risk_str = f"CVD risk: {regional_risks.get('cardiovascular_rr', 1.0)}, Diabetes risk: {regional_risks.get('diabetes_rr', 1.0)}"
        wealth_str = f"Category: {genetic_profile.wealth_psychology_category}, Delay factor: {wealth_effects.get('delay_coefficient', 0)}"
        
        result = self.disease_predictor(
            gene_profile=profile_str,
            regional_risk_factors=risk_str,
            wealth_psychology=wealth_str
        )
        
        return {
            "condition": result.primary_condition,
            "severity": result.severity_level,
            "symptoms": self._parse_symptoms(result.symptom_progression),
            "risk_factors": self._parse_risk_factors(result.risk_factors),
            "prognosis": result.prognosis
        }
    
    def _parse_symptoms(self, symptom_text: str) -> List[str]:
        """Parse symptom progression from DSPy output"""
        # Simple parsing - in production, use more sophisticated NLP
        symptoms = [s.strip() for s in symptom_text.split(',')]
        return symptoms[:6]  # Limit to 6 symptoms max
    
    def _parse_risk_factors(self, risk_text: str) -> List[str]:
        """Parse risk factors from DSPy output"""
        factors = [f.strip() for f in risk_text.split(',')]
        return factors[:4]  # Limit to 4 risk factors

class CommunicationStyleGenerator(dspy.Module):
    """Generate realistic communication styles using DSPy"""
    
    def __init__(self):
        super().__init__()
        self.style_predictor = dspy.ChainOfThought(CommunicationStyleSignature)
    
    def forward(self, genetic_profile: GeneticProfile, emergency_level: str) -> Dict[str, Any]:
        """Generate communication style based on profile"""
        
        # Calculate stress level based on emergency
        stress_map = {"mild": 3, "moderate": 5, "severe": 7, "critical": 9}
        stress_level = stress_map.get(emergency_level, 5)
        
        # Get cultural context
        regional_pattern = REGIONAL_PATTERNS.get(genetic_profile.state_region, REGIONAL_PATTERNS["north"])
        wealth_psychology = WEALTH_PSYCHOLOGY_CATEGORIES.get(
            genetic_profile.wealth_psychology_category,
            WEALTH_PSYCHOLOGY_CATEGORIES["aspirational_poor"]
        )
        
        profile_str = (f"Age: {genetic_profile.age}, Education: {genetic_profile.education_level}, "
                      f"Income: Q{genetic_profile.income_quintile}, Region: {genetic_profile.state_region}")
        
        cultural_str = (f"Language: {regional_pattern['language_preference']}, "
                       f"Family consultation: {regional_pattern['family_consultation_weight']}, "
                       f"Decision making: {regional_pattern.get('decision_making', 'individual')}",
                       f"Weath psychology: {wealth_psychology}")
        
        result = self.style_predictor(
            gene_profile=profile_str,
            wealth_psychology=genetic_profile.wealth_psychology_category,
            emergency_stress_level=str(stress_level),
            cultural_context=cultural_str
        )
        
        return {
            "style": result.communication_style,
            "language": result.language_preference,
            "emotional": result.emotional_expression,
            "authority": result.authority_relationship
        }

class MessageGenerator(dspy.Module):
    """Generate realistic patient messages using DSPy"""
    
    def __init__(self):
        super().__init__()
        self.initial_generator = dspy.ChainOfThought(InitialMessageSignature)
        self.followup_generator = dspy.ChainOfThought(FollowupMessageSignature)
    
    def generate_initial_message(self, patient: DynamicPatient) -> Dict[str, str]:
        """Generate initial patient message"""
        
        symptoms_str = ", ".join(patient.symptoms_progression[:2])  # First 2 symptoms
        style_str = f"Style: {patient.communication_style}, Language: {patient.language_preference}"
        cultural_str = f"Region: {patient.genetic_profile.state_region}, Education: {patient.genetic_profile.education_level}"
        urgency_map = {"mild": "low", "moderate": "medium", "severe": "high", "critical": "urgent"}
        urgency = urgency_map.get(patient.severity_level, "medium")
        
        result = self.initial_generator(
            symptoms=symptoms_str,
            communication_style=style_str,
            cultural_context=cultural_str,
            urgency_perception=urgency
        )
        
        return {
            "message": result.initial_message,
            "tone": result.emotional_tone,
            "detail_level": result.information_sharing_level
        }
    
    def generate_followup_message(self, patient: DynamicPatient, turns: List[ConversationTurn], 
                                 turn_number: int) -> Dict[str, str]:
        """Generate follow-up message with symptom progression"""
        
        # Build conversation history
        history_str = ""
        for turn in turns[-3:]:  # Last 3 turns
            history_str += f"Turn {turn.turn_number}: Patient: {turn.patient_message[:100]}... "
            if turn.agent_next_question:
                history_str += f"Agent: {turn.agent_next_question[:100]}... "
        
        # Current symptoms (may have progressed)
        current_symptoms = self._get_current_symptoms(patient, turn_number)
        
        # Patient psychology profile
        psychology_str = (f"Style: {patient.communication_style}, Trust: {patient.trust_in_healthcare}, "
                         f"Family involvement: {patient.family_involvement}")
        
        # Symptom progression pattern
        progression_str = f"Severity: {patient.severity_level}, Progression: {patient.prognosis}"
        
        last_turn = turns[-1] if turns else None
        agent_question = last_turn.agent_next_question if last_turn else "How are you feeling?"
        
        result = self.followup_generator(
            conversation_history=history_str,
            current_symptoms=current_symptoms,
            agent_question=agent_question,
            patient_psychology=psychology_str,
            symptom_progression=progression_str
        )
        
        return {
            "message": result.followup_message,
            "symptoms": result.symptom_evolution,
            "emotion": result.emotional_state,
            "info": result.information_revealed
        }
    
    def _get_current_symptoms(self, patient: DynamicPatient, turn_number: int) -> str:
        """Get current symptoms based on turn and progression"""
        if patient.severity_level in ["severe", "critical"]:
            # Symptoms get worse over time
            symptom_idx = min(turn_number - 1, len(patient.symptoms_progression) - 1)
            return patient.symptoms_progression[symptom_idx]
        else:
            # Symptoms remain stable
            return patient.symptoms_progression[0] if patient.symptoms_progression else "discomfort"

class BehavioralPhenotyper(dspy.Module):
    """Calculate behavioral phenotype using epistatic interactions"""
    
    def __init__(self):
        super().__init__()
        self.phenotype_calculator = dspy.ChainOfThought(BehavioralPhenotypingSignature)
    
    def forward(self, genetic_profile: GeneticProfile) -> Dict[str, Any]:
        """Calculate behavioral phenotype from gene interactions"""
        
        # Calculate main effects (βᵢ × Geneᵢ)
        main_effects = self._calculate_main_effects(genetic_profile)
        
        # Calculate epistatic interactions (γᵢⱼ × Geneᵢ × Geneⱼ)
        epistatic_effects = self._calculate_epistatic_interactions(genetic_profile)
        
        # Environmental factors
        env_factors = self._get_environmental_factors(genetic_profile)
        
        result = self.phenotype_calculator(
            main_effects=f"Education effect: {main_effects.get('education', 0):.2f}, Age effect: {main_effects.get('age', 0):.2f}",
            epistatic_interactions=f"Education×Income: {epistatic_effects.get('edu_income', 0):.2f}, Tech×Literacy: {epistatic_effects.get('tech_literacy', 0):.2f}",
            environmental_factors=f"Region: {genetic_profile.state_region}, Urban/Rural: {genetic_profile.residence_type}, Residential environment: {env_factors}"
        )
        
        return {
            "phenotype": result.behavioral_phenotype,
            "decision_style": result.decision_making_style,
            "family_involvement": self._parse_involvement_level(result.family_involvement_level),
            "trust_level": self._parse_trust_level(result.trust_in_healthcare)
        }
    
    def _calculate_main_effects(self, profile: GeneticProfile) -> Dict[str, float]:
        """Calculate main genetic effects"""
        from tests.poc.data.patient_genes import MAIN_EFFECT_COEFFICIENTS
        
        effects = {}
        
        # Age effect
        effects["age"] = MAIN_EFFECT_COEFFICIENTS.get("age", 0.15) * (profile.age / 50.0)
        
        # Education effect
        edu_map = {"none": 0, "primary": 1, "secondary": 2, "graduate": 3, "postgraduate": 4}
        edu_score = edu_map.get(profile.education_level, 1)
        effects["education"] = MAIN_EFFECT_COEFFICIENTS.get("education_level", 0.28) * edu_score
        
        # Income effect
        effects["income"] = MAIN_EFFECT_COEFFICIENTS.get("income_quintile", 0.22) * profile.income_quintile
        
        return effects
    
    def _calculate_epistatic_interactions(self, profile: GeneticProfile) -> Dict[str, float]:
        """Calculate gene-gene interaction effects"""
        interactions = {}
        
        # Education × Income interaction
        edu_map = {"none": 0, "primary": 1, "secondary": 2, "graduate": 3, "postgraduate": 4}
        edu_score = edu_map.get(profile.education_level, 1)
        interactions["edu_income"] = EPISTATIC_INTERACTIONS.get(("education_level", "income_quintile"), 0.25) * edu_score * profile.income_quintile
        
        # Smartphone × Internet literacy interaction
        interactions["tech_literacy"] = EPISTATIC_INTERACTIONS.get(("smartphone_access", "internet_literacy"), 0.30) * int(profile.smartphone_access) * profile.internet_literacy
        
        return interactions
    
    def _get_environmental_factors(self, profile: GeneticProfile) -> Dict[str, Any]:
        """Get environmental modifying factors"""
        return {
            "region": profile.state_region,
            "urban_rural": profile.residence_type,
            "healthcare_distance": profile.distance_to_healthcare
        }
    
    def _parse_involvement_level(self, involvement_text: str) -> float:
        """Parse family involvement level from text"""
        # Simple parsing - in production use more sophisticated NLP
        if "high" in involvement_text.lower():
            return 0.8
        elif "low" in involvement_text.lower():
            return 0.2
        else:
            return 0.5
    
    def _parse_trust_level(self, trust_text: str) -> float:
        """Parse trust level from text"""
        if "high" in trust_text.lower() or "trust" in trust_text.lower():
            return 0.8
        elif "low" in trust_text.lower() or "distrust" in trust_text.lower():
            return 0.3
        else:
            return 0.6

# ==================== ADVANCED PATIENT GENERATOR ====================

class AdvancedDynamicPatientGenerator:
    """Generate patients using complete gene profiles and DSPy modules"""
    
    def __init__(self):
        self.disease_generator = DiseaseGenerator()
        self.style_generator = CommunicationStyleGenerator()
        self.phenotyper = BehavioralPhenotyper()
        self.message_generator = MessageGenerator()
    
    def generate_patient(self, patient_id: Optional[str] = None) -> DynamicPatient:
        """Generate a complete dynamic patient"""
        if not patient_id:
            patient_id = f"patient_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Generate genetic profile with epistatic interactions
        genetic_profile = self._generate_genetic_profile()
        
        # Generate disease using DSPy
        disease_info = self.disease_generator(genetic_profile)
        
        # Generate behavioral phenotype using epistatic interactions
        behavior_info = self.phenotyper(genetic_profile)
        
        # Generate communication style
        style_info = self.style_generator(genetic_profile, disease_info["severity"])
        
        return DynamicPatient(
            patient_id=patient_id,
            genetic_profile=genetic_profile,
            
            # Disease information (DSPy generated)
            primary_condition=disease_info["condition"],
            severity_level=disease_info["severity"],
            symptoms_progression=disease_info["symptoms"],
            risk_factors=disease_info["risk_factors"],
            prognosis=disease_info["prognosis"],
            
            # Behavioral traits (DSPy generated)
            communication_style=style_info["style"],
            language_preference=style_info["language"],
            emotional_expression=style_info["emotional"],
            decision_making_style=behavior_info["decision_style"],
            family_involvement=behavior_info["family_involvement"],
            trust_in_healthcare=behavior_info["trust_level"],
            
            # Expected outcomes (for evaluation)
            expected_outcome=self._determine_expected_outcome(disease_info["condition"], disease_info["severity"]),
            red_flags_present=self._determine_red_flags(disease_info["condition"], disease_info["severity"]),
            optimal_turn_count=self._estimate_optimal_turns(disease_info["severity"])
        )
    
    def _generate_genetic_profile(self) -> GeneticProfile:
        """Generate complete 20-factor genetic profile"""
        
        # Sample base profile and add variations
        base_profiles = list(PATIENT_GENE_EXPRESSIONS.values())
        base_profile = np.random.choice(base_profiles)
        
        # Add genetic variation
        age_variation = np.random.randint(-15, 16)
        age = max(18, min(85, base_profile["age"] + age_variation))
        
        # Determine wealth psychology category
        wealth_category = self._assign_wealth_psychology(base_profile)
        
        # Calculate epistatic score
        epistatic_score = self._calculate_epistatic_score(base_profile)
        
        # Calculate behavioral phenotype score
        phenotype_score = epistatic_score + np.random.normal(0, 0.1)
        
        return GeneticProfile(
            age=age,
            gender=base_profile["gender"],
            education_level=base_profile["education_level"],
            income_quintile=base_profile["income_quintile"],
            ethnicity=base_profile["ethnicity"],
            state_region=base_profile["state_region"],
            city_tier=base_profile["city_tier"],
            
            health_insurance=bool(base_profile["health_insurance"]),
            distance_to_healthcare=base_profile["distance_to_healthcare"],
            smartphone_access=bool(base_profile["smartphone_access"]),
            internet_literacy=base_profile["internet_literacy"],
            
            tobacco_use=base_profile["tobacco_use"],
            alcohol_consumption=base_profile["alcohol_consumption"],
            physical_activity=base_profile["physical_activity"],
            occupation=base_profile["occupation"],
            marital_status=base_profile["marital_status"],
            
            residence_type=base_profile["residence_type"],
            caste_category=base_profile["caste_category"],
            time_of_day=base_profile["time_of_day"],
            season=base_profile["season"],
            
            epistatic_score=epistatic_score,
            behavioral_phenotype_score=phenotype_score,
            wealth_psychology_category=wealth_category
        )
    
    def _assign_wealth_psychology(self, profile: Dict) -> str:
        """Assign wealth psychology category based on profile"""
        income = profile["income_quintile"]
        education = profile["education_level"]
        
        # Complex assignment based on income-education interactions
        if income >= 4 and education in ["graduate", "postgraduate"]:
            return np.random.choice(["generational_wealth", "aspirational_poor"], p=[0.3, 0.7])
        elif income <= 2 and education in ["none", "primary"]:
            return np.random.choice(["hidden_wealth", "survival_rich"], p=[0.8, 0.2])
        else:
            return "aspirational_poor"
    
    def _calculate_epistatic_score(self, profile: Dict) -> float:
        """Calculate epistatic interaction score"""
        score = 0.0
        
        # Education × Income interaction
        edu_map = {"none": 0, "primary": 1, "secondary": 2, "graduate": 3, "postgraduate": 4}
        edu_score = edu_map.get(profile["education_level"], 1)
        score += EPISTATIC_INTERACTIONS.get(("education_level", "income_quintile"), 0.25) * edu_score * profile["income_quintile"]
        
        # Smartphone × Internet literacy interaction
        score += EPISTATIC_INTERACTIONS.get(("smartphone_access", "internet_literacy"), 0.30) * profile["smartphone_access"] * profile["internet_literacy"]
        
        # Age × Gender interaction
        gender_score = 1 if profile["gender"] == "female" else 0
        score += EPISTATIC_INTERACTIONS.get(("age", "gender"), 0.18) * (profile["age"] / 50.0) * gender_score
        
        return score
    
    def _determine_expected_outcome(self, condition: str, severity: str) -> str:
        """Determine expected medical outcome"""
        if severity == "critical":
            return "emergency"
        elif severity == "severe":
            return "emergency" if "chest" in condition.lower() or "heart" in condition.lower() else "routine"
        elif severity == "moderate":
            return "routine"
        else:
            return "self_care"
    
    def _determine_red_flags(self, condition: str, severity: str) -> List[str]:
        """Determine red flags present"""
        red_flags = []
        
        if "chest" in condition.lower() or "heart" in condition.lower():
            if severity in ["severe", "critical"]:
                red_flags.extend(["crushing_chest_pain", "chest_pain_radiation"])
            if severity == "critical":
                red_flags.append("severe_dyspnea")
        
        if "headache" in condition.lower() or "neurological" in condition.lower():
            if severity == "critical":
                red_flags.extend(["thunderclap_headache", "neck_stiffness"])
        
        if "respiratory" in condition.lower() or "breathing" in condition.lower():
            if severity in ["severe", "critical"]:
                red_flags.append("severe_dyspnea")
        
        return red_flags
    
    def _estimate_optimal_turns(self, severity: str) -> int:
        """Estimate optimal conversation turns"""
        return {
            "critical": 2,
            "severe": 4,
            "moderate": 8,
            "mild": 12
        }.get(severity, 8)

# ==================== ENHANCED CONVERSATION RUNNER ====================

class AdvancedMultiTurnConversationRunner:
    """Run complete conversations with advanced DSPy message generation"""
    
    def __init__(self, chat_orchestrator: ChatOrchestrator):
        self.orchestrator = chat_orchestrator
        self.message_generator = MessageGenerator()
        self.max_turns = 20
        self.min_turns = 3
    
    async def run_complete_conversation(self, patient: DynamicPatient) -> PatientConversation:
        """Run complete conversation with dynamic message generation"""
        conversation_id = str(uuid4())
        turns = []
        start_time = time.time()
        
        turn_count = 0
        is_complete = False
        
        while turn_count < self.max_turns and not is_complete:
            turn_count += 1
            
            # Generate patient message using DSPy
            if turn_count == 1:
                message_info = self.message_generator.generate_initial_message(patient)
                message = message_info["message"]
                emotional_tone = message_info["tone"]
                detail_level = message_info["detail_level"]
            else:
                message_info = self.message_generator.generate_followup_message(patient, turns, turn_count)
                message = message_info["message"]
                emotional_tone = message_info["emotion"]
                detail_level = message_info["info"]
            
            # Process through orchestrator
            from src.app2.models.schemas.multiturn_chat import (
                MultiTurnChatRequest, StakeholderRole, ChatProvider
            )
            
            request = MultiTurnChatRequest(
                conversation_id=conversation_id,
                user_message=message,
                stakeholder_role=StakeholderRole.PATIENT,
                stakeholder_id=patient.patient_id,
                patient_age=patient.genetic_profile.age,
                patient_gender=patient.genetic_profile.gender
            )
            
            # Process through orchestrator
            result = await self.orchestrator.process_conversation_turn(request)
            
            # Extract results
            agent_result = result.get("agent_result", {})
            
            turn = ConversationTurn(
                turn_number=turn_count,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                patient_message=message,
                emotional_tone=emotional_tone,
                information_revealed=detail_level,
                
                agent_response=agent_result.get("response"),
                agent_next_question=agent_result.get("next_question"),
                
                outcome=agent_result.get("outcome", "inconclusive"),
                confidence=agent_result.get("confidence", 50),
                confidence_about=agent_result.get("confidence_about", ""),
                red_flags=agent_result.get("red_flags", []),
                reasoning=agent_result.get("reasoning", ""),
                
                is_complete=agent_result.get("is_complete", False),
                completion_reason=agent_result.get("completion_reason"),
                
                symptom_evolution=message_info.get("symptoms", "stable"),
                emotional_state=emotional_tone
            )
            
            turns.append(turn)
            
            # Check completion
            is_complete = self._check_completion(turn, turn_count, patient)
            
            # Realistic delay
            await asyncio.sleep(0.5)
        
        end_time = time.time()
        
        # Calculate advanced metrics
        accuracy_score = self._calculate_accuracy(turns, patient)
        confidence_progression = [turn.confidence for turn in turns]
        red_flag_accuracy = self._calculate_red_flag_accuracy(turns, patient)
        symptom_accuracy = self._calculate_symptom_progression_accuracy(turns, patient)
        behavioral_consistency = self._calculate_behavioral_consistency(turns, patient)
        communication_authenticity = self._calculate_communication_authenticity(turns, patient)
        
        return PatientConversation(
            conversation_id=conversation_id,
            patient=patient,
            turns=turns,
            total_duration_seconds=end_time - start_time,
            final_outcome=turns[-1].outcome if turns else "incomplete",
            accuracy_score=accuracy_score,
            confidence_progression=confidence_progression,
            red_flag_detection_accuracy=red_flag_accuracy,
            symptom_progression_accuracy=symptom_accuracy,
            behavioral_consistency_score=behavioral_consistency,
            communication_authenticity=communication_authenticity
        )
    
    def _check_completion(self, turn: ConversationTurn, turn_count: int, patient: DynamicPatient) -> bool:
        """Enhanced completion logic"""
        if turn.outcome == "emergency":
            return True
        
        if turn.is_complete and turn_count >= self.min_turns:
            return True
        
        if turn_count >= patient.optimal_turn_count and turn.confidence >= 80:
            return True
        
        if turn_count >= self.max_turns:
            return True
        
        return False
    
    def _calculate_accuracy(self, turns: List[ConversationTurn], patient: DynamicPatient) -> float:
        """Calculate outcome accuracy"""
        if not turns:
            return 0.0
        
        final_outcome = turns[-1].outcome
        expected = patient.expected_outcome
        
        if final_outcome == expected:
            return 1.0
        elif (expected == "emergency" and final_outcome == "routine") or \
             (expected == "routine" and final_outcome == "emergency"):
            return 0.5
        else:
            return 0.0
    
    def _calculate_red_flag_accuracy(self, turns: List[ConversationTurn], patient: DynamicPatient) -> float:
        """Calculate red flag detection accuracy"""
        if not patient.red_flags_present:
            return 1.0
        
        detected_flags = set()
        for turn in turns:
            for flag in turn.red_flags:
                detected_flags.add(flag.lower().replace(" ", "_"))
        
        expected_flags = set(flag.lower() for flag in patient.red_flags_present)
        
        if not expected_flags:
            return 1.0
        
        intersection = len(detected_flags.intersection(expected_flags))
        return intersection / len(expected_flags)
    
    def _calculate_symptom_progression_accuracy(self, turns: List[ConversationTurn], patient: DynamicPatient) -> float:
        """Calculate how well symptom progression was captured"""
        if patient.severity_level in ["mild", "moderate"]:
            return 1.0  # No progression expected
        
        # Check if symptoms evolved appropriately in conversation
        evolution_score = 0.0
        for turn in turns[1:]:  # Skip first turn
            if "worse" in turn.symptom_evolution.lower() or "getting" in turn.symptom_evolution.lower():
                evolution_score += 1
        
        max_possible = max(1, len(turns) - 1)
        return min(1.0, evolution_score / max_possible)
    
    def _calculate_behavioral_consistency(self, turns: List[ConversationTurn], patient: DynamicPatient) -> float:
        """Calculate behavioral consistency across conversation"""
        if len(turns) < 2:
            return 1.0
        
        consistency_score = 0.0
        
        # Check emotional consistency
        emotions = [turn.emotional_tone for turn in turns if turn.emotional_tone]
        if len(set(emotions)) <= 2:  # Should have consistent emotional patterns
            consistency_score += 0.5
        
        # Check information sharing pattern
        detail_levels = [turn.information_revealed for turn in turns if turn.information_revealed]
        if len(set(detail_levels)) <= 2:  # Should have consistent detail sharing
            consistency_score += 0.5
        
        return min(1.0, consistency_score)
    
    def _calculate_communication_authenticity(self, turns: List[ConversationTurn], patient: DynamicPatient) -> float:
        """Calculate how authentic the communication feels"""
        # This is a simplified metric - in production, use more sophisticated NLP
        authenticity_score = 0.0
        
        for turn in turns:
            message_length = len(turn.patient_message.split())
            
            # Check if message length matches communication style
            if patient.communication_style == "detailed" and message_length >= 15:
                authenticity_score += 1
            elif patient.communication_style == "direct" and 5 <= message_length <= 15:
                authenticity_score += 1
            elif patient.communication_style in ["traditional", "casual"] and message_length >= 8:
                authenticity_score += 1
        
        return min(1.0, authenticity_score / max(1, len(turns)))

# ==================== TESTS ====================

@pytest.mark.asyncio
async def test_advanced_dynamic_patient_system():
    """Test the advanced dynamic patient conversation system"""
    
    generator = AdvancedDynamicPatientGenerator()
    
    # Create orchestrator
    orchestrator = ChatOrchestrator()
    await orchestrator.initialize()
    
    runner = AdvancedMultiTurnConversationRunner(orchestrator)
    
    # Generate diverse patients using full gene profiles
    patients = [generator.generate_patient() for _ in range(3)]  # Start with 3 for testing
    
    conversations = []
    
    print("🧬 Starting Advanced Dynamic Patient Conversations...")
    print("✨ Using DSPy signatures for all generation")
    print("🔬 Including epistatic interactions and wealth psychology")
    print(f"📊 Testing with {len(patients)} genetically diverse patients")
    
    for i, patient in enumerate(patients):
        print(f"\n🧬 Patient {i + 1}/{len(patients)}: {patient.patient_id}")
        print(f"   Genetics: Age {patient.genetic_profile.age}, {patient.genetic_profile.gender}, {patient.genetic_profile.ethnicity}")
        print(f"   Profile: {patient.genetic_profile.education_level}, Q{patient.genetic_profile.income_quintile}, {patient.genetic_profile.state_region}")
        print(f"   Psychology: {patient.genetic_profile.wealth_psychology_category}")
        print(f"   Condition: {patient.primary_condition} ({patient.severity_level})")
        print(f"   Communication: {patient.communication_style}")
        print(f"   Behavior: Family involvement {patient.family_involvement:.1f}, Trust {patient.trust_in_healthcare:.1f}")
        
        try:
            conversation = await runner.run_complete_conversation(patient)
            conversations.append(conversation)
            
            print(f"   ✅ Completed in {len(conversation.turns)} turns")
            print(f"   🎯 Final: {conversation.final_outcome} (accuracy: {conversation.accuracy_score:.2f})")
            print(f"   📈 Confidence: {conversation.confidence_progression}")
            print(f"   🚩 Red flag accuracy: {conversation.red_flag_detection_accuracy:.2f}")
            print(f"   🔄 Symptom tracking: {conversation.symptom_progression_accuracy:.2f}")
            print(f"   🎭 Behavioral consistency: {conversation.behavioral_consistency_score:.2f}")
            print(f"   💬 Communication authenticity: {conversation.communication_authenticity:.2f}")
            
            # Show conversation sample
            print(f"\n💬 CONVERSATION SAMPLE - {patient.patient_id}")
            print("-" * 100)
            for j, turn in enumerate(conversation.turns[:4]):  # First 4 turns
                print(f"Turn {turn.turn_number} ({turn.emotional_tone}):")
                print(f"👤 Patient: {turn.patient_message}")
                if turn.agent_response:
                    print(f"🤖 Agent: {turn.agent_response}")
                if turn.agent_next_question:
                    print(f"❓ Question: {turn.agent_next_question}")
                print(f"📊 {turn.outcome.upper()} (confidence: {turn.confidence}%)")
                if turn.red_flags:
                    print(f"🚩 Red Flags: {', '.join(turn.red_flags[:2])}")
                print(f"🔄 Symptom Evolution: {turn.symptom_evolution}")
                print()
            if len(conversation.turns) > 4:
                print(f"... {len(conversation.turns) - 4} more turns ...")
            print("-" * 100)
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    # Advanced analysis
    if conversations:
        print("\n🧬 ADVANCED SYSTEM ANALYSIS")
        print("=" * 80)
        
        # Overall metrics
        avg_turns = np.mean([len(c.turns) for c in conversations])
        avg_accuracy = np.mean([c.accuracy_score for c in conversations])
        avg_red_flag = np.mean([c.red_flag_detection_accuracy for c in conversations])
        avg_symptom = np.mean([c.symptom_progression_accuracy for c in conversations])
        avg_behavioral = np.mean([c.behavioral_consistency_score for c in conversations])
        avg_authenticity = np.mean([c.communication_authenticity for c in conversations])
        
        print(f"📊 Conversation Quality:")
        print(f"  Average turns: {avg_turns:.1f}")
        print(f"  Outcome accuracy: {avg_accuracy:.2f}")
        print(f"  Red flag detection: {avg_red_flag:.2f}")
        print(f"  Symptom progression tracking: {avg_symptom:.2f}")
        print(f"  Behavioral consistency: {avg_behavioral:.2f}")
        print(f"  Communication authenticity: {avg_authenticity:.2f}")
        
        # Genetic diversity analysis
        regions = [c.patient.genetic_profile.state_region for c in conversations]
        ethnicities = [c.patient.genetic_profile.ethnicity for c in conversations]
        wealth_categories = [c.patient.genetic_profile.wealth_psychology_category for c in conversations]
        
        print(f"\n🧬 Genetic Diversity:")
        print(f"  Regions: {set(regions)}")
        print(f"  Ethnicities: {set(ethnicities)}")
        print(f"  Wealth psychology: {set(wealth_categories)}")
        
        # Save results
        results_path = Path("advanced_dynamic_patient_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "system_type": "advanced_dynamic_dspy",
                "conversations": [asdict(c) for c in conversations],
                "summary": {
                    "avg_turns": float(avg_turns),
                    "avg_accuracy": float(avg_accuracy),
                    "avg_red_flag_detection": float(avg_red_flag),
                    "avg_symptom_tracking": float(avg_symptom),
                    "avg_behavioral_consistency": float(avg_behavioral),
                    "avg_communication_authenticity": float(avg_authenticity),
                    "total_patients": len(conversations),
                    "genetic_diversity": {
                        "regions": list(set(regions)),
                        "ethnicities": list(set(ethnicities)),
                        "wealth_categories": list(set(wealth_categories))
                    }
                }
            }, f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_path}")
    
    assert len(conversations) > 0, "Should complete at least one conversation"
    assert all(len(c.turns) >= 2 for c in conversations), "All conversations should have minimum turns"

@pytest.mark.asyncio
@pytest.mark.slow
async def test_large_scale_advanced_patient_evaluation():
    """Test with 100 patients using full genetic modeling"""
    
    generator = AdvancedDynamicPatientGenerator()
    orchestrator = ChatOrchestrator()
    await orchestrator.initialize()
    runner = AdvancedMultiTurnConversationRunner(orchestrator)
    
    # Generate 100 genetically diverse patients
    patients = [generator.generate_patient() for _ in range(100)]
    
    print("🧬 LARGE SCALE ADVANCED EVALUATION")
    print("✨ 100 Patients with Full Genetic Profiles")
    print("🔬 Epistatic Interactions + Wealth Psychology")
    print("🎭 DSPy-Generated Communication & Symptoms")
    print("⏱️  Expected duration: 4-8 hours")
    
    start_time = time.time()
    conversations = []
    
    for i, patient in enumerate(patients):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/100 patients ({elapsed/60:.1f}m elapsed)")
        
        try:
            conversation = await runner.run_complete_conversation(patient)
            conversations.append(conversation)
            
            if i % 25 == 0:
                print(f"🧬 Patient {i+1}: {patient.genetic_profile.ethnicity} {patient.genetic_profile.state_region} - {conversation.final_outcome} in {len(conversation.turns)} turns")
        
        except Exception as e:
            print(f"❌ Patient {i+1} failed: {str(e)}")
    
    total_time = time.time() - start_time
    
    # Comprehensive analysis
    print(f"\n🏆 FINAL ADVANCED RESULTS - {len(conversations)} Patients")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    
    # Analysis by genetic factors
    genetic_analysis = {}
    for conv in conversations:
        region = conv.patient.genetic_profile.state_region
        if region not in genetic_analysis:
            genetic_analysis[region] = []
        genetic_analysis[region].append(conv)
    
    print(f"\n🧬 Performance by Genetic Region:")
    for region, convs in genetic_analysis.items():
        avg_acc = np.mean([c.accuracy_score for c in convs])
        avg_turns = np.mean([len(c.turns) for c in convs])
        avg_auth = np.mean([c.communication_authenticity for c in convs])
        print(f"  {region}: {len(convs)} patients, {avg_acc:.2f} accuracy, {avg_turns:.1f} turns, {avg_auth:.2f} authenticity")
    
    # Wealth psychology analysis
    wealth_analysis = {}
    for conv in conversations:
        wealth_cat = conv.patient.genetic_profile.wealth_psychology_category
        if wealth_cat not in wealth_analysis:
            wealth_analysis[wealth_cat] = []
        wealth_analysis[wealth_cat].append(conv)
    
    print(f"\n💰 Performance by Wealth Psychology:")
    for category, convs in wealth_analysis.items():
        avg_acc = np.mean([c.accuracy_score for c in convs])
        avg_behav = np.mean([c.behavioral_consistency_score for c in convs])
        print(f"  {category}: {len(convs)} patients, {avg_acc:.2f} accuracy, {avg_behav:.2f} behavioral consistency")
    
    # Save comprehensive results
    results = {
        "advanced_genetic_system_results": {
            "total_patients": len(conversations),
            "total_time_hours": total_time / 3600,
            "conversations": [asdict(c) for c in conversations],
            "genetic_performance": {
                region: {
                    "count": len(convs),
                    "avg_accuracy": float(np.mean([c.accuracy_score for c in convs])),
                    "avg_turns": float(np.mean([len(c.turns) for c in convs])),
                    "avg_symptom_tracking": float(np.mean([c.symptom_progression_accuracy for c in convs])),
                    "avg_behavioral_consistency": float(np.mean([c.behavioral_consistency_score for c in convs])),
                    "avg_communication_authenticity": float(np.mean([c.communication_authenticity for c in convs]))
                }
                for region, convs in genetic_analysis.items()
            },
            "wealth_psychology_performance": {
                category: {
                    "count": len(convs),
                    "avg_accuracy": float(np.mean([c.accuracy_score for c in convs])),
                    "avg_behavioral_consistency": float(np.mean([c.behavioral_consistency_score for c in convs])),
                    "avg_family_involvement": float(np.mean([c.patient.family_involvement for c in convs]))
                }
                for category, convs in wealth_analysis.items()
            }
        }
    }
    
    with open("advanced_genetic_system_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("💾 Full genetic analysis saved to advanced_genetic_system_evaluation.json")
    print("🎯 Ready for genetic-based optimizer training!")
    
    assert len(conversations) >= 80, f"Should complete at least 80/100 conversations, got {len(conversations)}"
    assert len(set(c.patient.genetic_profile.state_region for c in conversations)) >= 3, "Should have regional diversity"
    assert len(set(c.patient.genetic_profile.wealth_psychology_category for c in conversations)) >= 2, "Should have wealth psychology diversity"
