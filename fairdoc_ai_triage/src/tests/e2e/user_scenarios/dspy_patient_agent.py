# DSPy Patient Agent - Simulates Realistic Indian Patients
"""
DSPy-powered patient simulation agent that represents diverse Indian patients
Uses DSPy modules to generate contextually appropriate responses based on:
- Patient profiles (demographics, socioeconomic background)
- Medical conditions (symptom progression)
- Communication styles (regional, educational, emotional)
"""

import dspy
from typing import Dict, List, Optional, Any, Tuple
import random
import time
import asyncio
import structlog

from src.tests.e2e.user_scenarios.patient_profiles import PatientProfile, get_patient_profile, EmotionalState
from src.tests.e2e.user_scenarios.medical_conditions import MedicalCondition, get_condition, get_condition_symptoms_at_turn, UrgencyLevel, TrustLevel
logger = structlog.get_logger(__name__)

class PatientResponseSignature(dspy.Signature):
    """DSPy signature for generating patient responses"""
    
    # Input context
    patient_background: str = dspy.InputField(desc="Patient demographics, location, socioeconomic background")
    medical_condition: str = dspy.InputField(desc="Current medical condition and symptom progression")
    communication_style: str = dspy.InputField(desc="English proficiency, local language influence, emotional state")
    conversation_context: str = dspy.InputField(desc="Previous conversation turns and current situation")
    agent_question: str = dspy.InputField(desc="Medical agent's current question or request")
    turn_number: int = dspy.InputField(desc="Current conversation turn number")
    
    # Output response
    patient_message: str = dspy.OutputField(desc="Patient's natural response reflecting their background and condition")
    emotional_state: str = dspy.OutputField(desc=f"Current emotional state: {'|'.join([e.value for e in EmotionalState])}")
    urgency_level: str = dspy.OutputField(desc=f"Patient's perceived urgency: {'|'.join([u.value for u in UrgencyLevel])}")
    communication_difficulty: str = dspy.OutputField(desc="Any communication challenges or hesitations")

class PatientEmotionalStateSignature(dspy.Signature):
    """DSPy signature for managing patient emotional state progression"""
    
    current_symptoms: str = dspy.InputField(desc="Current medical symptoms and progression")
    previous_emotional_state: str = dspy.InputField(desc="Previous emotional state")
    agent_interaction_tone: str = dspy.InputField(desc="How the medical agent is interacting")
    patient_personality: str = dspy.InputField(desc="Patient's baseline personality and anxiety levels")
    turn_number: int = dspy.InputField(desc="Conversation turn to gauge progression")
    
    new_emotional_state: str = dspy.OutputField(desc="Updated emotional state based on interaction")
    emotional_reasoning: str = dspy.OutputField(desc="Why the emotional state changed or remained same")
    trust_level: str = dspy.OutputField(desc=f"Patient's trust in the medical conversation: {'|'.join([t.value for t in TrustLevel])}")

class DSPyPatientAgent:
    """DSPy-powered patient simulation agent"""
    
    def __init__(self, patient_profile: PatientProfile, model_name: str):
        self.profile = patient_profile
        self.model_name = model_name
        # Initialize DSPy configuration
        lm = dspy.LM(
            f"ollama/{model_name}", 
            api_base="http://localhost:11434"
            )
        dspy.configure(lm=lm)

        # Initialize DSPy modules
        self.response_generator = dspy.ChainOfThought(PatientResponseSignature)
        self.emotional_processor = dspy.ChainOfThought(PatientEmotionalStateSignature)
        
        # Conversation state
        self.conversation_history = []
        self.turn_count = 0
        self.current_emotional_state = self.profile.primary_emotion  # Keep as enum
        self.conversation_active = True
        self.trust_level = "medium"  # Start with medium trust
        
        # Get medical condition details
        self.medical_condition = get_condition(self.profile.medical_condition)
        if not self.medical_condition:
            raise ValueError(f"Medical condition {self.profile.medical_condition} not found")
        
        logger.info("🤖 DSPy Patient Agent initialized",
                   patient_id=self.profile.patient_id,
                   condition=self.medical_condition.condition_name,
                   emotion=self.current_emotional_state)
    
    async def respond_to_agent(self, agent_question: str, agent_tone: str = "professional") -> Dict[str, Any]:
        """Generate patient response to medical agent question"""
        
        if not self.conversation_active:
            return self._create_conversation_ended_response()
        
        self.turn_count += 1
        
        try:
            # Update emotional state based on conversation progression
            await self._update_emotional_state(agent_question, agent_tone)
            
            # Get current symptoms for this turn
            current_symptoms = get_condition_symptoms_at_turn(
                self.profile.medical_condition, 
                self.turn_count
            )
            
            # Prepare context for DSPy
            patient_background = self._build_patient_background()
            communication_style = self._build_communication_style()
            conversation_context = self._build_conversation_context()
            
            # Generate response using DSPy
            response_result = self.response_generator(
                patient_background=patient_background,
                medical_condition=f"{self.medical_condition.name}: {current_symptoms}",
                communication_style=communication_style,
                conversation_context=conversation_context,
                agent_question=agent_question,
                turn_number=self.turn_count
            )
            
            # Apply communication style modifications
            modified_message = self._apply_communication_style(response_result.patient_message)
            
            # Add random pauses/delays based on patient characteristics
            await self._simulate_typing_delay()
            
            # Build response
            response = {
                "patient_message": modified_message,
                "emotional_state": response_result.emotional_state,
                "urgency_level": response_result.urgency_level,
                "communication_difficulty": response_result.communication_difficulty,
                "turn_number": self.turn_count,
                "trust_level": self.trust_level,
                "symptoms_severity": self._assess_symptom_severity(),
                "conversation_active": self.conversation_active
            }
            
            # Update conversation history
            self.conversation_history.append({
                "turn": self.turn_count,
                "agent_question": agent_question,
                "patient_response": modified_message,
                "emotional_state": response_result.emotional_state,
                "urgency_level": response_result.urgency_level
            })
            
            # Check if conversation should end (emergency or resolution)
            if self._should_end_conversation(response_result):
                self.conversation_active = False
                response["conversation_active"] = False
            
            logger.info("💬 Patient response generated",
                       patient_id=self.profile.patient_id,
                       turn=self.turn_count,
                       emotion=response_result.emotional_state,
                       urgency=response_result.urgency_level)
            
            return response
            
        except Exception as e:
            logger.error("❌ Patient response generation failed",
                        patient_id=self.profile.patient_id,
                        turn=self.turn_count,
                        error=str(e))
            
            # Fallback response
            return self._create_fallback_response(agent_question)
    
    async def _update_emotional_state(self, agent_question: str, agent_tone: str):
        """Update patient's emotional state based on interaction"""
        
        try:
            current_symptoms = get_condition_symptoms_at_turn(
                self.profile.medical_condition,
                self.turn_count
            )
            
            emotional_result = self.emotional_processor(
                current_symptoms=current_symptoms,
                previous_emotional_state=self.current_emotional_state,
                agent_interaction_tone=f"{agent_tone}: {agent_question}",
                patient_personality=f"Anxiety level: {self.profile.health_anxiety_level}, Trust in tech: {self.profile.trust_in_technology}",
                turn_number=self.turn_count
            )
            
            self.current_emotional_state = emotional_result.new_emotional_state
            self.trust_level = emotional_result.trust_level
            
        except Exception as e:
            logger.warning("⚠️ Emotional state update failed",
                          patient_id=self.profile.patient_id,
                          error=str(e))
            # Keep previous emotional state on failure
    
    def _build_patient_background(self) -> str:
        """Build comprehensive patient background for DSPy"""
        return f"""
        Patient: {self.profile.name}, {self.profile.age} years old, {self.profile.gender.value}
        Location: {self.profile.city}, {self.profile.state}
        Education: {self.profile.education.value}
        Economic Status: {self.profile.economic_status.value}
        Occupation: {self.profile.occupation}
        Previous Medical Experience: {self.profile.previous_medical_experience}
        Health Anxiety Level: {self.profile.health_anxiety_level}/100
        Trust in Technology: {self.profile.trust_in_technology}/100
        Family Influence: {self.profile.family_influence}/100
        """
    
    def _build_communication_style(self) -> str:
        """Build communication style description for DSPy"""
        style = self.profile.communication_style
        return f"""
        English Proficiency: {style.english_proficiency}
        Local Language Influence: {style.local_language_influence}%
        Typing Speed: {style.typing_speed}
        Vocabulary: {style.vocabulary}
        Sentence Structure: {style.sentence_structure}
        Cultural Expressions: {', '.join(style.cultural_expressions)}
        Current Emotional State: {self.current_emotional_state}
        Tech Comfort: {style.tech_comfort}
        """
    
    def _build_conversation_context(self) -> str:
        """Build conversation context for DSPy"""
        if not self.conversation_history:
            return "This is the first interaction with the medical agent."
        
        recent_turns = self.conversation_history[-2:]  # Last 2 turns for context
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"Turn {turn['turn']}: Agent asked '{turn['agent_question']}', Patient responded with {turn['emotional_state']} emotion")
        
        return " | ".join(context_parts)
    
    def _apply_communication_style(self, base_message: str) -> str:
        """Apply patient-specific communication style to the message"""
        
        style = self.profile.communication_style
        modified_message = base_message
        
        # Apply local language influence
        if style.local_language_influence > 50:
            # Add cultural expressions randomly
            if random.random() < 0.3 and style.cultural_expressions:
                expression = random.choice(style.cultural_expressions)
                modified_message = f"{modified_message} {expression}"
        
        # Apply emotional state modifications
        emotional_modifiers = self.profile.get_emotional_modifiers()
        
        # Repetition for anxious patients
        if (self.current_emotional_state == EmotionalState.ANXIOUS and 
            random.random() < emotional_modifiers["repetition_likelihood"]):
            # Add repetitive phrases
            anxious_additions = ["I'm really worried", "I don't know what to do", "Is this serious?"]
            addition = random.choice(anxious_additions)
            modified_message = f"{modified_message} {addition}"
        
        # Shorter messages for fearful patients
        if (self.current_emotional_state == EmotionalState.FEARFUL and 
            len(modified_message.split()) > 15):
            words = modified_message.split()
            modified_message = " ".join(words[:12]) + "..."
        
        # Length adjustments based on emotional state
        length_multiplier = emotional_modifiers["message_length_multiplier"]
        if length_multiplier != 1.0:
            if length_multiplier > 1.0:  # Longer messages (anxious)
                elaborations = [
                    "I'm not sure if I explained it properly",
                    "Let me tell you more details",
                    "I hope you understand what I mean"
                ]
                if random.random() < 0.4:
                    modified_message += f". {random.choice(elaborations)}"
        
        return modified_message.strip()
    
    async def _simulate_typing_delay(self):
        """Simulate realistic typing delays based on patient characteristics"""
        
        base_delay = 1.0  # Base 1 second
        
        # Adjust for typing speed
        speed_multipliers = {
            "slow": 2.0,
            "moderate": 1.0,
            "fast": 0.5
        }
        
        typing_multiplier = speed_multipliers.get(
            self.profile.communication_style.typing_speed, 1.0
        )
        
        # Adjust for emotional state
        emotional_delays = {
            "anxious": 1.5,
            "fearful": 2.0,
            "confused": 1.8,
            "worried": 1.3,
            "calm": 0.8,
            "sad": 1.4
        }
        
        emotional_multiplier = emotional_delays.get(self.current_emotional_state, 1.0)
        
        # Random pause for realism
        random_factor = random.uniform(0.5, 1.5)
        
        total_delay = base_delay * typing_multiplier * emotional_multiplier * random_factor
        
        # Cap delay at reasonable maximum
        total_delay = min(total_delay, 5.0)
        
        await asyncio.sleep(total_delay)
    
    def _assess_symptom_severity(self) -> str:
        """Assess current symptom severity based on condition and turn"""
        
        condition_severity = self.medical_condition.severity.value
        
        # Emergency conditions worsen quickly with base severity consideration
        if self.medical_condition.expected_outcome.value == "emergency_route_to_doctor":
            if self.turn_count >= 3:
                # Always severe for emergency conditions at turn 3+
                return "severe"
            elif self.turn_count >= 2:
                # Use condition severity or escalate to moderate_to_severe
                if condition_severity in ["severe", "critical"]:
                    return "severe"
                else:
                    return "moderate_to_severe"
            else:
                # Early turns: use base condition severity or default to moderate
                if condition_severity in ["severe", "critical"]:
                    return "moderate_to_severe"
                elif condition_severity == "moderate":
                    return "moderate"
                else:
                    return "mild_to_moderate"
        
        # Routine conditions stay relatively stable but consider base severity
        elif self.medical_condition.expected_outcome.value == "routine_doctor_consultation":
            if condition_severity == "severe":
                return "moderate_to_severe"
            elif condition_severity == "moderate":
                return "moderate"
            else:
                return "mild_to_moderate"
        
        # Self-care conditions are generally mild but still consider base severity
        else:
            if condition_severity in ["severe", "moderate"]:
                return "mild_to_moderate"
            else:
                return "mild"

    
    def _should_end_conversation(self, response_result) -> bool:
        """Determine if conversation should end based on patient state"""
        
        # End if emergency detected and patient understands urgency
        if (response_result.urgency_level in ["high", "critical"] and
            self.medical_condition.expected_outcome.value == "emergency_route_to_doctor" and
            self.turn_count >= 2):
            return True
        
        # End if self-care condition resolved
        if (self.medical_condition.expected_outcome.value == "self_care_advice" and
            self.turn_count >= 4 and
            response_result.emotional_state in ["calm", "hopeful"]):
            return True
        
        # End if maximum turns reached
        if self.turn_count >= self.medical_condition.maximum_turns_acceptable:
            return True
        
        return False
    
    def _create_conversation_ended_response(self) -> Dict[str, Any]:
        """Create response when conversation has ended"""
        return {
            "patient_message": "Thank you for your help. I understand what I need to do now.",
            "emotional_state": "calm",
            "urgency_level": "low",
            "communication_difficulty": "none",
            "turn_number": self.turn_count,
            "trust_level": self.trust_level,
            "symptoms_severity": self._assess_symptom_severity(),
            "conversation_active": False
        }
    
    def _create_fallback_response(self, agent_question: str) -> Dict[str, Any]:
        """Create fallback response when DSPy fails"""
        
        fallback_responses = [
            "I'm not feeling well and need help.",
            "Can you please help me understand what's wrong?",
            "I'm worried about my symptoms.",
            "I'm not sure how to explain this properly."
        ]
        
        return {
            "patient_message": random.choice(fallback_responses),
            "emotional_state": self.current_emotional_state,
            "urgency_level": "medium",
            "communication_difficulty": "system_error",
            "turn_number": self.turn_count,
            "trust_level": "medium",
            "symptoms_severity": "unknown",
            "conversation_active": True
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation"""
        return {
            "patient_id": self.profile.patient_id,
            "patient_name": self.profile.name,
            "medical_condition": self.medical_condition.name,
            "expected_outcome": self.medical_condition.expected_outcome.value,
            "total_turns": self.turn_count,
            "final_emotional_state": self.current_emotional_state,
            "final_trust_level": self.trust_level,
            "conversation_active": self.conversation_active,
            "conversation_history": self.conversation_history
        }

def create_patient_agent(patient_id: str, model_name: str) -> DSPyPatientAgent:
    """Factory function to create DSPy patient agent"""
    
    profile = get_patient_profile(patient_id)
    if not profile:
        raise ValueError(f"Patient profile {patient_id} not found")
    
    return DSPyPatientAgent(profile, model_name)

def create_diverse_patient_agents(model_name: str, count: int = 4) -> List[DSPyPatientAgent]:
    """Create diverse set of patient agents for testing"""
    
    from src.tests.e2e.user_scenarios.patient_profiles import get_diverse_patient_sample
    
    patient_ids = get_diverse_patient_sample(count)
    agents = []
    
    for patient_id in patient_ids:
        try:
            agent = create_patient_agent(patient_id, model_name)
            agents.append(agent)
        except Exception as e:
            logger.error("❌ Failed to create patient agent",
                        patient_id=patient_id, error=str(e))
    
    return agents

async def test_patient_agent_basic_functionality(patient_id: str, model_name: str) -> Dict[str, Any]:
    """Test basic functionality of a patient agent"""
    
    try:
        agent = create_patient_agent(patient_id, model_name)
        
        # Test basic response
        response1 = await agent.respond_to_agent("How are you feeling today?")
        assert "patient_message" in response1
        assert len(response1["patient_message"]) > 0
        
        # Test follow-up response
        response2 = await agent.respond_to_agent("Can you describe your symptoms?")
        assert response2["turn_number"] == 2
        
        summary = agent.get_conversation_summary()
        
        return {
            "patient_id": patient_id,
            "test_passed": True,
            "responses": [response1, response2],
            "summary": summary
        }
        
    except Exception as e:
        return {
            "patient_id": patient_id,
            "test_passed": False,
            "error": str(e)
        }
