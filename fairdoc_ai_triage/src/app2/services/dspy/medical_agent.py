"""
Multi-turn Medical Conversation Agent powered by DSPy
Implements progressive questioning using DSPy modules and programs
Enhanced with native DSPy reasoning separation for DeepSeek-R1
"""

import dspy
from typing import Dict, List, Optional, Any
from enum import Enum
import structlog
import asyncio
from dataclasses import dataclass

logger = structlog.get_logger(__name__)

class MedicalOutcome(str, Enum):
    """Medical triage outcomes following NICE guidelines"""
    EMERGENCY = "emergency_route_to_doctor"
    ROUTINE_DOCTOR = "routine_doctor_consultation"
    SELF_CARE = "self_care_advice"
    INCONCLUSIVE = "need_more_questions"
    SPAM_DETECTED = "spam_or_irrelevant"

@dataclass
class ConversationTurn:
    """Structure for single conversation turn"""
    symptoms: str
    agent_question: Optional[str]
    reasoning: str
    outcome: str
    confidence: int
    red_flags: List[str]

class MedicalTriageSignature(dspy.Signature):
    """Medical triage with separated reasoning and output"""
    # Input context
    current_symptoms: str = dspy.InputField(desc="Patient's current symptoms description")
    conversation_history: dspy.History = dspy.InputField(desc="Previous conversation turns")
    nice_protocols: str = dspy.InputField(desc="Relevant NICE protocol guidelines")
    
    # Reasoning output (for DeepSeek-R1 thinking)
    medical_reasoning: str = dspy.OutputField(desc="Step-by-step medical reasoning process")
    
    # Final decision outputs
    outcome_classification: str = dspy.OutputField(desc="Medical outcome: emergency|routine|self_care|inconclusive|spam")
    confidence_score: int = dspy.OutputField(desc="Confidence level 0-100")
    next_question: str = dspy.OutputField(desc="Next question, or 'COMPLETE' if done")
    red_flags: str = dspy.OutputField(desc="Concerning symptoms requiring attention")

class EmergencyDetectionSignature(dspy.Signature):
    """Specialized signature for emergency detection"""
    symptoms: str = dspy.InputField(desc="Current symptoms to analyze")
    medical_context: str = dspy.InputField(desc="Medical context and history")
    
    is_emergency: bool = dspy.OutputField(desc="True if emergency detected")
    emergency_reasoning: str = dspy.OutputField(desc="Reasoning for emergency classification")
    critical_flags: str = dspy.OutputField(desc="Critical red flags identified")

class MedicalReasoningModule(dspy.Module):
    """DSPy module for medical reasoning with thinking separation"""
    
    def __init__(self):
        super().__init__()
        # Use DSPy's ChainOfThoughtWithHint for enhanced reasoning
        self.medical_cot = dspy.ChainOfThoughtWithHint(MedicalTriageSignature)
        self.emergency_detector = dspy.ChainOfThought(EmergencyDetectionSignature)
    
    def forward(self, current_symptoms, conversation_history, nice_protocols):
        # First check for emergency conditions
        emergency_check = self.emergency_detector(
            symptoms=current_symptoms,
            medical_context=f"History: {conversation_history}\nProtocols: {nice_protocols}"
        )
        
        # If emergency detected, prioritize that path
        if emergency_check.is_emergency:
            hint = f"EMERGENCY DETECTED: {emergency_check.emergency_reasoning}"
        else:
            hint = "Proceed with standard triage assessment"
        
        # Main medical reasoning with hint
        result = self.medical_cot(
            current_symptoms=current_symptoms,
            conversation_history=conversation_history,
            nice_protocols=nice_protocols,
            hint=hint  # DSPy hint for guided reasoning
        )
        
        return result, emergency_check

class MedicalTriageProgram(dspy.Module):
    """DSPy program orchestrating medical triage workflow"""
    
    def __init__(self):
        super().__init__()
        self.reasoning_module = MedicalReasoningModule()
    
    def forward(self, symptoms, nice_context, history):
        # Use DSPy program to orchestrate the workflow
        medical_result, emergency_result = self.reasoning_module(
            current_symptoms=symptoms,
            conversation_history=history,
            nice_protocols=nice_context
        )
        
        return {
            'medical_reasoning': medical_result,
            'emergency_analysis': emergency_result
        }

class MedicalTriageAgent:
    """DSPy-powered medical triage agent with native reasoning separation"""
    
    def __init__(self, model_name: str = "deepseek-r1:8b", question_generator=None):
        """Initialize agent with DSPy program architecture"""
        self.model_name = model_name
        self._configure_dspy_with_thinking()
        
        # Use DSPy program instead of single signature
        self.triage_program = MedicalTriageProgram()
        
        # Initialize question generator
        if question_generator is None:
            from src.app2.services.dspy.question_generator import MedicalQuestionGenerator
            self.question_generator = MedicalQuestionGenerator(model_name=model_name)
        else:
            self.question_generator = question_generator
        
        # Conversation state tracking
        self.conversation_history = dspy.History(messages=[])
        self.turn_count = 0
        
        logger.info("🩺 Medical Triage Agent initialized", model=model_name)

    
    def _configure_dspy_with_thinking(self):
        """Configure DSPy with DeepSeek-R1 via proper Ollama integration"""
        try:
            # Method 1: Direct Ollama integration (Recommended)
            lm = dspy.LM(model=f'ollama/{self.model_name}')
            dspy.configure(lm=lm)
            logger.info("✅ DSPy configured with DeepSeek-R1 via Ollama")
            
        except Exception as e:
            # Method 2: Fallback to OpenAI-compatible endpoint
            try:
                lm = dspy.OpenAI(
                    api_base='http://localhost:11434/v1/',
                    api_key='ollama',  # Required but not validated
                    model=self.model_name,
                    model_type='chat'
                )
                dspy.configure(lm=lm)
                logger.info("✅ DSPy configured with DeepSeek-R1 via OpenAI-compatible API")
                
            except Exception as fallback_error:
                logger.error("❌ Both Ollama methods failed", 
                            primary_error=str(e),
                            fallback_error=str(fallback_error))
                raise

    async def process_turn(
        self,
        symptoms: str,
        nice_context: str,
        history: Optional[dspy.History] = None
    ) -> Dict[str, Any]:
        """Process medical turn using DSPy program"""
        if not symptoms.strip():
            raise ValueError("Symptoms cannot be empty")
        
        if history is None:
            history = self.conversation_history
        
        self.turn_count += 1
        
        try:
            # Use DSPy program for structured reasoning
            program_result = self.triage_program(
                symptoms=symptoms,
                nice_context=nice_context,
                history=history
            )
            
            # Extract results from DSPy program output
            medical_result = program_result['medical_reasoning']
            emergency_result = program_result['emergency_analysis']
            
            # Parse response using DSPy's native structure
            response = self._parse_dspy_response(medical_result, emergency_result)

            # Generate specialized questions using question generator
            if not response.get("is_complete") and response["outcome"] == "inconclusive":
                question_result = self.question_generator.suggest_questions(
                    symptom_text=symptoms,
                    conversation_context=str(history),
                    nice_context=nice_context,
                    max_questions=1
                )
                if question_result["questions"]:
                    response["next_question"] = question_result["questions"][0]
                    response["reasoning"] += f" | Question reasoning: {question_result.get('medical_reasoning', '')}"

            
            # Update conversation history
            self._update_history(history, symptoms, response)
            
            logger.info(
                "🔄 Medical turn processed",
                turn=self.turn_count,
                outcome=response["outcome"],
                confidence=response["confidence"],
                thinking_captured=len(response.get("thinking", "")) > 0
            )
            
            return response
            
        except Exception as e:
            logger.error("❌ Error processing medical turn", error=str(e))
            return self._create_error_response(str(e))
    
    def _parse_dspy_response(self, medical_result, emergency_result) -> Dict[str, Any]:
        """Parse DSPy program results using native structure"""
        # Extract red flags as list
        red_flags = []
        if hasattr(medical_result, 'red_flags') and medical_result.red_flags:
            red_flags = [flag.strip() for flag in medical_result.red_flags.split(',') if flag.strip()]
        
        # Add emergency red flags if detected
        if emergency_result.is_emergency and hasattr(emergency_result, 'critical_flags'):
            emergency_flags = [flag.strip() for flag in emergency_result.critical_flags.split(',') if flag.strip()]
            red_flags.extend(emergency_flags)
        
        # Validate outcome classification
        outcome = medical_result.outcome_classification.lower()
        if outcome not in [e.value.split('_')[0] for e in MedicalOutcome]:
            outcome = "emergency" if emergency_result.is_emergency else "inconclusive"
        
        # Ensure confidence bounds
        try:
            confidence = int(medical_result.confidence_score)
            # Boost confidence if emergency detected
            if emergency_result.is_emergency and confidence < 80:
                confidence = max(confidence, 85)
            confidence = max(0, min(100, confidence))
        except (ValueError, AttributeError):
            confidence = 85 if emergency_result.is_emergency else 50
        
        return {
            "outcome": outcome,
            "confidence": confidence,
            "next_question": (
                medical_result.next_question
                if medical_result.next_question != "COMPLETE"
                else None
            ),
            "reasoning": getattr(medical_result, 'medical_reasoning', ''),
            "thinking": getattr(medical_result, 'reasoning', ''),  # DSPy thinking output
            "red_flags": red_flags,
            "is_complete": medical_result.next_question == "COMPLETE",
            "emergency_detected": emergency_result.is_emergency
        }
    
    def _update_history(self, history: dspy.History, symptoms: str, response: Dict[str, Any]):
        """Update conversation history with DSPy structure"""
        turn_data = {
            "turn": self.turn_count,
            "current_symptoms": symptoms,
            "outcome_classification": response["outcome"],
            "next_question": response.get("next_question"),
            "reasoning": response["reasoning"],
            "confidence": response["confidence"],
            "red_flags": response["red_flags"],
            "emergency_detected": response.get("emergency_detected", False)
        }
        history.messages.append(turn_data)
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "outcome": "inconclusive",
            "confidence": 0,
            "next_question": "I'm sorry, I encountered an error. Could you please describe your symptoms again?",
            "reasoning": f"Error processing request: {error_msg}",
            "thinking": "",
            "red_flags": [],
            "is_complete": False,
            "emergency_detected": False
        }
    
    def reset_conversation(self):
        """Reset conversation state for new patient"""
        self.conversation_history = dspy.History(messages=[])
        self.turn_count = 0
        logger.info("🔄 Conversation reset")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            "turns": self.turn_count,
            "history_length": len(self.conversation_history.messages),
            "last_outcome": (
                self.conversation_history.messages[-1].get("outcome_classification")
                if self.conversation_history.messages
                else None
            )
        }
