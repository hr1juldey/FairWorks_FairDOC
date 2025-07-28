"""
Multi-turn Medical Conversation Agent powered by DSPy
Implements progressive questioning until conclusive outcome

Test-driven development approach with comprehensive error handling
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
    """Medical triage conversation with progressive questioning"""
    
    # Input context
    current_symptoms: str = dspy.InputField(
        desc="Patient's current symptoms description"
    )
    conversation_history: dspy.History = dspy.InputField(
        desc="Previous conversation turns with patient"
    )
    nice_protocols: str = dspy.InputField(
        desc="Relevant NICE protocol guidelines for symptoms"
    )
    
    # Output decision
    outcome_classification: str = dspy.OutputField(
        desc="Medical outcome: emergency|routine|self_care|inconclusive|spam"
    )
    confidence_score: int = dspy.OutputField(
        desc="Confidence level 0-100 in this classification"
    )  
    next_question: str = dspy.OutputField(
        desc="Next clarifying question, or 'COMPLETE' if done"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step medical reasoning for this decision"
    )
    red_flags: str = dspy.OutputField(
        desc="Concerning symptoms requiring immediate attention"
    )

class MedicalTriageAgent:
    """DSPy-powered medical triage agent with conversation state"""
    
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        """Initialize agent with Ollama model"""
        self.model_name = model_name
        self._configure_dspy()
        
        # Chain of thought reasoning for medical decisions
        self.predict = dspy.ChainOfThought(MedicalTriageSignature)
        
        # Conversation state tracking
        self.conversation_history = dspy.History(messages=[])
        self.turn_count = 0
        
        logger.info("🩺 Medical Triage Agent initialized", model=model_name)
    
    def _configure_dspy(self):
        """Configure DSPy with Ollama backend"""
        try:
            lm = dspy.LM(
                f'ollama_chat/{self.model_name}',
                api_base='http://localhost:11434',
                api_key=''
            )
            dspy.configure(lm=lm)
            logger.info("✅ DSPy configured with Ollama")
        except Exception as e:
            logger.error("❌ Failed to configure DSPy", error=str(e))
            raise
    
    async def process_turn(
        self, 
        symptoms: str, 
        nice_context: str,
        history: Optional[dspy.History] = None
    ) -> Dict[str, Any]:
        """Process one turn of medical conversation"""
        
        if not symptoms.strip():
            raise ValueError("Symptoms cannot be empty")
        
        if history is None:
            history = self.conversation_history
            
        self.turn_count += 1
        
        try:
            # Generate medical response using DSPy
            result = self.predict(
                current_symptoms=symptoms,
                conversation_history=history,
                nice_protocols=nice_context
            )
            
            # Parse and validate results
            response = self._parse_response(result)
            
            # Update conversation history  
            self._update_history(history, symptoms, response)
            
            logger.info(
                "🔄 Medical turn processed",
                turn=self.turn_count,
                outcome=response["outcome"],
                confidence=response["confidence"]
            )
            
            return response
            
        except Exception as e:
            logger.error("❌ Error processing medical turn", error=str(e))
            return self._create_error_response(str(e))
    
    def _parse_response(self, result) -> Dict[str, Any]:
        """Parse DSPy response into structured format"""
        
        # Extract red flags as list
        red_flags = []
        if hasattr(result, 'red_flags') and result.red_flags:
            red_flags = [
                flag.strip() 
                for flag in result.red_flags.split(',')
                if flag.strip()
            ]
        
        # Validate outcome classification
        outcome = result.outcome_classification.lower()
        if outcome not in [e.value.split('_')[0] for e in MedicalOutcome]:
            outcome = "inconclusive"
            
        # Ensure confidence is within bounds
        try:
            confidence = int(result.confidence_score)
            confidence = max(0, min(100, confidence))
        except (ValueError, AttributeError):
            confidence = 50
            
        return {
            "outcome": outcome,
            "confidence": confidence,
            "next_question": (
                result.next_question 
                if result.next_question != "COMPLETE" 
                else None
            ),
            "reasoning": getattr(result, 'reasoning', ''),
            "red_flags": red_flags,
            "is_complete": result.next_question == "COMPLETE"
        }
    
    def _update_history(
        self, 
        history: dspy.History, 
        symptoms: str, 
        response: Dict[str, Any]
    ):
        """Update conversation history with current turn"""
        
        turn_data = {
            "turn": self.turn_count,
            "current_symptoms": symptoms,
            "outcome_classification": response["outcome"],
            "next_question": response.get("next_question"),
            "reasoning": response["reasoning"],
            "confidence": response["confidence"],
            "red_flags": response["red_flags"]
        }
        
        history.messages.append(turn_data)
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "outcome": "inconclusive",
            "confidence": 0,
            "next_question": "I'm sorry, I encountered an error. Could you please describe your symptoms again?",
            "reasoning": f"Error processing request: {error_msg}",
            "red_flags": [],
            "is_complete": False
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
