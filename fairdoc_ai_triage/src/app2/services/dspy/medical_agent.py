"""
Multi-turn Medical Conversation Agent powered by DSPy
Implements progressive questioning until conclusive outcome
"""
import dspy
from typing import Dict, List, Optional, Tuple
from enum import Enum

class MedicalOutcome(str, Enum):
    EMERGENCY = "emergency_route_to_doctor"
    ROUTINE_DOCTOR = "routine_doctor_consultation"  
    SELF_CARE = "self_care_advice"
    INCONCLUSIVE = "need_more_questions"
    SPAM_DETECTED = "spam_or_irrelevant"

class MedicalTriageSignature(dspy.Signature):
    """Medical triage conversation with progressive questioning"""
    
    # Input context
    current_symptoms: str = dspy.InputField(desc="Patient's current symptoms description")
    conversation_history: dspy.History = dspy.InputField(desc="Previous Q&A pairs")
    nice_protocols: str = dspy.InputField(desc="Relevant NICE protocol guidelines")
    
    # Output decision
    outcome_classification: str = dspy.OutputField(desc="Medical outcome: emergency|routine|self_care|inconclusive|spam")
    confidence_score: int = dspy.OutputField(desc="Confidence 0-100 in this classification")
    next_question: str = dspy.OutputField(desc="Next clarifying question, or 'COMPLETE' if done")
    reasoning: str = dspy.OutputField(desc="Medical reasoning for this decision")
    red_flags: str = dspy.OutputField(desc="Any concerning symptoms requiring immediate attention")

class MedicalTriageAgent:
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        # Configure DSPy with Ollama
        lm = dspy.LM(f'ollama_chat/{model_name}', 
                     api_base='http://localhost:11434', 
                     api_key='')
        dspy.configure(lm=lm)
        
        self.predict = dspy.ChainOfThought(MedicalTriageSignature)
        self.conversation_history = dspy.History(messages=[])
    
    async def process_turn(self, 
                          symptoms: str, 
                          nice_context: str,
                          history: Optional[dspy.History] = None) -> Dict:
        """Process one turn of medical conversation"""
        
        if history is None:
            history = self.conversation_history
            
        # Generate medical response
        result = self.predict(
            current_symptoms=symptoms,
            conversation_history=history,
            nice_protocols=nice_context
        )
        
        # Update conversation history
        history.messages.append({
            "current_symptoms": symptoms,
            "outcome_classification": result.outcome_classification,
            "next_question": result.next_question,
            "reasoning": result.reasoning
        })
        
        return {
            "outcome": result.outcome_classification,
            "confidence": int(result.confidence_score),
            "next_question": result.next_question if result.next_question != "COMPLETE" else None,
            "reasoning": result.reasoning,
            "red_flags": result.red_flags,
            "is_complete": result.next_question == "COMPLETE"
        }
