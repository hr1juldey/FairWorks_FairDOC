"""
Clean Conversation Logger for Medical Triage Chats
"""
import json
import structlog
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import UUID

logger = structlog.get_logger(__name__)

class ConversationLogger:
    """Clean, structured conversation logging for dashboard display"""
    
    def __init__(self, log_dir: str = "conversation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def log_conversation_turn(self, 
                            conversation_id: str,
                            turn_number: int,
                            patient_message: str,
                            agent_response: str,
                            medical_assessment: Dict[str, Any],
                            patient_info: Optional[Dict] = None):
        """Log a single conversation turn in clean format"""
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        turn_data = {
            "conversation_id": str(conversation_id),
            "turn_number": turn_number,
            "timestamp": timestamp,
            "patient": {
                "message": patient_message,
                "info": patient_info or {}
            },
            "agent": {
                "response": agent_response,
                "assessment": {
                    "outcome": medical_assessment.get("outcome", "unknown"),
                    "confidence": medical_assessment.get("confidence", 0),
                    "confidence_about": self._explain_confidence(medical_assessment),
                    "red_flags": medical_assessment.get("red_flags", []),
                    "reasoning": medical_assessment.get("reasoning", ""),
                    "next_question": medical_assessment.get("next_question"),
                    "is_complete": medical_assessment.get("is_complete", False)
                }
            }
        }
        
        # Write to conversation-specific file
        conv_file = self.log_dir / f"conversation_{conversation_id}.jsonl"
        with open(conv_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(turn_data, ensure_ascii=False) + '\n')
            
        # Also write to master log for dashboard
        master_file = self.log_dir / "all_conversations.jsonl"
        with open(master_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(turn_data, ensure_ascii=False) + '\n')
            
        # Clean console log
        self._log_to_console(turn_data)
    
    def _explain_confidence(self, assessment: Dict[str, Any]) -> str:
        """Explain what the confidence score refers to"""
        outcome = assessment.get("outcome", "unknown")
        confidence = assessment.get("confidence", 0)
        red_flags = len(assessment.get("red_flags", []))
        
        explanations = []
        
        if outcome == "emergency":
            explanations.append(f"emergency classification (found {red_flags} red flags)")
        elif outcome == "routine":
            explanations.append("routine doctor consultation needed")
        elif outcome == "self_care":
            explanations.append("self-care recommendation")
        elif outcome == "inconclusive":
            explanations.append("need for more information")
            
        if assessment.get("reasoning"):
            explanations.append("clinical reasoning accuracy")
            
        return f"{confidence}% confident about: " + ", ".join(explanations)
    
    def _log_to_console(self, turn_data: Dict[str, Any]):
        """Clean console logging for development"""
        conv_id = turn_data["conversation_id"][:16]
        turn = turn_data["turn_number"]
        
        print(f"\n🩺 CONVERSATION {conv_id} - Turn {turn}")
        print(f"👤 Patient: {turn_data['patient']['message']}")
        
        agent = turn_data['agent']
        assessment = agent['assessment']
        
        print(f"🤖 Agent: {agent['response']}")
        print(f"📊 Assessment: {assessment['outcome'].upper()} ({assessment['confidence_about']})")
        
        if assessment['red_flags']:
            print(f"🚩 Red Flags: {', '.join(assessment['red_flags'])}")
            
        if assessment['reasoning']:
            print(f"💭 Reasoning: {assessment['reasoning'][:500]}...")
            
        if assessment['is_complete']:
            print(f"✅ Conversation Complete: {assessment['outcome']}")
        else:
            print(f"❓ Next Question: {assessment.get('next_question', 'Continuing...')}")
        print("-" * 60)

# Global instance
conversation_logger = ConversationLogger()
