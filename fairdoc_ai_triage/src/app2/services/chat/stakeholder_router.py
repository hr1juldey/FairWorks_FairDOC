"""
Stakeholder Routing: Patient ↔ Fairdoc Agent ↔ Doctor/Admin
"""
from typing import Dict, List, Optional, Literal
from enum import Enum
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

StakeholderType = Literal["patient", "doctor", "admin", "fairdoc_agent"]

class MessageRoute(BaseModel):
    from_stakeholder: StakeholderType
    to_stakeholder: StakeholderType
    message_content: str
    conversation_id: str
    priority: Literal["low", "medium", "high", "emergency"]
    requires_human_review: bool = False

class StakeholderRouter:
    """Routes messages between patients, doctors, admin, and Fairdoc AI agent"""
    
    def __init__(self):
        self.active_conversations: Dict[str, Dict] = {}
        
    async def route_message(self, 
                          conversation_id: str,
                          from_stakeholder: StakeholderType,
                          message: str,
                          medical_outcome: Optional[str] = None) -> List[MessageRoute]:
        """Route message to appropriate stakeholders"""
        
        routes = []
        
        if from_stakeholder == "patient":
            # Patient message always goes to Fairdoc agent first
            routes.append(MessageRoute(
                from_stakeholder="patient",
                to_stakeholder="fairdoc_agent", 
                message_content=message,
                conversation_id=conversation_id,
                priority="medium"
            ))
            
            # If emergency detected, also alert doctor immediately
            if medical_outcome == "emergency":
                routes.append(MessageRoute(
                    from_stakeholder="fairdoc_agent",
                    to_stakeholder="doctor",
                    message_content=f"🚨 EMERGENCY: Patient conversation requires immediate attention. Original message: {message}",
                    conversation_id=conversation_id,
                    priority="emergency",
                    requires_human_review=True
                ))
        
        elif from_stakeholder == "fairdoc_agent":
            # Agent response goes back to patient
            routes.append(MessageRoute(
                from_stakeholder="fairdoc_agent",
                to_stakeholder="patient",
                message_content=message,
                conversation_id=conversation_id,
                priority="medium"
            ))
            
            # If routine doctor consultation needed, notify doctor
            if medical_outcome == "routine_doctor":
                routes.append(MessageRoute(
                    from_stakeholder="fairdoc_agent",
                    to_stakeholder="doctor", 
                    message_content=f"📋 New patient consultation needed. Summary: {message[:200]}...",
                    conversation_id=conversation_id,
                    priority="medium",
                    requires_human_review=True
                ))
        
        elif from_stakeholder == "doctor":
            # Doctor message goes to patient via agent (for consistency)
            routes.append(MessageRoute(
                from_stakeholder="doctor",
                to_stakeholder="patient",
                message_content=f"👩‍⚕️ Doctor: {message}",
                conversation_id=conversation_id,
                priority="high"
            ))
            
        elif from_stakeholder == "admin":
            # Admin can message anyone based on context
            # For now, default to patient
            routes.append(MessageRoute(
                from_stakeholder="admin",
                to_stakeholder="patient",
                message_content=f"🏥 Fairdoc Admin: {message}",
                conversation_id=conversation_id,
                priority="low"
            ))
        
        # Log all routes for monitoring
        logger.info("📨 Message routed",
                   conversation_id=conversation_id,
                   from_stakeholder=from_stakeholder,
                   routes_count=len(routes),
                   medical_outcome=medical_outcome)
        
        return routes
    
    async def get_conversation_stakeholders(self, conversation_id: str) -> List[StakeholderType]:
        """Get list of stakeholders involved in conversation"""
        # This would query from database/Redis in full implementation
        return ["patient", "fairdoc_agent"]  # Default minimal stakeholders
