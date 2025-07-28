"""
Gold Standards Database Model for Fairdoc AI V2

Simple table + JSON blob for gold-standard medical dialogues
Used for DSPy optimization and training (≤150 LOC)
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Text, JSON, Integer, DateTime, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from pydantic import BaseModel

from src.app.core.database import Base
from src.app2.models.schemas.medical_triage import MedicalOutcome
from src.app2.models.schemas.multiturn_chat import ConversationState

class GoldStandardDB(Base):
    """
    Gold standard medical conversation for DSPy training
    Contains expert-validated multi-turn medical triage dialogues
    """
    __tablename__ = "gold_standards_v2"
    
    # Primary identification
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id = Column(String(50), nullable=False, unique=True, index=True)
    
    # Case metadata
    condition_name = Column(String(200), nullable=False, index=True)
    initial_symptoms = Column(Text, nullable=False)
    expected_outcome = Column(String(30), nullable=False, index=True)
    difficulty_level = Column(String(20), nullable=False, default="medium")  # easy|medium|hard
    nice_protocol_code = Column(String(50), nullable=True, index=True)
    
    # Gold standard dialogue (JSON format)
    conversation_turns = Column(JSON, nullable=False)  # List of turn objects
    expert_reasoning = Column(Text, nullable=False)  # Medical reasoning
    red_flags_expected = Column(JSON, nullable=False, default=list)
    
    # Quality metrics
    turn_count = Column(Integer, nullable=False)
    expert_confidence = Column(Integer, nullable=False)  # 0-100
    
    # Training metadata
    is_validated = Column(Boolean, nullable=False, default=True)
    validation_date = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(100), nullable=False, default="system")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Database indexes for DSPy optimization
    __table_args__ = (
        Index('idx_gold_condition_outcome', 'condition_name', 'expected_outcome'),
        Index('idx_gold_difficulty', 'difficulty_level'),
        Index('idx_gold_validated', 'is_validated'),
    )
    
    @classmethod
    def from_conversation_state(cls, conversation: ConversationState, expert_data: Dict[str, Any]) -> GoldStandardDB:
        """Create gold standard from completed conversation with expert validation"""
        return cls(
            case_id=f"gold_{conversation.conversation_id}",
            condition_name=expert_data.get("condition_name", "Unknown"),
            initial_symptoms=conversation.initial_symptoms,
            expected_outcome=conversation.current_outcome.value,
            difficulty_level=expert_data.get("difficulty", "medium"),
            nice_protocol_code=expert_data.get("nice_protocol"),
            conversation_turns=[turn.model_dump() for turn in conversation.turns],
            expert_reasoning=expert_data.get("reasoning", ""),
            red_flags_expected=conversation.red_flags_detected,
            turn_count=conversation.turn_count,
            expert_confidence=expert_data.get("confidence", 95),
            created_by=expert_data.get("expert_id", "system")
        )
    
    def to_dspy_example(self) -> Dict[str, Any]:
        """Convert to DSPy training example format"""
        return {
            "case_id": self.case_id,
            "initial_symptoms": self.initial_symptoms,
            "conversation_turns": self.conversation_turns,
            "expected_outcome": self.expected_outcome,
            "expert_reasoning": self.expert_reasoning,
            "nice_protocol": self.nice_protocol_code,
            "red_flags": self.red_flags_expected,
            "difficulty": self.difficulty_level
        }

# Seed data for essential medical conditions
GOLD_STANDARD_SEED_DATA = [
    {
        "case_id": "gold_headache_001",
        "condition_name": "Tension Headache",
        "initial_symptoms": "I have a headache that feels like a tight band around my head",
        "expected_outcome": "self_care",
        "difficulty_level": "easy",
        "nice_protocol_code": "NG127_HEADACHE",
        "conversation_turns": [
            {
                "turn_number": 1,
                "user_message": "I have a headache that feels like a tight band around my head",
                "agent_question": "How long have you had this headache, and is this a new type of headache for you?",
                "medical_outcome": "inconclusive",
                "confidence_score": 60
            },
            {
                "turn_number": 2,
                "user_message": "I've had it for about 2 hours. I get these sometimes when I'm stressed at work",
                "agent_question": "Have you experienced any nausea, vision changes, or neck stiffness with this headache?",
                "medical_outcome": "inconclusive", 
                "confidence_score": 75
            },
            {
                "turn_number": 3,
                "user_message": "No, none of those symptoms. Just the tight feeling",
                "agent_response": "Based on your symptoms, this appears to be a tension headache. You can manage this with rest, hydration, and over-the-counter pain relief.",
                "medical_outcome": "self_care",
                "confidence_score": 90
            }
        ],
        "expert_reasoning": "Classic tension headache presentation with no red flags. Stress trigger, bilateral pressure sensation, no neurological symptoms. Appropriate for self-care management.",
        "red_flags_expected": [],
        "turn_count": 3,
        "expert_confidence": 95
    },
    {
        "case_id": "gold_chest_pain_001", 
        "condition_name": "Cardiac Chest Pain",
        "initial_symptoms": "I'm having crushing chest pain that goes down my left arm",
        "expected_outcome": "emergency",
        "difficulty_level": "easy",
        "nice_protocol_code": "CG95_CHEST_PAIN",
        "conversation_turns": [
            {
                "turn_number": 1,
                "user_message": "I'm having crushing chest pain that goes down my left arm",
                "agent_response": "This sounds like a potential heart attack. You need immediate medical attention. Please call 999 or go to the nearest emergency department now.",
                "medical_outcome": "emergency",
                "confidence_score": 95,
                "red_flags": ["crushing_pain", "left_arm_radiation"]
            }
        ],
        "expert_reasoning": "Classic presentation of acute myocardial infarction with crushing pain and left arm radiation. Immediate emergency care required.",
        "red_flags_expected": ["crushing_pain", "left_arm_radiation"],
        "turn_count": 1,
        "expert_confidence": 98
    }
]
