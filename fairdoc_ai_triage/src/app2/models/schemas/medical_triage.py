"""
Pydantic v2 schemas for medical triage decisions and conversation turns.
Keeps a strict ≤200 LOC budget while mirroring enums used by DSPy services.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "MedicalOutcome",
    "RedFlagIndicator",
    "TriageDecision",
    "ConversationTurn",
]


class MedicalOutcome(str, Enum):
    """Unified outcome labels across services (snake-case identifiers)."""

    EMERGENCY = "emergency_route_to_doctor"
    ROUTINE_DOCTOR = "routine_doctor_consultation"
    SELF_CARE = "self_care_advice"
    INCONCLUSIVE = "need_more_questions"
    SPAM = "spam_or_irrelevant"


class RedFlagIndicator(BaseModel):
    """Represents a single red-flag symptom detected during triage."""

    symptom: str = Field(..., min_length=1, description="Canonical name of red-flag symptom")
    critical: bool = Field(
        default=True, description="Whether this red flag alone triggers emergency routing"
    )

    @field_validator("symptom")
    def _strip_symptom(cls, v: str) -> str:  # noqa: D401
        return v.strip().lower()


class TriageDecision(BaseModel):
    """Decision output from *MedicalTriageAgent* or gold-standard dataset."""

    outcome: MedicalOutcome = Field(..., description="Final outcome classification")
    confidence: int = Field(..., ge=0, le=100, description="Confidence 0–100")
    next_question: Optional[str] = Field(
        None, description="Clarifying question or *None* if triage complete"
    )
    reasoning: Optional[str] = Field("", description="Model reasoning (optional)")
    red_flags: List[RedFlagIndicator] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    """Schema for a single Q⇄A pair in a medical triage conversation."""

    turn_number: int = Field(..., ge=1)

    user_message: str = Field(..., min_length=1, max_length=1_000)
    agent_response: Optional[str] = Field(None, description="Raw text sent to user")
    agent_question: Optional[str] = Field(None, description="Follow-up question asked")

    medical_outcome: MedicalOutcome = Field(
        default=MedicalOutcome.INCONCLUSIVE, description="Outcome *after* this turn"
    )
    confidence_score: int = Field(
        default=0, ge=0, le=100, description="Agent confidence after this turn"
    )

    red_flags: List[RedFlagIndicator] = Field(default_factory=list)
    reasoning: Optional[str] = Field(None, description="Agent reasoning snippet")

    class Config:
        """Enable orm-mode for seamless SQLAlchemy integration."""

        orm_mode = True
