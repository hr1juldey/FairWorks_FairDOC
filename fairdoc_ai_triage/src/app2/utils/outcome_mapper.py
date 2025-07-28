"""
Outcome Mapper Utility for Fairdoc AI V2
Ensures **1-to-1** mapping between `medical_triage.MedicalOutcome`
(used by DSPy agent & internal logic) and `multiturn_chat.MedicalOutcome`
(used by API DTOs). Centralising the mapping eliminates brittle
string-based conversions sprinkled across the codebase.

*Keeping this helper under 120 LOC facilitates unit-level validation*
without bloating higher-level orchestrator modules.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Union

from src.app2.models.schemas.medical_triage import MedicalOutcome as _TriageOutcome
from src.app2.models.schemas.multiturn_chat import MedicalOutcome as _ChatOutcome

__all__ = [
    "OutcomeMapper",
    "Outcome",  # unified alias for type hints
]

# ---------------------------------------------------------------------------
# Public alias used across services to accept either enum type
# ---------------------------------------------------------------------------
Outcome = Union[_TriageOutcome, _ChatOutcome]


class OutcomeMapper:  # pylint: disable=too-few-public-methods
    """Bidirectional enum mapper with runtime safeguards.

    Example
    -------
    >>> OutcomeMapper.to_chat(_TriageOutcome.EMERGENCY)
    <MedicalOutcome.EMERGENCY: 'emergency'>
    >>> OutcomeMapper.to_triage(_ChatOutcome.SELF_CARE)
    <MedicalOutcome.SELF_CARE: 'self_care_advice'>
    """

    # _canonical → (_TriageOutcome, _ChatOutcome)
    _MAP: Dict[str, tuple[_TriageOutcome, _ChatOutcome]] = {
        "emergency": (
            _TriageOutcome.EMERGENCY,
            _ChatOutcome.EMERGENCY,
        ),
        "routine_doctor": (
            _TriageOutcome.ROUTINE_DOCTOR,
            _ChatOutcome.ROUTINE_DOCTOR,
        ),
        "self_care": (
            _TriageOutcome.SELF_CARE,
            _ChatOutcome.SELF_CARE,
        ),
        "inconclusive": (
            _TriageOutcome.INCONCLUSIVE,
            _ChatOutcome.INCONCLUSIVE,
        ),
        "spam_detected": (
            _TriageOutcome.SPAM,
            _ChatOutcome.SPAM_DETECTED,
        ),
    }

    # -------- Validation at import time --------
    for _label, (_t, _c) in _MAP.items():  # pragma: no cover
        assert isinstance(_t, _TriageOutcome), _label
        assert isinstance(_c, _ChatOutcome), _label
    del _label, _t, _c

    # ---------------------------------------------------------------------
    # Conversion helpers
    # ---------------------------------------------------------------------
    @classmethod
    def canonical(cls, outcome: Outcome | str) -> str:
        """Return canonical *snake-case* label for *outcome*."""
        if isinstance(outcome, Enum):
            raw = outcome.value
        else:
            raw = str(outcome).lower()
        # Normalise common substrings
        for key in cls._MAP:
            if key in raw:
                return key
        return "inconclusive"  # safe fallback

    # High-level helpers ---------------------------------------------------
    @classmethod
    def to_chat(cls, outcome: Outcome | str) -> _ChatOutcome:
        """Convert to API-facing enum."""
        return cls._MAP[cls.canonical(outcome)][1]

    @classmethod
    def to_triage(cls, outcome: Outcome | str) -> _TriageOutcome:
        """Convert to DSPy-facing enum."""
        return cls._MAP[cls.canonical(outcome)][0]

    @classmethod
    def is_equivalent(cls, triage: _TriageOutcome, chat: _ChatOutcome) -> bool:
        """Check if two enum values represent the same logical outcome."""
        return cls.to_chat(triage) == chat

    # ------------------------------------------------------------------
    # Convenience synonyms used by orchestration layer (reduce imports)
    # ------------------------------------------------------------------
    to_api = to_chat
    to_internal = to_triage
