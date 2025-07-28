"""
NICE Lookup Service for Fairdoc AI
Provides lightweight symptom-to-protocol retrieval under 120 LOC.
"""
from __future__ import annotations

import re
from functools import cached_property
from typing import Dict, List, Optional

import structlog

# Seed data is defined in the database layer to avoid duplication.
from src.app2.models.database.nice_protocols import NICE_SEED_DATA

logger = structlog.get_logger(__name__)


class NICELookupService:  # pylint: disable=too-few-public-methods
    """Map free-text symptoms to a NICE guideline snippet.

    The service relies on *exact* keyword matches against the
    ``primary_symptoms`` field inside each protocol.  This keeps
    complexity low while allowing future extension to fuzzy search or
    vector retrieval without touching the public interface.
    """

    def __init__(self, seed_data: Optional[List[Dict]] = None) -> None:
        self._protocols = seed_data or NICE_SEED_DATA

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @cached_property
    def _index(self) -> Dict[str, Dict]:
        """Lower-case keyword → protocol mapping built once per instance."""
        index: Dict[str, Dict] = {}
        for proto in self._protocols:
            for symptom in proto["primary_symptoms"]:
                index[symptom.lower()] = proto
        logger.debug("NICE index built", size=len(index))
        return index

    @staticmethod
    def _clean_text(text: str) -> List[str]:
        """Tokenise *very* roughly—good enough for keyword lookup."""
        return re.findall(r"[a-zA-Z_]+", text.lower())

    @staticmethod
    def _format_protocol(proto: Dict) -> str:
        """Flatten the protocol dict into a readable multi-line string."""
        questions = proto["initial_questions"] + proto["follow_up_questions"]
        return (
            f"Condition: {proto['condition_name']}\n"
            f"Emergency criteria: {proto['emergency_criteria']}\n"
            f"Routine criteria: {proto['routine_criteria']}\n"
            f"Self-care criteria: {proto['self_care_criteria']}\n"
            "Questions:\n  • " + "\n  • ".join(questions)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find_relevant_protocols(self, symptom_text: str) -> Dict[str, str]:
        """Return the first matching protocol for *symptom_text*.

        The caller always receives a mapping with *protocol_code* and
        *protocol_text* keys, even when no match is found (values set to
        "NONE" / "").
        """
        for token in self._clean_text(symptom_text):
            proto = self._index.get(token)
            if proto:
                logger.info(
                    "Protocol matched", keyword=token, code=proto["protocol_code"]
                )
                return {
                    "protocol_code": proto["protocol_code"],
                    "protocol_text": self._format_protocol(proto),
                }

        logger.warning("No NICE protocol match", text=symptom_text)
        return {"protocol_code": "NONE", "protocol_text": ""}
