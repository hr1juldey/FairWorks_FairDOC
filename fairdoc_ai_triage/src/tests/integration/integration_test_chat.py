# Integration Test for V2 Chat Endpoint

# File path: src/tests/integration/test_chat_api.py


"""
Integration test for the V2 multi-turn chat endpoint.
Uses FastAPI's AsyncClient with dependency overrides so the test runs
completely offline (no Redis / DSPy / Postgres required).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from uuid import uuid4

from src.app2.main_v2 import app
from src.app2.services.chat.chat_orchestrator import ChatOrchestrator
from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatResponse,
    MedicalOutcome,
)

# ---------------------------------------------------------------------------
# Test helpers / stubs
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_orchestrator(monkeypatch):
    """Patch ChatOrchestrator so we do not hit Redis or DSPy during tests."""

    async def _fake_process(self, request):  # pylint: disable=unused-argument
        """Return deterministic orchestrator result."""
        return {
            "conversation_id": "conv_test_1",
            "agent_result": {
                "outcome": "inconclusive",
                "confidence": 80,
                "next_question": "How long have you had the headache?",
                "reasoning": "",
                "red_flags": [],
                "is_complete": False,
            },
            "updated_state": {
                "turn_count": 1,
                "status": "active",
                "current_outcome": "inconclusive",
            },
            "nice_context": {
                "protocol_code": "NG127_HEADACHE",
                "protocol_text": "Condition: Headache\n...",
            },
            "message_routes": [],
            "requires_emergency_alert": False,
            "requires_persistence": False,
        }

    async def _fake_build(self, orchestration_result):  # pylint: disable=unused-argument
        """Return minimal valid API response so downstream asserts stay simple."""
        return MultiTurnChatResponse(
            conversation_id=uuid4(),
            medical_outcome=MedicalOutcome.INCONCLUSIVE,
            confidence_score=80.0,
            turn_count=1,
        )

    # Patch instance methods on ChatOrchestrator
    monkeypatch.setattr(ChatOrchestrator, "process_conversation_turn", _fake_process, raising=True)
    monkeypatch.setattr(ChatOrchestrator, "build_chat_response", _fake_build, raising=True)


# ---------------------------------------------------------------------------
# Actual integration test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_endpoint_integration(stub_orchestrator):  # noqa: D401 ‑ pylint: disable=unused-argument
    """Ensure /api/v2/medical/chat returns 200 and minimal response schema."""

    async with AsyncClient(app=app, base_url="http://testserver") as client:
        payload = {
            "user_message": "I have a severe headache.",
            "stakeholder_role": "patient",
        }
        response = await client.post("/api/v2/medical/chat", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["medical_outcome"] == "need_more_questions"
    assert body["confidence_score"] == 80.0
