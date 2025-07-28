"""
Integration tests for Fairdoc AI V2 multi-turn chat flow (≤200 LOC).

Runs against a real FastAPI app instance but patches heavy dependencies
so the tests remain fast and deterministic.  Requires pytest-asyncio and
httpx.

File: src/tests/integration/test_multiturn_chat_integration.py
"""
from __future__ import annotations

import uuid
from typing import Dict, Any

import pytest
from httpx import AsyncClient
from fastapi import status

# FastAPI application entry point
from src.app2.main_v2 import app

# ---------------------------------------------------------------------------
# Lightweight mocks for heavy services (DSPy / Redis / NICE)
# ---------------------------------------------------------------------------
class _FakeAgent:  # no DSPy / network calls
    async def process_turn(self, symptoms: str, nice_context: str = "", **_) -> Dict[str, Any]:
        """Return deterministic outputs based on keywords."""
        if "chest" in symptoms.lower():
            return {
                "outcome": "emergency",
                "confidence": 95,
                "next_question": None,
                "reasoning": "Chest pain -> possible MI",
                "red_flags": ["chest_pain"],
                "is_complete": True,
            }
        return {
            "outcome": "inconclusive",
            "confidence": 60,
            "next_question": "How long have you had the headache?",
            "reasoning": "Need duration",
            "red_flags": [],
            "is_complete": False,
        }

    def reset_conversation(self):
        pass


class _MemoryQueue:
    """In-memory substitute for Redis ConversationQueue."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        pass

    async def start_conversation(self, user_id: str, initial_symptoms: str) -> str:
        conv_id = f"conv_{uuid.uuid4()}"
        self._store[conv_id] = {
            "conversation_id": conv_id,
            "user_id": user_id,
            "turn_count": 0,
            "status": "active",
            "conversation_history": [],
            "current_outcome": "inconclusive",
        }
        return conv_id

    async def update_conversation_turn(self, conversation_id: str, user_response: str, agent_result: Dict):
        state = self._store[conversation_id]
        state["turn_count"] += 1
        state["current_outcome"] = agent_result["outcome"]
        return state

    async def get_conversation_state(self, conversation_id: str):
        return self._store.get(conversation_id)


class _DummyLookup:
    def find_relevant_protocols(self, _sym: str):
        return {"protocol_code": "NONE", "protocol_text": ""}


class _DummyRouter:
    async def route_message(self, *_, **__):
        return []


# ---------------------------------------------------------------------------
# Dependency monkeypatch helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_dependencies(monkeypatch):
    """Patch heavy services with lightweight stand-ins for all tests."""
    from src.app2.core import dependencies_v2 as deps

    monkeypatch.setattr(deps, "_medical_agent", _FakeAgent())
    monkeypatch.setattr(deps, "_conversation_queue", _MemoryQueue())
    monkeypatch.setattr(deps, "_nice_lookup", _DummyLookup())
    monkeypatch.setattr(deps, "_stakeholder_router", _DummyRouter())

    # Ensure API layer receives patched deps
    monkeypatch.setattr(deps, "get_medical_agent", lambda: deps._medical_agent)
    monkeypatch.setattr(deps, "get_conversation_queue", lambda: deps._conversation_queue)
    monkeypatch.setattr(deps, "get_nice_lookup", lambda: deps._nice_lookup)
    monkeypatch.setattr(deps, "get_stakeholder_router", lambda: deps._stakeholder_router)


# ---------------------------------------------------------------------------
# Async HTTP client fixture
# ---------------------------------------------------------------------------
@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_happy_path_headache_flow(client):
    """Start new conversation, expect follow-up question then reset."""
    payload = {
        "user_message": "I have a headache",
        "stakeholder_role": "patient",
    }
    r1 = await client.post("/api/v2/medical/chat", json=payload)
    assert r1.status_code == status.HTTP_200_OK
    data1 = r1.json()
    conv_id = data1["conversation_id"]
    assert data1["medical_outcome"] == "inconclusive"
    assert data1["next_question"]

    # Follow-up turn should return same conv_id
    payload2 = {
        "conversation_id": conv_id,
        "user_message": "It started today",
        "stakeholder_role": "patient",
    }
    r2 = await client.post("/api/v2/medical/chat", json=payload2)
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conv_id

    # Fetch state endpoint
    s = await client.get(f"/api/v2/medical/chat/{conv_id}/state")
    assert s.status_code == 200
    assert s.json()["turn_count"] >= 1

    # Reset conversation
    res = await client.post(f"/api/v2/medical/chat/{conv_id}/reset")
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_emergency_chest_pain(client):
    """Chest pain should trigger emergency outcome immediately."""
    payload = {
        "user_message": "Crushing chest pain and sweating",
        "stakeholder_role": "patient",
    }
    resp = await client.post("/api/v2/medical/chat", json=payload)
    body = resp.json()
    assert body["medical_outcome"] == "emergency"
    assert body["is_conversation_complete"] is True
    assert body["red_flags"]


@pytest.mark.asyncio
async def test_invalid_conversation_id_returns_404(client):
    bad_id = str(uuid.uuid4())
    r = await client.get(f"/api/v2/medical/chat/{bad_id}/history")
    assert r.status_code == status.HTTP_404_NOT_FOUND
