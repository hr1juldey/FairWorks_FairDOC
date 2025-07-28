"""
Unit tests for src/app2/services/context/redis_queue.py
------------------------------------------------------

This test suite validates the behaviour of the **ConversationQueue**
service without requiring a real Redis server.  It relies on
`fakeredis.aioredis.FakeRedis`, patching the `Redis.from_url` call used
inside the implementation so that all Redis commands operate on an
in-memory store.

Key Scenarios Covered
~~~~~~~~~~~~~~~~~~~~~
1. **Initialisation** – ``initialize`` creates and pings the connection.
2. **Conversation creation** – ``start_conversation`` persists the
   initial JSON document and adds the ID to the *pending* list.
3. **Turn update** – ``update_conversation_turn`` appends a new turn and
   when a terminal outcome is passed (``is_complete=True``) the
   conversation is marked *completed* and moved to the *completed* list.

The test file is kept well below the 200-line limit (≈120 LOC) and uses
``pytest-asyncio`` for async test execution.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Dict

import pytest
import pytest_asyncio
from fakeredis import aioredis
from redis.asyncio import Redis

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from src.app2.services.context.redis_queue import ConversationQueue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="function")
async def fake_redis() -> AsyncIterator[Redis]:
    """Provide an in-memory Redis replacement for each test."""
    async with aioredis.FakeRedis() as client:
        yield client


@pytest_asyncio.fixture(scope="function", autouse=True)
async def patched_redis(monkeypatch: pytest.MonkeyPatch, fake_redis: Redis):
    """Patch ``Redis.from_url`` to return *fake_redis* instead of a real pool."""

    async def _from_url(url: str, decode_responses: bool = True):  # noqa: D401
        # ``fakeredis`` already works with decoded responses.
        return fake_redis

    monkeypatch.setattr("src.app2.services.context.redis_queue.Redis.from_url", _from_url)
    yield


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
USER_ID = "user123"
SYMPTOMS = "severe headache and nausea"


def _minimal_agent_result(complete: bool = False) -> Dict:
    """Return a synthetic agent_result dictionary mimicking *MedicalTriageAgent* output."""
    return {
        "outcome": "inconclusive" if not complete else "self_care",
        "confidence": 80,
        "next_question": None if complete else "Have you taken any painkillers?",
        "reasoning": "test-reasoning",
        "red_flags": [],
        "is_complete": complete,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_start_conversation_persists_state(fake_redis: Redis):
    queue = ConversationQueue()
    await queue.initialize()

    conversation_id = await queue.start_conversation(USER_ID, SYMPTOMS)

    # Validate Redis keys
    state_raw = await fake_redis.get(f"{queue.state_prefix}:{conversation_id}")
    assert state_raw is not None, "Conversation state not stored in Redis"

    state = json.loads(state_raw)
    assert state["user_id"] == USER_ID
    assert state["turn_count"] == 1

    # Validate pending queue
    pending_id = await fake_redis.lpop(f"{queue.queue_prefix}:pending")
    assert pending_id == conversation_id


@pytest.mark.asyncio
async def test_update_conversation_turn_marks_completed(fake_redis: Redis):
    queue = ConversationQueue()
    await queue.initialize()

    conversation_id = await queue.start_conversation(USER_ID, SYMPTOMS)

    # Act – append a *completed* turn
    await queue.update_conversation_turn(
        conversation_id, "reply text", _minimal_agent_result(complete=True)
    )

    # State should now be completed and on *completed* list
    state = await queue.get_conversation_state(conversation_id)
    assert state is not None
    assert state["status"] == "completed"

    completed_id = await fake_redis.lpop(f"{queue.queue_prefix}:completed")
    assert completed_id == conversation_id

    # Ensure turn history grew
    assert len(state["conversation_history"]) == 2  # initial + new turn


@pytest.mark.asyncio
async def test_update_conversation_turn_appends_history(fake_redis: Redis):
    queue = ConversationQueue()
    await queue.initialize()

    conversation_id = await queue.start_conversation(USER_ID, SYMPTOMS)

    # Append an *ongoing* turn
    await queue.update_conversation_turn(
        conversation_id, "reply", _minimal_agent_result(complete=False)
    )

    state = await queue.get_conversation_state(conversation_id)
    assert state["turn_count"] == 2
    assert state["current_outcome"] == "inconclusive"

    # Last turn stored should match user response
    last_turn = state["conversation_history"][-1]
    assert last_turn["user_response"] == "reply"
    assert last_turn["outcome"] == "inconclusive"
