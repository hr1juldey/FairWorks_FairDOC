# V2 Integration Tests - Comprehensive Test Suite
# Tests the complete V2 medical triage system integration
# File: src/tests/integration/test_v2_integration.py


import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from fastapi.testclient import TestClient
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import json
from datetime import datetime, timedelta
import uuid


# Import V2 application and services
from src.app2.main_v2 import app
from src.app2.core.config_v2 import settings_v2
from src.app2.core.database_v2 import database_v2, BaseV2
from src.app2.models.schemas.multiturn_chat import (
    MultiTurnChatRequest,
    MultiTurnChatResponse,
    StakeholderRole,
    ChatProvider,
    MedicalOutcome
)
from src.app2.models.schemas.medical_triage import (
    MedicalOutcome as TriageOutcome,
    RedFlagIndicator,
    TriageDecision
)
from src.app2.services.dspy.medical_agent import MedicalTriageAgent
from src.app2.services.context.redis_queue import ConversationQueue
from src.app2.services.context.nice_lookup import NICELookupService
from src.app2.services.chat.stakeholder_router import StakeholderRouter
from src.app2.services.chat.chat_orchestrator import ChatOrchestrator


# Test Configuration
TEST_DB_URL = "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_fairdoc_v2"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use separate Redis DB for tests


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=300
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(BaseV2.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(BaseV2.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine):
    """Create test database session"""
    async_session = sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_redis():
    """Create test Redis connection"""
    redis_client = redis.from_url(TEST_REDIS_URL, decode_responses=True)
    
    # Clear test database
    await redis_client.flushdb()
    
    yield redis_client
    
    # Cleanup
    await redis_client.flushdb()
    await redis_client.close()


@pytest.fixture
def mock_dspy_agent():
    """Mock DSPy medical agent for testing"""
    agent = MagicMock(spec=MedicalTriageAgent)
    
    async def mock_process_turn(symptoms, nice_context=""):
        """Mock agent response based on symptoms"""
        if "chest pain" in symptoms.lower():
            return {
                "outcome": "emergency",
                "confidence": 95,
                "next_question": None,
                "reasoning": "Chest pain requires immediate evaluation",
                "red_flags": ["chest_pain", "potential_cardiac_event"],
                "is_complete": True
            }
        elif "headache" in symptoms.lower():
            return {
                "outcome": "inconclusive", 
                "confidence": 60,
                "next_question": "How long have you had this headache?",
                "reasoning": "Need more information about headache duration",
                "red_flags": [],
                "is_complete": False
            }
        else:
            return {
                "outcome": "inconclusive",
                "confidence": 50,
                "next_question": "Can you describe your symptoms in more detail?",
                "reasoning": "Insufficient information for diagnosis",
                "red_flags": [],
                "is_complete": False
            }
    
    agent.process_turn = AsyncMock(side_effect=mock_process_turn)
    agent.reset_conversation = MagicMock()
    
    return agent


@pytest.fixture
async def conversation_queue(test_redis):
    """Create test conversation queue"""
    queue = ConversationQueue()
    queue.redis = test_redis
    return queue


@pytest.fixture
def nice_lookup():
    """Create NICE lookup service"""
    return NICELookupService()


@pytest.fixture
def stakeholder_router():
    """Create stakeholder router"""
    return StakeholderRouter()


@pytest.fixture
async def chat_orchestrator(mock_dspy_agent, conversation_queue, nice_lookup, stakeholder_router):
    """Create chat orchestrator with mocked dependencies"""
    orchestrator = ChatOrchestrator()
    orchestrator.medical_agent = mock_dspy_agent
    orchestrator.conversation_queue = conversation_queue
    orchestrator.nice_lookup = nice_lookup
    orchestrator.stakeholder_router = stakeholder_router
    return orchestrator


@pytest.fixture
async def test_client():
    """Create FastAPI test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestV2DatabaseIntegration:
    """Test V2 database integration"""
    
    async def test_database_connection(self, test_session):
        """Test database connectivity and basic operations"""
        from sqlalchemy import text
        
        # Test basic connection
        result = await test_session.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        assert row[0] == 1
        
        # Test transaction
        await test_session.execute(text("CREATE TEMPORARY TABLE test_table (id INTEGER)"))
        await test_session.execute(text("INSERT INTO test_table (id) VALUES (1)"))
        
        result = await test_session.execute(text("SELECT id FROM test_table"))
        row = result.fetchone()
        assert row[0] == 1


    async def test_database_session_rollback(self, test_session):
        """Test database session rollback functionality"""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError
        
        # Setup: Create table and insert first record outside pytest.raises
        await test_session.execute(text("CREATE TEMPORARY TABLE test_rollback (id INTEGER PRIMARY KEY)"))
        await test_session.execute(text("INSERT INTO test_rollback (id) VALUES (1)"))
        
        # Only the operation that should fail goes inside pytest.raises (single statement)
        with pytest.raises(IntegrityError, match="duplicate key"):
            await test_session.execute(text("INSERT INTO test_rollback (id) VALUES (1)"))  # Duplicate key


class TestV2RedisIntegration:
    """Test V2 Redis integration"""
    
    async def test_redis_connection(self, test_redis):
        """Test Redis connectivity"""
        await test_redis.set("test_key", "test_value")
        value = await test_redis.get("test_key")
        assert value == "test_value"
        
        await test_redis.delete("test_key")
        value = await test_redis.get("test_key")
        assert value is None


    async def test_conversation_queue_operations(self, conversation_queue):
        """Test conversation queue Redis operations"""
        # Start conversation
        conv_id = await conversation_queue.start_conversation(
            user_id="test_user_123",
            initial_symptoms="I have a headache"
        )
        
        assert conv_id.startswith("conv_test_user_123_")
        
        # Get conversation state
        state = await conversation_queue.get_conversation_state(conv_id)
        assert state is not None
        assert state["user_id"] == "test_user_123"
        assert state["initial_symptoms"] == "I have a headache"
        assert state["status"] == "active"
        
        # Update conversation turn
        agent_result = {
            "outcome": "inconclusive",
            "confidence": 60,
            "next_question": "How long have you had this headache?",
            "reasoning": "Need more information",
            "red_flags": [],
            "is_complete": False
        }
        
        updated_state = await conversation_queue.update_conversation_turn(
            conversation_id=conv_id,
            user_response="About 2 hours",
            agent_result=agent_result
        )
        
        assert updated_state["turn_count"] == 2
        assert len(updated_state["conversation_history"]) == 1


class TestV2ServicesIntegration:
    """Test V2 services integration"""
    
    async def test_nice_lookup_integration(self, nice_lookup):
        """Test NICE lookup service"""
        # Test headache lookup
        result = nice_lookup.find_relevant_protocols("I have a severe headache")
        assert result["protocol_code"] == "NG127_HEADACHE"
        assert "headache" in result["protocol_text"].lower()
        
        # Test chest pain lookup  
        result = nice_lookup.find_relevant_protocols("chest pain and shortness of breath")
        assert result["protocol_code"] == "CG95_CHEST_PAIN"
        assert "chest pain" in result["protocol_text"].lower()
        
        # Test no match
        result = nice_lookup.find_relevant_protocols("random unrelated symptoms")
        assert result["protocol_code"] == "NONE"
        assert result["protocol_text"] == ""


    async def test_stakeholder_router_integration(self, stakeholder_router):
        """Test stakeholder routing logic"""
        conv_id = "test_conv_123"
        
        # Test patient message routing
        routes = await stakeholder_router.route_message(
            conversation_id=conv_id,
            from_stakeholder="patient",
            message="I have chest pain",
            medical_outcome="emergency"
        )
        
        assert len(routes) == 2  # To agent + emergency route to doctor
        assert routes[0].to_stakeholder == "fairdoc_agent"
        assert routes[1].to_stakeholder == "doctor"
        assert routes[1].priority == "emergency"
        
        # Test routine routing
        routes = await stakeholder_router.route_message(
            conversation_id=conv_id,
            from_stakeholder="fairdoc_agent",
            message="You should see a doctor",
            medical_outcome="routine_doctor"
        )
        
        assert len(routes) == 2  # To patient + to doctor
        assert any(route.to_stakeholder == "patient" for route in routes)
        assert any(route.to_stakeholder == "doctor" for route in routes)


    async def test_chat_orchestrator_integration(self, chat_orchestrator):
        """Test chat orchestrator end-to-end flow"""
        # Create test request
        request = MultiTurnChatRequest(
            user_message="I have severe chest pain",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="test_patient_456",
            chat_provider=ChatProvider.API_DIRECT
        )
        
        # Process conversation turn
        result = await chat_orchestrator.process_conversation_turn(request)
        
        # Verify orchestration result
        assert "conversation_id" in result
        assert "agent_result" in result
        assert "updated_state" in result
        assert "nice_context" in result
        assert "message_routes" in result
        
        # Check agent result
        agent_result = result["agent_result"]
        assert agent_result["outcome"] == "emergency"
        assert agent_result["confidence"] == 95
        assert agent_result["is_complete"] is True
        
        # Check emergency alert flag
        assert result["requires_emergency_alert"] is True
        
        # Build chat response
        response = chat_orchestrator.build_chat_response(result)
        assert isinstance(response, MultiTurnChatResponse)
        assert response.medical_outcome == MedicalOutcome.EMERGENCY
        assert response.confidence_score == 95


class TestV2APIIntegration:
    """Test V2 API endpoints integration"""
    
    @patch('src.app2.core.dependencies_v2.get_medical_agent')
    @patch('src.app2.core.dependencies_v2.get_conversation_queue')
    @patch('src.app2.core.dependencies_v2.get_nice_lookup')
    @patch('src.app2.core.dependencies_v2.get_stakeholder_router')
    async def test_multiturn_chat_endpoint(
        self, 
        mock_stakeholder_router,
        mock_nice_lookup,
        mock_conversation_queue,
        mock_medical_agent,
        test_client
    ):
        """Test multi-turn chat API endpoint"""
        # Setup mocks
        mock_medical_agent.return_value = mock_dspy_agent
        mock_conversation_queue.return_value = conversation_queue
        mock_nice_lookup.return_value = nice_lookup
        mock_stakeholder_router.return_value = stakeholder_router
        
        # Test request payload
        payload = {
            "user_message": "I have a headache",
            "stakeholder_role": "patient",
            "stakeholder_id": "test_patient_789",
            "chat_provider": "api_direct"
        }
        
        # Make API request
        response = await test_client.post(
            "/api/v2/medical/chat",
            json=payload
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "conversation_id" in data
        assert "agent_message" in data
        assert "medical_outcome" in data
        assert "confidence_score" in data
        assert "turn_count" in data


    async def test_health_check_endpoint(self, test_client):
        """Test V2 health check endpoint"""
        response = await test_client.get("/api/v2/health")
        
        # May return 503 if services not initialized, but should respond
        assert response.status_code in [200, 503]
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "services" in data


    async def test_api_info_endpoint(self, test_client):
        """Test V2 API info endpoint"""
        response = await test_client.get("/api/v2/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Fairdoc AI Triage System V2"
        assert data["version"] == "v2.6-stable"
        assert "features" in data
        assert "endpoints" in data


class TestV2ErrorHandling:
    """Test V2 error handling and edge cases"""
    
    async def test_invalid_conversation_id(self, chat_orchestrator):
        """Test handling of invalid conversation ID"""
        request = MultiTurnChatRequest(
            conversation_id=uuid.uuid4(),  # Non-existent ID
            user_message="Test message",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="test_patient"
        )
        
        # Fixed: Use pytest.raises with match parameter
        with pytest.raises(ValueError, match="Conversation .* not found"):
            await chat_orchestrator.process_conversation_turn(request)


    async def test_redis_connection_failure(self, conversation_queue):
        """Test Redis connection failure handling"""
        # Close Redis connection to simulate failure
        await conversation_queue.redis.close()
        
        # Fixed: Use more specific exception instead of bare Exception
        with pytest.raises((redis.ConnectionError, redis.RedisError)):
            await conversation_queue.start_conversation(
                user_id="test_user",
                initial_symptoms="test symptoms"
            )


    async def test_empty_message_handling(self, chat_orchestrator):
        """Test handling of empty messages"""
        request = MultiTurnChatRequest(
            user_message="",  # Empty message
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="test_patient"
        )
        
        # Fixed: Use pytest.raises instead of try-except-assert
        with pytest.raises(ValueError, match="empty|required"):
            await chat_orchestrator.process_conversation_turn(request)


class TestV2DataFlow:
    """Test complete V2 data flow scenarios"""
    
    async def test_emergency_scenario_flow(self, chat_orchestrator):
        """Test complete emergency scenario data flow"""
        # Step 1: Patient reports emergency symptoms
        request = MultiTurnChatRequest(
            user_message="I have crushing chest pain radiating to my left arm and I'm sweating",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="emergency_patient_001"
        )
        
        result = await chat_orchestrator.process_conversation_turn(request)
        
        # Verify emergency detection
        assert result["agent_result"]["outcome"] == "emergency"
        assert result["requires_emergency_alert"] is True
        
        # Check message routing includes emergency route
        routes = result["message_routes"]
        emergency_routes = [r for r in routes if r.priority == "emergency"]
        assert len(emergency_routes) > 0
        
        # Verify NICE protocol integration
        nice_context = result["nice_context"]
        assert nice_context["protocol_code"] != "NONE"


    async def test_multi_turn_conversation_flow(self, chat_orchestrator):
        """Test multi-turn conversation flow"""
        # Turn 1: Initial symptoms
        request1 = MultiTurnChatRequest(
            user_message="I have a headache",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="patient_multiturn_001"
        )
        
        result1 = await chat_orchestrator.process_conversation_turn(request1)
        conv_id = result1["conversation_id"]
        
        # Should be inconclusive and ask follow-up
        assert result1["agent_result"]["outcome"] == "inconclusive"
        assert result1["agent_result"]["next_question"] is not None
        
        # Turn 2: Follow-up response
        request2 = MultiTurnChatRequest(
            conversation_id=uuid.UUID(conv_id),
            user_message="It started about 2 hours ago and it's getting worse",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="patient_multiturn_001"
        )
        
        result2 = await chat_orchestrator.process_conversation_turn(request2)
        
        # Verify conversation continuity
        assert result2["conversation_id"] == conv_id
        assert result2["updated_state"]["turn_count"] >= 2


@pytest.mark.asyncio
class TestV2Performance:
    """Test V2 system performance characteristics"""
    
    async def test_concurrent_conversations(self, chat_orchestrator):
        """Test handling multiple concurrent conversations"""
        import asyncio
        
        async def create_conversation(user_id):
            request = MultiTurnChatRequest(
                user_message=f"Test message from user {user_id}",
                stakeholder_role=StakeholderRole.PATIENT,
                stakeholder_id=f"concurrent_user_{user_id}"
            )
            return await chat_orchestrator.process_conversation_turn(request)
        
        # Create 5 concurrent conversations
        tasks = [create_conversation(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent conversation failed: {result}")
            assert "conversation_id" in result


    async def test_response_time_benchmark(self, chat_orchestrator):
        """Basic response time benchmark"""
        import time
        
        request = MultiTurnChatRequest(
            user_message="Quick test message",
            stakeholder_role=StakeholderRole.PATIENT,
            stakeholder_id="benchmark_patient"
        )
        
        start_time = time.time()
        result = await chat_orchestrator.process_conversation_turn(request)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 2 seconds (with mocked DSPy)
        assert response_time < 2.0
        assert "conversation_id" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
