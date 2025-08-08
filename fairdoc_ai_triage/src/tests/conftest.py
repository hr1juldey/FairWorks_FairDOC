"""
Fairdoc AI Triage System - Test Configuration
Clean, modular test setup for V1/V2 compatibility testing
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import Dict, Any, Generator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from unittest.mock import AsyncMock
from src.app2.core.config_v2 import settings_v2
import structlog
# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

logger = structlog.get_logger(__name__)


# Centralized test environment variables
TEST_ENV_VARS = {
    "APP_NAME": "Fairdoc AI Triage System",
    "NEXT_GEN": "true",
    "ENVIRONMENT": "testing",
    "DEBUG": "true",
    "SECRET_KEY": "test-secret-key-123",
    "ALLOWED_HOSTS": '["localhost", "127.0.0.1"]',
    "ALLOWED_ORIGINS": '["http://localhost:3000"]',
    "DATABASE_URL": "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db",
    "TEST_DATABASE_URL": "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db_test",
    "REDIS_URL": "redis://:test_pass@localhost:6379/0",
    "MINIO_ENDPOINT": "localhost:9000",
    "MINIO_ACCESS_KEY": "test_minio",
    "MINIO_SECRET_KEY": "test_minio_key",
    "MINIO_BUCKET_NAME": "test-bucket",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL": settings_v2.DSPY_MODEL_NAME,
    "RAVEN_WEBHOOK_URL": "http://localhost:8080/test",
    "RAVEN_API_KEY": "test-raven-key",
    "RAVEN_SECRET": "test-raven-secret",
    "JWT_SECRET_KEY": "test-jwt-secret",
    "CELERY_BROKER_URL": "redis://:test_pass@localhost:6379/1",
    "CELERY_RESULT_BACKEND": "redis://:test_pass@localhost:6379/2",
    # V2 specific variables
    "FAIRDOC_V2_ENABLED": "true",
    "FAIRDOC_V2_DSPy_MODEL": settings_v2.FAIRDOC_V2_DSPy_MODEL,
    "FAIRDOC_V2_MAX_CONVERSATION_TURNS": "8",
    "TELEGRAM_BOT_TOKEN": "test-telegram-token",
}

# === Core Test Infrastructure ===

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up isolated test environment for all tests"""
    original_env = os.environ.copy()
    os.environ.update(TEST_ENV_VARS)
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# Replace this in your conftest.py

@pytest.fixture(scope='session')
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
@pytest.fixture(scope="session", autouse=True)
def shared_dspy_config() -> Generator[None, None, None]:
    """
    Establishes a single DSPy configuration for the entire test session.
    This is the core fix for the multiple instance creation problem.
    The 'autouse=True' ensures this fixture runs automatically for the session.
    """
    logger.info("🤖 Initializing shared DSPy configuration for test session...")
    from src.app2.core.dspy_config_v2 import ensure_dspy_configured, get_llm_provider

    success = ensure_dspy_configured(settings_v2.FAIRDOC_V2_DSPy_MODEL)
    if not success:
        pytest.fail("Critical error: Failed to configure shared DSPy for tests.", pytrace=False)
    
    provider = get_llm_provider()
    logger.info(f"✅ Shared DSPy configured. Default model: {provider._default_model}")
    
    yield
    
    logger.info("🧹 Tearing down shared DSPy test session.")

@pytest.fixture(autouse=True)
def reset_dspy_state():
    """Reset DSPy state between tests to avoid interference"""
    import dspy
    # Reset any global DSPy configuration
    dspy.configure(lm=None)
    yield
    # Cleanup after test
    dspy.configure(lm=None)


# === V1 Database Test Fixtures ===

@pytest_asyncio.fixture(scope="session")
async def test_engine_v1():
    """Create V1 test database engine with proper cleanup"""
    from src.app.core.config import get_test_settings
    from src.app.core.database import Base
    
    test_settings = get_test_settings()
    engine = create_async_engine(
        test_settings.database_url,
        echo=False,
        pool_pre_ping=True
    )
    
    # Setup database schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup database schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session_v1(test_engine_v1):
    """Create isolated V1 database session per test"""
    async_session = sessionmaker(
        test_engine_v1, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

# === V1 API Test Fixtures ===

@pytest_asyncio.fixture
async def test_client_v1(test_db_session_v1):
    """Create V1 test client with database override"""
    from src.app.main import app
    from src.app.core.database import get_db_session
    
    async def override_get_db():
        yield test_db_session_v1
    
    app.dependency_overrides[get_db_session] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()

# === V2 Test Fixtures ===

@pytest_asyncio.fixture
async def test_client_v2():
    """Create V2 test client when available"""
    try:
        from src.app2.main_v2 import app_v2
        async with AsyncClient(app=app_v2, base_url="http://test") as client:
            yield client
    except ImportError:
        # V2 not available in test environment
        yield None

@pytest_asyncio.fixture
async def mock_redis_v2():
    """Mock Redis client for V2 testing"""
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.get.return_value = None
    mock_redis.lpush.return_value = 1
    yield mock_redis

@pytest_asyncio.fixture
async def mock_ollama_service():
    """Mock Ollama AI service for testing"""
    mock_service = AsyncMock()
    mock_service.generate_response.return_value = {
        "response": "Test medical response",
        "confidence": 85,
        "reasoning": "Test reasoning for medical decision"
    }
    yield mock_service

# === Sample Test Data Fixtures (Fixed PT022) ===

@pytest.fixture
def sample_medical_query():
    """Sample medical query for V1/V2 testing"""
    return {
        "user_id": "test_user_123",
        "message": "I have been experiencing chest pain for the last hour",
        "timestamp": "2024-01-15T10:30:00Z",
        "metadata": {
            "source": "raven_chat",
            "urgency": "high"
        }
    }

@pytest.fixture
def sample_multiturn_conversation():
    """Sample multi-turn conversation for V2 testing"""
    return {
        "conversation_id": "conv_test_123",
        "user_id": "test_user_123",
        "turns": [
            {
                "turn": 1,
                "user_message": "I have chest pain",
                "agent_response": "Can you describe the pain? Is it sharp or crushing?",
                "outcome": "inconclusive",
                "confidence": 60
            },
            {
                "turn": 2,
                "user_message": "It's crushing and radiates to my left arm",
                "agent_response": "This requires immediate medical attention",
                "outcome": "emergency",
                "confidence": 95
            }
        ]
    }

@pytest.fixture
def sample_context_data():
    """Sample medical context data for testing"""
    return {
        "conversation_id": "conv_123",
        "user_id": "test_user_123",
        "medical_history": {
            "conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "allergies": ["penicillin"]
        },
        "current_symptoms": ["chest pain", "shortness of breath"]
    }

@pytest.fixture
def sample_nice_protocol():
    """Sample NICE protocol data for V2 testing"""
    return {
        "protocol_code": "CG95_CHEST_PAIN_TEST",
        "condition_name": "Chest Pain Assessment",
        "primary_symptoms": ["chest_pain", "chest_discomfort"],
        "red_flag_symptoms": ["crushing_pain", "left_arm_radiation", "sweating"],
        "initial_questions": [
            "Is the pain crushing or heavy?",
            "Does it radiate to your arm or jaw?"
        ],
        "emergency_criteria": "Crushing chest pain with radiation, sweating, SOB",
        "evidence_level": "A"
    }

# === Mock Services and Utilities (Fixed PT022) ===

@pytest.fixture
def mock_external_services():
    """Mock external chat services for testing"""
    return {
        "raven_chat": AsyncMock(),
        "telegram": AsyncMock(),
        "whatsapp": AsyncMock()
    }

@pytest.fixture
def test_headers():
    """Standard HTTP headers for API testing"""
    return {
        "Content-Type": "application/json",
        "X-Test-Environment": "true",
        "User-Agent": "Fairdoc-Test-Client/1.0"
    }

@pytest.fixture
def clean_env():
    """Clean environment fixture for isolated config testing"""
    original_env = os.environ.copy()
    # Clear specific environment variables
    for key in list(os.environ.keys()):
        if key.startswith(('APP_', 'NEXT_', 'FAIRDOC_', 'DATABASE_', 'REDIS_')):
            del os.environ[key]
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
