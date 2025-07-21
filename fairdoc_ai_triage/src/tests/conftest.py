"""
Medical AI Triage System - Test Configuration
"""

import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.app.core.config import get_test_settings
from src.app.core.database import Base, get_db_session
from src.app.main import app

# Test settings
test_settings = get_test_settings()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        echo=True,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session(test_engine):
    """Create test database session"""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def test_client(test_db_session):
    """Create test client with database dependency override"""
    
    async def override_get_db():
        yield test_db_session
    
    app.dependency_overrides[get_db_session] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_medical_query():
    """Sample medical query for testing"""
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
def sample_context_data():
    """Sample context data for testing"""
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
