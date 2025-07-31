"""
Unit Tests for Database Initialization Service
Tests database setup, seed data loading, and error handling
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.exc import IntegrityError, OperationalError

# Test environment setup
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'DATABASE_URL': 'postgresql+asyncpg://test:test@localhost/test',
    'REDIS_URL': 'redis://localhost:6379/0',
    'MINIO_ENDPOINT': 'localhost:9000',
    'MINIO_ACCESS_KEY': 'test',
    'MINIO_SECRET_KEY': 'test',
    'OLLAMA_BASE_URL': 'http://localhost:11434',
    'RAVEN_WEBHOOK_URL': 'http://localhost:8080/webhook',
    'RAVEN_API_KEY': 'test-key',
    'RAVEN_SECRET': 'test-secret',
    'JWT_SECRET_KEY': 'jwt-secret',
    'CELERY_BROKER_URL': 'redis://localhost:6379/1',
    'CELERY_RESULT_BACKEND': 'redis://localhost:6379/2'
}):
    from src.app2.services.database.initialization_service import (
        DatabaseInitializationService,
        database_initialization,
        initialize_database_on_startup
    )

@pytest.fixture
def db_service():
    """Create fresh database initialization service"""
    return DatabaseInitializationService()

@pytest.fixture
def mock_session():
    """Mock async database session"""
    session = AsyncMock()
    session.execute.return_value = MagicMock()
    session.commit.return_value = None
    return session

@pytest.fixture
def mock_nice_data():
    """Mock NICE protocol data"""
    return [
        {
            "protocol_code": "TEST_CHEST_PAIN",
            "condition_name": "Test Chest Pain",
            "category": "Cardiovascular",  # Will be filtered out
            "primary_symptoms": ["chest_pain"],
            "red_flag_symptoms": ["crushing_pain"],
            "initial_questions": ["When did pain start?"],
            "follow_up_questions": ["History of heart disease?"],
            "emergency_criteria": "Severe chest pain",
            "routine_criteria": "Mild chest pain",
            "self_care_criteria": "Minor discomfort",
            "evidence_level": "A",
            "fhir_code": "29857009"
        }
    ]

@pytest.fixture
def mock_gold_data():
    """Mock gold standards data"""
    return [
        {
            "title": "Test Emergency Scenario",
            "description": "Test emergency conversation",
            "primary_symptom": "chest_pain",
            "expected_outcome": "emergency_route_to_doctor",
            "patient_age": 58,
            "patient_gender": "male",
            "expected_red_flags": ["crushing_pain"],
            "should_escalate": True,
            "conversation_dialogue": [{"turn": 1, "test": "data"}],
            "relevant_protocols": ["TEST_CHEST_PAIN"],
            "minimum_confidence_threshold": 85.0,
            "expected_turn_count": 2,
            "max_acceptable_turns": 3,
            "created_by": "test_doctor"
        }
    ]

class TestDatabaseInitializationService:
    """Test database initialization service functionality"""
    
    @pytest.mark.asyncio
    async def test_initialize_database_success(self, db_service):
        """Test successful database initialization"""
        with patch('src.app2.services.database.initialization_service.get_async_session') as mock_get_session:
            with patch('src.app2.services.database.initialization_service.NICE_SEED_DATA', []):
                with patch('src.app2.services.database.initialization_service.GOLD_STANDARDS_SEED_DATA', []):
                    
                    mock_session = AsyncMock()
                    mock_get_session.return_value.__aenter__.return_value = mock_session
                    
                    # Mock validation queries
                    mock_result = MagicMock()
                    mock_result.scalar.return_value = 15  # Sufficient data
                    mock_session.execute.return_value = mock_result
                    
                    result = await db_service.initialize_database()
                    
                    assert result["status"] == "completed"
                    assert db_service.initialized is True
                    assert "total_time_ms" in result
                    mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_database_already_initialized(self, db_service):
        """Test skipping when already initialized"""
        db_service.initialized = True
        
        result = await db_service.initialize_database()
        
        assert result["status"] == "already_initialized"
        assert result["skipped"] is True
    
    @pytest.mark.asyncio
    async def test_initialize_database_force_reload(self, db_service):
        """Test force reload functionality"""
        db_service.initialized = True
        
        with patch('src.app2.services.database.initialization_service.get_async_session') as mock_get_session:
            with patch('src.app2.services.database.initialization_service.NICE_SEED_DATA', []):
                with patch('src.app2.services.database.initialization_service.GOLD_STANDARDS_SEED_DATA', []):
                    
                    mock_session = AsyncMock()
                    mock_get_session.return_value.__aenter__.return_value = mock_session
                    
                    # Mock validation queries
                    mock_result = MagicMock()
                    mock_result.scalar.return_value = 10
                    mock_session.execute.return_value = mock_result
                    
                    result = await db_service.initialize_database(force_reload=True)
                    
                    assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_load_nice_protocols_success(self, db_service, mock_session, mock_nice_data):
        """Test successful NICE protocols loading"""
        with patch('src.app2.services.database.initialization_service.NICE_SEED_DATA', mock_nice_data):
            with patch('src.app2.services.database.initialization_service.NICEProtocol') as mock_protocol:
                
                # Mock no existing protocol
                mock_existing = MagicMock()
                mock_existing.fetchone.return_value = None
                mock_session.execute.return_value = mock_existing
                
                result = await db_service._load_nice_protocols(mock_session, False)
                
                assert result["loaded"] == 1
                assert result["skipped"] == 0
                assert result["errors"] == 0
                mock_protocol.assert_called_once()
                mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_nice_protocols_skip_existing(self, db_service, mock_session, mock_nice_data):
        """Test skipping existing NICE protocols"""
        with patch('src.app2.services.database.initialization_service.NICE_SEED_DATA', mock_nice_data):
            
            # Mock existing protocol found
            mock_existing = MagicMock()
            mock_existing.fetchone.return_value = "existing_id"
            mock_session.execute.return_value = mock_existing
            
            result = await db_service._load_nice_protocols(mock_session, False)
            
            assert result["loaded"] == 0
            assert result["skipped"] == 1
            assert result["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_load_nice_protocols_with_errors(self, db_service, mock_session, mock_nice_data):
        """Test handling errors during NICE protocols loading"""
        with patch('src.app2.services.database.initialization_service.NICE_SEED_DATA', mock_nice_data):
            with patch('src.app2.services.database.initialization_service.NICEProtocol') as mock_protocol:
                
                # Mock no existing protocol
                mock_existing = MagicMock()
                mock_existing.fetchone.return_value = None
                mock_session.execute.return_value = mock_existing
                
                # Mock protocol creation error
                mock_protocol.side_effect = Exception("Protocol creation failed")
                
                result = await db_service._load_nice_protocols(mock_session, False)
                
                assert result["loaded"] == 0
                assert result["skipped"] == 0
                assert result["errors"] == 1
    
    @pytest.mark.asyncio
    async def test_load_gold_standards_success(self, db_service, mock_session, mock_gold_data):
        """Test successful gold standards loading"""
        with patch('src.app2.services.database.initialization_service.GOLD_STANDARDS_SEED_DATA', mock_gold_data):
            with patch('src.app2.services.database.initialization_service.validate_gold_standards') as mock_validate:
                with patch('src.app2.services.database.initialization_service.GoldStandardDialogue') as mock_gold:
                    
                    # Mock validation
                    mock_validate.return_value = {"coverage_balanced": True}
                    
                    # Mock no existing standard
                    mock_existing = MagicMock()
                    mock_existing.fetchone.return_value = None
                    mock_session.execute.return_value = mock_existing
                    
                    result = await db_service._load_gold_standards(mock_session, False)
                    
                    assert result["loaded"] == 1
                    assert result["skipped"] == 0
                    assert result["errors"] == 0
                    mock_gold.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_loaded_data_success(self, db_service, mock_session):
        """Test successful data validation"""
        # Mock sufficient counts
        mock_result = MagicMock()
        mock_result.scalar.return_value = 15
        mock_session.execute.return_value = mock_result
        
        # Should not raise exception
        await db_service._validate_loaded_data(mock_session)
    
    @pytest.mark.asyncio
    async def test_validate_loaded_data_insufficient_nice(self, db_service, mock_session):
        """Test validation failure with insufficient NICE protocols"""
        # Mock insufficient NICE count, sufficient gold count
        mock_results = [5, 10]  # nice_count=5, gold_count=10
        mock_result = MagicMock()
        mock_result.scalar.side_effect = mock_results
        mock_session.execute.return_value = mock_result
        
        with pytest.raises(RuntimeError, match="Insufficient NICE protocols"):
            await db_service._validate_loaded_data(mock_session)
    
    @pytest.mark.asyncio
    async def test_get_initialization_status_success(self, db_service):
        """Test getting initialization status successfully"""
        db_service.initialized = True
        
        with patch('src.app2.services.database.initialization_service.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Mock counts
            mock_result = MagicMock()
            mock_result.scalar.side_effect = [25, 15]  # nice=25, gold=15
            mock_session.execute.return_value = mock_result
            
            status = await db_service.get_initialization_status()
            
            assert status["initialized"] is True
            assert status["nice_protocols_count"] == 25
            assert status["gold_standards_count"] == 15
            assert status["status"] == "ready"
    
    @pytest.mark.asyncio
    async def test_get_initialization_status_error(self, db_service):
        """Test getting initialization status with error"""
        with patch('src.app2.services.database.initialization_service.get_async_session') as mock_get_session:
            mock_get_session.side_effect = OperationalError("DB Error", None, None)
            
            status = await db_service.get_initialization_status()
            
            assert status["initialized"] is False
            assert "error" in status
            assert status["status"] == "error"

class TestServiceIntegration:
    """Test service integration and singleton behavior"""
    
    def test_singleton_instance(self):
        """Test singleton database initialization service"""
        assert database_initialization is not None
        assert isinstance(database_initialization, DatabaseInitializationService)
    
    @pytest.mark.asyncio
    async def test_initialize_database_on_startup(self):
        """Test startup initialization function"""
        with patch.object(database_initialization, 'initialize_database') as mock_init:
            mock_init.return_value = {"status": "completed"}
            
            await initialize_database_on_startup()
            
            mock_init.assert_called_once()

class TestErrorHandling:
    """Test comprehensive error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, db_service):
        """Test handling database connection failures"""
        with patch('src.app2.services.database.initialization_service.get_async_session') as mock_get_session:
            mock_get_session.side_effect = OperationalError("Connection failed", None, None)
            
            with pytest.raises(RuntimeError, match="Database initialization failed"):
                await db_service.initialize_database()
    
    @pytest.mark.asyncio
    async def test_commit_failure(self, db_service):
        """Test handling commit failures"""
        with patch('src.app2.services.database.initialization_service.get_async_session') as mock_get_session:
            with patch('src.app2.services.database.initialization_service.NICE_SEED_DATA', []):
                with patch('src.app2.services.database.initialization_service.GOLD_STANDARDS_SEED_DATA', []):
                    
                    mock_session = AsyncMock()
                    mock_session.commit.side_effect = IntegrityError("Commit failed", None, None)
                    mock_get_session.return_value.__aenter__.return_value = mock_session
                    
                    with pytest.raises(RuntimeError, match="Database initialization failed"):
                        await db_service.initialize_database()
