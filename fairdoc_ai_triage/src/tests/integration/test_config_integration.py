"""
Integration tests for V1/V2 configuration compatibility
Tests environment variable sharing, config isolation, and feature flag behavior
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Test environment variables
TEST_ENV_VARS = {
    "APP_NAME": "Fairdoc AI Triage System",
    "NEXT_GEN": "true",
    "ENVIRONMENT": "testing",
    "DEBUG": "true",
    "SECRET_KEY": "test-secret-key-123",
    "ALLOWED_HOSTS": '["localhost", "127.0.0.1"]',
    "ALLOWED_ORIGINS": '["http://localhost:3000"]',
    "DATABASE_URL": "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db",
    "REDIS_URL": "redis://:test_pass@localhost:6379/0",
    "MINIO_ENDPOINT": "localhost:9000",
    "MINIO_ACCESS_KEY": "test_minio",
    "MINIO_SECRET_KEY": "test_minio_key",
    "MINIO_BUCKET_NAME": "test-bucket",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL": "deepseek-r1:8b",
    "RAVEN_WEBHOOK_URL": "http://localhost:8080/test",
    "RAVEN_API_KEY": "test-raven-key",
    "RAVEN_SECRET": "test-raven-secret",
    "JWT_SECRET_KEY": "test-jwt-secret",
    "CELERY_BROKER_URL": "redis://:test_pass@localhost:6379/1",
    "CELERY_RESULT_BACKEND": "redis://:test_pass@localhost:6379/2",
    # V2 specific
    "FAIRDOC_V2_ENABLED": "true",
    "FAIRDOC_V2_DSPy_MODEL": "deepseek-r1:8b",
    "FAIRDOC_V2_MAX_CONVERSATION_TURNS": "8",
    "TELEGRAM_BOT_TOKEN": "test-telegram-token",
}

@pytest.fixture
def clean_env():
    """Clean environment fixture with proper teardown"""
    original_env = os.environ.copy()
    # Clear all env vars
    for key in list(os.environ.keys()):
        if key.startswith(('APP_', 'NEXT_', 'FAIRDOC_', 'DATABASE_', 'REDIS_')):
            del os.environ[key]
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def test_env(clean_env):
    """Set up test environment variables - Fixed PT022"""
    os.environ.update(TEST_ENV_VARS)
    return TEST_ENV_VARS  # Changed from yield to return (no teardown needed)

class TestConfigurationIsolation:
    """Test V1 and V2 config isolation and compatibility"""
    
    def test_v1_config_loads_with_v2_env_vars(self, test_env):
        """Test V1 config ignores V2-specific environment variables"""
        from src.app.core.config import Settings
        
        v1_settings = Settings()
        
        # V1 should load successfully
        assert v1_settings.APP_NAME == "Fairdoc AI Triage System"
        assert v1_settings.NEXT_GEN is True
        assert v1_settings.DATABASE_URL == TEST_ENV_VARS["DATABASE_URL"]
        
        # V1 should ignore V2-specific vars (due to extra="ignore")
        assert not hasattr(v1_settings, 'FAIRDOC_V2_ENABLED')
        assert not hasattr(v1_settings, 'FAIRDOC_V2_DSPy_MODEL')

    def test_v2_config_loads_all_vars(self, test_env):
        """Test V2 config loads both V1 and V2 variables"""
        from src.app2.core.config_v2 import SettingsV2
        
        v2_settings = SettingsV2()
        
        # V2 should load V1 vars
        assert v2_settings.APP_NAME == "Fairdoc AI Triage System V2"
        assert v2_settings.DATABASE_URL == TEST_ENV_VARS["DATABASE_URL"]
        
        # V2 should load V2-specific vars
        assert v2_settings.FAIRDOC_V2_ENABLED is True
        assert v2_settings.FAIRDOC_V2_DSPy_MODEL == "deepseek-r1:8b"
        assert v2_settings.FAIRDOC_V2_MAX_CONVERSATION_TURNS == 8

    def test_v2_feature_flag_logic(self, test_env):
        """Test V2 feature flag evaluation logic"""
        from src.app2.core.config_v2 import SettingsV2
        
        # Test V2 enabled
        v2_settings = SettingsV2()
        assert v2_settings.is_v2_enabled is True
        
        # Test V2 disabled via FAIRDOC_V2_ENABLED
        os.environ["FAIRDOC_V2_ENABLED"] = "false"
        v2_settings_disabled = SettingsV2()
        assert v2_settings_disabled.is_v2_enabled is False

    def test_dspy_model_config_property(self, test_env):
        """Test V2 DSPy model configuration property"""
        from src.app2.core.config_v2 import SettingsV2
        
        v2_settings = SettingsV2()
        dspy_config = v2_settings.dspy_model_config
        
        assert dspy_config["model_name"] == "deepseek-r1:8b"
        assert dspy_config["api_base"] == "http://localhost:11434"
        assert dspy_config["max_turns"] == 8

class TestEnvironmentVariableValidation:
    """Test environment variable validation and error handling"""
    
    def test_missing_required_fields_v1(self, clean_env):
        """Test V1 config fails gracefully with missing required fields"""
        from src.app.core.config import Settings
        
        # Set minimal required fields
        os.environ.update({
            "SECRET_KEY": "test-key",
            "DATABASE_URL": "postgresql://test",
            "REDIS_URL": "redis://test",
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "test",
            "MINIO_SECRET_KEY": "test",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "RAVEN_WEBHOOK_URL": "http://test",
            "RAVEN_API_KEY": "test",
            "RAVEN_SECRET": "test",
            "JWT_SECRET_KEY": "test",
            "CELERY_BROKER_URL": "redis://test",
            "CELERY_RESULT_BACKEND": "redis://test"
        })
        
        # Should load successfully with defaults
        v1_settings = Settings()
        assert v1_settings.ENVIRONMENT == "development"
        assert v1_settings.DEBUG is True

    def test_config_caching(self, test_env):
        """Test configuration caching behavior"""
        from src.app.core.config import get_settings
        from src.app2.core.config_v2 import get_settings_v2
        
        # Should return same instance (cached)
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
        
        # V2 should also cache
        v2_settings1 = get_settings_v2()
        v2_settings2 = get_settings_v2()
        assert v2_settings1 is v2_settings2

    @patch('src.app.core.config.Settings')
    def test_config_error_handling(self, mock_settings, test_env):
        """Test configuration error handling - Fixed PT012"""
        # Fixed: Move import outside pytest.raises block
        from src.app.core.config import get_settings
        
        # Simulate config loading error
        mock_settings.side_effect = Exception("Config loading failed")
        
        with pytest.raises(Exception, match="Config loading failed"):
            get_settings()  # Single statement in pytest.raises block

class TestConfigurationIntegration:
    """Test integration scenarios between V1 and V2 configs"""
    
    def test_config_isolation_no_interference(self, test_env):
        """Test that V1 and V2 configs don't interfere with each other"""
        from src.app.core.config import Settings
        from src.app2.core.config_v2 import SettingsV2
        
        # Load both configs
        v1_settings = Settings()
        v2_settings = SettingsV2()
        
        # Verify isolation
        assert v1_settings.APP_NAME != v2_settings.APP_NAME
        assert v1_settings.DATABASE_URL == v2_settings.DATABASE_URL  # Shared
        
        # Modify V2 and ensure V1 unaffected
        assert v2_settings.FAIRDOC_V2_ENABLED is True
        assert not hasattr(v1_settings, 'FAIRDOC_V2_ENABLED')

    def test_test_settings_override(self, test_env):
        """Test test-specific settings override production settings"""
        from src.app.core.config import TestSettings
        from src.app2.core.config_v2 import TestSettingsV2
        
        test_v1 = TestSettings()
        test_v2 = TestSettingsV2()
        
        # Test settings should override environment
        assert test_v1.ENVIRONMENT == "testing"
        assert test_v2.ENVIRONMENT == "testing"
        assert test_v2.FAIRDOC_V2_ENABLED is True  # Force enabled in tests

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
