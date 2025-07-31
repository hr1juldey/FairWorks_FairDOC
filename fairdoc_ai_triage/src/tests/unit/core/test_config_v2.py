"""
Unit Tests for V2 Configuration Management
Tests environment loading, validation, and V2-specific settings
"""
import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

# Define minimal test environment BEFORE importing config_v2
TEST_ENV = {
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
}

# Patch environment before importing config_v2
with patch.dict(os.environ, TEST_ENV):
    from src.app2.core.config_v2 import SettingsV2, TestSettingsV2, get_settings_v2, get_test_settings_v2


class TestSettingsV2:
    """Test V2 configuration loading and validation"""
    
    def test_environment_variable_loading(self):
        """Test configuration loads from environment correctly"""
        with patch.dict(os.environ, {
            **TEST_ENV,
            'APP_NAME': 'Fairdoc AI Triage System',
            'ENVIRONMENT': 'development',
            'FAIRDOC_V2_ENABLED': 'true',
            'FAIRDOC_V2_DSPy_MODEL': 'deepseek-r1:8b',
            'FAIRDOC_V2_MAX_CONVERSATION_TURNS': '8'
        }):
            settings = SettingsV2()
            
            # Test V2 specific settings
            assert settings.FAIRDOC_V2_ENABLED is True
            assert settings.FAIRDOC_V2_DSPy_MODEL == "deepseek-r1:8b"
            assert settings.FAIRDOC_V2_MAX_CONVERSATION_TURNS == 8
            
            # Test inherited V1 settings
            assert settings.APP_NAME == "Fairdoc AI Triage System"
            assert settings.ENVIRONMENT == "development"
            assert settings.SECRET_KEY == "test-secret-key"
    
    def test_comma_separated_list_parsing(self):
        """Test ALLOWED_HOSTS and ALLOWED_ORIGINS parsing"""
        with patch.dict(os.environ, {
            **TEST_ENV,
            'ALLOWED_HOSTS': '["localhost", "127.0.0.1", "0.0.0.0"]',
            'ALLOWED_ORIGINS': '["http://localhost:3000", "http://localhost:8000"]'
        }):
            settings = SettingsV2()
            
            assert settings.ALLOWED_HOSTS == ["localhost", "127.0.0.1", "0.0.0.0"]
            assert settings.ALLOWED_ORIGINS == ["http://localhost:3000", "http://localhost:8000"]
    
    def test_v2_enabled_property(self):
        """Test is_v2_enabled property logic"""
        with patch.dict(os.environ, {
            **TEST_ENV,
            'NEXT_GEN': 'True',
            'FAIRDOC_V2_ENABLED': 'true'
        }):
            settings = SettingsV2()
            assert settings.is_v2_enabled is True
        
        # Test when V2 is disabled
        with patch.dict(os.environ, {
            **TEST_ENV,
            'NEXT_GEN': 'True',
            'FAIRDOC_V2_ENABLED': 'false'
        }):
            settings = SettingsV2()
            assert settings.is_v2_enabled is False
    
    def test_dspy_model_config_property(self):
        """Test DSPy model configuration property"""
        with patch.dict(os.environ, {
            **TEST_ENV,
            'FAIRDOC_V2_DSPy_MODEL': 'deepseek-r1:8b',
            'FAIRDOC_V2_MAX_CONVERSATION_TURNS': '8'
        }):
            settings = SettingsV2()
            config = settings.dspy_model_config
            
            assert config["model_name"] == "deepseek-r1:8b"
            assert config["api_base"] == "http://localhost:11434"
            assert config["max_turns"] == 8
    
    def test_chat_provider_validation(self):
        """Test chat provider literal validation"""
        with patch.dict(os.environ, {
            **TEST_ENV,
            'FAIRDOC_V2_CHAT_PROVIDER': 'raven'
        }):
            settings = SettingsV2()
            assert settings.FAIRDOC_V2_CHAT_PROVIDER == "raven"


class TestTestSettingsV2:
    """Test V2 test-specific configuration"""
    

    
    def test_settings_caching(self):
        """Test settings are cached via lru_cache"""
        with patch.dict(os.environ, TEST_ENV):
            settings1 = get_settings_v2()
            settings2 = get_settings_v2()
            
            # Should be the same instance due to caching
            assert settings1 is settings2
            
            test_settings1 = get_test_settings_v2()
            test_settings2 = get_test_settings_v2()
            
            assert test_settings1 is test_settings2


class TestConfigValidation:
    """Test configuration validation and error handling"""
    

    
    def test_invalid_conversation_turns_type(self):
        """Test invalid conversation turns type validation"""
        with patch.dict(os.environ, {
            **TEST_ENV,
            'FAIRDOC_V2_MAX_CONVERSATION_TURNS': 'invalid'
        }):
            with pytest.raises(ValidationError):
                SettingsV2()

    def test_app_name_override_from_env(self):
        """Test APP_NAME can be overridden from environment"""
        with patch.dict(os.environ, {
            **TEST_ENV,
            'APP_NAME': 'Custom Fairdoc Name'
        }):
            settings = SettingsV2()
            assert settings.APP_NAME == "Custom Fairdoc Name"
    
    def test_actual_env_file_values(self):
        """Test configuration reads from actual .env file values"""
        with patch.dict(os.environ, {
            **TEST_ENV,
            'APP_NAME': 'Fairdoc AI Triage System',
            'NEXT_GEN': 'True',
            'ENVIRONMENT': 'development',
            'FAIRDOC_V2_ENABLED': 'true',
            'FAIRDOC_V2_DSPy_MODEL': 'deepseek-r1:8b',
            'FAIRDOC_V2_MAX_CONVERSATION_TURNS': '8',
            'FAIRDOC_V2_CHAT_PROVIDER': 'raven'
        }):
            settings = SettingsV2()
            
            # Validate actual .env values work
            assert settings.APP_NAME == "Fairdoc AI Triage System"
            assert settings.NEXT_GEN is True
            assert settings.ENVIRONMENT == "development"
            assert settings.FAIRDOC_V2_ENABLED is True
            assert settings.FAIRDOC_V2_DSPy_MODEL == "deepseek-r1:8b"
            assert settings.FAIRDOC_V2_MAX_CONVERSATION_TURNS == 8
            assert settings.FAIRDOC_V2_CHAT_PROVIDER == "raven"
            assert settings.is_v2_enabled is True
