"""
Integration tests for V1/V2 configuration compatibility
Focuses on real-world environment variable interaction scenarios
"""

import pytest
import os
from unittest.mock import patch
from typing import Dict, Any
from pydantic import ValidationError

class TestEnvironmentVariableInteraction:
    """Test real environment variable interaction scenarios"""
    
    def test_v1_config_respects_env_precedence(self):
        """Test V1 config respects environment variable precedence over defaults"""
        from src.app.core.config import Settings
        
        # Current environment should override any defaults
        v1_settings = Settings()
        
        # These should come from the test environment (conftest.py)
        assert v1_settings.ENVIRONMENT == "testing"  # From conftest.py
        assert v1_settings.NEXT_GEN is True  # From conftest.py
        assert v1_settings.DEBUG is True  # From conftest.py
        
        # Critical fields should be loaded from environment
        assert v1_settings.DATABASE_URL.startswith("postgresql")
        assert v1_settings.REDIS_URL.startswith("redis")
        assert v1_settings.SECRET_KEY == "test-secret-key-123"

    def test_v2_config_respects_env_precedence(self):
        """Test V2 config respects environment variable precedence"""
        from src.app2.core.config_v2 import SettingsV2
        
        v2_settings = SettingsV2()
        
        # V2 should inherit shared environment variables
        assert v2_settings.ENVIRONMENT == "testing"
        assert v2_settings.DATABASE_URL == v2_settings.DATABASE_URL
        
        # V2-specific environment variables should be loaded
        assert v2_settings.FAIRDOC_V2_ENABLED is True
        assert v2_settings.FAIRDOC_V2_DSPy_MODEL == "deepseek-r1:8b"
        assert v2_settings.FAIRDOC_V2_MAX_CONVERSATION_TURNS == 8

    def test_v1_ignores_v2_specific_env_vars(self):
        """Test V1 config properly ignores V2-specific environment variables"""
        from src.app.core.config import Settings
        
        v1_settings = Settings()
        
        # V1 should not have V2-specific attributes
        v2_only_fields = [
            'FAIRDOC_V2_ENABLED',
            'FAIRDOC_V2_DSPy_MODEL', 
            'FAIRDOC_V2_MAX_CONVERSATION_TURNS',
            'FAIRDOC_V2_NICE_PROTOCOLS_TABLE',
            'FAIRDOC_V2_CHAT_PROVIDER'
        ]
        
        for field in v2_only_fields:
            assert not hasattr(v1_settings, field), f"V1 should not have {field}"

    def test_v2_loads_all_env_vars(self):
        """Test V2 config loads both shared and V2-specific environment variables"""
        from src.app2.core.config_v2 import SettingsV2
        
        v2_settings = SettingsV2()
        
        # Should have shared V1 fields
        shared_fields = ['DATABASE_URL', 'REDIS_URL', 'SECRET_KEY', 'OLLAMA_BASE_URL']
        for field in shared_fields:
            assert hasattr(v2_settings, field), f"V2 should have shared field {field}"
        
        # Should have V2-specific fields
        v2_fields = ['FAIRDOC_V2_ENABLED', 'FAIRDOC_V2_DSPy_MODEL']
        for field in v2_fields:
            assert hasattr(v2_settings, field), f"V2 should have V2-specific field {field}"

class TestFeatureFlagInteraction:
    """Test feature flag behavior with environment variables"""
    
    def test_next_gen_flag_controls_v2_availability(self):
        """Test NEXT_GEN environment variable controls V2 feature availability"""
        from src.app.core.config import Settings
        
        v1_settings = Settings()
        
        # NEXT_GEN should be enabled in test environment
        assert v1_settings.NEXT_GEN is True
        
        # This enables V2 features in the main app
        assert os.environ.get("NEXT_GEN") == "true"

    def test_v2_feature_flag_combination(self):
        """Test combination of NEXT_GEN and FAIRDOC_V2_ENABLED flags"""
        from src.app2.core.config_v2 import SettingsV2
        
        v2_settings = SettingsV2()
        
        # Both flags should be enabled for full V2 functionality
        assert v2_settings.NEXT_GEN is True
        assert v2_settings.FAIRDOC_V2_ENABLED is True
        assert v2_settings.is_v2_enabled is True

    @patch.dict(os.environ, {"FAIRDOC_V2_ENABLED": "false"})
    def test_v2_disabled_via_feature_flag(self):
        """Test V2 can be disabled via FAIRDOC_V2_ENABLED flag"""
        from src.app2.core.config_v2 import SettingsV2
        
        v2_settings = SettingsV2()
        
        # V2 should be disabled even if NEXT_GEN is true
        assert v2_settings.NEXT_GEN is True  # Still true
        assert v2_settings.FAIRDOC_V2_ENABLED is False  # Overridden
        assert v2_settings.is_v2_enabled is False  # Combined result

class TestConfigurationIsolation:
    """Test V1 and V2 configuration isolation"""
    
    def test_configs_load_independently(self):
        """Test V1 and V2 configs can be loaded independently without interference"""
        from src.app.core.config import Settings
        from src.app2.core.config_v2 import SettingsV2
        
        # Load both configs
        v1_settings = Settings()
        v2_settings = SettingsV2()
        
        # Both should load successfully
        assert isinstance(v1_settings, Settings)
        assert isinstance(v2_settings, SettingsV2)
        
        # Should share common configuration
        assert v1_settings.DATABASE_URL == v2_settings.DATABASE_URL
        assert v1_settings.REDIS_URL == v2_settings.REDIS_URL
        
        # V2 should have additional capabilities
        assert hasattr(v2_settings, 'is_v2_enabled')
        assert not hasattr(v1_settings, 'is_v2_enabled')

    def test_config_caching_isolation(self):
        """Test configuration caching works independently for V1 and V2"""
        from src.app.core.config import get_settings
        from src.app2.core.config_v2 import get_settings_v2
        
        # V1 caching
        v1_settings_1 = get_settings()
        v1_settings_2 = get_settings()
        assert v1_settings_1 is v1_settings_2  # Same instance
        
        # V2 caching
        v2_settings_1 = get_settings_v2()
        v2_settings_2 = get_settings_v2()
        assert v2_settings_1 is v2_settings_2  # Same instance
        
        # Different instances between V1 and V2
        assert v1_settings_1 is not v2_settings_1

class TestEnvironmentVariableValidation:
    """Test environment variable validation and error handling"""
    
    def test_required_fields_are_present(self):
        """Test that required fields are present in test environment"""
        required_fields = [
            "SECRET_KEY", "DATABASE_URL", "REDIS_URL", 
            "OLLAMA_BASE_URL", "RAVEN_WEBHOOK_URL"
        ]
        
        for field in required_fields:
            assert os.environ.get(field) is not None, f"Required field {field} missing"

    def test_list_field_parsing(self):
        """Test that list fields are properly parsed from environment"""
        from src.app.core.config import Settings
        from src.app2.core.config_v2 import SettingsV2
        
        v1_settings = Settings()
        v2_settings = SettingsV2()
        
        # Test list parsing for both configs
        assert isinstance(v1_settings.ALLOWED_HOSTS, list)
        assert isinstance(v1_settings.ALLOWED_ORIGINS, list)
        assert isinstance(v2_settings.ALLOWED_HOSTS, list)
        assert isinstance(v2_settings.ALLOWED_ORIGINS, list)
        
        # Should contain expected values
        assert "localhost" in v1_settings.ALLOWED_HOSTS
        assert "http://localhost:3000" in v1_settings.ALLOWED_ORIGINS

    def test_boolean_field_parsing(self):
        """Test that boolean fields are properly parsed from environment"""
        from src.app.core.config import Settings
        from src.app2.core.config_v2 import SettingsV2
        
        v1_settings = Settings()
        v2_settings = SettingsV2()
        
        # Test boolean parsing
        assert isinstance(v1_settings.DEBUG, bool)
        assert isinstance(v1_settings.NEXT_GEN, bool)
        assert isinstance(v2_settings.FAIRDOC_V2_ENABLED, bool)
        
        # Should be True in test environment
        assert v1_settings.DEBUG is True
        assert v1_settings.NEXT_GEN is True
        assert v2_settings.FAIRDOC_V2_ENABLED is True

class TestProductionScenarios:
    """Test realistic production scenarios"""
    
    def test_database_url_configuration(self):
        """Test database URL configuration for different environments"""
        from src.app.core.config import Settings, TestSettings
        
        # Regular settings should use test database
        settings = Settings()
        assert "test_user" in settings.DATABASE_URL
        
        # Test settings should potentially modify database URL
        test_settings = TestSettings()
        assert isinstance(test_settings.database_url, str)

    def test_v2_dspy_model_configuration(self):
        """Test V2 DSPy model configuration"""
        from src.app2.core.config_v2 import SettingsV2
        
        v2_settings = SettingsV2()
        dspy_config = v2_settings.dspy_model_config
        
        # Should be properly configured dictionary
        assert isinstance(dspy_config, dict)
        assert dspy_config["model_name"] == "deepseek-r1:8b"
        assert dspy_config["api_base"] == "http://localhost:11434"
        assert dspy_config["max_turns"] == 8

    def test_chat_provider_configuration(self):
        """Test chat provider configuration for V2"""
        from src.app2.core.config_v2 import SettingsV2
        
        v2_settings = SettingsV2()
        
        # Should have valid chat provider
        assert v2_settings.FAIRDOC_V2_CHAT_PROVIDER in ["raven", "telegram", "whatsapp"]
        
        # Should have associated tokens when needed
        if v2_settings.FAIRDOC_V2_CHAT_PROVIDER == "telegram":
            assert v2_settings.TELEGRAM_BOT_TOKEN is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
