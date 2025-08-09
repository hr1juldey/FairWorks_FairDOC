"""
Fairdoc AI Triage System - V2 Application Configuration

Extends V1 config with advanced medical triage features and DSPy LLM management
"""

from functools import lru_cache
from pydantic import BaseModel, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from typing import List, Optional, Literal
from src.shared.config_shared import SharedSettings, shared_settings
class SettingsV2(BaseSettings):
    """V2 Application settings - includes all V1 + V2 specific features"""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Ignore any other environment variables
    )

    # V2-Specific Features
    NEXT_GEN: bool = True
    FAIRDOC_V2_ENABLED: bool = True

    # Validation handled by parent SharedSettings class

    # Database Configuration (V1 compatible)
    DATABASE_URL: str
    TEST_DATABASE_URL: Optional[str] = None

    # Redis Configuration
    REDIS_URL: str

    # Feature Flags
    NEXT_GEN: Optional[bool] = True
    FAIRDOC_V2_ENABLED: bool = False

    # === V2 Specific Medical Configuration ===
    FAIRDOC_V2_DSPy_MODEL: str = "gemma3n:e4b"
    FAIRDOC_V2_NICE_PROTOCOLS_TABLE: str = "nice_protocols_v2"
    FAIRDOC_V2_GOLD_STANDARDS_COUNT: int = 50
    FAIRDOC_V2_MAX_CONVERSATION_TURNS: int = 15
    FAIRDOC_V2_EMERGENCY_ALERT_WEBHOOK: Optional[str] = None

    # Chat Integration (V2 Advanced)
    FAIRDOC_V2_CHAT_PROVIDER: Literal["raven", "telegram", "whatsapp"] = "raven"
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    WHATSAPP_BUSINESS_TOKEN: Optional[str] = None

    # MinIO Object Storage (inherited)
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str = "fairdoc-ai-storage"

    # Ollama Configuration (V2 enhanced)
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str = "gemma3n:e4b"

    # AI Services (V2 extended)
    OPENAI_API_KEY: Optional[str] = None
    DSPY_LM_TYPE: str = "ollama"
    DSPY_MODEL_NAME: str = "gemma3n:e4b"
    
    # === NEW: Available Models Configuration ===
    AVAILABLE_MODELS: Optional[str] = None  # Comma-separated list of available models

    # Raven Chat Integration (inherited)
    RAVEN_WEBHOOK_URL: str
    RAVEN_API_KEY: str
    RAVEN_SECRET: str

    # Security (inherited)
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # Logging (inherited)
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENDPOINT: Optional[str] = None

    # Celery Configuration (inherited)
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # === V2 Medical Triage Specific ===
    @property
    def is_v2_enabled(self) -> bool:
        """Check if V2 features are enabled"""
        return self.FAIRDOC_V2_ENABLED and self.NEXT_GEN

    @property
    def dspy_model_config(self) -> dict:
        """DSPy model configuration for medical agents"""
        return {
            "model_name": self.FAIRDOC_V2_DSPy_MODEL,
            "api_base": self.OLLAMA_BASE_URL,
            "max_turns": self.FAIRDOC_V2_MAX_CONVERSATION_TURNS,
            "available_models": self.AVAILABLE_MODELS
        }

    @property
    def available_models_list(self) -> List[str]:
        """Get available models as a list"""
        if self.AVAILABLE_MODELS:
            return [model.strip() for model in self.AVAILABLE_MODELS.split(',')]
        return []

class TestSettingsV2(SettingsV2):
    """Test-specific settings for V2"""
    
    ENVIRONMENT: str = "testing"
    DEBUG: bool = True
    FAIRDOC_V2_ENABLED: bool = True  # Enable V2 in tests

    @property
    def database_url(self) -> str:
        return self.TEST_DATABASE_URL or self.DATABASE_URL.replace("fairdoc_ai", "fairdoc_ai_test")

@lru_cache()
def get_settings_v2() -> SettingsV2:
    """Get cached V2 application settings"""
    return SettingsV2()

@lru_cache()
def get_test_settings_v2() -> TestSettingsV2:
    """Get cached V2 test settings"""
    return TestSettingsV2()

# Global V2 settings instance
settings_v2 = get_settings_v2()
