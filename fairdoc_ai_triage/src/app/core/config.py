"""
Fairdoc AI Triage System - V1 Application Configuration
Isolated from V2 dependencies to maintain stability
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import BaseModel, field_validator, ConfigDict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """V1 Application settings - ignores V2-specific environment variables"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Critical: ignore V2 environment variables
    )
    
    # Core Application Settings (V1 only)
    APP_NAME: str = "Fairdoc AI Triage System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    @field_validator('ALLOWED_HOSTS', 'ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_comma_separated_list(cls, v):
        """Parse comma-separated strings into lists"""
        if isinstance(v, str):
            # Handle both bracketed and comma-separated formats
            v = v.strip('[]"\'')
            return [item.strip(' "\'') for item in v.split(',') if item.strip()]
        return v
    
    # Database Configuration
    DATABASE_URL: str
    TEST_DATABASE_URL: Optional[str] = None
    
    # Redis Configuration
    REDIS_URL: str
    
    # Feature Flag (V1 safe)
    NEXT_GEN: Optional[bool] = False
    
    # MinIO Object Storage
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str = "fairdoc-ai-storage"
    
    # Ollama Configuration (V1 compatible)
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str = "deepseek-r1:8b"
    
    # AI Services (V1 minimal)
    OPENAI_API_KEY: Optional[str] = None
    DSPY_LM_TYPE: str = "ollama"
    DSPY_MODEL_NAME: str = "deepseek-r1:8b"
    
    # Raven Chat Integration
    RAVEN_WEBHOOK_URL: str
    RAVEN_API_KEY: str
    RAVEN_SECRET: str
    
    # Security
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Logging (V1 basic)
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENDPOINT: Optional[str] = None
    
    # Celery Configuration
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

class TestSettings(Settings):
    """Test-specific settings for V1"""
    
    ENVIRONMENT: str = "testing"
    DEBUG: bool = True
    
    @property
    def database_url(self) -> str:
        return self.TEST_DATABASE_URL or self.DATABASE_URL.replace("fairdoc_ai", "fairdoc_ai_test")

@lru_cache()
def get_settings() -> Settings:
    """Get cached V1 application settings"""
    return Settings()

@lru_cache()
def get_test_settings() -> TestSettings:
    """Get cached V1 test settings"""
    return TestSettings()

# Global V1 settings instance
settings = get_settings()
