"""
Fairdoc AI Triage System - Application Configuration
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
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
            return [item.strip() for item in v.split(',')]
        return v
    
    # Database
    DATABASE_URL: str
    TEST_DATABASE_URL: Optional[str] = None
    
    # Redis
    REDIS_URL: str
    
    # MinIO
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str = "fairdoc-ai-storage"
    
    # Ollama
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str  # Keep as variable for flexibility
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    DSPY_LM_TYPE: str = "ollama"
    DSPY_MODEL_NAME: str  # Variable model name
    
    # Raven Chat
    RAVEN_WEBHOOK_URL: str
    RAVEN_API_KEY: str
    RAVEN_SECRET: str
    
    # Security
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Logging
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENDPOINT: Optional[str] = None
    
    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class TestSettings(Settings):
    """Test-specific settings"""
    
    ENVIRONMENT: str = "testing"
    DEBUG: bool = True
    
    @property
    def database_url(self) -> str:
        return self.TEST_DATABASE_URL or self.DATABASE_URL.replace("fairdoc_ai", "fairdoc_ai_test")


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


@lru_cache()
def get_test_settings() -> TestSettings:
    """Get cached test settings"""
    return TestSettings()


# Global settings instance
settings = get_settings()
