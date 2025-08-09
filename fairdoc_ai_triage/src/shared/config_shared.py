"""
Shared Configuration Settings for V1 and V2

Single responsibility: Common environment variables and validation logic
"""

from pydantic_settings import BaseSettings
from typing import List
import structlog

logger = structlog.get_logger(__name__)

class SharedSettings(BaseSettings):
    """Common settings used by both V1 and V2"""
    
    # Core Application Settings
    APP_NAME: str = "Fairdoc AI Triage System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str
    
    # Infrastructure (shared)
    DATABASE_URL: str
    TEST_DATABASE_URL: str = None
    REDIS_URL: str
    
    # MinIO Storage (shared)
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str = "fairdoc-ai-storage"
    
    # Ollama Configuration (shared)
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str = "gemma3n:e4b"
    
    # AI Services (shared)
    OPENAI_API_KEY: str = None
    DSPY_LM_TYPE: str = "ollama"
    DSPY_MODEL_NAME: str = "gemma3n:e4b"
    AVAILABLE_MODELS: str = "deepseek-r1:8b,gemma3n:e4b,mistral:7b,gpt-oss:20b"
    
    # Chat Integration (shared)
    RAVEN_WEBHOOK_URL: str
    RAVEN_API_KEY: str
    RAVEN_SECRET: str
    
    # Security (shared)
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Monitoring (shared)
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: str = None
    PROMETHEUS_ENDPOINT: str = None
    
    # Celery (shared)
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # Network Settings (problematic fields handled as strings)
    ALLOWED_HOSTS: str = "localhost,127.0.0.1,0.0.0.0"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000,http://localhost:8002"
    
    @property
    def allowed_hosts_list(self) -> List[str]:
        """Get allowed hosts as list"""
        return [host.strip() for host in self.ALLOWED_HOSTS.split(',')]
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Get allowed origins as list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',')]
    
    @property
    def available_models_list(self) -> List[str]:
        """Get available models as list"""
        return [model.strip() for model in self.AVAILABLE_MODELS.split(',')]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore V2-specific env vars not defined in shared settings
# Global shared instance
shared_settings = SharedSettings()
