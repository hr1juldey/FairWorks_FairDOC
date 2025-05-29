"""
V1 Core Configuration Management - NO SECURITY
Pure configuration management, security handled separately
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path
from functools import lru_cache

class V1Settings(BaseSettings):
    """V1 API Configuration - Environment and operational settings only"""
    
    # Environment Information
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Application Info
    APP_NAME: str = "Fairdoc AI Triage API v1"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001  # V1 runs on 8001
    WORKERS: int = 1
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://fairdoc:password@localhost:5432/fairdoc_v0"
    DATABASE_ECHO: bool = True
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 10
    
    # MinIO Configuration
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "fairdoc-files"
    MINIO_SECURE: bool = False
    
    # ChromaDB Configuration
    CHROMADB_URL: str = "http://localhost:8000"
    CHROMADB_PERSIST_PATH: str = "./data/chromadb"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODELS: List[str] = ["gemma3:4b", "deepseek-r1:14b", "qwen2.5-coder:14b"]
    OLLAMA_TIMEOUT: int = 300
    OLLAMA_MAX_TOKENS: int = 2048
    OLLAMA_TEMPERATURE: float = 0.1
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/dicom",
        "audio/wav",
        "audio/mp3"
    ]
    
    # NICE Protocol Configuration
    NICE_PROTOCOLS_COUNT: int = 10
    NICE_QUESTIONS_REQUIRED: int = 80  # Minimum completion percentage
    
    # Medical Triage Configuration
    URGENCY_THRESHOLD_HIGH: float = 0.8
    URGENCY_THRESHOLD_MEDIUM: float = 0.6
    URGENCY_THRESHOLD_LOW: float = 0.4
    
    # API Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    # WebSocket Configuration
    WEBSOCKET_ENABLED: bool = True
    WEBSOCKET_PING_INTERVAL: int = 20
    WEBSOCKET_PING_TIMEOUT: int = 10
    WEBSOCKET_MAX_CONNECTIONS: int = 100
    
    # Emergency Services Configuration (Mock)
    EMERGENCY_SERVICE_ENABLED: bool = True
    MOCK_HOSPITAL_API: str = "http://localhost:9999/hospital"
    MOCK_AMBULANCE_API: str = "http://localhost:9999/ambulance"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_TO_FILE: bool = True
    LOG_FILE_PATH: str = "./logs/v1_api.log"
    
    # Paths Configuration
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT.lower() in ["development", "dev", "local"]
    
    @property
    def is_testing(self) -> bool:
        """Check if running in test mode"""
        return self.ENVIRONMENT.lower() in ["testing", "test", "staging"]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT.lower() in ["production", "prod"]
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        for directory in [self.DATA_DIR, self.LOGS_DIR, self.UPLOADS_DIR, self.REPORTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        if self.is_production:
            return [
                "https://fairdoc.ai",
                "https://api.fairdoc.ai",
                "https://triage.fairdoc.ai"
            ]
        elif self.is_testing:
            return [
                "https://test.fairdoc.ai",
                "https://staging.fairdoc.ai"
            ]
        else:  # Development
            return [
                "http://localhost:3000",   # React
                "http://localhost:8080",   # Vue
                "http://localhost:32123",  # Mesop
                "http://localhost:8501",   # Streamlit
                "http://127.0.0.1:32123",
                "http://127.0.0.1:8501",
                "*"  # Allow all in development
            ]

@lru_cache()
def get_v1_settings() -> V1Settings:
    """Get cached V1 settings instance"""
    settings = V1Settings()
    settings.create_directories()
    return settings


# Global V1 settings instance
v1_settings = get_v1_settings()

# Export settings
__all__ = ["V1Settings", "get_v1_settings", "v1_settings"]
